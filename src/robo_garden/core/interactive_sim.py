"""InteractiveSim: persistent MuJoCo simulation driven by Studio UI events.

The Design Studio needs a physics engine that stays alive across many
user interactions (joint sliders, apply-force clicks, pause/reset) and
continuously streams the world state out to the Isaac Sim viewport for
RTX rendering.

This replaces the old ``SessionViewer`` (which opened its own MuJoCo window
per robot iteration) for Studio mode.  Physics authority stays in MuJoCo;
Isaac Sim is a kinematic mirror.

Lifecycle
---------
    sim = InteractiveSim()
    sim.load_robot(mjcf_path=..., name="go2_walker")     # thread boots
    sim.apply_joint_target("FR_hip_joint", 0.3)          # called from UI
    sim.apply_force(body="trunk", force=(10, 0, 0))
    sim.pause() / sim.resume() / sim.reset()
    sim.close()                                           # on app exit

State is streamed outbound via ``frame_callback(robot_name, frames, timesteps)``
in small batches (default 10 frames ~ 50 Hz out at 500 Hz physics).  A
separate ``meta_callback(meta_dict)`` fires once per robot load so the UI
can build joint sliders with correct ranges.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import mujoco
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command types (UI -> simulator thread)
# ---------------------------------------------------------------------------

class _LoadRobot(NamedTuple):
    mjcf_path: str      # prefer file path for mesh resolution
    mjcf_xml: str       # fallback if no path
    name: str


class _JointTarget(NamedTuple):
    joint: str
    value: float


class _ApplyForce(NamedTuple):
    body: str
    force: tuple[float, float, float]
    torque: tuple[float, float, float]
    duration: float     # seconds the force stays applied


class _Pause(NamedTuple):
    pass


class _Resume(NamedTuple):
    pass


class _Reset(NamedTuple):
    pass


class _Step(NamedTuple):
    n: int


class _Stop(NamedTuple):
    pass


# ---------------------------------------------------------------------------
# Joint / body metadata extraction
# ---------------------------------------------------------------------------

@dataclass
class JointInfo:
    name: str
    type: str           # "hinge" | "slide" | "ball" | "free"
    range: tuple[float, float]
    ctrl_range: tuple[float, float]
    actuator: str       # "" if joint has no actuator


def _introspect_model(model: mujoco.MjModel) -> tuple[list[JointInfo], list[str]]:
    """Extract joint + body names / ranges for UI slider generation."""
    type_names = {
        mujoco.mjtJoint.mjJNT_FREE: "free",
        mujoco.mjtJoint.mjJNT_BALL: "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }

    # Build joint -> actuator map
    joint_to_actuator: dict[int, str] = {}
    for a_id in range(model.nu):
        j_id = int(model.actuator_trnid[a_id, 0])
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id) or f"act_{a_id}"
        joint_to_actuator[j_id] = act_name

    joints: list[JointInfo] = []
    for j_id in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or f"joint_{j_id}"
        jtype = type_names.get(model.jnt_type[j_id], "unknown")
        if model.jnt_limited[j_id]:
            lo, hi = float(model.jnt_range[j_id, 0]), float(model.jnt_range[j_id, 1])
        else:
            # Hinge/slide unbounded — expose a soft +/- pi (hinge) or +/- 1m (slide) range
            lo, hi = (-3.14159, 3.14159) if jtype == "hinge" else (-1.0, 1.0)

        act_name = joint_to_actuator.get(j_id, "")
        if act_name:
            a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if model.actuator_ctrllimited[a_id]:
                ctrl_lo = float(model.actuator_ctrlrange[a_id, 0])
                ctrl_hi = float(model.actuator_ctrlrange[a_id, 1])
            else:
                ctrl_lo, ctrl_hi = lo, hi
        else:
            ctrl_lo, ctrl_hi = lo, hi

        joints.append(JointInfo(
            name=jname,
            type=jtype,
            range=(lo, hi),
            ctrl_range=(ctrl_lo, ctrl_hi),
            actuator=act_name,
        ))

    bodies: list[str] = []
    for b_id in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
        bodies.append(bname)

    return joints, bodies


# ---------------------------------------------------------------------------
# InteractiveSim
# ---------------------------------------------------------------------------

FrameCallback = Callable[[str, np.ndarray, list[float]], None]
MetaCallback = Callable[[dict], None]


class InteractiveSim:
    """Persistent background MuJoCo simulator driven by UI events.

    Parameters
    ----------
    frame_callback:
        Called with ``(robot_name, frames_Nxnq, timesteps)`` each ``batch_size``
        physics steps.  Hook the Isaac Sim bridge here.
    meta_callback:
        Called with a dict describing joints / bodies once per robot load.
    batch_size:
        Number of physics steps per outbound frame batch.  Default 10 yields
        ~50 Hz packets at the 500 Hz default physics rate.
    physics_hz:
        Target physics step rate (soft — may slip if callback is slow).
    """

    def __init__(
        self,
        frame_callback: FrameCallback | None = None,
        meta_callback: MetaCallback | None = None,
        batch_size: int = 10,
        physics_hz: int = 500,
    ) -> None:
        self._cmd_queue: queue.Queue = queue.Queue()
        self._frame_cb = frame_callback
        self._meta_cb = meta_callback
        self._batch_size = batch_size
        self._target_dt = 1.0 / float(physics_hz)

        self._thread: threading.Thread | None = None
        self._paused = threading.Event()
        self._running = threading.Event()
        self._lock = threading.Lock()

        # Live state (read-only from outside the sim thread)
        self._current_robot: str = ""
        self._current_joints: list[JointInfo] = []
        self._current_bodies: list[str] = []
        self._latest_qpos: np.ndarray | None = None
        self._diverged: bool = False

    # ------------------------------------------------------------------
    # Public commands (thread-safe)
    # ------------------------------------------------------------------

    def load_robot(self, mjcf_path: str | Path = "", mjcf_xml: str = "", name: str = "robot") -> None:
        """Load a robot.  Prefer ``mjcf_path`` so relative meshdir paths resolve."""
        self._cmd_queue.put(_LoadRobot(
            mjcf_path=str(mjcf_path) if mjcf_path else "",
            mjcf_xml=mjcf_xml,
            name=name,
        ))
        self._ensure_thread()

    def apply_joint_target(self, joint: str, value: float) -> None:
        self._cmd_queue.put(_JointTarget(joint=joint, value=float(value)))

    def apply_force(
        self,
        body: str,
        force: tuple[float, float, float] = (0.0, 0.0, 0.0),
        torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
        duration: float = 0.1,
    ) -> None:
        self._cmd_queue.put(_ApplyForce(
            body=body, force=tuple(force), torque=tuple(torque), duration=float(duration)
        ))

    def pause(self) -> None:
        self._cmd_queue.put(_Pause())

    def resume(self) -> None:
        self._cmd_queue.put(_Resume())

    def step(self, n: int = 1) -> None:
        self._cmd_queue.put(_Step(n=int(n)))

    def reset(self) -> None:
        self._cmd_queue.put(_Reset())

    def close(self) -> None:
        self._cmd_queue.put(_Stop())
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def current_robot(self) -> str:
        return self._current_robot

    @property
    def joints(self) -> list[JointInfo]:
        return list(self._current_joints)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="interactive-sim"
                )
                self._thread.start()
                log.info("InteractiveSim: thread started")

    def _run(self) -> None:
        model: mujoco.MjModel | None = None
        data: mujoco.MjData | None = None
        force_expiry: dict[int, float] = {}  # body_id -> sim time when xfrc zeroed
        frame_batch: list[np.ndarray] = []
        ts_batch: list[float] = []
        paused = False
        pending_steps = 0
        sim_time = 0.0

        while True:
            # --- Drain all pending commands (never block physics for UI) ---
            try:
                while True:
                    cmd = self._cmd_queue.get_nowait()
                    if isinstance(cmd, _Stop):
                        log.info("InteractiveSim: stop received")
                        return
                    elif isinstance(cmd, _LoadRobot):
                        try:
                            new_model = self._compile(cmd)
                        except Exception as exc:
                            log.warning(f"InteractiveSim: load failed — {exc}")
                            continue
                        model = new_model
                        data = mujoco.MjData(model)
                        mujoco.mj_resetData(model, data)
                        self._current_robot = cmd.name
                        self._current_joints, self._current_bodies = _introspect_model(model)
                        self._diverged = False
                        force_expiry.clear()
                        frame_batch.clear()
                        ts_batch.clear()
                        sim_time = 0.0
                        paused = False
                        self._emit_meta()
                        log.info(
                            f"InteractiveSim: loaded '{cmd.name}' "
                            f"(nq={model.nq}, nu={model.nu})"
                        )
                    elif model is None:
                        # Other commands are no-ops until a robot is loaded
                        continue
                    elif isinstance(cmd, _JointTarget):
                        self._apply_joint_target(model, data, cmd.joint, cmd.value)
                    elif isinstance(cmd, _ApplyForce):
                        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cmd.body)
                        if b_id >= 0:
                            data.xfrc_applied[b_id, :3] = cmd.force
                            data.xfrc_applied[b_id, 3:] = cmd.torque
                            force_expiry[b_id] = sim_time + cmd.duration
                    elif isinstance(cmd, _Pause):
                        paused = True
                    elif isinstance(cmd, _Resume):
                        paused = False
                    elif isinstance(cmd, _Step):
                        pending_steps += cmd.n
                    elif isinstance(cmd, _Reset):
                        mujoco.mj_resetData(model, data)
                        force_expiry.clear()
                        sim_time = 0.0
                        self._diverged = False
            except queue.Empty:
                pass

            if model is None or data is None:
                time.sleep(0.05)
                continue

            # --- Decide whether to step ---
            should_step = (not paused) or (pending_steps > 0)
            if pending_steps > 0:
                pending_steps -= 1

            if should_step:
                # Expire any timed-out xfrc forces
                if force_expiry:
                    for b_id in list(force_expiry.keys()):
                        if sim_time >= force_expiry[b_id]:
                            data.xfrc_applied[b_id, :] = 0.0
                            del force_expiry[b_id]

                try:
                    mujoco.mj_step(model, data)
                except Exception as exc:
                    log.warning(f"InteractiveSim: mj_step error — {exc}")
                    self._diverged = True
                    paused = True
                    continue

                sim_time += model.opt.timestep

                if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                    self._diverged = True
                    paused = True
                    log.warning("InteractiveSim: physics diverged, pausing")
                    continue

                self._latest_qpos = data.qpos.copy()

                frame_batch.append(data.qpos.astype(np.float32).copy())
                ts_batch.append(float(sim_time))
                if len(frame_batch) >= self._batch_size:
                    self._emit_batch(model.nq, frame_batch, ts_batch)
                    frame_batch.clear()
                    ts_batch.clear()

                # Sleep to pace physics (rough — good enough for visual playback)
                time.sleep(max(0.0, self._target_dt))
            else:
                # Paused: flush any pending batch and sleep briefly
                if frame_batch:
                    self._emit_batch(model.nq, frame_batch, ts_batch)
                    frame_batch.clear()
                    ts_batch.clear()
                time.sleep(0.02)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile(cmd: _LoadRobot) -> mujoco.MjModel:
        if cmd.mjcf_path:
            return mujoco.MjModel.from_xml_path(cmd.mjcf_path)
        if cmd.mjcf_xml:
            return mujoco.MjModel.from_xml_string(cmd.mjcf_xml)
        raise ValueError("LoadRobot requires either mjcf_path or mjcf_xml")

    @staticmethod
    def _apply_joint_target(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_name: str,
        value: float,
    ) -> None:
        """Apply a slider value.

        Preferred path: find an actuator on this joint and write to ``data.ctrl``.
        Fallback: if no actuator, directly set the first qpos slot of the joint
        (useful for passive kinematic poking but will be overwritten by physics
        next step for unactuated degrees of freedom).
        """
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if j_id < 0:
            return

        # Find actuator driving this joint
        for a_id in range(model.nu):
            if int(model.actuator_trnid[a_id, 0]) == j_id:
                if model.actuator_ctrllimited[a_id]:
                    lo = float(model.actuator_ctrlrange[a_id, 0])
                    hi = float(model.actuator_ctrlrange[a_id, 1])
                    value = float(np.clip(value, lo, hi))
                data.ctrl[a_id] = value
                return

        # No actuator — set qpos directly for manual posing
        qadr = int(model.jnt_qposadr[j_id])
        data.qpos[qadr] = value

    def _emit_batch(self, nq: int, frames: list[np.ndarray], timesteps: list[float]) -> None:
        if self._frame_cb is None or not frames:
            return
        arr = np.stack(frames, axis=0)
        try:
            self._frame_cb(self._current_robot, arr, list(timesteps))
        except Exception as exc:
            log.warning(f"InteractiveSim frame callback failed: {exc}")

    def _emit_meta(self) -> None:
        if self._meta_cb is None:
            return
        meta = {
            "name": self._current_robot,
            "joints": [
                {
                    "name": j.name,
                    "type": j.type,
                    "range": list(j.range),
                    "ctrl_range": list(j.ctrl_range),
                    "actuator": j.actuator,
                }
                for j in self._current_joints
            ],
            "bodies": list(self._current_bodies),
        }
        try:
            self._meta_cb(meta)
        except Exception as exc:
            log.warning(f"InteractiveSim meta callback failed: {exc}")
