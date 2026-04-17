"""LivePolicyPlayer: 60 Hz MuJoCo loop that streams qpos batches to Isaac Sim.

Creates its own MuJoCo model/data (does not share state with InteractiveSim) so
it can run concurrently with Design-mode simulation without locking conflicts.

Usage::

    from robo_garden.core.live_player import LivePolicyPlayer

    player = LivePolicyPlayer(
        policy_fn=load_policy_fn(variant),
        mjcf_xml=merged_xml,
        robot_name="go2_walker",
        frame_callback=bridge.stream_qpos_batch,
    )
    player.start()      # returns immediately; loop is daemon thread
    # ... later:
    player.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import mujoco
import numpy as np

log = logging.getLogger(__name__)


class LivePolicyPlayer:
    """Persistent daemon thread: steps MuJoCo with a policy, streams frames.

    Parameters
    ----------
    policy_fn:
        ``obs_array → action_array`` callable.  obs shape (nq + nv,),
        action shape (nu,) in [-1, 1] — scaled to ctrlrange internally.
    mjcf_xml:
        Merged MJCF XML string defining the physics model.
    robot_name:
        Robot identifier forwarded in each ``frame_callback`` call.
    frame_callback:
        ``(robot_name, frames_Nxnq, timesteps_list) → None``.  Called every
        ``batch_size`` physics steps.  Runs on the physics thread — keep it fast.
    physics_hz:
        Target physics steps per second (default 500).
    batch_size:
        Physics steps per outbound frame batch (default 16 → ~30 Hz packets).
    """

    def __init__(
        self,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        mjcf_xml: str,
        robot_name: str,
        frame_callback: Callable,
        physics_hz: int = 500,
        batch_size: int = 16,
    ) -> None:
        self._policy_fn = policy_fn
        self._mjcf_xml = mjcf_xml
        self._robot_name = robot_name
        self._frame_cb = frame_callback
        self._target_dt = 1.0 / float(physics_hz)
        self._batch_size = batch_size

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # unpaused by default

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the playback loop in a daemon thread.  No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._paused.set()
        self._thread = threading.Thread(
            target=self._loop,
            name="live-policy-player",
            daemon=True,
        )
        self._thread.start()
        log.info(f"LivePolicyPlayer: started for {self._robot_name!r}")

    def stop(self) -> None:
        """Signal the loop to stop and wait (up to 2 s) for the thread to join."""
        self._stop_event.set()
        self._paused.set()   # unblock if waiting on pause
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info(f"LivePolicyPlayer: stopped for {self._robot_name!r}")

    def pause(self) -> None:
        """Suspend physics steps (frames stop streaming)."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume physics steps after a pause."""
        self._paused.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Physics loop (daemon thread)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            model = mujoco.MjModel.from_xml_string(self._mjcf_xml)
            data = mujoco.MjData(model)
        except Exception as exc:
            log.error(f"LivePolicyPlayer: failed to compile model — {exc}")
            return

        nq = model.nq
        nu = model.nu

        # Pre-compute action scale/bias for [-1,1] → ctrlrange mapping
        if nu > 0 and model.actuator_ctrlrange.shape[0] == nu:
            act_lo = model.actuator_ctrlrange[:, 0].copy()
            act_hi = model.actuator_ctrlrange[:, 1].copy()
        else:
            act_lo = np.full(max(nu, 1), -1.0)
            act_hi = np.full(max(nu, 1), 1.0)

        act_scale = 0.5 * (act_hi - act_lo)
        act_bias = 0.5 * (act_hi + act_lo)

        mujoco.mj_resetData(model, data)

        frame_buf: list[np.ndarray] = []
        time_buf: list[float] = []
        sim_time = 0.0

        while not self._stop_event.is_set():
            tick_start = time.perf_counter()

            if not self._paused.wait(timeout=0.05):
                # Paused — re-check stop every 50 ms
                continue

            # Build observation
            obs = np.concatenate(
                [data.qpos, data.qvel], axis=0, dtype=np.float32
            )

            # Policy inference
            try:
                raw_action = np.asarray(
                    self._policy_fn(obs), dtype=np.float32
                ).reshape(-1)
                if raw_action.shape[0] != nu and nu > 0:
                    raw_action = np.zeros(nu, dtype=np.float32)
            except Exception as exc:
                log.debug(f"LivePolicyPlayer: policy error — {exc}")
                raw_action = np.zeros(nu, dtype=np.float32)

            # Apply scaled ctrl
            if nu > 0:
                scaled = np.clip(raw_action, -1.0, 1.0) * act_scale + act_bias
                data.ctrl[:] = scaled

            mujoco.mj_step(model, data)
            sim_time += float(model.opt.timestep)

            frame_buf.append(data.qpos[:nq].copy())
            time_buf.append(sim_time)

            if len(frame_buf) >= self._batch_size:
                frames = np.stack(frame_buf)
                ts = list(time_buf)
                frame_buf.clear()
                time_buf.clear()
                try:
                    self._frame_cb(self._robot_name, frames, ts)
                except Exception as exc:
                    log.debug(f"LivePolicyPlayer: frame_callback error — {exc}")

            # Soft rate-limiting
            elapsed = time.perf_counter() - tick_start
            remaining = self._target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # Flush remaining frames on clean exit
        if frame_buf:
            try:
                self._frame_cb(
                    self._robot_name, np.stack(frame_buf), list(time_buf)
                )
            except Exception:
                pass
