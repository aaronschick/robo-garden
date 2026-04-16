"""WebSocket message constructors for the Isaac Sim bridge protocol."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

# Message type constants
# --- Existing one-way robo-garden -> Isaac Sim messages ---
LOAD_ROBOT = "LOAD_ROBOT"
LOAD_ROBOT_ACK = "LOAD_ROBOT_ACK"
SIM_FRAME_BATCH = "SIM_FRAME_BATCH"
SIM_END = "SIM_END"
# Training lifecycle:
#   TRAIN_RUN_START  — fired once per run with static metadata (robot, algo,
#                      total_timesteps, started_at).  UI uses this to reset
#                      its reward sparkline.
#   TRAIN_UPDATE     — periodic progress (timestep, mean_reward, best_reward,
#                      elapsed_s, timesteps_per_second).
#   TRAIN_RUN_END    — final reward curve, checkpoint path, success flag.
#   TRAIN_HISTORY    — seed of recent runs sent on Studio connect.
TRAIN_RUN_START = "TRAIN_RUN_START"
TRAIN_UPDATE = "TRAIN_UPDATE"
TRAIN_RUN_END = "TRAIN_RUN_END"
TRAIN_HISTORY = "TRAIN_HISTORY"
PING = "PING"
PONG = "PONG"

# --- Bidirectional messages added for the Design Studio ---
# Isaac Sim UI -> robo-garden backend
CHAT_MESSAGE = "CHAT_MESSAGE"       # {"text": str}
JOINT_TARGET = "JOINT_TARGET"       # {"joint": str, "value": float}
APPLY_FORCE = "APPLY_FORCE"         # {"body": str, "force": [fx,fy,fz], "torque": [tx,ty,tz]}
PAUSE = "PAUSE"                     # {}
RESUME = "RESUME"                   # {}
STEP = "STEP"                       # {"n": int}
RESET = "RESET"                     # {}
APPROVE_DESIGN = "APPROVE_DESIGN"   # {"robot_name": str, "environment_name": str, "notes": str}
UNAPPROVE_DESIGN = "UNAPPROVE_DESIGN"  # {}  — flip back to design phase

# robo-garden backend -> Isaac Sim UI
CHAT_REPLY = "CHAT_REPLY"           # {"text": str, "session_id": str}
TOOL_STATUS = "TOOL_STATUS"         # {"tool": str, "status": str, "detail": str}
TOOL_RESULT = "TOOL_RESULT"         # {"tool": str, "summary": str, "success": bool, "result": dict}
PHASE_CHANGED = "PHASE_CHANGED"     # {"phase": "design"|"training", "approved_robot": str|None, ...}
ROBOT_META = "ROBOT_META"           # {"name": str, "joints": [{"name": str, "range": [lo,hi], "type": str}], "bodies": [str]}
GATE_STATUS = "GATE_STATUS"         # {"robot_loaded": bool, "env_loaded": bool, "sim_stable": bool, "can_approve": bool}


def make_load_robot(name: str, path: Path, fmt: str = "mjcf") -> dict:
    """Tell Isaac Sim to import a robot from the filesystem.

    The path is always emitted as a resolved forward-slash (posix) string so
    Isaac Sim's MJCF/URDF importers never see Windows-style backslashes,
    which can cause their asset-resolver to drop referenced mesh files.
    """
    try:
        posix_path = Path(path).resolve().as_posix()
    except OSError:
        # .resolve() raises on non-existent paths only when strict=True (not
        # the default here), but also on Windows when the path is malformed.
        # Fall back to a best-effort as_posix() without resolving symlinks.
        posix_path = Path(path).as_posix()
    return {
        "type": LOAD_ROBOT,
        "name": name,
        "path": posix_path,
        "format": fmt,
        "ts": time.time(),
    }


def make_sim_frame_batch(
    robot_name: str,
    frames: np.ndarray,
    timesteps: list[float],
    nq: int,
) -> dict:
    """Batch of joint position frames for playback.

    For robots with nq < 12, uses plain JSON arrays (debuggable).
    For larger robots, uses base64-encoded float32 for efficiency.
    """
    msg: dict = {
        "type": SIM_FRAME_BATCH,
        "robot_name": robot_name,
        "batch_size": len(frames),
        "nq": nq,
        "timesteps": timesteps,
        "ts": time.time(),
    }
    if nq < 12:
        msg["qpos_json"] = frames.tolist()
    else:
        msg["qpos_b64"] = base64.b64encode(
            frames.astype(np.float32).tobytes()
        ).decode("ascii")
    return msg


def make_sim_end(robot_name: str, stable: bool, diverged: bool, summary: dict) -> dict:
    """Signal end of simulation playback."""
    return {
        "type": SIM_END,
        "robot_name": robot_name,
        "stable": stable,
        "diverged": diverged,
        "summary": summary,
        "ts": time.time(),
    }


def make_train_run_start(
    run_id: str,
    robot_name: str,
    environment_name: str,
    algorithm: str,
    total_timesteps: int,
    reward_function_id: str = "",
    **kwargs,
) -> dict:
    """Announce the start of a training run — UI resets its progress panel."""
    return {
        "type": TRAIN_RUN_START,
        "run_id": run_id,
        "robot_name": robot_name,
        "environment_name": environment_name,
        "algorithm": algorithm,
        "total_timesteps": int(total_timesteps),
        "reward_function_id": reward_function_id,
        "started_at": time.time(),
        "ts": time.time(),
        **kwargs,
    }


def make_train_update(
    robot_name: str,
    timestep: int,
    mean_reward: float,
    *,
    run_id: str = "",
    best_reward: float | None = None,
    elapsed_s: float | None = None,
    total_timesteps: int | None = None,
    algorithm: str = "",
    timesteps_per_second: float | None = None,
    **kwargs,
) -> dict:
    """Periodic training progress update driving the Studio progress panel."""
    msg: dict = {
        "type": TRAIN_UPDATE,
        "run_id": run_id,
        "robot_name": robot_name,
        "timestep": int(timestep),
        "mean_reward": float(mean_reward),
        "algorithm": algorithm,
        "ts": time.time(),
    }
    if best_reward is not None:
        msg["best_reward"] = float(best_reward)
    if elapsed_s is not None:
        msg["elapsed_s"] = float(elapsed_s)
    if total_timesteps is not None:
        msg["total_timesteps"] = int(total_timesteps)
    if timesteps_per_second is not None:
        msg["timesteps_per_second"] = float(timesteps_per_second)
    msg.update(kwargs)
    return msg


def make_train_run_end(
    run_id: str,
    robot_name: str,
    success: bool,
    *,
    best_reward: float | None = None,
    training_time_seconds: float | None = None,
    total_timesteps: int | None = None,
    checkpoint_path: str = "",
    reward_curve: list[tuple[int, float]] | None = None,
    error: str = "",
    **kwargs,
) -> dict:
    """Signal end of a training run — UI freezes the progress panel + appends history."""
    return {
        "type": TRAIN_RUN_END,
        "run_id": run_id,
        "robot_name": robot_name,
        "success": bool(success),
        "best_reward": None if best_reward is None else float(best_reward),
        "training_time_seconds": (
            None if training_time_seconds is None else float(training_time_seconds)
        ),
        "total_timesteps": (
            None if total_timesteps is None else int(total_timesteps)
        ),
        "checkpoint_path": checkpoint_path,
        "reward_curve": [
            [int(ts), float(r)] for ts, r in (reward_curve or [])
        ],
        "error": error,
        "ended_at": time.time(),
        "ts": time.time(),
        **kwargs,
    }


def make_train_history(runs: list[dict]) -> dict:
    """Seed the Studio's run-history list on connect (most recent first)."""
    return {
        "type": TRAIN_HISTORY,
        "runs": list(runs),
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# Studio messages (UI -> backend)
# ---------------------------------------------------------------------------


def make_chat_message(text: str) -> dict:
    """User typed a chat message in the Studio; forward to Claude."""
    return {"type": CHAT_MESSAGE, "text": text, "ts": time.time()}


def make_joint_target(joint: str, value: float) -> dict:
    """User moved a joint slider; apply as ctrl target in InteractiveSim."""
    return {"type": JOINT_TARGET, "joint": joint, "value": float(value), "ts": time.time()}


def make_apply_force(
    body: str,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    duration: float = 0.1,
) -> dict:
    """User clicked a body in Apply-Force mode; push it for `duration` seconds."""
    return {
        "type": APPLY_FORCE,
        "body": body,
        "force": list(force),
        "torque": list(torque),
        "duration": float(duration),
        "ts": time.time(),
    }


def make_pause() -> dict:
    return {"type": PAUSE, "ts": time.time()}


def make_resume() -> dict:
    return {"type": RESUME, "ts": time.time()}


def make_step(n: int = 1) -> dict:
    return {"type": STEP, "n": int(n), "ts": time.time()}


def make_reset() -> dict:
    return {"type": RESET, "ts": time.time()}


def make_approve_design(robot_name: str, environment_name: str, notes: str = "") -> dict:
    """User clicked Promote-to-Training; tell backend to run approve_for_training."""
    return {
        "type": APPROVE_DESIGN,
        "robot_name": robot_name,
        "environment_name": environment_name,
        "notes": notes,
        "ts": time.time(),
    }


def make_unapprove_design() -> dict:
    return {"type": UNAPPROVE_DESIGN, "ts": time.time()}


# ---------------------------------------------------------------------------
# Studio messages (backend -> UI)
# ---------------------------------------------------------------------------


def make_chat_reply(text: str, session_id: str = "") -> dict:
    return {"type": CHAT_REPLY, "text": text, "session_id": session_id, "ts": time.time()}


def make_tool_status(tool: str, status: str, detail: str = "") -> dict:
    return {"type": TOOL_STATUS, "tool": tool, "status": status, "detail": detail, "ts": time.time()}


def make_tool_result(tool: str, summary: str, success: bool, result: dict) -> dict:
    return {
        "type": TOOL_RESULT,
        "tool": tool,
        "summary": summary,
        "success": bool(success),
        "result": result,
        "ts": time.time(),
    }


def make_phase_changed(
    phase: str,
    approved_robot: str | None = None,
    approved_environment: str | None = None,
) -> dict:
    return {
        "type": PHASE_CHANGED,
        "phase": phase,
        "approved_robot": approved_robot,
        "approved_environment": approved_environment,
        "ts": time.time(),
    }


def make_robot_meta(name: str, joints: list[dict], bodies: list[str]) -> dict:
    """Describe the structure of the currently loaded robot for the Studio UI.

    ``joints`` is a list of ``{"name": str, "range": [lo, hi], "type": str,
    "ctrl_range": [lo, hi], "actuator": str}`` entries — enough for the UI to
    auto-generate sliders with correct limits.  ``bodies`` is a flat list of
    body names (used by Apply-Force picking).
    """
    return {
        "type": ROBOT_META,
        "name": name,
        "joints": joints,
        "bodies": bodies,
        "ts": time.time(),
    }


def make_gate_status(
    robot_loaded: bool,
    env_loaded: bool,
    sim_ran: bool,
    sim_stable: bool,
    can_approve: bool,
    phase: str = "design",
    missing: list[str] | None = None,
) -> dict:
    """Gate checklist broadcast after any state change — drives the approval panel."""
    return {
        "type": GATE_STATUS,
        "robot_loaded": bool(robot_loaded),
        "env_loaded": bool(env_loaded),
        "sim_ran": bool(sim_ran),
        "sim_stable": bool(sim_stable),
        "can_approve": bool(can_approve),
        "phase": phase,
        "missing": list(missing or []),
        "ts": time.time(),
    }
