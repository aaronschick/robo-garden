"""WebSocket message constructors for the Isaac Sim bridge protocol."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np

# Message type constants
LOAD_ROBOT = "LOAD_ROBOT"
LOAD_ROBOT_ACK = "LOAD_ROBOT_ACK"
SIM_FRAME_BATCH = "SIM_FRAME_BATCH"
SIM_END = "SIM_END"
TRAIN_UPDATE = "TRAIN_UPDATE"
PING = "PING"
PONG = "PONG"


def make_load_robot(name: str, path: Path, fmt: str = "mjcf") -> dict:
    """Tell Isaac Sim to import a robot from the filesystem."""
    return {
        "type": LOAD_ROBOT,
        "name": name,
        "path": str(path),
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


def make_train_update(robot_name: str, timestep: int, mean_reward: float, **kwargs) -> dict:
    """Training progress update for Isaac Sim overlay display."""
    return {
        "type": TRAIN_UPDATE,
        "robot_name": robot_name,
        "timestep": timestep,
        "mean_reward": mean_reward,
        "ts": time.time(),
        **kwargs,
    }
