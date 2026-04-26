"""Dispatch a training job to Isaac Lab (Windows-native, no WSL needed).

Isaac Lab runs inside Isaac Sim's Python 3.11 venv at C:\\isaac-venv.  Unlike
the Brax/WSL path, no Linux subprocess is needed: Isaac Lab runs directly on
Windows via the NVIDIA Omniverse stack.

Usage (routed automatically by handle_train when robot is in ISAAC_LAB_ROBOTS):

    ROBO_GARDEN_ISAAC_PYTHON = C:\\isaac-venv\\Scripts\\python.exe  (default)

    result = run_in_isaac_lab(
        robot_name="urchin_v2",
        total_timesteps=500_000,
        num_envs=64,
        run_id="run_20260417_001",
        progress_callback=my_cb,
    )

The training script emits ``__RG_PROGRESS__ <json>`` lines to stdout; this
dispatcher parses them and forwards each tick to ``progress_callback``.
On exit the script writes ``result.json`` to the checkpoint directory;
this function reads it and returns the same result shape as wsl_dispatch.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

PROGRESS_PREFIX = "__RG_PROGRESS__"
RESULT_FILE_NAME = "result.json"

# Maps robot_name → absolute path to the Isaac Lab train.py script.
# Paths are relative to PROJECT_ROOT and resolved at call time.
_ISAAC_LAB_ROBOTS: dict[str, str] = {
    "urchin_v2": "workspace/robots/urchin_v2/scripts/train.py",
}


def _isaac_python() -> str:
    return os.environ.get(
        "ROBO_GARDEN_ISAAC_PYTHON",
        r"C:\isaac-venv\Scripts\python.exe",
    )


def is_enabled(robot_name: str) -> bool:
    """True iff *robot_name* has an Isaac Lab config AND the Isaac Python exists."""
    if robot_name not in _ISAAC_LAB_ROBOTS:
        return False
    if sys.platform != "win32":
        return False
    return Path(_isaac_python()).exists()


def run_in_isaac_lab(
    *,
    robot_name: str,
    total_timesteps: int,
    num_envs: int,
    run_id: str,
    progress_callback: Callable[[int, dict], None] | None = None,
) -> dict[str, Any]:
    """Spawn Isaac Lab training and block until done.

    Returns a dict with keys:
        success, best_reward, reward_curve, checkpoint_path,
        training_time_seconds, error
    """
    from robo_garden.config import PROJECT_ROOT, CHECKPOINTS_DIR

    train_script_rel = _ISAAC_LAB_ROBOTS.get(robot_name)
    if train_script_rel is None:
        return _failure(f"No Isaac Lab training script registered for '{robot_name}'.")

    train_script = (PROJECT_ROOT / train_script_rel).resolve()
    if not train_script.exists():
        return _failure(f"Isaac Lab train script not found: {train_script}")

    isaac_py = _isaac_python()
    checkpoint_dir = CHECKPOINTS_DIR / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        isaac_py,
        str(train_script),
        "--headless",
        "--num-envs", str(num_envs),
        "--timesteps", str(total_timesteps),
        "--checkpoint-dir", str(checkpoint_dir),
        "--run-id", run_id,
    ]

    log.info(f"Launching Isaac Lab training: {' '.join(cmd)}")
    start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        return _failure(
            f"Isaac Python not found at {isaac_py!r}. "
            "Install Isaac Sim: see isaac_server/README.md. "
            "Override with ROBO_GARDEN_ISAAC_PYTHON env var."
        )

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if line.startswith(PROGRESS_PREFIX):
            if progress_callback is not None:
                try:
                    payload = json.loads(line[len(PROGRESS_PREFIX):])
                    step = int(payload.pop("step", 0))
                    progress_callback(step, payload)
                except Exception as exc:
                    log.debug(f"Progress parse error: {line!r} ({exc})")
        else:
            print(line, flush=True)

    proc.wait()
    elapsed = time.time() - start

    result_path = checkpoint_dir / RESULT_FILE_NAME
    if proc.returncode != 0 or not result_path.exists():
        rc = int(proc.returncode or 0)
        return _failure(
            f"Isaac Lab process exited with code {rc} and no result.json. "
            f"Checkpoint dir: {checkpoint_dir}",
            training_time_seconds=elapsed,
        )

    try:
        blob = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _failure(f"Could not read result.json: {exc}", training_time_seconds=elapsed)

    blob.setdefault("training_time_seconds", elapsed)
    return blob


def _failure(msg: str, training_time_seconds: float = 0.0) -> dict[str, Any]:
    return {
        "success": False,
        "best_reward": float("-inf"),
        "reward_curve": [],
        "checkpoint_path": "",
        "training_time_seconds": training_time_seconds,
        "error": msg,
    }
