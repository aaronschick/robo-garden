"""Dispatch a Claude-driven training job into WSL2 for GPU-backed training.

This module is the Windows-side half of the "gym mode on GPU" pipeline:

   ``handle_train`` (Windows, Claude session)
      └── ``run_in_wsl(...)`` in this module
            └── spawns ``wsl.exe -d Ubuntu-22.04 -- bash -c "… uv run robo-garden
                --mode train --wsl-worker <job_dir>"``
                  └── ``_run_wsl_worker`` in ``cli.py`` (Linux-side)
                        └── ``MuJoCoMJXEngine.train(...)``  (Brax/JAX/GPU)

The Linux-side worker and the Windows-side dispatcher communicate via two
files inside ``workspace/_wsl_jobs/<run_id>/``:

* ``job.json``    — input spec: robot_xml / env_xml paths, reward source,
                    training hyperparameters.
* ``result.json`` — written by the worker on exit with best_reward,
                    reward_curve, checkpoint_path, success flag, error text.

Progress is streamed over stdout as JSONL lines prefixed with
``__RG_PROGRESS__`` so the Windows-side parser can forward each tick to the
in-process ``progress_callback`` (which in turn drives the Isaac Sim
training panel and the persistent run history).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Cache for WSL auto-detection. None = not yet probed.
_WSL_AVAILABLE: bool | None = None

# Shared sentinel strings. Must match what cli.py --wsl-worker emits.
PROGRESS_PREFIX = "__RG_PROGRESS__"
RESULT_FILE_NAME = "result.json"
JOB_FILE_NAME = "job.json"


@dataclass
class WSLJobSpec:
    """Everything the Linux-side worker needs to run training.

    Paths are stored as Windows absolute paths; the worker translates them
    to ``/mnt/c/...`` on its side. Reward source is embedded directly (not
    a path) so the worker doesn't depend on the Windows filesystem being
    readable mid-training.
    """

    run_id: str
    robot_xml: str
    env_mjcf: str
    reward_fn_code: str
    robot_name: str
    environment_name: str
    algorithm: str
    total_timesteps: int
    num_envs: int
    max_episode_steps: int

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "robot_xml": self.robot_xml,
            "env_mjcf": self.env_mjcf,
            "reward_fn_code": self.reward_fn_code,
            "robot_name": self.robot_name,
            "environment_name": self.environment_name,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "num_envs": self.num_envs,
            "max_episode_steps": self.max_episode_steps,
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "WSLJobSpec":
        return cls(
            run_id=blob["run_id"],
            robot_xml=blob["robot_xml"],
            env_mjcf=blob.get("env_mjcf", ""),
            reward_fn_code=blob.get("reward_fn_code", ""),
            robot_name=blob.get("robot_name", ""),
            environment_name=blob.get("environment_name", ""),
            algorithm=blob.get("algorithm", "ppo"),
            total_timesteps=int(blob["total_timesteps"]),
            num_envs=int(blob.get("num_envs", 64)),
            max_episode_steps=int(blob.get("max_episode_steps", 1000)),
        )


def _windows_to_wsl_path(p: str) -> str:
    """``C:\\Users\\foo`` → ``/mnt/c/Users/foo`` so the Linux side can cd into it."""
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _jobs_root() -> Path:
    from robo_garden.config import WORKSPACE_DIR

    root = WORKSPACE_DIR / "_wsl_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_job(spec: WSLJobSpec) -> Path:
    """Persist *spec* under ``workspace/_wsl_jobs/<run_id>/`` and return the dir."""
    job_dir = _jobs_root() / spec.run_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / JOB_FILE_NAME).write_text(
        json.dumps(spec.to_json(), indent=2),
        encoding="utf-8",
    )
    return job_dir


def load_spec(job_dir: Path) -> WSLJobSpec:
    """Linux-side entry: load the spec dumped by ``stage_job``."""
    blob = json.loads((job_dir / JOB_FILE_NAME).read_text(encoding="utf-8"))
    return WSLJobSpec.from_json(blob)


def write_result(job_dir: Path, result: dict[str, Any]) -> None:
    """Linux-side: write the final result blob so the Windows side can read it."""
    (job_dir / RESULT_FILE_NAME).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def emit_progress(step: int, metrics: dict[str, Any]) -> None:
    """Linux-side helper: emit a JSONL progress line the dispatcher parses.

    Printed to stdout (not stderr) so it survives the subprocess pipe; the
    Windows side identifies these lines by the ``__RG_PROGRESS__`` prefix
    and forwards them to the in-process callback.
    """
    payload = {"step": int(step), **{k: float(v) if isinstance(v, (int, float)) else v
                                       for k, v in metrics.items()}}
    # flush immediately so the Windows side gets updates in real time.
    print(f"{PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


def run_in_wsl(
    *,
    run_id: str,
    robot_xml: str,
    env_mjcf: str,
    reward_fn_code: str,
    robot_name: str,
    environment_name: str,
    algorithm: str,
    total_timesteps: int,
    num_envs: int,
    max_episode_steps: int,
    progress_callback: Callable[[int, dict], None] | None = None,
    distro: str | None = None,
) -> dict[str, Any]:
    """Windows-side entry: run a training job inside WSL2 and return results.

    Blocks until the WSL subprocess exits. Progress ticks from the worker
    are parsed out of stdout and forwarded to ``progress_callback`` so the
    existing Isaac Sim bridge and history plumbing keep working as if the
    training were in-process.

    Returns a dict with the same shape as ``TrainingResult``-via-``handle_train``:
        {"success": bool, "best_reward": float, "reward_curve": list,
         "checkpoint_path": str, "training_time_seconds": float, "error": str}
    """
    from robo_garden.config import PROJECT_ROOT

    distro = distro or os.environ.get("ROBO_GARDEN_WSL_DISTRO", "Ubuntu-22.04")

    spec = WSLJobSpec(
        run_id=run_id,
        robot_xml=robot_xml,
        env_mjcf=env_mjcf,
        reward_fn_code=reward_fn_code,
        robot_name=robot_name,
        environment_name=environment_name,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
    )
    job_dir = stage_job(spec)

    wsl_project = _windows_to_wsl_path(str(PROJECT_ROOT))
    wsl_job_dir = _windows_to_wsl_path(str(job_dir))

    # Same preamble as _launch_wsl_training in cli.py: source the user's
    # shell profile so uv is on PATH, pin the ext4 venv location so we
    # don't clobber the Windows-side .venv, and fix COLUMNS so Rich doesn't
    # shred stdout one character at a time when captured through a pipe.
    source_profile = (
        'source "$HOME/.profile" 2>/dev/null; '
        'source "$HOME/.cargo/env" 2>/dev/null; '
        'export PATH="$HOME/.local/bin:$PATH"; '
        'export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1; '
        'export COLUMNS=120 LINES=40; '
        'export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/robo-garden/venv}"; '
    )
    worker_cmd = (
        f"cd '{wsl_project}' && "
        f"uv run robo-garden --no-isaac --mode train --wsl-worker '{wsl_job_dir}'"
    )
    full_cmd = source_profile + worker_cmd

    log.info(f"Dispatching training to WSL (distro={distro}, run_id={run_id})")
    log.info(f"Job dir: {job_dir}")

    start = time.time()
    try:
        proc = subprocess.Popen(
            ["wsl.exe", "-d", distro, "--", "bash", "-c", full_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "best_reward": float("-inf"),
            "reward_curve": [],
            "checkpoint_path": "",
            "training_time_seconds": 0.0,
            "error": (
                "wsl.exe not found on PATH. Install WSL2 + Ubuntu-22.04: "
                "'wsl --install -d Ubuntu-22.04'."
            ),
        }

    # Pump stdout. Any line starting with __RG_PROGRESS__ is a structured
    # callback tick; everything else is human-readable log output we echo
    # straight through so the user sees setup/error chatter too.
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if line.startswith(PROGRESS_PREFIX):
            if progress_callback is not None:
                try:
                    payload = json.loads(line[len(PROGRESS_PREFIX):])
                    step = int(payload.pop("step", 0))
                    progress_callback(step, payload)
                except Exception as exc:  # pragma: no cover — best-effort parsing
                    log.debug(f"Could not parse progress line: {line!r} ({exc})")
        else:
            # Echo non-progress lines to our stdout so the Claude-driven user
            # still sees dependency imports, warnings, tracebacks.
            print(line, flush=True)

    proc.wait()
    elapsed = time.time() - start

    result_path = job_dir / RESULT_FILE_NAME
    if proc.returncode != 0 or not result_path.exists():
        # Subprocess crashed before it could write result.json; synthesise
        # a failure record so the caller gets uniform shape. Keeping the job
        # dir around (we don't delete it) makes post-mortem easy.
        rc = int(proc.returncode or 0)
        rc_hint = ""
        if rc == 130:
            # 128 + SIGINT(2): Ctrl+C, closing the terminal, or session teardown —
            # not the Linux OOM killer (that is usually SIGKILL → exit 137).
            rc_hint = (
                " Code 130 is SIGINT: the worker was interrupted (often Ctrl+C or "
                "the host terminal/window closed), not a normal training failure. "
            )
        elif rc == 137:
            rc_hint = " Code 137 is often SIGKILL (e.g. Linux OOM killer or `kill -9`). "

        spacer = f" {rc_hint.strip()} " if rc_hint else " "

        return {
            "success": False,
            "best_reward": float("-inf"),
            "reward_curve": [],
            "checkpoint_path": "",
            "training_time_seconds": elapsed,
            "error": (
                f"WSL worker exited with code {rc} and did not "
                f"write {RESULT_FILE_NAME}.{spacer}"
                f"Inspect {job_dir} for the staged job spec; from WSL run:\n"
                f"  cd {wsl_job_dir} && uv run robo-garden --no-isaac --mode train --wsl-worker ."
            ),
        }

    try:
        result_blob = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "success": False,
            "best_reward": float("-inf"),
            "reward_curve": [],
            "checkpoint_path": "",
            "training_time_seconds": elapsed,
            "error": f"Could not read result.json: {exc}",
        }

    # Back-fill training_time_seconds if the worker didn't (shouldn't happen
    # but defensive). We trust the worker's value otherwise since it's more
    # precise than our wall-clock timing.
    result_blob.setdefault("training_time_seconds", elapsed)

    # Stream the post-training rollout frames to Isaac viewport so the user
    # can see the robot's behaviour after GPU training completes.
    wsl_frames_path = result_blob.get("rollout_frames_path", "")
    if wsl_frames_path:
        win_frames_path = _wsl_to_windows_path(wsl_frames_path)
        try:
            from pathlib import Path as _P
            import numpy as _np
            from robo_garden.isaac import get_bridge as _get_bridge

            frames_file = _P(win_frames_path)
            if frames_file.exists():
                data = _np.load(str(frames_file))
                qpos = data["qpos"]
                timesteps = data["timesteps"].tolist()
                bridge = _get_bridge()
                if bridge.connected and len(qpos) > 0:
                    bridge.stream_qpos_batch(spec.robot_name, qpos, timesteps)
                    log.info(
                        f"Streamed {len(qpos)} post-training rollout frames to Isaac viewport."
                    )
        except Exception as exc:
            log.debug(f"Could not stream WSL rollout frames to Isaac: {exc}")

    return result_blob


def _wsl_to_windows_path(p: str) -> str:
    """/mnt/c/Users/foo → C:\\Users\\foo for reading WSL-written files on Windows."""
    if p.startswith("/mnt/") and len(p) > 6:
        drive = p[5].upper()
        rest = p[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return p


def _detect_wsl(distro: str | None = None) -> bool:
    """Check if wsl.exe is on PATH and *distro* is registered.

    Uses ``wsl.exe --list --quiet`` (exit 0 when distros present, UTF-16-LE
    output). Fast (<200 ms, no bash cold-start). Returns False on any error.
    """
    import shutil

    if shutil.which("wsl.exe") is None:
        return False

    target = distro or os.environ.get("ROBO_GARDEN_WSL_DISTRO", "Ubuntu-22.04")
    try:
        result = subprocess.run(
            ["wsl.exe", "--list", "--quiet"],
            capture_output=True,
            timeout=5,
            encoding="utf-16-le",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
        log.debug(f"_detect_wsl: {exc}")
        return False

    if result.returncode != 0:
        return False

    distros = [
        line.strip().strip("\x00")
        for line in result.stdout.splitlines()
        if line.strip().strip("\x00")
    ]
    found = target in distros
    log.debug(f"_detect_wsl: distros={distros!r}, target={target!r}, found={found}")
    return found


def is_enabled() -> bool:
    """Return True iff GPU training should be dispatched to WSL2.

    Three-state logic (always False on Linux):

    * ``ROBO_GARDEN_TRAIN_IN_WSL=0`` / ``false`` / ``no`` / ``off`` → False (force CPU)
    * ``ROBO_GARDEN_TRAIN_IN_WSL=1`` (any truthy value) → True (force WSL)
    * Unset (default) → auto-detect via ``_detect_wsl()``, result cached for
      the lifetime of the process.
    """
    global _WSL_AVAILABLE

    if sys.platform != "win32":
        return False

    flag = os.environ.get("ROBO_GARDEN_TRAIN_IN_WSL", "").strip().lower()

    if flag in ("0", "false", "no", "off"):
        return False
    if flag:  # non-empty, non-disable value → explicit enable
        return True

    # Unset → auto-detect, cached after first probe
    if _WSL_AVAILABLE is None:
        _WSL_AVAILABLE = _detect_wsl()
        if _WSL_AVAILABLE:
            log.info(
                "WSL2 auto-detected: GPU training will be dispatched to WSL2. "
                "Set ROBO_GARDEN_TRAIN_IN_WSL=0 to force CPU."
            )
        else:
            log.info(
                "WSL2 not detected: falling back to SB3 PPO (CPU). "
                "Run scripts/setup_wsl2.ps1 to enable GPU training."
            )

    return _WSL_AVAILABLE
