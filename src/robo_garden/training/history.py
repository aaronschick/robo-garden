"""Append-only training run history stored as JSONL.

Each line is a record written by ``handle_train`` when a run finishes, shaped
roughly like::

    {
        "run_id": "run_20260416_211000_a1b2c3",
        "robot_name": "go2_walker",
        "environment_name": "flat_trot_ground",
        "algorithm": "ppo",
        "total_timesteps": 200000,
        "best_reward": 4.73,
        "training_time_seconds": 312.4,
        "started_at": 1750000000.0,
        "ended_at": 1750000312.4,
        "success": true,
        "checkpoint_path": "workspace/checkpoints/sb3_ppo_1750000312",
        "reward_function_id": "reward_ab12cd34",
        "error": ""
    }

The file lives at ``workspace/runs/runs.jsonl`` and is never rewritten — we
only append — so history survives crashes and doubles as an audit trail.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from robo_garden.config import RUNS_DIR

log = logging.getLogger(__name__)

RUNS_FILE = RUNS_DIR / "runs.jsonl"


def append_run(record: dict, path: Path | None = None) -> None:
    """Append one run record to the history file.

    Best-effort — IO errors are logged but do not propagate, because we never
    want a history-write failure to blow up an otherwise successful run.
    """
    target = path or RUNS_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        log.warning(f"Could not append run record to {target}: {exc}")


def load_recent(limit: int = 20, path: Path | None = None) -> list[dict]:
    """Return the most recent ``limit`` run records, newest first.

    Skips malformed lines silently so a corrupted history line doesn't
    kill the Studio startup.
    """
    target = path or RUNS_FILE
    if not target.exists():
        return []

    records: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        log.warning(f"Could not read run history from {target}: {exc}")
        return []

    records.reverse()
    return records[:max(0, int(limit))]


def filter_runs(
    records: Iterable[dict],
    *,
    robot_name: str | None = None,
    successful_only: bool = False,
) -> list[dict]:
    """Optional post-filter used by history queries."""
    out = []
    for r in records:
        if robot_name and r.get("robot_name") != robot_name:
            continue
        if successful_only and not r.get("success", False):
            continue
        out.append(r)
    return out
