"""Model save/load for JAX params and PyTorch state dicts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_checkpoint(params, path: Path, metadata: dict | None = None) -> Path:
    """Save training checkpoint (JAX params or PyTorch state dict).

    TODO (Phase 4): Implement for both JAX and PyTorch backends.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    log.info(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(path: Path) -> dict:
    """Load a training checkpoint.

    TODO (Phase 4): Implement for both JAX and PyTorch backends.
    """
    meta_path = path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    return {"params": None, "metadata": metadata}
