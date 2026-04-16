"""Model save/load for JAX params and PyTorch state dicts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_checkpoint(params, path: Path, metadata: dict | None = None) -> Path:
    """Save training checkpoint (JAX params or PyTorch state dict)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    if params is not None:
        try:
            import orbax.checkpoint as ocp  # type: ignore
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(path, params)
        except ImportError:
            import numpy as np
            import jax
            try:
                leaves, treedef = jax.tree_util.tree_flatten(params)
                np_leaves = [np.array(l) for l in leaves]
                np.savez(path.with_suffix(".npz"), *np_leaves)
                import pickle
                treedef_path = path.with_suffix(".treedef")
                treedef_path.write_bytes(pickle.dumps(treedef))
            except Exception:
                import pickle
                path.with_suffix(".pkl").write_bytes(pickle.dumps(params))

    log.info(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(path: Path) -> dict:
    """Load a training checkpoint."""
    meta_path = path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    params = None
    try:
        import orbax.checkpoint as ocp  # type: ignore
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(path)
    except (ImportError, Exception):
        npz_path = path.with_suffix(".npz")
        pkl_path = path.with_suffix(".pkl")
        if npz_path.exists():
            import numpy as np
            import pickle
            data = np.load(npz_path)
            leaves = [data[k] for k in sorted(data.files)]
            treedef_path = path.with_suffix(".treedef")
            if treedef_path.exists():
                treedef = pickle.loads(treedef_path.read_bytes())
                import jax
                params = jax.tree_util.tree_unflatten(treedef, leaves)
            else:
                params = leaves
        elif pkl_path.exists():
            import pickle
            params = pickle.loads(pkl_path.read_bytes())

    return {"params": params, "metadata": metadata}
