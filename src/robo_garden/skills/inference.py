"""Load a trained policy checkpoint into a callable obs → action function.

Supports:
  - SB3 PPO checkpoints (``policy.zip`` inside the checkpoint dir)
  - Brax PPO checkpoints (Orbax directory + sibling ``.json`` with obs/action sizes)

Returns a zero-action fallback when neither format is detected so callers
always get a safe callable without needing try/except.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from robo_garden.skills import VariantSpec

log = logging.getLogger(__name__)


def load_policy_fn(
    variant: VariantSpec,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``obs_array → action_array`` for the variant's checkpoint.

    The returned callable is synchronous (no background threads) and safe to
    call repeatedly from a 60 Hz loop.
    """
    from robo_garden.config import WORKSPACE_DIR

    raw_path = variant.checkpoint_path
    p = Path(raw_path)
    if not p.is_absolute():
        p = WORKSPACE_DIR / p

    if not p.exists():
        log.warning(f"inference: checkpoint not found at {p!r}, using zero policy")
        return _zero_policy(0)

    # --- SB3 PPO (policy.zip inside the directory) ---
    if (p / "policy.zip").exists():
        return _load_sb3(p)

    # --- Brax / JAX PPO (Orbax dir + sibling JSON with obs/action sizes) ---
    meta_path = p.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "obs_size" in meta and "action_size" in meta:
                return _load_brax(p, meta)
        except Exception as exc:
            log.warning(f"inference: bad Brax metadata at {meta_path}: {exc}")

    log.warning(f"inference: unrecognised checkpoint format at {p!r}, using zero policy")
    return _zero_policy(0)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _zero_policy(action_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    def _fn(obs: np.ndarray) -> np.ndarray:
        return np.zeros(max(1, action_dim), dtype=np.float32)
    return _fn


def _load_sb3(checkpoint_dir: Path) -> Callable[[np.ndarray], np.ndarray]:
    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint_dir / "policy"), device="cpu")
        log.info(f"inference: loaded SB3 PPO from {checkpoint_dir}")

        def _fn(obs: np.ndarray) -> np.ndarray:
            obs2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action, _ = model.predict(obs2d, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return _fn
    except Exception as exc:
        log.warning(f"inference: SB3 load failed ({exc}), using zero policy")
        return _zero_policy(0)


def _load_brax(checkpoint_dir: Path, meta: dict) -> Callable[[np.ndarray], np.ndarray]:
    try:
        import jax
        import jax.numpy as jnp
        from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
        from robo_garden.training.checkpoints import load_checkpoint

        obs_size = int(meta["obs_size"])
        action_size = int(meta["action_size"])

        ckpt = load_checkpoint(checkpoint_dir)
        params = ckpt.get("params")
        if params is None:
            log.warning("inference: Brax checkpoint has no params, using zero policy")
            return _zero_policy(action_size)

        network = ppo_networks.make_ppo_networks(obs_size, action_size)
        make_inference_fn = ppo_networks.make_inference_fn(network)
        inference_fn = make_inference_fn(params)
        rng = jax.random.PRNGKey(0)

        log.info(
            f"inference: loaded Brax PPO from {checkpoint_dir} "
            f"(obs={obs_size}, act={action_size})"
        )

        def _fn(obs: np.ndarray) -> np.ndarray:
            jax_obs = jnp.array(obs, dtype=jnp.float32).reshape(1, -1)
            action, _ = inference_fn(jax_obs, rng)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return _fn
    except Exception as exc:
        log.warning(f"inference: Brax load failed ({exc}), using zero policy")
        action_size = meta.get("action_size", 0)
        return _zero_policy(int(action_size))
