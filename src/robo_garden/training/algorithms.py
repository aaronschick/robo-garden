"""RL algorithm wrappers: Brax PPO (JAX-native for MJX) and SB3 fallback."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def create_brax_ppo_trainer(
    env,
    learning_rate: float = 3e-4,
    num_timesteps: int = 1_000_000,
    batch_size: int = 256,
    num_envs: int = 128,
):
    """Create a Brax PPO trainer for MJX vectorized environments.

    Brax PPO operates natively on JAX arrays — zero CPU-GPU transfer overhead.
    This is the primary training algorithm for local GPU training.

    TODO (Phase 4): Implement Brax PPO integration with:
    - brax.training.agents.ppo.train
    - Custom network architecture
    - Checkpoint callbacks
    """
    log.warning("Brax PPO trainer not yet implemented")
    return None


def create_sb3_ppo_trainer(
    env,
    learning_rate: float = 3e-4,
    num_timesteps: int = 1_000_000,
    batch_size: int = 256,
):
    """Create a Stable Baselines3 PPO trainer as fallback.

    SB3 uses PyTorch and NumPy — involves CPU-GPU transfer per step.
    Use this for CPU-mode debugging or when Brax PPO isn't suitable.

    TODO (Phase 4): Implement SB3 integration.
    """
    log.warning("SB3 PPO trainer not yet implemented")
    return None
