"""MuJoCo + MJX local training engine: primary backend for RTX 3070."""

from __future__ import annotations

import logging
import time

from robo_garden.training.models import TrainingConfig, TrainingResult

log = logging.getLogger(__name__)


class MuJoCoMJXEngine:
    """Training engine using MuJoCo MJX (JAX GPU acceleration).

    This is the primary local training backend, targeting 64-256 parallel
    environments on an RTX 3070 (8GB VRAM).

    Implementation phases:
    - Phase 4: vectorized_env.py (jax.vmap batched stepping)
    - Phase 4: Brax PPO integration
    - Phase 7: Curriculum learning support
    """

    def __init__(self):
        self.model = None
        self.config = None

    def setup(self, robot_mjcf: str, env_mjcf: str, config: TrainingConfig) -> None:
        """Initialize MJX model and vectorized environment."""
        import mujoco
        from mujoco import mjx

        self.config = config
        self.model = mujoco.MjModel.from_xml_string(robot_mjcf)
        self.mjx_model = mjx.put_model(self.model)
        log.info(
            f"MJX engine initialized: {config.num_envs} envs, "
            f"{config.total_timesteps} total steps"
        )

    def train(self, reward_fn_code: str, callback=None) -> TrainingResult:
        """Run PPO training with MJX vectorized environments.

        TODO (Phase 4): Implement full training loop with:
        - vectorized_env.py for jax.vmap batched stepping
        - Brax PPO for JAX-native policy optimization
        - Checkpoint saving at intervals
        """
        log.warning("MJX training not yet implemented - returning placeholder result")
        return TrainingResult(
            config=self.config,
            training_time_seconds=0.0,
        )

    def evaluate(self, checkpoint_path: str, num_episodes: int = 10) -> dict:
        """Evaluate a trained policy checkpoint."""
        return {"status": "not_implemented"}

    def cleanup(self) -> None:
        """Release GPU resources."""
        self.model = None
        self.mjx_model = None
