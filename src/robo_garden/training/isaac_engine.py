"""Isaac Lab cloud training engine: secondary backend for A100/H100 scaling."""

from __future__ import annotations

import logging

from robo_garden.training.models import TrainingConfig, TrainingResult

log = logging.getLogger(__name__)


class IsaacLabEngine:
    """Training engine using Isaac Lab (requires Linux + high-end GPU).

    This is the cloud-scale backend for training with 2048+ parallel environments
    on A100/H100 GPUs. Requires Isaac Sim + Isaac Lab installation.

    Used via Dockerfile.cloud when scaling beyond local hardware.

    TODO (Phase 8+): Implement Isaac Lab integration:
    - MJCF -> URDF -> Isaac Lab asset conversion
    - RSL-RL or RL-Games training integration
    - Remote job submission and monitoring
    """

    def setup(self, robot_mjcf: str, env_mjcf: str, config: TrainingConfig) -> None:
        raise NotImplementedError(
            "Isaac Lab engine requires Linux + Omniverse. "
            "Use Dockerfile.cloud for cloud deployment."
        )

    def train(self, reward_fn_code: str, callback=None) -> TrainingResult:
        raise NotImplementedError("Isaac Lab training requires cloud deployment.")

    def evaluate(self, checkpoint_path: str, num_episodes: int = 10) -> dict:
        raise NotImplementedError("Isaac Lab evaluation requires cloud deployment.")

    def cleanup(self) -> None:
        pass
