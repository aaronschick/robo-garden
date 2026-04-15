"""Abstract TrainingEngine protocol: interface for local and cloud backends."""

from __future__ import annotations

from typing import Protocol

from robo_garden.training.models import TrainingConfig, TrainingResult


class TrainingEngine(Protocol):
    """Protocol for training backends (MuJoCo/MJX local, Isaac Lab cloud)."""

    def setup(self, robot_mjcf: str, env_mjcf: str, config: TrainingConfig) -> None:
        """Initialize the training environment and algorithm."""
        ...

    def train(self, reward_fn_code: str, callback=None) -> TrainingResult:
        """Run the full training loop.

        Args:
            reward_fn_code: Python source code for the reward function.
            callback: Optional callback(timestep, metrics) for progress updates.
        """
        ...

    def evaluate(self, checkpoint_path: str, num_episodes: int = 10) -> dict:
        """Evaluate a trained policy."""
        ...

    def cleanup(self) -> None:
        """Release resources."""
        ...
