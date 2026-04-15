"""Data models for reward design."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardStats:
    """Statistics from evaluating a reward function during training."""
    mean_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    success_rate: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    correlation_with_task: float | None = None


@dataclass
class RewardFunction:
    """A reward function with its source code and evaluation results."""
    code: str
    iteration: int = 0
    training_stats: RewardStats | None = None
    task_description: str = ""
