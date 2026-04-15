"""Data models for the Training Gym."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CurriculumStage:
    """A single stage in a training curriculum."""
    name: str
    environment_params: dict = field(default_factory=dict)
    max_timesteps: int = 500_000


@dataclass
class CurriculumConfig:
    """Progressive difficulty configuration."""
    stages: list[CurriculumStage] = field(default_factory=list)
    advance_threshold: float = 0.8  # metric threshold to advance


@dataclass
class TrainingConfig:
    """Configuration for an RL training run."""
    algorithm: str = "ppo"               # "ppo" | "sac"
    engine: str = "mujoco_mjx"           # "mujoco_mjx" | "isaac_lab"
    num_envs: int = 128                  # parallel environments
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    max_episode_steps: int = 1000
    curriculum: CurriculumConfig | None = None
    checkpoint_interval: int = 50_000
    device: str = "auto"                 # "cpu" | "cuda" | "auto"


@dataclass
class TrainingResult:
    """Results from a completed training run."""
    config: TrainingConfig
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[float] = field(default_factory=list)
    success_rate: float | None = None
    best_reward: float = float("-inf")
    training_time_seconds: float = 0.0
    checkpoint_path: Path | None = None
    reward_curve: list[tuple[int, float]] = field(default_factory=list)
