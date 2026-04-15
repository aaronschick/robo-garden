"""Eureka-style iterative reward generation loop.

Loop: Claude generates reward code -> short training run -> collect stats ->
      Claude refines reward -> repeat 3-5 iterations -> keep the best.

Reference: https://eureka-research.github.io/
"""

from __future__ import annotations

import logging

from robo_garden.rewards.models import RewardFunction, RewardStats
from robo_garden.rewards.reward_runner import compile_reward_function

log = logging.getLogger(__name__)


class EurekaLoop:
    """Manages the iterative reward generation and evaluation cycle.

    TODO (Phase 5): Full implementation with:
    - Claude API calls for reward generation/refinement
    - Short training runs for evaluation
    - Statistics collection and comparison
    - Best-of-N selection
    """

    def __init__(self, max_iterations: int = 5, eval_timesteps: int = 200_000):
        self.max_iterations = max_iterations
        self.eval_timesteps = eval_timesteps
        self.history: list[RewardFunction] = []
        self.best: RewardFunction | None = None

    def run_iteration(
        self,
        task_description: str,
        observation_space_desc: str,
        reward_code: str,
        previous_stats: RewardStats | None = None,
    ) -> RewardFunction:
        """Run a single Eureka iteration: compile, evaluate, record.

        Args:
            task_description: What the robot should learn.
            observation_space_desc: Description of the observation space.
            reward_code: Claude-generated reward function code.
            previous_stats: Stats from previous iteration for refinement.

        Returns:
            RewardFunction with compiled code (stats populated after training).
        """
        # Compile the reward function
        fn = compile_reward_function(reward_code)

        reward_fn = RewardFunction(
            code=reward_code,
            iteration=len(self.history),
            task_description=task_description,
        )
        self.history.append(reward_fn)

        log.info(f"Eureka iteration {reward_fn.iteration}: reward function compiled successfully")
        return reward_fn

    def get_best(self) -> RewardFunction | None:
        """Return the best reward function based on training stats."""
        evaluated = [rf for rf in self.history if rf.training_stats is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda rf: rf.training_stats.mean_reward)

    def format_feedback_for_claude(self, reward_fn: RewardFunction) -> str:
        """Format evaluation results as feedback for Claude's next iteration."""
        if reward_fn.training_stats is None:
            return "No training stats available yet."

        stats = reward_fn.training_stats
        lines = [
            f"Iteration {reward_fn.iteration} results:",
            f"  Mean reward: {stats.mean_reward:.4f}",
            f"  Max reward: {stats.max_reward:.4f}",
            f"  Min reward: {stats.min_reward:.4f}",
        ]
        if stats.success_rate is not None:
            lines.append(f"  Success rate: {stats.success_rate:.1%}")
        if stats.reward_components:
            lines.append("  Reward components:")
            for name, value in stats.reward_components.items():
                lines.append(f"    {name}: {value:.4f}")

        return "\n".join(lines)
