"""Eureka-style iterative reward generation loop.

Loop: Claude generates reward code -> short training run -> collect stats ->
      Claude refines reward -> repeat 3-5 iterations -> keep the best.

Reference: https://eureka-research.github.io/
"""

from __future__ import annotations

import logging
import re

from robo_garden.rewards.models import RewardFunction, RewardStats
from robo_garden.rewards.reward_runner import compile_reward_function


def _extract_code_block(text: str) -> str:
    """Extract Python code from a markdown code block, or return text as-is."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

log = logging.getLogger(__name__)


class EurekaLoop:
    """Manages the iterative reward generation and evaluation cycle."""

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
        robot_mjcf: str = "",
        env_mjcf: str = "",
        previous_stats: RewardStats | None = None,
    ) -> RewardFunction:
        """Run a single Eureka iteration: compile, short-train, record stats.

        Args:
            task_description: What the robot should learn.
            observation_space_desc: Description of the observation space.
            reward_code: Claude-generated reward function code.
            robot_mjcf: MJCF XML for the robot (required to run training).
            env_mjcf: MJCF XML for the environment (optional).
            previous_stats: Stats from previous iteration for context.

        Returns:
            RewardFunction with training_stats populated if robot_mjcf provided.
        """
        compile_reward_function(reward_code)  # raises ValueError if invalid

        reward_fn = RewardFunction(
            code=reward_code,
            iteration=len(self.history),
            task_description=task_description,
        )
        self.history.append(reward_fn)
        log.info(f"Eureka iteration {reward_fn.iteration}: reward function compiled successfully")

        if robot_mjcf:
            try:
                from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
                from robo_garden.training.models import TrainingConfig

                config = TrainingConfig(
                    num_envs=64,
                    total_timesteps=self.eval_timesteps,
                )
                engine = MuJoCoMJXEngine()
                engine.setup(robot_mjcf, env_mjcf, config)
                result = engine.train(reward_fn_code=reward_code)
                engine.cleanup()

                rewards = [r for _, r in result.reward_curve]
                reward_fn.training_stats = RewardStats(
                    mean_reward=float(sum(rewards) / len(rewards)) if rewards else 0.0,
                    max_reward=float(max(rewards)) if rewards else 0.0,
                    min_reward=float(min(rewards)) if rewards else 0.0,
                )

                if self.best is None or (
                    reward_fn.training_stats.mean_reward
                    > (self.best.training_stats.mean_reward if self.best.training_stats else float("-inf"))
                ):
                    self.best = reward_fn

                log.info(
                    f"Eureka iteration {reward_fn.iteration}: "
                    f"mean_reward={reward_fn.training_stats.mean_reward:.3f}"
                )
            except Exception as e:
                log.warning(f"Eureka training run failed: {e}")

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

    def run_with_claude(
        self,
        task_description: str,
        observation_space_desc: str,
        robot_mjcf: str = "",
        env_mjcf: str = "",
    ) -> "RewardFunction | None":
        """Run iterative reward refinement, asking Claude to improve the reward each iteration.

        Returns the best RewardFunction found, or None if all iterations failed.
        """
        import anthropic as _anthropic
        from robo_garden.claude.prompts import REWARD_GENERATION_PROMPT
        from robo_garden.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Write a reward function for the following task:\n"
                    f"Task: {task_description}\n"
                    f"Observation space: {observation_space_desc}\n\n"
                    "Return ONLY a Python code block with the compute_reward function."
                ),
            }
        ]

        no_improvement_streak = 0

        for iteration in range(self.max_iterations):
            log.info(f"Eureka iteration {iteration + 1}/{self.max_iterations}: calling Claude")

            response = client.messages.create(
                model=CLAUDE_MODEL,
                system=REWARD_GENERATION_PROMPT,
                messages=messages,
                max_tokens=2048,
            )

            assistant_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            code = _extract_code_block(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})

            try:
                reward_fn = self.run_iteration(
                    task_description,
                    observation_space_desc,
                    code,
                    robot_mjcf,
                    env_mjcf,
                )
            except Exception as exc:
                log.error(f"Eureka iteration {iteration + 1}: reward code invalid — {exc}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"The reward function raised an error: {exc}\n"
                        "Please fix it and return ONLY a Python code block."
                    ),
                })
                no_improvement_streak += 1
                if no_improvement_streak >= 2:
                    log.info("Eureka: stopping early — 2 consecutive failures")
                    break
                continue

            if reward_fn.training_stats is not None:
                if self.best is None or (
                    reward_fn.training_stats.mean_reward
                    > (self.best.training_stats.mean_reward if self.best.training_stats else float("-inf"))
                ):
                    self.best = reward_fn
                    no_improvement_streak = 0
                    log.info(f"Eureka iteration {iteration + 1}: new best mean_reward={reward_fn.training_stats.mean_reward:.4f}")
                else:
                    no_improvement_streak += 1
                    log.info(f"Eureka iteration {iteration + 1}: no improvement (streak={no_improvement_streak})")
            else:
                log.info(f"Eureka iteration {iteration + 1}: compiled OK, no training stats")
                no_improvement_streak = 0

            if no_improvement_streak >= 2:
                log.info("Eureka: stopping early — 2 consecutive iterations without improvement")
                break

            feedback = self.format_feedback_for_claude(reward_fn)
            messages.append({
                "role": "user",
                "content": (
                    f"{feedback}\n\n"
                    "Based on these results, improve the reward function. "
                    "Return ONLY a Python code block with the updated compute_reward function."
                ),
            })

        return self.get_best()
