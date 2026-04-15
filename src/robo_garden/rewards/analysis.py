"""Reward function visualization and analysis."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from robo_garden.rewards.models import RewardFunction

console = Console()


def print_reward_comparison(reward_functions: list[RewardFunction]):
    """Print a comparison table of reward function iterations."""
    table = Table(title="Reward Function Comparison")
    table.add_column("Iteration", style="bold")
    table.add_column("Mean Reward")
    table.add_column("Max Reward")
    table.add_column("Success Rate")
    table.add_column("Components")

    for rf in reward_functions:
        if rf.training_stats is None:
            table.add_row(str(rf.iteration), "N/A", "N/A", "N/A", "Not evaluated")
            continue

        stats = rf.training_stats
        components = ", ".join(
            f"{k}={v:.3f}" for k, v in stats.reward_components.items()
        )
        table.add_row(
            str(rf.iteration),
            f"{stats.mean_reward:.4f}",
            f"{stats.max_reward:.4f}",
            f"{stats.success_rate:.1%}" if stats.success_rate is not None else "N/A",
            components or "N/A",
        )

    console.print(table)
