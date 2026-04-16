"""TUI screens for each space."""

from robo_garden.tui.screens.chat import ChatScreen
from robo_garden.tui.screens.building import BuildingScreen
from robo_garden.tui.screens.environments import EnvironmentsScreen
from robo_garden.tui.screens.training import TrainingScreen
from robo_garden.tui.screens.rewards import RewardsScreen

__all__ = ["ChatScreen", "BuildingScreen", "EnvironmentsScreen", "TrainingScreen", "RewardsScreen"]
