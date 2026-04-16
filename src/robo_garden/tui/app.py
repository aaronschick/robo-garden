"""Main Textual application with tab-based navigation across all 5 spaces."""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane

from robo_garden.tui.screens.chat import ChatScreen
from robo_garden.tui.screens.building import BuildingScreen
from robo_garden.tui.screens.environments import EnvironmentsScreen
from robo_garden.tui.screens.training import TrainingScreen
from robo_garden.tui.screens.rewards import RewardsScreen


class RoboGardenApp(App):
    """Robo Garden: Claude-Powered Robot Studio."""

    TITLE = "Robo Garden"
    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Chat", id="tab-chat"):
                yield ChatScreen()
            with TabPane("Building", id="tab-building"):
                yield BuildingScreen()
            with TabPane("Environments", id="tab-environments"):
                yield EnvironmentsScreen()
            with TabPane("Training", id="tab-training"):
                yield TrainingScreen()
            with TabPane("Rewards", id="tab-rewards"):
                yield RewardsScreen()
        yield Footer()
