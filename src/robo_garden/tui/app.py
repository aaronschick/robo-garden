"""Main Textual application with tab-based navigation across all 5 spaces.

TODO (Phase 6): Implement full TUI with:
- Chat screen (Claude conversation with streaming)
- Building screen (robot design + validation results)
- Environments screen (terrain/object configuration)
- Training screen (live reward curves, FPS)
- Rewards screen (Eureka iteration comparison)
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, TabbedContent, TabPane


class RoboGardenApp(App):
    """Robo Garden: Claude-Powered Robot Studio."""

    TITLE = "Robo Garden"
    CSS = """
    Screen {
        background: $surface;
    }
    #placeholder {
        margin: 2;
        padding: 1 2;
        border: solid $primary;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "focus_chat", "Chat"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Chat", id="tab-chat"):
                yield Static(
                    "Claude Chat - Type your robot design ideas here.\n"
                    "(Full implementation in Phase 6)",
                    id="placeholder",
                )
            with TabPane("Building", id="tab-building"):
                yield Static(
                    "Robot Building Space\n"
                    "Design robots with real actuators and materials.\n"
                    "(Full implementation in Phase 6)",
                    id="placeholder",
                )
            with TabPane("Environments", id="tab-environments"):
                yield Static(
                    "Environment Building Space\n"
                    "Create terrain and training environments.\n"
                    "(Full implementation in Phase 6)",
                    id="placeholder",
                )
            with TabPane("Training", id="tab-training"):
                yield Static(
                    "Training Gym\n"
                    "Run RL training and monitor progress.\n"
                    "(Full implementation in Phase 6)",
                    id="placeholder",
                )
            with TabPane("Rewards", id="tab-rewards"):
                yield Static(
                    "Incentive Design\n"
                    "Design and iterate reward functions.\n"
                    "(Full implementation in Phase 6)",
                    id="placeholder",
                )
        yield Footer()
