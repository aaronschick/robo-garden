"""Stub screen used by not-yet-implemented mode tabs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class StubScreen(Widget):
    """Placeholder for a mode tab that will be implemented in a future phase."""

    def __init__(self, title: str, description: str, phase: str) -> None:
        super().__init__()
        self._title = title
        self._description = description
        self._phase = phase

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold cyan]{self._title}[/bold cyan]\n\n"
            f"{self._description}\n\n"
            f"[dim]Coming in {self._phase}.[/dim]",
            markup=True,
        )
