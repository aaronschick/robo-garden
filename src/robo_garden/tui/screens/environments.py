"""Environments screen: lists saved environment XML files."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import RichLog


class EnvironmentsScreen(Widget):
    """Lists saved environment XML files from the workspace environments directory."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)

    def on_mount(self) -> None:
        self._scan_and_display()

    def _scan_and_display(self) -> None:
        from robo_garden.config import WORKSPACE_DIR

        log = self.query_one("#log", RichLog)
        log.clear()

        env_dir = WORKSPACE_DIR / "environments"
        if not env_dir.exists():
            log.write("[dim]No environments saved yet.[/dim]")
            return

        xml_files = sorted(env_dir.glob("*.xml"))
        if not xml_files:
            log.write("[dim]No environments saved yet.[/dim]")
            return

        log.write(f"[bold]Saved Environments[/bold] ({len(xml_files)} found)\n")
        for path in xml_files:
            size = path.stat().st_size
            log.write(f"  [cyan]{path.name}[/cyan]  ({size:,} bytes)")

    def action_refresh(self) -> None:
        self._scan_and_display()
