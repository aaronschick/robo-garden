"""Chat widget: Claude conversation with tool-use display.

Delegates to the canonical agentic loop in claude/client.py via Session.chat().
"""

from __future__ import annotations

import asyncio
import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Input, RichLog


def _tool_label(name: str, tool_input: dict) -> str:
    for key in ("name", "robot_name", "query", "environment_name", "task"):
        if key in tool_input and tool_input[key]:
            return str(tool_input[key])[:40]
    for v in tool_input.values():
        if isinstance(v, str) and v.strip():
            return v.strip()[:40]
    return json.dumps(tool_input)[:40]


class ChatScreen(Widget):
    """Interactive chat widget for robot design conversations with Claude."""

    BINDINGS = [
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._session = None
        self._reward_iteration = 0

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)
        yield Input(placeholder="Describe a robot...", id="input")

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[dim]Robo Garden ready. Describe a robot to get started.[/dim]")
        self.query_one("#input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        event.input.clear()
        log = self.query_one("#log", RichLog)
        log.write(f"\n[bold cyan]You:[/bold cyan] {message}")
        self.run_worker(self._send_message(message), exclusive=False)

    async def _send_message(self, message: str) -> None:
        log = self.query_one("#log", RichLog)

        try:
            session = self._get_session()
        except Exception as e:
            log.write(f"[bold red]Session error:[/bold red] {e}")
            return

        log.write("[dim italic]Claude is thinking…[/dim italic]")

        tool_lines: list[str] = []

        def on_tool_call(name: str, tool_input: dict) -> None:
            label = _tool_label(name, tool_input)
            tool_lines.append(f"[dim italic]  → tool: {name} ({label})[/dim italic]")

        def on_tool_result(name: str, result: dict) -> None:
            if name == "generate_robot" and result.get("success"):
                from robo_garden.tui.screens.building import BuildingScreen
                self.app.call_from_thread(
                    self.app.query_one(BuildingScreen).refresh_display, result
                )
            elif name == "generate_reward" and result.get("success"):
                from robo_garden.tui.screens.rewards import RewardsScreen
                self._reward_iteration += 1
                reward_id = result.get("reward_function_id", "?")
                self.app.call_from_thread(
                    self.app.query_one(RewardsScreen).add_iteration,
                    self._reward_iteration,
                    0.0,
                    0.0,
                    result.get("task_description", reward_id),
                )
            elif name == "train" and result.get("success"):
                from robo_garden.tui.screens.training import TrainingScreen
                reward_curve = result.get("reward_curve", [])
                for step, mean_r in reward_curve:
                    self.app.call_from_thread(
                        self.app.query_one(TrainingScreen).log_update,
                        step,
                        {"eval/episode_reward": mean_r},
                    )

        try:
            response = await asyncio.to_thread(
                session.chat, message, on_tool_call=on_tool_call, on_tool_result=on_tool_result
            )
        except Exception as e:
            log.write(f"[bold red]Error:[/bold red] {e}")
            return

        for line in tool_lines:
            log.write(line)

        log.write(f"[bold green]Claude:[/bold green] {response}")

    def _get_session(self):
        if self._session is None:
            from robo_garden.claude.session import Session
            self._session = Session()
        return self._session

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()
        self.query_one("#log", RichLog).write("[dim]Log cleared.[/dim]")
