"""Conversation session management with history and state."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable

from robo_garden.claude.client import create_client, run_agentic_loop
from robo_garden.claude.prompts import SYSTEM_PROMPT


def _resolve_prompt(text: str) -> str:
    """If text starts with @, load contents from the referenced file.

    Resolution order for relative paths:
      1. workspace/prompts/<path>
      2. Path as-is (absolute or relative to cwd)

    Raises FileNotFoundError if the file cannot be found.
    Returns text unchanged if it does not start with @.
    """
    if not text.startswith("@"):
        return text

    from robo_garden.config import PROMPTS_DIR

    raw = text[1:].strip()
    candidate = Path(raw)

    if not candidate.is_absolute():
        in_prompts = PROMPTS_DIR / raw
        if in_prompts.exists():
            candidate = in_prompts

    if not candidate.exists():
        raise FileNotFoundError(
            f"Prompt file not found: '{raw}'\n"
            f"  Looked in: {PROMPTS_DIR / raw}\n"
            f"  And as absolute/relative path: {Path(raw).resolve()}"
        )

    return candidate.read_text(encoding="utf-8").strip()


class Session:
    """A conversation session with Claude for robot design."""

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []
        self.client = create_client()
        self.robots: dict[str, dict] = {}  # name -> robot data

        from robo_garden.viewer.session_viewer import SessionViewer
        self.viewer = SessionViewer()

    def chat(
        self,
        user_message: str,
        on_status: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, dict], None] | None = None,
    ) -> str:
        """Send a message and get Claude's response, handling tool calls automatically."""
        self.messages.append({"role": "user", "content": user_message})
        response, self.messages = run_agentic_loop(
            client=self.client,
            system_prompt=SYSTEM_PROMPT,
            messages=self.messages,
            on_status=on_status,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
        return response

    def close(self) -> None:
        """Release resources (viewer window, threads)."""
        self.viewer.close()


def _make_tool_result_handler(session: Session) -> Callable[[str, dict], None]:
    """Return a callback that opens/refreshes the viewer on successful generate_robot."""
    def _on_tool_result(name: str, result: dict) -> None:
        if name == "generate_robot" and result.get("success"):
            robot_path = result.get("robot_path", "")
            robot_name = result.get("robot_name", "robot")
            if robot_path and Path(robot_path).exists():
                try:
                    xml = Path(robot_path).read_text(encoding="utf-8")
                    session.viewer.show(xml, title=f"Robo Garden — {robot_name}")
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(f"Viewer update failed: {exc}")
    return _on_tool_result


def _send_with_spinner(
    session: Session,
    user_input: str,
    on_tool_result: Callable[[str, dict], None] | None = None,
) -> str:
    """Send a message to Claude, displaying a live spinner with status updates."""
    from rich.console import Console

    console = Console()

    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots") as status:
        def on_status(msg: str) -> None:
            status.update(f"[bold cyan]{msg}[/bold cyan]")

        return session.chat(user_input, on_status=on_status, on_tool_result=on_tool_result)


def run_chat(initial_prompt: str | None = None) -> None:
    """Run an interactive chat session in the terminal.

    Args:
        initial_prompt: If provided, sent as the first message before the
            interactive loop begins (e.g. loaded from --prompt-file).
    """
    from rich.console import Console
    from rich.rule import Rule

    console = Console()
    console.print("[bold green]Robo Garden[/bold green] — Claude-Powered Robot Studio")
    console.print(
        "Type your message or [bold]@filename.txt[/bold] to load from workspace/prompts/. "
        "Type [bold]quit[/bold] to exit.\n"
    )

    session = Session()
    on_tool_result = _make_tool_result_handler(session)

    try:
        if initial_prompt:
            preview = initial_prompt[:120] + ("..." if len(initial_prompt) > 120 else "")
            console.print(f"[bold]You:[/bold] {preview}")
            response = _send_with_spinner(session, initial_prompt, on_tool_result)
            console.print()
            console.print(Rule(style="dim"))
            console.print(f"[bold green]Claude:[/bold green] {response}")
            console.print(Rule(style="dim"))
            console.print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye.[/dim]")
                break
            if not user_input:
                continue

            try:
                user_input = _resolve_prompt(user_input)
            except FileNotFoundError as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}\n")
                continue

            response = _send_with_spinner(session, user_input, on_tool_result)
            console.print()
            console.print(Rule(style="dim"))
            console.print(f"[bold green]Claude:[/bold green] {response}")
            console.print(Rule(style="dim"))
            console.print()

    finally:
        session.close()
