"""Conversation session management with history and state."""

from __future__ import annotations

import uuid

from robo_garden.claude.client import create_client, run_agentic_loop
from robo_garden.claude.prompts import SYSTEM_PROMPT


class Session:
    """A conversation session with Claude for robot design."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []
        self.client = create_client()
        self.robots: dict[str, dict] = {}  # name -> robot data

    def chat(self, user_message: str) -> str:
        """Send a message and get Claude's response, handling tool calls automatically."""
        self.messages.append({"role": "user", "content": user_message})
        response, self.messages = run_agentic_loop(
            client=self.client,
            system_prompt=SYSTEM_PROMPT,
            messages=self.messages,
        )
        return response


def run_chat():
    """Run an interactive chat session in the terminal."""
    print("Robo Garden - Claude-Powered Robot Studio")
    print("Type your robot design ideas. Type 'quit' to exit.\n")

    session = Session()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = session.chat(user_input)
        print(f"\nClaude: {response}\n")
