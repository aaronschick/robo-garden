"""Anthropic SDK wrapper with agentic tool-use conversation loop."""

from __future__ import annotations

import json
import logging

import anthropic

from robo_garden.claude.tools import TOOLS
from robo_garden.claude.tool_handlers import dispatch_tool
from robo_garden.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

log = logging.getLogger(__name__)


def create_client() -> anthropic.Anthropic:
    """Create an Anthropic client."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def run_agentic_loop(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    max_iterations: int = 20,
) -> tuple[str, list[dict]]:
    """Run the Claude tool-use agentic loop until Claude returns a final text response.

    Args:
        client: Anthropic client instance.
        system_prompt: System prompt defining Claude's role and context.
        messages: Conversation history in Anthropic messages format.
        max_iterations: Safety limit on tool-use iterations.

    Returns:
        Tuple of (final_text_response, updated_messages).
    """
    for _ in range(max_iterations):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            system=system_prompt,
            messages=messages,
            tools=TOOLS,
            max_tokens=4096,
        )

        # Collect content blocks
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done
        if response.stop_reason == "end_turn" or not tool_calls:
            return "\n".join(text_parts), messages

        # Process tool calls and send results back
        tool_results = []
        for tool_call in tool_calls:
            log.info(f"Tool call: {tool_call.name}({json.dumps(tool_call.input)[:200]}...)")
            result = dispatch_tool(tool_call.name, tool_call.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached in agentic loop.", messages
