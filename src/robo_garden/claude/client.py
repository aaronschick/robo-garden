"""Anthropic SDK wrapper with agentic tool-use conversation loop."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

import anthropic

from robo_garden.claude.tools import TOOLS
from robo_garden.claude.tool_handlers import dispatch_tool
from robo_garden.claude.gating import tools_for_phase
from robo_garden.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

log = logging.getLogger(__name__)

# Human-readable label for each tool
_TOOL_LABELS: dict[str, str] = {
    "generate_robot": "Generating robot",
    "simulate": "Simulating",
    "evaluate": "Evaluating",
    "generate_environment": "Building environment",
    "generate_reward": "Writing reward function",
    "train": "Training",
    "query_catalog": "Searching catalog",
    "approve_for_training": "Approving design",
}


def _tool_status(name: str, input: dict) -> str:
    """One-line status string for a tool call."""
    label = _TOOL_LABELS.get(name, name)
    detail = (
        input.get("name")
        or input.get("robot_name")
        or input.get("simulation_id")
        or input.get("query")
        or ""
    )
    extra = ""
    if name == "simulate" and "duration_seconds" in input:
        extra = f", {input['duration_seconds']}s"
    if name == "train" and "total_timesteps" in input:
        extra = f", {input['total_timesteps']:,} steps"
    return f"{label}: {detail}{extra}" if detail else label


def _tool_result_summary(name: str, result: dict) -> str:
    """One-line summary of a tool result."""
    if not result.get("success", True):
        errors = result.get("errors", result.get("error", "unknown error"))
        return f"failed — {errors}"
    if name == "generate_robot":
        path = result.get("robot_path", "")
        warnings = len(result.get("warnings", []))
        suffix = f", {warnings} warning(s)" if warnings else ""
        return f"saved {path.split('/')[-1].split(chr(92))[-1]}{suffix}"
    if name == "simulate":
        stable = result.get("stable")
        dur = result.get("duration", "")
        return f"{dur}s — {'stable' if stable else 'unstable'}"
    if name == "train":
        best = result.get("best_reward")
        t = result.get("training_time_seconds")
        return f"best_reward={best:.3f}, {t:.1f}s" if best is not None and t is not None else "done"
    if name == "evaluate":
        metrics = result.get("metrics", {})
        return ", ".join(f"{k}={v:.3f}" for k, v in list(metrics.items())[:3]) or "done"
    if name == "approve_for_training":
        if result.get("approved"):
            return (
                f"{result.get('robot_name', '?')} + {result.get('environment_name', '?')} "
                "→ training unlocked"
            )
        unmet = result.get("unmet_preconditions", [])
        return f"not approved — {len(unmet)} precondition(s) unmet"
    return "done"


def _create_with_retry(
    client: anthropic.Anthropic,
    on_status: Callable[[str], None] | None = None,
    **kwargs,
) -> anthropic.types.Message:
    """Call client.messages.create with exponential backoff on rate-limit errors."""
    delays = [15, 30, 60]
    for attempt, delay in enumerate(delays, start=1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt > len(delays):
                raise
            msg = f"Rate limit — waiting {delay}s (retry {attempt}/{len(delays)})"
            if on_status:
                on_status(msg)
            else:
                print(f"\n[{msg}]")
            time.sleep(delay)
            if on_status:
                on_status("Thinking...")
    return client.messages.create(**kwargs)


def create_client() -> anthropic.Anthropic:
    """Create an Anthropic client."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def run_agentic_loop(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    max_iterations: int = 20,
    on_status: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, dict], None] | None = None,
    phase_getter: Callable[[], str] | None = None,
) -> tuple[str, list[dict]]:
    """Run the Claude tool-use agentic loop until Claude returns a final text response.

    Args:
        client: Anthropic client instance.
        system_prompt: System prompt defining Claude's role and context.
        messages: Conversation history in Anthropic messages format.
        max_iterations: Safety limit on tool-use iterations.
        on_status: Optional callback called with a one-line status string whenever
            the loop changes state (thinking, calling a tool, waiting on rate limit).
        on_tool_call: Optional callback fired before each tool dispatch with
            (tool_name, tool_input).
        on_tool_result: Optional callback fired after each tool dispatch with
            (tool_name, result_dict).
        phase_getter: Optional callable returning the current session phase
            ("design" or "training").  Re-evaluated on each iteration so that a
            successful ``approve_for_training`` call mid-turn unlocks training
            tools on the very next iteration.  Defaults to "design".

    Returns:
        Tuple of (final_text_response, updated_messages).
    """
    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    for iteration in range(max_iterations):
        _status("Thinking...")
        current_phase = phase_getter() if phase_getter else "design"
        allowed_tools = tools_for_phase(current_phase, TOOLS)
        response = _create_with_retry(
            client,
            on_status=on_status,
            model=CLAUDE_MODEL,
            system=system_prompt,
            messages=messages,
            tools=allowed_tools,
            max_tokens=16384,
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
            _status(_tool_status(tool_call.name, tool_call.input))
            log.info(f"Tool call: {tool_call.name}({json.dumps(tool_call.input)[:200]}...)")
            if on_tool_call:
                on_tool_call(tool_call.name, tool_call.input)
            result = dispatch_tool(tool_call.name, tool_call.input)
            summary = _tool_result_summary(tool_call.name, result)
            _status(f"{_TOOL_LABELS.get(tool_call.name, tool_call.name)}: {summary}")
            if on_tool_result:
                on_tool_result(tool_call.name, result)
            log.info(f"Tool result: {tool_call.name} → {summary}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached in agentic loop.", messages
