"""Dispatch Claude tool_use blocks to local execution."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Registry of tool name -> handler function
_HANDLERS: dict[str, Any] = {}


def register_handler(tool_name: str):
    """Decorator to register a tool handler function."""
    def decorator(fn):
        _HANDLERS[tool_name] = fn
        return fn
    return decorator


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call to its registered handler.

    Returns a dict that will be JSON-serialized as the tool_result content.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        log.error(f"Unknown tool: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return handler(tool_input)
    except Exception as e:
        log.exception(f"Tool {tool_name} failed")
        return {"error": str(e)}


# --- Handler implementations ---
# These will be filled in during Phase 1-5 implementation.
# Each handler bridges a Claude tool call to the corresponding module.


@register_handler("generate_robot")
def handle_generate_robot(input: dict) -> dict:
    """Validate Claude-generated MJCF and create a Robot object."""
    from robo_garden.core.formats import validate_mjcf, model_info

    mjcf_xml = input.get("mjcf_xml", "")
    result = validate_mjcf(mjcf_xml)

    if not result.valid:
        return {"success": False, "errors": result.errors}

    info = model_info(result.model)
    return {
        "success": True,
        "robot_name": input.get("name", "unnamed"),
        "model_info": info,
        "warnings": result.warnings,
    }


@register_handler("simulate")
def handle_simulate(input: dict) -> dict:
    """Run a physics simulation and return results."""
    # Stub: will be implemented in Phase 1
    return {"status": "not_implemented", "message": "Simulation handler coming in Phase 1"}


@register_handler("evaluate")
def handle_evaluate(input: dict) -> dict:
    """Evaluate simulation results against criteria."""
    return {"status": "not_implemented", "message": "Evaluation handler coming in Phase 2"}


@register_handler("generate_environment")
def handle_generate_environment(input: dict) -> dict:
    """Create an environment from Claude's specification."""
    return {"status": "not_implemented", "message": "Environment handler coming in Phase 3"}


@register_handler("generate_reward")
def handle_generate_reward(input: dict) -> dict:
    """Validate and register a reward function."""
    return {"status": "not_implemented", "message": "Reward handler coming in Phase 5"}


@register_handler("train")
def handle_train(input: dict) -> dict:
    """Launch an RL training run."""
    return {"status": "not_implemented", "message": "Training handler coming in Phase 4"}


@register_handler("query_catalog")
def handle_query_catalog(input: dict) -> dict:
    """Search actuator/material/robot catalogs."""
    return {"status": "not_implemented", "message": "Catalog handler coming in Phase 2"}
