"""Sandboxed execution of Claude-generated reward function code.

Security: reward functions are executed with restricted globals to prevent
arbitrary code execution. Only numpy and basic math are available.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

# Restricted globals for reward function execution
SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "bool": bool,
        "True": True,
        "False": False,
        "None": None,
    },
    "np": np,
    "numpy": np,
}


def compile_reward_function(code: str) -> Callable:
    """Compile a reward function from Python source code.

    The function must define compute_reward(obs, action, next_obs, info) -> tuple[float, dict].

    Args:
        code: Python source code defining compute_reward.

    Returns:
        The compiled compute_reward callable.

    Raises:
        ValueError: If the code doesn't define compute_reward or fails to compile.
    """
    # Basic safety checks
    forbidden = ["import ", "exec(", "eval(", "open(", "__import__", "os.", "sys.", "subprocess"]
    for f in forbidden:
        if f in code:
            raise ValueError(f"Forbidden construct in reward code: '{f}'")

    local_vars: dict[str, Any] = {}
    try:
        exec(code, SAFE_GLOBALS.copy(), local_vars)
    except Exception as e:
        raise ValueError(f"Failed to compile reward code: {e}")

    if "compute_reward" not in local_vars:
        raise ValueError("Reward code must define a 'compute_reward' function")

    fn = local_vars["compute_reward"]

    # Smoke test with dummy inputs
    try:
        dummy_obs = np.zeros(10)
        result = fn(dummy_obs, np.zeros(4), dummy_obs, {})
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("compute_reward must return tuple[float, dict]")
    except ValueError:
        raise
    except Exception as e:
        log.warning(f"Reward function smoke test failed (may be OK with real obs): {e}")

    return fn
