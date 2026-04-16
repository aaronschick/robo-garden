"""Sandboxed execution of Claude-generated reward function code.

Security: reward functions are executed with restricted globals to prevent
arbitrary code execution. Only numpy and basic math are available.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

# Warn at most once per reward-fn id about an IndexError in compute_reward.
# Without this the training log fills with thousands of identical tracebacks.
_reported_shape_warn: set[int] = set()

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
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "True": True,
        "False": False,
        "None": None,
        # Exception types Claude-written rewards routinely use for defensive
        # shape / range assertions.  Omitting these turns a clean
        # `raise ValueError(...)` into an opaque `NameError` that the smoke
        # test misclassifies as a real bug.
        "AssertionError": AssertionError,
        "ValueError": ValueError,
        "IndexError": IndexError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "Exception": Exception,
    },
    "np": np,
    "numpy": np,
}


def compile_reward_function(
    code: str,
    *,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> Callable:
    """Compile a reward function from Python source code.

    The function must define compute_reward(obs, action, next_obs, info) -> tuple[float, dict].

    Args:
        code: Python source code defining compute_reward.
        obs_dim: If provided, the smoke test uses this exact observation
            length so the reported error matches the real training shape.
        action_dim: Matching action length for the smoke test.

    Returns:
        The compiled compute_reward callable.

    Raises:
        ValueError: If the code doesn't define compute_reward or fails to compile.
    """
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

    # Smoke test with realistic shapes when available, otherwise a 10-wide
    # stand-in. Shape mismatches are the #1 cause of runtime failures, so we
    # surface them as errors (not warnings) whenever we know the real dims.
    smoke_obs_dim = obs_dim if obs_dim and obs_dim > 0 else 10
    smoke_act_dim = action_dim if action_dim and action_dim > 0 else 4
    try:
        dummy_obs = np.zeros(smoke_obs_dim, dtype=np.float32)
        dummy_act = np.zeros(smoke_act_dim, dtype=np.float32)
        result = fn(dummy_obs, dummy_act, dummy_obs, {})
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("compute_reward must return tuple[float, dict]")
    except ValueError:
        raise
    except Exception as e:
        if obs_dim is not None:
            raise ValueError(
                f"Reward function smoke test failed with real shapes "
                f"(obs_dim={smoke_obs_dim}, action_dim={smoke_act_dim}): {e}. "
                f"Remember: obs layout is concat(qpos, qvel)."
            )
        # The stand-in shape (obs_dim=10, act_dim=4) is rarely right for real
        # robots; if Claude's reward did a legitimate sanity check on shape
        # it can trip here even though training would work fine.  Log at
        # debug level so casual users aren't alarmed.
        log.debug(
            f"Reward smoke test tripped on stand-in shape "
            f"(obs_dim={smoke_obs_dim}, action_dim={smoke_act_dim}): {e}. "
            f"Pass robot_name to generate_reward (or approve a design first) "
            f"so the smoke test uses real dimensions."
        )

    return fn


def safe_reward(fn: Callable, *, fallback: float = 0.0) -> Callable:
    """Wrap *fn* so IndexError/ValueError in the user code returns ``fallback``.

    Without this, a single bad index crashes the entire training run. We log
    the first failure in detail (so the problem is still visible) but swallow
    subsequent ones silently.
    """

    fn_id = id(fn)

    def _wrapped(obs, action, next_obs, info):
        try:
            r = fn(obs, action, next_obs, info)
        except (IndexError, ValueError) as exc:
            if fn_id not in _reported_shape_warn:
                _reported_shape_warn.add(fn_id)
                log.warning(
                    f"compute_reward raised {type(exc).__name__}: {exc}. "
                    f"obs.shape={np.asarray(obs).shape}, "
                    f"action.shape={np.asarray(action).shape}. "
                    f"Returning {fallback} for this and all subsequent failures "
                    f"(warning suppressed hereafter)."
                )
            return (float(fallback), {"_error": str(exc)})
        return r

    return _wrapped
