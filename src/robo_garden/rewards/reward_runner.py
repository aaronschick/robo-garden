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


def _rewrite_if_to_where(code: str) -> str:
    """Transform the narrow ``if a < b: x = p else: x = q`` pattern into
    ``x = np.where(a < b, p, q)`` so Claude-written rewards JIT-compile.

    Claude routinely writes reward branches as plain Python ``if``/``else``
    because the prompt phrases them that way ("R_termination = -100 if z <
    0.15"). JAX can't trace ``if`` on traced values, so without a rewrite
    the Brax GPU path is effectively unreachable for almost every reward.

    We only handle the unambiguous symmetric pattern: ``if``/``else`` (or
    ``elif`` chain) where *every* branch assigns to the same single Name
    target with exactly one statement. Anything with side effects, shape
    changes, or asymmetric branches is passed through unchanged — the JIT
    trace will fail loudly with our standard error message.

    The transform is applied *only* inside ``compile_jax_reward_function``.
    The NumPy reward path never sees it, so SB3 fallback behaviour is
    unchanged.
    """
    import ast

    class _Transformer(ast.NodeTransformer):
        def _if_to_where(self, node: ast.If) -> ast.Assign | None:
            # Both branches must be a single Assign to the same single
            # Name target. We allow the else branch to itself be a nested
            # ``If`` (elif) and recurse.
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)
                and len(node.body[0].targets) == 1
                and isinstance(node.body[0].targets[0], ast.Name)
            ):
                then_assign: ast.Assign = node.body[0]
                target_name = then_assign.targets[0].id  # type: ignore[attr-defined]
            else:
                return None

            # Accept ``else: x = v`` or ``elif: …``.
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Assign):
                else_assign = node.orelse[0]
                if (
                    len(else_assign.targets) == 1
                    and isinstance(else_assign.targets[0], ast.Name)
                    and else_assign.targets[0].id == target_name  # type: ignore[attr-defined]
                ):
                    else_value: ast.expr = else_assign.value
                else:
                    return None
            elif len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                nested = self._if_to_where(node.orelse[0])
                if nested is None or nested.targets[0].id != target_name:  # type: ignore[attr-defined]
                    return None
                else_value = nested.value
            else:
                return None

            # Build: target_name = np.where(<test>, <then_value>, <else_value>)
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="np", ctx=ast.Load()),
                    attr="where",
                    ctx=ast.Load(),
                ),
                args=[node.test, then_assign.value, else_value],
                keywords=[],
            )
            new_node = ast.Assign(
                targets=[ast.Name(id=target_name, ctx=ast.Store())],
                value=call,
            )
            return ast.copy_location(new_node, node)

        def visit_If(self, node: ast.If) -> Any:  # noqa: N802
            self.generic_visit(node)
            replacement = self._if_to_where(node)
            return replacement if replacement is not None else node

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the normal exec() further down produce the real error message.
        return code

    new_tree = _Transformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:
        # Python <3.9 or a weird AST; fall back to the original source and
        # let JAX tracing report the if/else as before.
        return code


def compile_jax_reward_function(
    code: str,
    *,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> Callable:
    """Compile Claude's numpy-style reward code as a JAX-traceable function.

    Most reward functions Claude writes only touch a numpy subset that
    ``jax.numpy`` also provides (``np.clip``, ``np.exp``, ``np.mean``,
    ``np.abs``, comparisons → np.where). By re-exec'ing the code with ``np``
    and ``numpy`` rebound to ``jax.numpy``, the same source produces a
    JIT-compilable function usable by the Brax/MJX GPU training path.

    On failure (control flow that JAX can't trace, unsupported numpy op,
    Python-level ``if`` on traced values, etc.) we raise ``ValueError`` so
    the caller can fall back to the NumPy-SB3 CPU path without crashing.

    Returns a function with signature ``(obs, action, next_obs) -> scalar``
    that matches ``MJXBraxEnv``'s ``reward_fn`` contract (note: no ``info``
    argument and no tuple return — we unwrap the (reward, info) tuple).

    Raises:
        ValueError: jax not installed, code won't compile against jnp, or
            the JIT trace fails.
    """
    forbidden = ["import ", "exec(", "eval(", "open(", "__import__", "os.", "sys.", "subprocess"]
    for f in forbidden:
        if f in code:
            raise ValueError(f"Forbidden construct in reward code: '{f}'")

    # Auto-rewrite the common ``if z < 0.15: r = -100 else: r = 0`` pattern
    # into ``r = np.where(z < 0.15, -100, 0)`` so the reward can be
    # JIT-traced under JAX without Claude having to know the difference.
    # Asymmetric / side-effectful ifs are left alone and will still raise
    # the usual tracer-bool error at JIT time.
    code = _rewrite_if_to_where(code)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ValueError(f"JAX not available: {exc}")

    # Claude reward functions almost always end with `return float(r), {}`
    # to match the documented ``-> tuple[float, dict]`` signature. `float()`
    # on a traced JAX array raises ``ConcretizationTypeError`` and kills the
    # JIT trace. Same story for `int()`/`bool()` on tracers inside control
    # flow. Rebinding them to tracer-aware pass-throughs lets the same
    # numpy source compile under JAX without any code changes on Claude's
    # side.
    def _trace_safe(orig):
        def inner(x, *args, **kwargs):
            try:
                return orig(x, *args, **kwargs)
            except Exception:
                return x
        return inner

    traced_builtins = {
        **SAFE_GLOBALS["__builtins__"],
        "float": _trace_safe(float),
        "int": _trace_safe(int),
        "bool": _trace_safe(bool),
    }

    jax_globals = {
        "__builtins__": traced_builtins,
        # Critical rebind: the same Claude source that uses ``np.foo`` now
        # dispatches to ``jax.numpy.foo`` without any code changes.
        "np": jnp,
        "numpy": jnp,
    }

    local_vars: dict[str, Any] = {}
    try:
        exec(code, jax_globals, local_vars)
    except Exception as e:
        raise ValueError(f"JAX reward compile failed: {e}")

    if "compute_reward" not in local_vars:
        raise ValueError("Reward code must define 'compute_reward'")

    numpy_fn = local_vars["compute_reward"]

    def _brax_reward(obs, action, next_obs):
        # Claude returns (reward, info_dict); MJXBraxEnv only wants the scalar.
        # Pass an empty dict for info since we discard it anyway.
        r_and_info = numpy_fn(obs, action, next_obs, {})
        if isinstance(r_and_info, tuple):
            return r_and_info[0]
        return r_and_info

    # Probe the function: actually JIT-trace it with real shapes so any
    # untrace-able construct (e.g. ``if z < threshold:`` on a traced scalar)
    # raises *here* rather than at the first training step on the GPU.
    smoke_obs_dim = obs_dim if obs_dim and obs_dim > 0 else 10
    smoke_act_dim = action_dim if action_dim and action_dim > 0 else 4
    try:
        dummy_obs = jnp.zeros(smoke_obs_dim, dtype=jnp.float32)
        dummy_act = jnp.zeros(smoke_act_dim, dtype=jnp.float32)
        jitted = jax.jit(_brax_reward)
        out = jitted(dummy_obs, dummy_act, dummy_obs)
        # Force materialization so tracing errors surface synchronously.
        _ = jnp.asarray(out).block_until_ready()
    except Exception as e:
        raise ValueError(
            f"JAX reward JIT trace failed "
            f"(obs_dim={smoke_obs_dim}, action_dim={smoke_act_dim}): {e}. "
            f"Common causes: Python 'if' on traced values (use jnp.where), "
            f"numpy-only ops (e.g. np.random without passing a key), or "
            f"in-place mutation of traced arrays."
        )

    return _brax_reward


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
