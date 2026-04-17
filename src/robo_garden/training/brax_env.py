"""JAX-native Brax environment backed by MuJoCo MJX.

Used by MuJoCoMJXEngine._train_brax when JAX + MJX + Brax are available
(Linux / WSL2 with CUDA).  This module is intentionally imported lazily so
that it is safe to import robo_garden on Windows where mujoco-mjx/brax are
not installed.
"""

from __future__ import annotations

import re
from typing import Callable


def _mjx_compat_xml(mjcf_xml: str) -> str:
    """Patch MJCF for MJX compatibility.

    MJX does not implement all collision geometry pairs — notably
    (CYLINDER, BOX) and (CYLINDER, PLANE) are missing.  The standard
    workaround is to replace collision-active cylinder geoms with capsules,
    which MJX fully supports and which approximate cylinders well for RL.

    Visual-only geoms (``contype="0" conaffinity="0"``) are left unchanged
    so rendered meshes stay correct.
    """
    def _replace_cylinder(m: re.Match) -> str:
        tag = m.group(0)
        # Leave purely visual geoms (no collision contribution) unchanged.
        has_contype0 = bool(re.search(r'contype\s*=\s*"0"', tag))
        has_conaff0 = bool(re.search(r'conaffinity\s*=\s*"0"', tag))
        if has_contype0 and has_conaff0:
            return tag
        return re.sub(r'type\s*=\s*"cylinder"', 'type="capsule"', tag)

    # Match each <geom ...> element (single-line; MJCF geoms are always one line)
    return re.sub(
        r'<geom\b[^>]*type\s*=\s*"cylinder"[^>]*/?>',
        _replace_cylinder,
        mjcf_xml,
    )


def _make_state(pipeline_state, obs, reward, done, metrics=None):
    """Construct a Brax State, tolerating API differences across versions.

    Brax >= 0.9 added an ``info`` field to ``State``; older builds do not have
    it.  We try both constructors and fall back gracefully.
    """
    from brax.envs.base import State  # type: ignore

    kwargs = dict(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics or {},
    )
    try:
        return State(**kwargs, info={})
    except TypeError:
        return State(**kwargs)


class MJXBraxEnv:
    """Brax-compatible environment that steps MuJoCo physics via JAX/MJX.

    Implements the ``brax.envs.base.Env`` protocol (``observation_size``,
    ``action_size``, ``reset``, ``step``) so it can be passed directly to
    ``brax.training.agents.ppo.train``.

    All methods are pure-JAX and JIT-compilable.  Reward and done functions
    MUST use ``jax.numpy`` (not ``numpy``) to remain traceable.

    Args:
        mjcf_xml: MJCF model string.
        reward_fn: ``(obs, action, next_obs) -> jax.Array`` — must be
            JIT-compilable.  Defaults to a small control-effort penalty.
        done_fn: ``(next_obs) -> bool jax.Array`` for early termination.
            Defaults to NaN-divergence check only.
        init_noise_scale: Std-dev of Gaussian noise added to qpos/qvel at reset.
    """

    def __init__(
        self,
        mjcf_xml: str,
        reward_fn: Callable | None = None,
        done_fn: Callable | None = None,
        init_noise_scale: float = 0.01,
    ):
        import mujoco
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        self._jax = jax
        self._jnp = jnp
        self._mjx = mjx

        self._mj_model = mujoco.MjModel.from_xml_string(_mjx_compat_xml(mjcf_xml))
        self._mx = mjx.put_model(self._mj_model)

        _base_mj_data = mujoco.MjData(self._mj_model)
        self._base_data = mjx.put_data(self._mj_model, _base_mj_data)

        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._init_noise_scale = init_noise_scale

        self._nq = self._mj_model.nq
        self._nv = self._mj_model.nv
        self._nu = self._mj_model.nu

        # Pre-compute actuator-control scaling.  PPO outputs actions in [-1, 1]
        # (tanh-squashed Gaussian by default), but MuJoCo expects torques in
        # each actuator's ctrlrange — e.g. [-23.7, 23.7] Nm for Go2.  Without
        # this rescale we'd feed the physics engine torques ~24× smaller than
        # the robot needs to move its own weight, and PPO would converge to a
        # "do nothing" policy even with a good reward.
        import numpy as _np
        ctrlrange = _np.asarray(self._mj_model.actuator_ctrlrange, dtype=_np.float32)
        if ctrlrange.ndim == 2 and ctrlrange.shape[0] == self._nu and self._nu > 0:
            lo = ctrlrange[:, 0]
            hi = ctrlrange[:, 1]
            # Actuators with ctrllimited="false" report (0, 0); treat those as
            # identity so we don't collapse their range to zero.
            unlimited = hi <= lo
            lo = _np.where(unlimited, -1.0, lo).astype(_np.float32)
            hi = _np.where(unlimited, 1.0, hi).astype(_np.float32)
            self._ctrl_lo = jnp.asarray(lo)
            self._ctrl_hi = jnp.asarray(hi)
            self._ctrl_center = (self._ctrl_lo + self._ctrl_hi) * jnp.float32(0.5)
            self._ctrl_halfspan = (self._ctrl_hi - self._ctrl_lo) * jnp.float32(0.5)
        else:
            # No actuators — stub out scaling so .step() still runs (useful
            # for debugging purely-passive models).
            self._ctrl_lo = jnp.zeros((self._nu,), dtype=jnp.float32)
            self._ctrl_hi = jnp.zeros((self._nu,), dtype=jnp.float32)
            self._ctrl_center = jnp.zeros((self._nu,), dtype=jnp.float32)
            self._ctrl_halfspan = jnp.ones((self._nu,), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Brax Env protocol
    # ------------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        return self._nq + self._nv

    @property
    def action_size(self) -> int:
        return self._nu

    def reset(self, rng):
        """Reset to a randomly perturbed initial state."""
        jax = self._jax
        jnp = self._jnp

        rng, key_q, key_v = jax.random.split(rng, 3)
        scale = self._init_noise_scale
        qpos = self._base_data.qpos + jax.random.normal(key_q, (self._nq,)) * scale
        qvel = self._base_data.qvel + jax.random.normal(key_v, (self._nv,)) * scale

        data = self._base_data.replace(qpos=qpos, qvel=qvel)
        data = self._mjx.forward(self._mx, data)
        obs = self._get_obs(data)

        return _make_state(
            pipeline_state=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
        )

    def step(self, state, action):
        """Advance the simulation by one timestep."""
        jnp = self._jnp

        prev_obs = state.obs
        # Clip the raw policy output to [-1, 1] defensively (PPO's squashing
        # usually keeps it there, but an unsquashed action head could overshoot
        # and produce NaNs through the linear rescale below).
        raw_action = jnp.clip(action[: self._nu], -1.0, 1.0)
        ctrl = self._ctrl_center + raw_action * self._ctrl_halfspan

        data = state.pipeline_state.replace(ctrl=ctrl)
        data = self._mjx.step(self._mx, data)
        obs = self._get_obs(data)

        if self._reward_fn is not None:
            reward = self._reward_fn(prev_obs, raw_action, obs)
        else:
            reward = jnp.float32(-0.1) * jnp.sum(raw_action ** 2)

        diverged = jnp.any(jnp.isnan(data.qpos))
        if self._done_fn is not None:
            early_done = self._done_fn(obs)
            done = jnp.where(
                jnp.logical_or(diverged, early_done),
                jnp.float32(1.0),
                jnp.float32(0.0),
            )
        else:
            done = jnp.where(diverged, jnp.float32(1.0), jnp.float32(0.0))

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self, data):
        jnp = self._jnp
        return jnp.concatenate([data.qpos, data.qvel])
