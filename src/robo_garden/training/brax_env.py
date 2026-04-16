"""JAX-native Brax environment backed by MuJoCo MJX.

Used by MuJoCoMJXEngine._train_brax when JAX + MJX + Brax are available
(Linux / WSL2 with CUDA).  This module is intentionally imported lazily so
that it is safe to import robo_garden on Windows where mujoco-mjx/brax are
not installed.
"""

from __future__ import annotations

from typing import Callable


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

        self._mj_model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._mx = mjx.put_model(self._mj_model)

        _base_mj_data = mujoco.MjData(self._mj_model)
        self._base_data = mjx.put_data(self._mj_model, _base_mj_data)

        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._init_noise_scale = init_noise_scale

        self._nq = self._mj_model.nq
        self._nv = self._mj_model.nv
        self._nu = self._mj_model.nu

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
        data = state.pipeline_state.replace(ctrl=action[: self._nu])
        data = self._mjx.step(self._mx, data)
        obs = self._get_obs(data)

        if self._reward_fn is not None:
            reward = self._reward_fn(prev_obs, action, obs)
        else:
            reward = jnp.float32(-0.1) * jnp.sum(action ** 2)

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
