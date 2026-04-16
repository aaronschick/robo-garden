"""MJX batched environment using jax.vmap for GPU-parallel simulation."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Gymnasium spaces imported lazily to avoid hard dep at module level
_GYM_AVAILABLE = True
try:
    from gymnasium import spaces
except ImportError:
    _GYM_AVAILABLE = False


class MJXVectorizedEnv:
    """Vectorized MuJoCo environment using MJX + JAX for GPU parallelism.

    All data stays on GPU as JAX arrays. CPU fallback activates automatically
    when MJX/CUDA is unavailable.
    """

    def __init__(
        self,
        mjcf_xml: str,
        num_envs: int = 128,
        max_episode_steps: int = 1000,
        seed: int = 0,
        reward_fn=None,
    ):
        self.mjcf_xml = mjcf_xml
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.reward_fn = reward_fn  # callable(obs, action, next_obs) -> float, or None
        self._initialized = False
        self._use_mjx = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """JIT-compile the vectorized step. Called lazily on first reset()."""
        import mujoco

        self._mj_model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        self._nq = self._mj_model.nq
        self._nv = self._mj_model.nv
        self._nu = self._mj_model.nu
        # Floating-base robots have nq > nv (quaternion adds one extra DOF)
        self._floating_base = self._mj_model.nq > self._mj_model.nv
        self._obs_dim = (self._nq - (1 if self._floating_base else 0)) + self._nv + self._nu

        try:
            import jax
            import jax.numpy as jnp
            from mujoco import mjx

            self._jax = jax
            self._jnp = jnp
            self._mjx = mjx

            self._mx = mjx.put_model(self._mj_model)
            _base_data = mujoco.MjData(self._mj_model)
            self._base_mjx_data = mjx.put_data(self._mj_model, _base_data)

            # JIT-compiled batched step: (model, batch_data) -> batch_data
            self._batch_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
            self._use_mjx = True
            log.info(
                f"MJX vectorized env: {self.num_envs} envs, "
                f"obs_dim={self._obs_dim}, nu={self._nu}, devices={jax.devices()}"
            )
        except Exception as e:
            log.warning(f"MJX unavailable ({e}), falling back to CPU sequential stepping")
            self._use_mjx = False
            self._cpu_datas = [
                mujoco.MjData(self._mj_model) for _ in range(self.num_envs)
            ]

        self._step_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._initialized = True

    # ------------------------------------------------------------------
    # Gymnasium-style interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset all environments. Returns (obs [num_envs, obs_dim], info)."""
        if not self._initialized:
            self.initialize()

        self._step_counts[:] = 0

        if self._use_mjx:
            import jax
            keys = jax.random.split(jax.random.PRNGKey(self.seed), self.num_envs)

            def _reset_single(key):
                d = self._base_mjx_data.replace(
                    qpos=self._base_mjx_data.qpos + jax.random.normal(key, (self._nq,)) * 0.01,
                )
                return d

            self._batch_data = jax.vmap(_reset_single)(keys)
            obs = self._extract_obs_mjx(self._batch_data)
        else:
            import mujoco
            for d in self._cpu_datas:
                mujoco.mj_resetData(self._mj_model, d)
            obs = self._extract_obs_cpu()

        return obs, {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all environments. Returns (obs, rewards, terminated, truncated, info)."""
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")

        actions = np.asarray(actions, dtype=np.float32)
        prev_obs = self._extract_obs_mjx(self._batch_data) if self._use_mjx else self._extract_obs_cpu()

        if self._use_mjx:
            jnp = self._jnp
            jax = self._jax

            ctrl = jnp.array(actions, dtype=jnp.float32)

            def _apply_ctrl(data, u):
                return data.replace(ctrl=u)

            self._batch_data = jax.vmap(_apply_ctrl)(self._batch_data, ctrl)
            self._batch_data = self._batch_step(self._mx, self._batch_data)
            obs = self._extract_obs_mjx(self._batch_data)

            qpos_np = np.array(self._batch_data.qpos)
            terminated = np.any(np.isnan(qpos_np), axis=1)
        else:
            import mujoco
            for i, d in enumerate(self._cpu_datas):
                d.ctrl[:] = actions[i, : self._nu]
                mujoco.mj_step(self._mj_model, d)
            obs = self._extract_obs_cpu()
            terminated = np.zeros(self.num_envs, dtype=bool)

        self._step_counts += 1
        truncated = self._step_counts >= self.max_episode_steps

        # Default reward: penalise control effort; forward if obs has velocity
        if self.reward_fn is not None:
            rewards = np.array([
                self.reward_fn(prev_obs[i], actions[i], obs[i])
                for i in range(self.num_envs)
            ], dtype=np.float32)
        else:
            ctrl_cost = 0.1 * np.sum(actions ** 2, axis=-1)
            rewards = -ctrl_cost

        # Auto-reset envs that are done
        done = terminated | truncated
        if done.any():
            reset_obs, _ = self._partial_reset(done)
            obs[done] = reset_obs[done]
            self._step_counts[done] = 0

        return obs, rewards, terminated, truncated, {"done": done}

    def close(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    def _extract_obs_mjx(self, batch_data) -> np.ndarray:
        jnp = self._jnp
        qpos = batch_data.qpos
        qvel = batch_data.qvel
        ctrl = batch_data.ctrl
        if self._floating_base:
            qpos = qpos[:, 1:]  # drop root x translation for translation-invariance
        obs = jnp.concatenate([qpos, qvel, ctrl], axis=-1)
        return np.array(obs, dtype=np.float32)

    def _extract_obs_cpu(self) -> np.ndarray:
        obs = []
        for d in self._cpu_datas:
            qpos = d.qpos[1:] if self._floating_base else d.qpos
            obs.append(np.concatenate([qpos, d.qvel, d.ctrl]))
        return np.array(obs, dtype=np.float32)

    def _partial_reset(self, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        """Reset only environments where mask is True."""
        if self._use_mjx:
            import jax
            jnp = self._jnp
            keys = jax.random.split(jax.random.PRNGKey(self.seed + int(self._step_counts.sum())), self.num_envs)

            def _maybe_reset(data, key, should_reset):
                fresh = self._base_mjx_data.replace(
                    qpos=self._base_mjx_data.qpos + jax.random.normal(key, (self._nq,)) * 0.01,
                )
                return jax.tree.map(lambda f, b: jnp.where(should_reset, b, f), data, fresh)

            mask_jnp = jnp.array(mask)
            self._batch_data = jax.vmap(_maybe_reset)(self._batch_data, keys, mask_jnp)
            obs = self._extract_obs_mjx(self._batch_data)
        else:
            import mujoco
            obs = self._extract_obs_cpu()
            for i, should_reset in enumerate(mask):
                if should_reset:
                    mujoco.mj_resetData(self._mj_model, self._cpu_datas[i])
            obs = self._extract_obs_cpu()

        return obs, {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        if not self._initialized:
            self.initialize()
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        if not self._initialized:
            self.initialize()
        return self._nu
