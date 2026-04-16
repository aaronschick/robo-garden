"""Single gymnasium.Env wrapping any MuJoCo MJCF — SB3-compatible."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class MuJoCoGymEnv(gym.Env):
    """Single gymnasium.Env wrapping a MuJoCo model loaded from MJCF XML.

    Compatible with stable-baselines3. Passes a standard (obs, action, next_obs)
    tuple to ``reward_fn``; if None a small control-effort penalty is used.

    Args:
        mjcf_xml: MJCF model string.
        max_episode_steps: Truncation horizon per episode.
        reward_fn: ``(obs, action, next_obs) -> float`` or None.
        done_fn: ``(next_obs) -> bool`` for early termination or None.
        action_scale: Multiplier applied to raw ``ctrlrange`` of actuators.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mjcf_xml: str,
        max_episode_steps: int = 500,
        reward_fn: Callable | None = None,
        done_fn: Callable | None = None,
        init_noise: float = 0.05,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for MuJoCoGymEnv")

        self.mjcf_xml = mjcf_xml
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.init_noise = init_noise

        self._model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._data = mujoco.MjData(self._model)

        nq = self._model.nq
        nv = self._model.nv
        nu = self._model.nu
        obs_dim = nq + nv

        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(nu, -1.0, dtype=np.float32),
            high=np.full(nu, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        noise = self._rng.uniform(-self.init_noise, self.init_noise, self._model.nq)
        self._data.qpos[:] += noise
        self._data.qvel[:] += self._rng.uniform(-self.init_noise, self.init_noise, self._model.nv)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        prev_obs = self._get_obs()

        # Scale action [-1, 1] → actual ctrlrange
        act_range = self._model.actuator_ctrlrange
        if act_range.shape[0] > 0:
            lo, hi = act_range[:, 0], act_range[:, 1]
            scaled = lo + (np.asarray(action, dtype=np.float32) + 1.0) * 0.5 * (hi - lo)
        else:
            scaled = np.asarray(action, dtype=np.float32)

        self._data.ctrl[:] = scaled
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()

        if self.reward_fn is not None:
            raw_r = self.reward_fn(prev_obs, action, obs)
            reward = float(raw_r[0]) if isinstance(raw_r, tuple) else float(raw_r)
        else:
            reward = float(-0.1 * np.sum(action ** 2))

        diverged = bool(np.any(np.isnan(obs)))
        early_done = bool(self.done_fn(obs)) if self.done_fn is not None else False
        terminated = diverged or early_done
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos, self._data.qvel]).astype(np.float32)


# ---------------------------------------------------------------------------
# Cartpole helpers
# ---------------------------------------------------------------------------

def cartpole_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
    """NumPy reward for cartpole balancing — used by SB3 (CPU) path.

    obs layout (from MJCF): [cart_pos, pole_angle, cart_vel, pole_ang_vel]
    Maximised when pole is upright (angle ≈ 0) and cart is near centre.
    """
    pole_angle = float(next_obs[1])
    cart_pos = float(next_obs[0])
    ctrl_cost = 0.001 * float(np.sum(action ** 2))
    return float(np.cos(pole_angle)) - 0.1 * cart_pos ** 2 - ctrl_cost


def cartpole_done(next_obs: np.ndarray) -> bool:
    """Terminate when the pole falls past ±45° or the cart leaves the track."""
    pole_angle = abs(float(next_obs[1]))
    cart_pos = abs(float(next_obs[0]))
    return pole_angle > 0.785 or cart_pos > 2.3  # 45°, track limit


# ---------------------------------------------------------------------------
# JAX-native variants (JIT-compilable) — used by MJX/Brax GPU path
# ---------------------------------------------------------------------------

def cartpole_reward_jax(obs, action, next_obs):
    """JAX-native cartpole reward — JIT-compilable, for Brax PPO on GPU.

    Identical logic to cartpole_reward but uses jnp so it survives JAX tracing.
    """
    import jax.numpy as jnp

    pole_angle = next_obs[1]
    cart_pos = next_obs[0]
    return jnp.cos(pole_angle) - jnp.float32(0.1) * cart_pos ** 2 - jnp.float32(0.001) * jnp.sum(action ** 2)


def cartpole_done_jax(next_obs) -> bool:
    """JAX-native early-termination check for cartpole — JIT-compilable."""
    import jax.numpy as jnp

    pole_angle = jnp.abs(next_obs[1])
    cart_pos = jnp.abs(next_obs[0])
    return jnp.logical_or(pole_angle > jnp.float32(0.785), cart_pos > jnp.float32(2.3))
