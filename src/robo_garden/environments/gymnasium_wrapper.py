"""Wrap MuJoCo environments as Gymnasium Env for standardized RL interface."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class MuJoCoRobotEnv(gym.Env):
    """A Gymnasium environment wrapping a MuJoCo robot + environment.

    This is the bridge between Robo Garden's robot/environment definitions
    and standard RL training loops.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot_mjcf: str,
        reward_fn=None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(robot_mjcf)
        self.data = mujoco.MjData(self.model)
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0

        # Observation: qpos + qvel
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Action: control signals
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        info = {"time": self.data.time, "step": self._step_count}

        # Compute reward
        if self.reward_fn is not None:
            reward, reward_info = self.reward_fn(obs, action, obs, info)
            info.update(reward_info)
        else:
            reward = 0.0

        terminated = bool(np.any(np.isnan(obs)))
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info
