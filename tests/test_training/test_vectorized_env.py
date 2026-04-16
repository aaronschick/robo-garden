"""Tests for MJXVectorizedEnv in CPU fallback mode."""
from __future__ import annotations

import numpy as np
import pytest

from robo_garden.training.vectorized_env import MJXVectorizedEnv

PENDULUM_MJCF = """<mujoco>
  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-180 180"/>
      <geom type="capsule" size="0.05 0.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="1"/>
  </actuator>
</mujoco>"""


def test_vectorized_env_init():
    env = MJXVectorizedEnv(PENDULUM_MJCF, num_envs=2, max_episode_steps=10)
    assert env.num_envs == 2
    assert env.obs_dim > 0
    assert env.action_dim > 0


def test_vectorized_env_reset():
    env = MJXVectorizedEnv(PENDULUM_MJCF, num_envs=2, max_episode_steps=10)
    obs, info = env.reset()
    assert obs.shape == (2, env.obs_dim)


def test_vectorized_env_step():
    env = MJXVectorizedEnv(PENDULUM_MJCF, num_envs=2, max_episode_steps=10)
    env.reset()
    actions = np.zeros((2, env.action_dim), dtype=np.float32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert obs.shape == (2, env.obs_dim)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
