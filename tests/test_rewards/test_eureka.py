"""Tests for EurekaLoop — compile-only path (no robot_mjcf)."""
from __future__ import annotations

import pytest

from robo_garden.rewards.eureka import EurekaLoop

GOOD_REWARD = """
def compute_reward(obs, action, next_obs, info):
    return float(-np.sum(action**2)), {"ctrl_cost": float(-np.sum(action**2))}
"""

BAD_REWARD = """
def compute_reward(obs, action, next_obs, info):
    import os
    return 0.0, {}
"""


def test_eureka_compile_only():
    """Single iteration with no robot_mjcf — validates compile path only."""
    loop = EurekaLoop(max_iterations=1)
    result = loop.run_iteration(
        task_description="minimize control effort",
        observation_space_desc="joint positions and velocities",
        reward_code=GOOD_REWARD,
    )
    assert result is not None
    assert loop.history


def test_eureka_rejects_bad_reward():
    """Bad reward code (forbidden import) should raise or return error."""
    loop = EurekaLoop(max_iterations=1)
    with pytest.raises(Exception):
        loop.run_iteration(
            task_description="test",
            observation_space_desc="obs",
            reward_code=BAD_REWARD,
        )
