"""Tests for sandboxed reward function execution."""

import pytest
import numpy as np

from robo_garden.rewards.reward_runner import compile_reward_function


GOOD_REWARD = """
def compute_reward(obs, action, next_obs, info):
    forward_vel = obs[0] if len(obs) > 0 else 0.0
    energy = float(np.sum(np.square(action)))
    reward = forward_vel - 0.01 * energy
    return reward, {"forward_vel": forward_vel, "energy": energy}
"""

BAD_REWARD_IMPORT = """
import os
def compute_reward(obs, action, next_obs, info):
    return 0.0, {}
"""

BAD_REWARD_NO_FUNCTION = """
x = 42
"""


def test_compile_good_reward():
    fn = compile_reward_function(GOOD_REWARD)
    obs = np.array([1.0, 0.0, 0.0])
    reward, info = fn(obs, np.zeros(2), obs, {})
    assert isinstance(reward, float)
    assert "forward_vel" in info


def test_reject_import():
    with pytest.raises(ValueError, match="Forbidden"):
        compile_reward_function(BAD_REWARD_IMPORT)


def test_reject_missing_function():
    with pytest.raises(ValueError, match="compute_reward"):
        compile_reward_function(BAD_REWARD_NO_FUNCTION)
