"""Tests for simulation runner."""

import mujoco
import numpy as np

from robo_garden.core.simulation import simulate


def test_simulate_pendulum(minimal_mjcf):
    model = mujoco.MjModel.from_xml_string(minimal_mjcf)
    result = simulate(model, duration=1.0)
    assert not result.diverged
    assert result.num_steps > 0
    assert result.qpos_trajectory.shape[0] == result.num_steps


def test_simulate_quadruped_passive(quadruped_mjcf):
    model = mujoco.MjModel.from_xml_string(quadruped_mjcf)
    result = simulate(model, duration=0.5)
    assert not result.diverged
    assert result.num_steps > 0
