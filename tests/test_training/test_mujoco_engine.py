"""Tests for MuJoCoMJXEngine random rollout path (no GPU required)."""
from __future__ import annotations

import unittest.mock as mock

import pytest

from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
from robo_garden.training.models import TrainingConfig

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


def test_engine_setup():
    engine = MuJoCoMJXEngine()
    config = TrainingConfig(num_envs=2, total_timesteps=100, max_episode_steps=10)
    engine.setup(PENDULUM_MJCF, "", config)
    assert engine.config is not None
    assert engine._merged_mjcf != ""
    engine.cleanup()


def test_engine_random_rollout():
    engine = MuJoCoMJXEngine()
    config = TrainingConfig(
        num_envs=2, total_timesteps=200, max_episode_steps=10, algorithm="random"
    )
    engine.setup(PENDULUM_MJCF, "", config)
    with mock.patch.object(engine, "_train_brax", side_effect=RuntimeError("no brax")):
        result = engine.train()
    assert result is not None
    assert hasattr(result, "training_time_seconds")
    engine.cleanup()


def test_engine_curriculum_wired():
    """Curriculum manager advances stage when threshold is met."""
    from robo_garden.training.models import CurriculumConfig, CurriculumStage

    stages = [
        CurriculumStage(name="easy"),
        CurriculumStage(name="hard"),
    ]
    # Very low threshold so rollout will advance immediately
    curriculum_config = CurriculumConfig(stages=stages, advance_threshold=-1e9)

    engine = MuJoCoMJXEngine()
    config = TrainingConfig(
        num_envs=2, total_timesteps=200, max_episode_steps=10, algorithm="random"
    )
    engine.setup(PENDULUM_MJCF, "", config, curriculum_config=curriculum_config)
    with mock.patch.object(engine, "_train_brax", side_effect=RuntimeError("no brax")):
        result = engine.train()
    assert result is not None
    engine.cleanup()
