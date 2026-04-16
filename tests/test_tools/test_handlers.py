"""Integration tests for the tool handler pipeline via dispatch_tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from robo_garden.claude.tool_handlers import dispatch_tool
from robo_garden.config import ROBOTS_DIR, WORKSPACE_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_workspace():
    """Remove any test robot/env files before and after each test."""
    def _cleanup():
        for pattern in ["test_*.xml", "test_*.urdf"]:
            for path in ROBOTS_DIR.glob(pattern):
                path.unlink(missing_ok=True)
        env_dir = WORKSPACE_DIR / "environments"
        if env_dir.exists():
            for path in env_dir.glob("test_*.xml"):
                path.unlink(missing_ok=True)

    _cleanup()
    yield
    _cleanup()


# ---------------------------------------------------------------------------
# MJCF fixtures
# ---------------------------------------------------------------------------

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

ENV_MJCF = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
  </worldbody>
</mujoco>"""

GOOD_REWARD = """
def compute_reward(obs, action, next_obs, info):
    return float(-np.sum(action**2)), {"ctrl_cost": float(-np.sum(action**2))}
"""

BAD_REWARD = """
def compute_reward(obs, action, next_obs, info):
    import os  # forbidden
    return 0.0, {}
"""


# ---------------------------------------------------------------------------
# Test 1: generate_robot → simulate → evaluate round-trip
# ---------------------------------------------------------------------------

def test_generate_robot_returns_success():
    result = dispatch_tool("generate_robot", {"name": "test_pendulum", "robot_xml": PENDULUM_MJCF})
    assert result["success"] is True
    assert result["robot_name"] == "test_pendulum"
    assert "model_info" in result


def test_simulate_after_generate():
    dispatch_tool("generate_robot", {"name": "test_pendulum", "robot_xml": PENDULUM_MJCF})
    result = dispatch_tool("simulate", {"robot_name": "test_pendulum", "duration_seconds": 0.5})
    assert result["success"] is True
    assert "stable" in result
    assert "simulation_id" in result


def test_evaluate_after_simulate():
    dispatch_tool("generate_robot", {"name": "test_pendulum", "robot_xml": PENDULUM_MJCF})
    dispatch_tool("simulate", {"robot_name": "test_pendulum", "duration_seconds": 0.5})
    result = dispatch_tool("evaluate", {
        "simulation_id": "test_pendulum",
        "metrics": ["stability", "diverged", "com_height"],
    })
    assert result["success"] is True
    assert "stability" in result["metrics"]
    assert 0.0 <= result["metrics"]["stability"] <= 1.0


def test_evaluate_without_simulate_returns_error():
    result = dispatch_tool("evaluate", {"simulation_id": "nonexistent_robot", "metrics": ["stability"]})
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Test 2: generate_robot with invalid MJCF returns error
# ---------------------------------------------------------------------------

def test_generate_robot_invalid_mjcf():
    result = dispatch_tool("generate_robot", {"name": "test_bad", "robot_xml": "<not valid xml"})
    assert result["success"] is False
    assert "errors" in result


# ---------------------------------------------------------------------------
# Test 3: simulate missing robot returns error
# ---------------------------------------------------------------------------

def test_simulate_missing_robot():
    result = dispatch_tool("simulate", {"robot_name": "does_not_exist_xyz", "duration_seconds": 0.5})
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Test 4: generate_environment saves file
# ---------------------------------------------------------------------------

def test_generate_environment():
    result = dispatch_tool("generate_environment", {"name": "test_flat", "mjcf_xml": ENV_MJCF})
    assert result["success"] is True
    assert "env_path" in result
    assert Path(result["env_path"]).exists()


# ---------------------------------------------------------------------------
# Test 5: generate_reward validates and stores
# ---------------------------------------------------------------------------

def test_generate_reward_valid():
    result = dispatch_tool("generate_reward", {
        "task_description": "minimize control effort",
        "reward_code": GOOD_REWARD,
    })
    assert result["success"] is True
    assert "reward_function_id" in result


def test_generate_reward_invalid():
    result = dispatch_tool("generate_reward", {
        "task_description": "test",
        "reward_code": BAD_REWARD,
    })
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Test 6: query_catalog returns results
# ---------------------------------------------------------------------------

def test_query_catalog_actuators():
    result = dispatch_tool("query_catalog", {"catalog": "actuators", "query": "dynamixel"})
    assert result["catalog"] == "actuators"
    assert result["count"] >= 0


def test_query_catalog_robots():
    result = dispatch_tool("query_catalog", {"catalog": "robots", "query": "ant"})
    assert result["catalog"] == "robots"
