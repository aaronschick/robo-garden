"""Bridge to robot_descriptions library and MuJoCo Menagerie for reference models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ReferenceRobot:
    """A reference robot from an external library."""
    name: str
    source: str  # "robot_descriptions" | "menagerie"
    format: str  # "mjcf" | "urdf"
    path: str
    description: str = ""


def list_reference_robots() -> list[ReferenceRobot]:
    """List available reference robots from installed libraries."""
    robots = []

    try:
        import robot_descriptions
        # robot_descriptions.py provides loaders for many robots
        # Each robot has a module with MJCF_PATH and/or URDF_PATH
        for name in _KNOWN_ROBOTS:
            robots.append(ReferenceRobot(
                name=name,
                source="robot_descriptions",
                format="mjcf",
                path=f"robot_descriptions.{name}_mj_description",
            ))
    except ImportError:
        log.warning("robot_descriptions package not installed")

    return robots


def load_reference_mjcf(robot_name: str) -> str | None:
    """Load MJCF XML for a reference robot by name."""
    try:
        module_name = f"robot_descriptions.{robot_name}_mj_description"
        import importlib
        mod = importlib.import_module(module_name)
        mjcf_path = getattr(mod, "MJCF_PATH", None)
        if mjcf_path:
            from pathlib import Path
            return Path(mjcf_path).read_text()
    except Exception as e:
        log.warning(f"Could not load reference robot '{robot_name}': {e}")
    return None


# Well-known robots available in robot_descriptions
_KNOWN_ROBOTS = [
    "go2",           # Unitree Go2 quadruped
    "h1",            # Unitree H1 humanoid
    "fr3",           # Franka FR3 arm
    "ur5e",          # Universal Robots UR5e
    "shadow_hand",   # Shadow Dexterous Hand
    "a1",            # Unitree A1 quadruped
    "anymal_c",      # ANYmal C quadruped
    "iiwa14",        # KUKA iiwa 14
    "panda",         # Franka Panda arm
]
