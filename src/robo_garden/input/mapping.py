"""ControlMapping: maps gamepad axes/buttons → robot joint ctrl targets.

A mapping can be loaded from a JSON file at ``workspace/robots/<name>/gamepad.json``
or constructed automatically from the robot's joint list (default mode).

Default auto-mapping
--------------------
Axis ``i`` → the ``i``-th actuated (non-free, non-ball) joint, scaled from
[-1, 1] to the joint's ``ctrl_range``.  Up to 6 axes are mapped by default.

JSON format
-----------
::

    {
        "deadzone": 0.10,
        "axis_to_joints": {
            "0": [{"joint": "FR_thigh_joint", "scale": 1.0}],
            "1": [{"joint": "FL_thigh_joint", "scale": -1.0}]
        },
        "button_actions": {
            "0": "reset",
            "1": "pause"
        }
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.core.interactive_sim import JointInfo
    from robo_garden.input.gamepad import GamepadState

log = logging.getLogger(__name__)

_MAX_AUTO_AXES = 6


@dataclass
class ControlMapping:
    """Maps controller axes to joint ctrl targets and buttons to actions."""

    # axis_index -> list of (joint_name, scale)
    axis_to_joints: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    # button_index -> action string ("reset" | "pause")
    button_actions: dict[int, str] = field(default_factory=dict)
    deadzone: float = 0.10

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_joints(
        cls,
        joints: "list[JointInfo]",
        deadzone: float = 0.10,
    ) -> "ControlMapping":
        """Build a default 1:1 axis→joint mapping from the robot's joint list."""
        actuated = [
            j for j in joints
            if j.type not in ("free", "ball") and j.actuator
        ]
        axis_to_joints = {
            i: [(j.name, 1.0)]
            for i, j in enumerate(actuated[:_MAX_AUTO_AXES])
        }
        return cls(axis_to_joints=axis_to_joints, deadzone=deadzone)

    @classmethod
    def load(cls, path: Path) -> "ControlMapping":
        """Load a mapping from a JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning(f"ControlMapping.load: could not read {path}: {exc}")
            return cls()

        axis_to_joints: dict[int, list[tuple[str, float]]] = {}
        for k, entries in data.get("axis_to_joints", {}).items():
            axis_to_joints[int(k)] = [(e["joint"], float(e.get("scale", 1.0))) for e in entries]

        button_actions = {int(k): v for k, v in data.get("button_actions", {}).items()}
        return cls(
            axis_to_joints=axis_to_joints,
            button_actions=button_actions,
            deadzone=float(data.get("deadzone", 0.10)),
        )

    @classmethod
    def load_for_robot(cls, robot_name: str) -> "ControlMapping | None":
        """Try to load ``workspace/robots/<robot_name>/gamepad.json``.

        Returns None if the file does not exist (caller should fall back to
        ``from_joints``).
        """
        from robo_garden.config import ROBOTS_DIR
        path = ROBOTS_DIR / robot_name / "gamepad.json"
        if not path.exists():
            return None
        return cls.load(path)

    # ------------------------------------------------------------------
    # Runtime application
    # ------------------------------------------------------------------

    def apply(
        self,
        state: "GamepadState",
        joints_by_name: "dict[str, JointInfo]",
    ) -> dict[str, float]:
        """Map gamepad state to ``{joint_name: ctrl_value}`` dict.

        Values are clamped to each joint's ``ctrl_range``.  Axes below the
        deadzone threshold are treated as zero.
        """
        result: dict[str, float] = {}
        for axis_idx, joint_mappings in self.axis_to_joints.items():
            if axis_idx >= len(state.axes):
                continue
            raw = float(state.axes[axis_idx])
            if abs(raw) < self.deadzone:
                raw = 0.0

            for joint_name, scale in joint_mappings:
                ji = joints_by_name.get(joint_name)
                if ji is None:
                    continue
                lo, hi = ji.ctrl_range
                mid = 0.5 * (lo + hi)
                half = 0.5 * (hi - lo)
                value = mid + raw * half * scale
                result[joint_name] = max(lo, min(hi, value))

        return result

    def button_pressed(self, state: "GamepadState", button_index: int) -> bool:
        return bool(state.buttons.get(f"button_{button_index}", False))
