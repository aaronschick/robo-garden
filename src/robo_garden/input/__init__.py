"""Gamepad input package for Robo Garden.

Provides a 60 Hz polling daemon (GamepadRunner) that maps controller axes and
buttons to robot joint control targets.  pygame is imported lazily so the
package is safe to import on machines without a display.
"""

from robo_garden.input.gamepad import GamepadDevice, GamepadState
from robo_garden.input.mapping import ControlMapping
from robo_garden.input.runner import GamepadRunner

__all__ = ["GamepadDevice", "GamepadState", "ControlMapping", "GamepadRunner"]
