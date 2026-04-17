"""GamepadDevice: thin wrapper around a pygame joystick.

pygame is imported lazily inside GamepadDevice so the module is safe to import
on machines without a display server or without pygame installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class GamepadState:
    """Snapshot of all gamepad inputs at a single point in time."""

    axes: list[float] = field(default_factory=list)
    # "button_0", "button_1", ... → bool
    buttons: dict[str, bool] = field(default_factory=dict)
    connected: bool = False
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "axes": list(self.axes),
            "buttons": dict(self.buttons),
            "connected": self.connected,
            "ts": self.ts,
        }


_DISCONNECTED = GamepadState(axes=[], buttons={}, connected=False)


class GamepadDevice:
    """Polls one pygame joystick.  Thread-safe for read; init/close from one thread."""

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._joystick = None
        self._pygame_ready = False
        self._init_pygame()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            n = pygame.joystick.get_count()
            if n == 0:
                log.info("GamepadDevice: no joystick detected")
                return
            joy = pygame.joystick.Joystick(min(self._device_id, n - 1))
            joy.init()
            self._joystick = joy
            self._pygame_ready = True
            log.info(
                f"GamepadDevice: connected to '{joy.get_name()}' "
                f"(axes={joy.get_numaxes()}, buttons={joy.get_numbuttons()})"
            )
        except Exception as exc:
            log.warning(f"GamepadDevice: pygame init failed — {exc}")

    def poll(self) -> GamepadState:
        """Return the current gamepad state.  Drains pygame event queue."""
        if not self._pygame_ready or self._joystick is None:
            return _DISCONNECTED

        try:
            import pygame
            pygame.event.pump()
            joy = self._joystick

            if not joy.get_init():
                return _DISCONNECTED

            axes = [joy.get_axis(i) for i in range(joy.get_numaxes())]
            buttons = {
                f"button_{i}": bool(joy.get_button(i))
                for i in range(joy.get_numbuttons())
            }
            return GamepadState(
                axes=axes,
                buttons=buttons,
                connected=True,
                ts=time.time(),
            )
        except Exception as exc:
            log.debug(f"GamepadDevice.poll: {exc}")
            return _DISCONNECTED

    def close(self) -> None:
        try:
            if self._joystick is not None:
                self._joystick.quit()
                self._joystick = None
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        return self._pygame_ready and self._joystick is not None
