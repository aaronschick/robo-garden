"""GamepadRunner: 60 Hz daemon thread that polls the gamepad and fires callbacks.

Usage::

    runner = GamepadRunner(
        on_state=my_callback,        # called every tick at 60 Hz
        on_echo=bridge_send_fn,      # called at ~10 Hz for UI echo
    )
    runner.start()
    # ... later:
    runner.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from robo_garden.input.gamepad import GamepadDevice, GamepadState

log = logging.getLogger(__name__)

_POLL_HZ = 60
_ECHO_DIVISOR = 6   # fire on_echo every 6th tick ≈ 10 Hz


class GamepadRunner:
    """Polls one gamepad at 60 Hz in a daemon thread.

    Parameters
    ----------
    on_state:
        Called with the latest ``GamepadState`` on every tick.  Runs on the
        polling thread — keep it fast (no blocking I/O).
    on_echo:
        Optional.  Called at ~10 Hz with the current state dict (suitable for
        forwarding to the Isaac Sim UI via WebSocket).
    device_id:
        Pygame joystick index.  Default 0 = first connected controller.
    """

    def __init__(
        self,
        on_state: Callable[[GamepadState], None],
        on_echo: Callable[[dict], None] | None = None,
        device_id: int = 0,
    ) -> None:
        self._on_state = on_state
        self._on_echo = on_echo
        self._device_id = device_id

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest: GamepadState | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start polling in a daemon thread.  No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="gamepad-runner",
            daemon=True,
        )
        self._thread.start()
        log.info("GamepadRunner: started")

    def stop(self) -> None:
        """Signal the loop to stop and wait up to 2 s."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("GamepadRunner: stopped")

    def latest(self) -> GamepadState | None:
        """Return the most recently polled state (or None before first poll)."""
        with self._lock:
            return self._latest

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Poll loop (daemon thread)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        device = GamepadDevice(self._device_id)
        target_dt = 1.0 / _POLL_HZ
        tick = 0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            state = device.poll()

            with self._lock:
                self._latest = state

            try:
                self._on_state(state)
            except Exception as exc:
                log.debug(f"GamepadRunner on_state error: {exc}")

            tick += 1
            if tick >= _ECHO_DIVISOR:
                tick = 0
                if self._on_echo is not None:
                    try:
                        self._on_echo(state.to_dict())
                    except Exception as exc:
                        log.debug(f"GamepadRunner on_echo error: {exc}")

            elapsed = time.perf_counter() - t0
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

        device.close()
