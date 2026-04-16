"""IsaacBridge: WebSocket client that streams robot/simulation state to Isaac Sim.

Design principles:
- All public methods are safe to call when disconnected (return False / no-op).
- Uses a background daemon thread + asyncio event loop so callers stay synchronous.
- A threading.Event signals connection success/failure back to connect().
- Queue is capped to 200 messages; overflow silently drops (display is best-effort).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_STOP = None  # sentinel: None in the queue means "disconnect and stop"


class IsaacBridge:
    def __init__(self) -> None:
        self._url: str | None = None
        self._connected: bool = False
        self._send_queue: queue.Queue = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._ready: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, url: str = "ws://localhost:8765") -> bool:
        """Attempt to connect to the Isaac Sim WebSocket server.

        Blocks for up to 3 seconds waiting for the connection to establish.
        Returns True if connected, False if Isaac Sim is not reachable.
        """
        self._url = url
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="isaac-bridge"
        )
        self._thread.start()
        self._ready.wait(timeout=3.0)
        return self._connected

    def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._send_queue.put(_STOP)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Public API (all no-ops when disconnected)
    # ------------------------------------------------------------------

    def send_robot(self, name: str, robot_path: Path, fmt: str = "mjcf") -> bool:
        """Tell Isaac Sim to import a robot from the shared workspace."""
        from robo_garden.isaac.protocol import make_load_robot
        return self._send(make_load_robot(name, robot_path, fmt=fmt))

    def stream_simulation(
        self,
        result,
        robot_name: str,
        batch_size: int = 10,
    ) -> None:
        """Queue all simulation frames for playback in Isaac Sim.

        Args:
            result: SimulationResult from core/simulation.py
            robot_name: Name matching the previously loaded robot.
            batch_size: Frames per WebSocket message (default 10 → 50 pkt/s at 500Hz).
        """
        if not self._connected:
            return

        from robo_garden.isaac.protocol import make_sim_frame_batch, make_sim_end

        nq = result.qpos_trajectory.shape[1]
        num_steps = result.num_steps

        for batch_start in range(0, num_steps, batch_size):
            batch_end = min(batch_start + batch_size, num_steps)
            frames = result.qpos_trajectory[batch_start:batch_end]
            timesteps = [
                float(result.timestep * i) for i in range(batch_start, batch_end)
            ]
            self._send(make_sim_frame_batch(robot_name, frames, timesteps, nq))

        self._send(make_sim_end(robot_name, result.stable, result.diverged, result.summary))

    def send_training_update(self, robot_name: str, timestep: int, mean_reward: float, **kwargs) -> None:
        """Send a training progress update for the Isaac Sim overlay."""
        from robo_garden.isaac.protocol import make_train_update
        self._send(make_train_update(robot_name, timestep, mean_reward, **kwargs))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> bool:
        if not self._connected:
            return False
        try:
            self._send_queue.put_nowait(json.dumps(msg, default=str))
            return True
        except queue.Full:
            log.debug("Isaac bridge send queue full, dropping message")
            return False

    def _run_loop(self) -> None:
        """Entry point for the background sender thread."""
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        try:
            import websockets

            async with websockets.connect(self._url, open_timeout=3.0) as ws:
                self._connected = True
                self._ready.set()
                log.info(f"Isaac bridge connected to {self._url}")

                while True:
                    try:
                        msg = self._send_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                        continue

                    if msg is _STOP:
                        break
                    await ws.send(msg)

        except Exception as e:
            log.debug(f"Isaac bridge: {e}")
        finally:
            self._connected = False
            self._ready.set()  # unblock connect() on failure
