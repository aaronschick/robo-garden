"""IsaacBridge: bidirectional WebSocket client for the Studio UI.

Design principles:
- All public methods are safe to call when disconnected (return False / no-op).
- Uses a background daemon thread + asyncio event loop so callers stay synchronous.
- A threading.Event signals connection success/failure back to connect().
- Outbound queue is capped to 200 messages; overflow silently drops (display
  is best-effort — dropping a frame is better than blocking physics).
- Inbound messages (CHAT_MESSAGE, JOINT_TARGET, ...) are delivered by calling
  the registered ``on_message`` callback ON THE ASYNC THREAD.  The callback
  must be thread-safe or forward work to its own thread/queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable

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
        self._on_message: Callable[[dict], None] | None = None

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

    def set_on_message(self, callback: Callable[[dict], None] | None) -> None:
        """Register a callback fired for every inbound message from Isaac Sim.

        The callback runs on the asyncio/bridge thread — keep it short and
        thread-safe (ideally just dispatch to a queue).  Pass ``None`` to
        clear.
        """
        self._on_message = callback

    # ------------------------------------------------------------------
    # Public send API (all no-ops when disconnected)
    # ------------------------------------------------------------------

    def send_robot(self, name: str, robot_path: Path, fmt: str = "mjcf") -> bool:
        """Tell Isaac Sim to import a robot from the shared workspace."""
        from robo_garden.isaac.protocol import make_load_robot
        return self._send(make_load_robot(name, robot_path, fmt=fmt))

    def send_raw(self, msg: dict) -> bool:
        """Send an already-constructed protocol message."""
        return self._send(msg)

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

    def stream_qpos_batch(
        self,
        robot_name: str,
        frames: np.ndarray,
        timesteps: list[float],
    ) -> None:
        """Stream a batch of live qpos frames from InteractiveSim to Isaac Sim.

        Unlike ``stream_simulation``, this is called continuously as the
        background MuJoCo loop produces new state — no SIM_END at the end.
        """
        if not self._connected or frames.size == 0:
            return
        from robo_garden.isaac.protocol import make_sim_frame_batch
        nq = frames.shape[1]
        self._send(make_sim_frame_batch(robot_name, frames, timesteps, nq))

    def send_training_update(
        self,
        robot_name: str,
        timestep: int,
        mean_reward: float,
        **kwargs,
    ) -> None:
        """Send a training progress update for the Isaac Sim overlay."""
        from robo_garden.isaac.protocol import make_train_update
        self._send(make_train_update(robot_name, timestep, mean_reward, **kwargs))

    def send_train_run_start(
        self,
        run_id: str,
        robot_name: str,
        environment_name: str,
        algorithm: str,
        total_timesteps: int,
        **kwargs,
    ) -> None:
        """Announce the start of a training run to the Studio UI."""
        from robo_garden.isaac.protocol import make_train_run_start
        self._send(
            make_train_run_start(
                run_id=run_id,
                robot_name=robot_name,
                environment_name=environment_name,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                **kwargs,
            )
        )

    def send_train_run_end(
        self,
        run_id: str,
        robot_name: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Announce completion (or failure) of a training run."""
        from robo_garden.isaac.protocol import make_train_run_end
        self._send(
            make_train_run_end(
                run_id=run_id,
                robot_name=robot_name,
                success=success,
                **kwargs,
            )
        )

    def send_train_history(self, runs: list[dict]) -> None:
        """Seed the Studio's run history list with recent runs on connect."""
        from robo_garden.isaac.protocol import make_train_history
        self._send(make_train_history(runs))

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

                # Run sender + receiver concurrently
                sender = asyncio.create_task(self._sender(ws))
                receiver = asyncio.create_task(self._receiver(ws))
                done, pending = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()

        except Exception as e:
            log.debug(f"Isaac bridge: {e}")
        finally:
            self._connected = False
            self._ready.set()  # unblock connect() on failure

    async def _sender(self, ws) -> None:
        while True:
            try:
                msg = self._send_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue

            if msg is _STOP:
                break
            await ws.send(msg)

    async def _receiver(self, ws) -> None:
        """Forward inbound messages to the registered on_message callback."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"Bridge: invalid JSON inbound: {raw[:80]}")
                continue
            if self._on_message is not None:
                try:
                    self._on_message(msg)
                except Exception as exc:
                    log.warning(f"Bridge on_message callback failed: {exc}")
