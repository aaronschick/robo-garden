"""Isaac Sim standalone server for robo-garden.

Runs inside Isaac Sim's Python 3.11 environment. Opens a WebSocket server
on port 8765, receives commands from robo-garden (Python 3.12), and drives
the Isaac Sim viewport.

Architecture:
  - Main thread:   Isaac Sim Kit render loop (world.step + simulation_app.update)
  - Asyncio thread: WebSocket server — receives messages, puts into _msg_queue
  - Main thread drains _msg_queue between render steps

This separation is required: world.step() must run on the main thread;
asyncio.run() cannot share the main thread with Kit's event loop.

Launch:
    .\\isaac_server\\launch.ps1              (GUI mode, default)
    .\\isaac_server\\launch.ps1 --headless   (no viewport, for testing)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import sys
import threading
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("isaac_server")

# Parse flags before SimulationApp init
_headless = "--headless" in sys.argv

HOST = "0.0.0.0"
PORT = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--port"), "8765"))

# Queue for passing messages from the WS thread to the main (Kit) thread
_msg_queue: queue.Queue = queue.Queue(maxsize=1000)

# ---------------------------------------------------------------------------
# Isaac Sim init — must happen before any omni.* imports
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

SIM_CONFIG = {
    "headless": _headless,
    # GUI mode: RayTracedLighting gives best visuals within 8GB VRAM budget
    # Headless mode: RasterizerOpenGL avoids RTX init crash (no window needed)
    "renderer": "RasterizerOpenGL" if _headless else "RayTracedLighting",
    "anti_aliasing": 0,
    "width": 1280,
    "height": 720,
}
log.info(f"Starting Isaac Sim (headless={_headless})...")
simulation_app = SimulationApp(SIM_CONFIG)

# Now safe to import omni modules
import omni
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView

# MJCF importer — try Isaac Sim 5.x path first, fall back to 4.x
try:
    from isaacsim.asset.importer.mjcf import _mjcf as mjcf_importer  # type: ignore
    from isaacsim.asset.importer.mjcf.mjcf_importer import ImportConfig  # type: ignore
    MJCF_IMPORTER_AVAILABLE = True
    log.info("MJCF importer ready (isaacsim 5.x)")
except ImportError:
    try:
        from omni.importer.mjcf import _mjcf as mjcf_importer  # type: ignore
        from omni.importer.mjcf.mjcf_importer import ImportConfig  # type: ignore
        MJCF_IMPORTER_AVAILABLE = True
        log.info("MJCF importer ready (omni 4.x)")
    except ImportError:
        mjcf_importer = None
        ImportConfig = None
        MJCF_IMPORTER_AVAILABLE = False
        log.warning("MJCF importer not available — robots cannot be loaded")

# Global state (accessed from main thread only)
world = World(stage_units_in_meters=1.0)
robots: dict[str, ArticulationView] = {}

# Shared flag: main thread sets this to signal WS thread to stop
_shutdown = threading.Event()

# ---------------------------------------------------------------------------
# WebSocket server (runs in a background asyncio thread)
# ---------------------------------------------------------------------------

async def _ws_handler(ws) -> None:
    addr = getattr(ws, "remote_address", "?")
    log.info(f"Client connected: {addr}")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"Invalid JSON: {raw[:80]}")
                continue

            msg_type = msg.get("type")
            if msg_type == "PING":
                await ws.send(json.dumps({"type": "PONG", "ts": msg.get("ts")}))
            else:
                # Forward everything else to the main thread
                msg["_ws_ref"] = id(ws)
                try:
                    _msg_queue.put_nowait(msg)
                except queue.Full:
                    log.warning("Message queue full, dropping")
    except Exception as exc:
        log.info(f"Client {addr} disconnected: {exc}")


def _run_ws_server() -> None:
    """Entry point for the WebSocket background thread."""
    import websockets

    async def _serve():
        log.info(f"WebSocket server listening on ws://{HOST}:{PORT}")
        async with websockets.serve(_ws_handler, HOST, PORT):
            while not _shutdown.is_set():
                await asyncio.sleep(0.1)

    asyncio.run(_serve())


# ---------------------------------------------------------------------------
# Message processing — called from the main Kit thread
# ---------------------------------------------------------------------------

def _process_messages() -> None:
    """Drain the message queue; called once per Kit render step."""
    while True:
        try:
            msg = _msg_queue.get_nowait()
        except queue.Empty:
            break
        _dispatch(msg)


def _dispatch(msg: dict) -> None:
    msg_type = msg.get("type")
    if msg_type == "LOAD_ROBOT":
        _handle_load_robot(msg)
    elif msg_type == "SIM_FRAME_BATCH":
        _handle_sim_frame_batch(msg)
    elif msg_type == "SIM_END":
        _handle_sim_end(msg)
    elif msg_type == "TRAIN_UPDATE":
        log.info(
            f"Training [{msg.get('robot_name')}] "
            f"step={msg.get('timestep')} "
            f"reward={msg.get('mean_reward', 0):.3f}"
        )


def _handle_load_robot(msg: dict) -> None:
    name = msg["name"]
    path = msg["path"]
    fmt = msg.get("format", "mjcf")
    log.info(f"Loading robot '{name}' from {path} (format={fmt})")

    try:
        # Remove existing robot with same name
        if name in robots:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim:
                omni.kit.commands.execute("DeletePrims", paths=[f"/World/{name}"])
            del robots[name]

        if fmt == "urdf":
            # URDF importer — try Isaac Sim 5.x path first, fall back to 4.x
            try:
                from isaacsim.asset.importer.urdf import _urdf as urdf_importer  # type: ignore
                from isaacsim.asset.importer.urdf.urdf_importer import ImportConfig as UrdfConfig  # type: ignore
            except ImportError:
                from omni.importer.urdf import _urdf as urdf_importer  # type: ignore
                from omni.importer.urdf.urdf_importer import ImportConfig as UrdfConfig  # type: ignore

            urdf_config = UrdfConfig()
            urdf_config.fix_base = False
            urdf_config.import_inertia_tensor = True
            success = urdf_importer.import_robot(path, "/World", urdf_config)
            if not success:
                raise RuntimeError("URDF importer returned failure")
        else:
            if not MJCF_IMPORTER_AVAILABLE:
                log.error("MJCF importer not available")
                return

            config = ImportConfig()
            config.fix_base = False
            config.import_inertia_tensor = True
            config.default_drive_type = 1  # position drive

            success = mjcf_importer.import_robot(path, "/World", config)
            if not success:
                raise RuntimeError("MJCF importer returned failure")

        world.reset()

        prim_path_expr = f"/World/{name}*"
        articulation = ArticulationView(prim_paths_expr=prim_path_expr)
        world.scene.add(articulation)
        world.reset()

        robots[name] = articulation
        log.info(f"Robot '{name}' ready — {articulation.num_dof} DOFs")

    except Exception as exc:
        log.error(f"Failed to load '{name}': {exc}")


# Frame playback state
_playback: dict = {}  # robot_name -> list of qpos frames (consumed one per step)


def _handle_sim_frame_batch(msg: dict) -> None:
    name = msg.get("robot_name", "")
    nq = msg["nq"]

    if "qpos_b64" in msg:
        raw = base64.b64decode(msg["qpos_b64"])
        frames = np.frombuffer(raw, dtype=np.float32).reshape(msg["batch_size"], nq)
    else:
        frames = np.array(msg["qpos_json"], dtype=np.float32)

    if name not in _playback:
        _playback[name] = []
    _playback[name].extend(frames.tolist())


def _handle_sim_end(msg: dict) -> None:
    name = msg.get("robot_name", "")
    _playback.pop(name, None)  # clear remaining frames
    if name in robots:
        articulation = robots[name]
        articulation.set_joint_positions(np.zeros(articulation.num_dof))
    log.info(
        f"Sim ended — '{name}' stable={msg.get('stable')} diverged={msg.get('diverged')}"
    )


def _apply_next_playback_frames() -> None:
    """Apply one pending frame per robot per render step."""
    for name, frames in list(_playback.items()):
        if not frames:
            del _playback[name]
            continue
        if name not in robots:
            continue
        articulation = robots[name]
        frame = np.array(frames.pop(0), dtype=np.float32)
        n_dof = articulation.num_dof
        positions = frame[-n_dof:] if len(frame) > n_dof else frame
        articulation.set_joint_positions(positions)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Start WebSocket server in background thread
    ws_thread = threading.Thread(target=_run_ws_server, daemon=True, name="ws-server")
    ws_thread.start()

    # Set up world
    world.scene.add_default_ground_plane()
    world.reset()
    log.info("Isaac Sim ready — waiting for robo-garden connection on port 8765")

    # Kit render loop — must stay on main thread
    while simulation_app.is_running():
        _process_messages()
        _apply_next_playback_frames()
        world.step(render=True)
        simulation_app.update()

    _shutdown.set()
    log.info("Shutting down")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
