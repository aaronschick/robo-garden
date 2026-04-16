"""Isaac Sim standalone server for robo-garden.

Runs inside Isaac Sim's Python 3.11 environment. Opens a WebSocket server
on port 8765, receives commands from the robo-garden backend (Python 3.12),
drives the Isaac Sim viewport, and hosts the Studio UI extension.

Architecture:
  - Main thread:    Isaac Sim Kit render loop (world.step + simulation_app.update)
  - Asyncio thread: WebSocket server — receives messages, dispatches to either
                    the physics main thread (LOAD_ROBOT, SIM_FRAME_BATCH, ...)
                    or the Studio UI (CHAT_REPLY, GATE_STATUS, ROBOT_META, ...)
  - Studio UI can also produce outbound messages (CHAT_MESSAGE, JOINT_TARGET,
    APPLY_FORCE, APPROVE_DESIGN) via ``broadcast(msg)`` which pushes them to
    every connected client (there is typically only one — the robo-garden
    backend).

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

# Queues + state shared between the asyncio thread and the main Kit thread
# Inbound queue is large because InteractiveSim can burst SIM_FRAME_BATCH
# messages faster than the 60Hz Kit render loop can drain them.
_msg_queue: queue.Queue = queue.Queue(maxsize=8000)          # inbound -> physics
_outbound_queue: queue.Queue = queue.Queue(maxsize=1000)     # Studio UI -> clients
_clients: set = set()                                        # connected ws instances
_clients_lock = threading.Lock()
# Robot names that exist in our Kit stage.  Updated on main thread from
# _handle_load_robot; read from the asyncio thread to skip frames for
# robots we could not import.  Set instead of dict for cheap reads.
_loaded_robot_names: set[str] = set()
_loaded_names_lock = threading.Lock()
# Most-recent LOAD_ROBOT (name -> {path, fmt}) so we can skip exact duplicates.
# See _handle_load_robot for rationale.
_last_load: dict[str, dict] = {}

# Message types routed to the Studio UI (not the physics main thread).  The
# Studio UI extension registers a callback via ``register_studio_listener``
# to receive these.
_STUDIO_UI_TYPES = {
    "CHAT_REPLY",
    "TOOL_STATUS",
    "TOOL_RESULT",
    "PHASE_CHANGED",
    "ROBOT_META",
    "GATE_STATUS",
    "TRAIN_RUN_START",
    "TRAIN_UPDATE",
    "TRAIN_RUN_END",
    "TRAIN_HISTORY",
}

# Callback set by the Studio UI extension, if loaded.  Signature: (msg: dict).
_studio_listener = None


def register_studio_listener(callback) -> None:
    """Called by ext_studio on load to receive inbound studio messages."""
    global _studio_listener
    _studio_listener = callback
    log.info("Studio UI listener registered")


def broadcast(msg: dict) -> None:
    """Send `msg` to all connected clients (typically the robo-garden backend).

    Called from the Studio UI extension (main Kit thread) to push user events.
    Safe to call at any time — overflow drops silently.
    """
    try:
        _outbound_queue.put_nowait(json.dumps(msg, default=str))
    except queue.Full:
        log.warning("Outbound queue full, dropping studio event")


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
import omni.kit.app
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView


def _ext_exists(ext_name: str) -> bool:
    """Return True if an extension with this name is discoverable in any registry."""
    try:
        mgr = omni.kit.app.get_app().get_extension_manager()
        # get_extension_id returns a non-empty fullname (name-version) when found
        ext_id = mgr.get_extension_id(ext_name)
        if ext_id:
            return True
        # Fallback: search the full list
        for info in mgr.get_extensions():
            if info.get("name") == ext_name:
                return True
    except Exception:
        pass
    return False


def _enable_ext(ext_name: str) -> bool:
    """Enable a Kit extension by name.  Returns True iff it's now enabled.

    Silently skips extensions that don't exist in any registry (so we don't
    spam the log with "Can't find extension" errors for 4.x names that are
    absent on a 5.x install).
    """
    try:
        mgr = omni.kit.app.get_app().get_extension_manager()
        if mgr.is_extension_enabled(ext_name):
            return True
        if not _ext_exists(ext_name):
            return False
        mgr.set_extension_enabled_immediate(ext_name, True)
        return mgr.is_extension_enabled(ext_name)
    except Exception as exc:
        log.warning(f"Could not enable extension {ext_name}: {exc}")
        return False


# In Isaac Sim 5.x the asset importers are lazy-loaded extensions.  Enable
# them before importing.  We try both the 5.x and 4.x names but skip ones
# that don't exist on the current install.
for _ext_id in (
    "isaacsim.asset.importer.mjcf",
    "isaacsim.asset.importer.urdf",
    "omni.importer.mjcf",
    "omni.importer.urdf",
):
    if _enable_ext(_ext_id):
        log.info(f"Enabled extension: {_ext_id}")

# MJCF importer — try Isaac Sim 5.x path first, fall back to 4.x.
# 5.x: ImportConfig lives on _mjcf; 4.x: on a submodule called mjcf_importer.
mjcf_importer = None  # the low-level C++ interface (carb.interface)
_MjcfImportConfig = None  # factory callable
MJCF_IMPORTER_AVAILABLE = False

try:
    from isaacsim.asset.importer.mjcf import _mjcf as _mjcf_mod  # type: ignore
    mjcf_importer = _mjcf_mod.acquire_mjcf_interface()
    _MjcfImportConfig = _mjcf_mod.ImportConfig
    MJCF_IMPORTER_AVAILABLE = True
    log.info("MJCF importer ready (isaacsim 5.x)")
except ImportError as exc:
    log.debug(f"isaacsim 5.x MJCF importer unavailable: {exc}")
    try:
        from omni.importer.mjcf import _mjcf as _mjcf_mod  # type: ignore
        from omni.importer.mjcf.mjcf_importer import ImportConfig as _OldMjcfConfig  # type: ignore
        mjcf_importer = _mjcf_mod.acquire_mjcf_interface()
        _MjcfImportConfig = _OldMjcfConfig
        MJCF_IMPORTER_AVAILABLE = True
        log.info("MJCF importer ready (omni 4.x)")
    except ImportError as exc2:
        log.warning(
            f"MJCF importer not available — even after enabling "
            f"isaacsim.asset.importer.mjcf. Robots cannot be loaded. "
            f"Last error: {exc2}"
        )

# Global state (accessed from main thread only)
world = World(stage_units_in_meters=1.0)
robots: dict[str, ArticulationView] = {}

# When True, the viewport is purely a playback mirror of an external physics
# engine (MuJoCo/InteractiveSim running in the robo-garden backend).  The Kit
# world still steps (to advance rendering) but articulations are written
# kinematically from SIM_FRAME_BATCH messages rather than physically simulated.
_kinematic_only = True

# Shared flag: main thread sets this to signal WS thread to stop
_shutdown = threading.Event()


# ---------------------------------------------------------------------------
# WebSocket server (runs in a background asyncio thread)
# ---------------------------------------------------------------------------

async def _ws_handler(ws) -> None:
    addr = getattr(ws, "remote_address", "?")
    log.info(f"Client connected: {addr}")
    with _clients_lock:
        _clients.add(ws)
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
            elif msg_type in _STUDIO_UI_TYPES:
                # Forward to the Studio UI extension if loaded
                if _studio_listener is not None:
                    try:
                        _studio_listener(msg)
                    except Exception as exc:
                        log.warning(f"Studio listener error: {exc}")
            else:
                # Drop playback frames for robots that never imported — this
                # prevents the queue from flooding when MJCF import fails or
                # the client streams faster than Kit can render.
                if msg_type == "SIM_FRAME_BATCH":
                    name = msg.get("robot_name", "")
                    with _loaded_names_lock:
                        known = name in _loaded_robot_names
                    if not known:
                        continue
                # Forward to the physics main thread
                msg["_ws_ref"] = id(ws)
                try:
                    _msg_queue.put_nowait(msg)
                except queue.Full:
                    log.warning("Message queue full, dropping")
    except Exception as exc:
        log.info(f"Client {addr} disconnected: {exc}")
    finally:
        with _clients_lock:
            _clients.discard(ws)


async def _outbound_pump() -> None:
    """Drain the outbound queue and fan messages out to every client."""
    while not _shutdown.is_set():
        try:
            raw = _outbound_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        with _clients_lock:
            targets = list(_clients)
        for ws in targets:
            try:
                await ws.send(raw)
            except Exception as exc:
                log.debug(f"Outbound send failed: {exc}")


def _run_ws_server() -> None:
    """Entry point for the WebSocket background thread."""
    import websockets

    async def _serve():
        log.info(f"WebSocket server listening on ws://{HOST}:{PORT}")
        async with websockets.serve(_ws_handler, HOST, PORT):
            pump = asyncio.create_task(_outbound_pump())
            try:
                while not _shutdown.is_set():
                    await asyncio.sleep(0.1)
            finally:
                pump.cancel()

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


def _maybe_setattr(obj, attr: str, value) -> bool:
    """Best-effort ``setattr`` that swallows AttributeError.

    Isaac Sim's importer ImportConfig classes drop fields between releases
    (e.g. ``default_drive_type`` was removed in 5.1).  We use this so we can
    keep one set-up path that works across 4.x and 5.x without version sniffing.
    Returns True iff the assignment succeeded.
    """
    if not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
        return True
    except (AttributeError, TypeError) as exc:
        log.debug(f"Could not set {type(obj).__name__}.{attr}={value!r}: {exc}")
        return False


def _list_world_children() -> list[str]:
    """Return the immediate child prim paths under ``/World`` for debugging."""
    try:
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            return []
        return [str(p.GetPath()) for p in world_prim.GetChildren()]
    except Exception:
        return []


def _report_load_failure(name: str, reason: str, extra: dict | None = None) -> None:
    """Deliver a load-robot error to the Studio dock chat as a TOOL_RESULT.

    Runs on the main Kit thread.  The Studio extension's listener handles
    TOOL_RESULT by updating the status label, so the user sees the real
    failure instead of a silently-empty viewport.
    """
    payload = {
        "type": "TOOL_RESULT",
        "tool": "generate_robot",
        "summary": f"Isaac viewport: {reason}",
        "success": False,
        "result": {
            "robot_name": name,
            "reason": reason,
            "world_children": _list_world_children(),
            **(extra or {}),
        },
        "ts": time.time(),
    }
    # Prefer an in-process delivery to the extension listener; fall back to
    # broadcasting over WS so the robo-garden backend sees it too.
    if _studio_listener is not None:
        try:
            _studio_listener(payload)
        except Exception as exc:
            log.debug(f"Studio listener could not receive load failure: {exc}")
    try:
        _outbound_queue.put_nowait(json.dumps(payload, default=str))
    except queue.Full:
        pass


def _auto_frame_prim(prim_path: str) -> None:
    """Point the active viewport camera at ``prim_path`` so the robot is
    guaranteed to be in frame after an import.  Silent on any failure —
    framing is a convenience, not a correctness requirement.
    """
    try:
        # Kit 106+ (Isaac Sim 5.x): the "FramePrimsCommand" accepts a list of
        # prim paths and re-aims the active viewport camera at their AABB.
        omni.kit.commands.execute("FramePrimsCommand", prim_to_move=[prim_path])
        return
    except Exception as exc:
        log.debug(f"FramePrimsCommand failed: {exc}")
    try:
        # Fallback path via omni.kit.viewport.utility (available on most 5.x
        # installs).  frame_viewport_prims takes an iterable of prim paths.
        from omni.kit.viewport.utility import (  # type: ignore
            frame_viewport_prims,
            get_active_viewport,
        )
        viewport = get_active_viewport()
        if viewport is not None:
            frame_viewport_prims(viewport, prim_paths=[prim_path])
    except Exception as exc:
        log.debug(f"frame_viewport_prims failed: {exc}")


def _handle_load_robot(msg: dict) -> None:
    name = msg["name"]
    path = msg["path"]
    fmt = msg.get("format", "mjcf")

    # Defensive path normalisation — the backend already sends posix paths
    # (see make_load_robot) but belt-and-braces in case an older client is
    # talking to us.
    path = path.replace("\\", "/")

    # Idempotency: drop a LOAD_ROBOT that exactly matches what's already in the
    # stage.  Claude frequently calls generate_robot twice in quick succession
    # (once to read the catalog entry, once to rename it) and the MJCF importer
    # cannot survive two overlapping imports — its second call returns an
    # empty prim path, polluting the stage.
    last = _last_load.get(name)
    if last and last.get("path") == path and last.get("fmt") == fmt:
        log.info(
            f"LOAD_ROBOT for '{name}' is identical to the previous import "
            f"({path}); skipping re-import."
        )
        return

    log.info(f"Loading robot '{name}' from {path} (format={fmt})")

    try:
        # Remove existing robot with same name.  In 5.x the stage traverse +
        # DeletePrims may leave dangling references if we re-import too fast,
        # so we also flush pending USD updates via world.step(render=False).
        if name in robots:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"/World/{name}")
            if prim:
                omni.kit.commands.execute("DeletePrims", paths=[f"/World/{name}"])
            del robots[name]
            with _loaded_names_lock:
                _loaded_robot_names.discard(name)
            try:
                world.step(render=False)
            except Exception:
                pass

        prim_path = f"/World/{name}"
        # Final prim-path root used by ArticulationView below.  Updated from
        # the importer's return value when available — the MJCF importer
        # sometimes decorates the requested root with a suffix.
        articulation_root = prim_path

        if fmt == "urdf":
            # URDF importer — in 5.x we dispatch the URDFParseAndImportFile
            # Kit command, which is the supported public API and handles
            # parsing + stage insertion in one step.
            try:
                from isaacsim.asset.importer.urdf import _urdf as _urdf_mod  # type: ignore
                urdf_config = _urdf_mod.ImportConfig()
                cmd_name = "URDFParseAndImportFile"
                api_5x = True
            except ImportError:
                from omni.importer.urdf.urdf_importer import ImportConfig as _UrdfConfig  # type: ignore
                urdf_config = _UrdfConfig()
                cmd_name = "URDFParseAndImportFile"
                api_5x = False

            _maybe_setattr(urdf_config, "fix_base", False)
            _maybe_setattr(urdf_config, "import_inertia_tensor", True)
            _maybe_setattr(urdf_config, "make_default_prim", True)
            _maybe_setattr(urdf_config, "create_physics_scene", False)

            ok, _ = omni.kit.commands.execute(
                cmd_name,
                urdf_path=path,
                import_config=urdf_config,
                dest_path="",  # import in-memory
            )
            if not ok:
                raise RuntimeError(f"URDF import command failed (api_5x={api_5x})")
        else:
            if not MJCF_IMPORTER_AVAILABLE:
                reason = "MJCF importer extension not loaded"
                log.error(reason)
                _report_load_failure(name, reason)
                return

            config = _MjcfImportConfig()
            # The set of fields on ImportConfig changed between Isaac Sim 4.x
            # and 5.x (e.g. ``default_drive_type`` was dropped in 5.1).  Set
            # each one only if it actually exists so we stay compatible with
            # both API revisions.
            _maybe_setattr(config, "fix_base", False)
            _maybe_setattr(config, "import_inertia_tensor", True)
            # 1 = position drive in 4.x; 5.x uses per-joint drive types
            # configured by the MJCF parser, so the field simply isn't there.
            _maybe_setattr(config, "default_drive_type", 1)
            # 5.x exposes per-attribute knobs we want sane defaults for.
            _maybe_setattr(config, "make_default_prim", True)
            _maybe_setattr(config, "create_physics_scene", False)
            _maybe_setattr(config, "self_collision", False)
            log.info(
                "MJCF ImportConfig fields: "
                + ", ".join(
                    sorted(a for a in dir(config) if not a.startswith("_"))
                )
            )

            # 5.x: _mjcf interface exposes create_asset_mjcf(path, prim_path,
            # config, dest_path).  An empty dest_path imports in-memory into
            # the currently open stage.  That path triggers a family of
            # "Ill-formed SdfPath" warnings on some Kit 5.1 builds when MuJoCo
            # <material> elements are copied — and the importer then bails out
            # with an empty return path.  Writing to a temp USD first and
            # letting the main stage reference it sidesteps the buggy copy.
            import tempfile
            from pathlib import Path as _P

            tmp_dir = _P(tempfile.gettempdir()) / "robo_garden_mjcf"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            dest_path = (tmp_dir / f"{name}.usd").as_posix()

            created_path = ""
            try:
                created_path = mjcf_importer.create_asset_mjcf(
                    path, prim_path, config, dest_path
                )
            except Exception as exc:
                log.warning(
                    f"MJCF import via temp USD failed ({exc!r}); retrying in-memory"
                )

            if not created_path:
                # Fallback to the original in-memory behavior.  Kept because
                # some Kit builds reject file writes under %TEMP% (sandboxing).
                log.info("MJCF temp-USD import returned empty; retrying in-memory")
                created_path = mjcf_importer.create_asset_mjcf(path, prim_path, config, "")

            if not created_path:
                raise RuntimeError(
                    f"MJCF importer returned empty prim path for {path} "
                    f"(tried both file '{dest_path}' and in-memory import)"
                )
            articulation_root = str(created_path)
            log.info(f"MJCF imported at {articulation_root}")

        # Log what was actually created so we can diagnose future mismatches.
        children = _list_world_children()
        log.info(f"/World children after load: {children}")

        world.reset()

        # Build the ArticulationView pointing at the exact prim the importer
        # created.  Previously we used f"/World/{name}*" which is a regex
        # (so `r*` means zero-or-more r's, not "tree under /World/{name}")
        # and often failed to match when the MJCF root got renamed.
        articulation = ArticulationView(prim_paths_expr=articulation_root)
        world.scene.add(articulation)
        world.reset()

        num_dof = int(getattr(articulation, "num_dof", 0) or 0)
        if num_dof == 0:
            # Robot geometry may have landed in the stage but no articulation
            # prim was picked up — give the user something actionable.
            reason = (
                f"ArticulationView matched 0 DOF at {articulation_root!r}. "
                f"Importer likely placed the robot under a different prim path."
            )
            log.error(reason)
            _report_load_failure(
                name,
                reason,
                extra={"articulation_root": articulation_root},
            )
            # Keep going — the meshes may still render even without an
            # articulation binding, so we still register and frame the prim.

        # Kinematic playback: pause physics so our SIM_FRAME_BATCH frames are
        # the sole source of motion.  The Kit render loop still runs.
        if _kinematic_only:
            try:
                world.pause()
            except Exception:
                pass

        robots[name] = articulation
        with _loaded_names_lock:
            _loaded_robot_names.add(name)
        _last_load[name] = {"path": path, "fmt": fmt}
        log.info(
            f"Robot '{name}' ready at {articulation_root} — {num_dof} DOFs"
        )

        # Point the camera at the robot so it's visible even if the import
        # placed it far from the default camera position.
        _auto_frame_prim(articulation_root)

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.error(f"Failed to load '{name}': {reason}")
        with _loaded_names_lock:
            _loaded_robot_names.discard(name)
        _report_load_failure(name, reason)


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
# Studio UI extension loading
# ---------------------------------------------------------------------------

def _try_load_studio_extension() -> None:
    """Attempt to import and initialize the Studio UI extension.

    The extension is pure-Python and runs inside the same Kit process — it
    uses ``omni.ui`` to contribute a dockable window.  If Kit's UI subsystem
    is unavailable (e.g. headless mode), the extension fails gracefully.
    """
    if _headless:
        log.info("Headless mode — skipping Studio UI extension")
        return
    try:
        # Make isaac_server importable as a package
        import sys as _sys
        from pathlib import Path as _Path
        _here = _Path(__file__).parent
        if str(_here.parent) not in _sys.path:
            _sys.path.insert(0, str(_here.parent))
        from isaac_server.ext_studio.extension import StudioExtension  # type: ignore
        StudioExtension(
            broadcast_fn=broadcast,
            register_listener_fn=register_studio_listener,
        ).on_startup()
        log.info("Studio UI extension loaded")
    except Exception as exc:
        log.warning(f"Studio UI extension not loaded: {exc}")


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
    if _kinematic_only:
        try:
            world.pause()
        except Exception:
            pass

    _try_load_studio_extension()
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
