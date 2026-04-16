"""Design Studio runner: glue between InteractiveSim, IsaacBridge, and Session.

Flow (all inside a single Python process; Isaac Sim is a separate process
reached over WebSocket):

    Isaac Sim UI  <---WebSocket--->  IsaacBridge  <---->  Session (Claude)
                                         |                   |
                                         |                   v
                                         |              tool_handlers
                                         v                   |
                                      InteractiveSim <-------+
                                        (MuJoCo)

On startup:
  1. Spawn InteractiveSim; hook its frame callback into the bridge.
  2. Connect the bridge to Isaac Sim (hard error if unreachable).
  3. Register a bridge.on_message handler that translates UI events into
     Session.chat() / InteractiveSim commands / approve_for_training.
  4. Watch tool_handlers for generate_robot success → load into InteractiveSim.

Runs the main thread idle until user quits (Ctrl-C or Isaac Sim exit).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from robo_garden.claude.session import Session
from robo_garden.claude.tool_handlers import get_sim_result, set_studio_mode
from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR, APPROVED_DIR
from robo_garden.core.interactive_sim import InteractiveSim
from robo_garden.isaac import get_bridge
from robo_garden.isaac.protocol import (
    make_chat_reply,
    make_gate_status,
    make_phase_changed,
    make_robot_meta,
    make_tool_result,
    make_tool_status,
)

log = logging.getLogger(__name__)


class Studio:
    """Orchestrates the Design Studio runtime."""

    def __init__(self, isaac_url: str = "ws://localhost:8765") -> None:
        self._isaac_url = isaac_url
        self._bridge = get_bridge()
        # Claim exclusive ownership of Isaac Sim LOAD_ROBOT dispatch.  Without
        # this, tool_handlers.handle_generate_robot would also send LOAD_ROBOT
        # from the Claude tool-call path, and the resulting double-import
        # races with the delete-prim step and frequently leaves the viewport
        # empty.  See src/robo_garden/claude/tool_handlers.py::_studio_mode.
        set_studio_mode(True)
        self._session = Session(enable_viewer=False)
        self._sim = InteractiveSim(
            frame_callback=self._on_frames,
            meta_callback=self._on_meta,
        )
        self._robot_loaded: str = ""
        self._env_loaded: str = ""
        self._chat_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to Isaac Sim.  Returns False if unreachable."""
        if not self._bridge.connect(self._isaac_url):
            return False
        self._bridge.set_on_message(self._on_inbound)
        self._broadcast_phase()
        self._broadcast_gate()
        self._seed_train_history()
        log.info(f"Studio connected to {self._isaac_url}")
        return True

    def _seed_train_history(self, limit: int = 20) -> None:
        """Ship recent training runs to the Studio UI so the history list
        isn't empty on fresh connect. Best-effort; failures are logged only."""
        try:
            from robo_garden.training.history import load_recent

            runs = load_recent(limit=limit)
            if runs:
                self._bridge.send_train_history(runs)
                log.info(f"Studio: seeded {len(runs)} historical runs to UI")
        except Exception as exc:
            log.warning(f"Studio: could not seed train history: {exc}")

    def close(self) -> None:
        self._sim.close()
        self._bridge.disconnect()
        self._session.close()
        set_studio_mode(False)

    def run_forever(self) -> None:
        """Block the main thread until Isaac Sim disconnects or Ctrl-C."""
        try:
            while self._bridge.connected:
                time.sleep(0.25)
        except KeyboardInterrupt:
            log.info("Studio: Ctrl-C — exiting")

    # ------------------------------------------------------------------
    # Inbound WebSocket messages (from Isaac Sim UI)
    # ------------------------------------------------------------------

    def _on_inbound(self, msg: dict) -> None:
        msg_type = msg.get("type")
        try:
            if msg_type == "CHAT_MESSAGE":
                threading.Thread(
                    target=self._handle_chat,
                    args=(msg.get("text", ""),),
                    daemon=True,
                    name="studio-chat",
                ).start()
            elif msg_type == "JOINT_TARGET":
                self._sim.apply_joint_target(msg["joint"], float(msg["value"]))
            elif msg_type == "APPLY_FORCE":
                self._sim.apply_force(
                    body=msg["body"],
                    force=tuple(msg.get("force", (0, 0, 0))),
                    torque=tuple(msg.get("torque", (0, 0, 0))),
                    duration=float(msg.get("duration", 0.1)),
                )
            elif msg_type == "PAUSE":
                self._sim.pause()
            elif msg_type == "RESUME":
                self._sim.resume()
            elif msg_type == "STEP":
                self._sim.step(int(msg.get("n", 1)))
            elif msg_type == "RESET":
                self._sim.reset()
            elif msg_type == "APPROVE_DESIGN":
                self._handle_approve(msg)
            elif msg_type == "UNAPPROVE_DESIGN":
                self._session.phase = "design"
                self._broadcast_phase()
                self._broadcast_gate()
        except Exception as exc:
            log.warning(f"Studio inbound handler error ({msg_type}): {exc}")

    def _handle_chat(self, text: str) -> None:
        """Run one Claude chat turn; stream status + reply back to UI."""
        if not text.strip():
            return

        def on_status(msg: str) -> None:
            self._bridge.send_raw(make_tool_status("", msg, ""))

        def on_tool_call(name: str, tool_input: dict) -> None:
            detail = (
                tool_input.get("name")
                or tool_input.get("robot_name")
                or tool_input.get("query")
                or ""
            )
            self._bridge.send_raw(make_tool_status(name, "running", str(detail)))

        def on_tool_result(name: str, result: dict) -> None:
            summary = self._summarise_result(name, result)
            self._bridge.send_raw(make_tool_result(
                tool=name,
                summary=summary,
                success=bool(result.get("success", True)),
                result={k: v for k, v in result.items() if self._json_safe(v)},
            ))
            self._on_tool_side_effects(name, result)

        with self._chat_lock:
            try:
                reply = self._session.chat(
                    text,
                    on_status=on_status,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                )
            except Exception as exc:
                log.exception("Studio chat failed")
                self._bridge.send_raw(make_chat_reply(
                    f"[error: {exc}]", session_id=self._session.session_id
                ))
                return

        self._bridge.send_raw(make_chat_reply(reply, session_id=self._session.session_id))
        self._broadcast_gate()

    # ------------------------------------------------------------------
    # Tool side effects (update InteractiveSim, Isaac viewport, UI)
    # ------------------------------------------------------------------

    def _on_tool_side_effects(self, name: str, result: dict) -> None:
        if not result.get("success"):
            return

        if name == "generate_robot":
            robot_name = result.get("robot_name", "")
            mjcf_path = result.get("mjcf_path") or result.get("robot_path", "")
            if mjcf_path and Path(mjcf_path).exists():
                self._robot_loaded = robot_name
                # Mirror in Isaac Sim viewport
                self._bridge.send_robot(
                    robot_name,
                    Path(mjcf_path),
                    fmt=result.get("format", "mjcf"),
                )
                # Drive our authoritative MuJoCo sim
                self._sim.load_robot(mjcf_path=mjcf_path, name=robot_name)
            self._broadcast_gate()

        elif name == "generate_environment":
            self._env_loaded = result.get("env_name", "")
            self._broadcast_gate()

        elif name == "simulate":
            self._broadcast_gate()

        elif name == "approve_for_training":
            if result.get("approved"):
                self._broadcast_phase()
                self._broadcast_gate()

    # ------------------------------------------------------------------
    # InteractiveSim callbacks
    # ------------------------------------------------------------------

    def _on_frames(self, robot_name: str, frames, timesteps) -> None:
        self._bridge.stream_qpos_batch(robot_name, frames, timesteps)

    def _on_meta(self, meta: dict) -> None:
        self._bridge.send_raw(make_robot_meta(
            name=meta.get("name", ""),
            joints=meta.get("joints", []),
            bodies=meta.get("bodies", []),
        ))

    # ------------------------------------------------------------------
    # Approve handler (UI button bypass -> tool dispatch)
    # ------------------------------------------------------------------

    def _handle_approve(self, msg: dict) -> None:
        """User clicked the Promote button; dispatch approve_for_training directly."""
        from robo_garden.claude.tool_handlers import dispatch_tool

        result = dispatch_tool("approve_for_training", {
            "robot_name": msg.get("robot_name", self._robot_loaded),
            "environment_name": msg.get("environment_name", self._env_loaded),
            "notes": msg.get("notes", "Approved via Studio UI button"),
        })
        summary = (
            "approved" if result.get("approved")
            else f"blocked: {len(result.get('unmet_preconditions', []))} precondition(s)"
        )
        self._bridge.send_raw(make_tool_result(
            tool="approve_for_training",
            summary=summary,
            success=bool(result.get("success")),
            result=result,
        ))
        self._broadcast_phase()
        self._broadcast_gate()

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast_phase(self) -> None:
        self._bridge.send_raw(make_phase_changed(
            phase=self._session.phase,
            approved_robot=self._session.approved_robot,
            approved_environment=self._session.approved_environment,
        ))

    def _broadcast_gate(self) -> None:
        robot_name = self._session.approved_robot or self._robot_loaded or self._sim.current_robot
        sim = get_sim_result(robot_name) if robot_name else None

        robot_loaded = bool(robot_name) and self._robot_path_exists(robot_name)
        env_loaded = bool(self._env_loaded) and (ENVIRONMENTS_DIR / f"{self._env_loaded}.xml").exists()
        sim_ran = sim is not None
        sim_stable = bool(sim is not None and not sim.diverged)

        missing: list[str] = []
        if not robot_loaded:
            missing.append("generate_robot first")
        if not env_loaded:
            missing.append("generate_environment first")
        if not sim_ran:
            missing.append("call simulate (passive) first")
        elif not sim_stable:
            missing.append("latest simulate diverged — fix physics")

        can_approve = robot_loaded and env_loaded and sim_ran and sim_stable

        self._bridge.send_raw(make_gate_status(
            robot_loaded=robot_loaded,
            env_loaded=env_loaded,
            sim_ran=sim_ran,
            sim_stable=sim_stable,
            can_approve=can_approve,
            phase=self._session.phase,
            missing=missing,
        ))

    @staticmethod
    def _robot_path_exists(robot_name: str) -> bool:
        if not robot_name:
            return False
        return (
            (ROBOTS_DIR / f"{robot_name}.xml").exists()
            or (ROBOTS_DIR / f"{robot_name}.urdf").exists()
        )

    @staticmethod
    def _summarise_result(name: str, result: dict) -> str:
        if not result.get("success", True):
            return f"failed — {result.get('error') or result.get('errors') or '?'}"
        if name == "generate_robot":
            return f"saved {result.get('robot_name', '?')}"
        if name == "simulate":
            return (
                f"{result.get('duration', '?')}s "
                f"{'stable' if result.get('stable') else 'unstable'}"
            )
        if name == "train":
            br = result.get("best_reward")
            return f"best_reward={br:.3f}" if isinstance(br, (int, float)) else "done"
        if name == "approve_for_training":
            return "approved" if result.get("approved") else "blocked"
        return "done"

    @staticmethod
    def _json_safe(value: Any) -> bool:
        try:
            json.dumps(value, default=str)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Entry points for cli.py
# ---------------------------------------------------------------------------


def run_studio(
    isaac_url: str = "ws://localhost:8765",
    initial_prompt: str | None = None,
) -> None:
    """Launch the Design Studio (expects Isaac Sim already running).

    If *initial_prompt* is provided, it is fed to Claude as the opening turn
    once the bridge is up — convenient for seeding with a saved prompt file
    (e.g. ``workspace/prompts/go2_walker.txt``).  Replies stream into the
    Studio chat panel as usual.
    """
    from rich.console import Console

    console = Console()
    studio = Studio(isaac_url=isaac_url)

    console.print(
        f"[bold cyan]Robo Garden Studio[/] — connecting to Isaac Sim at "
        f"[bold]{isaac_url}[/]"
    )
    if not studio.connect():
        console.print(
            "[bold red]Error:[/] could not reach Isaac Sim at "
            f"{isaac_url}.\nStart it with:\n  ./isaac_server/launch.ps1"
        )
        return

    console.print(
        "[bold green]Connected.[/] Use the Robo Garden Studio panel in Isaac Sim. "
        "Ctrl-C here to quit."
    )

    if initial_prompt:
        preview = initial_prompt.splitlines()[0][:100] if initial_prompt else ""
        console.print(f"[dim]Seeding Studio chat with prompt: {preview}...[/]")
        # Dispatch on a worker thread so the main thread can stay in run_forever
        import threading as _threading
        _threading.Thread(
            target=studio._handle_chat,
            args=(initial_prompt,),
            daemon=True,
            name="studio-seed",
        ).start()

    try:
        studio.run_forever()
    finally:
        studio.close()


def load_approved_manifest(name_or_path: str) -> dict:
    """Resolve an approval manifest by name or filesystem path."""
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        # Try APPROVED_DIR
        candidate = APPROVED_DIR / name_or_path
        if not candidate.exists() and not candidate.suffix:
            candidate = candidate.with_suffix(".json")
        p = candidate
    if not p.exists():
        raise FileNotFoundError(f"Approved manifest not found: {name_or_path}")
    return json.loads(p.read_text(encoding="utf-8"))
