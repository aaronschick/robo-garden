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
from robo_garden.claude.tool_handlers import get_catalog_path, get_sim_result, set_studio_mode
from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR, APPROVED_DIR
from robo_garden.core.interactive_sim import InteractiveSim
from robo_garden.isaac import get_bridge
from robo_garden.isaac.protocol import (
    make_chat_reply,
    make_gamepad_input,
    make_gate_status,
    make_mode_changed,
    make_phase_changed,
    make_policy_list,
    make_policy_playback_status,
    make_robot_meta,
    make_skill_list,
    make_tool_result,
    make_tool_status,
)
from robo_garden.modes.base import AVAILABLE_MODES

log = logging.getLogger(__name__)


def _build_training_context(timesteps: int | None, num_envs: int | None) -> str:
    """Build a system-prompt context block from CLI training overrides.

    Returns an empty string when neither value is provided (no override needed).
    """
    if not timesteps and not num_envs:
        return ""
    lines = [
        "## CLI Training Defaults",
        "The user launched with explicit training scale flags. "
        "Use these values when calling the `train` tool unless the user explicitly requests different ones:",
    ]
    if timesteps:
        lines.append(f"- total_timesteps: {timesteps:,}")
    if num_envs:
        lines.append(f"- num_envs: {num_envs}")
    return "\n".join(lines)


class Studio:
    """Orchestrates the Design Studio runtime."""

    def __init__(
        self,
        isaac_url: str = "ws://localhost:8765",
        training_timesteps: int | None = None,
        training_num_envs: int | None = None,
    ) -> None:
        self._isaac_url = isaac_url
        self._bridge = get_bridge()
        # Claim exclusive ownership of Isaac Sim LOAD_ROBOT dispatch.  Without
        # this, tool_handlers.handle_generate_robot would also send LOAD_ROBOT
        # from the Claude tool-call path, and the resulting double-import
        # races with the delete-prim step and frequently leaves the viewport
        # empty.  See src/robo_garden/claude/tool_handlers.py::_studio_mode.
        set_studio_mode(True)
        extra_context = _build_training_context(training_timesteps, training_num_envs)
        self._session = Session(enable_viewer=False, extra_context=extra_context)
        self._sim = InteractiveSim(
            frame_callback=self._on_frames,
            meta_callback=self._on_meta,
        )
        self._robot_loaded: str = ""
        self._env_loaded: str = ""
        self._chat_lock = threading.Lock()
        self._mode: str = "home"
        self._live_player = None        # LivePolicyPlayer | None
        self._gamepad_runner = None     # GamepadRunner | None
        self._gamepad_mapping = None    # ControlMapping | None (rebuilt on robot load)

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
        self._broadcast_mode()
        self._seed_train_history()
        self._seed_skill_list()
        self._seed_policy_list()
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

    def _seed_skill_list(self) -> None:
        """Broadcast current Skills Library contents on connect. Best-effort."""
        try:
            from robo_garden.skills.registry import list_skills
            skills = [s.to_dict() for s in list_skills()]
            self._bridge.send_raw(make_skill_list(skills))
            log.info(f"Studio: seeded {len(skills)} skills to UI")
        except Exception as exc:
            log.warning(f"Studio: could not seed skill list: {exc}")

    def _seed_policy_list(self) -> None:
        """Broadcast current Policy Composer contents on connect. Best-effort."""
        try:
            from robo_garden.skills.registry import list_policies
            policies = [p.to_dict() for p in list_policies()]
            self._bridge.send_raw(make_policy_list(policies))
            log.info(f"Studio: seeded {len(policies)} policies to UI")
        except Exception as exc:
            log.warning(f"Studio: could not seed policy list: {exc}")

    def _broadcast_policy_list(self, robot_name: str | None = None) -> None:
        try:
            from robo_garden.skills.registry import list_policies
            policies = [p.to_dict() for p in list_policies(robot_name=robot_name)]
            self._bridge.send_raw(make_policy_list(policies, robot_name=robot_name))
        except Exception as exc:
            log.warning(f"Studio: could not broadcast policy list: {exc}")

    def _broadcast_skill_list(self, robot_name: str | None = None) -> None:
        """Broadcast updated skill list (e.g. after promote). Best-effort."""
        try:
            from robo_garden.skills.registry import list_skills
            skills = [s.to_dict() for s in list_skills(robot_name=robot_name)]
            self._bridge.send_raw(make_skill_list(skills, robot_name=robot_name))
        except Exception as exc:
            log.warning(f"Studio: could not broadcast skill list: {exc}")

    # ------------------------------------------------------------------
    # Gamepad
    # ------------------------------------------------------------------

    def _start_gamepad(self) -> None:
        """Start GamepadRunner for Simulate mode.  Best-effort — no error if pygame absent."""
        if self._gamepad_runner is not None and self._gamepad_runner.running:
            return
        try:
            from robo_garden.input.runner import GamepadRunner

            def _on_state(state) -> None:
                self._on_gamepad_state(state)

            def _on_echo(state_dict: dict) -> None:
                try:
                    self._bridge.send_raw(make_gamepad_input(
                        axes=state_dict.get("axes", []),
                        buttons=state_dict.get("buttons", {}),
                        connected=state_dict.get("connected", False),
                    ))
                except Exception:
                    pass

            self._gamepad_runner = GamepadRunner(on_state=_on_state, on_echo=_on_echo)
            self._gamepad_runner.start()
        except Exception as exc:
            log.info(f"Studio: gamepad unavailable — {exc}")

    def _stop_gamepad(self) -> None:
        if self._gamepad_runner is not None:
            self._gamepad_runner.stop()
            self._gamepad_runner = None
            self._gamepad_mapping = None

    def _on_gamepad_state(self, state) -> None:
        """Direct teleop: apply gamepad axes to joints (only when no policy is playing)."""
        if self._live_player is not None:
            return   # policy drives the robot; gamepad is echo-only in Phase 5
        if not state.connected:
            return

        joints = self._sim.joints
        if not joints:
            return

        # Build (or reuse) the control mapping for the current robot
        if self._gamepad_mapping is None:
            from robo_garden.input.mapping import ControlMapping
            loaded = ControlMapping.load_for_robot(self._robot_loaded)
            if loaded is not None:
                self._gamepad_mapping = loaded
            else:
                self._gamepad_mapping = ControlMapping.from_joints(joints)

        joints_by_name = {j.name: j for j in joints}
        ctrl_targets = self._gamepad_mapping.apply(state, joints_by_name)
        for joint_name, value in ctrl_targets.items():
            self._sim.apply_joint_target(joint_name, value)

        # Check for reset / pause button actions
        for btn_idx, action in self._gamepad_mapping.button_actions.items():
            if state.buttons.get(f"button_{btn_idx}"):
                if action == "reset":
                    self._sim.reset()
                elif action == "pause":
                    self._sim.pause()

    def close(self) -> None:
        self._stop_gamepad()
        if self._live_player is not None:
            self._live_player.stop()
            self._live_player = None
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
            elif msg_type == "REVIEW_RUN":
                threading.Thread(
                    target=self._handle_review_run,
                    args=(msg.get("run_id", "latest"),),
                    daemon=True,
                    name="studio-review",
                ).start()
            elif msg_type == "MODE_REQUEST":
                self.set_mode(msg.get("mode", "home"), msg.get("context", {}))
            elif msg_type == "SKILL_PROMOTE":
                threading.Thread(
                    target=self._handle_skill_promote,
                    args=(msg,),
                    daemon=True,
                    name="studio-promote",
                ).start()
            elif msg_type == "POLICY_PLAYBACK_START":
                threading.Thread(
                    target=self._handle_playback_start,
                    args=(msg,),
                    daemon=True,
                    name="studio-playback",
                ).start()
            elif msg_type == "POLICY_PLAYBACK_STOP":
                threading.Thread(
                    target=self._handle_playback_stop,
                    daemon=True,
                    name="studio-playback-stop",
                ).start()
            elif msg_type == "POLICY_SAVE":
                threading.Thread(
                    target=self._handle_policy_save,
                    args=(msg,),
                    daemon=True,
                    name="studio-policy-save",
                ).start()
            elif msg_type == "POLICY_DELETE":
                threading.Thread(
                    target=self._handle_policy_delete,
                    args=(msg,),
                    daemon=True,
                    name="studio-policy-delete",
                ).start()
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
                self._gamepad_mapping = None   # force rebuild for new robot
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

        elif name == "promote_skill":
            self._broadcast_skill_list()

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

    def _handle_review_run(self, run_id: str) -> None:
        """User clicked Replay on a history item; dispatch review_run tool directly."""
        from robo_garden.claude.tool_handlers import dispatch_tool
        from robo_garden.isaac.protocol import make_tool_result, make_tool_status

        self._bridge.send_raw(make_tool_status("review_run", "running", run_id))
        result = dispatch_tool("review_run", {
            "run_id": run_id,
            "num_frames": 150,
            "render_video": False,
        })
        summary = (
            f"Replaying {result.get('num_frames', 0)} frames of {result.get('robot_name', '?')}"
            if result.get("success")
            else f"replay failed: {result.get('error', '?')}"
        )
        self._bridge.send_raw(make_tool_result(
            tool="review_run",
            summary=summary,
            success=bool(result.get("success")),
            result=result,
        ))

    def _handle_skill_promote(self, msg: dict) -> None:
        """User clicked ★ Skill in the run history; promote the run to the Skills Library."""
        from robo_garden.claude.tool_handlers import dispatch_tool

        run_id = msg.get("run_id", "")
        skill_id = msg.get("skill_id", "").strip()
        display_name = msg.get("display_name", "").strip()
        task_description = msg.get("task_description", "")

        if not skill_id or not display_name:
            log.warning("Studio: SKILL_PROMOTE missing skill_id or display_name")
            return

        self._bridge.send_raw(make_tool_status("promote_skill", "running", display_name))
        result = dispatch_tool("promote_skill", {
            "run_id": run_id,
            "skill_id": skill_id,
            "display_name": display_name,
            "task_description": task_description,
        })
        self._bridge.send_raw(make_tool_result(
            tool="promote_skill",
            summary=(
                f"'{display_name}' saved as skill {result.get('variant_id', '')}"
                if result.get("success")
                else f"promote failed: {result.get('error', '?')}"
            ),
            success=bool(result.get("success")),
            result=result,
        ))
        if result.get("success"):
            self._broadcast_skill_list()

    def _handle_playback_start(self, msg: dict) -> None:
        """Load policy from the skills library and start LivePolicyPlayer.

        Supports two paths:
        - Single skill: {robot_name, skill_id, variant_id}
        - Composed policy: {robot_name, policy_name}  → PolicyRunner
        """
        from robo_garden.core.live_player import LivePolicyPlayer
        from robo_garden.skills.inference import load_policy_fn
        from robo_garden.skills.registry import get_variant
        from robo_garden.training.mujoco_engine import _merge_mjcf

        robot_name = msg.get("robot_name", "")
        policy_name = msg.get("policy_name", "")

        # Stop any running player first
        if self._live_player is not None:
            self._live_player.stop()
            self._live_player = None

        # Resolve robot MJCF
        robot_xml = self._resolve_robot_xml(robot_name)
        if not robot_xml:
            self._bridge.send_raw(make_policy_playback_status(
                "stopped", robot_name=robot_name,
                error=f"robot '{robot_name}' MJCF not found",
            ))
            return

        if policy_name:
            # -- Composed policy path --
            from robo_garden.skills.policy import PolicyRunner, PolicyEntry
            from robo_garden.skills.registry import get_policy

            spec = get_policy(robot_name, policy_name)
            if spec is None:
                self._bridge.send_raw(make_policy_playback_status(
                    "stopped", robot_name=robot_name,
                    error=f"policy '{policy_name}' not found",
                ))
                return

            entries: list[PolicyEntry] = []
            env_name = ""
            for ref in spec.skills:
                variant = get_variant(robot_name, ref.skill_id, ref.variant_id)
                if variant is None:
                    log.warning(f"Studio: playback — variant {ref.variant_id!r} not found, skipping")
                    continue
                pfn = load_policy_fn(variant)
                entries.append(PolicyEntry(
                    skill_id=ref.skill_id,
                    variant_id=ref.variant_id,
                    trigger=ref.trigger,
                    policy_fn=pfn,
                ))
                if not env_name:
                    env_name = variant.environment_name

            if not entries:
                self._bridge.send_raw(make_policy_playback_status(
                    "stopped", robot_name=robot_name,
                    error=f"policy '{policy_name}' has no loadable variants",
                ))
                return

            policy_runner = PolicyRunner(entries)
            gamepad_runner = self._gamepad_runner

            def gamepad_fn():
                return gamepad_runner.latest() if gamepad_runner is not None else None

            policy_fn = policy_runner.as_policy_fn(gamepad_fn)
            display_id = policy_name

        else:
            # -- Single skill path --
            skill_id = msg.get("skill_id", "")
            variant_id = msg.get("variant_id", "")

            variant = get_variant(robot_name, skill_id, variant_id)
            if variant is None:
                self._bridge.send_raw(make_policy_playback_status(
                    "stopped", skill_id=skill_id, robot_name=robot_name,
                    error=f"variant {variant_id!r} not found",
                ))
                return

            env_name = variant.environment_name
            policy_fn = load_policy_fn(variant)
            display_id = skill_id

        # Resolve environment MJCF (best-effort)
        env_xml = ""
        if env_name:
            env_path = ENVIRONMENTS_DIR / f"{env_name}.xml"
            if env_path.exists():
                env_xml = env_path.read_text(encoding="utf-8")

        merged_xml = _merge_mjcf(robot_xml, env_xml)

        # Load robot into Isaac viewport if not already showing this robot
        if robot_name != self._robot_loaded:
            robot_path = ROBOTS_DIR / f"{robot_name}.xml"
            if not robot_path.exists():
                robot_path = ROBOTS_DIR / f"{robot_name}.urdf"
            cat = get_catalog_path(robot_name)
            if cat is not None:
                robot_path = cat
            if robot_path.exists():
                self._bridge.send_robot(robot_name, robot_path, fmt="mjcf")
                self._robot_loaded = robot_name

        self._live_player = LivePolicyPlayer(
            policy_fn=policy_fn,
            mjcf_xml=merged_xml,
            robot_name=robot_name,
            frame_callback=self._bridge.stream_qpos_batch,
        )
        self._live_player.start()

        self._bridge.send_raw(make_policy_playback_status(
            "playing",
            skill_id=display_id,
            robot_name=robot_name,
            variant_id=msg.get("variant_id", ""),
        ))
        log.info(f"Studio: playback started for {robot_name}/{display_id}")

    def _handle_playback_stop(self) -> None:
        if self._live_player is not None:
            robot_name = self._live_player._robot_name
            self._live_player.stop()
            self._live_player = None
            self._bridge.send_raw(make_policy_playback_status("stopped", robot_name=robot_name))
            log.info("Studio: playback stopped")

    def _handle_policy_save(self, msg: dict) -> None:
        """User saved a composed policy from the Policy Composer UI."""
        from robo_garden.skills import PolicySpec, PolicySkillRef
        from robo_garden.skills.registry import save_policy

        robot_name = msg.get("robot_name", "")
        policy_name = msg.get("policy_name", "").strip()
        if not robot_name or not policy_name:
            log.warning("Studio: POLICY_SAVE missing robot_name or policy_name")
            return

        skills = [
            PolicySkillRef(
                skill_id=s.get("skill_id", ""),
                variant_id=s.get("variant_id", ""),
                trigger=s.get("trigger", ""),
            )
            for s in msg.get("skills", [])
        ]
        spec = PolicySpec(
            policy_name=policy_name,
            robot_name=robot_name,
            composition=msg.get("composition", "switcher"),
            skills=skills,
        )
        try:
            path = save_policy(spec)
            log.info(f"Studio: saved policy {robot_name}/{policy_name} → {path}")
            self._bridge.send_raw(make_tool_result(
                tool="policy_save",
                summary=f"Policy '{policy_name}' saved ({len(skills)} skills)",
                success=True,
                result={"policy_name": policy_name, "robot_name": robot_name},
            ))
            self._broadcast_policy_list(robot_name)
        except Exception as exc:
            log.warning(f"Studio: could not save policy: {exc}")
            self._bridge.send_raw(make_tool_result(
                tool="policy_save",
                summary=f"Save failed: {exc}",
                success=False,
                result={"error": str(exc)},
            ))

    def _handle_policy_delete(self, msg: dict) -> None:
        from robo_garden.skills.registry import delete_policy

        robot_name = msg.get("robot_name", "")
        policy_name = msg.get("policy_name", "")
        ok = delete_policy(robot_name, policy_name)
        log.info(f"Studio: delete policy {robot_name}/{policy_name} → {ok}")
        self._broadcast_policy_list(robot_name)

    def _resolve_robot_xml(self, robot_name: str) -> str:
        """Return MJCF XML string for *robot_name*, or '' if not found."""
        cat = get_catalog_path(robot_name)
        if cat is not None:
            try:
                from robo_garden.claude.tool_handlers import _absolutize_asset_paths
                xml = cat.read_text(encoding="utf-8")
                return _absolutize_asset_paths(xml, cat.parent)
            except Exception as exc:
                log.warning(f"Studio: could not read catalog robot {robot_name}: {exc}")
                return ""
        for ext in (".xml", ".urdf"):
            p = ROBOTS_DIR / f"{robot_name}{ext}"
            if p.exists():
                return p.read_text(encoding="utf-8")
        return ""

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

    def set_mode(self, mode: str, context: dict | None = None) -> None:
        """Switch activity mode and broadcast MODE_CHANGED to all UI clients."""
        if mode not in AVAILABLE_MODES:
            log.warning(f"Studio: unknown mode {mode!r}, ignoring")
            return
        prev = self._mode
        self._mode = mode
        log.info(f"Studio: mode {prev!r} → {mode!r}")

        # Start gamepad when entering Simulate or Compose, stop when leaving both
        _gamepad_modes = {"simulate", "compose"}
        if mode in _gamepad_modes and prev not in _gamepad_modes:
            self._start_gamepad()
        elif mode not in _gamepad_modes and prev in _gamepad_modes:
            self._stop_gamepad()

        self._broadcast_mode(context or {})

    def _broadcast_mode(self, context: dict | None = None) -> None:
        self._bridge.send_raw(make_mode_changed(
            mode=self._mode,
            available_modes=AVAILABLE_MODES,
            context=context or {},
        ))

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
    training_timesteps: int | None = None,
    training_num_envs: int | None = None,
) -> None:
    """Launch the Design Studio (expects Isaac Sim already running).

    If *initial_prompt* is provided, it is fed to Claude as the opening turn
    once the bridge is up — convenient for seeding with a saved prompt file
    (e.g. ``workspace/prompts/go2_walker.txt``).  Replies stream into the
    Studio chat panel as usual.

    *training_timesteps* and *training_num_envs* are injected into Claude's
    system prompt so it uses these values when calling the ``train`` tool,
    overriding the schema defaults (1M timesteps, 128 envs).
    """
    from rich.console import Console

    console = Console()
    studio = Studio(
        isaac_url=isaac_url,
        training_timesteps=training_timesteps,
        training_num_envs=training_num_envs,
    )

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
