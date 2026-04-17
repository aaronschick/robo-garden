"""Studio UI extension for the Isaac Sim server.

Runs inside the Isaac Sim Kit process.  Builds a dockable ``omni.ui`` window
with a mode navigation bar at the top followed by mode-specific panels:

    +---------------------------------------------------+
    |  [Home] [Design] [Sim] [Train] [Skills] [...] ... |  ← mode bar
    +---------------------------------------------------+
    |  Phase: design                                    |  ← always visible
    +---------------------------------------------------+
    |  Chat with Claude                                 |  ← always visible
    |  (history)                                        |
    |  [input field]                [Send]              |
    +---------------------------------------------------+
    |  <active mode panels>                             |
    |   Design:  Robot Controls / Toolbar / Approval / Training
    |   Train:   stub (Phase 7)
    |   Simulate: stub (Phase 4)
    |   Skills:  stub (Phase 3)
    |   Compose: stub (Phase 6)
    |   Deploy:  stub (Phase 8)
    |   Home:    welcome message                        |
    +---------------------------------------------------+

User actions are serialised into studio protocol messages and broadcast to
every connected WebSocket client (the robo-garden backend) via the
``broadcast_fn`` injected at construction.  Backend replies arrive through
``register_listener_fn`` as dict messages.

All UI callbacks run on the Kit main thread; broadcast is thread-safe.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger("ext_studio")

BroadcastFn = Callable[[dict], None]
RegisterListenerFn = Callable[[Callable[[dict], None]], None]


# ---------------------------------------------------------------------------
# Studio protocol — UI-side message builders.
#
# The extension runs inside Isaac Sim's Python venv which does NOT have the
# `robo_garden` package installed, so we cannot import from
# robo_garden.isaac.protocol.  These tiny helpers build the exact same dicts
# and must stay in sync with src/robo_garden/isaac/protocol.py.
# ---------------------------------------------------------------------------

def _make_chat_message(text: str) -> dict:
    return {"type": "CHAT_MESSAGE", "text": text, "ts": time.time()}


def _make_joint_target(joint: str, value: float) -> dict:
    return {
        "type": "JOINT_TARGET",
        "joint": joint,
        "value": float(value),
        "ts": time.time(),
    }


def _make_apply_force(
    body: str,
    force=(0.0, 0.0, 0.0),
    torque=(0.0, 0.0, 0.0),
    duration: float = 0.1,
) -> dict:
    return {
        "type": "APPLY_FORCE",
        "body": body,
        "force": list(force),
        "torque": list(torque),
        "duration": float(duration),
        "ts": time.time(),
    }


def _make_pause() -> dict:
    return {"type": "PAUSE", "ts": time.time()}


def _make_resume() -> dict:
    return {"type": "RESUME", "ts": time.time()}


def _make_step(n: int = 1) -> dict:
    return {"type": "STEP", "n": int(n), "ts": time.time()}


def _make_reset() -> dict:
    return {"type": "RESET", "ts": time.time()}


def _make_approve_design(robot_name: str, environment_name: str, notes: str = "") -> dict:
    return {
        "type": "APPROVE_DESIGN",
        "robot_name": robot_name,
        "environment_name": environment_name,
        "notes": notes,
        "ts": time.time(),
    }


def _make_unapprove_design() -> dict:
    return {"type": "UNAPPROVE_DESIGN", "ts": time.time()}


def _make_mode_request(mode: str) -> dict:
    return {"type": "MODE_REQUEST", "mode": mode, "context": {}, "ts": time.time()}


def _make_skill_promote(run_id: str, skill_id: str, display_name: str, task_description: str = "") -> dict:
    return {
        "type": "SKILL_PROMOTE",
        "run_id": run_id,
        "skill_id": skill_id,
        "display_name": display_name,
        "task_description": task_description,
        "ts": time.time(),
    }


def _make_policy_save(
    robot_name: str,
    policy_name: str,
    skills: list[dict],
    composition: str = "switcher",
) -> dict:
    return {
        "type": "POLICY_SAVE",
        "robot_name": robot_name,
        "policy_name": policy_name,
        "composition": composition,
        "skills": skills,
        "ts": time.time(),
    }


def _make_policy_delete(robot_name: str, policy_name: str) -> dict:
    return {"type": "POLICY_DELETE", "robot_name": robot_name, "policy_name": policy_name, "ts": time.time()}


def _make_policy_playback_start_composed(robot_name: str, policy_name: str) -> dict:
    return {"type": "POLICY_PLAYBACK_START", "robot_name": robot_name, "policy_name": policy_name, "ts": time.time()}


# ---------------------------------------------------------------------------
# Mode metadata: order and display labels for the nav bar.
# ---------------------------------------------------------------------------

_MODES: list[tuple[str, str]] = [
    ("home",    "Home"),
    ("design",  "Design"),
    ("simulate","Sim"),
    ("train",   "Train"),
    ("skills",  "Skills"),
    ("compose", "Compose"),
    ("deploy",  "Deploy"),
]


class StudioExtension:
    """Owns the Studio dockable window and routes events to the WS server."""

    def __init__(
        self,
        broadcast_fn: BroadcastFn,
        register_listener_fn: RegisterListenerFn,
    ) -> None:
        self._broadcast = broadcast_fn
        self._register = register_listener_fn

        self._window = None
        self._root_scroll = None
        self._chat_log = None
        self._chat_input = None
        self._status_label = None
        self._joint_container = None
        self._bodies_combo = None
        self._gate_label = None
        self._promote_button = None
        self._phase_label = None

        self._joints: list[dict] = []
        self._bodies: list[str] = []
        self._apply_force_mode: bool = False
        self._paused: bool = False
        self._current_robot: str = ""
        self._approved_env: str = ""
        self._last_gate: dict = {}

        # Training progress panel state
        self._train_header_label = None
        self._train_status_label = None
        self._train_backend_label = None
        self._train_progress_bar = None
        self._train_progress_text = None
        self._train_metric_labels: dict = {}
        self._train_sparkline_label = None      # fallback if ui.Plot unavailable
        self._train_plot_container = None       # holds ui.Plot or sparkline
        self._train_plot_mode: str = "unknown"  # "plot" | "sparkline"
        self._train_history_container = None
        self._train_rewards: list[float] = []
        self._train_reward_curve: list[tuple[int, float]] = []
        self._train_run_id: str = ""
        self._train_runs_history: list[dict] = []
        self._train_best: float = float("-inf")
        # Per-component reward breakdown
        self._train_component_container = None
        self._train_last_components: dict = {}
        # Mid-training rollout preview buttons
        self._train_preview_container = None
        self._train_previews: list[dict] = []   # [{timestep, num_frames}, ...]

        # Mode navigation state
        self._current_mode: str = "home"
        self._mode_buttons: dict[str, object] = {}   # mode key → Button widget
        self._mode_panels: dict[str, object] = {}    # mode key → VStack container

        # Skills Library state
        self._skills_list_container = None
        self._skill_entries: list[dict] = []

        # Simulate mode playback state
        self._sim_skill_combo = None
        self._sim_play_btn = None
        self._sim_pause_btn = None
        self._sim_stop_btn = None
        self._sim_status_label = None
        self._sim_selected_idx: int = 0
        self._sim_playing: bool = False

        # Gamepad panel state
        self._gamepad_conn_label = None
        self._gamepad_axes_label = None
        self._gamepad_btns_label = None

        # Policy Composer state
        self._compose_rows: list[dict] = []       # {"skill_idx": int, "trigger": str}
        self._compose_rows_container = None
        self._compose_name_field = None
        self._compose_status_label = None
        self._compose_policy_list_container = None
        self._compose_policies: list[dict] = []   # PolicySpec.to_dict() entries
        self._compose_playing: bool = False
        self._compose_add_skill_combo = None
        self._compose_add_trigger_combo = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_startup(self) -> None:
        """Build the window and subscribe to inbound messages."""
        import omni.ui as ui

        self._window = ui.Window(
            "Robo Garden Studio",
            width=520,
            height=920,
            dockPreference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            self._root_scroll = ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with self._root_scroll:
                with ui.VStack(spacing=6):
                    # ── Always-visible chrome ──────────────────────────────
                    self._build_mode_bar(ui)
                    ui.Separator()
                    self._build_phase_banner(ui)
                    ui.Separator()
                    self._build_chat_panel(ui)
                    ui.Separator()

                    # ── Home panel ─────────────────────────────────────────
                    _home = ui.VStack(spacing=6)
                    with _home:
                        self._build_home_panel(ui)
                    self._mode_panels["home"] = _home

                    # ── Design panels (all current Studio panels) ──────────
                    _design = ui.VStack(spacing=6)
                    with _design:
                        self._build_robot_panel(ui)
                        ui.Separator()
                        self._build_interaction_toolbar(ui)
                        ui.Separator()
                        self._build_approval_panel(ui)
                        ui.Separator()
                        self._build_training_panel(ui)
                    self._mode_panels["design"] = _design

                    # ── Train panel (stub — Phase 7) ───────────────────────
                    _train = ui.VStack(spacing=6)
                    with _train:
                        ui.Label("Training", style={"font_size": 16})
                        ui.Label(
                            "Start a training run from Design mode chat.\n"
                            "Full training dashboard coming in Phase 7.",
                            style={"color": 0xFF888888},
                            word_wrap=True,
                        )
                    self._mode_panels["train"] = _train

                    # ── Simulate panel ─────────────────────────────────────
                    _simulate = ui.VStack(spacing=6)
                    with _simulate:
                        ui.Label("Simulate", style={"font_size": 16})
                        ui.Label(
                            "Select a skill and click Play to run live policy rollout.",
                            style={"color": 0xFF888888},
                            word_wrap=True,
                        )
                        ui.Separator()
                        with ui.HStack(height=24, spacing=6):
                            ui.Label("Skill:", width=40)
                            self._sim_skill_combo = ui.ComboBox(
                                0, "(no skills — promote a run first)"
                            )
                            self._sim_skill_combo.model.add_item_changed_fn(
                                self._on_sim_skill_changed
                            )
                        with ui.HStack(height=26, spacing=4):
                            self._sim_play_btn = ui.Button(
                                "▶ Play", width=70,
                                clicked_fn=self._on_sim_play,
                                enabled=False,
                            )
                            self._sim_pause_btn = ui.Button(
                                "⏸ Pause", width=70,
                                clicked_fn=self._on_sim_pause,
                                enabled=False,
                            )
                            self._sim_stop_btn = ui.Button(
                                "■ Stop", width=70,
                                clicked_fn=self._on_sim_stop,
                                enabled=False,
                            )
                        self._sim_status_label = ui.Label(
                            "",
                            style={"color": 0xFF888888},
                            word_wrap=True,
                        )
                        ui.Separator()
                        ui.Label("Gamepad", style={"font_size": 14})
                        self._gamepad_conn_label = ui.Label(
                            "○ Not connected",
                            style={"color": 0xFF888888},
                        )
                        self._gamepad_axes_label = ui.Label(
                            "Axes: —",
                            style={"font_size": 11, "color": 0xFFCCCCCC},
                            word_wrap=False,
                        )
                        self._gamepad_btns_label = ui.Label(
                            "Btns: —",
                            style={"font_size": 11, "color": 0xFFCCCCCC},
                            word_wrap=False,
                        )
                    self._mode_panels["simulate"] = _simulate

                    # ── Skills panel ───────────────────────────────────────
                    _skills = ui.VStack(spacing=6)
                    with _skills:
                        ui.Label("Skills Library", style={"font_size": 16})
                        ui.Label(
                            "Trained behaviors promoted from completed runs.",
                            style={"color": 0xFF888888},
                            word_wrap=True,
                        )
                        with ui.ScrollingFrame(
                            height=200,
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                        ):
                            self._skills_list_container = ui.VStack(spacing=2)
                            with self._skills_list_container:
                                ui.Label("(no skills yet — promote a training run)", style={"color": 0xFF888888})
                    self._mode_panels["skills"] = _skills

                    # ── Compose panel ──────────────────────────────────────
                    _compose = ui.VStack(spacing=6)
                    with _compose:
                        self._build_compose_panel(ui)
                    self._mode_panels["compose"] = _compose

                    # ── Deploy panel (stub — Phase 8) ──────────────────────
                    _deploy = ui.VStack(spacing=6)
                    with _deploy:
                        ui.Label("Deploy / Export", style={"font_size": 16})
                        ui.Label(
                            "Export checkpoints for real-world deployment.\n"
                            "Coming in Phase 8.",
                            style={"color": 0xFF888888},
                            word_wrap=True,
                        )
                    self._mode_panels["deploy"] = _deploy

        # Start in home mode (backend will send MODE_CHANGED on connect)
        self._set_mode_panels("home")
        self._register(self._on_backend_message)
        log.info("StudioExtension window ready")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_mode_bar(self, ui) -> None:
        """Navigation bar: one button per activity mode."""
        with ui.HStack(height=28, spacing=2):
            for key, label in _MODES:
                btn = ui.Button(
                    label,
                    height=26,
                    clicked_fn=lambda m=key: self._on_mode_button(m),
                    style={"font_size": 11},
                )
                self._mode_buttons[key] = btn

    def _build_home_panel(self, ui) -> None:
        """Welcome screen shown on startup before the user selects a mode."""
        ui.Spacer(height=16)
        ui.Label(
            "Robo Garden",
            style={"font_size": 22, "color": 0xFF4ED4FF},
            alignment=ui.Alignment.CENTER,
            width=ui.Fraction(1),
        )
        ui.Spacer(height=6)
        ui.Label(
            "Claude-powered robot design and training studio.\n"
            "Select a mode above to get started.",
            style={"color": 0xFF888888, "font_size": 13},
            word_wrap=True,
            alignment=ui.Alignment.CENTER,
            width=ui.Fraction(1),
        )
        ui.Spacer(height=12)
        ui.Label(
            "Design  —  create robots with Claude\n"
            "Simulate  —  drive robots with a controller\n"
            "Train  —  run and monitor training jobs\n"
            "Skills  —  browse trained behaviors\n"
            "Compose  —  combine skills into policy versions",
            style={"color": 0xFFAAAAAA, "font_size": 12},
            word_wrap=True,
        )

    def _build_phase_banner(self, ui) -> None:
        with ui.HStack(height=24):
            ui.Label("Phase:", width=60)
            self._phase_label = ui.Label("design", style={"color": 0xFF4ED4FF})

    def _build_chat_panel(self, ui) -> None:
        ui.Label("Chat with Claude", style={"font_size": 16})
        with ui.ScrollingFrame(
            height=220,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self._chat_log = ui.Label(
                "Ready. Describe a robot to get started.\n",
                word_wrap=True,
                alignment=ui.Alignment.LEFT_TOP,
                width=ui.Fraction(1),
            )
        with ui.HStack(height=26, spacing=4):
            self._chat_input = ui.StringField(width=ui.Fraction(1))
            ui.Button("Send", width=60, clicked_fn=self._on_send_chat)
        self._status_label = ui.Label(
            "",
            style={"color": 0xFF888888},
            word_wrap=True,
            width=ui.Fraction(1),
        )

    def _build_robot_panel(self, ui) -> None:
        ui.Label("Robot Controls", style={"font_size": 16})
        self._joint_container = ui.VStack(spacing=2)
        with self._joint_container:
            ui.Label("(no robot loaded)", style={"color": 0xFF888888})

    def _build_interaction_toolbar(self, ui) -> None:
        ui.Label("Interaction", style={"font_size": 16})
        with ui.HStack(height=26, spacing=4):
            ui.Button("Pause", clicked_fn=self._on_pause)
            ui.Button("Resume", clicked_fn=self._on_resume)
            ui.Button("Step", clicked_fn=self._on_step)
            ui.Button("Reset", clicked_fn=self._on_reset)
        with ui.HStack(height=26, spacing=4):
            ui.Label("Apply force to body:", width=130)
            self._bodies_combo = ui.ComboBox(0, "(select body)")
            ui.Button("Push +X", width=70, clicked_fn=lambda: self._on_apply_force(10.0, 0, 0))
            ui.Button("Push +Z", width=70, clicked_fn=lambda: self._on_apply_force(0, 0, 10.0))

    def _build_approval_panel(self, ui) -> None:
        ui.Label("Promote to Training", style={"font_size": 16})
        self._gate_label = ui.Label(
            "Gate: waiting for robot / environment / passive sim",
            word_wrap=True,
            style={"color": 0xFFAAAAAA},
        )
        with ui.HStack(height=26, spacing=4):
            self._promote_button = ui.Button(
                "Promote to Training",
                enabled=False,
                clicked_fn=self._on_promote,
            )
            ui.Button("Back to Design", clicked_fn=self._on_unapprove)

    def _build_training_panel(self, ui) -> None:
        """Training Progress: live status + reward sparkline + run history."""
        ui.Label("Training Progress", style={"font_size": 16})
        self._train_header_label = ui.Label(
            "(no active run)",
            style={"color": 0xFF888888},
            word_wrap=True,
        )
        self._train_backend_label = ui.Label(
            "",
            style={"font_size": 13, "color": 0xFFAAAAAA},
            word_wrap=False,
        )
        self._train_status_label = ui.Label(
            "", style={"color": 0xFFAAAAAA}, word_wrap=True
        )

        with ui.HStack(height=18, spacing=6):
            self._train_progress_bar = ui.ProgressBar(width=ui.Fraction(1))
            self._train_progress_text = ui.Label("0%", width=50)
        self._train_progress_bar.model.set_value(0.0)

        with ui.HStack(height=20, spacing=12):
            self._train_metric_labels["mean"] = ui.Label("mean: —", width=120)
            self._train_metric_labels["best"] = ui.Label("best: —", width=120)
            self._train_metric_labels["tps"] = ui.Label("t/s: —", width=100)
            self._train_metric_labels["elapsed"] = ui.Label("elapsed: —", width=120)

        # Reward curve: try omni.ui.Plot, fall back to text sparkline
        self._train_plot_container = ui.VStack(height=72)
        with self._train_plot_container:
            self._train_sparkline_label = ui.Label(
                "Reward curve will appear here once a run starts.",
                style={"font_size": 13, "color": 0xFFCCDDFF},
                word_wrap=False,
            )
        self._train_plot_mode = "sparkline"   # upgraded to "plot" on first update

        # Per-component reward breakdown (populated by TRAIN_REWARD_BREAKDOWN)
        ui.Label("Reward breakdown", style={"font_size": 13, "color": 0xFFAAAAAA})
        self._train_component_container = ui.VStack(spacing=2)
        with self._train_component_container:
            ui.Label("(no component data yet)", style={"color": 0xFF666666, "font_size": 11})

        # Mid-training rollout previews (populated by TRAIN_ROLLOUT_PREVIEW)
        ui.Label("In-run previews", style={"font_size": 13, "color": 0xFFAAAAAA})
        with ui.ScrollingFrame(
            height=30,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        ):
            self._train_preview_container = ui.HStack(spacing=4)
            with self._train_preview_container:
                ui.Label("(none yet)", style={"color": 0xFF666666, "font_size": 11})

        ui.Label("Run history", style={"font_size": 14})
        with ui.ScrollingFrame(
            height=140,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self._train_history_container = ui.VStack(spacing=2)
            with self._train_history_container:
                ui.Label("(no previous runs)", style={"color": 0xFF888888})

    def _build_compose_panel(self, ui) -> None:
        """Policy Composer: pick skills + triggers → save a named policy."""
        ui.Label("Policy Composer", style={"font_size": 16})
        ui.Label(
            "Combine trained skills into a named policy version.",
            style={"color": 0xFF888888},
            word_wrap=True,
        )
        ui.Separator()

        # -- Add skill row --
        ui.Label("Add skill to composition:", style={"font_size": 13})
        with ui.HStack(height=24, spacing=4):
            ui.Label("Skill:", width=36)
            self._compose_add_skill_combo = ui.ComboBox(
                0, "(no skills — promote a run first)"
            )
            ui.Label("Trigger:", width=46)
            self._compose_add_trigger_combo = ui.ComboBox(
                0, "", "button_0", "button_1", "button_2", "button_3",
                "button_4", "button_5", "button_6", "button_7",
            )
            ui.Button("+ Add", width=50, clicked_fn=self._on_compose_add_skill)

        ui.Separator()
        ui.Label("Composition:", style={"font_size": 13})
        with ui.ScrollingFrame(
            height=120,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self._compose_rows_container = ui.VStack(spacing=2)
            with self._compose_rows_container:
                ui.Label("(add skills above)", style={"color": 0xFF888888})

        ui.Separator()
        with ui.HStack(height=24, spacing=6):
            ui.Label("Policy name:", width=90)
            self._compose_name_field = ui.StringField(width=ui.Fraction(1))
            self._compose_name_field.model.set_value("my_policy")
        with ui.HStack(height=26, spacing=4):
            ui.Button("Save Policy", width=90, clicked_fn=self._on_compose_save)
            ui.Button("▶ Test", width=70, clicked_fn=self._on_compose_test)
            ui.Button("■ Stop", width=70, clicked_fn=self._on_compose_stop)
        self._compose_status_label = ui.Label(
            "", style={"color": 0xFF888888}, word_wrap=True
        )

        ui.Separator()
        ui.Label("Saved policies", style={"font_size": 14})
        with ui.ScrollingFrame(
            height=120,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self._compose_policy_list_container = ui.VStack(spacing=2)
            with self._compose_policy_list_container:
                ui.Label("(no saved policies)", style={"color": 0xFF888888})

    # ------------------------------------------------------------------
    # Mode navigation
    # ------------------------------------------------------------------

    def _on_mode_button(self, mode: str) -> None:
        self._broadcast(_make_mode_request(mode))

    def _set_mode_panels(self, mode: str) -> None:
        """Show the panel for *mode*, hide all others."""
        self._current_mode = mode
        for key, container in self._mode_panels.items():
            if container is not None:
                try:
                    container.visible = (key == mode)
                except Exception:
                    pass
        self._update_mode_buttons(mode)

    def _update_mode_buttons(self, active_mode: str) -> None:
        for key, btn in self._mode_buttons.items():
            if btn is None:
                continue
            try:
                if key == active_mode:
                    btn.set_style({"Button": {
                        "background_color": 0xFF4ED4FF,
                        "color": 0xFF000000,
                    }})
                else:
                    btn.set_style({})
            except Exception:
                pass

    def _on_mode_changed(self, msg: dict) -> None:
        mode = msg.get("mode", "home")
        self._set_mode_panels(mode)
        log.info(f"StudioExtension: mode → {mode!r}")

    # ------------------------------------------------------------------
    # User event handlers (Kit main thread)
    # ------------------------------------------------------------------

    def _append_chat(self, prefix: str, text: str) -> None:
        if self._chat_log is None:
            return
        current = self._chat_log.text or ""
        self._chat_log.text = current + f"{prefix} {text}\n\n"

    def _on_send_chat(self) -> None:
        if self._chat_input is None:
            return
        text = (self._chat_input.model.as_string or "").strip()
        if not text:
            return
        self._chat_input.model.set_value("")
        self._append_chat("You:", text)
        self._status_label.text = "Claude is thinking..."
        self._broadcast(_make_chat_message(text))

    def _on_pause(self) -> None:
        self._broadcast(_make_pause())
        self._paused = True

    def _on_resume(self) -> None:
        self._broadcast(_make_resume())
        self._paused = False

    def _on_step(self) -> None:
        self._broadcast(_make_step(1))

    def _on_reset(self) -> None:
        self._broadcast(_make_reset())

    def _on_apply_force(self, fx: float, fy: float, fz: float) -> None:
        if not self._bodies:
            return
        idx = 0
        if self._bodies_combo is not None:
            try:
                idx = int(self._bodies_combo.model.get_item_value_model().as_int)
            except Exception:
                idx = 0
        idx = max(0, min(idx, len(self._bodies) - 1))
        body = self._bodies[idx]
        self._broadcast(_make_apply_force(body, force=(fx, fy, fz), duration=0.1))

    def _on_promote(self) -> None:
        if not self._current_robot:
            return
        self._broadcast(_make_approve_design(
            self._current_robot,
            self._approved_env or "flat_ground",
            notes="Approved via Studio UI",
        ))

    def _on_unapprove(self) -> None:
        self._broadcast(_make_unapprove_design())

    def _on_replay_run(self, run_id: str) -> None:
        self._broadcast({"type": "REVIEW_RUN", "run_id": run_id})
        if self._status_label is not None:
            self._status_label.text = f"Replaying run {run_id[:18]}..."

    def _on_promote_to_skill(self, run_id: str, robot_name: str) -> None:
        """User clicked ★ Skill — build a slug from the robot name and promote."""
        import re as _re
        slug = _re.sub(r"[^a-z0-9_]+", "_", robot_name.lower()).strip("_") or "skill"
        display = robot_name.replace("_", " ").title()
        self._broadcast(_make_skill_promote(
            run_id=run_id,
            skill_id=slug,
            display_name=display,
        ))
        if self._status_label is not None:
            self._status_label.text = f"Promoting run {run_id[:18]} → '{display}'..."

    def _on_skill_list(self, msg: dict) -> None:
        self._skill_entries = list(msg.get("skills") or [])
        self._render_skill_list()
        self._rebuild_skill_combo()
        self._render_compose_rows()  # skill labels may have changed

    def _render_skill_list(self) -> None:
        import omni.ui as ui

        if self._skills_list_container is None:
            return
        self._skills_list_container.clear()
        with self._skills_list_container:
            if not self._skill_entries:
                ui.Label("(no skills yet — promote a training run)", style={"color": 0xFF888888})
                return
            for s in self._skill_entries:
                robot = s.get("robot_name", "?")
                skill_id = s.get("skill_id", "?")
                display = s.get("display_name") or skill_id
                best = s.get("best_reward")
                n_variants = s.get("variant_count", 0)
                best_txt = f"{best:+.3f}" if best is not None else "—"
                desc = s.get("task_description", "")
                with ui.HStack(height=20, spacing=6):
                    ui.Label("★", width=16, style={"color": 0xFFFFDD44})
                    ui.Label(display, width=120, style={"font_size": 12})
                    ui.Label(robot, width=90, style={"font_size": 11, "color": 0xFF888888})
                    ui.Label(f"best={best_txt}", width=80, style={"font_size": 11})
                    ui.Label(f"{n_variants}v", width=30, style={"font_size": 11, "color": 0xFF888888})
                if desc:
                    ui.Label(f"  {desc}", style={"font_size": 11, "color": 0xFF888888}, word_wrap=True)

    def _rebuild_skill_combo(self) -> None:
        """Repopulate the simulate-mode and compose add-skill ComboBoxes from _skill_entries."""
        import omni.ui as _ui
        no_skills_label = "(no skills — promote a run first)"

        def _repopulate(combo):
            if combo is None:
                return
            try:
                combo_model = combo.model
                root_items = combo_model.get_item_children(None)
                for item in list(root_items):
                    combo_model.remove_item(item)
                if not self._skill_entries:
                    combo_model.append_child_item(None, _ui.SimpleStringModel(no_skills_label))
                else:
                    for s in self._skill_entries:
                        label = f"{s.get('display_name') or s.get('skill_id', '?')}  ({s.get('robot_name', '?')})"
                        combo_model.append_child_item(None, _ui.SimpleStringModel(label))
            except Exception as exc:
                log.debug(f"_rebuild_skill_combo: {exc}")

        _repopulate(self._sim_skill_combo)
        _repopulate(self._compose_add_skill_combo)

        if self._sim_play_btn is not None:
            self._sim_play_btn.enabled = bool(self._skill_entries)

    def _on_sim_skill_changed(self, model, _item) -> None:
        try:
            self._sim_selected_idx = int(model.get_item_value_model().as_int)
        except Exception:
            self._sim_selected_idx = 0

    def _on_sim_play(self) -> None:
        idx = self._sim_selected_idx
        if not self._skill_entries or idx >= len(self._skill_entries):
            return
        s = self._skill_entries[idx]
        robot_name = s.get("robot_name", "")
        skill_id = s.get("skill_id", "")
        variant_id = s.get("active_variant_id") or s.get("active_variant", "")
        self._broadcast({
            "type": "POLICY_PLAYBACK_START",
            "robot_name": robot_name,
            "skill_id": skill_id,
            "variant_id": variant_id,
        })
        if self._sim_status_label is not None:
            self._sim_status_label.text = f"Starting {s.get('display_name', skill_id)}..."

    def _on_sim_pause(self) -> None:
        if self._sim_playing:
            self._broadcast({"type": "PAUSE"})
        else:
            self._broadcast({"type": "RESUME"})

    def _on_sim_stop(self) -> None:
        self._broadcast({"type": "POLICY_PLAYBACK_STOP"})

    def _on_gamepad_input(self, msg: dict) -> None:
        connected = bool(msg.get("connected", False))
        axes = msg.get("axes") or []
        buttons = msg.get("buttons") or {}

        if self._gamepad_conn_label is not None:
            if connected:
                self._gamepad_conn_label.text = "● Connected"
                self._gamepad_conn_label.set_style({"color": 0xFF00FF80})
            else:
                self._gamepad_conn_label.text = "○ Not connected"
                self._gamepad_conn_label.set_style({"color": 0xFF888888})

        if self._gamepad_axes_label is not None and axes:
            parts = [f"A{i}:{v:+.2f}" for i, v in enumerate(axes[:8])]
            self._gamepad_axes_label.text = "Axes: " + "  ".join(parts)

        if self._gamepad_btns_label is not None and buttons:
            pressed = [k for k, v in buttons.items() if v]
            if pressed:
                self._gamepad_btns_label.text = "Btns: " + " ".join(f"[{b}]" for b in pressed[:8])
            else:
                self._gamepad_btns_label.text = "Btns: —"

    def _on_playback_status(self, msg: dict) -> None:
        state = msg.get("state", "stopped")
        skill_id = msg.get("skill_id", "")
        error = msg.get("error", "")

        self._sim_playing = (state == "playing")

        if self._sim_pause_btn is not None:
            self._sim_pause_btn.enabled = (state == "playing")
            self._sim_pause_btn.text = "⏸ Pause" if state == "playing" else "▶ Resume"
        if self._sim_stop_btn is not None:
            self._sim_stop_btn.enabled = (state == "playing")
        if self._sim_play_btn is not None:
            self._sim_play_btn.enabled = (state != "playing") and bool(self._skill_entries)

        if self._sim_status_label is not None:
            if state == "playing":
                self._sim_status_label.text = f"Playing: {skill_id}"
                self._sim_status_label.set_style({"color": 0xFF00FF80})
            elif state == "stopped" and error:
                self._sim_status_label.text = f"Error: {error}"
                self._sim_status_label.set_style({"color": 0xFFFF6666})
            else:
                self._sim_status_label.text = "Stopped"
                self._sim_status_label.set_style({"color": 0xFF888888})

    # ------------------------------------------------------------------
    # Policy Composer handlers
    # ------------------------------------------------------------------

    _TRIGGERS = [
        "", "button_0", "button_1", "button_2", "button_3",
        "button_4", "button_5", "button_6", "button_7",
    ]

    def _on_compose_add_skill(self) -> None:
        """Add one skill row to the composition list."""
        if not self._skill_entries:
            return
        try:
            skill_idx = int(self._compose_add_skill_combo.model.get_item_value_model().as_int)
        except Exception:
            skill_idx = 0
        skill_idx = max(0, min(skill_idx, len(self._skill_entries) - 1))

        try:
            trig_idx = int(self._compose_add_trigger_combo.model.get_item_value_model().as_int)
        except Exception:
            trig_idx = 0
        trigger = self._TRIGGERS[trig_idx] if trig_idx < len(self._TRIGGERS) else ""

        self._compose_rows.append({"skill_idx": skill_idx, "trigger": trigger})
        self._render_compose_rows()

    def _render_compose_rows(self) -> None:
        import omni.ui as ui

        if self._compose_rows_container is None:
            return
        self._compose_rows_container.clear()
        with self._compose_rows_container:
            if not self._compose_rows:
                ui.Label("(add skills above)", style={"color": 0xFF888888})
                return
            for i, row in enumerate(self._compose_rows):
                sidx = row["skill_idx"]
                if sidx < len(self._skill_entries):
                    s = self._skill_entries[sidx]
                    skill_label = s.get("display_name") or s.get("skill_id", "?")
                    robot_label = s.get("robot_name", "?")
                else:
                    skill_label = "(unknown)"
                    robot_label = "?"
                trigger = row.get("trigger", "") or "(default)"
                with ui.HStack(height=20, spacing=4):
                    ui.Label(f"{i+1}.", width=16, style={"font_size": 11})
                    ui.Label(skill_label, width=110, style={"font_size": 12})
                    ui.Label(robot_label, width=70, style={"font_size": 11, "color": 0xFF888888})
                    ui.Label(f"→ {trigger}", width=90, style={"font_size": 11, "color": 0xFFFFDD44})
                    ui.Button(
                        "✗", width=22, height=18,
                        style={"font_size": 11, "color": 0xFFFF6666},
                        clicked_fn=lambda _i=i: self._on_compose_remove_row(_i),
                    )

    def _on_compose_remove_row(self, idx: int) -> None:
        if 0 <= idx < len(self._compose_rows):
            self._compose_rows.pop(idx)
            self._render_compose_rows()

    def _on_compose_save(self) -> None:
        if not self._compose_rows or not self._skill_entries:
            if self._compose_status_label is not None:
                self._compose_status_label.text = "Add at least one skill before saving."
            return
        policy_name = ""
        if self._compose_name_field is not None:
            try:
                policy_name = self._compose_name_field.model.as_string.strip()
            except Exception:
                pass
        if not policy_name:
            if self._compose_status_label is not None:
                self._compose_status_label.text = "Enter a policy name first."
            return

        # Infer robot name from first skill
        first_s = self._skill_entries[self._compose_rows[0]["skill_idx"]]
        robot_name = first_s.get("robot_name", "")

        skills = []
        for row in self._compose_rows:
            sidx = row["skill_idx"]
            if sidx >= len(self._skill_entries):
                continue
            s = self._skill_entries[sidx]
            skills.append({
                "skill_id": s.get("skill_id", ""),
                "variant_id": s.get("active_variant_id") or s.get("active_variant", ""),
                "trigger": row.get("trigger", ""),
            })

        self._broadcast(_make_policy_save(robot_name, policy_name, skills))
        if self._compose_status_label is not None:
            self._compose_status_label.text = f"Saving '{policy_name}'..."

    def _on_compose_test(self) -> None:
        policy_name = ""
        if self._compose_name_field is not None:
            try:
                policy_name = self._compose_name_field.model.as_string.strip()
            except Exception:
                pass
        if not policy_name or not self._skill_entries:
            return
        # Infer robot from first skill row or first skill entry
        robot_name = ""
        if self._compose_rows and self._skill_entries:
            sidx = self._compose_rows[0].get("skill_idx", 0)
            if sidx < len(self._skill_entries):
                robot_name = self._skill_entries[sidx].get("robot_name", "")
        if not robot_name and self._skill_entries:
            robot_name = self._skill_entries[0].get("robot_name", "")
        self._broadcast(_make_policy_playback_start_composed(robot_name, policy_name))
        if self._compose_status_label is not None:
            self._compose_status_label.text = f"Starting '{policy_name}'..."

    def _on_compose_stop(self) -> None:
        self._broadcast({"type": "POLICY_PLAYBACK_STOP"})
        self._compose_playing = False
        if self._compose_status_label is not None:
            self._compose_status_label.text = "Stopped."

    def _on_policy_list(self, msg: dict) -> None:
        self._compose_policies = list(msg.get("policies") or [])
        self._render_compose_policy_list()

    def _render_compose_policy_list(self) -> None:
        import omni.ui as ui

        if self._compose_policy_list_container is None:
            return
        self._compose_policy_list_container.clear()
        with self._compose_policy_list_container:
            if not self._compose_policies:
                ui.Label("(no saved policies)", style={"color": 0xFF888888})
                return
            for p in self._compose_policies:
                pname = p.get("policy_name", "?")
                robot = p.get("robot_name", "?")
                n_skills = len(p.get("skills", []))
                with ui.HStack(height=20, spacing=6):
                    ui.Label(pname, width=110, style={"font_size": 12})
                    ui.Label(robot, width=80, style={"font_size": 11, "color": 0xFF888888})
                    ui.Label(f"{n_skills} skills", width=60, style={"font_size": 11})
                    ui.Button(
                        "▶ Play", width=60, height=18,
                        style={"font_size": 11},
                        clicked_fn=lambda _r=robot, _p=pname: self._broadcast(
                            _make_policy_playback_start_composed(_r, _p)
                        ),
                    )
                    ui.Button(
                        "✗", width=22, height=18,
                        style={"font_size": 11, "color": 0xFFFF6666},
                        clicked_fn=lambda _r=robot, _p=pname: self._on_compose_delete_policy(_r, _p),
                    )

    def _on_compose_delete_policy(self, robot_name: str, policy_name: str) -> None:
        self._broadcast(_make_policy_delete(robot_name, policy_name))
        if self._compose_status_label is not None:
            self._compose_status_label.text = f"Deleted '{policy_name}'."

    # ------------------------------------------------------------------

    def _on_joint_slider_changed(self, joint_name: str, model) -> None:
        try:
            value = float(model.as_float)
        except Exception:
            return
        self._broadcast(_make_joint_target(joint_name, value))

    # ------------------------------------------------------------------
    # Inbound message handling
    # ------------------------------------------------------------------

    def _on_backend_message(self, msg: dict) -> None:
        """Called by isaac_server when a backend message arrives.

        Runs on the WS asyncio thread — dispatch UI mutations to the Kit
        thread via omni.kit.app.get_app().post_to_main_thread.
        """
        msg_type = msg.get("type", "?")
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()

            def _apply(msg=msg):
                try:
                    self._apply_backend_message(msg)
                except Exception as exc:
                    log.warning(
                        f"StudioExtension: {msg_type} handler raised — {exc}"
                    )

            posted = False
            if hasattr(app, "post_to_main_thread"):
                try:
                    app.post_to_main_thread(_apply)
                    posted = True
                except Exception as exc:
                    log.warning(
                        f"StudioExtension: post_to_main_thread({msg_type}) failed — {exc}"
                    )
            if not posted:
                try:
                    stream = app.get_update_event_stream()
                    subscription_holder: dict = {}

                    def _once(e, _m=msg):
                        sub = subscription_holder.pop("sub", None)
                        if sub is not None:
                            sub.unsubscribe()
                        _apply(_m)

                    subscription_holder["sub"] = (
                        stream.create_subscription_to_pop(_once, name="ext_studio_apply")
                    )
                    posted = True
                except Exception as exc:
                    log.debug(
                        f"StudioExtension: update-stream dispatch failed — {exc}"
                    )

            if not posted:
                _apply()
        except Exception as exc:
            log.warning(
                f"StudioExtension: failed to dispatch {msg_type} — {exc}"
            )

    def _apply_backend_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "CHAT_REPLY":
            self._append_chat("Claude:", msg.get("text", ""))
            if self._status_label is not None:
                self._status_label.text = ""
        elif msg_type == "TOOL_STATUS":
            if self._status_label is not None:
                self._status_label.text = f"{msg.get('tool', '?')}: {msg.get('status', '')}"
        elif msg_type == "TOOL_RESULT":
            if self._status_label is not None:
                self._status_label.text = f"{msg.get('tool', '?')}: {msg.get('summary', '')}"
        elif msg_type == "ROBOT_META":
            self._rebuild_joint_sliders(msg)
        elif msg_type == "GATE_STATUS":
            self._update_gate(msg)
        elif msg_type == "PHASE_CHANGED":
            self._update_phase(msg)
        elif msg_type == "MODE_CHANGED":
            self._on_mode_changed(msg)
        elif msg_type == "TRAIN_RUN_START":
            log.info(f"TRAIN_RUN_START run_id={msg.get('run_id')}")
            self._on_train_run_start(msg)
        elif msg_type == "TRAIN_UPDATE":
            self._on_train_update(msg)
        elif msg_type == "TRAIN_RUN_END":
            log.info(
                f"TRAIN_RUN_END run_id={msg.get('run_id')} "
                f"success={msg.get('success')}"
            )
            self._on_train_run_end(msg)
        elif msg_type == "TRAIN_HISTORY":
            log.info(f"TRAIN_HISTORY runs={len(msg.get('runs', []))}")
            self._on_train_history(msg)
        elif msg_type == "TRAIN_REWARD_BREAKDOWN":
            self._on_train_reward_breakdown(msg)
        elif msg_type == "TRAIN_ROLLOUT_PREVIEW":
            self._on_train_rollout_preview(msg)
        elif msg_type == "SKILL_LIST":
            log.info(f"SKILL_LIST skills={len(msg.get('skills', []))}")
            self._on_skill_list(msg)
        elif msg_type == "POLICY_LIST":
            log.info(f"POLICY_LIST policies={len(msg.get('policies', []))}")
            self._on_policy_list(msg)
        elif msg_type == "POLICY_PLAYBACK_STATUS":
            self._on_playback_status(msg)
            # Also update compose panel status
            state = msg.get("state", "stopped")
            self._compose_playing = (state == "playing")
            if self._compose_status_label is not None:
                try:
                    if state == "playing":
                        self._compose_status_label.text = f"Playing: {msg.get('skill_id', '')}"
                        self._compose_status_label.set_style({"color": 0xFF00FF80})
                    elif msg.get("error"):
                        self._compose_status_label.text = f"Error: {msg['error']}"
                        self._compose_status_label.set_style({"color": 0xFFFF6666})
                    else:
                        self._compose_status_label.text = "Stopped."
                        self._compose_status_label.set_style({"color": 0xFF888888})
                except Exception:
                    pass
        elif msg_type == "GAMEPAD_INPUT":
            self._on_gamepad_input(msg)

    def _rebuild_joint_sliders(self, meta: dict) -> None:
        import omni.ui as ui

        self._current_robot = meta.get("name", "")
        self._joints = list(meta.get("joints", []))
        self._bodies = list(meta.get("bodies", []))

        if self._joint_container is None:
            return

        self._joint_container.clear()
        with self._joint_container:
            ui.Label(f"Robot: {self._current_robot}", style={"font_size": 14})
            ui.Label(
                f"{len(self._joints)} joint(s) — {len(self._bodies)} body(ies)",
                style={"color": 0xFF888888},
            )
            for j in self._joints:
                if j.get("type") in ("free", "ball"):
                    continue
                ctrl_range = j.get("ctrl_range") or j.get("range") or [-1.0, 1.0]
                lo, hi = float(ctrl_range[0]), float(ctrl_range[1])
                jname = j["name"]
                with ui.HStack(height=22):
                    ui.Label(jname, width=130, style={"font_size": 11})
                    mid = 0.5 * (lo + hi)
                    slider = ui.FloatSlider(min=lo, max=hi, step=(hi - lo) / 200.0)
                    slider.model.set_value(mid)

                    def _make_cb(name=jname, m=slider.model):
                        return lambda _m: self._on_joint_slider_changed(name, m)

                    slider.model.add_value_changed_fn(_make_cb())

        if self._bodies_combo is not None:
            try:
                combo_model = self._bodies_combo.model
                root_item = combo_model.get_item_children(None)
                for item in root_item:
                    combo_model.remove_item(item)
                for body in self._bodies:
                    combo_model.append_child_item(
                        None, ui.SimpleStringModel(body)
                    )
            except Exception:
                pass

    def _update_gate(self, msg: dict) -> None:
        self._last_gate = msg
        lines = []
        lines.append(("✓" if msg.get("robot_loaded") else "•") + " robot loaded")
        lines.append(("✓" if msg.get("env_loaded") else "•") + " environment loaded")
        lines.append(("✓" if msg.get("sim_ran") else "•") + " passive sim ran")
        lines.append(("✓" if msg.get("sim_stable") else "•") + " sim did not diverge")
        missing = msg.get("missing") or []
        if missing:
            lines.append("")
            lines.append("Missing:")
            for m in missing:
                lines.append(f"  - {m}")

        if self._gate_label is not None:
            self._gate_label.text = "\n".join(lines)
        if self._promote_button is not None:
            self._promote_button.enabled = bool(msg.get("can_approve"))

    def _update_phase(self, msg: dict) -> None:
        phase = msg.get("phase", "design")
        if self._phase_label is not None:
            self._phase_label.text = phase
            self._phase_label.set_style(
                {"color": 0xFF00FF80 if phase == "training" else 0xFF4ED4FF}
            )
        if msg.get("approved_environment"):
            self._approved_env = msg["approved_environment"]

    # ------------------------------------------------------------------
    # Training panel handlers
    # ------------------------------------------------------------------

    def _on_train_run_start(self, msg: dict) -> None:
        self._train_run_id = msg.get("run_id", "")
        self._train_rewards = []
        self._train_reward_curve = []
        self._train_best = float("-inf")

        header = (
            f"Run {self._train_run_id[:18]}  |  "
            f"{msg.get('robot_name', '?')} on {msg.get('environment_name', '?')}  |  "
            f"{msg.get('algorithm', 'ppo').upper()}"
        )
        if self._train_header_label is not None:
            self._train_header_label.text = header
            self._train_header_label.set_style({"color": 0xFFFFDD88})
        if self._train_backend_label is not None:
            self._train_backend_label.text = "Backend: detecting..."
            self._train_backend_label.set_style({"font_size": 13, "color": 0xFFAAAAAA})
        if self._train_status_label is not None:
            self._train_status_label.text = (
                f"running — target {msg.get('total_timesteps', 0):,} timesteps"
            )
        if self._train_progress_bar is not None:
            self._train_progress_bar.model.set_value(0.0)
        if self._train_progress_text is not None:
            self._train_progress_text.text = "0%"
        for key, prefix in (
            ("mean", "mean"),
            ("best", "best"),
            ("tps", "t/s"),
            ("elapsed", "elapsed"),
        ):
            label = self._train_metric_labels.get(key)
            if label is not None:
                label.text = f"{prefix}: —"
        self._train_previews = []
        self._train_last_components = {}
        if self._train_preview_container is not None:
            try:
                self._train_preview_container.clear()
                import omni.ui as ui
                with self._train_preview_container:
                    ui.Label("(none yet)", style={"color": 0xFF666666, "font_size": 11})
            except Exception:
                pass
        if self._train_component_container is not None:
            try:
                self._train_component_container.clear()
                import omni.ui as ui
                with self._train_component_container:
                    ui.Label("(no component data yet)", style={"color": 0xFF666666, "font_size": 11})
            except Exception:
                pass
        self._rebuild_reward_plot([])

        if self._phase_label is not None:
            self._phase_label.text = "training (run active)"
            self._phase_label.set_style({"color": 0xFFFFDD88})
        if self._root_scroll is not None:
            try:
                self._root_scroll.scroll_y = 10_000
            except Exception:
                pass

    def _on_train_update(self, msg: dict) -> None:
        mean_r = float(msg.get("mean_reward", 0.0))
        best_r = msg.get("best_reward")
        if best_r is not None:
            self._train_best = float(best_r)
        elif mean_r > self._train_best:
            self._train_best = mean_r

        timestep = int(msg.get("timestep", 0))
        self._train_reward_curve.append((timestep, mean_r))
        self._train_rewards.append(mean_r)
        if len(self._train_rewards) > 80:
            self._train_rewards = self._train_rewards[-80:]

        total = int(msg.get("total_timesteps", 0) or 0)
        pct = (timestep / total) if total > 0 else 0.0
        pct = max(0.0, min(1.0, pct))
        if self._train_progress_bar is not None:
            self._train_progress_bar.model.set_value(pct)
        if self._train_progress_text is not None:
            self._train_progress_text.text = f"{pct * 100:.0f}%"

        backend = msg.get("backend", "")
        if backend and self._train_backend_label is not None:
            self._train_backend_label.text = f"Backend: {backend}"
            if "GPU" in backend or "Brax" in backend:
                color = 0xFF00FF80
            elif "Random" in backend:
                color = 0xFFFF6666
            else:
                color = 0xFFFFDD44
            self._train_backend_label.set_style({"font_size": 13, "color": color})

        tps = msg.get("timesteps_per_second")
        elapsed = msg.get("elapsed_s")
        labels = self._train_metric_labels
        if labels.get("mean") is not None:
            labels["mean"].text = f"mean: {mean_r:.3f}"
        if labels.get("best") is not None:
            labels["best"].text = f"best: {self._train_best:.3f}"
        if labels.get("tps") is not None:
            labels["tps"].text = f"t/s: {tps:,.0f}" if tps is not None else "t/s: —"
        if labels.get("elapsed") is not None:
            labels["elapsed"].text = f"elapsed: {self._fmt_duration(elapsed)}"

        self._rebuild_reward_plot(self._train_rewards)
        if self._train_status_label is not None:
            self._train_status_label.text = (
                f"step {timestep:,}"
                + (f" / {total:,}" if total else "")
            )

    def _on_train_run_end(self, msg: dict) -> None:
        success = bool(msg.get("success"))
        best = msg.get("best_reward")
        err = msg.get("error", "")
        training_time = msg.get("training_time_seconds")

        if self._train_header_label is not None:
            col = 0xFF00FF80 if success else 0xFFFF6666
            label = "completed" if success else "failed"
            self._train_header_label.set_style({"color": col})
            best_txt = f" best={best:.3f}" if success and best is not None else ""
            self._train_header_label.text = (
                f"Run {msg.get('run_id', '')[:18]} {label}"
                f" in {self._fmt_duration(training_time)}{best_txt}"
            )
        if self._train_status_label is not None:
            self._train_status_label.text = err if err else ("done" if success else "failed")
        if success and self._train_progress_bar is not None:
            self._train_progress_bar.model.set_value(1.0)
            if self._train_progress_text is not None:
                self._train_progress_text.text = "100%"

        backend_txt = ""
        if self._train_backend_label is not None:
            lbl = self._train_backend_label.text or ""
            if lbl.startswith("Backend: "):
                backend_txt = lbl[len("Backend: "):]
        record = {
            "run_id": msg.get("run_id", ""),
            "robot_name": msg.get("robot_name", ""),
            "best_reward": best,
            "training_time_seconds": training_time,
            "total_timesteps": msg.get("total_timesteps"),
            "success": success,
            "error": err,
            "ended_at": msg.get("ended_at"),
            "backend": backend_txt,
        }
        self._train_runs_history = [record] + self._train_runs_history
        self._train_runs_history = self._train_runs_history[:50]
        self._render_run_history()

    def _on_train_history(self, msg: dict) -> None:
        runs = msg.get("runs") or []
        self._train_runs_history = list(runs)[:50]
        self._render_run_history()

    # ------------------------------------------------------------------
    # Training panel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_duration(seconds) -> str:
        if seconds is None:
            return "—"
        try:
            s = int(float(seconds))
        except Exception:
            return "—"
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _sparkline(values: list[float], width: int = 60) -> str:
        if not values:
            return "▁" * width
        blocks = "▁▂▃▄▅▆▇█"
        series = values[-width:]
        lo = min(series)
        hi = max(series)
        span = hi - lo if hi > lo else 1.0
        out = []
        for v in series:
            idx = int(round((v - lo) / span * (len(blocks) - 1)))
            out.append(blocks[max(0, min(len(blocks) - 1, idx))])
        return "".join(out) + f"  [{lo:+.2f} → {hi:+.2f}]"

    def _rebuild_reward_plot(self, rewards: list[float]) -> None:
        """Rebuild the reward curve using omni.ui.Plot, falling back to text sparkline."""
        if self._train_plot_container is None:
            return
        try:
            import omni.ui as ui

            data = rewards[-60:] if rewards else [0.0]
            lo = min(data)
            hi = max(data)
            if hi <= lo:
                hi = lo + 1.0

            self._train_plot_container.clear()
            with self._train_plot_container:
                try:
                    ui.Plot(
                        ui.Type.LINE,
                        lo, hi,
                        *data,
                        width=ui.Fraction(1),
                        height=72,
                        style={
                            "color": 0xFF4ED4FF,
                            "secondary_color": 0xFF1A2A3A,
                        },
                    )
                    self._train_plot_mode = "plot"
                except Exception:
                    # ui.Plot not available (older Isaac Sim) — use text sparkline
                    self._train_sparkline_label = ui.Label(
                        self._sparkline(rewards) if rewards else "▁" * 40,
                        style={"font_size": 13, "color": 0xFFCCDDFF},
                        word_wrap=False,
                    )
                    self._train_plot_mode = "sparkline"
        except Exception as exc:
            log.debug(f"_rebuild_reward_plot: {exc}")

    def _on_train_reward_breakdown(self, msg: dict) -> None:
        components = msg.get("components") or {}
        if not components:
            return
        self._train_last_components = dict(components)
        if self._train_component_container is None:
            return
        try:
            import omni.ui as ui

            self._train_component_container.clear()
            with self._train_component_container:
                if not components:
                    ui.Label("(no component data)", style={"color": 0xFF666666, "font_size": 11})
                    return
                # Sort by absolute value descending
                items = sorted(components.items(), key=lambda kv: abs(kv[1]), reverse=True)
                total_abs = sum(abs(v) for _, v in items) or 1.0
                for name, val in items[:8]:   # cap at 8 rows
                    frac = min(1.0, abs(val) / total_abs)
                    bar = "█" * max(1, int(frac * 20))
                    color = 0xFF00FF80 if val >= 0 else 0xFFFF6666
                    with ui.HStack(height=16, spacing=4):
                        ui.Label(name[:18], width=110, style={"font_size": 11})
                        ui.Label(f"{val:+.3f}", width=60, style={"font_size": 11, "color": color})
                        ui.Label(bar, style={"font_size": 10, "color": color})
        except Exception as exc:
            log.debug(f"_on_train_reward_breakdown: {exc}")

    def _on_train_rollout_preview(self, msg: dict) -> None:
        timestep = int(msg.get("timestep", 0))
        num_frames = int(msg.get("num_frames", 0))
        run_id = msg.get("run_id", "")

        self._train_previews.append({"timestep": timestep, "num_frames": num_frames})
        if len(self._train_previews) > 8:
            self._train_previews = self._train_previews[-8:]

        if self._train_preview_container is None:
            return
        try:
            import omni.ui as ui

            self._train_preview_container.clear()
            with self._train_preview_container:
                for p in self._train_previews:
                    ts = p["timestep"]
                    label = f"step={ts:,}"
                    ui.Button(
                        label,
                        width=80, height=22,
                        style={"font_size": 10},
                        clicked_fn=lambda _id=run_id, _ts=ts: self._on_replay_run(_id),
                    )
        except Exception as exc:
            log.debug(f"_on_train_rollout_preview: {exc}")

    def _render_run_history(self) -> None:
        import omni.ui as ui

        if self._train_history_container is None:
            return
        self._train_history_container.clear()
        with self._train_history_container:
            if not self._train_runs_history:
                ui.Label("(no previous runs)", style={"color": 0xFF888888})
                return
            for r in self._train_runs_history[:15]:
                success = bool(r.get("success"))
                marker = "✓" if success else "✗"
                color = 0xFF00FF80 if success else 0xFFFF6666
                best = r.get("best_reward")
                best_txt = f"{best:+.3f}" if best is not None else "—"
                robot = r.get("robot_name", "?")
                dur = self._fmt_duration(r.get("training_time_seconds"))
                steps = r.get("total_timesteps")
                steps_txt = f"{int(steps):,}" if steps else "—"
                run_id = r.get("run_id", "")
                backend_hint = r.get("backend", "")
                if "GPU" in backend_hint or "Brax" in backend_hint:
                    backend_color = 0xFF00FF80
                elif backend_hint and "Random" not in backend_hint:
                    backend_color = 0xFFFFDD44
                else:
                    backend_color = 0xFF888888
                with ui.HStack(height=20, spacing=6):
                    ui.Label(marker, width=16, style={"color": color})
                    ui.Label(robot, width=90, style={"font_size": 12})
                    ui.Label(f"best={best_txt}", width=80, style={"font_size": 12})
                    ui.Label(f"steps={steps_txt}", width=90, style={"font_size": 12})
                    ui.Label(dur, width=60, style={"font_size": 12, "color": 0xFF888888})
                    if backend_hint:
                        ui.Label(backend_hint[:14], width=90, style={"font_size": 11, "color": backend_color})
                    if success and run_id:
                        ui.Button(
                            "▶ Replay",
                            width=70,
                            height=18,
                            style={"font_size": 11},
                            clicked_fn=lambda _id=run_id: self._on_replay_run(_id),
                        )
                        ui.Button(
                            "★ Skill",
                            width=60,
                            height=18,
                            style={"font_size": 11},
                            clicked_fn=lambda _id=run_id, _r=robot: self._on_promote_to_skill(_id, _r),
                        )
