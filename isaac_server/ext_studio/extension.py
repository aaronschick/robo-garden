"""Studio UI extension for the Isaac Sim server.

Runs inside the Isaac Sim Kit process.  Builds a dockable ``omni.ui`` window
with four sections:

    +--------------------------------------------+
    |  Chat           | Robot Controls           |
    |  (history)      |  joint sliders           |
    |  [input field]  |                          |
    |                 |                          |
    +--------------------------------------------+
    |  Interaction toolbar:                      |
    |  [Pause] [Step] [Reset] [Apply-Force mode] |
    +--------------------------------------------+
    |  Approval: gate checklist + Promote button |
    +--------------------------------------------+

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
        self._train_progress_bar = None
        self._train_progress_text = None
        self._train_metric_labels: dict = {}
        self._train_sparkline_label = None
        self._train_history_container = None
        self._train_rewards: list[float] = []     # recent mean_reward samples
        self._train_reward_curve: list[tuple[int, float]] = []
        self._train_run_id: str = ""
        self._train_runs_history: list[dict] = []
        self._train_best: float = float("-inf")

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
        # Wrap the whole stack in a scrolling frame so no panel (particularly
        # the Training Progress panel at the bottom) is ever clipped off the
        # window when the user's screen is smaller than the 920 px design
        # height.  Prior to this, the training panel existed but was not
        # reachable.
        with self._window.frame:
            self._root_scroll = ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            with self._root_scroll:
                with ui.VStack(spacing=6):
                    self._build_phase_banner(ui)
                    ui.Separator()
                    self._build_chat_panel(ui)
                    ui.Separator()
                    self._build_robot_panel(ui)
                    ui.Separator()
                    self._build_interaction_toolbar(ui)
                    ui.Separator()
                    self._build_approval_panel(ui)
                    ui.Separator()
                    self._build_training_panel(ui)

        # Subscribe to inbound backend messages
        self._register(self._on_backend_message)
        log.info("StudioExtension window ready")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
            )
        with ui.HStack(height=26):
            self._chat_input = ui.StringField()
            ui.Spacer(width=4)
            ui.Button("Send", width=60, clicked_fn=self._on_send_chat)
        self._status_label = ui.Label("", style={"color": 0xFF888888})

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
        """Training Progress: live status + reward sparkline + run history.

        Populated from TRAIN_RUN_START / TRAIN_UPDATE / TRAIN_RUN_END / TRAIN_HISTORY
        messages broadcast by the backend.  Hidden-but-present when idle so the
        user sees the panel as soon as ``handle_train`` kicks off a run.
        """
        ui.Label("Training Progress", style={"font_size": 16})
        self._train_header_label = ui.Label(
            "(no active run)",
            style={"color": 0xFF888888},
            word_wrap=True,
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

        # Reward sparkline — unicode block chars drawn into a Label.  Keeps us
        # free of omni.ui charting APIs that vary between Kit versions.
        self._train_sparkline_label = ui.Label(
            "Reward curve will appear here once a run starts.",
            style={"font_size": 13, "color": 0xFFCCDDFF},
            word_wrap=False,
        )

        ui.Label("Run history", style={"font_size": 14})
        with ui.ScrollingFrame(
            height=140,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self._train_history_container = ui.VStack(spacing=2)
            with self._train_history_container:
                ui.Label("(no previous runs)", style={"color": 0xFF888888})

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
        thread via omni.kit.app.get_app().post_to_main_thread.  omni.ui
        operations on a background thread can crash Kit.
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
            # Prefer the one-shot post-to-main-thread when available.  Some
            # Kit 5.x builds drop post_to_main_thread, so we also accept
            # ``get_update_event_stream`` + a one-shot subscription.
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
                # Last-resort direct call.  omni.ui on a non-main thread is
                # risky but better than dropping the message silently.
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
                    continue  # no scalar slider for these
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

        # Refresh body combo
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
                # ComboBox API varies between Kit versions; ignore if we can't mutate
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
        if self._train_sparkline_label is not None:
            self._train_sparkline_label.text = "▁" * 40

        # Call attention to the panel: tint the phase banner and auto-scroll
        # the root frame so the panel is visible even if the user was reading
        # the chat log.  Without these cues, users couldn't tell a run had
        # actually kicked off.
        if self._phase_label is not None:
            self._phase_label.text = "training (run active)"
            self._phase_label.set_style({"color": 0xFFFFDD88})
        if self._root_scroll is not None:
            try:
                self._root_scroll.scroll_y = 10_000  # clamps to max
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

        if self._train_sparkline_label is not None:
            self._train_sparkline_label.text = self._sparkline(self._train_rewards)
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

        # Synthesise a history record from the end message so the run shows
        # up even before Studio reconnects and reads runs.jsonl.
        record = {
            "run_id": msg.get("run_id", ""),
            "robot_name": msg.get("robot_name", ""),
            "best_reward": best,
            "training_time_seconds": training_time,
            "total_timesteps": msg.get("total_timesteps"),
            "success": success,
            "error": err,
            "ended_at": msg.get("ended_at"),
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
                with ui.HStack(height=20, spacing=6):
                    ui.Label(marker, width=16, style={"color": color})
                    ui.Label(robot, width=120, style={"font_size": 12})
                    ui.Label(f"best={best_txt}", width=110, style={"font_size": 12})
                    ui.Label(f"steps={steps_txt}", width=120, style={"font_size": 12})
                    ui.Label(dur, style={"font_size": 12, "color": 0xFF888888})
