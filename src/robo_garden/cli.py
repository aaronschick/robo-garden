"""Main entry point for the Robo Garden application."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Force UTF-8 output on Windows so Rich can render Unicode block characters.
# Must be set before any I/O happens (reconfigure works on Python 3.7+).
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # stdout is not a TextIOWrapper (e.g. piped) — skip


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

_SPARKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of floats as a Unicode block sparkline of fixed width."""
    if not values:
        return " " * width
    win = values[-width:]
    lo, hi = min(win), max(win)
    if hi == lo:
        return "▄" * len(win) + " " * (width - len(win))
    chars = [_SPARKS[int((v - lo) / (hi - lo) * 8)] for v in win]
    return "".join(chars).ljust(width)


# ---------------------------------------------------------------------------
# Live training display
# ---------------------------------------------------------------------------

def _run_training_live(
    robot_name: str,
    robot_path: Path,
    total_timesteps: int,
    num_envs: int,
    reward_fn: Callable | None = None,
    done_fn: Callable | None = None,
    jax_reward_fn: Callable | None = None,
    jax_done_fn: Callable | None = None,
    max_episode_steps: int | None = None,
    wsl_run_id: str | None = None,
    reward_fn_code: str = "",
    use_wsl: bool = False,
) -> None:
    """Run training with a Rich Live terminal display.

    When ``use_wsl=True`` (Windows with WSL2 GPU available), the job is
    dispatched to the WSL worker via ``wsl_dispatch.run_in_wsl()`` and
    ``__RG_PROGRESS__`` ticks drive the same Rich display as the local path.

    On Linux/WSL2 (JAX GPU available), ``jax_reward_fn`` / ``jax_done_fn`` are
    forwarded to the Brax PPO path.  On Windows without WSL, ``reward_fn`` /
    ``done_fn`` (numpy) are used by the SB3 PPO fallback.

    ``max_episode_steps`` defaults to 500, which is fine for cartpole at
    dt=0.02 (10 s episodes) but far too short for locomotion at dt=0.002
    (just 1 s).  Locomotion tasks should pass ``max_episode_steps=2500`` or
    more so the policy has time to develop a gait before truncation.
    """
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.columns import Columns
    from rich.text import Text
    from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
    from robo_garden.training.models import TrainingConfig

    console = Console()

    _ep_steps = max_episode_steps if max_episode_steps is not None else 500

    state: dict = {
        "step": 0,
        "mean_reward": 0.0,
        "best_reward": float("-inf"),
        "reward_history": [],
        "algorithm": "Brax PPO (GPU/JAX)" if use_wsl else "starting…",
        "start_time": time.time(),
    }

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task(f"Training {robot_name}", total=total_timesteps)

    def _make_panel() -> Panel:
        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="bold cyan", min_width=16)
        tbl.add_column(min_width=20)

        elapsed = time.time() - state["start_time"]
        fps = state["step"] / max(elapsed, 1e-6)
        eta_s = (total_timesteps - state["step"]) / max(fps, 1e-6)

        tbl.add_row("Robot", robot_name)
        tbl.add_row("Algorithm", state["algorithm"])
        tbl.add_row("Timestep", f"{state['step']:,} / {total_timesteps:,}")
        tbl.add_row("Mean reward", f"{state['mean_reward']:>10.3f}")
        tbl.add_row("Best reward", f"{state['best_reward']:>10.3f}")
        tbl.add_row("FPS", f"{fps:>10.0f}")
        tbl.add_row("ETA", f"{eta_s / 60:>8.1f} min")

        spark = _sparkline(state["reward_history"], width=40)
        return Panel(
            Columns([tbl, Text("\n")]),
            title=f"[bold magenta]Robo Garden — {robot_name} training[/]",
            subtitle=f"Reward trend  [green]{spark}[/]",
            border_style="bright_blue",
        )

    def _on_step(step: int, metrics: dict) -> None:
        mean_r = float(metrics.get("eval/episode_reward", metrics.get("mean_reward", 0.0)))
        backend = metrics.get("_backend", "")
        if backend and not backend.endswith("compiling...") and not backend.endswith("VRAM..."):
            state["algorithm"] = backend
        state["step"] = step
        state["mean_reward"] = mean_r
        if mean_r > state["best_reward"]:
            state["best_reward"] = mean_r
        state["reward_history"].append(mean_r)
        progress.update(task_id, completed=min(step, total_timesteps))

    console.print(
        f"\n[bold cyan]Robo Garden[/] — training [bold]{robot_name}[/] "
        f"for [bold]{total_timesteps:,}[/] timesteps"
        + (" [dim](WSL2 GPU)[/dim]" if use_wsl else "")
        + "\n"
    )

    if use_wsl:
        # ── WSL GPU path ─────────────────────────────────────────────────
        # Dispatch to the WSL worker via the same run_in_wsl() path used by
        # handle_train (gym/tui mode).  Progress ticks flow back over
        # __RG_PROGRESS__ stdout and drive the same Rich panel as the local path.
        from datetime import datetime, timezone
        from uuid import uuid4
        from robo_garden.training import wsl_dispatch

        run_id = wsl_run_id or (
            f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        )

        with Live(console=console, refresh_per_second=4, transient=False) as live:
            def _live_callback(step: int, metrics: dict) -> None:
                _on_step(step, metrics)
                live.update(_make_panel())

            wsl_result = wsl_dispatch.run_in_wsl(
                run_id=run_id,
                robot_xml=robot_path.read_text(encoding="utf-8"),
                env_mjcf="",
                reward_fn_code=reward_fn_code,
                robot_name=robot_name,
                environment_name=robot_name,
                algorithm="ppo",
                total_timesteps=total_timesteps,
                num_envs=num_envs,
                max_episode_steps=_ep_steps,
                progress_callback=_live_callback,
            )
            live.update(_make_panel())

        if wsl_result.get("success"):
            console.print(
                f"\n[bold green]Done![/]  "
                f"best_reward=[bold]{wsl_result.get('best_reward', 0.0):.3f}[/]  "
                f"time=[bold]{wsl_result.get('training_time_seconds', 0.0):.1f}s[/]  "
                f"checkpoint=[dim]{wsl_result.get('checkpoint_path', '')}[/]"
            )
        else:
            console.print(f"\n[bold red]WSL training failed:[/] {wsl_result.get('error', 'unknown error')}")
        return

    # ── Local path (engine runs in-process) ──────────────────────────────
    config = TrainingConfig(
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        max_episode_steps=_ep_steps,
    )
    engine = MuJoCoMJXEngine()
    engine.setup(robot_path.read_text(encoding="utf-8"), "", config)

    # Patch engine methods to capture algorithm name before they run
    _orig_brax = engine._train_brax
    _orig_sb3 = engine._train_sb3
    _orig_random = engine._train_random_rollout

    def _patched_brax(jrf=None, jdf=None, cb=None):
        state["algorithm"] = "Brax PPO (GPU/JAX)"
        return _orig_brax(jrf, jdf, cb)

    def _patched_sb3(rf=None, df=None, cb=None):
        state["algorithm"] = "SB3 PPO (CPU)"
        return _orig_sb3(rf, df, cb)

    def _patched_random(cb=None):
        state["algorithm"] = "Random rollout (no learning)"
        return _orig_random(cb)

    engine._train_brax = _patched_brax
    engine._train_sb3 = _patched_sb3
    engine._train_random_rollout = _patched_random

    # Wire rollout streaming to the Isaac Sim bridge if connected.
    from robo_garden.isaac import get_bridge as _get_bridge_live
    _live_bridge = _get_bridge_live()
    _merged_for_rollout = robot_path.read_text(encoding="utf-8")

    def _rollout_cb(timestep: int, policy_apply) -> None:
        if not _live_bridge.connected:
            return
        try:
            from robo_garden.training.rollout import sample_rollout
            rollout = sample_rollout(
                _merged_for_rollout,
                policy_apply,
                num_frames=120,
                seed=int(timestep) & 0xFFFF,
            )
            if rollout.qpos.shape[0] > 0:
                _live_bridge.stream_qpos_batch(robot_name, rollout.qpos, list(rollout.timesteps))
        except Exception as exc:
            pass  # best-effort — never interrupt training for a rollout failure

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        def _live_callback(step: int, metrics: dict) -> None:
            _on_step(step, metrics)
            live.update(_make_panel())

        result = engine.train(
            "",
            reward_fn=reward_fn,
            done_fn=done_fn,
            jax_reward_fn=jax_reward_fn,
            jax_done_fn=jax_done_fn,
            callback=_live_callback,
            rollout_callback=_rollout_cb if _live_bridge.connected else None,
        )
        engine.cleanup()
        live.update(_make_panel())

    console.print(
        f"\n[bold green]Done![/]  "
        f"best_reward=[bold]{result.best_reward:.3f}[/]  "
        f"time=[bold]{result.training_time_seconds:.1f}s[/]  "
        f"checkpoint=[dim]{result.checkpoint_path}[/]"
    )

    # Write rollout frames so the Windows side can stream them to Isaac Sim.
    # Only relevant when running as a WSL worker (wsl_run_id is set).
    if wsl_run_id:
        try:
            import numpy as np
            from robo_garden.config import WORKSPACE_DIR
            from robo_garden.training.rollout import sample_rollout

            job_dir = WORKSPACE_DIR / "_wsl_jobs" / wsl_run_id
            job_dir.mkdir(parents=True, exist_ok=True)
            rollout = sample_rollout(
                robot_path.read_text(encoding="utf-8"),
                None,  # zero-action rollout shows passive dynamics
                num_frames=150,
                seed=42,
            )
            if rollout.qpos.shape[0] > 0:
                np.savez(
                    str(job_dir / "rollout_frames.npz"),
                    qpos=rollout.qpos,
                    timesteps=np.array(rollout.timesteps),
                )
        except Exception as exc:
            pass  # best-effort — don't crash training over a rollout write failure


# ---------------------------------------------------------------------------
# WSL2 path helper (used by wsl_dispatch and _run_training_live)
# ---------------------------------------------------------------------------

def _windows_to_wsl_path(p: str) -> str:
    """Convert ``C:\\Users\\...`` to ``/mnt/c/Users/...`` for WSL2."""
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _run_wsl_worker(job_dir: Path) -> None:
    """Linux-side worker: run a staged training job and write result.json.

    Invoked as a subprocess by ``wsl_dispatch.run_in_wsl`` on the Windows
    side.  Reads the job spec dumped into ``job_dir/job.json``, compiles
    the reward function for both NumPy (SB3 fallback) and JAX (Brax GPU),
    runs ``MuJoCoMJXEngine.train`` with real progress emission, and
    finally persists a result dict to ``job_dir/result.json``.

    All communication back to the Windows-side caller happens via:
      * JSONL progress lines on stdout  (prefix ``__RG_PROGRESS__``)
      * the final ``result.json`` file

    Everything else printed goes to stdout verbatim and is passed through
    to the Claude session's terminal, useful for dependency-import warnings
    and real Python tracebacks.
    """
    import time as _time
    from robo_garden.training import wsl_dispatch

    spec = wsl_dispatch.load_spec(job_dir)

    # Rewrite Windows absolute paths (C:/... or C:\...) in MJCF XML to
    # WSL equivalents (/mnt/c/...) so MuJoCo can open mesh/texture files
    # from inside the Linux environment.
    import re as _re

    def _rewire_xml_paths(xml: str) -> str:
        return _re.sub(
            r'([A-Za-z]):[/\\]([^"\'<>\s]+)',
            lambda m: f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}",
            xml,
        )

    spec.robot_xml = _rewire_xml_paths(spec.robot_xml)
    spec.env_mjcf = _rewire_xml_paths(spec.env_mjcf)

    print(f"WSL worker starting (run_id={spec.run_id}, "
          f"robot={spec.robot_name}, envs={spec.num_envs}, "
          f"timesteps={spec.total_timesteps:,})", flush=True)

    error_text = ""
    success = False
    best_reward = float("-inf")
    reward_curve: list[tuple[int, float]] = []
    checkpoint_path = ""
    started_at = _time.time()

    try:
        from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
        from robo_garden.training.models import TrainingConfig

        # Probe the actual observation / action dimensions from the robot
        # MJCF so the reward smoke-tests and JIT traces use real shapes.
        # Claude-generated rewards routinely include a defensive
        # ``if len(obs) < N: raise ValueError(...)`` — tracing that against
        # a 10-wide stand-in always fails and we'd wrongly disable the
        # Brax GPU path.
        obs_dim: int | None = None
        action_dim: int | None = None
        try:
            import mujoco as _mj
            _probe = _mj.MjModel.from_xml_string(spec.robot_xml)
            obs_dim = int(_probe.nq + _probe.nv)
            action_dim = int(_probe.nu)
            print(f"Probed MJCF dims: obs_dim={obs_dim}, action_dim={action_dim}", flush=True)
        except Exception as exc:
            print(f"WARN: could not probe MJCF dims ({exc}); smoke-testing "
                  f"with stand-in shapes (obs=10, action=4)", flush=True)

        # Compile Claude's reward source into a NumPy callable (for the SB3
        # fallback) and, best-effort, a JAX callable (unlocks the Brax/GPU
        # path). The JAX compile is allowed to fail — we just lose GPU.
        numpy_reward_fn = None
        jax_reward_fn = None
        if spec.reward_fn_code:
            try:
                from robo_garden.rewards.reward_runner import (
                    compile_reward_function,
                    safe_reward,
                )
                _np_raw = compile_reward_function(
                    spec.reward_fn_code,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )
                _np_safe = safe_reward(_np_raw, fallback=0.0)
                numpy_reward_fn = (
                    lambda obs, action, next_obs, _r=_np_safe:
                        float(_r(obs, action, next_obs, {})[0])
                )
            except Exception as exc:
                print(f"WARN: NumPy reward compile failed: {exc}", flush=True)

            try:
                from robo_garden.rewards.reward_runner import (
                    compile_jax_reward_function,
                )
                jax_reward_fn = compile_jax_reward_function(
                    spec.reward_fn_code,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                )
                print("JAX reward traced successfully; Brax/GPU path enabled.", flush=True)
            except Exception as exc:
                print(
                    f"NOTE: reward not JAX-traceable ({exc}); falling back to "
                    f"SB3/CPU. To use the Brax GPU path, rewrite the reward to "
                    f"use only numpy ops that jax.numpy supports and avoid "
                    f"Python-level 'if' on obs values (use jnp.where instead).",
                    flush=True,
                )

        config = TrainingConfig(
            algorithm=spec.algorithm,
            num_envs=spec.num_envs,
            total_timesteps=spec.total_timesteps,
            max_episode_steps=spec.max_episode_steps,
        )

        engine = MuJoCoMJXEngine()
        engine.setup(spec.robot_xml, spec.env_mjcf, config)

        def _progress(step: int, metrics: dict) -> None:
            nonlocal best_reward
            mean_r = float(metrics.get("eval/episode_reward", metrics.get("mean_reward", 0.0)))
            if mean_r > best_reward:
                best_reward = mean_r
            reward_curve.append((int(step), mean_r))
            emit_payload: dict = {
                "eval/episode_reward": mean_r,
                "mean_reward": mean_r,
                "best_reward": best_reward,
                "elapsed_s": _time.time() - started_at,
                "total_timesteps": config.total_timesteps,
            }
            if "_backend" in metrics:
                emit_payload["_backend"] = metrics["_backend"]
            wsl_dispatch.emit_progress(step, emit_payload)

        result = engine.train(
            reward_fn_code=spec.reward_fn_code,
            reward_fn=numpy_reward_fn,
            jax_reward_fn=jax_reward_fn,
            callback=_progress,
        )
        engine.cleanup()

        success = True
        best_reward = float(result.best_reward)
        reward_curve = list(result.reward_curve)
        checkpoint_path = str(result.checkpoint_path)
    except Exception as exc:
        import traceback
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"WSL worker FAILED: {error_text}", flush=True)
        traceback.print_exc()

    ended_at = _time.time()

    # Generate a short rollout for post-training viewport playback on Windows.
    # Brax/JAX trained policies can't be easily reconstructed outside the
    # training context (inference fn not serialized), so we run a zero-action
    # rollout — this at least shows the robot's passive stability and joint
    # layout, confirming the physics are sane.
    rollout_frames_path = ""
    if success and spec.robot_xml:
        try:
            from robo_garden.training.rollout import sample_rollout
            from robo_garden.training.mujoco_engine import _merge_mjcf

            merged_for_rollout = _merge_mjcf(spec.robot_xml, spec.env_mjcf)
            rollout = sample_rollout(
                merged_for_rollout,
                None,  # zero-action policy — Brax params not serializable here
                num_frames=150,
                seed=42,
            )
            if rollout.qpos.shape[0] > 0:
                frames_path = job_dir / "rollout_frames.npz"
                np.savez(
                    str(frames_path),
                    qpos=rollout.qpos,
                    timesteps=np.array(rollout.timesteps, dtype=np.float32),
                )
                rollout_frames_path = str(frames_path)
                print(f"Rollout frames saved ({len(rollout.qpos)} frames)", flush=True)
        except Exception as exc:
            print(f"WARN: could not generate rollout frames: {exc}", flush=True)

    wsl_dispatch.write_result(job_dir, {
        "success": success,
        "best_reward": best_reward,
        "reward_curve": reward_curve,
        "checkpoint_path": checkpoint_path,
        "training_time_seconds": ended_at - started_at,
        "error": error_text,
        "rollout_frames_path": rollout_frames_path,
    })
    print(
        f"WSL worker finished "
        f"(success={success}, best_reward={best_reward:.3f}, "
        f"time={ended_at - started_at:.1f}s)",
        flush=True,
    )


def _run_gym(args) -> None:
    """Training Gym: load an approved manifest and launch a chat session with
    training tools unlocked.

    Reuses the terminal chat UI but pre-seeds Session.phase = "training" so
    generate_reward + train are advertised to Claude from the first turn.
    """
    from robo_garden.claude.session import Session, _resolve_prompt, _send_with_spinner
    from robo_garden.studio import load_approved_manifest
    from rich.console import Console
    from rich.rule import Rule

    console = Console()

    if not args.approved:
        console.print(
            "[bold red]Error:[/] --mode gym requires --approved <manifest-name>. "
            "Approve a design in --mode studio first, then pass the manifest "
            "filename (without .json) from workspace/approved/."
        )
        sys.exit(1)

    try:
        manifest = load_approved_manifest(args.approved)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    robot_name = manifest["robot_name"]
    env_name = manifest["environment_name"]
    robot_path = manifest.get("robot_path", "")

    console.print(
        f"[bold green]Training Gym[/] — approved design: "
        f"[bold]{robot_name}[/] + [bold]{env_name}[/]"
    )
    console.print(
        f"[dim]Manifest approved {manifest.get('approved_at_utc', '?')}[/]"
    )
    console.print(
        "[dim]All training tools (generate_reward, train) are unlocked. "
        "Type 'quit' to exit.[/]\n"
    )

    # Push the approved robot into the Isaac Sim viewport at startup so the
    # user sees something instead of an empty stage.  Studio mode does this
    # via _on_tool_side_effects; gym mode skips the design tools entirely so
    # we have to drive it directly here from the manifest.
    from pathlib import Path as _Path
    from robo_garden.isaac import get_bridge as _get_bridge

    if robot_path and _Path(robot_path).exists():
        _bridge = _get_bridge()
        if _bridge.connected:
            fmt = "urdf" if robot_path.lower().endswith(".urdf") else "mjcf"
            try:
                _bridge.send_robot(robot_name, _Path(robot_path), fmt=fmt)
                console.print(
                    f"[dim]Mirrored approved robot to Isaac Sim viewport "
                    f"({robot_name}, {fmt}).[/]"
                )
            except Exception as exc:  # pragma: no cover — defensive
                console.print(
                    f"[yellow]Warning:[/] could not push robot to Isaac Sim: {exc}"
                )
        else:
            console.print(
                "[dim]Isaac Sim bridge not connected — viewport will stay empty. "
                "Start isaac_server first to see the robot.[/]"
            )
    else:
        console.print(
            f"[yellow]Warning:[/] approved manifest has no usable robot_path "
            f"({robot_path!r}); viewport will be empty."
        )

    from robo_garden.studio import _build_training_context
    _gym_context = _build_training_context(args.timesteps, args.envs)
    session = Session(phase="training", enable_viewer=False, extra_context=_gym_context)
    session.approved_robot = robot_name
    session.approved_environment = env_name

    initial = None
    if args.prompt_file:
        try:
            initial = _resolve_prompt(f"@{args.prompt_file}")
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            sys.exit(1)

    try:
        if initial:
            preview = initial[:120] + ("..." if len(initial) > 120 else "")
            console.print(f"[bold]You:[/bold] {preview}")
            response = _send_with_spinner(session, initial, on_tool_result=None)
            console.print()
            console.print(Rule(style="dim"))
            console.print(f"[bold green]Claude:[/bold green] {response}")
            console.print(Rule(style="dim"))
            console.print()

        while True:
            try:
                user_input = input("Gym: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            try:
                user_input = _resolve_prompt(user_input)
            except FileNotFoundError as exc:
                console.print(f"[bold red]Error:[/] {exc}\n")
                continue
            response = _send_with_spinner(session, user_input, on_tool_result=None)
            console.print()
            console.print(Rule(style="dim"))
            console.print(f"[bold green]Claude:[/bold green] {response}")
            console.print(Rule(style="dim"))
            console.print()
    finally:
        session.close()


def _try_connect_bridge(url: str) -> None:
    """Attempt to connect the Isaac Sim bridge, print one-line status."""
    from robo_garden.isaac import get_bridge
    bridge = get_bridge()
    connected = bridge.connect(url)
    if connected:
        print(f"Isaac Sim bridge connected at {url}")
    else:
        print("Isaac Sim bridge not available (running without 3D visualization)")


# Track auto-spawned Isaac Sim subprocess so we can opt-in to clean it up
# at exit (left running by default — Isaac Sim takes 30-60 s to boot, so
# users typically prefer to keep it warm across multiple invocations).
_isaac_proc = None


def _isaac_port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Quick TCP probe — True iff something is listening on (host, port)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def _parse_ws_url(url: str) -> tuple[str, int]:
    """Extract (host, port) from a ws:// or wss:// URL with sensible defaults."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "wss" else 8765)
    return host, port


def _ensure_isaac_server_running(
    url: str,
    *,
    venv_path: str = "C:/isaac-venv",
    boot_timeout_seconds: float = 180.0,
) -> bool:
    """Make sure an Isaac Sim server is reachable at *url*; spawn one if not.

    Returns True iff the port is open by the time we return (regardless of
    whether we spawned the process ourselves or it was already there).

    On Windows this opens Isaac Sim in a new console window so its log output
    stays visible to the user.  On other platforms we currently just print a
    helpful message and return False — the auto-launch path is Windows-only
    today because Isaac Sim ships as a Windows venv at ``C:/isaac-venv``.
    """
    global _isaac_proc

    host, port = _parse_ws_url(url)

    if _isaac_port_is_open(host, port):
        print(f"Isaac Sim server already running at {url}")
        return True

    project_root = Path(__file__).parent.parent.parent
    launcher = project_root / "isaac_server" / "launch.ps1"

    if sys.platform != "win32" or not launcher.exists():
        print(
            "Isaac Sim auto-launch is currently Windows-only.\n"
            f"  Start it manually: {launcher}\n"
            "  Then re-run this command (it will reconnect on the next boot)."
        )
        return False

    print(f"Isaac Sim not running — launching {launcher.name} ...")
    print("(first boot can take 30-60 seconds; the new console will show progress)")

    try:
        new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        _isaac_proc = subprocess.Popen(
            [
                "powershell",
                "-NoExit",
                "-ExecutionPolicy", "Bypass",
                "-File", str(launcher),
                "-VenvPath", venv_path,
                "-Port", str(port),
            ],
            creationflags=new_console,
            cwd=str(project_root),
        )
    except Exception as exc:
        print(f"Failed to spawn Isaac Sim: {exc}")
        return False

    # Poll the port until the server starts accepting connections, with a
    # progress dot every 2 s so the user knows we haven't hung.
    import time as _time

    deadline = _time.monotonic() + boot_timeout_seconds
    last_dot = 0.0
    while _time.monotonic() < deadline:
        if _isaac_port_is_open(host, port, timeout=0.3):
            # The TCP socket can bind a fraction of a second before the WS
            # handler is ready to accept upgrades.  Give it a small grace
            # period so the very first bridge.connect() doesn't bounce.
            _time.sleep(1.0)
            print(f"\nIsaac Sim server ready at {url}")
            return True
        if _isaac_proc.poll() is not None:
            print(
                f"\nIsaac Sim subprocess exited early "
                f"(code {_isaac_proc.returncode}). Check the new console for errors."
            )
            return False
        now = _time.monotonic()
        if now - last_dot >= 2.0:
            print(".", end="", flush=True)
            last_dot = now
        _time.sleep(0.25)

    print(
        f"\nIsaac Sim did not start within {boot_timeout_seconds:.0f}s. "
        "Check the new console window for errors."
    )
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Robo Garden: Claude-powered robot creation studio"
    )
    parser.add_argument(
        "--mode",
        choices=["tui", "chat", "train", "studio", "gym"],
        default="chat",
        help=(
            "Run mode: "
            "tui (full textual interface), "
            "chat (terminal Claude conversation), "
            "train (raw training run for a known robot), "
            "studio (Design Studio — requires Isaac Sim server), "
            "gym (Training Gym — load an approved manifest, unlock training tools)."
        ),
    )
    parser.add_argument("--robot", help="Robot name (for train mode) or path to robot MJCF file")
    parser.add_argument("--env", help="Path to environment MJCF file")
    parser.add_argument(
        "--approved",
        metavar="MANIFEST",
        help=(
            "Name (or path) of an approval manifest in workspace/approved/. "
            "Required for --mode gym; enables generate_reward + train."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=64,
        help="Number of parallel environments (default: 64)",
    )
    parser.add_argument(
        "--no-isaac",
        action="store_true",
        help="Disable Isaac Sim bridge (skip connection attempt)",
    )
    parser.add_argument(
        "--no-auto-isaac",
        action="store_true",
        help=(
            "Don't auto-launch the Isaac Sim server in --mode studio/gym "
            "(assume the user started it externally)"
        ),
    )
    parser.add_argument(
        "--isaac-venv",
        default="C:/isaac-venv",
        metavar="PATH",
        help=(
            "Path to the Isaac Sim Python venv used by the auto-launcher "
            "(Windows only; default: C:/isaac-venv)"
        ),
    )
    parser.add_argument(
        "--wsl",
        action="store_true",
        help=(
            "Run training inside WSL2 for JAX/MJX GPU acceleration "
            "(Windows only — streams output back to this terminal)"
        ),
    )
    parser.add_argument(
        "--wsl-worker",
        metavar="JOB_DIR",
        help=(
            "INTERNAL. Linux-side training worker for Claude-driven runs "
            "dispatched via ROBO_GARDEN_TRAIN_IN_WSL; reads job.json from "
            "JOB_DIR and writes result.json when done."
        ),
    )
    parser.add_argument(
        "--wsl-run-id",
        metavar="RUN_ID",
        help=(
            "INTERNAL. Run ID for this training session; the Linux worker "
            "writes rollout_frames.npz to workspace/_wsl_jobs/<RUN_ID>/."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        metavar="FILE",
        help="Send file contents as the opening message (relative paths resolve from workspace/prompts/)",
    )

    args = parser.parse_args()

    # Attempt Isaac Sim bridge connection unless disabled.  Studio mode owns
    # the bridge itself (sets up message handlers before connect), so skip the
    # pre-connect there to avoid a duplicate singleton connection.
    from robo_garden.config import ISAAC_BRIDGE_URL, ISAAC_BRIDGE_ENABLED

    # Auto-launch the Isaac Sim server for the modes that depend on a live
    # 3D viewport (studio, gym).  Skipped if --no-isaac, --no-auto-isaac, or
    # the bridge is globally disabled.  Block until the port is open so the
    # subsequent bridge connect/_run_gym dispatch sees a live server.
    if (
        not args.no_isaac
        and not args.no_auto_isaac
        and ISAAC_BRIDGE_ENABLED != "off"
        and args.mode in ("studio", "gym", "train")
    ):
        _ensure_isaac_server_running(
            ISAAC_BRIDGE_URL,
            venv_path=args.isaac_venv,
        )

    if (
        not args.no_isaac
        and ISAAC_BRIDGE_ENABLED != "off"
        and args.mode not in ("studio",)
    ):
        _try_connect_bridge(ISAAC_BRIDGE_URL)

    if args.mode == "tui":
        from robo_garden.tui.app import RoboGardenApp
        app = RoboGardenApp()
        app.run()
    elif args.mode == "chat":
        from robo_garden.claude.session import run_chat, _resolve_prompt
        initial = None
        if args.prompt_file:
            try:
                initial = _resolve_prompt(f"@{args.prompt_file}")
            except FileNotFoundError as exc:
                print(f"Error: {exc}")
                sys.exit(1)
        run_chat(initial_prompt=initial)
    elif args.mode == "studio":
        from robo_garden.studio import run_studio
        from robo_garden.claude.session import _resolve_prompt
        seed = None
        if args.prompt_file:
            try:
                seed = _resolve_prompt(f"@{args.prompt_file}")
            except FileNotFoundError as exc:
                print(f"Error: {exc}")
                sys.exit(1)
        run_studio(
            isaac_url=ISAAC_BRIDGE_URL,
            initial_prompt=seed,
            training_timesteps=args.timesteps,
            training_num_envs=args.envs,
        )
    elif args.mode == "gym":
        _run_gym(args)
    elif args.mode == "train":
        # Internal: Linux-side worker for Claude-driven gym-mode runs.
        # Branches *before* the --robot check because the worker takes its
        # robot/env/reward directly from job.json instead of CLI flags.
        if args.wsl_worker:
            _run_wsl_worker(Path(args.wsl_worker))
            return

        robot_name = args.robot
        if not robot_name:
            print("Error: --mode train requires --robot <name>")
            sys.exit(1)

        from robo_garden.config import ROBOTS_DIR

        robot_path = ROBOTS_DIR / f"{robot_name}.xml"
        if not robot_path.exists():
            robot_path = ROBOTS_DIR / f"{robot_name}.urdf"
        if not robot_path.exists():
            print(f"Error: robot '{robot_name}' not found in workspace. Generate it first.")
            sys.exit(1)

        # Load robot into Isaac Sim viewport before training starts so the
        # user can see the robot and watch rollout frames stream in.
        from robo_garden.isaac import get_bridge as _get_bridge
        _train_bridge = _get_bridge()
        if _train_bridge.connected:
            fmt = "urdf" if str(robot_path).endswith(".urdf") else "mjcf"
            _train_bridge.send_robot(robot_name, robot_path, fmt=fmt)

        # Built-in reward functions for known robots.  Without one of these
        # (and no Claude-generated reward in this mode), MJXBraxEnv falls
        # back to a pure control-effort penalty whose optimum is ``action=0``
        # → robot collapses. See brax_env.py for the default.
        from robo_garden.training.gym_env import BUILTIN_REWARD_SOURCE
        reward_fn = None
        done_fn = None
        jax_reward_fn = None
        jax_done_fn = None
        locomotion_horizon = None
        if robot_name == "cartpole":
            from robo_garden.training.gym_env import (
                cartpole_reward,
                cartpole_reward_jax,
                cartpole_done_jax,
            )
            reward_fn = cartpole_reward
            jax_reward_fn = cartpole_reward_jax
            jax_done_fn = cartpole_done_jax
        elif robot_name == "go2_walker":
            from robo_garden.training.gym_env import (
                go2_walker_reward,
                go2_walker_done,
                go2_walker_reward_jax,
                go2_walker_done_jax,
            )
            reward_fn = go2_walker_reward
            done_fn = go2_walker_done
            jax_reward_fn = go2_walker_reward_jax
            jax_done_fn = go2_walker_done_jax
            # Locomotion needs multi-second episodes for a gait to emerge; 500
            # steps × dt=0.002 = 1 s is too short for walking policies.  5 s
            # gives the policy time to push the trunk forward several body
            # lengths before truncation.
            locomotion_horizon = 2500
        elif robot_name == "urchin_v2":
            from robo_garden.training.gym_env import (
                urchin_v2_reward,
                urchin_v2_reward_jax,
            )
            reward_fn = urchin_v2_reward
            jax_reward_fn = urchin_v2_reward_jax
            # Same rationale as go2_walker: rolling a 30 cm ball from rest
            # takes a couple of seconds of coordinated voice-coil activity,
            # and 1 s episodes truncate before the policy ever sees motion.
            locomotion_horizon = 2500

        # ``--approved`` is consumed by --mode gym; in --mode train we only use
        # it as a context string for logging so users are not silently confused
        # when they pass it and it has no effect on the reward function.
        if getattr(args, "approved", None):
            print(
                f"Note: --approved '{args.approved}' is informational in "
                f"--mode train; reward function comes from the built-in "
                f"for robot '{robot_name}'."
            )

        # Auto-detect WSL2 GPU availability.  --wsl forces it; without it,
        # wsl_dispatch.is_enabled() checks for wsl.exe + Ubuntu-22.04 and
        # returns True automatically when WSL2 is set up.  No explicit --wsl
        # flag needed — GPU is preferred whenever it's available.
        from robo_garden.training import wsl_dispatch as _wsl
        _use_wsl = bool(args.wsl) or _wsl.is_enabled()

        _run_training_live(
            robot_name,
            robot_path,
            args.timesteps,
            args.envs,
            reward_fn=reward_fn,
            done_fn=done_fn,
            jax_reward_fn=jax_reward_fn,
            jax_done_fn=jax_done_fn,
            max_episode_steps=locomotion_horizon,
            wsl_run_id=getattr(args, "wsl_run_id", None),
            reward_fn_code=BUILTIN_REWARD_SOURCE.get(robot_name, ""),
            use_wsl=_use_wsl,
        )


if __name__ == "__main__":
    main()
