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
    jax_reward_fn: Callable | None = None,
    jax_done_fn: Callable | None = None,
) -> None:
    """Run training with a Rich Live terminal display.

    On Linux/WSL2 (JAX GPU available), ``jax_reward_fn`` / ``jax_done_fn`` are
    forwarded to the Brax PPO path.  On Windows, ``reward_fn`` (numpy) is used
    by the SB3 PPO fallback.
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

    config = TrainingConfig(
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        max_episode_steps=500,
    )
    engine = MuJoCoMJXEngine()
    engine.setup(robot_path.read_text(encoding="utf-8"), "", config)

    state: dict = {
        "step": 0,
        "mean_reward": 0.0,
        "best_reward": float("-inf"),
        "reward_history": [],
        "algorithm": "starting…",
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
        state["step"] = step
        state["mean_reward"] = mean_r
        if mean_r > state["best_reward"]:
            state["best_reward"] = mean_r
        state["reward_history"].append(mean_r)
        progress.update(task_id, completed=min(step, total_timesteps))

    # Patch engine methods to capture algorithm name before they run
    _orig_brax = engine._train_brax
    _orig_sb3 = engine._train_sb3
    _orig_random = engine._train_random_rollout

    def _patched_brax(jrf=None, jdf=None, cb=None):
        state["algorithm"] = "Brax PPO (GPU/JAX)"
        return _orig_brax(jrf, jdf, cb)

    def _patched_sb3(rf=None, cb=None):
        state["algorithm"] = "SB3 PPO (CPU)"
        return _orig_sb3(rf, cb)

    def _patched_random(cb=None):
        state["algorithm"] = "Random rollout (no learning)"
        return _orig_random(cb)

    engine._train_brax = _patched_brax
    engine._train_sb3 = _patched_sb3
    engine._train_random_rollout = _patched_random

    console.print(
        f"\n[bold cyan]Robo Garden[/] — training [bold]{robot_name}[/] "
        f"for [bold]{total_timesteps:,}[/] timesteps\n"
    )

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        def _live_callback(step: int, metrics: dict) -> None:
            _on_step(step, metrics)
            live.update(_make_panel())

        result = engine.train(
            "",
            reward_fn=reward_fn,
            jax_reward_fn=jax_reward_fn,
            jax_done_fn=jax_done_fn,
            callback=_live_callback,
        )
        engine.cleanup()
        live.update(_make_panel())

    console.print(
        f"\n[bold green]Done![/]  "
        f"best_reward=[bold]{result.best_reward:.3f}[/]  "
        f"time=[bold]{result.training_time_seconds:.1f}s[/]  "
        f"checkpoint=[dim]{result.checkpoint_path}[/]"
    )


# ---------------------------------------------------------------------------
# WSL2 launch (Windows → Linux GPU training)
# ---------------------------------------------------------------------------

def _windows_to_wsl_path(p: str) -> str:
    """Convert ``C:\\Users\\...`` to ``/mnt/c/Users/...`` for WSL2."""
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _launch_wsl_training(robot_name: str, timesteps: int, num_envs: int) -> None:
    """Run training inside WSL2, streaming output back to this terminal.

    Translates the Windows project path to its WSL2 ``/mnt/…`` equivalent,
    then invokes ``wsl bash -c "…"`` as a subprocess so the Rich live display
    renders inside the Windows Terminal.
    """
    import subprocess
    from robo_garden.config import PROJECT_ROOT

    wsl_path = _windows_to_wsl_path(str(PROJECT_ROOT))

    # Source profile so uv / cargo-installed binaries are on PATH in the
    # non-interactive shell that wsl.exe spawns.
    source_profile = (
        'source "$HOME/.profile" 2>/dev/null; '
        'source "$HOME/.cargo/env" 2>/dev/null; '
        'export PATH="$HOME/.local/bin:$PATH"; '
    )
    train_cmd = (
        f'cd \'{wsl_path}\' && '
        f'uv run robo-garden --no-isaac --mode train --robot {robot_name} '
        f'--timesteps {timesteps} --envs {num_envs}'
    )

    full_cmd = source_profile + train_cmd
    print(f"Launching WSL2 training: {train_cmd}\n")

    try:
        proc = subprocess.Popen(
            ["wsl", "bash", "-c", full_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        print("Error: wsl.exe not found. Is WSL2 installed?")
        print("Install it with:  wsl --install -d Ubuntu-22.04")
        sys.exit(1)

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


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

    session = Session(phase="training", enable_viewer=False)
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
        and args.mode in ("studio", "gym")
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
        run_studio(isaac_url=ISAAC_BRIDGE_URL, initial_prompt=seed)
    elif args.mode == "gym":
        _run_gym(args)
    elif args.mode == "train":
        robot_name = args.robot
        if not robot_name:
            print("Error: --mode train requires --robot <name>")
            sys.exit(1)

        # --wsl: delegate to WSL2 and stream output back (Windows only)
        if args.wsl:
            _launch_wsl_training(robot_name, args.timesteps, args.envs)
            return

        from robo_garden.config import ROBOTS_DIR

        robot_path = ROBOTS_DIR / f"{robot_name}.xml"
        if not robot_path.exists():
            robot_path = ROBOTS_DIR / f"{robot_name}.urdf"
        if not robot_path.exists():
            print(f"Error: robot '{robot_name}' not found in workspace. Generate it first.")
            sys.exit(1)

        # Built-in reward functions for known robots
        reward_fn = None
        jax_reward_fn = None
        jax_done_fn = None
        if robot_name == "cartpole":
            from robo_garden.training.gym_env import (
                cartpole_reward,
                cartpole_reward_jax,
                cartpole_done_jax,
            )
            reward_fn = cartpole_reward
            jax_reward_fn = cartpole_reward_jax
            jax_done_fn = cartpole_done_jax

        _run_training_live(
            robot_name,
            robot_path,
            args.timesteps,
            args.envs,
            reward_fn=reward_fn,
            jax_reward_fn=jax_reward_fn,
            jax_done_fn=jax_done_fn,
        )


if __name__ == "__main__":
    main()
