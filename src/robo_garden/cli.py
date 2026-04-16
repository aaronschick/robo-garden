"""Main entry point for the Robo Garden application."""

import argparse
import sys


def _try_connect_bridge(url: str) -> None:
    """Attempt to connect the Isaac Sim bridge, print one-line status."""
    from robo_garden.isaac import get_bridge
    bridge = get_bridge()
    connected = bridge.connect(url)
    if connected:
        print(f"Isaac Sim bridge connected at {url}")
    else:
        print("Isaac Sim bridge not available (running without 3D visualization)")


def main():
    parser = argparse.ArgumentParser(
        description="Robo Garden: Claude-powered robot creation studio"
    )
    parser.add_argument(
        "--mode",
        choices=["tui", "chat", "train"],
        default="chat",
        help="Run mode: tui (full interface), chat (Claude conversation), train (run training)",
    )
    parser.add_argument("--robot", help="Robot name (for train mode) or path to robot MJCF file")
    parser.add_argument("--env", help="Path to environment MJCF file")
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
        "--prompt-file",
        metavar="FILE",
        help="Send file contents as the opening message (relative paths resolve from workspace/prompts/)",
    )

    args = parser.parse_args()

    # Attempt Isaac Sim bridge connection unless disabled
    from robo_garden.config import ISAAC_BRIDGE_URL, ISAAC_BRIDGE_ENABLED
    if not args.no_isaac and ISAAC_BRIDGE_ENABLED != "off":
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
    elif args.mode == "train":
        robot_name = args.robot
        if not robot_name:
            print("Error: --mode train requires --robot <name>")
            sys.exit(1)

        from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
        from robo_garden.training.models import TrainingConfig
        from robo_garden.config import ROBOTS_DIR

        robot_path = ROBOTS_DIR / f"{robot_name}.xml"
        if not robot_path.exists():
            robot_path = ROBOTS_DIR / f"{robot_name}.urdf"
        if not robot_path.exists():
            print(f"Error: robot '{robot_name}' not found in workspace. Generate it first.")
            sys.exit(1)

        print(f"Training {robot_name} for {args.timesteps:,} timesteps with {args.envs} envs...")

        engine = MuJoCoMJXEngine()
        config = TrainingConfig(
            num_envs=args.envs,
            total_timesteps=args.timesteps,
            max_episode_steps=1000,
        )
        engine.setup(robot_path.read_text(encoding="utf-8"), "", config)

        def progress(step, metrics):
            mean_r = metrics.get("eval/episode_reward", metrics.get("mean_reward", 0))
            print(f"  step={step:>10,}  mean_reward={mean_r:>8.3f}")

        result = engine.train("", callback=progress)
        engine.cleanup()

        print(f"\nDone. best_reward={result.best_reward:.3f}  time={result.training_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
