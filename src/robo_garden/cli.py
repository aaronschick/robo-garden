"""Main entry point for the Robo Garden application."""

import argparse
import sys


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
    parser.add_argument("--robot", help="Path to robot MJCF file")
    parser.add_argument("--env", help="Path to environment MJCF file")

    args = parser.parse_args()

    if args.mode == "tui":
        from robo_garden.tui.app import RoboGardenApp
        app = RoboGardenApp()
        app.run()
    elif args.mode == "chat":
        from robo_garden.claude.session import run_chat
        run_chat()
    elif args.mode == "train":
        print("Training mode not yet implemented. Use chat mode to set up training via Claude.")
        sys.exit(1)


if __name__ == "__main__":
    main()
