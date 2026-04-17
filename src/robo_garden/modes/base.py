"""ModeController protocol and mode registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from robo_garden.studio import Studio

AVAILABLE_MODES: list[str] = [
    "home",
    "design",
    "simulate",
    "train",
    "skills",
    "compose",
    "deploy",
]


@runtime_checkable
class ModeController(Protocol):
    """Protocol every activity-mode controller must satisfy.

    activate/deactivate wire and unwire subsystems (e.g. live player,
    gamepad runner) as the user switches modes.  handle_message lets each
    mode intercept WebSocket messages before Studio's default dispatch;
    return True to mark the message consumed.
    """

    name: str

    def activate(self, studio: "Studio", context: dict) -> None: ...
    def deactivate(self, studio: "Studio") -> None: ...
    def handle_message(self, msg: dict) -> bool: ...
