"""Design mode controller — Claude-driven MJCF authoring (current Studio behavior).

For Phase 1 this is a thin pass-through: all Design logic lives in Studio.
Future phases may move more behavior here as the mode system matures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class DesignModeController:
    name = "design"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Design mode activated")

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Design mode deactivated")

    def handle_message(self, msg: dict) -> bool:
        return False  # Studio's default dispatch handles all Design messages
