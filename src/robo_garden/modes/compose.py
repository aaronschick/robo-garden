"""Policy Composer mode controller — combine skills into named policy versions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class ComposeModeController:
    name = "compose"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Compose mode activated")
        # Refresh UI lists so the composer has current data
        studio._broadcast_skill_list()
        studio._broadcast_policy_list()

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Compose mode deactivated")
        # Stop any policy test running from the composer
        if studio._live_player is not None:
            studio._live_player.stop()
            studio._live_player = None

    def handle_message(self, msg: dict) -> bool:
        return False
