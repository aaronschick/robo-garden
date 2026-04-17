"""Simulate mode controller — load any robot and drive it interactively.

Phase 4: activating/deactivating Simulate mode starts/stops the LivePolicyPlayer
if one is loaded.  Phase 5 will add gamepad input.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class SimulateModeController:
    name = "simulate"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Simulate mode activated")
        # Seed skill list so the panel is populated on first switch
        studio._seed_skill_list()

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Simulate mode deactivated")
        # Stop live playback when leaving Simulate mode
        if studio._live_player is not None:
            studio._live_player.stop()
            studio._live_player = None

    def handle_message(self, msg: dict) -> bool:
        return False
