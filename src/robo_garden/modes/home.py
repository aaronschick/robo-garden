"""Home mode controller — launch screen, no active subsystems."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class HomeModeController:
    name = "home"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Home mode activated")

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Home mode deactivated")

    def handle_message(self, msg: dict) -> bool:
        return False
