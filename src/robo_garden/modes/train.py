"""Train mode controller — run and monitor training jobs.

Phase 1 stub.  Phase 7 will add the omni.ui.Plot reward curve,
per-component breakdown panel, and mid-training rollout previews here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class TrainModeController:
    name = "train"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Train mode activated")

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Train mode deactivated")

    def handle_message(self, msg: dict) -> bool:
        return False
