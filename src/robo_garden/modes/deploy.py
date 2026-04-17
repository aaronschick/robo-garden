"""Deploy / Export mode controller — export checkpoints for deployment.

Phase 1 stub.  Phase 8 adds ONNX export for SB3 skills and param/XML
bundles for Brax skills.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class DeployModeController:
    name = "deploy"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Deploy mode activated")

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Deploy mode deactivated")

    def handle_message(self, msg: dict) -> bool:
        return False
