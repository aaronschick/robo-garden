"""Skills Library mode controller — browse and play back trained behaviors.

Phase 1 stub.  Phase 3 adds the skills data model and promote flow;
Phase 4 adds live viewport playback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_garden.studio import Studio

log = logging.getLogger(__name__)


class SkillsModeController:
    name = "skills"

    def activate(self, studio: "Studio", context: dict) -> None:
        log.debug("Skills mode activated")

    def deactivate(self, studio: "Studio") -> None:
        log.debug("Skills mode deactivated")

    def handle_message(self, msg: dict) -> bool:
        return False
