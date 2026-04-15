"""Curriculum learning: progressive difficulty for robust training."""

from __future__ import annotations

import logging

from robo_garden.training.models import CurriculumConfig

log = logging.getLogger(__name__)


class CurriculumManager:
    """Manages training curriculum, advancing difficulty based on performance.

    TODO (Phase 7): Implement curriculum advancement with:
    - Performance threshold checking
    - Environment parameter modification
    - Stage transition logging
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_stage = 0

    @property
    def current_stage_config(self):
        if self.current_stage < len(self.config.stages):
            return self.config.stages[self.current_stage]
        return None

    def should_advance(self, metric_value: float) -> bool:
        """Check if performance warrants advancing to next stage."""
        return metric_value >= self.config.advance_threshold

    def advance(self) -> bool:
        """Advance to the next curriculum stage. Returns True if advanced."""
        if self.current_stage < len(self.config.stages) - 1:
            self.current_stage += 1
            log.info(f"Curriculum advanced to stage {self.current_stage}: "
                     f"{self.config.stages[self.current_stage].name}")
            return True
        return False
