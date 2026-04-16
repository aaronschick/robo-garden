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

    def get_env_params(self, stage_index: int) -> dict:
        """Return physics parameter overrides for a given curriculum stage."""
        if not self.config.stages or stage_index >= len(self.config.stages):
            return {}
        stage = self.config.stages[stage_index]
        difficulty = stage_index / max(len(self.config.stages) - 1, 1)
        return {
            "gravity_scale": 1.0,
            "friction_range": (max(0.3, 1.0 - difficulty * 0.4), min(1.5, 1.0 + difficulty * 0.4)),
            "stage_name": stage.name,
            "difficulty": difficulty,
        }
