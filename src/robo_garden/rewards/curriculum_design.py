"""Claude-assisted curriculum stage design for progressive training difficulty.

TODO (Phase 7): Implement Claude-driven curriculum generation.
"""

from __future__ import annotations

from robo_garden.training.models import CurriculumConfig, CurriculumStage


def create_default_locomotion_curriculum() -> CurriculumConfig:
    """Create a default curriculum for locomotion training."""
    return CurriculumConfig(
        stages=[
            CurriculumStage(
                name="stand",
                environment_params={"terrain_type": "flat"},
                max_timesteps=200_000,
            ),
            CurriculumStage(
                name="walk_flat",
                environment_params={"terrain_type": "flat", "target_velocity": 0.5},
                max_timesteps=500_000,
            ),
            CurriculumStage(
                name="walk_rough",
                environment_params={"terrain_type": "rough", "roughness": 0.03},
                max_timesteps=500_000,
            ),
            CurriculumStage(
                name="walk_stairs",
                environment_params={"terrain_type": "stairs", "step_height": 0.05},
                max_timesteps=500_000,
            ),
        ],
        advance_threshold=0.8,
    )
