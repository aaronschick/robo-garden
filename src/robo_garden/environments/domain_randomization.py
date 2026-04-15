"""Domain randomization: randomize physics parameters per episode for robust training."""

from __future__ import annotations

import mujoco
import numpy as np

from robo_garden.environments.models import DomainRandomizationConfig


def randomize_model(
    model: mujoco.MjModel,
    config: DomainRandomizationConfig,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply domain randomization to a MuJoCo model in-place.

    Call this at the start of each episode to vary physics parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    if config.friction_range is not None:
        lo, hi = config.friction_range
        for i in range(model.ngeom):
            model.geom_friction[i, 0] = rng.uniform(lo, hi)

    if config.mass_scale_range is not None:
        lo, hi = config.mass_scale_range
        scale = rng.uniform(lo, hi)
        for i in range(model.nbody):
            model.body_mass[i] *= scale

    if config.gravity_range is not None:
        lo, hi = config.gravity_range
        model.opt.gravity[2] = rng.uniform(lo, hi)

    if config.actuator_strength_range is not None:
        lo, hi = config.actuator_strength_range
        for i in range(model.nu):
            model.actuator_gear[i, 0] *= rng.uniform(lo, hi)
