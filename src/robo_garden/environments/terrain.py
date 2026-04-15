"""Terrain generation: heightfields, stairs, ramps, rough terrain."""

from __future__ import annotations

import numpy as np


def generate_heightfield(
    width: int = 100,
    length: int = 100,
    roughness: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a heightfield array using Perlin-like noise.

    Args:
        width: Grid width in cells.
        length: Grid length in cells.
        roughness: Height variation in meters.
        seed: Random seed for reproducibility.

    Returns:
        2D numpy array of height values.
    """
    rng = np.random.default_rng(seed)
    # Simple multi-octave noise approximation
    heightfield = np.zeros((length, width))
    for octave in range(4):
        scale = 2 ** octave
        freq = max(1, width // (8 * scale))
        noise = rng.random((length // freq + 2, width // freq + 2))
        # Bilinear upscale
        from numpy import interp
        x = np.linspace(0, noise.shape[1] - 1, width)
        y = np.linspace(0, noise.shape[0] - 1, length)
        xg, yg = np.meshgrid(x, y)
        xi = xg.astype(int).clip(0, noise.shape[1] - 2)
        yi = yg.astype(int).clip(0, noise.shape[0] - 2)
        xf = xg - xi
        yf = yg - yi
        upscaled = (
            noise[yi, xi] * (1 - xf) * (1 - yf)
            + noise[yi, xi + 1] * xf * (1 - yf)
            + noise[yi + 1, xi] * (1 - xf) * yf
            + noise[yi + 1, xi + 1] * xf * yf
        )
        heightfield += upscaled * roughness / (2 ** octave)

    return heightfield


def generate_stairs(
    num_steps: int = 10,
    step_height: float = 0.05,
    step_depth: float = 0.3,
    width: float = 2.0,
) -> str:
    """Generate MJCF XML for a staircase.

    Returns partial MJCF XML with box geoms representing stairs.
    """
    geoms = []
    for i in range(num_steps):
        x = i * step_depth
        z = (i + 1) * step_height / 2
        height = (i + 1) * step_height
        geoms.append(
            f'<geom type="box" pos="{x:.3f} 0 {z:.3f}" '
            f'size="{step_depth/2:.3f} {width/2:.3f} {height/2:.3f}" '
            f'rgba="0.6 0.6 0.6 1"/>'
        )
    return "\n        ".join(geoms)
