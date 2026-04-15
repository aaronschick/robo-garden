"""Data models for the Environment Building Space."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    type: str = "flat"  # "flat" | "heightfield" | "stairs" | "rough" | "mixed"
    size: tuple[float, float] = (10.0, 10.0)  # meters (width, length)
    params: dict = field(default_factory=dict)  # type-specific parameters


@dataclass
class ObjectConfig:
    """Configuration for an object in the environment."""
    type: str = "box"  # "box" | "sphere" | "cylinder" | "mesh"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, ...] = (0.1, 0.1, 0.1)
    mass: float = 1.0
    friction: float = 1.0


@dataclass
class PhysicsConfig:
    """Physics simulation parameters."""
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_iterations: int = 4


@dataclass
class DomainRandomizationConfig:
    """Ranges for domain randomization during training."""
    friction_range: tuple[float, float] | None = None
    mass_scale_range: tuple[float, float] | None = None
    gravity_range: tuple[float, float] | None = None
    actuator_strength_range: tuple[float, float] | None = None


@dataclass
class EnvironmentConfig:
    """Complete environment specification."""
    name: str
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    objects: list[ObjectConfig] = field(default_factory=list)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    domain_randomization: DomainRandomizationConfig | None = None
    mjcf_xml: str = ""
