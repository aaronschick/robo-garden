"""Data models for the Robot Building Space: actuators, materials, assignments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Actuator:
    """A real-world actuator with physical specifications."""
    id: str                       # e.g., "dynamixel_xm430_w350"
    name: str                     # e.g., "Dynamixel XM430-W350-T"
    type: str                     # "servo" | "bldc" | "stepper" | "linear"
    torque_nm: float              # Peak torque in Newton-meters
    speed_rpm: float              # No-load speed in RPM
    voltage: float                # Operating voltage
    weight_g: float               # Weight in grams
    price_usd: float | None = None
    max_angle_deg: float | None = None  # None = continuous rotation
    interface: str = ""           # "ttl" | "rs485" | "pwm" | "canbus"
    source: str = ""              # Manufacturer/vendor
    url: str = ""                 # Product page URL


@dataclass
class ActuatorAssignment:
    """Maps a joint in the MJCF to a real-world actuator."""
    joint_name: str
    actuator: Actuator
    gear_ratio: float = 1.0


@dataclass
class Material:
    """A real-world material with physical properties."""
    id: str                       # e.g., "pla_standard"
    name: str                     # e.g., "PLA (Standard)"
    type: str                     # "3d_print" | "metal" | "composite" | "elastomer"
    density_kg_m3: float
    yield_strength_mpa: float
    elastic_modulus_gpa: float
    printable: bool
    cost_per_kg_usd: float | None = None
    max_temp_c: float | None = None
    notes: str = ""


@dataclass
class MaterialAssignment:
    """Maps a link in the MJCF to a real-world material."""
    link_name: str
    material: Material
