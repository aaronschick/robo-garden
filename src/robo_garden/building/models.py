"""Data models for the Robot Building Space: actuators, materials, assignments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Actuator:
    """A real-world actuator with physical specifications.

    Two actuator families are represented in the same dataclass:

    * Rotary (``type in {"servo", "bldc", "stepper"}``) — populates
      ``torque_nm`` and ``speed_rpm``. ``force_n`` / ``speed_mps`` /
      ``stroke_mm`` are ``None``.
    * Linear (``type == "linear"``) — populates ``force_n``,
      ``speed_mps`` and ``stroke_mm``. ``torque_nm`` / ``speed_rpm`` are
      ``None``.

    Use :pyattr:`kind` to dispatch without re-parsing ``type``.
    """

    id: str                       # e.g., "dynamixel_xm430_w350"
    name: str                     # e.g., "Dynamixel XM430-W350-T"
    type: str                     # "servo" | "bldc" | "stepper" | "linear"
    voltage: float                # Operating voltage
    weight_g: float | None = None  # Weight in grams (null allowed when sourcing is incomplete)
    # Rotary specs — required for rotary actuators, null for linear.
    torque_nm: float | None = None  # Peak torque in Newton-meters
    speed_rpm: float | None = None  # No-load speed in RPM
    # Linear specs — required for linear actuators, null for rotary.
    force_n: float | None = None    # Peak force in Newtons
    speed_mps: float | None = None  # Max linear speed in m/s
    stroke_mm: float | None = None  # Usable stroke in millimeters
    price_usd: float | None = None
    max_angle_deg: float | None = None  # None = continuous rotation
    interface: str = ""           # "ttl" | "rs485" | "pwm" | "canbus"
    source: str = ""              # Manufacturer/vendor
    url: str = ""                 # Product page URL
    notes: str = ""               # Free-form sourcing notes

    @property
    def kind(self) -> str:
        """Return ``"linear"`` or ``"rotary"`` based on declared ``type``.

        Dispatch key for downstream code (validator, generators) so they
        don't have to re-parse the YAML string contract.
        """
        if self.type == "linear":
            return "linear"
        return "rotary"


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
