"""Core Robot domain object: the central data structure flowing through all spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import mujoco

from robo_garden.building.models import ActuatorAssignment, MaterialAssignment


@dataclass
class RobotMetadata:
    """Provenance and tracking information for a robot design."""
    created_at: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    iteration: int = 0
    prompt: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Robot:
    """A robot design with its MJCF description and real-world component mappings."""
    name: str
    mjcf_xml: str
    actuators: list[ActuatorAssignment] = field(default_factory=list)
    materials: list[MaterialAssignment] = field(default_factory=list)
    metadata: RobotMetadata = field(default_factory=RobotMetadata)

    def to_mj_model(self) -> mujoco.MjModel:
        """Compile the MJCF XML into a MuJoCo model."""
        return mujoco.MjModel.from_xml_string(self.mjcf_xml)

    def save(self, directory: Path) -> Path:
        """Save robot MJCF to a file."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.xml"
        path.write_text(self.mjcf_xml)
        return path

    @classmethod
    def load(cls, path: Path) -> Robot:
        """Load a robot from an MJCF file."""
        xml = path.read_text()
        return cls(name=path.stem, mjcf_xml=xml)
