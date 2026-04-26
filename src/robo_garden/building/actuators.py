"""Actuator database: load, query, and filter real-world actuators."""

from __future__ import annotations

from pathlib import Path

import yaml

from robo_garden.building.models import Actuator
from robo_garden.config import DATA_DIR

_catalog: list[Actuator] | None = None


def load_catalog() -> list[Actuator]:
    """Load all actuator catalogs from YAML files."""
    global _catalog
    if _catalog is not None:
        return _catalog

    _catalog = []
    actuator_dir = DATA_DIR / "actuators"
    for yaml_file in sorted(actuator_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for entry in data.get("actuators", []):
            _catalog.append(Actuator(**entry))

    return _catalog


def find_actuator(actuator_id: str) -> Actuator | None:
    """Find an actuator by its ID."""
    for a in load_catalog():
        if a.id == actuator_id:
            return a
    return None


def query_actuators(
    min_torque_nm: float | None = None,
    max_weight_g: float | None = None,
    actuator_type: str | None = None,
    max_price_usd: float | None = None,
) -> list[Actuator]:
    """Query actuators matching filter criteria."""
    results = load_catalog()

    if min_torque_nm is not None:
        # Linear actuators have torque_nm = None; exclude them from a torque query.
        results = [a for a in results if a.torque_nm is not None and a.torque_nm >= min_torque_nm]
    if max_weight_g is not None:
        results = [a for a in results if a.weight_g is not None and a.weight_g <= max_weight_g]
    if actuator_type is not None:
        results = [a for a in results if a.type == actuator_type]
    if max_price_usd is not None:
        results = [a for a in results if a.price_usd is not None and a.price_usd <= max_price_usd]

    return results
