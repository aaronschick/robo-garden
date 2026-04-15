"""Materials database: load, query, and filter real-world materials."""

from __future__ import annotations

from pathlib import Path

import yaml

from robo_garden.building.models import Material
from robo_garden.config import DATA_DIR

_catalog: list[Material] | None = None


def load_catalog() -> list[Material]:
    """Load all material catalogs from YAML files."""
    global _catalog
    if _catalog is not None:
        return _catalog

    _catalog = []
    material_dir = DATA_DIR / "materials"
    for yaml_file in sorted(material_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for entry in data.get("materials", []):
            _catalog.append(Material(**entry))

    return _catalog


def find_material(material_id: str) -> Material | None:
    """Find a material by its ID."""
    for m in load_catalog():
        if m.id == material_id:
            return m
    return None


def query_materials(
    printable: bool | None = None,
    min_strength_mpa: float | None = None,
    material_type: str | None = None,
) -> list[Material]:
    """Query materials matching filter criteria."""
    results = load_catalog()

    if printable is not None:
        results = [m for m in results if m.printable == printable]
    if min_strength_mpa is not None:
        results = [m for m in results if m.yield_strength_mpa >= min_strength_mpa]
    if material_type is not None:
        results = [m for m in results if m.type == material_type]

    return results
