"""MJCF tweaks for Isaac Sim 5.x MJCF → USD import.

Isaac Sim 5.1's MJCF importer can fail on MuJoCo Menagerie assets with:

- Named ``<asset><material/>`` references (USD ``Invalid empty path`` / ``Used null prim``).
- Per-geom ``rgba`` after inlining — the importer still builds ``material_rgba``
  USD prims and hits the same broken code path.

For **viewport mirroring** we strip all material-related attributes from every
``<geom>`` and drop ``<material>`` / ``<texture>`` from ``<asset>``.  MuJoCo
defaults apply; Isaac no longer runs the flaky material copy path.  Meshes and
collision geometry are unchanged.
"""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger("isaac_server.mjcf_compat")

# Isaac 5.1 MJCF→USD: avoid any geom-driven USD material creation.
_GEOM_STRIP_ATTRS = frozenset(
    {
        "material",
        "materialclass",
        "rgba",
        "emission",
        "specular",
        "shininess",
        "reflectance",
    }
)


def build_isaac_viewport_mjcf_xml(xml: str) -> str:
    """Return MJCF text safe-ish for Isaac 5.1 import (may equal input)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return xml

    changed = False

    def strip_geoms(elem: ET.Element) -> None:
        nonlocal changed
        if elem.tag == "geom":
            for k in list(elem.attrib):
                if k in _GEOM_STRIP_ATTRS:
                    del elem.attrib[k]
                    changed = True
        for child in elem:
            strip_geoms(child)

    strip_geoms(root)

    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag in ("material", "texture"):
                asset.remove(child)
                changed = True

    if not changed:
        return xml

    try:
        return ET.tostring(root, encoding="unicode")
    except Exception as exc:
        log.warning(f"MJCF Isaac compat: could not serialize tree: {exc}")
        return xml


def write_isaac_viewport_mjcf_beside_source(src: Path) -> Path | None:
    """Write a temp MJCF next to ``src`` (same folder as mesh assets).

    Returns ``None`` if the transform did not change the file (nothing to do).
    """
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning(f"MJCF compat: cannot read {src}: {exc}")
        return None

    new_xml = build_isaac_viewport_mjcf_xml(text)
    if new_xml == text:
        return None

    parent = src.resolve().parent
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".xml",
            prefix=f".{src.stem}_rg_isaac_",
            dir=str(parent),
            delete=False,
            newline="\n",
        ) as tf:
            tf.write(new_xml)
            out = Path(tf.name)
    except OSError as exc:
        log.warning(f"MJCF compat: cannot write temp MJCF in {parent}: {exc}")
        return None

    log.info(f"MJCF compat: wrote Isaac-viewport MJCF (geom materials stripped): {out}")
    return out
