"""Build a clean .blend bundle of the urchin_v3 robot for texturing.

Run with:

    blender --background --python scripts/build_urchin_v3_blend.py

Produces: workspace/robots/urchin_v3/assets/urchin_v3.blend

The output is a static, non-animated, non-rigged mesh bundle organized
into a `urchin_v3` top-level collection with `base_shell` at the root
and `panels` holding all 42 vc_NN panels, each named after its URDF
link and placed at its URDF-derived world pose. Every mesh gets a
default Principled BSDF material slot and smooth shading with sharp
edges preserved via an "Auto Smooth" modifier (Blender 4.1+ replacement
for the legacy mesh.auto_smooth property). A camera and sun light are
added framing the robot so the .blend opens to a sensible view.
"""

from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy  # type: ignore
from mathutils import Euler, Matrix, Vector  # type: ignore


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "workspace" / "robots" / "urchin_v3" / "assets"
URDF_DIR = ASSETS_DIR / "urdf"
URDF_PATH = URDF_DIR / "urchin_v3.urdf"
MESHES_DIR = URDF_DIR / "meshes"
OUTPUT_BLEND = ASSETS_DIR / "urchin_v3.blend"


# -----------------------------------------------------------------------------
# URDF parsing
# -----------------------------------------------------------------------------

def _parse_xyz(s: str | None) -> Vector:
    if not s:
        return Vector((0.0, 0.0, 0.0))
    parts = [float(p) for p in s.strip().split()]
    return Vector(parts)


def _parse_rpy(s: str | None) -> Euler:
    if not s:
        return Euler((0.0, 0.0, 0.0), "XYZ")
    r, p, y = [float(v) for v in s.strip().split()]
    return Euler((r, p, y), "XYZ")


def _origin_matrix(origin_elem) -> Matrix:
    if origin_elem is None:
        return Matrix.Identity(4)
    xyz = _parse_xyz(origin_elem.get("xyz"))
    rpy = _parse_rpy(origin_elem.get("rpy"))
    rot = rpy.to_matrix().to_4x4()
    trans = Matrix.Translation(xyz)
    return trans @ rot


def parse_urdf(urdf_path: Path):
    """Return (root_link, {link_name: world_pose_matrix}, {link_name: mesh_rel_path}).

    For each <link> with a <visual><geometry><mesh>, compute its world pose
    by walking the joint chain from the URDF root outward. Each joint
    contributes its <origin>; each link's <visual><origin> contributes the
    extra visual offset *inside* the link frame.
    """
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    links: dict[str, ET.Element] = {}
    for link in robot.findall("link"):
        links[link.get("name")] = link

    # joint_map[child] = (parent, joint_origin_matrix)
    joint_map: dict[str, tuple[str, Matrix]] = {}
    children: set[str] = set()
    for joint in robot.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        origin = _origin_matrix(joint.find("origin"))
        joint_map[child] = (parent, origin)
        children.add(child)

    # Root = link that is not a child of any joint.
    roots = [n for n in links if n not in children]
    if len(roots) != 1:
        raise RuntimeError(f"Expected 1 URDF root link, got {roots}")
    root_link = roots[0]

    def link_world_pose(link_name: str) -> Matrix:
        # Walk up joint chain, composing from root.
        chain = []
        current = link_name
        while current in joint_map:
            parent, jorigin = joint_map[current]
            chain.append(jorigin)
            current = parent
        # chain is child-first; root is `current` now. Compose root->...->link.
        mat = Matrix.Identity(4)
        for jorigin in reversed(chain):
            mat = mat @ jorigin
        return mat

    poses: dict[str, Matrix] = {}
    meshes: dict[str, str] = {}
    for name, link in links.items():
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_elem = visual.find("geometry/mesh")
        if mesh_elem is None:
            continue
        visual_origin = _origin_matrix(visual.find("origin"))
        poses[name] = link_world_pose(name) @ visual_origin
        meshes[name] = mesh_elem.get("filename")

    return root_link, poses, meshes


# -----------------------------------------------------------------------------
# Blender scene build
# -----------------------------------------------------------------------------

def _clear_scene() -> None:
    # Remove all objects.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Purge orphan data blocks.
    for coll in (
        bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
        bpy.data.lights, bpy.data.images, bpy.data.collections,
    ):
        for block in list(coll):
            if getattr(block, "users", 0) == 0:
                coll.remove(block)


def _ensure_collection(name: str, parent=None):
    parent = parent or bpy.context.scene.collection
    for child in parent.children:
        if child.name == name:
            return child
    coll = bpy.data.collections.new(name)
    parent.children.link(coll)
    return coll


def _move_object_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _add_default_material(obj, material_name: str):
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    # Leave default Principled BSDF as the starting point.
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def _apply_smooth_with_sharp_edges(obj) -> None:
    """Blender 4.1+ removed mesh.use_auto_smooth. Replacement recipe:
    shade smooth + add a 'Smooth by Angle' geometry-nodes modifier
    (bpy.ops.object.shade_auto_smooth) which bakes the same behavior
    into a modifier the artist can tweak."""
    # Select only this object.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Shade smooth first.
    bpy.ops.object.shade_smooth()
    # Auto-smooth operator (adds the Smooth-by-Angle modifier).
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(30.0))
    except (AttributeError, RuntimeError):
        # Fallback for Blender <4.1: set the legacy property if it exists.
        if hasattr(obj.data, "use_auto_smooth"):
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(30.0)


def _import_obj(obj_path: Path) -> object:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="Y", up_axis="Z")
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    if len(new_objs) != 1:
        # Join if multiple pieces came in (shouldn't happen for these OBJs).
        if len(new_objs) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for o in new_objs:
                o.select_set(True)
            bpy.context.view_layer.objects.active = new_objs[0]
            bpy.ops.object.join()
            return bpy.context.view_layer.objects.active
        raise RuntimeError(f"No object imported from {obj_path}")
    return new_objs[0]


def build_scene() -> list[str]:
    """Build the urchin_v3 scene. Returns a list of warnings."""
    warnings: list[str] = []

    print(f"[build_urchin_v3_blend] URDF:    {URDF_PATH}")
    print(f"[build_urchin_v3_blend] Meshes:  {MESHES_DIR}")
    print(f"[build_urchin_v3_blend] Output:  {OUTPUT_BLEND}")

    root_link, poses, meshes = parse_urdf(URDF_PATH)
    print(f"[build_urchin_v3_blend] URDF root link: {root_link}")
    print(f"[build_urchin_v3_blend] Visual links:   {len(meshes)}")

    _clear_scene()

    # Unit scale = meters (Blender default, but be explicit).
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"

    # Collection hierarchy.
    root_coll = _ensure_collection("urchin_v3")
    panels_coll = _ensure_collection("panels", parent=root_coll)

    # Import in a deterministic order: root_link first, then vc_NN sorted.
    link_order = [root_link] + sorted(n for n in meshes if n != root_link)

    imported_count = 0
    for link_name in link_order:
        mesh_rel = meshes.get(link_name)
        if not mesh_rel:
            warnings.append(f"link {link_name!r} has no visual mesh")
            continue
        mesh_abs = (URDF_DIR / mesh_rel).resolve()
        if not mesh_abs.exists():
            warnings.append(f"missing mesh file: {mesh_abs}")
            continue

        obj = _import_obj(mesh_abs)
        obj.name = link_name
        obj.data.name = f"{link_name}_mesh"

        # Apply the URDF-derived world pose. For urchin_v3 all matrices
        # are identity (joint origins are all zero and vertices already
        # carry world-space positions), but we apply the matrix anyway
        # so the script is correct for any URDF.
        obj.matrix_world = poses[link_name]

        _add_default_material(obj, f"{link_name}_mat")
        _apply_smooth_with_sharp_edges(obj)

        target_coll = root_coll if link_name == root_link else panels_coll
        _move_object_to_collection(obj, target_coll)
        imported_count += 1

    print(f"[build_urchin_v3_blend] Imported {imported_count} meshes.")

    # Camera + sun light, framed on the robot.
    # Determine bounding box of all imported objects.
    min_v = Vector((float("inf"),) * 3)
    max_v = Vector((float("-inf"),) * 3)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            for i in range(3):
                min_v[i] = min(min_v[i], world_corner[i])
                max_v[i] = max(max_v[i], world_corner[i])
    center = (min_v + max_v) * 0.5
    extent = max((max_v - min_v)[i] for i in range(3))
    cam_dist = max(extent * 2.2, 0.6)

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    root_coll.objects.link(cam_obj)
    cam_pos = center + Vector((cam_dist * 0.8, -cam_dist * 0.9, cam_dist * 0.55))
    cam_obj.location = cam_pos
    # Aim at center.
    direction = (center - cam_pos).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    scene.camera = cam_obj

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 4.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.location = center + Vector((0.5, -0.5, 1.0))
    sun_obj.rotation_euler = Euler((math.radians(45), 0, math.radians(30)), "XYZ")
    root_coll.objects.link(sun_obj)

    # Viewport shading = Material Preview for the 3D viewports.
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"

    return warnings


def save_blend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def main() -> int:
    if not URDF_PATH.exists():
        print(f"[build_urchin_v3_blend] ERROR: URDF not found: {URDF_PATH}")
        return 2
    warnings = build_scene()
    save_blend(OUTPUT_BLEND)
    size = OUTPUT_BLEND.stat().st_size
    print(f"[build_urchin_v3_blend] Saved: {OUTPUT_BLEND}  ({size/1024:.1f} KiB)")
    if warnings:
        print("[build_urchin_v3_blend] Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("[build_urchin_v3_blend] No warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
