"""Object placement for simulation environments."""

from __future__ import annotations

from robo_garden.environments.models import ObjectConfig


def object_to_mjcf(obj: ObjectConfig) -> str:
    """Convert an ObjectConfig to an MJCF geom element."""
    pos = f"{obj.position[0]:.3f} {obj.position[1]:.3f} {obj.position[2]:.3f}"

    if obj.type == "box":
        size = " ".join(f"{s/2:.3f}" for s in obj.size[:3])
        return f'<body pos="{pos}"><geom type="box" size="{size}" mass="{obj.mass}" friction="{obj.friction}"/></body>'
    elif obj.type == "sphere":
        r = obj.size[0] / 2
        return f'<body pos="{pos}"><geom type="sphere" size="{r:.3f}" mass="{obj.mass}"/></body>'
    elif obj.type == "cylinder":
        r, h = obj.size[0] / 2, obj.size[1] / 2
        return f'<body pos="{pos}"><geom type="cylinder" size="{r:.3f} {h:.3f}" mass="{obj.mass}"/></body>'
    else:
        return f'<!-- Unknown object type: {obj.type} -->'
