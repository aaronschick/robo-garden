"""Compose terrain + objects into complete environment MJCF."""

from __future__ import annotations

from robo_garden.environments.models import EnvironmentConfig
from robo_garden.environments.objects import object_to_mjcf


def generate_environment_mjcf(config: EnvironmentConfig) -> str:
    """Generate a complete MJCF environment from an EnvironmentConfig.

    This produces a standalone MJCF that can be merged with a robot MJCF
    using MuJoCo's include mechanism.
    """
    objects_xml = "\n        ".join(object_to_mjcf(obj) for obj in config.objects)

    return f"""<mujoco model="{config.name}">
  <option timestep="{config.physics.timestep}" gravity="{' '.join(str(g) for g in config.physics.gravity)}"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="{config.terrain.size[0]/2} {config.terrain.size[1]/2} 0.1" material="grid"/>
    {objects_xml}
  </worldbody>
</mujoco>"""
