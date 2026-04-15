"""Shared test fixtures for Robo Garden tests."""

import pytest


MINIMAL_MJCF = """<mujoco model="test_pendulum">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.5" mass="0.1"/>
      <body name="tip" pos="0 0 0.5">
        <geom type="sphere" size="0.05" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1"/>
  </actuator>
</mujoco>"""


QUADRUPED_MJCF = """<mujoco model="test_quadruped">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1"/>
    <body name="torso" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom type="box" size="0.15 0.08 0.04" mass="1.0"/>

      <body name="leg_fr" pos="0.1 0.08 0">
        <joint name="hip_fr" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.1"/>
        <body name="shin_fr" pos="0 0 -0.15">
          <joint name="knee_fr" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.05"/>
        </body>
      </body>

      <body name="leg_fl" pos="0.1 -0.08 0">
        <joint name="hip_fl" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.1"/>
        <body name="shin_fl" pos="0 0 -0.15">
          <joint name="knee_fl" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.05"/>
        </body>
      </body>

      <body name="leg_br" pos="-0.1 0.08 0">
        <joint name="hip_br" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.1"/>
        <body name="shin_br" pos="0 0 -0.15">
          <joint name="knee_br" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.05"/>
        </body>
      </body>

      <body name="leg_bl" pos="-0.1 -0.08 0">
        <joint name="hip_bl" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.1"/>
        <body name="shin_bl" pos="0 0 -0.15">
          <joint name="knee_bl" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip_fr" gear="1"/>
    <motor joint="knee_fr" gear="1"/>
    <motor joint="hip_fl" gear="1"/>
    <motor joint="knee_fl" gear="1"/>
    <motor joint="hip_br" gear="1"/>
    <motor joint="knee_br" gear="1"/>
    <motor joint="hip_bl" gear="1"/>
    <motor joint="knee_bl" gear="1"/>
  </actuator>
</mujoco>"""


@pytest.fixture
def minimal_mjcf():
    return MINIMAL_MJCF


@pytest.fixture
def quadruped_mjcf():
    return QUADRUPED_MJCF
