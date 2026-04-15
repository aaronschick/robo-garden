"""Visual preview of robot designs using MuJoCo viewer."""

from __future__ import annotations

import mujoco
import mujoco.viewer


def launch_viewer(mjcf_xml: str, duration: float | None = None):
    """Launch the MuJoCo interactive viewer for a robot.

    Args:
        mjcf_xml: MJCF XML string defining the robot.
        duration: If set, auto-close after this many seconds. None = interactive.
    """
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    if duration is not None:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        mujoco.viewer.launch(model, data)
