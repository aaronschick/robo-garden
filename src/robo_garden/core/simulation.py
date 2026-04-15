"""MuJoCo simulation runner: load, step, collect state trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    duration: float
    timestep: float
    num_steps: int
    qpos_trajectory: np.ndarray  # (num_steps, nq)
    qvel_trajectory: np.ndarray  # (num_steps, nv)
    com_trajectory: np.ndarray   # (num_steps, 3) - center of mass
    stable: bool                 # Did the robot remain upright?
    diverged: bool               # Did the simulation diverge (NaN)?
    summary: dict = field(default_factory=dict)


def simulate(
    model: mujoco.MjModel,
    duration: float = 2.0,
    ctrl: np.ndarray | None = None,
    policy_fn=None,
) -> SimulationResult:
    """Run a forward simulation and collect trajectory data.

    Args:
        model: Compiled MuJoCo model.
        duration: Simulation duration in seconds.
        ctrl: Fixed control signal (if None, zero control).
        policy_fn: Optional callable(obs) -> action for closed-loop control.

    Returns:
        SimulationResult with state trajectories and stability assessment.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    num_steps = int(duration / model.opt.timestep)
    qpos_traj = np.zeros((num_steps, model.nq))
    qvel_traj = np.zeros((num_steps, model.nv))
    com_traj = np.zeros((num_steps, 3))

    diverged = False
    initial_com_z = None

    for i in range(num_steps):
        # Apply control
        if policy_fn is not None:
            obs = np.concatenate([data.qpos, data.qvel])
            data.ctrl[:] = policy_fn(obs)
        elif ctrl is not None:
            data.ctrl[:] = ctrl

        mujoco.mj_step(model, data)

        # Record state
        qpos_traj[i] = data.qpos.copy()
        qvel_traj[i] = data.qvel.copy()

        # Compute center of mass
        com = np.zeros(3)
        mujoco.mj_subtreeVel(model, data)  # Updates subtree COM
        com[:] = data.subtree_com[0]  # Root body COM
        com_traj[i] = com

        if initial_com_z is None:
            initial_com_z = com[2]

        # Check for divergence
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            diverged = True
            break

    # Stability assessment: did COM height drop by more than 50%?
    if initial_com_z and initial_com_z > 0:
        final_com_z = com_traj[min(i, num_steps - 1), 2]
        stable = final_com_z > initial_com_z * 0.5
    else:
        stable = True

    return SimulationResult(
        duration=duration,
        timestep=model.opt.timestep,
        num_steps=i + 1 if diverged else num_steps,
        qpos_trajectory=qpos_traj[: i + 1] if diverged else qpos_traj,
        qvel_trajectory=qvel_traj[: i + 1] if diverged else qvel_traj,
        com_trajectory=com_traj[: i + 1] if diverged else com_traj,
        stable=stable and not diverged,
        diverged=diverged,
        summary={
            "initial_com_z": float(initial_com_z) if initial_com_z else 0.0,
            "final_com_z": float(com_traj[min(i, num_steps - 1), 2]),
            "max_velocity": float(np.max(np.abs(qvel_traj[: i + 1]))),
        },
    )
