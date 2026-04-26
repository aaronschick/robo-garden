"""Single gymnasium.Env wrapping any MuJoCo MJCF — SB3-compatible."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


class MuJoCoGymEnv(gym.Env):
    """Single gymnasium.Env wrapping a MuJoCo model loaded from MJCF XML.

    Compatible with stable-baselines3. Passes a standard (obs, action, next_obs)
    tuple to ``reward_fn``; if None a small control-effort penalty is used.

    Args:
        mjcf_xml: MJCF model string.
        max_episode_steps: Truncation horizon per episode.
        reward_fn: ``(obs, action, next_obs) -> float`` or None.
        done_fn: ``(next_obs) -> bool`` for early termination or None.
        action_scale: Multiplier applied to raw ``ctrlrange`` of actuators.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mjcf_xml: str,
        max_episode_steps: int = 500,
        reward_fn: Callable | None = None,
        done_fn: Callable | None = None,
        init_noise: float = 0.05,
    ):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for MuJoCoGymEnv")

        self.mjcf_xml = mjcf_xml
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.init_noise = init_noise

        self._model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._data = mujoco.MjData(self._model)

        nq = self._model.nq
        nv = self._model.nv
        nu = self._model.nu
        obs_dim = nq + nv

        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(nu, -1.0, dtype=np.float32),
            high=np.full(nu, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self.last_reward_components: dict = {}

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        noise = self._rng.uniform(-self.init_noise, self.init_noise, self._model.nq)
        self._data.qpos[:] += noise
        self._data.qvel[:] += self._rng.uniform(-self.init_noise, self.init_noise, self._model.nv)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        prev_obs = self._get_obs()

        # Scale action [-1, 1] → actual ctrlrange
        act_range = self._model.actuator_ctrlrange
        if act_range.shape[0] > 0:
            lo, hi = act_range[:, 0], act_range[:, 1]
            scaled = lo + (np.asarray(action, dtype=np.float32) + 1.0) * 0.5 * (hi - lo)
        else:
            scaled = np.asarray(action, dtype=np.float32)

        self._data.ctrl[:] = scaled
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()

        if self.reward_fn is not None:
            raw_r = self.reward_fn(prev_obs, action, obs)
            if isinstance(raw_r, tuple) and len(raw_r) == 2:
                reward = float(raw_r[0])
                self.last_reward_components = raw_r[1] if isinstance(raw_r[1], dict) else {}
            else:
                reward = float(raw_r)
                self.last_reward_components = {}
        else:
            reward = float(-0.1 * np.sum(action ** 2))
            self.last_reward_components = {}

        diverged = bool(np.any(np.isnan(obs)))
        early_done = bool(self.done_fn(obs)) if self.done_fn is not None else False
        terminated = diverged or early_done
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos, self._data.qvel]).astype(np.float32)


# ---------------------------------------------------------------------------
# Cartpole helpers
# ---------------------------------------------------------------------------

def cartpole_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
    """NumPy reward for cartpole balancing — used by SB3 (CPU) path.

    obs layout (from MJCF): [cart_pos, pole_angle, cart_vel, pole_ang_vel]
    Maximised when pole is upright (angle ≈ 0) and cart is near centre.
    """
    pole_angle = float(next_obs[1])
    cart_pos = float(next_obs[0])
    ctrl_cost = 0.001 * float(np.sum(action ** 2))
    return float(np.cos(pole_angle)) - 0.1 * cart_pos ** 2 - ctrl_cost


def cartpole_done(next_obs: np.ndarray) -> bool:
    """Terminate when the pole falls past ±45° or the cart leaves the track."""
    pole_angle = abs(float(next_obs[1]))
    cart_pos = abs(float(next_obs[0]))
    return pole_angle > 0.785 or cart_pos > 2.3  # 45°, track limit


# ---------------------------------------------------------------------------
# JAX-native variants (JIT-compilable) — used by MJX/Brax GPU path
# ---------------------------------------------------------------------------

def cartpole_reward_jax(obs, action, next_obs):
    """JAX-native cartpole reward — JIT-compilable, for Brax PPO on GPU.

    Identical logic to cartpole_reward but uses jnp so it survives JAX tracing.
    """
    import jax.numpy as jnp

    pole_angle = next_obs[1]
    cart_pos = next_obs[0]
    return jnp.cos(pole_angle) - jnp.float32(0.1) * cart_pos ** 2 - jnp.float32(0.001) * jnp.sum(action ** 2)


def cartpole_done_jax(next_obs) -> bool:
    """JAX-native early-termination check for cartpole — JIT-compilable."""
    import jax.numpy as jnp

    pole_angle = jnp.abs(next_obs[1])
    cart_pos = jnp.abs(next_obs[0])
    return jnp.logical_or(pole_angle > jnp.float32(0.785), cart_pos > jnp.float32(2.3))


# ---------------------------------------------------------------------------
# Go2 walker helpers
# ---------------------------------------------------------------------------
#
# Observation layout (from workspace/robots/go2_walker.xml): obs = [qpos, qvel]
# with nq=19, nv=18, obs_dim=37.
#   qpos[0:3]   = trunk xyz                 qvel[0:3]   = trunk linear velocity
#   qpos[3:7]   = trunk quaternion (wxyz)   qvel[3:6]   = trunk angular velocity
#   qpos[7:19]  = 12 hinge joint angles     qvel[6:18]  = 12 joint velocities
#
# Action layout: 12 motors, one per joint, order matches <actuator> block,
# ctrlrange = [-23.7, 23.7] Nm.  PPO emits actions in [-1, 1] (tanh-squashed);
# MJXBraxEnv / MuJoCoGymEnv scale that to ctrlrange before applying torque.

_GO2_STAND_HEIGHT = 0.41       # target trunk z (m) for a nominally-standing Go2
_GO2_COLLAPSE_Z = 0.15         # below this z, episode is terminated with a big
                               # negative reward — prevents the policy from
                               # learning to lie flat and collect cheap rewards.


def go2_walker_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
    """NumPy reward for Go2 forward-trotting (SB3 / CPU path).

    Matches the reward structure in workspace/prompts/go2_walker_phase2_train.txt:
      R = 2.0*R_forward + 1.0*R_height + 0.5*R_smooth + 0.1*R_energy + R_termination

    A collapse (trunk below 15 cm) ends the episode with -100 so the policy
    must stay upright to collect forward-velocity reward.
    """
    vx = float(next_obs[19])           # qvel[0] = trunk forward velocity
    z = float(next_obs[2])             # qpos[2] = trunk height
    qvel_prev = obs[19 + 6 : 19 + 18]  # 12 prior joint velocities
    qvel_next = next_obs[19 + 6 : 19 + 18]
    joint_accel = qvel_next - qvel_prev
    ctrl = np.asarray(action, dtype=np.float32)

    r_forward = np.clip(vx, 0.0, 4.0)
    r_height = np.exp(-10.0 * (z - _GO2_STAND_HEIGHT) ** 2)
    r_smooth = -float(np.mean(np.abs(joint_accel)))
    r_energy = -float(np.mean(np.abs(ctrl * qvel_next)))
    r_term = -100.0 if z < _GO2_COLLAPSE_Z else 0.0

    return float(
        2.0 * r_forward
        + 1.0 * r_height
        + 0.5 * r_smooth
        + 0.1 * r_energy
        + r_term
    )


def go2_walker_done(next_obs: np.ndarray) -> bool:
    """Terminate when trunk collapses below 15 cm."""
    return bool(next_obs[2] < _GO2_COLLAPSE_Z)


def go2_walker_reward_jax(obs, action, next_obs):
    """JAX-native Go2 forward-trotting reward — JIT-compilable for Brax PPO."""
    import jax.numpy as jnp

    vx = next_obs[19]
    z = next_obs[2]
    qvel_prev = obs[19 + 6 : 19 + 18]
    qvel_next = next_obs[19 + 6 : 19 + 18]
    joint_accel = qvel_next - qvel_prev

    r_forward = jnp.clip(vx, 0.0, 4.0)
    r_height = jnp.exp(jnp.float32(-10.0) * (z - jnp.float32(_GO2_STAND_HEIGHT)) ** 2)
    r_smooth = -jnp.mean(jnp.abs(joint_accel))
    r_energy = -jnp.mean(jnp.abs(action * qvel_next))
    r_term = jnp.where(z < jnp.float32(_GO2_COLLAPSE_Z), jnp.float32(-100.0), jnp.float32(0.0))

    return (
        jnp.float32(2.0) * r_forward
        + jnp.float32(1.0) * r_height
        + jnp.float32(0.5) * r_smooth
        + jnp.float32(0.1) * r_energy
        + r_term
    )


def go2_walker_done_jax(next_obs):
    """JAX-native collapse-termination check — JIT-compilable."""
    import jax.numpy as jnp

    return next_obs[2] < jnp.float32(_GO2_COLLAPSE_Z)


# ---------------------------------------------------------------------------
# Urchin v2 (spherical 30-voice-coil ball) helpers
# ---------------------------------------------------------------------------
#
# Observation layout (workspace/robots/urchin_v2.xml): obs = [qpos, qvel]
# with nq=49, nv=48, obs_dim=97.
#   qpos[0:3]    = shell xyz          qvel[0:3]    = shell linear velocity
#   qpos[3:7]    = shell quat         qvel[3:6]    = shell angular velocity
#   qpos[7:19]   = 12 passive sol ext qvel[6:18]   = 12 passive sol velocities
#   qpos[19:49]  = 30 voice-coil ext  qvel[18:48]  = 30 voice-coil velocities
#
# Action: 30 voice-coil motors (solenoid actuators disabled for Stage A
# training; the 12 sol_* joints remain in the model as passive spring feet).

_URCHIN_FLOOR_Z = 0.10         # below this the shell is clipping through the
                               # ground plane; terminate / penalise heavily.


def urchin_v2_reward(obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
    """NumPy reward for urchin_v2 free-rolling (SB3 / CPU path).

    Encourages any XY motion up to 1.5 m/s, keeps the shell above the floor,
    and gently penalises voice-coil chatter and raw actuator effort.
    """
    vx = float(next_obs[49])
    vy = float(next_obs[50])
    speed_xy = float(np.sqrt(vx * vx + vy * vy))
    z = float(next_obs[2])
    ctrl = np.asarray(action, dtype=np.float32)

    r_speed = float(np.clip(speed_xy, 0.0, 1.5))
    r_alive = 0.1 if z > _URCHIN_FLOOR_Z else -5.0
    r_smooth = -0.005 * float(np.mean(np.abs(next_obs[67:97])))
    r_energy = -0.001 * float(np.sum(ctrl * ctrl))
    return r_speed + r_alive + r_smooth + r_energy


def urchin_v2_reward_jax(obs, action, next_obs):
    """JAX-native urchin_v2 reward — JIT-compilable for Brax PPO."""
    import jax.numpy as jnp

    vx = next_obs[49]
    vy = next_obs[50]
    speed_xy = jnp.sqrt(vx * vx + vy * vy)
    z = next_obs[2]

    r_speed = jnp.clip(speed_xy, jnp.float32(0.0), jnp.float32(1.5))
    r_alive = jnp.where(z > jnp.float32(_URCHIN_FLOOR_Z), jnp.float32(0.1), jnp.float32(-5.0))
    r_smooth = jnp.float32(-0.005) * jnp.mean(jnp.abs(next_obs[67:97]))
    r_energy = jnp.float32(-0.001) * jnp.sum(action * action)
    return r_speed + r_alive + r_smooth + r_energy


# ---------------------------------------------------------------------------
# Built-in reward source strings for --mode train WSL dispatch
# ---------------------------------------------------------------------------
# These are the compute_reward source strings that wsl_dispatch.run_in_wsl()
# embeds in job.json so the WSL worker can compile them.  They mirror the
# logic of the NumPy reward functions above but written as standalone
# compute_reward(obs, action, next_obs, info) -> tuple[float, dict] bodies.

BUILTIN_REWARD_SOURCE: dict[str, str] = {
    "cartpole": """\
def compute_reward(obs, action, next_obs, info):
    pole_angle = next_obs[1]
    cart_pos = next_obs[0]
    ctrl_cost = 0.001 * np.sum(action ** 2)
    r = np.cos(pole_angle) - 0.1 * cart_pos ** 2 - ctrl_cost
    return float(r), {}
""",
    "go2_walker": """\
def compute_reward(obs, action, next_obs, info):
    vx = next_obs[19]
    z = next_obs[2]
    qvel_prev = obs[25:37]
    qvel_next = next_obs[25:37]
    joint_accel = qvel_next - qvel_prev
    r_forward = np.clip(vx, 0.0, 4.0)
    r_height = np.exp(-10.0 * (z - 0.41) ** 2)
    r_smooth = -float(np.mean(np.abs(joint_accel)))
    r_energy = -float(np.mean(np.abs(action * qvel_next)))
    r_term = -100.0 if z < 0.15 else 0.0
    r = 2.0 * r_forward + 1.0 * r_height + 0.5 * r_smooth + 0.1 * r_energy + r_term
    return float(r), {}
""",
    "urchin_v2": """\
def compute_reward(obs, action, next_obs, info):
    vx = next_obs[49]
    vy = next_obs[50]
    speed_xy = np.sqrt(vx * vx + vy * vy)
    z = next_obs[2]
    r_speed = np.clip(speed_xy, 0.0, 1.5)
    r_alive = np.where(z > 0.10, 0.1, -5.0)
    r_smooth = -0.005 * np.mean(np.abs(next_obs[67:97]))
    r_energy = -0.001 * np.sum(action * action)
    r = 1.0 * r_speed + r_alive + r_smooth + r_energy
    return float(r), {"speed": speed_xy, "z": z}
""",
}
