"""Smoke test for the if-to-where AST rewrite + JAX compile roundtrip.

Uses the exact reward structure from go2_walker_phase2_train.txt, including
the Python-level `if z < 0.15: r_term = -100.0 else: r_term = 0.0` pattern
that used to force the SB3 fallback.
"""
from robo_garden.rewards.reward_runner import (
    _rewrite_if_to_where,
    compile_jax_reward_function,
    compile_reward_function,
)
import numpy as np
import jax.numpy as jnp

# This is what Claude actually emitted during the last gym run — or close
# enough to it: trunk collapse termination written as a plain if/else.
REWARD = '''
def compute_reward(obs, action, next_obs, info):
    vx = next_obs[19]
    z = next_obs[2]
    qvel_prev = obs[25:37]
    qvel_next = next_obs[25:37]
    joint_accel = qvel_next - qvel_prev

    r_forward = np.clip(vx, 0.0, 4.0)
    r_height = np.exp(-10.0 * (z - 0.41) ** 2)
    r_smooth = -np.mean(np.abs(joint_accel))
    r_energy = -np.mean(np.abs(action * qvel_next))
    if z < 0.15:
        r_term = -100.0
    else:
        r_term = 0.0
    r = 2.0 * r_forward + 1.0 * r_height + 0.5 * r_smooth + 0.1 * r_energy + r_term
    return float(r), {}
'''

print("=== source before rewrite ===")
print(REWARD)
print("=== source after rewrite ===")
print(_rewrite_if_to_where(REWARD))

print("=== numpy compile (untouched) ===")
fn_np = compile_reward_function(REWARD, obs_dim=37, action_dim=12)
print("numpy OK")

print("=== JAX compile (should now succeed thanks to if->where) ===")
fn_jax = compile_jax_reward_function(REWARD, obs_dim=37, action_dim=12)
print("JAX OK — Brax GPU path unlocked")

obs = np.zeros(37, dtype=np.float32)
next_obs = np.zeros(37, dtype=np.float32)
next_obs[2] = 0.41
next_obs[19] = 1.0
print("numpy value (standing+moving): ", fn_np(obs, np.zeros(12), next_obs, {}))
print("jax   value (standing+moving): ",
      float(fn_jax(jnp.asarray(obs), jnp.zeros(12), jnp.asarray(next_obs))))

# Also check the collapse case — must hit the -100 branch.
next_obs[2] = 0.10
print("numpy value (collapsed z=0.10):", fn_np(obs, np.zeros(12), next_obs, {}))
print("jax   value (collapsed z=0.10):",
      float(fn_jax(jnp.asarray(obs), jnp.zeros(12), jnp.asarray(next_obs))))
