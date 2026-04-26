# Urchin Spin-Panel Variant — Design & Build Plan

## Context

The current urchin family (`urchin_v3`, `urchin_lite_v1`) rolls by **extending panels radially** (prismatic spring joints) in a contact-dipole pattern — rear panels push off the ground, front-top panels bulge to shift the CoM forward, sustained rolling emerges. This works (Phase 1 chain was video-approved 2026-04-22) but uses compliant contact-impulse locomotion, not a true wheeled gait.

This plan swaps that propulsion mechanism: panels stay at a fixed radial position but **spin** about an axis tangent to the sphere surface. The panel rim (a slight protrusion past the spherical hull) contacts the ground and drives the body via friction — an omniwheel-style locomotion. Differential spin rates across panels induce body rotation two ways:

1. **Ground-contact mode** — rim friction on the bottom-most panel pushes the body translationally; reaction torque about the wheel axle rotates the body, bringing adjacent panels into contact, chaining propulsion.
2. **Airborne mode** — spinning up/down airborne panels dumps angular momentum into the body (reaction-wheel / CMG effect), allowing mid-roll attitude correction without any external contact.

Tilt is **not** actively commanded on the panel (no extra DOF); it emerges from differential wheel-torque coupling into the body frame.

**Why try this.** Current contactpush gait has narrow learnability — ang-vel rewards had to be carefully gated, BC pretraining was required, and reward-hacking risks surfaced repeatedly during Phase 1 tuning. A wheeled gait, if feasible, has a much higher natural ceiling speed and may be easier to specify reward-wise (just "translate toward goal").

---

## Key design decisions (and what they assume)

| Decision | Choice | Risk / rationale |
|---|---|---|
| Start body | **`urchin_lite_v1`** (8 panels, octahedral) | Simpler to debug — 8 wheels is already rich enough to validate the mechanism. Scale to v3 later if lite works. |
| Panel geometry | **Short cylinder (disc), axis tangent to sphere surface, rim protrudes ~5 mm past the sphere hull** | A perfectly-flush disc cannot make ground contact — rim must poke out. 5 mm is small enough not to change the "sphere"-like body shape but large enough for reliable contact given the 0.060 m panel scale. |
| Joint type | **revolute (hinge), unlimited range, axis = tangent vector in panel plane** | Standard wheel joint. Tangent vector picked consistently per face (cross of face normal with body +Z, renormalized). |
| Actuator | **Velocity-motor via `ImplicitActuatorCfg(stiffness=0.0, damping=0.2, effort_limit≈2 N·m, velocity_limit≈30 rad/s)`** | No position spring — pure velocity tracking. Effort limit bounds friction impulse per step. |
| Panel mass | **keep 0.060 kg** (same as lite) | Preserves overall body moment of inertia ≈ same as current lite; reward-tuning experience transfers better. |
| Action space | **8-D direct per-panel velocity command, each ∈ [-1, 1] → ±vmax** | Drop SH-basis decomposition for v1 — easier to sanity-check scripted oracles. Can re-introduce SH in a follow-up if symmetry exploit helps learning. |
| Observation | **128-D**: `lin_acc(3) + ang_vel(3) + projected_gravity(3) + joint_vel(8) + to_goal(2) + contact(8) + prev_action(8) + panel_normals_body(24) + body_quat(4) + reserve` — reduce from 137-D to the needed dims | Drop `joint_pos` (wheels don't have meaningful angular position — they're continuous), keep `joint_vel` (how fast each wheel is spinning), add `prev_action` (common for velocity control). |
| Contact signal | **proxy = `‖projected_gravity × panel_normal‖ × sign(contact_force)`** at panel (tactile booleans from PhysX contact sensors if available, else thresholded `joint_vel` discrepancy) | Current contactpush uses spring-deflection as a tactile proxy; spin panels need a replacement. |
| Reward | **reuse `rolling_reward` + `progress_reward` + `distance_penalty`; drop `aspherity_penalty` (no extensions to penalize) and the contactpush oracle; add `action_rate_penalty` to damp wheel thrashing** | rolling_reward is geometry-agnostic (measures ang_vel aligned to translation). Keep it. |
| Baseline oracle | **"spin-all-forward"**: rotate every panel such that the tangent-velocity component aligned with the to-goal direction is positive. Works because the bottom panel is the only one whose rim contacts ground on any given tick, so only it contributes translation — and its direction is correct by construction. | Replaces the contactpush scripted oracle for BC pretraining and sanity checks. |

---

## Feasibility: where this could break

1. **Flush-panel contact failure.** If the ~5 mm rim protrusion is too small relative to sphere curvature + body settling penetration, no contact ever occurs. **Mitigation:** start with 5 mm protrusion; if smoke shows zero contact impulse, increase to 10 mm and re-run.

2. **Multi-contact fight.** An octahedron resting on a face has **one** panel in clean contact, but during rolling there are transient 2-panel or edge-contact moments where two wheels with different tangent axes both touch ground. Their velocities may be kinematically inconsistent → wheel scrub, energy loss, noisy reward. **Mitigation:** rolling_reward's Gaussian peak on pure-rolling kinematics already penalizes scrub; should self-correct in RL. If it doesn't, add a scrub penalty.

3. **Reaction-torque authority is weak for primary drive.** Airborne panels produce reaction torque proportional to (panel moment of inertia) × (angular accel), which for a 0.060 kg disc of radius 0.03 m is ~2.7e-5 kg·m². Body MoI is ~1e-3 kg·m² (~40× larger). So a 30 rad/s panel spin-up in 0.1 s yields only ~0.75 rad/s body attitude shift per panel — real but modest. **Mitigation:** don't rely on airborne reaction torque for primary propulsion; use it as attitude trim only. This is a property of the design, not a bug — the wheeled ground-contact is the main drive.

4. **No position-feedback on wheels.** Unlike prismatic joints, hinge joint_pos grows unboundedly for spinning wheels — can't include in obs. Policy must operate on joint_vel alone. **Mitigation:** this is standard for wheeled robots; include prev_action in obs to give the policy a memory surrogate.

---

## Build plan

### Files to create (mirror structure from `workspace/robots/urchin_lite_v1/`)

```
workspace/robots/urchin_spin_lite_v1/
├── assets/urdf/urchin_spin_lite_v1.urdf          # generated
├── urchin_spin_lite_v1/
│   ├── __init__.py
│   ├── urchin_spin_lite_v1_cfg.py                # ArticulationCfg w/ velocity ImplicitActuatorCfg
│   └── urchin_env_cfg.py                          # reward weights, obs/action defs
├── scripts/
│   ├── build_urdf.py                              # hinge joints + protruding disc panels
│   ├── scripted_spin_oracle.py                    # "spin-all-forward" BC dataset generator
│   └── train.py                                   # copy from urchin_lite_v1 + tweak
└── urdf_meta.json                                 # face normals + panel-rim-offset (5mm)
```

### Implementation steps

1. **Fork URDF builder.** Copy `workspace/robots/urchin_lite_v1/scripts/build_urdf.py` → new location. Change:
   - Panel geometry from box(0.040×0.040×0.004) → cylinder(radius=0.030, length=0.004), oriented with cylinder axis = **tangent vector** (not face normal).
   - Panel parent-link offset = `face_normal × (sphere_radius + 0.005)` — +5 mm rim protrusion.
   - Joint type `prismatic` → `revolute`, axis = the disc's rotation axis (geometrically set so rim presents to ground when the opposite face is down).
   - Unlimited joint range (no `<limit/>`).
   - Add `<friction>` with μ=1.0 to panel rim geom for grip.

2. **Fork cfg module.** Copy `urchin_lite_v1/urchin_lite_v1_cfg.py:100-128` style `ImplicitActuatorCfg`. Change:
   - `stiffness=0.0` (no position tracking)
   - `damping=0.2` (mild velocity damping for numerical stability)
   - `effort_limit=2.0` N·m
   - `velocity_limit=30.0` rad/s
   - `joint_names_expr=[r"panel_\d+"]` (regex unchanged from lite)

3. **Fork env cfg.** Copy `urchin_lite_v1/urchin_env_cfg.py`. Change:
   - **Remove** `aspherity_weight` entirely.
   - **Remove** contactpush oracle import / action-decoder call path.
   - **Set** action space to 8-D direct: `actions = tanh(policy_out)`, `target_vel = actions * 30.0`.
   - **Keep** `rolling_reward_weight=4.0`, `rolling_motion_threshold=0.10`, `rolling_duration_steps=15`, `progress_reward_weight=40.0`, `distance_penalty_weight=0.05`.
   - **Add** `action_rate_penalty_weight=0.02` on `||a_t - a_{t-1}||²`.
   - **Redefine** observations per the 128-D layout in the design table above. Update `OBS_*_SLICE` constants accordingly and document them like `scripted_roll.py:65-80`.

4. **Write new scripted oracle.** Replace `compute_contactpush_oracle` with `compute_spin_forward_oracle`:
   - Input: `panel_normals_body` (8×3), `to_goal_b` (3), body `ang_vel_b` (3).
   - For each panel i: compute tangent-velocity direction of rim at ground-contact point (cross of panel_normal with gravity_body), project onto to_goal direction, use sign to command panel spin velocity.
   - Output: 8-D velocity command in [-1, 1].

5. **Copy & tweak `train.py`.** Base: `workspace/robots/urchin_lite_v1/scripts/train.py` (Isaac Lab PPO, skrl 2.0). Tweaks:
   - Change `--robot-name` default to `urchin_spin_lite_v1`.
   - Update checkpoint dir to `workspace/checkpoints/urchin_spin_lite_v1/<run_id>`.
   - Keep skrl 2.0 pipeline fixes ([N,1] shapes, `inputs["observations"]` key).

6. **Smoke test chain** (start here — all else blocked on this passing):
   ```
   scripts/smoke_urchin_spin_lite_v1.sh   # 100k steps, 4 envs, single bearing (0°)
   ```
   Confirm: no physics explosion, non-zero contact impulses, body displacement > 0.3 m by end.

7. **BC pretrain + Phase-1 chain** (if smoke passes):
   - Generate 50k scripted-oracle rollouts via `scripted_spin_oracle.py`.
   - BC warmstart (10 epochs) → PPO 1M steps → render + user video-review at 0°/45°/90°.
   - No chain handoff on `best_reward` without video approval.

---

## Critical files to reference (reuse, don't rewrite)

| Purpose | File |
|---|---|
| URDF builder skeleton | `workspace/robots/urchin_lite_v1/scripts/build_urdf.py` |
| Spring→velocity actuator change reference | `workspace/robots/urchin_v3/urchin_v3/urchin_v3_cfg.py:100-123` |
| rolling_reward implementation (keep verbatim) | `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:859-990` |
| Obs-slice documentation pattern | `workspace/robots/urchin_v3/scripts/scripted_roll.py:65-80` |
| skrl 2.0 training pipeline | `workspace/robots/urchin_lite_v1/scripts/train.py` |
| Phase 1 chain script to adapt | `scripts/run_urchin_v3_phase1_chain.sh` |

---

## Verification (end-to-end)

1. **Physics smoke (manual, no training):** Launch the new robot in Isaac Sim view mode with scripted-oracle actions only. Confirm visually that panels spin and body translates. Success criterion: body CoM moves ≥ 0.3 m in 5 seconds of scripted-forward commands.

2. **Contact sanity:** Log panel rim contact-force magnitude for each of 8 panels over one scripted rollout. Expect: at any instant exactly ~1 panel has significant normal force; that panel's rim tangential force magnitude > 0; total force integrates to the net body impulse.

3. **100k-step PPO smoke on flat ground, 0° bearing:** Success = median episode return > baseline (random policy return, measured separately), no NaNs, rolling_reward actively firing (non-zero mean).

4. **Phase-1 style chain (only if smoke passes):** 1M steps × 3 bearings (0°, 45°, 90°). Render policy-video via `scripts/render_policy_video.py`. User video-review before any handoff.

5. **Head-to-head baseline comparison:** Same 3 bearings, compare distance-to-goal curves vs. `pathB_chain_run1` (the approved v3 baseline). Does the spin variant reach goal faster? Same? Or fail to learn? This is the falsifier.

---

## Out of scope for this plan

- Scaling to `urchin_v3` (42 panels) — defer until lite spin-variant is proven.
- Hybrid push+spin panels — orthogonal design question, different plan.
- Curriculum beyond Phase 1 — reapply the Phase-1→Phase-6 template once baseline is approved.
