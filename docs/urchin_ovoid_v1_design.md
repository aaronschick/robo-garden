# Urchin Ovoid Variant — Design & Future-Test Plan

## Context

Current urchin family (`urchin_v3`, `urchin_lite_v1`) is built on an approximately spherical body: 42 prismatic "panels" suspended on passive springs over a central shell, propelled by a contact-dipole oracle that extends rear-ground panels while bulging the front-top to shift CoM forward. Phase 1 flat-ground rolling was video-approved 2026-04-22 and cleared for Phase 2 (arena-scale spawn/goal).

A spherical body has one axis of symmetry per rotation plane and roughly isotropic moment of inertia — every rolling direction is mechanically equivalent. An **ovoid** (prolate spheroid, "egg-shaped") breaks that symmetry, and the symmetry break is the point:

- **Short (transverse) axis** has the smaller moment of inertia → cheap to spin up via contact impulses. This is the natural rolling axis.
- **Long (polar) axis** has the larger moment of inertia → expensive to spin up but stores more angular momentum per rad/s.
- **Intermediate-axis dynamics**: the Dzhanibekov / tennis-racket regime sits between these. An ovoid rolling on the short axis is spinning about an *unstable* intermediate axis of inertia; perturbations amplify. This is either a liability (body flips unpredictably) or an exploitable feature (flips become a controlled mode-switch into long-axis spin or hop).

The user's hypothesis: **store angular momentum via sustained short-axis rolling, then transfer it into a jump, spin, or pivot along the long axis.** This is a qualitatively new locomotion repertoire — not just "faster urchin."

**Scope.** This is a design mock-up for future testing, not a committed workstream. It belongs in `docs/` alongside `urchin_spin_lite_v1_design.md` as a parallel exploration. The baseline (Phase A short-axis rolling) must work before the novel phases (B spin, C energy transfer) are worth attempting. Most v3 infrastructure (panel physics, SH residual, BC+PPO+scaler-refit pipeline) transfers unchanged; the novel parts are geometry, the two-mode oracle, the rolling-mode observation, and the energy-transfer reward phase.

---

## Key design decisions (and what they assume)

| Decision | Choice | Risk / rationale |
|---|---|---|
| Body shape | **Prolate spheroid, semi-axes a=b=0.12 m (short), c=0.22 m (long). Aspect ratio ~1.8.** | Enough anisotropy to feel ovoid and produce meaningful inertia-tensor asymmetry (I_long / I_short ≈ 0.54 for a uniform solid spheroid at this aspect), small enough to roll on the short axis without spontaneously tipping onto the long axis at rest. Start at aspect **1.5** and escalate to 1.8 only after Phase A is clean. |
| Panel count | **Keep 42.** | Preserves reward-tuning transferability from v3. Reducing count first-order just reduces contact redundancy — not the variable under study. |
| Panel placement | **Ellipsoidal-area-weighted Fibonacci spiral** (not unit-sphere spiral) | Prevents polar caps from being over-sampled. The spiral parameter maps to arc-length-weighted sites on the spheroid surface. |
| Panel axes | **Outward ellipsoidal normals**: `n = normalize((x/a², y/b², z/c²))`, not `x/‖x‖`. | Radial-from-center axes would misalign panels at the poles (axes pointing closer to tangent than normal). The gradient-of-level-set form is geometrically correct. |
| Panel travel | **Unchanged**: [0, 0.060] m, rest=0.010 m, spring k=800 N/m, d=1.0 N·s/m, m=25 g. | Keeps v3 reward weights, BC hyperparameters, and smoothness penalties transferable. Panel-scale physics isn't the variable. |
| Actuation | **Unchanged**: 42 prismatic, PD outer loop k_p=80 / k_d=8, LPF α=0.30, effort 60 N. | Same. |
| Action space | **Keep 9-D SH (l≤2), same `residual_scale=0.5` bias toward oracle.** | Formally, an ovoid calls for spheroidal harmonics; pragmatically, l≤2 SH + learned residual should cover it. If slow-learning shows up as a symptom, fall back to **principal-axis-aligned SH** (same 9 modes rotated to body frame axes) — cheap change, documented in risk 6 below. |
| Observation | **~141-D**: v3's 137 plus 4 new dims. Add `rolling_axis_b (3)` (LPF'd `ω/‖ω‖`, τ=0.3 s) and `rolling_mode_scalar (1)` = `|ω · ĉ_long| / ‖ω‖` (0 on short-axis roll, 1 on long-axis spin). Optionally, a constant `body_shape_b=(a,b,c)` if the policy needs explicit geometry. | Makes mode dispatch legible to both oracle and reward. LPF is critical — instantaneous axis flicks near ‖ω‖≈0 would chatter the mode gate. |
| Oracle | **Two-mode extension of `compute_contactpush_oracle`** with soft-blend gate = `sigmoid(4·(mode_scalar − 0.5))`. | See Oracle section below for both modes and the blend. |
| Rewards | **All v3 terms kept verbatim** (progress, velocity, distance penalty, goal bonus, rolling_reward, aspherity, action smoothness). **Three new terms** for Phase A/B/C, each gated to its phase. | See Rewards section below. |
| Training pipeline | **Identical to v3.** BC pretrain (20 epochs, ~2000 episodes, `--bc-post-log-std=-1.6`) → PPO with annealed forward-KL `beta_bc` reg → scaler-refit between phases A/B/C. | Zero pipeline code change — only config. |

---

## Oracle: two modes, soft-blended

Extend `compute_contactpush_oracle` (workspace/robots/urchin_v3/scripts/scripted_roll.py:372-448) to dispatch on the filtered `rolling_mode_scalar`:

1. **Short-axis rolling mode** (default, gate weight → 1 when mode_scalar < 0.5):
   - Contact-dipole as in v3, **but** project `to_goal_b` onto the short-axis plane before computing front/rear. "Front" is the component of to-goal perpendicular to ĉ_long; "rear" is its negative. This prevents the oracle from commanding any long-axis-aligned contact pushes during Phase A.
   - Top-front CoM bulge uses the ellipsoidal "top" (highest-z panel after `body_shape` normalization), not the geometric north pole — otherwise the bulge sits on the long-axis cap and never touches the ground during short-axis rolling.

2. **Long-axis spin mode** (gate weight → 1 when mode_scalar > 0.5):
   - Identify the long-axis equator: panels with small `|n_z_body|` after rotating to principal-axis frame, i.e. normals roughly perpendicular to ĉ_long.
   - Extend those panels in a **tangential-push pattern**: for each equatorial panel, extend amplitude proportional to `(n × ĉ_long) · v̂_tangential_goal`, where `v̂_tangential_goal` is the desired tangential velocity direction for the commanded spin sense.
   - Non-equatorial panels stay at rest.

3. **Blend**: `action_raw = (1 − g) · short_axis_out + g · long_axis_out`, with `g = sigmoid(4·(mode_scalar_filtered − 0.5))`. EMA the mode scalar with τ = 0.5 s before sigmoiding — instantaneous gating chatters.

---

## Rewards: three new phase-gated terms

All existing v3 reward terms (urchin_env_cfg.py:859) are kept unchanged. Add:

- **`short_axis_bonus`** (Phase A only, weight `k_short = 3.0`):
  `k_short · (1 − |ω · ĉ_long| / ‖ω‖) · ‖ω‖`, gated on forward progress with the same 15-step duration gate used by `rolling_reward`. Incentivizes rolling *specifically* on the short axis, not generically rolling. Kills the degenerate "spin in place on long axis" reward-hack for Phase A.

- **`long_axis_spin_bonus`** (Phase B only, weight `k_spin = 2.0`):
  `k_spin · |ω · ĉ_long|`, additively gated on `(stored_ang_mom_magnitude > threshold)` — only counts once the body has accumulated angular momentum first. Prevents the "statically spin a 30 rad/s motor" degenerate solution.

- **`energy_transfer_bonus`** (Phase C stretch, weight `k_xfer = 5.0`):
  Reward the *event* where, within a 0.3 s window, short-axis angular momentum decreases by ≥ 30% **and** long-axis angular momentum + vertical CoM velocity (m·g·h-weighted) increases. Detected via windowed cross-correlation on buffered `ω` and `v_z` histories. **High reward-hacking risk** — always gate on a downstream verifier: a re-landing within 0.5 m of the transfer-event location OR continued forward progress post-spin. Expect this term to require 2–3 tuning iterations; it is not a one-shot spec.

Phase-gating is implemented in the same annealing-schedule style as v3's `progress_anneal_steps` — each term has (start_step, end_step, weight_at_end) tuple; terms outside their phase hold weight=0.

---

## Curriculum: pre-Phase-1 chain (A → B → C → re-enter v3 Phase 1)

| Phase | Spawn | Objective | Success metric | Steps |
|---|---|---|---|---|
| **A — Short-axis rolling** | Long axis horizontal, perpendicular to goal; Goal ±0.3 rad, 1.0 m. | Video-visible short-axis roll toward goal, maintained. | ≥ 0.3 m/s sustained 5 s, visible in render at 3 bearings. User video-approval. | 2M + BC warmstart |
| **B — Long-axis spin in place** | Long axis vertical, zero goal distance. | Sustain spin about ĉ_long. | ≥ 6 rad/s `|ω · ĉ_long|` sustained 5 s. User video-approval. | 1M |
| **C — Transfer (stretch)** | Short-axis rolling; after 3 s rolling, goal teleports 90° bearing + elevation 0.2 m. | Discover an energy-transfer that reaches elevated goal (jump, pivot, or spin-hop). | Goal bonus triggered from elevated state. **Not** cross-correlation alone. | 3M |
| **→ Phase 1 (v3 original)** | Random goal direction, flat ground. | As in `docs/urchin_v3_continued_curriculum.md`. | Unchanged. | 1.5M |

After Phase C (or if C is skipped), the ovoid re-enters v3's Phase 1→6 roadmap. Phases 2–6 (arena-scale spawn, disturbance robustness, terrain, multi-goal) require no ovoid-specific changes — just the ovoid robot config and the Phase-A checkpoint.

Scaler-freeze policy between A/B/C matches v3's convention (memory: `feedback_scaler_refit_not_bc_pretrain.md`): when obs distribution changes across a phase boundary, run first segment with `--freeze-scaler-after-warmstart off` then flip to `on`, and use `--scaler-refit-dataset` when only the scaler needs updating.

---

## Feasibility: where this could break

1. **Intermediate-axis instability is working against us, not for us.** A prolate spheroid rolling on its short axis is spinning about an axis with inertia between I_min and I_max — the classic unstable intermediate axis (Dzhanibekov regime). Perturbations amplify and the body wants to flip onto the long axis. **Mitigation:** a 60-line passive-dynamics sanity script, run *before* any RL, measures time-to-flip from an initial short-axis spin. If the flip timescale is < 1 s at realistic panel compliance, Phase A is not viable as "rolling" — downgrade to "rocking" (oscillating ground contact without full rotations) and rewrite the short-axis bonus accordingly. **This is a gating experiment.** Do not commit BC recording or PPO chain time until the passive-dynamics result is in hand.

2. **Aspect ratio too aggressive → tips at rest.** 1.8 may be too much. **Mitigation:** start the URDF at aspect 1.5; only escalate after Phase A is clean at 1.5.

3. **Oracle mode-gate chatters near ‖ω‖ ≈ 0.** When the body is nearly stationary, `ω/‖ω‖` is numerically unstable; `rolling_mode_scalar` flicks between 0 and 1. **Mitigation:** EMA filter with τ = 0.5 s on `mode_scalar` before sigmoiding, and add a magnitude gate: below `‖ω‖ < 0.5 rad/s`, force mode_scalar = 0 (default to short-axis mode).

4. **Reward hacking on the transfer bonus.** Cross-correlation-window rewards are exactly the sort of shaped signal PPO exploits — expect the policy to find that panel-jiggle produces the "signature" without any actual energy transfer. **Mitigation:** always gate Phase C on a downstream physical verifier (re-landing location, post-spin forward progress). Document upfront that this term will require iteration; budget 2–3 tuning rounds.

5. **BC dataset cost for Phase A.** The short-axis oracle is novel enough that v3's BC dataset doesn't transfer. **Budget:** ~1 day of oracle authoring and smoke-rendering, plus ~3 hrs of WSL data generation for ~2000 episodes. This is the main non-training engineering cost.

6. **SH basis may not cleanly represent short-axis dipoles.** If PPO learns the residual slowly or the oracle alone can't achieve smooth short-axis rolling, the l≤2 SH basis may be the bottleneck. **Fallback:** rotate the SH basis into body principal axes (ĉ_long = body Z), keeping the 9 coefficient slots. One-file change in the SH constructor. Cheaper than switching to true spheroidal harmonics.

7. **Zombie WSL / blender processes during long chains.** Unrelated to ovoid design but applies (memory: `feedback_zombie_processes.md`) — monitor during Phase A/B/C runs. Same risk profile as any multi-M-step PPO chain on this hardware.

---

## Build plan — files to create

Mirror the `workspace/robots/urchin_spin_lite_v1/` tree:

```
workspace/robots/urchin_ovoid_v1/
├── assets/urdf/urchin_ovoid_v1.urdf              # generated
├── urchin_ovoid_v1/
│   ├── __init__.py
│   ├── urchin_ovoid_v1_cfg.py                    # ArticulationCfg: inertia for prolate spheroid
│   └── urchin_env_cfg.py                          # obs slices, reward weights, phase schedules
├── scripts/
│   ├── build_urdf.py                              # ellipsoidal-normal Fibonacci spiral
│   ├── scripted_ovoid_oracle.py                   # two-mode oracle + soft-blend gate
│   ├── record_bc_dataset.py                       # copy v3, swap oracle
│   ├── passive_dynamics_sanity.py                 # gating experiment for risk 1
│   ├── scripted_ovoid_video.py                    # pre-RL oracle render
│   └── train.py                                   # copy v3 train.py, no code change
└── urdf_meta.json                                 # (a, b, c), 42 panel sites
```

### Implementation steps

1. **Passive-dynamics sanity first.** Write `passive_dynamics_sanity.py` before anything else. Drop a provisional ovoid URDF (aspect 1.5, all panels at rest, zero action) into MuJoCo; seed with initial angular velocity around the short axis; measure time until `|ω · ĉ_long| / ‖ω‖` exceeds 0.3 (flip threshold). Average over 20 seeds. **If t_flip < 1 s at aspect 1.5, stop and rewrite the curriculum.** See risk 1.

2. **Fork URDF builder.** Copy `workspace/robots/urchin_v3/scripts/build_urdf.py` equivalent; swap the sphere sampler for an ellipsoidal-area-weighted Fibonacci spiral and the panel-axis computation for the gradient-of-level-set form.

3. **Fork cfg module.** Copy `urchin_v3/urchin_v3/urchin_v3_cfg.py`. Update the base-link inertia tensor to the prolate-spheroid closed form. Panel springs/damping/mass unchanged. Actuator block unchanged.

4. **Fork env cfg.** Copy `urchin_v3/urchin_v3/urchin_env_cfg.py`. Add: `rolling_axis_b`, `rolling_mode_scalar`, optional `body_shape_b` to obs slices. Add `short_axis_bonus`, `long_axis_spin_bonus`, `energy_transfer_bonus` with phase-gated weights. All v3 terms preserved.

5. **Write two-mode oracle.** `scripted_ovoid_oracle.py` — same signature as `compute_contactpush_oracle`, returns blended action. Short-axis branch projects `to_goal_b` onto the short-axis plane; long-axis branch pushes equatorial panels tangentially. Document with the same obs-slice comment pattern as `scripted_roll.py:65-80`.

6. **Pre-RL oracle video.** `scripted_ovoid_video.py` — render both modes at 5 seeds before any RL. Confirm visually that short-axis mode rolls and long-axis mode spins. Gate before BC recording.

7. **BC dataset.** `record_bc_dataset.py` — ~2000 episodes of short-axis oracle rollouts for Phase A warmstart.

8. **Phase A smoke.** 100k-step PPO smoke run (same sizing as `scripts/render_phase1_smoke.sh`). Success = median episode return above random baseline, rolling_reward + short_axis_bonus both firing, no NaNs.

9. **Phase A full chain.** 2M steps at 3 bearings (0°, 45°, 90°). Render with `scripts/render_policy_video.py`. **User video-approval before handoff** (memory: `feedback_verify_peak_before_handoff.md`, `feedback_checkpoint_handoff.md`). No handoff on `best_reward` alone.

10. **Phases B, C** only if A is video-approved.

---

## Critical files to reference (reuse, don't rewrite)

| Purpose | File |
|---|---|
| Reward composition template | `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:859` |
| Articulation physics (spring/damping) | `workspace/robots/urchin_v3/urchin_v3/urchin_v3_cfg.py` |
| Oracle to generalize | `workspace/robots/urchin_v3/scripts/scripted_roll.py:372` |
| SH basis (decide whether to rotate-align) | `workspace/robots/urchin_v3/scripts/scripted_roll.py:57` |
| Training pipeline (no changes expected) | `workspace/robots/urchin_v3/scripts/train.py` |
| BC regularization reference | `docs/urchin_v3_bc_reg_design.md` |
| Curriculum template for post-C re-entry | `docs/urchin_v3_continued_curriculum.md` |
| Design-doc structure template | `docs/urchin_spin_lite_v1_design.md` |
| Phase 1 chain script to adapt | `scripts/run_urchin_v3_phase1_chain.sh` |

---

## Verification (end-to-end)

1. **Passive-dynamics sanity** (risk 1 gate): `passive_dynamics_sanity.py`, 20-seed average time-to-flip ≥ 1 s at aspect 1.5. Archive result in `workspace/eval/ovoid_passive_dynamics/`.

2. **Oracle rollout video** (pre-RL): `scripted_ovoid_video.py` at 5 seeds per mode. Visual confirmation that short-axis mode rolls and long-axis mode spins.

3. **Phase A smoke** (100k steps, 4 envs, bearing 0°): no NaNs, `rolling_reward > 0` mean, `short_axis_bonus > 0` mean, body CoM moves ≥ 0.5 m per episode on average.

4. **Phase A full chain** (2M × 3 bearings): user video-approval before handoff. Head-to-head distance-to-goal comparison vs. v3's `pathB_chain_run1` at the same bearings — faster? comparable? failed to learn? This is the falsifier for the whole exploration.

5. **Phases B, C** gated on A approval. Each phase has its own render + user-review before the next phase starts.

---

## Out of scope for this plan

- Oblate ovoid (pancake shape) — inertia-tensor inversion makes the long axis the easy-roll axis; different dynamics, separate plan.
- Active shape morphing (changing aspect ratio mid-episode via panel extensions alone) — plausibly implicit in the existing panel action space, but treating it as an explicit affordance is a separate design.
- Scaling beyond 42 panels — size is not the variable under study.
- Sim-to-real for ovoid geometry — hull fabrication is a manufacturing-track problem, not an RL one.
- Post-Phase-C curriculum changes — once C is clean, re-enter v3's Phase 1→6 as-is.
