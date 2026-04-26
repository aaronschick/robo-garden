# Urchin v3 Continued Curriculum Plan

**Author:** planning agent, 2026-04-21
**Predecessor:** `C:\Users\aaron\.claude\plans\i-am-running-into-eventual-wozniak.md` (Path B design)

---

## 1. Status summary (where we are on 2026-04-21)

What just landed:
- **Path B (residual-on-oracle) is validated at its simplest configuration.** Contactpush oracle computes inside the env per tick (`_pre_physics_step`, `C:\Users\aaron\Documents\repositories\robo-garden\workspace\robots\urchin_v3\urchin_v3\urchin_env_cfg.py:310-382`); policy outputs 9-D SH residual coefficients; combined field decodes through the existing LPF → spring-PD pipeline.
- **BC-warmstart video review: passed** (5 episodes, 1/5 showed transient disc-shape but recovered — a known Path B non-issue since baseline is the oracle).
- **pathB_chain_run1 (1M PPO steps) video review: passed.** User summary: "rolls reliably, only one case fell into puck shape" — this is the failure mode we deliberately traded for a productive baseline.
- **pathB_chain 2M×5 hardening chain is currently running** (`scripts/run_urchin_v3_harden_chain.sh`) to harden the policy across 10M total env-steps.

Where the robot is today:
- **Task:** roll from (−0.5, −0.5) to (+0.5, +0.5), fixed flat ground.
- **Diagonal:** 1.41 m. **Episode:** 8 s. **Goal radius:** 0.5 m.
- **Yaw curriculum:** ± 0.3 rad at start → fully random by 600k sim-steps.
- **Distance curriculum:** currently disabled (`dist_scale_start = dist_scale_end = 1.0`).
- **Residual scale schedule:** 0.3 → 1.0 linearly over env-steps [500k, 2.5M].
- **Goal set:** single fixed point per episode. One goal reached = terminated.

What the robot can and cannot yet do:
- **Can:** point itself roughly toward a known-heading goal and produce sustained forward rolling motion on flat ground under mild yaw randomization.
- **Cannot yet:** reach goals in arbitrary directions in the full arena, recover from off-axis yaw without re-finding the oracle's natural forward mode, handle terrain, handle obstacles, handle payload changes, handle any partial actuator failure, do anything requiring short-term memory (oracle has none), reach sequential waypoints, or stop at a pose.

The deliberate shape of the baseline is "productive rolling when oracle conditions hold." Everything in this roadmap is about expanding the region where conditions hold *and* teaching the residual to cover corner cases the oracle can't see.

---

## 2. Architectural principles before phases

Three claims constrain the whole roadmap:

1. **The oracle is a closed-form function of `(gravity_b, to_goal_b)` and has no memory, no terrain awareness, no obstacle awareness, no inertia awareness.** Any capability requiring one of those must be supplied by the residual. The oracle cannot be extended to cover them without converting it into a policy, at which point Path B reduces to Path A.
2. **The residual is a 9-D SH field with RMS-normalized basis — good for smooth corrections, bad for sharp ones.** That means residual-addressable perturbations must have smooth spatial signatures on the panel sphere. Isolated-panel faults (actuator failure, local contact with an obstacle) have sharp signatures; expect the residual to approximate them, not represent them.
3. **`best_checkpoint.pt` lies.** Memory `feedback_verify_peak_before_handoff.md`: every chain handoff and every phase graduation requires rendered-video sign-off, not a scalar threshold. I will not propose any phase that advances on `best_reward` alone.

Two downstream consequences the roadmap will lean on:

- **When the oracle stops being a good prior, the right response is to augment the oracle, not replace it.** Adding an *additional* closed-form term to `compute_contactpush_oracle` (e.g. a slope-compensating tilt bias) is cheap; replacing oracle with policy is expensive. Several phases below do the augment.
- **Long-horizon and memory-requiring capabilities need a second architectural change (recurrent policy or goal-conditioned policy with goal reset).** I flag where this lands; it is deferred as far as possible.

---

## 3. Capability roadmap

Six phases, in order. Each phase is a 2–4M step PPO chain run that extends or replaces the current harden chain. Total projected compute: ~30M env-steps over 3–4 months at 64 envs × ~25k steps/sec ≈ 20 wall-hours/Mstep. With verification, retries, and render/review cycles, plan for 4–6 months calendar.

### Phase 0 — Harden baseline (in progress)

**What it tests:** that the current Path B config is not a local fluke; 10M env-steps of 2M chain sections without collapse.

**Why first:** we have 1M-step video-positive evidence but no seed-variance story on longer runs. Before we add curriculum, we need to know the baseline sits in a basin, not on a ledge.

**No changes.** Let the chain finish. Review each 2M segment's render before the next starts.

**Success criteria:**
- All 5 chain runs complete without the `COLLAPSE_FRAC=0.5` gate firing.
- Each run's det-eval mean stays above 800 (oracle-alone baseline from verification-plan step 2, per Path B doc).
- Video review of the final chain checkpoint: 5/5 seeds roll to goal, no puck attractors (or ≤ 1/5 transient puck that recovers).

**Failure modes to watch for:**
- Silent collapse *within* the collapse-frac tolerance (best stays okay but robot behavior degrades — the same reward-hack concern as `harden_run2` on 2026-04-20). Mitigation: render after every 2M segment, not just at the end.
- Residual coefficient norms growing past |residual|~0.5 on contactpush inputs. That would indicate BC-KL anneal is letting the policy drift from oracle-zero faster than rewards justify.

### Phase 1 — Random goal direction (yaw-robust reach from any heading)

**What it tests:** The robot reaches a goal sampled uniformly on a ring of the current 1.41 m diagonal radius — any direction from spawn. Currently the goal is fixed at (+0.5, +0.5) and the robot can lean on the fact that goal heading is constant.

**Why it's next:** yaw randomization (already annealed to full 2π in Phase 0) trains orientation-invariant *state handling* but not *direction-invariant task handling*. The BC-seed user-observation — "one case fell into puck shape" — specifically correlates with off-axis yaw at spawn; widening the goal direction exposes more oracle-hostile initial headings and forces the residual to earn its keep. Also: this is the capability axis with the smallest implementation footprint (randomize one tensor), so it's the highest-ROI next move.

**Env changes needed:**
- Replace fixed `GOAL_XY` with a per-env resampled goal at the start of each episode. The change lives in `urchin_env_cfg.py:640 _reset_idx` and touches `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones` (all currently index a global goal tensor at file:line 333, 392, 432, 635).
- Introduce a per-env `_goal_xy` buffer of shape `(N, 2)` set in `_reset_idx`, replacing the module-level `GOAL_XY` broadcast.
- Add env config fields:
  - `goal_radius_m: float = 1.0` (distance from spawn)
  - `goal_direction_span: float = math.pi/6` at start, anneal to `2*math.pi` by `goal_direction_anneal_end_sim_steps = 1_500_000`.
- Curriculum: begin this phase with a 30° span centered on the old (+1,+1) direction, widen to 360°. Anneal is purely env-side and does not interact with `residual_scale` anneal.

**Oracle changes needed:** None. The oracle is already a closed-form function of `to_goal_b`; random goal directions are in its natural domain.

**Reward changes needed:** None. Progress, potential-field, distance penalty, rolling-reward all already use `delta = goal - pos_xy`; they pick up the per-env goal automatically once wiring is fixed. Goal bonus and goal termination likewise.

**BC requirements:** The existing BC dataset was recorded with fixed (+0.5,+0.5) and yaw jitter. Residual targets are derived from `(gravity_b, to_goal_b)` at each step, so BC MSE should be unaffected by goal direction at replay time. **Judgment call:** I'd still record a fresh BC dataset with goal-direction randomization to broaden the preprocessor's running-stats coverage (`to_goal_b` norm and angle distribution changes). ~30 min of recording. Override this if you'd rather just freeze the existing scaler and move on.

**Training harness:** Extend the existing chain. Add two 2M chain entries, `pathB_randgoal_run{1,2}`. Warmstart from the last verified Phase 0 final. Total new compute: 4M env-steps ≈ 45 wall-hours.

**Success criteria:**
- Det-eval over 64 goal directions × single seed: median reward ≥ 600 (calibrate once Phase 0 baseline is known; target is "no worse than Phase 0 baseline minus 25%, then recovering").
- Video review: 5 seeds × 6 goal directions (evenly spaced); ≥ 90% of the 30 episodes show sustained rolling to goal.
- Puck-attractor rate: ≤ 2/30.

**Failure modes to watch for:**
- **Oracle asymmetry blindspot.** The contactpush oracle uses `forward_b = normalize(to_goal_3 − (to_goal_3·down_b)·down_b)`. When the robot is in a transient off-axis orientation, `forward_b` can swing rapidly as `to_goal` projects through the body-frame `down`, producing oracle jitter. Residual must damp this. If it doesn't, det-eval will show stalls near specific initial yaws.
- **Spawn-goal axis symmetry breaking.** The yaw curriculum is centered on the *original* start→goal heading (`math.atan2(GOAL_XY[1]-START_XY[1], …)`). When the goal moves, the yaw-jitter center must move with it. Fix in `_reset_idx` yaw draw (line ~656).

### Phase 2 — Random spawn and goal (arena-scale reach)

**What it tests:** Robot spawns at a random point in the arena and must reach a random goal, up to the env's full supported 4.24 m diagonal.

**Why it's next:** Phase 1 fixes direction but keeps distance constant. The residual has seen ~1.4 m of trajectory; we need to know the policy generalizes past that. Also: most interesting downstream capabilities (terrain, obstacles, multi-goal) need longer episodes and varied spawn, so get the geometry right before layering physics in.

**Env changes needed:**
- Replace `START_XY` global with per-env `_spawn_xy` sampled each reset.
- New config:
  - `arena_half_extent: float = 2.0` (m) — sample both spawn and goal uniformly in [-a, +a]²
  - `min_spawn_goal_dist: float = 0.3` — rejection-resample pairs closer than this (keeps `prev_dist` from spawning inside goal_radius)
  - Distance curriculum: enable `dist_scale_start = 0.5, dist_scale_end = 1.0` over `dist_curriculum_{start,end}_sim_steps = [100_000, 1_500_000]`. Short distances first.
- Episode length: bump from 8 s to 15 s to allow ~2.5 m travel at 0.2 m/s sustained. Config: `URCHIN_EPISODE_S` env var.
- Potential-field shaping parameter `Φ(s) = exp(-dist / 0.7)` — 0.7 m decay is poorly tuned for 3+ m distances. Change to `exp(-dist / 1.5)` under this phase, or introduce a config field `potential_scale_m: float = 1.5`. File: `urchin_env_cfg.py:462`.

**Oracle changes needed:** None functionally, but note that `to_goal_b` magnitude varies widely now; the oracle normalizes it so behavior is scale-invariant. Residual input distribution changes — this is the main reason to re-train scaler stats (see BC note below).

**Reward changes needed:**
- `goal_radius` currently 0.5 m. At arena scale, 0.5 m may be too generous (free bonus on nearby goals); propose scaling with `min(0.5, 0.15 * dist)`. But this changes the value baseline, so apply with care. **Judgment call:** I'd leave goal_radius at 0.5 m for Phase 2 to avoid compounding changes, and address the bonus-calibration drift in a Phase 2.5 mini-tune if needed.
- Distance penalty `-0.05 * dist` is linear and small; at 3 m it's −0.15/step × 900 steps ≈ −135/ep, a meaningful floor. If this dominates, the policy will prefer to hold near center rather than approach 4 m goals. Monitor `mean_reward` vs `mean_distance_final` correlation; if the policy stops approaching distant goals, consider clipping the distance penalty at 2 m.

**BC requirements:** Record a new BC dataset with arena-scale spawn/goal sampling. ~2000 episodes × 15 s × 60 Hz ≈ 1.8M samples; 30 min record time. Critical: **refit the running state scaler** after appending arena-scale data. Residual targets re-derived on oracle-zero contactpush will still be near-zero in the mean, but the 95th percentile of `|to_goal_b|` will be ~5× larger, and `RunningStandardScaler` needs to see that before PPO starts or it will under-normalize the `to_goal` obs slice.

**Training harness:** New chain script `scripts/run_urchin_v3_arena_chain.sh`, forked from the existing harden chain, with CLI env-vars for the new fields. 2 × 2M warmstart chain.

**Success criteria:**
- Det-eval over 64 (spawn, goal) pairs × 3 seeds: median reward relative to oracle-alone on matched pairs, ratio ≥ 1.0 (residual adds value or is neutral).
- Success rate (reached within episode) ≥ 80% at 1.5 m distance, ≥ 60% at 3.5 m distance.
- Video review: 3 seeds × 8 pair conditions (near/far × N/S/E/W goal); no puck attractors, sustained rolling.

**Failure modes to watch for:**
- Episode timeout rate spikes at long distance — robot rolls correctly but too slowly. If so, add a small sustained-forward-velocity bonus (a cleaner version of the current `velocity_reward_weight`, currently 2.0 but ungated against the speed_gate cap). Hold this until you see it, because adding velocity bonuses has historically farmed wobble.
- Scaler drift: `to_goal_b` slice has running mean that depends on goal distribution. If scaler is frozen from Phase 1 and Phase 2 distribution differs, residual coefficients will skew. This is exactly the scenario the `--freeze-scaler-after-warmstart` guard was designed against; for a distribution change of this size, **unfreeze the scaler for the first 500k steps of this phase** (set `--freeze-scaler-after-warmstart off` for the warmstart run, then re-freeze for subsequent chain segments).

### Phase 3 — Disturbance robustness (external pushes, contact friction variation)

**What it tests:** Robot continues rolling toward goal while subjected to small external pushes (random impulse force on root body) and variable ground-contact friction. Exercises the residual's ability to correct off-nominal dynamics the oracle doesn't model.

**Why it's next:** Before terrain (Phase 4), we want confirmation that the residual can close a *dynamics* loop the oracle can't see. Pushes and friction-change are the cleanest such signal because:
- They don't change the goal-reach dispatch logic (Phases 1–2 still good).
- They don't need a new observation channel (robot feels them through lin_acc, ang_vel, contact).
- They're domain-randomization-compatible with sim-to-real if that's ever a direction (no signal in memory that it is — see Risks section).

**Env changes needed:**
- Add an `event_manager`-style disturbance that applies a random 3-D impulse every ~1 s with probability 0.3, amplitude sampled from `N(0, 0.5 N·s)`, at the root body. Isaac Lab supports this via `isaaclab.envs.mdp.events.apply_external_force_torque`.
- Material friction randomization: replace single-friction `terrain_type="plane"` with a physics material whose dynamic friction is sampled per-episode from `U(0.6, 1.2)`. This requires a `PhysicsMaterialCfg` per-env spawn in `urchin_v3_cfg.py`, not in `urchin_env_cfg.py`. Non-trivial but bounded; estimate 1 day to wire up.
- Curriculum: push amplitude anneals 0 → 0.5 N·s over 1M steps; friction range anneals [1.0, 1.0] → [0.6, 1.2] over the same window.

**Oracle changes needed:** None. The oracle is unchanged by external forces; that's the point — residual must compensate.

**Reward changes needed:** None. Pushes produce off-heading motion that the existing progress/potential-field naturally penalizes; the robot must correct.

**BC requirements:** None. BC was recorded under nominal physics; residual targets are still valid since oracle hasn't changed.

**Training harness:** Extend chain. 2M PPO segment on top of Phase 2 final.

**Success criteria:**
- Success rate on pushed-episode eval (1000 eps, random spawns/goals, pushes enabled at full amplitude) drops by ≤ 20% from Phase 2 baseline. If it drops 50%, residual is not robust enough and we need to widen BC with scripted-oracle-plus-noise demos before continuing.
- Video review: 10 episodes with visible pushes — at least 7 show robot recovering trajectory after disturbance.

**Failure modes to watch for:**
- **Residual learns to ignore pushes rather than counter them.** If pushes are small enough that the policy can route around them via longer paths, it will. Mitigation: monitor per-episode correlation between impulse timing and velocity-toward-goal dip; if policy consistently fails to correct, increase push amplitude or increase frequency.
- **Contact friction observation asymmetry.** Robot has no direct friction observation; it infers from `joint_vel` vs commanded target. Residual needs 2–3 control ticks to pick up the discrepancy; under high-friction episodes it may lunge and slip. Expected behavior, not a bug — grade against a time-to-recover metric, not an instantaneous one.

### Phase 4 — Sloped and rough terrain

**What it tests:** Robot rolls up modest slopes (up to 10°), across random height perturbations (amplitude ≤ 0.03 m = ~17% of ball radius), and along gentle traverses of a tilted ground plane. Two subphases: gradient (slopes), then stochastic (rough).

**Why it's next:** Phases 1–3 taught goal-direction and disturbance robustness on flat ground. Terrain is the first physical change that breaks the oracle's "contact = ~constant downward pointing direction" assumption: on a slope, the `down_b` vector used by contactpush is tilted relative to goal-direction, and on rough terrain contact changes frame-to-frame in ways the quasi-static oracle misses.

**Env changes needed:** This is the biggest env change in the roadmap.
- Replace `TerrainImporterCfg(terrain_type="plane")` in `urchin_env_cfg.py:67` with Isaac Lab's `TerrainGeneratorCfg` using `ROUGH_TERRAINS_CFG` or a custom `TerrainGeneratorCfg` with sub-terrain difficulty axes. Use the generator's curriculum levels (1–5) indexed by a config field.
- Config:
  - `terrain_difficulty: int = 0` (start flat) → `5` (rough)
  - `terrain_anneal_start_sim_steps: int = 0`, `terrain_anneal_end_sim_steps: int = 3_000_000`.
- Respawn logic: `_reset_idx` must now re-sample spawn height from terrain heightmap at the chosen XY, not fixed 0.17 m. Isaac Lab's `TerrainImporter` exposes a `get_terrain_height` helper.
- **Observation:** consider adding a local heightmap snippet (3×3 grid, 0.1 m spacing around robot) as a new observation. Without it, the residual is trying to correct terrain with only IMU+gravity cues, which is under-sensed. **Judgment call:** start *without* heightmap — the 137-D obs already carries `projected_gravity_b` and `lin_acc_b` which together signal slope and incipient collision. If Phase 4 plateaus, add heightmap in a Phase 4.5 mini-iteration. This avoids a mid-phase observation-dim change that invalidates BC.

**Oracle changes needed:** **Yes — this is the first phase that augments the oracle.** The contactpush oracle computes `forward_b = normalize(to_goal_3 − (to_goal_3·down_b)·down_b)` which is the *horizontal* direction toward goal in body frame. On a sloped terrain, the relevant "direction the robot should push" is the terrain-surface projection, not the horizontal one. Augment:

```
# in compute_contactpush_oracle, replace the down_b definition:
surface_normal_b = <robot-inferred terrain normal>  # from contact-panel down-vector avg, or from heightmap
down_b = surface_normal_b  # NOT projected_gravity
```

But we don't have a robot-frame terrain-normal observation by default. Cheaper augmentation: add a small slope-compensation term that biases the oracle toward "push harder on the up-slope side." I'd prototype this standalone first (as a scripted variant callable from `record_bc_dataset.py`) and only graduate the augmented oracle into the env once it beats vanilla contactpush on a fixed 5° slope in isolation. This is a 1–2 day side-investigation that gates phase entry.

**Reward changes needed:**
- Optional: terminate on ball flip (robot upside down for > 0.5 s) — monitor `projected_gravity_b[2]` sign. If contactpush fails on rough terrain, the most common failure is flipping and rolling incorrectly from an inverted spring geometry. Flip-termination signals failure clean.
- Distance penalty is computed on horizontal distance — correct under terrain because goal is specified in XY.

**BC requirements:** Record a BC dataset with the augmented oracle on a mix of flat and 5° slope terrains. ~3000 episodes. The residual target under augmented oracle on slope will be near-zero (the augmented oracle is the new baseline); under vanilla oracle on slope, it will be nonzero (residual compensates what vanilla oracle misses). Pick one: either augment-then-warmstart (cleaner), or leave oracle-vanilla and let the residual learn slope compensation from PPO reward alone (faster to set up, but high risk that it doesn't).

**Training harness:** New chain `scripts/run_urchin_v3_terrain_chain.sh`. Because terrain changes physics and potentially obs, **do not chain from Phase 3's `final_checkpoint.pt` without re-running BC first**. The scaler and value function for "flat arena" are not appropriate initial conditions for "terrain." Start this phase with a fresh BC warmstart on the terrain-aware dataset, then 3× 2M PPO segments.

**Success criteria:**
- Flat-terrain regression eval: success rate within 10% of Phase 2 baseline (we should not have broken flat).
- 5° slope: success rate ≥ 50% at full amplitude.
- Rough terrain (difficulty 3): success rate ≥ 40%.
- Video review: 10 episodes each of flat/slope/rough; at least 7/10 of each show sustained rolling.

**Failure modes to watch for:**
- **Oracle augmentation sign bugs.** Slope-compensation terms are easy to get backward (push harder toward downhill rather than uphill). Verify against scripted rollouts on a static 5° ramp before letting PPO touch it.
- **Ball flip + respawn thrash.** If flip termination fires often, training distribution gets dominated by "spawn → wobble → flip" episodes. Mitigation: give flipped episodes a large −100 termination penalty and a curriculum-gated max-flip-rate check; suspend terrain annealing if flip rate > 25%.
- **Rolling-reward denominator.** `sigma = (0.5 * expected_omega).clamp(min=0.3)` Gaussian kernel assumes flat-ground no-slip kinematics. On a slope, no-slip rolling has a different angular-velocity-to-translation relationship. Probably still close enough that the kernel works, but eyeball after first render.

### Phase 5 — Multi-goal sequencing (memory via explicit goal index)

**What it tests:** Robot visits goals A, then B, then C in sequence. Goal A is revealed on reset; B revealed when A is reached; C when B is reached. Three reached = episode success.

**Why it's next:** This is the first capability that genuinely requires state the policy can't infer from instantaneous observation — which goal we're targeting depends on history. It's also the gateway to any long-horizon task. Two architectural options:
1. **Goal-index-in-obs** (cheap, no arch change): append current-goal-XY to obs. Policy is still memoryless; "memory" lives in the env's goal-advance logic.
2. **Recurrent policy** (expensive, invasive): GRU cell before the action head; policy infers goal state from history.

I recommend option 1 unequivocally for Phase 5. Option 2 is an infrastructure lift that only pays off when we have tasks where *the goal itself is not observable* (e.g. deliver-from-memory). We don't have those yet.

**Env changes needed:**
- Goal is now a list of 2–4 XY points. `_reset_idx` samples the list; `_get_dones` advances an internal goal-index on reach; `_get_rewards` rewards reach-of-current-goal, not final-goal.
- Observation: `to_goal_b` already represents current goal — it automatically updates when the env advances the index. **No obs dim change needed.** Nice side-effect of the existing design.
- Episode length: 25 s for a 3-goal sequence at arena scale.
- New config:
  - `num_goals: int = 3`
  - `goal_sequence_strategy: str = "chain"` (each goal is next sampled pair) or `"random_list"` (all goals revealed via per-goal obs slots).

**Oracle changes needed:** None. Oracle sees current goal via obs, doesn't know it's one of a sequence.

**Reward changes needed:**
- Per-goal bonus: reach-of-intermediate-goal gets a smaller bonus than final (100 vs 500) to avoid incentivizing the policy to loop on intermediate goals without progressing.
- Potential-field Φ already uses current goal distance — works correctly with sequenced goals.

**BC requirements:** Record BC with random 3-goal sequences at arena scale. Residual targets on contactpush remain near-zero (oracle doesn't care about sequence). BC here is mostly to keep the value function calibrated under the new episode structure.

**Training harness:** New chain `multigoal_chain`. Warmstart from Phase 4 final. 2× 2M segments.

**Success criteria:**
- 3-goal success rate ≥ 60% on flat, arena-scale.
- 2-goal success rate ≥ 80%.
- Video review: 5 seeds × 3 sequence patterns; robot visibly "switches" to new goal on reach.

**Failure modes to watch for:**
- **Goal-switch transient.** At the moment goal-index advances, `to_goal_b` jumps to the next goal; `prev_dist` resets. If `_get_rewards` uses the stale `_prev_dist`, the first frame after a goal-reach computes a spurious huge progress term. Mirror the reset logic from `_reset_idx` but for goal-advance events. **This is a real bug risk; treat as a mandatory code review item.**
- **Policy games intermediate goals.** Oscillating near-boundary of intermediate goal to re-collect the bonus. Mitigation: count `reached[i]` once per episode; don't re-award.

### Phase 6 — Actuator failure robustness (6 panels frozen)

**What it tests:** Robot continues reaching goals with 1–6 panels whose actuators are stuck at their rest position (random selection per episode). Approximates real-world failure modes.

**Why it's next:** Ordering choice. Actuator failure breaks the oracle's spatial symmetry assumption (oracle commands 42 panels; 6 don't respond). This is the harshest residual test because the correction is *localized* — exactly what the smooth 9-D SH basis is worst at. Placing it last lets residual have all prior capabilities before we hit it with a high-l problem.

**Env changes needed:**
- Per-env: sample `num_frozen = U{0, 6}` and `frozen_indices` random subset of `[0..41]` at reset.
- In `_pre_physics_step` after the LPF, override `self._smoothed_targets[:, frozen_indices] = self.cfg.rest_pos`. Simple masking.
- New observation: a 42-D `actuator_health` binary mask, appended to obs. **Obs dim change from 137 to 179.** This invalidates BC data; re-record required.

**Oracle changes needed:** None — oracle commands are issued to all 42 panels; env silently drops the ones that are frozen. Residual job: command the remaining 36 panels *harder* to compensate.

**Reward changes needed:** None.

**BC requirements:** Full re-record at 179-D obs. The extra 42 dims are binary, scaler-normalized trivially.

**Training harness:** Fresh BC + new chain `failure_chain`. Cannot warmstart directly from Phase 5 (dim mismatch). Zero-pad Phase 5 policy's first-layer weights and initialize the new input dims with zero — crude but preserves most of the prior learning.

**Success criteria:**
- 0-frozen success rate: no regression vs Phase 5.
- 3-frozen success rate: ≥ 60%.
- 6-frozen success rate: ≥ 40%.
- Failure-rate scaling curve (freeze-count vs success) is monotonic and smooth.

**Failure modes to watch for:**
- **SH basis hits its expressivity ceiling.** The smooth 9-D residual cannot fully compensate for 6 arbitrarily-placed frozen panels; expect failure rate to saturate near some non-trivial rate even with perfect training. If so, we've found Path B's architectural ceiling — document it and move to Phase 7-pending (see Section 6 first concrete step). Not a failure of the phase, a finding.

---

## 4. Cross-cutting concerns

### 4.1 When to orthogonalize `gravity_b` from `to_goal_b` in the observation

Currently both live in the 137-D obs. The oracle already uses their cross-product to derive `forward_b`; the policy sees both raw. Under terrain (Phase 4) the two vectors become more correlated (slope ties them together). No change needed — orthogonalization was historically useful for hand-designed representations; for MLP policies with scaler, it's a wash. Leave alone.

### 4.2 When does the oracle grow from l≤2 SH content to l≤3?

The current contactpush oracle is a *spatial* field with contact gates — it is not actually limited to l≤2 (sigmoids produce high-l content that the SH basis truncates on residual projection, but the oracle itself applies the full 42-panel field directly to the env). The question doesn't arise for the oracle. It arises for the residual.

**Recommendation:** do not expand the SH basis past 9 coefficients (l≤2). If Phase 6 shows the basis is the binding constraint, the correct response is not adding more SH modes — it's switching the residual to a direct 42-D panel-field MLP output. That's a 3-line policy change and doesn't touch the oracle. Queue as a Phase 7 candidate; do not do it speculatively.

### 4.3 When do we need a second policy (hierarchical)?

**Answer: not in the 6 phases above.** Phases 1–6 are all single-policy. Hierarchical (goal-planner + reaching-policy) would be appropriate for tasks like "explore an unseen arena" or "deliver a payload given a map" — neither is in scope. If you want a stretch goal, pushing Phase 5 to 8-goal sequences at arena scale starts to feel hierarchical; treat that as a Phase 7 signal.

### 4.4 BC dataset management

Each phase's BC is non-trivially different:
- Phases 1–3: same oracle, different obs distribution. One re-record sufficient for Phase 2; Phase 1 can reuse existing data.
- Phase 4: augmented oracle + terrain. Full re-record.
- Phase 5: new episode structure. Re-record recommended.
- Phase 6: new obs dim. Full re-record mandatory.

**Storage:** 1.8M samples × 137 float32 ≈ 1 GB per dataset. Keep last 2; archive older.

### 4.5 Scaler freeze policy

The `--freeze-scaler-after-warmstart` flag has historically been set to `auto` on warmstart runs. That's right for phases that don't change obs *distribution* much (Phase 1). For phases that do (Phase 2 arena, Phase 4 terrain), run the first warmstart segment with `off`, then flip to `on` for subsequent segments. Explicit per-phase in each new chain script.

### 4.6 Residual-scale anneal

Current: 0.3 → 1.0 over [500k, 2.5M]. Each new phase should re-anchor this window to start from "whatever residual_scale the chain is at when the phase starts." Under the existing chain script this is implicit (the counter is per-run, not per-chain) but it will matter when adding new phases. Promote `residual_scale_init` / `final` / window to phase-level config so each chain script can reset them to the phase's desired starting point.

### 4.7 Sim-to-real prep

**Not in scope for this roadmap.** Memory search turned up no signal that a physical urchin v3 is being built. If that changes, the roadmap bends: Phase 3 (disturbances) becomes Phase 3+sim2real-DR (mass, inertia, damping, actuator-lag randomization), and we add a domain-randomization-aware BC recording tool. Flag for user confirmation before any hardware assumption is baked in.

---

## 5. Risks and off-ramps

### 5.1 "Puck attractor returns under curriculum"
**Risk:** user reports "only one fell into puck" on 1M pathB run. Widening yaw (Phase 0 end) or goal direction (Phase 1) may surface more puck episodes. Residual hasn't seen these states in BC.
**Off-ramp:** add a shape-penalty term gated on low-velocity-at-high-time (`aspherity_weight` bump from 8 to 12, motion-gate extended from 1 s to 2 s). Or add a "puck recovery" scripted demo to the BC mix — a fast-return-to-rest-pos sequence from a flat pose. 1 day either way.

### 5.2 "Oracle-induced drift on random goal headings"
**Risk:** the oracle's `forward_b` derivation through gravity projection is unstable when `down_b` and `to_goal_b` nearly align (steep goal direction relative to current up). Rare in flat-ground, common in terrain.
**Off-ramp:** add an epsilon clamp to the `forward_b` denominator. 10-line change in `scripted_roll.py:427`.

### 5.3 "Phase 4 terrain bricks the whole pipeline"
**Risk:** terrain is the largest environmental change in the roadmap; any one of (spawn-respawn logic, oracle augmentation, scaler refit) can silently break things.
**Off-ramp:** Phase 4 is gated on a standalone oracle-only rollout test (render a static 5° slope, verify the augmented contactpush produces a forward-rolling motion; if not, the augmentation is wrong, stop). Budget 3 days for the standalone before PPO.

### 5.4 "Chain collapse rate too high to make progress"
**Risk:** collapse-frac gate fires mid-chain, we spend more time diagnosing than training.
**Off-ramp:** tighten to 0.3 fraction and switch chain handoff from `final_checkpoint.pt` to `best_checkpoint.pt` — **but only with explicit user approval** per memory `feedback_checkpoint_handoff.md`. Include the diagnosis ask as a required agenda item before any chain-handoff change.

### 5.5 "We reach Phase 6 and it just doesn't work"
**Risk:** actuator failure exposes the SH-9 residual's expressivity limit.
**Off-ramp:** this is the most interesting "failure" — it's evidence, not a blocker. Document the ceiling, draft a Phase 7 proposal to replace the 9-D residual with a 42-D direct panel MLP output, and defer until user approves the architecture revision.

### 5.6 "Hardware (VRAM) saturates under terrain"
**Risk:** terrain generators add polycount; rough terrain with heightmap obs may push 64 envs over 8 GB.
**Off-ramp:** drop to 32 envs, accept 2× wall-clock training time. Memory `user_profile.md` notes 8 GB shared VRAM. Budget accordingly; if it happens, don't try to fight it.

---

## 6. First concrete next step (after 2M harden chain finishes)

**Do this first:** implement and ship Phase 1 (random goal direction), nothing else. It is the smallest incremental change that exposes the current policy to a meaningfully broader task distribution, and it has the lowest risk of breaking anything (no physics change, no oracle change, no obs change, no BC re-record strictly required).

Concrete next-session checklist:
1. Render final pathB_chain_run5 checkpoint on 5 seeds × current fixed goal. Confirm no regression vs pathB_chain_run1's 1M result.
2. In `urchin_env_cfg.py`:
   a. Replace module-level `GOAL_XY` constant with per-env `_goal_xy` buffer in `_reset_idx`.
   b. Add `goal_direction_span_start`, `goal_direction_span_end`, `goal_direction_anneal_start_sim_steps`, `goal_direction_anneal_end_sim_steps` config fields.
   c. Sample per-env goal at reset: start on a ring of radius 1.0 m at angle `uniform(heading - span/2, heading + span/2)`, center `heading` on a fixed base direction (e.g. +X) that does not rotate with yaw.
   d. Update all four consumers of `GOAL_XY` (_pre_physics_step, _get_observations, _get_rewards, _get_dones) to read the per-env buffer.
3. Fork `run_urchin_v3_harden_chain.sh` to `run_urchin_v3_randgoal_chain.sh`:
   a. Set `URCHIN_START_XY=-0.5,-0.5`, remove `URCHIN_GOAL_XY`, bump `URCHIN_EPISODE_S=10`.
   b. Warmstart from the verified Phase 0 final.
   c. Two chain entries: 1M (slow ramp from span 30° → 180°) and 2M (span 180° → 360°).
4. First 1M finishes → render 6 goal directions × 3 seeds = 18 episodes. Review. Gate the 2M segment on video approval, not on reward.

**Estimated calendar:** 1 day env wiring, 20 wall-hours first 1M, ½ day video review, 40 wall-hours second 2M, ½ day video review. ~3.5 days of wall time if everything works; ~1 week of calendar with review cycles.

---

## Decision points where I made judgment calls you may want to override

1. **Phase ordering:** I put random-goal-direction (Phase 1) before arena-scale (Phase 2) because it's cheaper and exposes residual more. You could argue arena-scale first is a better stress test. My pick was risk-minimization.
2. **No hierarchy / no recurrence through Phase 6.** A recurrent policy would make Phases 5–6 easier but is a big infrastructure change. I deferred it. If you have near-term plans for memory-requiring tasks (deliver-from-hidden-state), re-order.
3. **Heightmap obs deferred to Phase 4.5.** You could argue for including heightmap from Phase 4 start. I chose to try without first because obs-dim changes are expensive (BC re-record, scaler refit, first-layer weight shape mismatch).
4. **Sim-to-real treated as out-of-scope.** If you're planning real hardware within 6 months, Phase 3 (disturbances) must be replaced with full domain-randomization, and Phase 4 terrain should include physics-randomization.
5. **Augmented oracle in Phase 4 is a proposal, not a commitment.** If pre-phase investigation shows residual can handle moderate slopes unaided, skip the oracle augmentation. It's a pure-residual problem until proven otherwise.
6. **Actuator failure placed last.** It's the most likely phase to expose a basis ceiling; I put it where finding a ceiling is most informative rather than most disruptive.

---

### Critical Files for Implementation
- `C:\Users\aaron\Documents\repositories\robo-garden\workspace\robots\urchin_v3\urchin_v3\urchin_env_cfg.py` — per-env goal buffer, terrain swap, multi-goal logic, event manager for disturbances, actuator-freeze masking
- `C:\Users\aaron\Documents\repositories\robo-garden\workspace\robots\urchin_v3\scripts\scripted_roll.py` — `compute_contactpush_oracle` augmentations for slope compensation (Phase 4)
- `C:\Users\aaron\Documents\repositories\robo-garden\workspace\robots\urchin_v3\scripts\train.py` — BC retarget loop to handle new goal/obs distributions, per-phase residual-scale anneal config surface
- `C:\Users\aaron\Documents\repositories\robo-garden\workspace\robots\urchin_v3\scripts\record_bc_dataset.py` — dataset recorder needs per-phase config (random goal, arena scale, terrain, multi-goal, frozen-panel mask)
- `C:\Users\aaron\Documents\repositories\robo-garden\scripts\run_urchin_v3_harden_chain.sh` — template for new per-phase chain scripts (randgoal, arena, terrain, multigoal, failure)
