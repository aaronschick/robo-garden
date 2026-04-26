# Phase 4 — Corrective Data Loop: Design Proposal

**Status:** DRAFT — awaiting user sign-off before any implementation.
**Date:** 2026-04-24
**Scope:** Plan §12.4 (DAgger-style corrective data loop).
**Supersedes:** nothing yet; first Phase 4 design pass.

## 1. Goal

Close the imitation gap in off-distribution states for the
`phase3_oraclehalf` BC controller. The canonical target is **F3 — arc
clumsiness** (per `urchin_v3_phase3_bc_controller.md`): `arc_left` and
`arc_right` curl the right direction but wobble on the curved trajectory,
a ~6.5× per-element closed-loop MSE blow-up over open-loop val MSE that
is textbook BC state-distribution drift.

Non-goal: the 6 stubbed primitives (brake / stop_settle / wobble_idle /
slide / jump_{stationary,rolling}) remain out of scope — they are Phase 6
engine-channel work. F1 (style differentiation) and F2 (weak accelerate)
were resolved upstream by the partial-oracle retrain and are not targets
here.

## 2. Architecture

```
    +---------+         student π_BC                  supervisor π*
    |         | obs ---> ---------------> action ---> -------------> label
    |  env    |              ^                         (primitive +
    | (Isaac) |              |                          rolling_engine)
    |         | <--- action--+                              |
    +---------+                                             |
         |                                                  v
         |  student-induced obs ───────────────────►  record label
         |                                                  |
         |                                                  v
         +──────►  drift trigger (action_rmse, trajectory, tilt)
                        gates whether to record this step
                                 |
                                 v
                      append (obs, supervisor_action, cond) to HDF5
```

Exactly one new recorder script: `record_corrective_rollouts.py`. It
loads the BC checkpoint, runs the student-driven rollout, and at each
step also computes what the scripted primitive + rolling_engine would
have produced on the same obs. Drift-gated rows go into a new HDF5.
The student never sees the supervisor action during the rollout.

The supervisor is **phase-clocked** (phase advances at `env_dt` from t=0)
exactly like `record_primitive_dataset.py` already does at
lines 390–407. This is correct: rolling-engine phase is a wall-clock
variable, not a feedback variable.

## 3. Failure detector triggers

The detector runs per-env, per-step. Three candidate signals, ranked by
ease of implementation and ability to discriminate F3-style drift:

### 3.1 Action-space drift (RECOMMENDED for iter 1)

```
d_action_t = || student_action_sh_t  −  supervisor_action_sh(obs_t) ||_2
```

Trigger: `d_action_t > θ_action` for N consecutive steps (debounce).

Initial threshold calibration comes from the Phase 3 closed-loop
fidelity eval (`workspace/eval/bc_phase3/fidelity.json`): worst-combo
`action_rmse_mean` was 0.059–0.065 (`arc_right:snappy`, `accelerate:*`).
Proposed default: **θ_action = 0.06**, N = 3 steps. `--dry-run` mode
(§5) should be used to calibrate this before first real recording.

Cost: near-zero. The supervisor action is already needed as the label —
the detector just diffs it against the student action, which we have
from the forward pass.

### 3.2 Trajectory drift (DEFERRED to iter 2+)

For `straight_roll`: lateral y-error in the start→goal frame exceeds
threshold. For `arc_left`/`arc_right`: distance from the ideal arc
implied by `steering_bias`. For `accelerate`: speed-vs-expected ramp.

This is semantically more grounded but requires per-primitive "ideal
trajectory" math. The engine does not currently expose an ideal
trajectory — the relationship between `steering_bias` and turn radius
is implicit. Proposing to defer until iter 1 demonstrates whether
action-space drift alone converges. If iter 1 fails on F3, trajectory
drift becomes the iter 2 improvement.

### 3.3 Body-state / terminal-failure flags (RECOMMENDED as emergency gate)

- **Flipped**: `projected_gravity_b[z] > 0.5` (upside-down).
- **Puck onset**: `|v_xy| < 0.05 m/s` sustained for 1.0 s while the
  primitive is not a brake/stop primitive.
- **Early termination without reach**: env sets truncated=True and the
  goal was not reached.

These always record (no threshold needed) because any of them is a
terminal failure and the supervisor trajectory from that state is
strictly more valuable than the student's. They are not expected to
fire often in iter 1 (phase3_oraclehalf doesn't puck in the review
videos), but they catch catastrophic drift cheaply.

## 4. Targeted drift scenarios

The env's default arena sampler picks spawn/goal uniformly at
half-extent 2.0 m with `min_spawn_goal_dist = 0.5`. That distribution
is what BC was trained on. To force drift we need scenarios that bias
spawn/goal toward states the BC controller does not yet handle.

### 4.1 iter 1 scenario pack

| name | what it does | why |
|------|--------------|-----|
| `long_arc` | spawn/goal sampled with distance ∈ [2.5, 3.5] m AND lateral offset ∈ [1.5, 2.5] m relative to body heading. Forces 3s+ continuous curving. | F3 is an arc drift problem; the default arena picks many short straight runs. |
| `off_axis_spawn` | spawn with yaw offset sampled uniformly in [±45°, ±90°] from the start→goal axis. | Student must recover heading while rolling — the compounding state-distribution drift case. |
| `combined` | `long_arc` + `off_axis_spawn` simultaneously. | The adversarial corner. |
| `nominal` | default sampler; no forced drift. | Sanity baseline — if the drift rate in nominal goes UP on subsequent iterations that's a regression signal. |

### 4.2 Deferred scenarios (iter 2+ or later)

- `mid_rollout_perturbation` — inject impulsive lateral force at t=T/2.
  Requires a perturbation API we don't have yet.
- `extreme_commands` — steering_bias outside dataset jitter range
  (±0.5 vs recorded ±0.15). Risks labeling saturated-oracle states
  where the supervisor itself is not informative (panel clamp caps
  the target).
- `style_switch` — mid-episode style change. Requires a time-varying
  cond-input path in the student rollout, which doesn't exist.

## 5. CLI shape

Following the Phase 3 convention
(`record_primitive_dataset.py`, `bc_rollout_video.py`):

```
C:/isaac-venv/Scripts/python.exe \
    workspace/robots/urchin_v3/scripts/record_corrective_rollouts.py \
    --checkpoint workspace/checkpoints/bc/phase3_oraclehalf/model.pt \
    --num-envs 16 \
    --scenarios long_arc,off_axis_spawn,combined,nominal \
    --primitives arc_left,arc_right \
    --styles neutral,snappy \
    --episodes-per-combo-per-scenario 10 \
    --episode-s 8.0 \
    --trigger action_rmse \
    --trigger-threshold 0.06 \
    --trigger-consecutive-steps 3 \
    --pre-trigger-lookback 10 \
    --post-trigger-frames 60 \
    --emergency-gates flipped,puck_onset,early_trunc \
    --output workspace/datasets/urchin_v3_dagger_iter1_<ts>.h5 \
    --seed 0
```

Key flags:

- `--checkpoint` — BC student checkpoint. Default:
  `phase3_oraclehalf/model.pt`.
- `--scenarios` — comma-separated scenario names from the scenario
  registry defined in the new script. A scenario is a named generator
  of `(spawn, goal, initial_yaw, env_overrides)`.
- `--primitives`, `--styles` — combos to record. Iter 1 default
  restricts to arc_{left,right} × {neutral,snappy} = 4 combos.
- `--episodes-per-combo-per-scenario` — parallel rollouts per cell of
  the (primitive, style, scenario) cross product.
- `--trigger` — trigger family (`action_rmse` for iter 1).
- `--trigger-threshold`, `--trigger-consecutive-steps` — as defined in
  §3.1.
- `--pre-trigger-lookback`, `--post-trigger-frames` — window around the
  trigger moment. Supervisor's *recovery* after the trigger is the
  most valuable signal, so post-trigger is typically larger.
- `--emergency-gates` — always-on failure flags from §3.3.
- `--record-always` (not shown above) — debug mode, label every step
  regardless of triggers. Use it the first time to calibrate
  thresholds against actual drift-signal distributions.
- `--dry-run` — run the rollouts, compute triggers, print summary
  statistics, but do NOT write HDF5. Use to calibrate
  `--trigger-threshold`.

### Env-state constraints (non-negotiable)

The script must set these env overrides for every rollout, matching the
dataset BC was trained against:

- `URCHIN_ORACLE_AMPLITUDE=0.5`
- `URCHIN_RESIDUAL_SCALE_INIT=1.0`
- `URCHIN_RESIDUAL_SCALE_FINAL=1.0`
- `URCHIN_GOAL_SAMPLING_MODE=arena`
- `URCHIN_ARENA_HALF_EXTENT=2.0`
- `URCHIN_MIN_SPAWN_GOAL_DIST=0.5`
- `URCHIN_POTENTIAL_SCALE_M=1.5`

Any deviation would shift the obs distribution and corrupt DAgger
labels. The script should assert these at startup and bail if the env
cfg doesn't match.

## 6. Dataset strategy

### 6.1 One new HDF5 file per DAgger iteration

Do NOT append to `urchin_v3_primitives_oraclehalf_20260424_012337.h5`.
Its dataset shapes are fixed-size (allocated to
`total_transitions` at creation), and in-place append on a 3.15 GB
HDF5 file on NTFS is slow and a corruption hazard. Instead:

```
workspace/datasets/urchin_v3_primitives_oraclehalf_20260424_012337.h5  (iter 0, 4.15M rows)
workspace/datasets/urchin_v3_dagger_iter1_<ts>.h5                      (iter 1, expected 0.2-1.0M rows)
workspace/datasets/urchin_v3_dagger_iter2_<ts>.h5                      (iter 2, ...)
...
```

Each new file uses the same schema (same datasets, same column names)
so the trainer can concatenate.

### 6.2 Schema additions

Same schema as `record_primitive_dataset.py` (§ docstring) PLUS:

- `/drift_metric` (N,) float32 — the `d_action` value at record time.
  Useful for per-row reweighting at train time and for post-hoc
  threshold calibration.
- `/trigger_kind` (N,) uint8 — index into `attrs["trigger_kinds"]`
  (`action_rmse`, `flipped`, `puck_onset`, `early_trunc`, `always`).
- `attrs["source"] = "corrective_iter<N>"` — lets the trainer
  distinguish iter 0 (`primitive_library_v3`) from corrective batches.
- `attrs["parent_checkpoint"]` — path to the BC checkpoint that drove
  the rollout.
- `attrs["scenario_pack"]`, `attrs["trigger_cfg"]` — enough to
  reproduce.

### 6.3 Row count targets for iter 1

Budget proposal: 4 combos × 4 scenarios × 10 eps × 16 envs × ~300
captured steps ≈ 0.77M rows. At ~200 MB/M rows (similar to Phase 3),
that is ~150–200 MB. This is small enough to iterate on quickly and
large enough to move the needle on F3. If action_rmse triggers rarely
(low trigger rate), `--record-always` on a smaller rollout budget is
the fallback for the first calibration pass.

## 7. Retraining

Extend `train_bc.py` with two small additions:

### 7.1 `--dataset` accepts a list

```
--dataset ds_iter0.h5,ds_iter1.h5[,ds_iter2.h5,...]
```

The trainer concatenates `obs`, `action_sh`, `primitive_id`,
`style_id`, `source` across files. A new `source_weight` map lets us
up-weight corrective rows (e.g. `primitive_library_v3:1.0,
corrective_iter1:3.0`) because they are the rare off-distribution
examples we explicitly want the policy to fit.

### 7.2 `--resume <checkpoint.pt>`

Loads `state_dict`, scaler mean/std, arch — starts from there instead
of random init. Required because the rework plan §12.4 and the Phase 3
writeup both direct us to continue from `phase3_oraclehalf/model.pt`,
not retrain from scratch.

### 7.3 Scaler refit

DAgger by design shifts the obs distribution. Per the
`feedback_scaler_refit_not_bc_pretrain` memory: when only obs
distribution changes across phases, **refit the scaler** (don't
re-pretrain the policy). Default for Phase 4 retrain:
**scaler refit on the combined dataset, policy resumed from
phase3_oraclehalf**.

## 8. Iteration loop for Phase 4

```
iter 0: workspace/checkpoints/bc/phase3_oraclehalf/model.pt      (DONE)
        workspace/datasets/urchin_v3_primitives_oraclehalf_...h5 (DONE)

iter 1:
   (a) --dry-run on long_arc + off_axis_spawn to calibrate threshold
   (b) record_corrective_rollouts.py ... --output iter1.h5
   (c) train_bc.py --dataset iter0.h5,iter1.h5
                   --resume phase3_oraclehalf/model.pt
                   --out-dir workspace/checkpoints/bc/phase4_iter1
                   --epochs 10
   (d) bc_rollout_video.py --checkpoint phase4_iter1/model.pt
       → user video review — specifically inspect arc_{left,right}
                             for clumsiness reduction
   (e) IF user gates pass: proceed. IF clumsy: go to iter 2 with
       adjusted scenarios or trajectory-drift trigger.

iter 2+: same pattern, aggregating all prior HDF5s.
```

Each iteration's video gate is mandatory (per the
`feedback_verify_peak_before_handoff` memory); we do not build iter
N+1 on iter N unless the user approves iter N's video.

## 9. Evaluation

Two metrics per iteration, both on the held-out rollout:

1. **Per-primitive closed-loop `action_rmse`** against supervisor —
   same metric as Phase 3 fidelity eval. Target: `arc_left` and
   `arc_right` drop from ~0.04–0.05 (Phase 3) to <0.025.
2. **Reach events / 12 s** by combo (the same table the user gated
   Phase 3 on). Target: no regression on straight_roll or accelerate;
   improvement or tie on arc_{left,right}.

The video gate is authoritative over the metrics; the metrics are the
cheap sanity check between iterations.

## 10. Out-of-scope for Phase 4

- Anything that touches `rolling_engine.py` (phase structure, support
  logic, etc.) — that's Phase 5+ residual RL territory.
- Anything that touches `primitives.py` to change scripted behavior —
  the supervisor is the fixed reference in DAgger, not a moving
  target.
- Anything that touches `urchin_env_cfg.py` beyond adding scenario
  env-var knobs (spawn/goal distribution) — env physics is frozen.
- Training iteration budget >3 iterations without a plan review. If
  F3 does not resolve within 3 DAgger iterations it is a deeper
  problem (e.g. supervisor itself is wobbly on long arcs, or the
  BC policy architecture under-fits curved state-distributions), and
  we escalate to a design revision rather than throwing more data.

## 11. Open decisions requesting sign-off

Before I write a line of code:

**D1. Scope of iter 1 combos.**
Proposed: `arc_left,arc_right × neutral,snappy` (4 combos), since F3
is the target and straight_roll/accelerate are not F3-problematic
post-retrain. Alternative: full 4 primitives × 3 styles = 12 combos
(dataset-matched; safer but ~3× the rollout time).

**D2. Trigger choice for iter 1.**
Proposed: `action_rmse` alone at threshold ~0.06, calibrated via
`--dry-run`. Alternative: also include `trajectory` drift from day
one (more work; deferred by §3.2).

**D3. Multi-file dataset strategy.**
Proposed: new HDF5 per iteration, train_bc concatenates. Alternative:
make the Phase 3 HDF5 resizable and append in place. I recommend
multi-file for robustness; confirm.

**D4. `--record-always` for iter 1 calibration pass.**
Proposed: do one small `--record-always` run first to see what the
d_action distribution actually looks like on `long_arc`, then lock
threshold. This costs ~10 minutes of rollout time and saves a lot of
"the threshold was wrong and we recorded zero rows" debugging.

**D5. Retraining strategy.**
Proposed: `--resume phase3_oraclehalf/model.pt` + scaler refit on
combined dataset + 10 epochs. Alternative: train from scratch on
combined dataset (safer against catastrophic forgetting, but wastes
the ~80% of rows that are identical to iter 0). Per memory
`feedback_checkpoint_handoff` I need explicit approval before
switching from one to the other.

**D6. Targeted drift scenarios for iter 1.**
Proposed: `long_arc`, `off_axis_spawn`, `combined`, `nominal`. Defer
`mid_rollout_perturbation`, `extreme_commands`, `style_switch` to
iter 2+. Confirm.

**D7. Iteration cap before design review.**
Proposed: 3 iterations max without revisiting this design. If F3
still clumsy after iter 3, the drift detector or supervisor model is
structurally wrong and throwing data is not the fix.

## 12. Deliverables expected at Phase 4 closure

- `record_corrective_rollouts.py` (new, ~400 lines).
- `train_bc.py` additions: `--dataset` multi-file, `--resume`,
  optional `--source-weight` (~60 lines).
- `workspace/datasets/urchin_v3_dagger_iter{1..N}.h5`.
- `workspace/checkpoints/bc/phase4_iter{1..N}/model.pt`.
- Per-iteration `bc_rollout_video.py` output reviewed + gated.
- `docs/urchin_v3_phase4_corrective_loop.md` — closure writeup
  (authored at the END of Phase 4, not now), containing per-iter
  metrics, video verdicts, F3 before/after comparison, and the
  partial-oracle BC checkpoint chosen as the Phase 5 handoff.

## 13. Relation to existing code

No new modules; all new surface area is one script plus small
additions to an existing script. The supervisor IS the existing
`rolling_engine.forward()` + `primitives.PRIMITIVES[name].step()` call
chain that `record_primitive_dataset.py` already uses. The student
IS the existing MLP checkpoint that `bc_rollout_video.py` already
loads. The new script is essentially a merge of those two, with a
drift gate and HDF5 writer added between them.
