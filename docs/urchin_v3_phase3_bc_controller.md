# Phase 3 — BC Controller: Fidelity Review & Failure Catalog

Status: **CLOSED (2026-04-24)** — partial-oracle retrain passed user
video gate. 1st pass (2026-04-23, oracle-full) logged F1/F2/F3
failures; scripted-oracle render gate diagnosed F1+F2 as upstream
panel-clamp saturation (not BC conditioning); 2nd pass retrained at
`URCHIN_ORACLE_AMPLITUDE=0.5` with residual=1.0 resolved F1 (snappy=3
reaches vs lazy=1-2 in every primitive family). F3 (arc clumsiness) is
deferred as the canonical Phase 4 DAgger target. Fidelity eval was run
only against the oracle-full checkpoint and is not re-run for the
partial-oracle retrain — the closed-loop reach-count table in the user
gate is treated as sufficient for Phase 3 closure.

## Summary

Phase 3 of the urchin_v3 control rework (see
`urchin_v3_control_rework_plan.md` §12.3). A compact MLP policy was
behavior-cloned against the 4.15M-row primitive-family dataset recorded
in the prior turn, producing an imitation controller that takes an obs +
primitive-one-hot + style-one-hot and emits the 9-D SH action the
scripted primitive would have.

## Deliverables

| Artifact | Path |
|----------|------|
| Training script | `workspace/robots/urchin_v3/scripts/train_bc.py` |
| Fidelity eval | `workspace/robots/urchin_v3/scripts/eval_bc_fidelity.py` |
| Rollout video script | `workspace/robots/urchin_v3/scripts/bc_rollout_video.py` |
| Dataset | `workspace/datasets/urchin_v3_primitives_phase3.h5` (3.3 GB, 4.15M rows) |
| Checkpoint | `workspace/checkpoints/bc/phase3/model.pt` |
| Training history | `workspace/checkpoints/bc/phase3/history.json` |
| Fidelity JSON | `workspace/eval/bc_phase3/fidelity.json` |
| Per-combo videos | `workspace/checkpoints/bc_rollout_video/v3_bc_<ts>/<prim>__<style>.mp4` |

## Policy Architecture

MLP, 137 obs + 7 cond (4 primitive one-hot + 3 style one-hot) → 256 →
256 → 9 action_sh. 105,225 parameters. RunningStandardScaler-style
whitening on obs (mean/std fit on the train split of the dataset).
AdamW lr=3e-4, weight_decay=1e-5, batch=1024.

## Training Curve (15 epochs)

| metric | epoch 1 | epoch 15 | reduction |
|--------|---------|----------|-----------|
| train MSE | 9.04e-4 | 2.09e-4 | 4.3× |
| val MSE | 4.34e-4 | 2.15e-4 | 2.0× |

Train-val gap is ~3% at the end — no overfitting, but diminishing
returns past ~10 epochs. Curve stayed monotone-decreasing on val MSE.

## Per-combo val MSE (best epoch)

All 12 combos within 1.2e-4 to 2.7e-4 range — well balanced, no combo
is structurally harder than the others. `arc_right:lazy` is easiest
(1.22e-4) and `accelerate:snappy` is hardest (2.70e-4), which tracks
intuition (lazy = small amplitude, snappy + accelerate = large
amplitude-varying targets).

## Fidelity Eval (closed-loop rollout)

Method: for each combo, drop the BC policy into UrchinEnv (16 envs,
6 s = 360 steps, arena sampler at half-extent 2.0 m, fixed
primitive + style conditioning), step the env with the BC action, and
at every step also compute the action the *scripted* primitive would
have produced on the same obs. Two metrics per combo:

- `action_rmse` = sqrt(mean((bc_sh - scripted_sh)^2) across elements) —
  closed-loop fidelity metric. Low = BC stays behaviorally close to
  scripted under the state distribution BC itself induces.
- `reach_rate_per_env` = goal-reach events divided by num_envs. Since
  each env's 6 s window can contain multiple reach+reset cycles,
  rates > 1.0 are possible for fast primitives.

**Overall:** `action_rmse_mean = 0.0375`, `reach_rate_mean = 0.58/env`.

For reference, open-loop dataset val MSE at best epoch = 2.15e-4 per
element → per-element closed-loop MSE is ≈ 1.4e-3, a ~6.5× blow-up
typical of BC state-distribution drift.

| combo | action_rmse | reach/env | reach events | timeouts | notes |
|-------|-------------|-----------|--------------|----------|-------|
| straight_roll:neutral | 0.0156 | 0.75 | 12 | 5 | best mix of fidelity + reach rate |
| straight_roll:lazy    | 0.0117 | 0.50 | 8  | 8 | lowest rmse overall (lazy = small amp) |
| straight_roll:snappy  | 0.0188 | 0.62 | 10 | 8 | higher amp → modestly higher rmse |
| arc_left:neutral      | 0.0253 | 0.50 | 8  | 8 | arcs harder than straight |
| arc_left:lazy         | 0.0241 | 0.31 | 5  | 12 | **lowest reach rate** — lazy arc may not bend enough to hit goals |
| arc_left:snappy       | 0.0473 | 0.62 | 10 | 7 | |
| arc_right:neutral     | 0.0426 | 0.88 | 14 | 5 | |
| arc_right:lazy        | 0.0329 | 1.00 | 16 | 4 | **highest reach rate** (16/16) — arena geometry favors arc_right on this seed |
| arc_right:snappy      | 0.0597 | 0.56 | 9  | 8 | |
| accelerate:neutral    | 0.0562 | 0.44 | 7  | 9 | accelerate has hardest fidelity (time-varying target) |
| accelerate:lazy       | 0.0508 | 0.38 | 6  | 10 | |
| accelerate:snappy     | 0.0646 | 0.44 | 7  | 9 | **highest rmse** — largest amplitude-varying target |

Asymmetry note: `arc_left` reach rate (0.48 mean) is considerably
lower than `arc_right` (0.81 mean) despite similar per-combo dataset
balance. This is likely a seed/geometry artifact with num_envs=16 and
a single episode per combo — not a structural BC failure. Re-running
with `--episodes-per-combo 3+` would tighten the estimate.

Raw results: `workspace/eval/bc_phase3/fidelity.json`.

## Visual Review

Rendered batch: `workspace/checkpoints/bc_rollout_video/v3_bc_<ts>/`
(per-combo mp4, 12 s each, `--sampling-mode ring` anchored at
start=(0,0) → goal=(3,0) so the arc/accelerate geometry is visible
without per-reset resampling). Regenerate via:

```
C:/isaac-venv/Scripts/python.exe \
    workspace/robots/urchin_v3/scripts/bc_rollout_video.py \
    --checkpoint workspace/checkpoints/bc/phase3/model.pt \
    --out-dir workspace/checkpoints/bc_rollout_video \
    --seconds 12.0 --sampling-mode ring \
    --start-xy 0.0,0.0 --goal-xy 3.0,0.0
```

Review criteria per mp4:
- **straight_roll**: body rolls forward toward goal; no flip/wobble.
- **arc_left / arc_right**: curl toward the intended side is visible
  over the full 12 s (not just a drift).
- **accelerate**: speed grows during the rollout (ramp behavior).
- **styles**: `lazy` has visibly smaller amplitude than `neutral`;
  `snappy` has larger and more abrupt amplitude than `neutral`.

## Failure Catalog

Verdict from the 2026-04-23 user video review of all 12 mp4s in
`workspace/checkpoints/bc_rollout_video/v3_bc_20260423_232915/`:

### F1 — style mode-collapse on straight_roll

- **Observed:** `straight_roll:neutral`, `:lazy`, and `:snappy` all
  roll at roughly the same speed. Lazy is not visibly gentler and
  snappy is not visibly more aggressive.
- **Hypothesis:** Two candidates — not yet distinguished.
  1. BC conditioning collapse: the 3-D style one-hot is being
     under-used; the 105K-param MLP may have averaged across styles
     because the training loss per style is small relative to the
     primitive-selection signal.
  2. Upstream weakness: the scripted `with_style(straight_roll, ...)`
     itself may not produce visibly different motions, so BC learned
     a legitimate averaging. The primitive recording loop would need
     to be replayed to rule this out.
- **Phase 4 action:** First render the scripted oracle through
  `scripted_roll_video.py` for `straight_roll:{lazy,neutral,snappy}`
  before any DAgger. If the oracle shows three distinct speeds, the
  defect is BC conditioning → DAgger with style-stratified corrective
  rollouts. If the oracle looks the same three-ways, the defect is
  in `primitives.with_style` / `STYLES` gains → rework the style
  preset deltas and re-record the dataset.

### F2 — accelerate does not visibly accelerate

- **Observed:** `accelerate:{neutral,lazy,snappy}` all look like a
  slightly-slower `straight_roll`; speed does not visibly ramp over
  the 12 s rollout.
- **Hypothesis:** Consistent with the metric flag
  (highest action_rmse overall: 0.05–0.065). Most likely a BC
  conditioning collapse — the `accelerate` one-hot is being treated
  as near-identical to `straight_roll` — but upstream weakness in
  the `accelerate` primitive itself (ramp gain too small to read
  visually) is also possible.
- **Phase 4 action:** Same split as F1 — render scripted oracle
  first. If oracle ramps visibly, DAgger with an explicit
  `accelerate` supervisor that holds the ramping cfg for the full
  episode. If the oracle itself does not ramp, rework the
  primitive's amplitude schedule in `primitives.accelerate` before
  any retraining.

### F3 — arc clumsiness

- **Observed:** `arc_left` and `arc_right` *do* curl toward their
  intended side (so no conditioning collapse on the primitive axis),
  but the motion is "clumsy" — wobbly / not clean.
- **Hypothesis:** Classic BC state-distribution drift. Small
  per-step action errors compound during curved trajectories
  (the state distribution off the arc was underrepresented in the
  dataset). This tracks the ~6.5× per-element closed-loop MSE
  blow-up vs open-loop val MSE reported in §Fidelity Eval.
- **Phase 4 action:** Textbook DAgger use case — supervisor queries
  whenever the rollout position deviates from the ideal arc
  trajectory by a threshold, aggregate into the dataset,
  retrain from this BC checkpoint.

### Not failing (contradicts earlier metric flags)

- `arc_left:lazy` — despite metric flag (reach 0.31, lowest in the
  eval), the video shows visible left curl. The low reach rate was
  likely a seed/geometry artifact of `num_envs=16 × episodes=1`,
  not a BC defect. Re-eval with `episodes_per_combo ≥ 3` would
  confirm.
- `arc_{left,right}:snappy` — the metric flag was on rmse
  (0.047 / 0.060). Video shows they curl. Classified under F3 with
  the rest of the arcs — part of the clumsy-arc population, not a
  distinct failure.

### Failure priority for Phase 4

1. **F2 accelerate** — primitive-level capability gap; without it the
   controller has only one effective speed.
2. **F1 straight_roll styles** — the point of the style layer is
   character differentiation; this is what makes Phase 6 viable.
3. **F3 arc clumsiness** — quality issue on a working behavior; the
   standard DAgger path should handle it.

## 2026-04-24 Retrain — F1/F2 Diagnosis & Partial-Oracle Fix

Phase 3's F1 + F2 failures were classified via the scripted-oracle
render gate (rework plan 2026-04-23 directive: render oracle before
DAgger). Three render runs isolated the cause:

| Run | `URCHIN_ORACLE_AMPLITUDE` | `URCHIN_RESIDUAL_SCALE_{INIT,FINAL}` | Result |
|-----|---------------------------|--------------------------------------|--------|
| `v3_oracle_20260424_001830`    | 1.0 | 0.3 / 1.0 | oracle saturates 42-D panel to ±1.0 → residual carries ~0 visible signal. Styles all identical. |
| `v3_oracleoff_20260424_005158` | 0.0 | 1.0 / 1.0 | `straight_roll:lazy` stiction-locks at 5 cm for the full 12 s (pos=+0.051, dist=2.949). Snappy rolls; neutral rolls; lazy dies. |
| `v3_oraclehalf_20260424_010101`| 0.5 | 1.0 / 1.0 | lazy rolls; joint-level amplitudes differ (jp_mean: lazy=18 mm, neutral=22 mm, snappy=24 mm). Net COM speed stays roughly flat — the ±1 panel clamp caps it. |

**Root cause.** `rolling_engine.py:256` clamps the 42-D panel output to
`[-1,+1]`, and `compute_contactpush_oracle(..., amplitude=1.0)` is
calibrated to fill that entire range. The env then sums
`combined_raw = tau_oracle + residual_scale * residual_raw` and clamps
again (`urchin_env_cfg.py:806`). So whenever the oracle runs at full
amplitude the residual has zero headroom — F1/F2 were **not** BC
conditioning collapse, they were an upstream architectural saturation.

**Chosen fix.** Record the dataset at `URCHIN_ORACLE_AMPLITUDE=0.5` +
`URCHIN_RESIDUAL_SCALE_{INIT,FINAL}=1.0`. Half-amplitude oracle gives
lazy a baseline to ride on (avoids the stiction-lock) while leaving
enough headroom for snappy / accelerate residuals to show through.
This accepts the physical ceiling on net COM-speed differentiation
between styles and instead exploits the joint-level differentiation
the oracle+residual stack can actually express.

### Retrain deliverables (2026-04-24)

| Artifact | Path |
|----------|------|
| Dataset | `workspace/datasets/urchin_v3_primitives_oraclehalf_20260424_012337.h5` (3.15 GB, 4.15 M rows) |
| Smoke dataset | `workspace/datasets/urchin_v3_primitives_oraclehalf_smoke.h5` (56 MB, 69 K rows) |
| Checkpoint | `workspace/checkpoints/bc/phase3_oraclehalf/model.pt` (417 KB) |
| Review videos | `workspace/checkpoints/bc/phase3_oraclehalf/review_20260424_032933/v3_bc_20260424_032958/` (12 mp4s + summary.json) |

Dataset HDF5 `randomization_spec` now includes the three
`URCHIN_*` overrides so Phase 4 can identify the oracle configuration
the BC controller was trained against.

### BC retrain training curve

Final **val_mse = 1.46 × 10⁻⁴** (vs 2.15 × 10⁻⁴ on the oracle-full
dataset — 32% lower). Training curve stayed monotone-decreasing over
15 epochs. No overfit.

### Retrain review — reach events per 12 s rollout

Rendered via `bc_rollout_video.py --sampling-mode ring
--start-xy 0,0 --goal-xy 3,0` (single episode per combo, 12 s each).
"Reach events" = distinct goal-reach + reset cycles within the 12 s
window; higher = faster effective locomotion.

| combo | lazy | neutral | snappy |
|-------|------|---------|--------|
| straight_roll | 2 | 2 | **3** |
| arc_left      | 1 | 2 | **3** |
| arc_right     | 2 | 2 | **3** |
| accelerate    | 1 | 2 | **3** |

**Every primitive family shows snappy > neutral ≥ lazy by reach count.**
This is the first time Phase 3 has shown this clean a style
differentiation — F1 is resolved in the direction the partial-oracle
hypothesis predicted. Awaiting user visual gate for Phase 3 closure.

## Next: Phase 4 Corrective Loop

From the rework plan §12.4: once a failure catalog exists, the Phase 4
subtasks are:
1. failure detector instrumentation
2. targeted drift scenarios
3. scripted corrective supervisor queries
4. DAgger-style aggregation
5. retraining iterations

Phase 4 should re-use **the partial-oracle BC checkpoint**
(`workspace/checkpoints/bc/phase3_oraclehalf/model.pt`) as the starting
policy and append corrective rollouts to the same HDF5 (schema is
resizable). F3 (arc clumsiness) remains the canonical Phase 4 target —
the partial-oracle retrain did not address state-distribution drift on
curved trajectories, and it should not, because that is genuinely the
DAgger job.
