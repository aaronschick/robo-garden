# Velocity-Clamped Gait Re-Simulation — Reproducible Procedure (Agent 3)

**Purpose.** Determine whether the `compute_contactpush_oracle` rolling gait
survives realistic off-the-shelf linear-actuator velocity caps (0.4, 0.2,
0.1 m/s). Output feeds Agent 2's actuator shortlist: if the gait rolls at
0.1 m/s, hobby lead-screws are on the table; if it only rolls at 0.4 m/s,
we need voice coils.

The current sim pins per-panel velocity at **1.5 m/s** (a sim-side safety
margin — the rolling gait's measured `jv_max` is in the 0.3–0.5 m/s band
and `jv_mean` ≈ 0.04–0.10 m/s). This procedure re-runs the scripted oracle
(not a policy checkpoint) with `velocity_limit` clamped, captures the
same telemetry, and decides pass/fail per cap.

---

## 1. Config edit — single line

**File:** `workspace/robots/urchin_v3/urchin_v3/urchin_v3_cfg.py`

**Line:** 110 (inside the `ImplicitActuatorCfg(...)` block for `"panels"`).

`velocity_limit` on `ImplicitActuatorCfg` is the correct knob — Isaac Lab's
implicit actuator clamps joint-velocity command at this value before the
PhysX PD produces effort, exactly like a servo driver's speed loop. The
`effort_limit=60.0` field is unrelated (it's stall force) and should be
left alone for this test.

**Before (current, line 109–110):**
```python
effort_limit=60.0,
velocity_limit=1.5,
```

**After (example for the 0.2 m/s cap):**
```python
effort_limit=60.0,
velocity_limit=0.2,
```

Rerun with three separate values: `1.5` (baseline control), `0.4`, `0.2`,
`0.1`. Edit, run, edit back. Do **not** commit any of these values —
revert to `1.5` when done.

**Why not an env-var / CLI knob?** `ImplicitActuatorCfg` is frozen at
`URCHIN_V3_CFG` import time (module-level ArticulationCfg). Plumbing a
runtime override would touch 3 files. A four-line manual edit is faster
and lower-risk for a one-off feasibility test.

---

## 2. Reproducible run command

The existing `scripted_roll_video.py` already:
- drives `compute_contactpush_oracle` via `--mode contactpush`,
- logs `jv_max` / `jv_mean` (panel joint velocity) and `speed` every
  30 steps,
- renders an mp4 so you can eyeball rolling vs. stalling.

From Windows PowerShell at the repo root, one line per cap:

```powershell
$env:URCHIN_START_XY = "-0.5,-0.5"
$env:URCHIN_GOAL_XY  = "1.5,-0.5"   # pure +x, 2 m run, avoids diagonal
$env:URCHIN_EPISODE_S = "10.0"

# After editing velocity_limit in urchin_v3_cfg.py to the target value:
& C:/isaac-venv/Scripts/python.exe `
    workspace/robots/urchin_v3/scripts/scripted_roll_video.py `
    --mode contactpush --baseline-hz 0 `
    --seconds 10.0 --episodes 1 `
    --start-xy -0.5,-0.5 --goal-xy 1.5,-0.5 `
    --amplitude 1.0
```

`contactpush` is closed-loop (no phase wave), so `--baseline-hz 0` is
correct — confirmed by the scripted_roll_video help text. `--episodes 1`
keeps each clamp-test under 2 minutes wall time on a 3070.

Capture stdout to a file per cap so telemetry survives:

```powershell
... | Tee-Object -FilePath "workspace/_tasks_out/vclamp_v0p2_stdout.log"
```

Videos land in
`workspace/checkpoints/scripted_roll_video/v3_<timestamp>/urchin_v3_scripted-*.mp4`
— report the absolute path to the user (memory: video-artifacts rule).

---

## 3. Pass/fail metrics (all already logged)

Every 30 env-steps (= 0.5 s), `scripted_roll_video.py` prints:

| Logged field | What it measures | Threshold |
|---|---|---|
| `pos=(x, y, z)` | shell CoM world position | **PASS** if `|Δx| ≥ 1.0 m` at `t = 10 s` (starting from −0.5, −0.5 toward +1.5, −0.5) |
| `speed` | `‖v_xy‖` | **PASS** if sustained `speed ≥ 0.15 m/s` for ≥ 50% of run (avoids counting a brief lunge) |
| `jv_max` | peak per-panel joint velocity that step | Diagnostic: should sit ≈ at `velocity_limit` if the clamp is biting |
| `jv_mean` | mean \|panel velocity\| that step | Diagnostic: compare to unconstrained baseline ≈ 0.04–0.10 m/s |

**Additional computed check (ang-vel / v alignment).** The env reward code
at `urchin_env_cfg.py:605-606` defines `expected_omega = vel_toward / r`
with `r = 0.17 m`. We're not reading rewards here, but if you want the
rolling-quality signal, read `robot.data.root_ang_vel_w` after each step
and project onto `z × goal_dir_w`; ratio of that to `speed / 0.17`
should be ≈ 1.0 for true rolling vs. ≪ 1.0 for sliding/dragging. Leave
this for follow-up; the `|Δx| ≥ 1 m in 10 s` criterion already
discriminates pass/fail cleanly.

**Summary pass criterion:** robot translates ≥ 1 m along the goal axis in
10 s **and** `jv_max` pinned at the cap (proving the clamp was active,
not a no-op). Fail = stalls, wobbles, or `jv_max` barely touches the
cap (gait degenerated into low-velocity creep).

---

## 4. Expected outcomes — honest prediction

| Cap | Prediction | Reasoning |
|---|---|---|
| **1.5 m/s** (baseline) | PASS, `Δx ≈ 1.4–1.8 m`, `jv_max ≈ 0.4–0.5 m/s` | Matches Phase 1 smoke-approved behaviour. Clamp never binds. |
| **0.4 m/s** | **Likely PASS**, possibly with ~10–20 % speed loss | Baseline `jv_max` is already near 0.4 m/s during the dipole-alignment transition; clamp will bite only during transients. Mean panel velocity (0.04–0.10 m/s) is ~4× under the cap, so steady-state gait is unaffected. |
| **0.2 m/s** | **Marginal — probably partial**. Expect ~50 % forward progress (0.5–0.7 m in 10 s). | The dipole front/back-swap transient needs panels to traverse from ~0 mm to ~60 mm (or back) in roughly one baseline-hz period (~1 s at the default rate, faster under closed-loop `contactpush`). Clamp at 0.2 m/s means ≥ 0.3 s minimum transit per swap, which the contactpush oracle's contact-window phasing may not wait for. Result: lagged panel response → weaker contact impulse → Weeble lean rather than over-the-top roll. |
| **0.1 m/s** | **Likely FAIL.** Stalls in a ~2–4 cm Weeble tilt similar to the under-effort iterations mentioned in `urchin_v3_cfg.py:107-108`. | Full 60 mm stroke would take ≥ 0.6 s. The rolling cycle needs sub-200 ms contact impulses to beat the spring's settling. Expect `Δx < 0.3 m`. |

**Decision tree for Agent 2:**
- PASS at 0.2 m/s → hobby lead-screws (Actuonix L16-series at higher
  gear ratios, ~0.15–0.3 m/s) may suffice. Cost story is easy.
- PASS only at 0.4 m/s → voice coils or custom BLDC-driven linear stages
  required. Cost story gets expensive.
- FAIL even at 0.4 m/s → gait must be re-designed (higher gear ratio +
  lower-hz wave, or switch to impulsive `legpush` which uses shorter
  strokes). Unlikely but worth flagging.

---

## 5. Caveats

- **Headless vs. viewer.** `scripted_roll_video.py` sets
  `args.headless = True` and `args.enable_cameras = True`
  unconditionally. The MP4 is written; no Isaac Sim viewport opens.
  Faster and matches what CI-like smokes do. If you want to watch live,
  run through the Studio UI instead (outside the scope of this test).
- **Render cost.** Each 10 s episode takes ~45–75 s wall time on an
  RTX 3070 (render dominates on a 1-env sim). Total for the four caps
  (incl. 1.5 m/s baseline) ≈ 5 min.
- **No checkpoint needed.** This is scripted / oracle — no BC seed, no
  PPO policy. Skip the `SEED_CKPT` knob entirely. That's the whole
  point: we're isolating the *gait physics* from any learned residual.
- **Zombie processes.** Isaac Sim has a known habit of leaving
  `python.exe` / `kit.exe` alive after os._exit. After each run,
  Task-Manager-kill any leftover `python.exe` eating > 1 GB before
  starting the next cap (per memory: zombie-processes rule).
- **Black-frame videos.** If rendered mp4s are ~26 KB with black frames,
  that's GPU/Vulkan residue — kill all Isaac processes, restart, re-run
  (memory: render-after-chain-gpu rule).
- **Revert the edit.** After the final run, restore
  `velocity_limit=1.5` in `urchin_v3_cfg.py`. Do **not** commit a
  velocity-clamped config — it would silently cripple every subsequent
  training run.
- **One-at-a-time.** Don't try to parameterize `velocity_limit` over
  multiple simultaneous envs in one run. The field is frozen at config
  time and shared across all panels; a loop would need to rebuild the
  URDF articulation between iterations, which Isaac Lab doesn't support
  cleanly.
