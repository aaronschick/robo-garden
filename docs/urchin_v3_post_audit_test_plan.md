# urchin_v3 Post-Audit Test Plan

**Date:** 2026-04-21
**Purpose:** Hand-off for a fresh Claude Code session. The audit (`docs/urchin_v3_pipeline_audit_findings.md`) identified 2 BLOCKERs and 2 HIGH-severity issues in the eval/render/train pipeline; all code patches have been landed (not yet executed). This doc is the test plan that validates the patches and decides the next training direction.

## What just changed (read first)

Patches landed in this session, none executed:

1. **`eval_policy_reward.py`** — rolling_reward formula fixed to post-WP2/WP3 peaked-no-slip (was still pre-WP3 monotone-min, so every "noslip" eval was measuring the wrong reward surface). `final_pos` now reports the true terminal pose (was reporting post-auto-reset spawn). `inputs["observations"]` key normalized. New `--fixed-yaw` diagnostic flag.
2. **`render_policy_video.py`** — one MP4 per episode (was concatenating into one file plus a 1.7 KB placeholder). `--deterministic` no-op replaced with `--stochastic` escape hatch. `inputs["observations"]` key normalized.
3. **`train.py`** — `best_checkpoint.pt` is saved on periodic deterministic eval (64 envs, every `--det-eval-interval-sim-steps`, default 50k), not on a noisy 5-sample stochastic mean. New `--freeze-scaler-after-warmstart` flag prevents scaler drift from poisoning the warmstarted obs normalizer.

Full audit + patch rationale is in `docs/urchin_v3_pipeline_audit_findings.md`. Memory entries to consult:
- `project_urchin_v3_rolling` — task + contactpush gait context
- `project_urchin_v3_bc_reg_sweep_results` — β not the lever
- `project_urchin_v3_noslip_result` — WP2+WP3 failed video review
- `project_skrl2_pipeline_fixes` — [N,1] shapes + observations key gotchas
- `feedback_verify_peak_before_handoff` — always render + user-review new bests

## Environment setup (each terminal session)

```powershell
$env:URCHIN_START_XY = "-0.5,-0.5"
$env:URCHIN_GOAL_XY  = "0.5,0.5"
$env:URCHIN_EPISODE_S = "5.0"
```

These env vars are read at import time by `urchin_env_cfg.py`. If they're not set, the eval/render scripts default to these values anyway (their argparse defaults match), so they're belt-and-suspenders.

## Step 1 — Re-measure existing checkpoints with the fixed pipeline

**Goal:** Confirm the BLOCKER fixes work, and get a clean baseline on the BC seed and the noslip best. All read-only, no training.

### 1a — BC seed under the new reward surface (critical diagnostic)

```powershell
C:/isaac-venv/Scripts/python.exe `
  workspace/robots/urchin_v3/scripts/eval_policy_reward.py `
  --checkpoint workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt `
  --tag bc_new_surface --episodes 5 --seconds 5.0 `
  --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 `
  --json-out workspace/rewards/bc_new_surface/eval.json
```

### 1b — noslip best re-measured with the corrected rolling_reward formula

```powershell
C:/isaac-venv/Scripts/python.exe `
  workspace/robots/urchin_v3/scripts/eval_policy_reward.py `
  --checkpoint workspace/checkpoints/urchin_v3_smoke_noslip/best_checkpoint.pt `
  --tag noslip_fixed_eval --episodes 5 --seconds 5.0 `
  --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 `
  --json-out workspace/rewards/noslip_fixed_eval/best.json
```

### 1c — Same two checkpoints with `--fixed-yaw` (isolates orientation fragility)

```powershell
C:/isaac-venv/Scripts/python.exe `
  workspace/robots/urchin_v3/scripts/eval_policy_reward.py `
  --checkpoint workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt `
  --tag bc_fixed_yaw --episodes 5 --seconds 5.0 --fixed-yaw `
  --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 `
  --json-out workspace/rewards/bc_fixed_yaw/eval.json

C:/isaac-venv/Scripts/python.exe `
  workspace/robots/urchin_v3/scripts/eval_policy_reward.py `
  --checkpoint workspace/checkpoints/urchin_v3_smoke_noslip/best_checkpoint.pt `
  --tag noslip_fixed_yaw --episodes 5 --seconds 5.0 --fixed-yaw `
  --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 `
  --json-out workspace/rewards/noslip_fixed_yaw/best.json
```

### 1d — Sanity: every episode's `final_pos` should differ from spawn

In each eval output, inspect the five `final_pos=(...)` log lines. They MUST vary across episodes (with or without `--fixed-yaw`). If all five are still `(-0.5, -0.5, +0.168)`, Patch #2 didn't land correctly and we need to re-check `_last_root_pos_w` plumbing.

### Expected outputs + how they steer the decision

| Observation | Interpretation | Next direction |
|---|---|---|
| BC new-surface ≈ 600-700 | Reward surface isn't what broke BC behavior; PPO itself was degrading the warmstart | Try `--freeze-scaler-after-warmstart` before WP4 |
| BC new-surface 200-400 | WP2/WP3 made the reward stricter; BC's behavior earns less under it. The 37-at-step-20k ramp is healthy learning, not collapse | Go to WP4 (de-telescope progress) |
| BC new-surface < 100 | WP2/WP3 fundamentally broke BC's reward-correlation — peaked no-slip penalizes the specific gait BC learned | Reconsider WP2/WP3 itself; peaked no-slip may be too sharp |
| noslip_fixed_eval ≪ 260 (old) | Confirms the old rolling_reward formula inflated noslip scores; WP2/WP3 did NOT actually help | Supports the "reward surface is still wrong" thesis |
| `--fixed-yaw` ± stdev ≪ ±323 on noslip | 3/5-vs-2/5 bimodality was orientation-fragility, not reward hacking | Training with a yaw curriculum hold will help; OR live with it if full-yaw still averages decently |
| `--fixed-yaw` ± stdev still huge | Policy is fundamentally unreliable; not just orientation-dependent | WP4 is needed |

Summarize all four JSON outputs in a short table (mean ± std + per-term breakdown) before moving on.

## Step 2 — Re-render noslip best for honest per-episode visual review

```powershell
C:/isaac-venv/Scripts/python.exe `
  workspace/robots/urchin_v3/scripts/render_policy_video.py `
  --checkpoint workspace/checkpoints/urchin_v3_smoke_noslip/best_checkpoint.pt `
  --tag noslip_fixed_render --episodes 3 --seconds 5.0
```

Expect **3 distinct MP4s** (one per episode) in `workspace/checkpoints/policy_video/v3_noslip_fixed_render_*/`. Report the absolute paths per the `feedback_video_artifacts` memory.

User will review; use that verdict as the human tiebreaker on Step 1's data.

## Step 3 — Decision point (human-in-the-loop)

**Do NOT launch another 500k training run until Step 1 + Step 2 are reviewed.** Based on the matrix above, the next action is one of:

- **(a) WP4 — de-telescope progress.** Replace `progress = (prev_dist - dist) * weight` (which telescopes to `start_dist - end_dist` over an episode) with a dense potential-field term `progress = exp(-dist/0.7) * weight`. Full spec in `docs/urchin_v3_reward_fix_plan.md` §WP4. Launch from BC seed, LR=5e-5, 64 envs, 500k, `--freeze-scaler-after-warmstart`, `--det-eval-interval-sim-steps 50000`.
- **(b) Scaler-freeze solo test.** If Step 1 shows BC is still good under the new reward surface but PPO degrades it, try a 500k run that ONLY adds `--freeze-scaler-after-warmstart` (no reward change). Hypothesis: the normalizer drift was the real killer.
- **(c) Back off WP2/WP3.** If Step 1 shows BC scores <100 on the new surface, the peaked-no-slip term is too sharp. Revert to the monotone-min formula or widen `slip_tolerance`.
- **(d) Accept current policy + iterate downstream.** Unlikely given video verdict, but if somehow the noslip best scores well with `--fixed-yaw`, the issue is narrower than "training failed" and we can move to chaining.

Whichever branch is picked, the next 500k should **always** use the new training flags:
- `--det-eval-interval-sim-steps 50000` (so best_checkpoint.pt is saved on a real metric)
- `--freeze-scaler-after-warmstart` (remove the drift variable from the signal)

## Hard constraints (carry forward)

- **8 GB VRAM.** Serialize Isaac Sim runs — NEVER launch eval + render + training in parallel. Monitor for zombie `blender.exe` / `python.exe` / Isaac processes between runs (see `feedback_zombie_processes`).
- **BC seed is sacred.** `workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt` is the only known-good policy. Never overwrite it.
- **Video artifacts**: always preserve rendered MP4s and report absolute paths (`feedback_video_artifacts`).
- **Argparse signed args**: use `--start-xy=-0.5,-0.5` (with `=`), never `--start-xy -0.5,-0.5`.
- **Chain training is off the table** until a single 500k run beats BC visually. Do not touch `scripts/run_urchin_v3_harden_chain.sh` (`feedback_checkpoint_handoff`).
- **Don't trust `best_reward` alone.** Even after Patch #3, every new best checkpoint must be rendered and user-reviewed before handoff (`feedback_verify_peak_before_handoff`).

## Files to reference in the next session

- `docs/urchin_v3_pipeline_audit_findings.md` — full audit
- `docs/urchin_v3_pipeline_inspection_handoff.md` — pre-audit context
- `docs/urchin_v3_reward_fix_plan.md` — WP1-WP4 reward plan spec
- `workspace/checkpoints/urchin_v3_smoke_noslip/result.json` — reward_curve for the failed 500k noslip smoke
- `workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt` — BC seed (det=673 on OLD reward surface)

## What NOT to do

- Do not launch new PPO training in Step 1 or 2. Step 1 and 2 are read-only checkpoint evaluations; the whole point is to re-measure existing artifacts before spending another 500k of GPU time.
- Do not edit code in Step 1 or 2. Patches are landed; only open files to reference them.
- Do not modify `workspace/` contents except within the specific `workspace/rewards/` and `workspace/checkpoints/policy_video/` output directories listed above. These are gitignored output trees.
- Do not re-run the audit agents — the findings doc is authoritative and patches are applied.
