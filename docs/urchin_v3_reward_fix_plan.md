# Urchin v3 Reward Fix Plan

**Status:** WP2+WP3 merged and smoke-tested on 2026-04-20 — **failed video review**. Both best and final checkpoints barely initiate a roll before failing, majorly degraded from the BC seed. WP4 is next. See "2026-04-20 update" section below.
**Date:** 2026-04-20.
**Scope:** Urchin v3 RL training reward in `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py`.

## 2026-04-20 update — WP2+WP3 outcome

Applied WP2 (signed-forward `expected_omega`) + WP3 (peaked no-slip `rolling_reward`) as a single bundle. Ran 500k BC-warmstart smoke at `workspace/checkpoints/urchin_v3_smoke_noslip` (LR 5e-5, 64 envs, no BC-reg).

**Training shape:** peaked at step 173k (stochastic 161), drifted down to ~88 by 500k. Not monotonic.

**Det-eval:**
| Policy | Det mean | rolling_reward | progress |
|---|---|---|---|
| BC seed (ref) | **673** | — | — |
| noslip best | 260 ± 323 | +248 | +8 |
| noslip final | 21 ± 30 | +43 | +0.3 |

**Video review (user verdict):** both videos barely initiate a roll before failing. Majorly degraded from BC. The bimodal scoring on `noslip best` (2/5 seeds ~650, 3/5 ~0) was not sustained rolling — the high-scoring seeds were also transient partial rolls that reward_hacking-adjacent contact-firing can score on. **WP2+WP3 did not unstick the wobble basin.** Progress contributed only ~3% of the 260 best — progress term is too weak; rolling_reward still dominates and is still hackable.

Artifacts:
- `workspace/checkpoints/urchin_v3_smoke_noslip/{best,final}_checkpoint.pt`
- `workspace/rewards/noslip_eval/{best,final}.json`
- `workspace/checkpoints/policy_video/v3_v3_smoke_noslip_{best,final}_*/urchin_v3_policy-episode-1.mp4`

**Pipeline smell surfaced during review:** `eval_policy_reward.py` reported `final_pos=(-0.5,-0.5,+0.168)` on all 5 episodes (identical to spawn), while `render_policy_video.py` step logs suggested 0.37–0.55 m translation. User video confirmed the eval script was effectively right (near-zero sustained translation) and the render step logs were reading transient excursions, not final pose. Full handoff for a fresh inspection session at `docs/urchin_v3_pipeline_inspection_handoff.md`.

**Next action: WP4 (de-telescope progress).** See execution order below — revised to run WP4 next, independent of the WP1 branch logic. The WP2+WP3 result is already sufficient evidence that `progress` needs a restoring-field form before more rolling-term tweaking is worthwhile.


## Why this plan exists

PPO training from the BC-seed checkpoint (`workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt`) cannot produce a rolling policy that beats the BC seed itself. Two successive 500k–2M-step chains on 2026-04-20 both peaked early, drifted down, and produced final policies that visibly roll worse than the BC seed — or, in `harden_run2`, don't roll at all:

- **`harden_run1/final`** — best_reward 199.5. Rolls, but worse than BC.
- **`harden_run2/best`** — best_reward 309.2. Erratic wobble with spinning, does not translate. Confirmed reward-hack.
- **`smoke_vtgate/best`** — best_reward 241.9. "Barely initiates a roll" per 2026-04-20 visual review.

The 2026-04-20 minimal fix (gate `rolling_reward` on signed `vel_toward` instead of `|vel_xy|`) neutered the worst wobble hack but did not close the gap to the BC seed. The reward function needs deeper restructuring.

## Numeric diagnosis

See `urchin_env_cfg.py:346` (`_get_rewards`). Live term contributions in a 5-second episode (300 control steps at 60 Hz):

| Term | Weight | Episode bound | Telescopes? | Notes |
|---|---|---|---|---|
| `progress = (prev_dist - dist) × 40` | 40 | ±40 × net displacement | **yes** | 1.41m task ⇒ ≤ ~56 episode-total. Returning to start zeroes it out. |
| `vel_reward = vel·goal_dir × 2` | 2 | ±2 × net displacement | **yes** | ≤ ~3. Negligible. |
| `dist_pen = -0.05 × dist` | 0.05 | ≈ -15 | no | Negligible. |
| `goal_r = (dist < 0.5) × 500` | 500 | 0 | n/a | Episodes don't reach goal at 0.15 m/s × 5 s. Dead term. |
| **`rolling_reward = min(ang_proj, v/r) × 4 × gate`** | **4** | **~800** | no | Dominates by 10×. |
| `action_rate_pen = -0.03 × Δa²` | 0.03 | tiny | no | |

Rolling reward dominates; the episode signal is effectively `rolling_reward` plus noise.

**Rolling reward landscape** (after 2026-04-20 vt-gate fix):

| Policy | v_toward | ang_proj | speed_gate | rolling_reward/ep |
|---|---|---|---|---|
| Real rolling (BC-like) | 0.15 | 0.88 (≈v/r) | 0.75 | ~790 |
| Slow-translate + modest spin | 0.10 | 0.6 | 0.50 | ~360 |
| Slow-translate + spinning hard | 0.10 | 5.0 | 0.50 | ~354 (capped at v/r) |
| Wobble only, no translation | 0 | 5.0 | 0 | 0 |

Gate-fix correctly zeroes pure wobble. But "slow translate + spin" still scores ~350, above our smoke peak of 242. PPO has an easier attractor than real rolling.

## Critical unknown

**BC seed's own reward under this env has never been measured.** First eval in every training run is at step 20k, already PPO-modified. If BC scores ~800 alone, PPO is destroying the behavior faster than it can evaluate it; if BC scores ~250, the reward is poorly correlated with visual rolling quality. The work plan **must** resolve this first.

## Work packages

Each package below is self-contained and written as a subagent prompt. Spawn via `Agent` with `subagent_type: general-purpose` or `Explore` where indicated. Packages 1–3 can run in sequence; 4 and 5 depend on 1's outcome.

---

### WP1 — Measure BC seed baseline reward

**Goal:** Decide whether the problem is "PPO destroys BC instantly" (BC scores ~800) or "reward is miscalibrated to rolling" (BC scores ~250).

**Agent type:** `general-purpose`.

**Dependencies:** none.

**Prompt:**

> Measure the 5-second episode reward of the BC seed checkpoint under the current urchin_v3 reward function.
>
> Checkpoint: `workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt`
> Env overrides (match training and the BC recording): `URCHIN_START_XY=-0.5,-0.5`, `URCHIN_GOAL_XY=0.5,0.5`, `URCHIN_EPISODE_S=5.0`.
> Reward function lives at `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:346` (`_get_rewards`).
> Existing rollout script: `workspace/robots/urchin_v3/scripts/render_policy_video.py`. It uses deterministic policy mean. Extend it (or write a thin sibling) to also accumulate the per-step reward and print the episode total at the end of each episode. Do not change training behavior.
> Run 5 deterministic episodes. Report per-episode totals and the mean. Save a summary to `workspace/rewards/bc_seed_reward_measurement.md`.
> Then repeat the measurement for `workspace/checkpoints/urchin_v3_harden_run1/final_checkpoint.pt` and `workspace/checkpoints/urchin_v3_smoke_vtgate/best_checkpoint.pt` for side-by-side comparison. Render with the Windows isaac-venv: `C:/isaac-venv/Scripts/python.exe`.
> Output the mean reward table and your one-paragraph interpretation: does the number support "BC should score ~800" or "BC scores ≲ peak PPO"? That determines which WP we run next.

**Deliverable:** `workspace/rewards/bc_seed_reward_measurement.md` with mean episode reward for BC, run1/final, smoke/best. Interpretation paragraph.

---

### WP2 — `expected_omega` uses `vel_toward` (tiny follow-on)

**Goal:** Make the no-slip cap use *forward* translation, not total planar speed, so lateral wander doesn't unlock rolling credit.

**Agent type:** `general-purpose`.

**Dependencies:** WP1 complete (decide whether to run this at all — if BC already scores ~800, this is premature; if BC scores ≲400, proceed).

**Prompt:**

> In `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:427`, the current line is:
> ```python
> expected_omega = speed_xy / self.cfg.ball_radius
> ```
> Change this to use signed forward translation:
> ```python
> expected_omega = vel_toward.clamp(min=0.0) / self.cfg.ball_radius
> ```
> `vel_toward` is already computed at line 374. Update the nearby comment block to match — the cap now says "rolling speed should match forward translation." Do not touch other reward terms.
> Then launch a 500k smoke from BC seed (match the existing `smoke_vtgate` run's invocation: 64 envs, `--learning-rate 2e-5`, same env overrides, output dir `workspace/checkpoints/urchin_v3_smoke_vt_exp_omega`). Stream the reward curve. Render best + final when training exits. Report the reward trajectory, ep-total for best and final, and whether either render visually beats `smoke_vtgate/best`.

**Deliverable:** Code change committed to that one file + a short report in the agent summary with the reward curve and video paths.

---

### WP3 — No-slip rolling reward (rewrite `rolling_reward`)

**Goal:** Replace the monotone `min(ang_proj, v/r)` term with a peaked function that is maximal at `ang_proj ≈ v/r` and decays on both sides. Kills the "spin fast at low speed" attractor structurally.

**Agent type:** `general-purpose`.

**Dependencies:** WP1 complete. Whether to run WP3 *instead of* or *after* WP2 depends on WP1's number and WP2's outcome.

**Prompt:**

> In `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:428`, replace the current rolling_reward block:
> ```python
> rolling_reward = (
>     torch.minimum(ang_proj, expected_omega)
>     * self.cfg.rolling_reward_weight
>     * speed_gate
> )
> ```
> with a no-slip-quality formulation:
> ```python
> # Reward is maximal when ang_proj == expected_omega (pure rolling),
> # and decays smoothly on both sides. Prevents "spin faster than you
> # translate" hacks structurally.
> slip_tolerance = 0.3 * expected_omega + 0.2  # rad/s tolerance band
> slip_error = (ang_proj - expected_omega).abs()
> rolling_quality = (1.0 - slip_error / (slip_tolerance + 1e-6)).clamp(0.0, 1.0)
> rolling_reward = (
>     rolling_quality
>     * expected_omega
>     * self.cfg.rolling_reward_weight
>     * speed_gate
> )
> ```
> Update the comment above the block to describe the no-slip quality design. Preserve the 2026-04-20 vt-gate fix (`speed_gate` uses `vel_toward`).
> Note the old design paid `min(ang_proj, v/r)` regardless of over-spin; the new one pays ~0 for over-spin by double the translational rate. That is the intended tightening.
> Launch a 500k smoke from BC seed (output dir `workspace/checkpoints/urchin_v3_smoke_noslip`, same other args as smoke_vtgate). Stream the reward curve. Render best + final. Report.
>
> Independence note: if WP2 has already been merged, rebase this change on top of it. If not, this change is independent of WP2.

**Deliverable:** Code change + reward curve + render paths + one-paragraph interpretation: does the policy now visually roll better than `smoke_vtgate/best`?

---

### WP4 — De-telescope the progress term

**Goal:** The current `progress` term telescopes (net displacement only), so "go then come back" scores zero. Replace with a *potential-field* shaping that pays every step the policy is near the goal, so sustained progress is rewarded.

**Agent type:** `general-purpose`.

**Dependencies:** WP1 complete. Only run this if WP1 reveals progress/vel_reward shaping is the bottleneck (BC scores ~800 ⇒ PPO is just drifting in a flat reward surface; a non-telescoping shaping gives it a restoring force).

**Prompt:**

> In `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:370`, the current `progress = (prev_dist - dist) * self.cfg.progress_reward_weight` telescopes over an episode, summing to `40 × (d_0 - d_T)` regardless of path. A policy can roll forward and return to start for the same total reward as staying put.
>
> Replace `progress` with an **inverse-distance potential** that pays per-step proximity, not per-step change:
> ```python
> # Potential-field shaping: sustained forward motion is rewarded every
> # step, not telescopically. proximity = exp(-dist/l) in [0, 1].
> proximity_length = 0.7  # meters; task diagonal is 1.41m
> proximity = torch.exp(-dist / proximity_length)
> progress = proximity * self.cfg.progress_reward_weight * 0.05  # scale to ~0.5/step max
> ```
> Delete the `_prev_dist` caching lines that are no longer needed.
> Also in `UrchinEnvCfg` at line 99, lower `progress_reward_weight` from 40.0 to comment-match the new per-step scale (the 0.05 multiplier above already downscales, but consider clean constants).
>
> Run the usual 500k BC-warmstart smoke at `workspace/checkpoints/urchin_v3_smoke_potential`. Report. Pay attention: with potential shaping and no-slip rolling combined, the reward surface has a real gradient toward "fast rolling to goal."

**Deliverable:** Code change + report. Possibly substantial — includes removing `_prev_dist` bookkeeping.

---

### WP5 — BC-regularized PPO (architectural)

**Goal:** If WP1 shows BC scoring ~800 but early PPO scoring ~40, the policy is being destroyed by action noise / observation-scaler bootstrap in the first ~20k steps, before reward can even be measured. Fix by regularizing PPO toward the BC policy with a KL term.

**Agent type:** `general-purpose`.

**Dependencies:** WP1 complete *and* shows BC ≫ early-PPO reward. Only run this if reward fixes (WP2/WP3/WP4) don't close the gap.

**Prompt:**

> Investigate adding a KL-to-BC regularization term to the urchin_v3 PPO training loop. Training script: `workspace/robots/urchin_v3/scripts/train.py`. Uses skrl 2.0 PPO (see `project_skrl2_pipeline_fixes` memory).
>
> Design:
> - Load a frozen copy of the BC seed policy (same `final_checkpoint.pt`).
> - At each PPO update, compute `KL(current_policy || bc_policy)` on the collected states.
> - Add this KL × `beta_bc` to the PPO loss. Choose `beta_bc` such that the KL term is ~10% of the policy-loss magnitude at init.
> - Anneal `beta_bc` from e.g. 0.1 to 0.0 over the first 500k steps.
>
> First do a dry-run code sketch and commit to `docs/urchin_v3_bc_reg_design.md` with the skrl integration points identified (where to hook the loss, how to run the BC network in parallel without blowing VRAM). Do not run training yet — wait for human review.
>
> Why the dry-run first: this is a meaningful training-loop change. A design review prevents wasted RTX 3070 time. Flag any concerns about adding 2× policy forward-pass compute in the update step.

**Deliverable:** `docs/urchin_v3_bc_reg_design.md` with integration plan, estimated compute overhead, and open questions. No training run yet — wait for human approval.

---

## Execution order

**Revised 2026-04-20 after WP2+WP3 video failure.**

1. ~~**WP1**~~ — skipped. BC scores 673 on det-eval under the current reward, matching the "BC ≫ early-PPO" bucket implicitly. The historical question of WP1 (measure BC baseline) is answered by the noslip eval run's ref column.
2. ~~**WP2 + WP3**~~ — merged and smoke-tested. Failed video review (see top of doc).
3. **WP4 (next, priority)** — de-telescope progress. Potential-field shaping `progress = exp(-dist/0.7) * weight`. Rationale: the WP2+WP3 noslip run showed `progress` contributing only ~3% of the best reward (8 of 260) while `rolling_reward` contributed 95%. Even with a better-shaped rolling term, the absence of a sustained-translation gradient means PPO drifts toward reward-hackable contact motion. Progress must be dense and non-telescoping.
4. **Pipeline inspection (parallel)** — `docs/urchin_v3_pipeline_inspection_handoff.md` describes a fresh-session audit of `eval_policy_reward.py`, `render_policy_video.py`, and the train/eval parity surface. This runs in a separate session since it's a different kind of work (read-only audit) from WP4 (reward edit + smoke).
5. **WP5 (BC-reg architectural)** — only after WP4's smoke. If WP4 still cannot beat BC visually, escalate to the BC-reg design doc. Note the β sweep (β=0.1, 0.3) already ran and collapsed, so WP5 would need to revisit the design: either much lower β (0.01–0.05) sustained through the whole run, or the KL-against-BC formulation instead of the action-space L2 that β was parameterizing.
6. Do not chain-train (multi-stage harden chain) until a single 500k smoke beats the BC seed visually. `feedback_verify_peak_before_handoff` memory applies — render and human-review every new best.

## Known traps

- **Don't switch the handoff source between `final_checkpoint.pt` and `best_checkpoint.pt` without explicit user approval** (`feedback_checkpoint_handoff` memory).
- **Don't trust `best_reward`.** `harden_run2/best` scored 309 but visually did not roll. Every new best must be rendered and human-reviewed (`feedback_verify_peak_before_handoff` memory).
- **8 GB VRAM.** Don't run two Isaac Sim instances in parallel. Render-then-train-then-render is the pattern; parallel training + rendering has OOM'd before (`feedback_zombie_processes` memory).
- **Episode length (5 s) may not reach the 1.41 m goal at 0.15 m/s.** `goal_r=500` is currently a dead term. If we want goal-bonus signal, either raise episode time or lower task diagonal.
