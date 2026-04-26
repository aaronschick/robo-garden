# Urchin v3 Pipeline Status — Inspection Handoff

**Date: 2026-04-20**
**Purpose: hand off to a fresh Claude session that will inspect the training/eval/render pipeline for bugs causing persistent difficulty on the urchin_v3 rolling task.**

## What we're trying to do

Train a 42-panel compliant-blob robot (`urchin_v3`) to roll from `(-0.5,-0.5)` to `(0.5,0.5)` on flat ground using PPO in Isaac Lab. A behavior-cloning (BC) seed at `workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt` is the only known-good policy (det-eval reward **673**, visibly rolls). Every PPO fine-tune from that seed has collapsed.

## What we've tried

**1. BC-regularized PPO sweep (2026-04-20, prior session)** — β=0.1 and β=0.3 over 500k steps from the BC seed. Both collapsed on deterministic eval:

| Policy | Det-eval mean |
|---|---|
| BC seed (ref) | +673 |
| β=0.1 best / final | −12.9 / +37.5 |
| β=0.3 best / final | −11.99 / −15.19 |

Checkpoint inspection confirmed log_std did NOT drift (BC=−1.000, β=0.3 final=−1.024), so β is not the lever.

**2. Reward-surface diagnosis** — found the PPO attractor: "slow-translate + moderate spin" scores ~360/ep while real rolling scores ~790/ep. The rolling_reward was `min(ang_proj, expected_omega) * speed_gate`, monotone-increasing in `ang_proj` up to the cap — so "spin hard + translate slow" farmed nearly full credit.

**3. WP2 + WP3 bundled reward fix (this session, 2026-04-20 late)** in `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py`:
- **WP2:** `expected_omega` now uses signed forward translation (`vel_toward.clamp(min=0)/ball_radius`) instead of unsigned `|vel_xy|/r`. Sideways wander can no longer inflate the cap.
- **WP3:** Replaced monotone `min(ang_proj, expected_omega)` with a peaked no-slip-quality term — maxes at `ang_proj == expected_omega` (pure rolling), decays on both sides. `slip_tolerance = 0.3 * expected_omega + 0.2`.

Full diff is in the git-ignored `workspace/` tree; comment block above the rolling section documents both changes.

**4. 500k smoke with new reward** — launched from BC seed, LR=5e-5, 64 envs, NO BC-reg (the sweep proved β isn't the fix). Config: `URCHIN_START_XY=-0.5,-0.5 GOAL_XY=0.5,0.5 EPISODE_S=5.0`. Completed in 13.6 min. Artifacts at `workspace/checkpoints/urchin_v3_smoke_noslip/` (best + final + result.json + train.log).

## Current problems

**Problem 1 — Training shape is unhealthy.** Peak at step 173k (161 stochastic), then drifted down to ~88 by step 500k. Not monotonic. Reward curve in `result.json.reward_curve`.

**Problem 2 — Bimodal det-eval.** Best checkpoint scores **260 ± 323** (huge stdev):

| Policy | Det mean | rolling_reward | progress | Notes |
|---|---|---|---|---|
| noslip best | **260 ± 323** | +248 | +8 | 2/5 seeds ~650, 3/5 seeds ~0 |
| noslip final | 21 ± 30 | +43 | +0.3 | Hard collapse |
| BC seed (ref) | 673 | — | — | Robust |

`rolling_reward` dominates at 248; `progress` contributes only 8 (~3%) on the best — suggesting progress is too weak to force translation.

**Problem 3 — User video verdict (2026-04-20 post-review): both videos barely initiate a roll before failing; majorly degraded from the BC seed.**

This arbitrates the eval/render contradiction below: the **eval script was effectively right** (policy produces near-zero sustained translation), and the render agent's mid-episode step-log positions (0.37–0.55 m "gains") were transient excursions the policy couldn't hold. Net conclusion: WP2+WP3 did **not** unstick the wobble basin. The reward surface is still wrong, OR the pipeline is corrupting the policy between training and eval/render.

Original contradiction (now arbitrated by video review):
- `scripts/eval_policy_reward.py` reports `final_pos=(-0.5,-0.5,+0.168)` on **all 5 episodes** — identical to spawn → zero translation → "rolling_reward firing on in-place motion."
- `scripts/render_policy_video.py` step logs show 0.37–0.55 m translation toward the goal across all 3 episodes → misleading: transient excursions, not sustained rolling.

Pipeline-inspection implication: the render script may still be worth auditing for why its step logs suggested sustained rolling when the video showed a barely-initiated roll. Either the per-step position printout happens at a different lifecycle point than the video frame capture, or the position diff method is noise-dominated.

**Problem 4 — Render script oddity.** `--episodes 3` produces only 2 MP4 files per run: a 1.7 KB placeholder `-episode-0.mp4` and a 1.4 MB `-episode-1.mp4` containing 945 steps (3 concatenated episodes with teleports at t=300, t=600). Not obviously wrong, but the `--episodes` flag semantics are inconsistent.

**Problem 5 — Stochastic-vs-deterministic gap persists.** On β=0.3, training stochastic reached 237 while det-eval was −15 — a 250-unit chasm. On noslip, training peak 161 vs det-eval best 260 is less bad but still weird (det > stochastic suggests the "best" checkpoint saved at step 173k is a different policy than what the curve peak reflects). Potentially the evaluation pipeline is not faithfully reproducing training conditions (domain randomization? action scaling? observation normalization? episode length?).

## What to inspect (handoff request)

Please inspect the following scripts for bugs that could explain the above:

1. **`workspace/robots/urchin_v3/scripts/eval_policy_reward.py`** — does `final_pos` report the spawn or the actual terminal root position? Grep for where it reads the body state and when in the episode lifecycle it samples. Specifically: does it reset the env before logging? Is it reading `env.reset()`'s return instead of the post-rollout state?

2. **`workspace/robots/urchin_v3/scripts/render_policy_video.py`** — why does `--episodes 3` produce only 2 MP4s (one placeholder, one concatenated)? Is there an off-by-one on video writer init, or does each `env.reset()` mid-rollout not trigger a new video file?

3. **Determinism / scaling parity between train & eval** — confirm train.py and eval_policy_reward.py use the same:
   - Observation normalization (running mean/var, clipping)
   - Action scaling / clipping
   - Domain randomization (should be OFF for eval; confirm)
   - Episode length (`URCHIN_EPISODE_S`)
   - Physics substeps

   A 250-unit stochastic/deterministic gap is not normal PPO exploration noise.

4. **PPO stability during training** — why does the policy peak at 35% of training then regress? Things to sanity-check:
   - Is `best_checkpoint.pt` being overwritten only on improvement, and does `best_reward` match the curve max? (result.json says yes for this run: 160.68.)
   - Is there a learning rate or entropy schedule that changes mid-run?
   - Is the reference BC policy being re-loaded somewhere during training (KL target drift)?

5. **Reward term accounting** — the eval agent reported per-term breakdowns (`rolling_reward`, `progress`, `vel_reward`, `dist_pen`, `action_rate_pen`). Confirm these match the terms actually summed in `_get_rewards` in `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py`. If there are terms present in training that eval doesn't log (or vice versa), that's a divergence that could explain the stochastic/det gap.

## Plans to move forward (pending inspection)

Video review has ruled out Plan C (visible rolling + drift). The policy is degraded, not just mis-measured.

**Plan A — Pipeline inspection (proceed regardless)**: even with the user's verdict that the policies are degraded, the eval/render script discrepancies still warrant audit. Stochastic-vs-deterministic gaps of 250 units and `final_pos == spawn_pos` reports across 5 independent seeds are pipeline smells that would make *any* future training loop untrustworthy. Findings from inspection will either:
1. Explain why every PPO run from the BC seed has collapsed despite reward fixes (e.g., obs-normalization reset on load, action-scaling mismatch between train and eval, BC seed being loaded with wrong shape/device), in which case the fix is a pipeline patch, not more reward engineering; or
2. Confirm the pipeline is clean, in which case Plan B is the only path forward.

**Plan B — WP4 (de-telescope progress)**: current `progress = (prev_dist - dist) * weight` telescopes to `start_dist − end_dist` over an episode — a policy that rolls forward 0.3m then slides back pays the same as staying put. Replace with dense potential-field shaping `progress = exp(-dist/0.7) * weight` so every step near the goal pays. This gives PPO a restoring gradient toward sustained forward motion — currently the only restoring force is `rolling_reward`, which WP2+WP3 confirmed is still reward-hackable even after the peaked-no-slip fix. Full spec at `docs/urchin_v3_reward_fix_plan.md` WP4.

**Plan C (deprecated)** — was "if videos show visible rolling but training drift persists, add LR decay or low-β BC-reg." Video verdict rules this out.

## Critical files

- `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py` — reward function (WP2+WP3 applied at the rolling block)
- `workspace/robots/urchin_v3/scripts/train.py` — PPO entrypoint
- `workspace/robots/urchin_v3/scripts/eval_policy_reward.py` — deterministic per-term eval **(suspect)**
- `workspace/robots/urchin_v3/scripts/render_policy_video.py` — MP4 render **(minor suspect)**
- `workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt` — BC seed (only known-good, det=673)
- `workspace/checkpoints/urchin_v3_smoke_noslip/{best,final}_checkpoint.pt` — noslip smoke artifacts
- `workspace/rewards/noslip_eval/{best,final}.json` — det-eval JSONs
- `workspace/checkpoints/policy_video/v3_v3_smoke_noslip_{best,final}_*/urchin_v3_policy-episode-1.mp4` — rendered videos

## Memory-backed constraints (do not violate)

- **Don't switch chain handoff source between final/best without explicit user approval** (`feedback_checkpoint_handoff`).
- **Don't trust best_reward scalar — render and human-review every new best** (`feedback_verify_peak_before_handoff`). harden_run2/best scored 309 but was visibly wobbling.
- **8 GB VRAM** — no parallel Isaac Sim instances. Train → eval → render in series.
- **Argparse signed-arg bug** — use `--start-xy=-0.5,-0.5` (with `=`), not `--start-xy -0.5,-0.5`.
- **Chain training is off the table** until a single 500k smoke beats BC visually.
