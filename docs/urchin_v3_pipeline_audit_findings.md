The urchin_v3 rolling task has failed three successive 500k PPO fine-tunes from the
 BC seed (workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt,
 det-eval=673). The BC-reg β sweep (0.1, 0.3) and the WP2+WP3 peaked-no-slip
 reward rewrite all produced policies that user-reviewed as "barely initiates a
 roll before failing." Two pipeline smells block reward-engineering iteration:

 1. A 250-unit gap between stochastic training reward and deterministic eval
 reward on the same checkpoint (β=0.3: 237 train vs −15 eval).
 2. eval_policy_reward.py reports final_pos=(-0.5,-0.5,+0.168) identical to
 spawn on all 5 episodes of all runs, while render_policy_video.py
 step-log positions suggested 0.37–0.55 m translation. Videos confirmed
 near-zero sustained translation.

 Before running WP4 (de-telescope progress), we need confidence the measurement
 pipeline isn't lying. This doc is a read-only audit of eval/render/train scripts
 and their parity to identify BLOCKER / HIGH / LOW issues and a prioritized patch
 list.

 Final deliverable will be at docs/urchin_v3_pipeline_audit_findings.md
 (copied out of this plan after approval — cannot write there from plan mode).

 ---
 1. scripts/eval_policy_reward.py — final_pos, term accounting, lifecycle

 1.1 BLOCKER: final_pos is post-auto-reset, not episode terminal

 eval_policy_reward.py:374
 last_pos = env.scene["robot"].data.root_pos_w[0].cpu().tolist()
 runs immediately after env.step(action) inside the per-step loop.

 Isaac Lab's DirectRLEnv.step() lifecycle
 (isaaclab/source/isaaclab/isaaclab/envs/direct_rl_env.py:391-418) is:

 1. _get_dones()          → sets reset_terminated, reset_time_outs
 2. _get_rewards()        → uses pre-reset state (correct)
 3. _reset_idx(reset_ids) → auto-reset writes spawn pose to root_pos_w
 4. _get_observations()
 5. return (obs, reward, terminated, truncated, extras)

 So on the final step of any truncated/terminated episode (100% of 5 s timeout
 episodes), robot.data.root_pos_w is the newly-reset spawn pose when the
 eval loop reads it. This fully explains the final_pos = (-0.5, -0.5, +0.168)
 report on every episode — it is the reset target in _reset_idx
 (urchin_env_cfg.py:505-507: default[:,0]=start_x; default[:,1]=start_y; default[:,2]=0.17).

 Consequence: the final_dist printed in the eval log is always
 ‖spawn − goal‖ = √2 ≈ 1.414. It is not a diagnostic, it is spawn geometry.

 Crucially, the per-episode reward totals and per-term breakdowns are
 NOT affected. The reward is computed inside _get_rewards(), which runs
 before auto-reset, on the true terminal state. The 260±323 on best /
 21±30 on final and the per-term splits (rolling_reward=+248,
 progress=+8) are genuine.

 Patch: capture the true terminal root_pos_w inside the instrumented
 _get_rewards (which sees pre-reset state) and read it after the loop.

 Proposed diff (in the instrumented rewards, near line 207):
 self._prev_dist = dist.detach()
 self._last_root_pos_w = robot.data.root_pos_w.detach().clone()   # NEW
 and in the main loop, replace line 374:
 last_pos = env._last_root_pos_w[0].cpu().tolist()

 Severity: BLOCKER. Our only per-episode spatial-progress metric is
 lying. Without this fix we cannot diagnose "did the policy actually
 translate?" — we have to rely on user video review for every eval.

 1.2 HIGH: _prev_action / _prev_dist mutated by instrumented rewards — not tracked by _reset_idx

 The instrumented _get_rewards (lines 139–151, 207) re-creates self._prev_action
 and self._prev_dist tensors on the env instance. _reset_idx
 (urchin_env_cfg.py:528-529) reassigns these on reset ONLY IF they already exist.

 Order of operations in eval:
 1. env = UrchinEnv(...) → _reset_idx runs at init before instrumentation;
 neither attribute exists yet.
 2. _instrument_env(env) replaces _get_rewards.
 3. First step creates _prev_dist and _prev_action (lines 106-107, 143-145).
 4. Subsequent inter-episode auto-resets DO hit _reset_idx, which zeros
 _prev_action[env_ids] and updates _prev_dist[env_ids] correctly.

 So actually this is fine — the first episode starts with progress=0
 (prev_dist equals dist on first frame), and subsequent resets fix up
 correctly. Downgrading to LOW.

 1.3 HIGH: instrumented reward re-implementation vs urchin_env_cfg._get_rewards — subtle divergence risk

 The instrumented version (eval_policy_reward.py:95-223) is a hand-copy of
 urchin_env_cfg._get_rewards (urchin_env_cfg.py:346-485). I diff-reviewed
 line-by-line:

 ┌────────────────────────────────────────┬─────────────────────────────────────────┐
 │                  Term                  │                 Match?                  │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ progress, vel_reward, dist_pen, goal_r │ ✅ identical                            │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ surface_smooth_pen                     │ ✅ identical                            │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ action_rate_pen                        │ ✅ identical                            │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ crouch_reward                          │ ✅ identical                            │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ rolling_reward                         │ ❌ does not match WP2+WP3               │
 ├────────────────────────────────────────┼─────────────────────────────────────────┤
 │ gait_reward                            │ ✅ identical (both gated at weight 0.0) │
 └────────────────────────────────────────┴─────────────────────────────────────────┘

 BLOCKER finding inside rolling_reward. The env's current post-WP2/WP3
 formula (urchin_env_cfg.py:425-448) is:

 expected_omega = vel_toward.clamp(min=0.0) / ball_radius           # WP2
 slip_tolerance = 0.3 * expected_omega + 0.2
 slip_error     = (ang_proj - expected_omega).abs()
 rolling_quality = (1.0 - slip_error / (slip_tolerance + eps)).clamp(0, 1)
 rolling_reward  = rolling_quality * expected_omega * weight * speed_gate  # peaked

 The eval instrumentation (eval_policy_reward.py:175-180) is still the
 pre-WP3 monotone version:

 expected_omega = speed_xy / self.cfg.ball_radius                   # unsigned |v|/r — PRE-WP2
 rolling_reward = (
     torch.minimum(ang_proj, expected_omega) * weight * speed_gate  # monotone — PRE-WP3
 )

 This means every "noslip" eval I ran was scoring the policy against the
 OLD reward surface. The per-term rolling_reward=+248 on the best
 noslip checkpoint was computed with the monotone-min formula — the policy
 was never measured against the actual training-time peaked-no-slip reward.

 Severity: BLOCKER. This plausibly explains everything:
 - The 260±323 "det mean" is on the old reward surface. Under the new
 (peaked) surface the same behavior might score 0 or negative
 (ang_proj > expected_omega hurts), which would align with the user's
 video verdict of "barely rolls."
 - The "rolling_reward=+248, progress=+8" signature suggesting reward
 hacking is computed against the WRONG formula.

 Patch: replace the rolling_reward block in _instrumented_rewards
 (lines 161-180) with the post-WP2/WP3 formula from urchin_env_cfg.py:425-448.
 Safer long-term fix: make the instrumented wrapper NOT re-implement the
 math — instead, call the original _get_rewards inside a stashing
 decorator that captures tensors from closure variables. But that's
 invasive; a line-for-line re-copy is fine for now.

 1.4 MEDIUM: _sim_step pinned past anneal end — confirm parity intent

 eval_policy_reward.py:252 pins _sim_step = anneal_end + 1M so
 anneal_scale = 1.0. This means all annealed terms (surface_smooth_pen,
 action_rate_pen, crouch_reward) fire at full weight during eval,
 matching post-anneal training conditions.

 For checkpoints trained past 700k sim steps (a_end), this is correct —
 that's the reward surface the policy optimized against. For an early
 checkpoint that never saw full anneal, this over-weights the penalty
 terms. The BC seed used a dedicated BC-loss-only warmup (anneal
 not relevant there) so this doesn't cause BC-seed mismeasurement.

 No patch needed; document intent.

 1.5 MEDIUM: eval Policy/Value read inputs["states"], train reads inputs["observations"]

 train.py:154,169:   self.net(inputs["observations"])
 eval_policy_reward.py:288,305:   self.net(inputs["states"])
 render_policy_video.py:128,143:   self.net(inputs["states"])

 Per project_skrl2_pipeline_fixes memory, skrl 2.0 puts rollout obs under
 both keys, update under "observations" only. Eval/render call
 policy.compute({"states": obs_scaled}, ...) directly (not agent.act),
 passing only the "states" key. Their Policy reads that key → works.
 The checkpoint state_dict keys are on self.net, identical topology both
 sides → agent.load() succeeds.

 So it's consistent in isolation, but the asymmetry is a footgun: if
 someone ever calls agent.act(...) in eval/render or ports the key fix
 from train, the eval Policy may silently read None.

 Patch (low severity): normalize to inputs["observations"] in eval and
 render too, and pass that key in the manual compute({...}) calls.

 1.6 LOW: ppo_cfg.state_preprocessor_kwargs = {"size": obs_space, ...}

 eval_policy_reward.py:316, render_policy_video.py:155: passes the
 full Box as size. skrl accepts this. Same in train.py:193. Parity
 holds.

 ---
 2. scripts/render_policy_video.py — why 3 episodes → 2 MP4 files

 2.1 LOW: --episodes 3 gets 2 MP4 files, one near-empty

 Root cause: the script uses a flat step loop of 945 steps total
 (line 72: total_steps = steps_per_ep * episodes * 1.05) with
 RecordVideo(video_length=total_steps, episode_trigger=lambda _: True)
 (lines 79-86). The video_length is set so large that the first video
 captures all episodes concatenated into one file — that is the 1.4 MB
 episode-1.mp4 with teleports at t=300, t=600.

 The 1.7 KB episode-0.mp4 is the placeholder opened by the first
 env.reset(seed=args.seed) on line 88 (BEFORE the second reset on
 line 180 kicks in with real rollouts). RecordVideo creates a video
 writer on the first reset, writes nothing (no steps yet), then when the
 next reset fires it closes the near-empty file and starts episode-1.

 There is no meaningful correctness bug, but the file-count is
 inconsistent with --episodes 3, and the concatenated video makes
 per-episode spatial analysis hard.

 Patch (optional, LOW):
 - Remove the outer env.reset() on line 88 (the one on line 180 is the
 real one).
 - Restructure as for ep in range(args.episodes): env.reset(...); for step in range(steps_per_ep): .... Drop video_length or set to  
 steps_per_ep. Emit one MP4 per episode.
 - Alternative minimal change: leave the flat loop; just drop line 88's
 reset and accept one concatenated video. Still kills the 1.7 KB file.

 2.2 LOW: --deterministic argparse flag is a no-op

 render_policy_video.py:29:
 parser.add_argument("--deterministic", action="store_true", default=True, ...)
 store_true + default=True means there is no way to make this False
 from the CLI. The --deterministic flag always evaluates True whether
 passed or not. Latent; not currently affecting runs.

 Patch: replace with action="store_false" + default True, OR make
 the default explicit and drop the flag.

 2.3 LOW: step-log positions don't match video — sampling aliasing, not a bug

 line 204: if i % 30 == 0 or i == total_steps - 1 prints pos at every
 30 steps. At env_dt = 1/60 s, 5 s episode = 300 steps, so transition
 steps are 300, 600 (auto-resets occur on those). The log at step 270
 captures late-episode transient excursions (e.g. 0.37-0.55 m forward
 lurches before the blob collapses). The video shows the full 300 frames
 of motion — including the collapse — so it looks "less rolling" than the
 log snapshot.

 This is sampling aliasing, not a bug. The robot physically does
 reach 0.37-0.55 m for a fraction of a second, then falls back toward
 spawn. The video is truth; the step log is a periodic snapshot.

 Documenting as expected behavior.

 ---
 3. Train ↔ eval parity

 3.1 Observation normalization

 - Both train and eval use RunningStandardScaler.
 - Train (train.py:462-463): train=not epoch — scaler stats update on
 first epoch of each PPO update, from sampled rollout obs.
 - Train (train.py:401-404): scaler is applied with train=True to
 _current_next_observations during GAE computation — also updates
 stats.
 - Eval (eval_policy_reward.py:360): scaler(obs, train=False) — never
 updates. ✅

 agent.load(ckpt_path) restores the scaler state_dict alongside models
 and optimizer, so eval inherits whatever running mean/variance the
 training checkpoint had. ✅

 HIGH finding: the scaler is still updating throughout training.
 When best_checkpoint.pt is saved at step 173k, its scaler reflects 173k's
 rollout distribution. On eval, policy + scaler are self-consistent — but
 the scaler stats are based on the stochastic policy's exploration
 distribution (std≈0.37). Under deterministic eval, obs distribution is
 narrower and shifted. Normalization isn't catastrophically wrong, but
 this is a secondary contributor to the stoch-vs-det gap.

 Patch (HIGH, optional for now): after BC warmstart, fit the scaler
 on a held-out rollout dataset and freeze it for the PPO phase.
 Requires a new flag --freeze-scaler-after-warmstart. Not urgent; fix
 1.3 is more likely to close the gap.

 3.2 Action scaling / clipping

 - Policy.compute returns raw MLP output (no tanh, no clip) in both
 train and eval.
 - GaussianMixin(clip_actions=False) both sides.
 - Clipping happens inside the env: _pre_physics_step
 (urchin_env_cfg.py:276-282) applies clamp(-sh_coeff_clip, sh_coeff_clip) then raw.clamp(-1, 1) then rest + span*raw
 clamped to stroke limits.
 - Both train and eval use the same env → same clipping. ✅

 3.3 Domain randomization — confirmed OFF

 UrchinEnvCfg has no events field, no action_noise_model, no
 observation_noise_model. Curriculum is deterministic functions of
 _sim_step (yaw span, dist scale, shaping anneal). No DR at train
 or eval. ✅

 HOWEVER: the yaw curriculum reset jitter is random
 (urchin_env_cfg.py:515-517: (torch.rand(...) - 0.5) * yaw_span).
 For eval, _sim_step is pinned past anneal end, which means
 yaw_span_end = 2π → every eval reset draws a fully random yaw.
 For a rolling policy, initial yaw matters a lot (a ball facing backward
 gets −40 on progress before it even learns to reorient).

 This is by design — it stress-tests the policy at the fully-annealed
 task distribution — but it's also the single biggest contributor to the
 bimodal det-eval 260±323: 2/5 seeds spawn facing the goal and roll
 (~650), 3/5 spawn rotated away and fail (~0).

 Recommendation (not a patch, a test protocol): for diagnostic evals,
 add a --fixed-yaw flag that zeros the jitter. Use for smoke tests to
 distinguish "policy can't roll" from "policy can roll but orientation-
 fragile." Keep random yaw for capstone evaluation.

 3.4 Episode length

 - Env reads URCHIN_EPISODE_S env var at module import time
 (urchin_env_cfg.py:62).
 - Both train.py (via dispatcher / shell wrapper) and
 eval_policy_reward.py:56 set this env var before importing
 urchin_env_cfg. ✅
 - args.seconds defaults to 5.0 in eval, matching the 5 s episode
 the noslip smoke was trained on. ✅

 3.5 Physics substeps / control frequency

 - cfg.sim.dt = 1/240, cfg.decimation = 4 → env_dt = 1/60.
 - Both paths instantiate UrchinEnvCfg() with no overrides. ✅

 3.6 Reset distribution

 - Start pose: default[:,0..2] = (start_x, start_y, 0.17) with
 distance curriculum scale. Eval pins _sim_step past anneal end →
 dist_scale = dist_scale_end = 1.0 → spawn at (-0.5, -0.5, 0.17). ✅
 (Matches the spawn the noslip smoke mostly trained on, since
 dist_curriculum_end_sim_steps = 400k, which 500k training passes.)
 - Yaw: see 3.3.
 - Joint pos: inherits robot's default_joint_pos on reset. Consistent. ✅

 Summary of parity: NO train/eval divergence beyond the scaler-keeps-
 learning issue (3.1, HIGH) and a latent yaw-randomness diagnostic
 headache (3.3). The stochastic-vs-deterministic 250-unit gap is NOT
 explained by parity drift. Probably driven by:
 - The WP2/WP3 reward surface has a narrow peak (ang_proj ≈ expected_omega),
 which stochastic exploration lucky-hits and deterministic misses. This
 is a reward-design issue, not a measurement bug.
 - Finding 1.3 (eval using pre-WP3 formula) means we're ALSO comparing
 apples to oranges until that's fixed.

 ---
 4. PPO stability — peak at 35%, drift to 500k

 4.1 HIGH: best_checkpoint.pt saved on noisy stochastic mean over too few episodes

 train.py:702-710:
 if (timestep + 1) % ROLLOUTS == 0 and recent_ep_rewards:
     mean_r = float(np.mean(recent_ep_rewards[-100:]))
     if mean_r > best_reward:
         best_reward = mean_r
         agent.save(str(checkpoint_dir / "best_checkpoint.pt"))
     reward_curve.append([actual_steps, mean_r])
     _emit(...)
     recent_ep_rewards.clear()

 Math: ROLLOUTS=24 timesteps × 64 envs = 1536 env-steps per window.
 Episode length 300 steps/env → at most 5 envs complete an episode per
 window; typically 0–5 completions. recent_ep_rewards.clear() drops
 everything each emit, so [-100:] is really [-0..5:] in most windows.

 So best_reward is the max-over-326-windows of a mean-over-5-episodes.
 Max-of-326 draws from a noisy 5-sample mean has a large positive bias.
 The "peak 161 at step 173k" is partly signal, partly selection bias. And
 once best_reward is set to that lucky high, later genuinely-similar
 windows can't overwrite it.

 This also pairs with feedback_verify_peak_before_handoff — already
 noted in memory.

 Patch (HIGH):
 - Raise sample size per eval. Either:
   - Keep a rolling buffer that DOESN'T clear, take mean of last N≥30
 episodes: recent_ep_rewards = recent_ep_rewards[-100:] (don't
 clear), emit mean every ROLLOUTS.
   - OR run a synchronous deterministic eval (e.g. every 50k steps, roll
 out N=20 deterministic episodes) and save on that.
 - The second is strictly better (matches eval methodology), at the cost
 of ~30-60 s per eval checkpoint. At 50k cadence over 500k, that's
 10 evals × 45 s = 7.5 min on a 14 min run. Worth it.

 4.2 HIGH: best_reward drift up from 37 at step 20k — policy was degraded at start

 result.json reward_curve[0] = [19968, 36.86]. This is 20k env-steps
 into a run warmstarted from BC seed (det-eval 673). BC seed stochastic
 reward under this same reward surface was not measured; under the OLD
 surface it would be close to 673.

 Two interpretations:
 - WP2/WP3 made the reward surface so much harsher that the BC policy
 scores 37 stochastic under it. PPO then spends 170k steps recovering
 to 161 — the policy IS learning, it's just not getting above the old
 BC det score.
 - Scaler starts at BC-era stats but immediately gets overwritten by
 early-PPO obs. Policy input distribution shifts → behavior degrades.

 Either way, the warmstart advantage is largely wasted.

 Patch (to diagnose, not a code patch): run a single deterministic
 eval of the BC seed under the CURRENT (WP2/WP3) reward surface BEFORE
 launching another PPO fine-tune. If BC scores 673 on old surface and
 e.g. 200 on new surface, Interpretation 1 is confirmed — the PPO curve
 shape is healthy, we just need a better reward surface. If BC still
 scores 650+ on new surface, Interpretation 2 is real and the
 scaler-keeps-learning path needs patching first (3.1).

 4.3 LOW: no learning-rate schedule, no entropy schedule

 train.py uses a fixed learning_rate and entropy_loss_scale=0.005.
 No cosine decay, no KL-adaptive. For noisy 42-panel contact dynamics at
 this scale, a fixed LR across 500k steps is plausibly too aggressive
 late-run. Not a bug, but a knob worth trying AFTER 1.3 is fixed.

 4.4 NOT A BUG: best_reward matches curve max

 result.json.best_reward = 160.68. Scanning reward_curve for the max
 value shows the curve peaks near 161 around step 173k. best_checkpoint.pt
 is saved only on mean_r > best_reward → they match by construction.
 ✅ No rollback or overwrite bug.

 4.5 NOT A BUG: frozen BC reference policy is not re-loaded mid-training

 In the BC-reg path (train.py:338-355), bc_policy is instantiated,
 loaded from bc_ckpt_path, set to eval mode, requires_grad_(False).
 The closure update_with_bc references bc_policy (captured by
 closure); nothing mutates it. The bc_scaler is a copy.deepcopy of
 agent state, also frozen at copy time.

 BUT note train.py:375: when bc_reg_scaler_checkpoint == load_checkpoint,
 the code does bc_scaler = copy.deepcopy(agent._state_preprocessor).eval().
 Since agent.load() already loaded the BC-era scaler from the load
 checkpoint, this deepcopy captures BC-era stats. ✅

 ---
 5. skrl 2.0 integration surface

 5.1 NOT A BUG: [N,1] shape fix present

 train.py:667-672 explicitly unsqueezes rewards/terminated/truncated
 to [N,1] before record_transition. ✅ Matches the memory note.

 5.2 NOT A BUG: inputs["observations"] key fix present in train

 train.py:154, 169 reads inputs["observations"]. ✅ Consistent with
 the skrl 2.0 fix from memory.

 5.3 NOT A BUG: agent.init() + enable_training_mode(True)

 train.py:213-214. Present per the memory note. ✅

 5.4 MEDIUM: BC-reg monkey-patch of PPO.update is verbatim from

 skrl 2.0.0, version-gated at train.py:303-311. Hard-fails on version
 mismatch. ✅ Well-guarded.

 A subtle concern: the patched update_with_bc calls
 self.value.act(_inputs, role="value") on line 505 with the SAME
 _inputs dict that contains taken_actions (injected on line 469).
 skrl's Value model ignores taken_actions, so this is harmless. ✅

 5.5 LOW: advantages normalization

 The monkey-patched update uses compute_gae and stores raw advantages
 via set_tensor_by_name("advantages", advantages). skrl 2.0 normalizes
 advantages inside its sampled-batch iterator if configured; the
 configured ppo_cfg doesn't touch advantages_mean_std or similar.
 This matches vanilla skrl default → no divergence from stock behavior.

 No issues found in the skrl integration surface. Prior fixes hold.

 ---
 6. Prioritized patch list

 #: 1
 File: eval_policy_reward.py:161-180
 Severity: BLOCKER
 What: Replace instrumented rolling_reward block with the post-WP2/WP3 peaked-no-slip formula from urchin_env_cfg.py:425-448
 Why: Every noslip eval was scoring against the OLD reward surface. Plausibly explains the whole "rolling_reward=+248 but video      
 shows
   no rolling" contradiction
 ────────────────────────────────────────
 #: 2
 File: eval_policy_reward.py:207, 374
 Severity: BLOCKER
 What: Capture env._last_root_pos_w inside instrumented _get_rewards (pre-reset state), read it instead of robot.data.root_pos_w     
   after the loop
 Why: All final_pos reports are currently spawn pose due to Isaac Lab's step→reset→obs ordering
 ────────────────────────────────────────
 #: 3
 File: train.py:702-710
 Severity: HIGH
 What: Replace noisy stochastic-mean best-save with periodic deterministic eval (e.g. 20 eps at every 50k sim steps); save on that.  
   Or, at minimum, stop clearing recent_ep_rewards and use a fixed window ≥30
 Why: Current path saves lucky 5-sample peaks (selection bias), misaligns with our actual eval metric
 ────────────────────────────────────────
 #: 4
 File: diagnosis, no code change
 Severity: HIGH
 What: Run eval of BC seed urchin_v3_contactpush_bc/final_checkpoint.pt under the CURRENT (WP2+WP3) reward surface — AFTER patch #1  
   lands
 Why: Decides whether the stoch-ramping-from-37 shape is reward-surface harshness (expected) or warmstart-broken (bug). Gates next   
   training cycle design
 ────────────────────────────────────────
 #: 5
 File: train.py
 Severity: HIGH
 What: Add --freeze-scaler-after-warmstart flag; snapshot scaler right after agent.load(--load-checkpoint) and disable its
 train=True
   updates for the PPO phase
 Why: Scaler drifting under stochastic rollouts is a plausible secondary driver of stoch-vs-det gap; decoupling it lets us measure   
   the reward-surface issue in isolation
 ────────────────────────────────────────
 #: 6
 File: eval_policy_reward.py:288,305; render_policy_video.py:128,143
 Severity: LOW
 What: Read inputs["observations"] instead of inputs["states"]; pass {"observations": ...} in manual compute({...}) calls
 Why: Eliminates train/eval key asymmetry; future-proofs against skrl internal changes
 ────────────────────────────────────────
 #: 7
 File: render_policy_video.py:29
 Severity: LOW
 What: Fix --deterministic argparse (currently always True)
 Why: Latent bug; not affecting current runs but a trap
 ────────────────────────────────────────
 #: 8
 File: render_policy_video.py:79-86, 88
 Severity: LOW
 What: Remove preliminary env.reset() on line 88 and/or restructure as per-episode loops so each episode gets its own MP4
 Why: Eliminates the 1.7 KB placeholder and the concatenated-episode video; makes per-episode visual review tractable
 ────────────────────────────────────────
 #: 9
 File: eval_policy_reward.py / new flag
 Severity: LOW
 What: Add --fixed-yaw that zeros the reset yaw jitter, for diagnostic smoke evals
 Why: Separates "policy can't roll" from "policy is orientation-fragile" — explains 2/5 seeds ~650 + 3/5 ~0 bimodality

 Recommended sequence before any new training cycle

 1. Land patch #1 (BLOCKER — eval reward formula). Re-run
 eval_policy_reward.py on the existing urchin_v3_smoke_noslip/best
 and final checkpoints. If the scores drop dramatically (e.g. 260 →
 negative), the "noslip fixed the reward, smoke failed" conclusion
 stands.
 2. Land patch #2 (BLOCKER — final_pos). Re-eval BC seed and noslip
 best; confirm reported final_pos varies across 5 seeds and matches
 what videos show. This gives us a trustworthy spatial metric for
 all future runs.
 3. Execute diagnostic #4: eval BC seed under current reward surface
 (now correctly measured). Outcome informs WP4 scope.
 4. Optionally land #3 and #5 before the next training cycle to
 remove selection bias and scaler drift from the signal. #3 is
 cheap; #5 is a small code change but requires one more
 regression test.
 5. Execute WP4 (de-telescope progress reward) or an alternate reward-
 surface change, informed by #4.

 Verification

 After patches are applied, verify on existing checkpoints without
 launching new training:

 # After patch 1 + 2:
 C:/isaac-venv/Scripts/python.exe \
   workspace/robots/urchin_v3/scripts/eval_policy_reward.py \
   --checkpoint workspace/checkpoints/urchin_v3_contactpush_bc/final_checkpoint.pt \
   --tag bc_new_surface --episodes 5 --seconds 5.0 \
   --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 \
   --json-out workspace/rewards/bc_new_surface/eval.json

 # Expect: final_pos values vary across the 5 episodes (not all spawn);
 # rolling_reward drops if peaked-no-slip is stricter than monotone-min.

 C:/isaac-venv/Scripts/python.exe \
   workspace/robots/urchin_v3/scripts/eval_policy_reward.py \
   --checkpoint workspace/checkpoints/urchin_v3_smoke_noslip/best_checkpoint.pt \
   --tag noslip_fixed_eval --episodes 5 --seconds 5.0 \
   --start-xy=-0.5,-0.5 --goal-xy=0.5,0.5 \
   --json-out workspace/rewards/noslip_fixed_eval/best.json

 No new training runs needed until patch #1 + #2 land and diagnostic #4
 reports.

 Constraints on implementation (carry forward)

 - BC seed is the only known-good policy — NEVER overwrite or reuse its
 directory.
 - Patch #2's instrumented-state cache must not leak across
 concurrent envs; guard with self._last_root_pos_w = pos.detach()...
 which lives on the env instance (num_envs=1 for eval/render so this
 is safe).
 - Eval/render run single-env only; skrl agent.load() on a single-env
 PPO context is known-working for this pipeline.
 - 8 GB VRAM — do not launch parallel Isaac Sim. Run eval → render in
 series.