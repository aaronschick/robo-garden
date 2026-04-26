# Urchin v3 BC-Regularized PPO: Design

**Status:** Design doc. Not implemented. Awaiting human review before WP5 execution.
**Date:** 2026-04-20.
**Scope:** Training script `workspace/robots/urchin_v3/scripts/train.py` (skrl 2.0 PPO).
**Prereq:** WP1 result in `workspace/rewards/bc_seed_reward_measurement.md` — BC seed
scores 673 mean vs. `smoke_vtgate/best` at 33. PPO is destroying the BC behavior
inside the first ~20k steps. See `docs/urchin_v3_reward_fix_plan.md` for context.

## TL;DR

Add an **analytic forward-KL penalty** `beta_bc * KL(pi_current || pi_bc)` to PPO's
per-mini-batch loss. Both policies are diagonal Gaussians with learned
`log_std_param`, so the KL is closed-form in the action-mean and the two log-stds
(no sampling needed). Freeze a second `Policy` module + a frozen snapshot of the
`RunningStandardScaler` taken at warmstart load. Anneal `beta_bc` linearly from
~0.1 to 0 over the first 500k environment steps. The integration point is a
single injected line right before the backward pass in skrl's PPO update loop;
we achieve that by overriding `agent.update` in `train.py` (no skrl source edit).

## Design decisions

### 1. Regularization form — forward KL, analytic

Both the current policy and the frozen BC policy expose
`.distribution(role="policy")` returning a `torch.distributions.Normal` with
mean shape `[B, act_dim]` and a broadcasted `log_std` vector. For two diagonal
Gaussians, forward KL has a closed form:

```
KL(N(mu_c, s_c) || N(mu_bc, s_bc)) =
    sum_d [ log(s_bc/s_c) + (s_c^2 + (mu_c - mu_bc)^2)/(2 s_bc^2) - 0.5 ]
```

We use `torch.distributions.kl.kl_divergence(dist_current, dist_bc)` which gives
this exactly — no Monte Carlo estimate, zero variance. Justification vs.
alternatives:

- **Forward KL** `KL(pi || pi_bc)` is mode-covering. Because BC is already our
  trusted anchor, we want pi to stay *inside* BC's support. This is the right
  direction. Reverse KL (mode-seeking on BC) would also work but zero-forces
  probability mass where BC is low — bad if BC's log-std is under-estimated.
- **MSE on means only** ignores log-std drift. Our `smoke_vtgate` run used
  `--bc-post-log-std=-1.0` (std ~0.37); PPO with `entropy_loss_scale=0.005`
  will drift `log_std_param` upward and the mean shift dominates the symptom,
  but the std blow-up is coupled. KL captures both.
- **Analytic vs. sampled KL.** We already have parameters; sampling is pure
  overhead. Skip it.

### 2. skrl integration point

File: `C:/isaac-venv/Lib/site-packages/skrl/agents/torch/ppo/ppo.py`.
Version `skrl==2.0.0`. The key code is the per-mini-batch backward in
`PPO.update`, **lines 397–417**:

```python
# compute policy loss
ratio = torch.exp(next_log_prob - sampled_log_prob)
surrogate = sampled_advantages * ratio
surrogate_clipped = sampled_advantages * torch.clip(
    ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
)
policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
# compute value loss
predicted_values, _ = self.value.act(inputs, role="value")
...
value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

# optimization step
self.optimizer.zero_grad()
self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()  # <-- HERE
```

The BC-KL term plugs into that `.backward()` sum at **line 417**. Since we
must not edit skrl, we monkey-patch `agent.update` from `train.py` by
assigning a bound method that is a verbatim copy of skrl's `update` with
two extra lines. A cleaner alternative — subclassing `PPO` — is fine too but
requires reproducing the full ~150-line method. Monkey-patch is the minimal
diff. Pin it to this exact skrl version in a comment; if skrl is upgraded,
re-copy `update`.

### 3. BC policy loading

Instantiate a second `Policy` module (same architecture), load weights from
`--bc-reg-checkpoint` (usually the same `final_checkpoint.pt` passed to
`--load-checkpoint`), call `.eval()`, and `p.requires_grad_(False)` on every
parameter. Keep on the same CUDA device.

**Memory cost.** Policy has `(90->256, 256->256, 256->42) + log_std(42)` ≈ 90k
params ≈ **360 KB** in fp32. Activations: one forward pass at batch
`num_envs*rollouts/mini_batches = 64*24/4 = 384` samples, two 256-unit hidden
layers ≈ `384 * 256 * 4 bytes * 2 ≈ 0.8 MB`. No grad graph retained for the BC
branch (`no_grad`), so no activation memory accumulates. Total overhead is in
the low-MB range, comfortably fits on an RTX 3070 Laptop that already uses
~5.5 GB at 64 envs.

### 4. Annealing schedule

Start `beta_bc = 0.1`, anneal **linearly** to 0 over the first 500k env steps,
then hold at 0. Justification:

- **Initial value 0.1.** At PPO init the policy loss magnitude is O(0.01–0.1)
  for clipped surrogates, and a freshly-loaded BC policy has `KL(pi || pi_bc)`
  near 0 that grows as training perturbs mu/log_std. A beta of 0.1 keeps the
  KL penalty comparable to policy loss *once KL rises to ~0.1 nats*, which is
  approximately the scale at which we observe behavior degrading. Range to
  explore in follow-up: 0.03 – 0.3.
- **Linear, not exponential.** Exponential makes sense when you want a long
  tail; here we specifically want beta_bc=0 by 500k so PPO can still refine
  the gait. A linear schedule crosses zero deterministically at the configured
  step. Implementation: `beta_t = beta_0 * max(0, 1 - step/anneal_steps)`.
- **Why 500k.** Matches the current smoke horizon. The hypothesis is that the
  first ~50k–100k steps are the danger zone (BC destroyed before first eval at
  20k); once PPO has taken hundreds of stable gradient steps anchored near BC,
  the anchor can relax. 500k gives 10× margin.

### 5. Preprocessor freeze

Critical. The frozen BC policy was trained against the `RunningStandardScaler`
state that existed at BC-pretrain end. If we let PPO's live scaler continue
updating (`train=not epoch` at `ppo.py:374`), observations fed to the frozen
BC policy drift and its action distribution becomes meaningless relative to
the anchor we wanted.

**Solution:** on load, `deepcopy` the agent's `_state_preprocessor` into a
second object `_bc_state_preprocessor`. Don't call it with `train=True` — only
`train=False`. Use it exclusively for BC forward-passes. The live PPO scaler
continues updating as normal. Spell this out in code:

```python
bc_scaler = copy.deepcopy(agent._state_preprocessor).eval()
for p in bc_scaler.parameters():
    p.requires_grad_(False)
# in the KL hook:
obs_for_bc = bc_scaler(sampled_observations, train=False)
```

Rationale: freezing the live scaler would break PPO's own value-loss logic
(the value preprocessor is separate, fine). Using an old BC-era scaler
snapshot for the anchor gives us the invariant we need.

### 6. CLI flags (to add to `train.py`)

```python
parser.add_argument("--bc-reg-coef", type=float, default=0.0, dest="bc_reg_coef",
                    help="Initial beta_bc weight for KL(pi||pi_bc). 0 disables.")
parser.add_argument("--bc-reg-anneal-steps", type=int, default=500_000,
                    dest="bc_reg_anneal_steps",
                    help="Env steps to linearly anneal beta_bc to 0.")
parser.add_argument("--bc-reg-checkpoint", type=str, default="",
                    dest="bc_reg_checkpoint",
                    help="Checkpoint whose policy is the frozen BC anchor. "
                         "Defaults to --load-checkpoint when empty.")
parser.add_argument("--bc-reg-kind", type=str, default="kl_fwd",
                    choices=("kl_fwd", "kl_rev", "mse_mean"), dest="bc_reg_kind",
                    help="Regularization form (default: forward KL).")
```

Default `--bc-reg-coef=0.0` keeps the code path inert when unused — existing
`--load-checkpoint` workflows are unaffected.

### 7. Compute overhead estimate

Per PPO update: `learning_epochs * mini_batches = 5 * 4 = 20` extra forward
passes through a 90k-param MLP at batch 384. Each forward is ~40 us on a 3070
Laptop (measured rough order for a 3-layer 256-wide MLP). 20 * 40 us = **0.8
ms per update**. PPO updates fire every `ROLLOUTS=24` env steps = ~0.4 s at
our 60 Hz × 64 envs throughput. Overhead fraction: 0.8 ms / 400 ms ≈
**0.2%**. Even with a 10× pessimism margin, well under 5%. The KL analytic
computation itself is a handful of elementwise ops on `[B, 42]` — negligible.

### 8. Fallback diagnostic

Log `KL(pi || pi_bc)` (the same term we're penalizing) at each eval emit.
Health bands:

- `KL < 0.01` continuously: **over-regularized**; policy can't diverge enough
  to learn. Reduce `beta_bc` by 3×.
- `KL` climbs monotonically past `1.0` in `< 20k` steps: **under-regularized**;
  BC anchor not doing anything. Increase `beta_bc` by 3×.
- `KL` rises slowly to ~0.3–1.0 over the anneal window, then can grow
  afterward: **healthy**.

Plumb this into the existing `_emit` payload as `bc_kl`.

## Concrete code sketch (~50 lines)

To be added to `train.py` immediately after `agent.init()` and the
`--load-checkpoint` block. This is a sketch for the human reviewer; WP5
implementation polishes it.

```python
# --- BC regularization hook --------------------------------------------------
if args.bc_reg_coef > 0.0:
    import copy, types, torch.distributions as D
    bc_ckpt = args.bc_reg_checkpoint or args.load_checkpoint
    assert bc_ckpt, "--bc-reg-coef>0 requires --bc-reg-checkpoint or --load-checkpoint"

    # Frozen BC policy: same architecture, load weights, no grad.
    bc_policy = Policy(obs_space, act_space, device)
    _raw = torch.load(bc_ckpt, map_location=device, weights_only=False)
    bc_policy.load_state_dict(_raw["policy"])
    bc_policy.eval()
    for p in bc_policy.parameters(): p.requires_grad_(False)

    # Frozen BC-era state scaler snapshot.
    bc_scaler = copy.deepcopy(agent._state_preprocessor).eval()
    for p in bc_scaler.parameters(): p.requires_grad_(False)

    # Total env steps for the anneal.
    bc_anneal = max(1, args.bc_reg_anneal_steps)
    beta0 = args.bc_reg_coef

    # Monkey-patch agent.update: verbatim copy of skrl 2.0.0 ppo.py update
    # with two injected lines right before `.backward()` (ppo.py:417).
    # Keep one unified copy; if skrl is upgraded, re-sync.
    _orig_update = agent.update
    def update_with_bc(self, *, timestep, timesteps):
        # ... (copy of skrl update body up to policy_loss / value_loss compute)
        #     for each mini-batch, after value_loss is computed, before backward:
        with torch.no_grad():
            dist_bc = bc_policy.distribution_from_obs(  # thin wrapper: act() then
                bc_scaler(sampled_observations, train=False))  #   return dist
        dist_cur = self.policy.distribution(role="policy")
        bc_kl = D.kl.kl_divergence(dist_cur, dist_bc).sum(-1).mean()
        env_step = timestep * num_envs
        beta_t = beta0 * max(0.0, 1.0 - env_step / bc_anneal)
        bc_loss = beta_t * bc_kl
        # replace the existing backward line with:
        #   self.scaler.scale(policy_loss + entropy_loss + value_loss + bc_loss).backward()
        # ...
        self.track_data("BC / KL(pi||pi_bc)", bc_kl.item())
        self.track_data("BC / beta_bc", beta_t)
    agent.update = types.MethodType(update_with_bc, agent)
    print(f"[urchin_v3] BC regularization enabled: beta0={beta0} "
          f"anneal={bc_anneal} anchor={bc_ckpt}", flush=True)
# ----------------------------------------------------------------------------
```

`Policy.distribution_from_obs(o)` is a two-liner helper we add: run `self.act({"observations": o, "states": o}, role="policy")` then return
`self.distribution(role="policy")`. Because GaussianMixin caches the last
distribution on `.act`, this is already the natural API.

## Open questions / risks

- **Monkey-patch fragility.** If skrl is updated, the copied `update` body
  diverges. Mitigation: pin skrl version in the project's isaac-venv, add a
  version-check assert in `train.py`.
- **BC log-std mismatch.** The BC checkpoint was saved with
  `log_std_param = -1.0` (per `--bc-post-log-std`). If the current policy's
  log_std drifts far from -1.0, the forward KL's variance-ratio term dominates.
  May want to tune `entropy_loss_scale` *down* alongside enabling BC reg so
  log_std has less pressure to drift. Open question: does the current 0.005
  need to go to 0.001?
- **Mini-batch correlation.** KL on the same 384 samples gets counted 5× per
  update (once per learning epoch). This is consistent with how policy_loss
  is accumulated in the current PPO, so effectively the BC loss "weight" is
  5× beta_bc relative to a single-epoch optimizer. Accounted for in the
  suggested 0.1 initial value.
- **Interaction with KL early-stop.** `ppo.py:388` early-stops a mini-batch
  when the PPO approximate-KL-to-old-policy exceeds `cfg.kl_threshold`. This
  is a *different* KL (to the pre-update policy) from BC-KL. They don't
  conflict, but if we're fighting against a strong BC anchor the PPO-KL won't
  blow past `kl_threshold` — so the early-stop won't mis-fire. Good.
- **Value function is not regularized.** Only policy. We don't have a BC
  value to anchor to, and the value net is free to adapt. This is desirable
  — we want the critic to learn *this* env's returns, not mimic BC's
  (undefined) value estimate.

## Proposed test plan (WP5b)

1. **Sanity run, 50k steps, `--bc-reg-coef=0.1`, headless, 64 envs.** Confirm:
   (a) training completes with no OOM, (b) `bc_kl` telemetry is non-zero and
   rises, (c) VRAM usage stays within 1 GB of pre-BC-reg baseline.
2. **Coefficient sweep.** Three 500k smokes at `beta_bc ∈ {0.03, 0.1, 0.3}`.
   Output dirs `workspace/checkpoints/urchin_v3_bcreg_{003,010,030}`. Render
   best + final for each. Criterion: does the 20k-step eval reward now track
   BC's 673 instead of collapsing to 33? Winning run is one where early eval
   ≥ 500 and late eval exceeds BC.
3. **Diagnostic review.** Plot `bc_kl` vs. step for each sweep. Verify the
   health bands from §8 match observed dynamics. If the `beta_bc=0.1` run
   stays with `bc_kl < 0.01` through 100k, that's over-regularized; if
   `beta_bc=0.03` has `bc_kl` past 1.0 at 20k, under-regularized.
4. **Visual review.** The run with the highest 500k reward gets rendered and
   human-reviewed (`feedback_verify_peak_before_handoff` applies). Only then
   does it become a handoff candidate.
5. **Compare to WP3.** If WP3 (no-slip peaked rolling reward) has landed,
   re-run the winning BC-reg config on the new reward to isolate gains
   (stacked vs. independent effects).

## References

- WP1 baseline: `workspace/rewards/bc_seed_reward_measurement.md`
- Plan doc: `docs/urchin_v3_reward_fix_plan.md` (WP5 section at line 174)
- Train script: `workspace/robots/urchin_v3/scripts/train.py`
- Upstream PPO: `C:/isaac-venv/Lib/site-packages/skrl/agents/torch/ppo/ppo.py:319–468`
- GaussianMixin distribution: `skrl/models/torch/gaussian.py:124`
