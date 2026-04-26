# RL Alternatives: CMA-ES and Symmetry Augmentation

**Status:** Design notes, not yet implemented.
**Scope:** Complements the existing Brax PPO training path (`training/`) and the Eureka-style reward loop (`rewards/`).
**Primary motivating robot:** `urchin_v2` — 42-panel puffer sphere with high-order icosahedral symmetry and no kinematic reference motions.

## TL;DR

PPO (Brax, GPU) remains the right default for everything this project
trains. But two auxiliary techniques can give outsized wins on robots
whose morphology has strong symmetry or whose gait is hard to discover
from random exploration:

1. **CMA-ES** — a gradient-free evolutionary optimizer used to *seed*
   open-loop gait primitives before PPO refines them into closed-loop
   policies. Especially useful when panel activation is near-discrete
   ("which subset fires, in what sequence") and PPO's random exploration
   rarely stumbles onto a working gait.
2. **Symmetry augmentation** — exploit the fact that a rotated or
   reflected rollout is also a physically valid rollout. Multiplies
   effective training data by the order of the symmetry group, for
   free. On `urchin_v2` that's up to ~60×.

Neither replaces PPO. Both plug in *around* it.

## When to reach for these

| Situation | Recommended addition |
|-----------|---------------------|
| PPO learning a gait from scratch on a novel morphology, plateau at low reward | CMA-ES warmup on an open-loop controller, imitation-init PPO |
| Robot has obvious geometric symmetry (bilateral, radial, icosahedral) | Symmetry augmentation in rollout buffer |
| Panel/actuator activations are near-binary in the optimal gait | CMA-ES is especially strong here — gradients through near-discrete controls are noisy |
| Small policy (≤ ~10k parameters) | CMA-ES viable as a full optimizer, not just warmup |
| Large policy (MLP with >100k parameters) | Stay with PPO; CMA-ES hits the curse of dimensionality |
| Sparse/shaped reward that fights itself | Neither helps directly — fix the reward first (Eureka loop) |

## Part 1: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

### What it is

A **gradient-free black-box optimizer**. Instead of computing ∇θ via
backprop, it treats the policy's parameter vector θ as a point in
ℝⁿ and searches that space directly with an adaptive Gaussian sampler.

### The algorithm, in one page

Maintain a multivariate Gaussian over parameter space: mean **μ** ∈ ℝⁿ,
covariance **Σ** ∈ ℝⁿˣⁿ, step size σ ∈ ℝ.

Each generation:

1. **Sample** N candidate parameter vectors:
   θᵢ ~ μ + σ · 𝒩(0, Σ), for i = 1..N
2. **Evaluate** each by running a full episode in sim and recording
   total reward Rᵢ. (Embarrassingly parallel — one sim rollout per θᵢ.)
3. **Select** the top-k performers (typically k = N/2).
4. **Update μ** toward the weighted mean of the top-k.
5. **Update Σ** to stretch along directions the winners moved — the
   covariance-adaptation heuristic learns which parameter axes matter
   and aligns the search ellipsoid along them.
6. **Update σ** using a cumulative step-length heuristic — grows when
   progress is consistent, shrinks when the search is oscillating.

The adaptation of **Σ** is the clever part. It turns CMA-ES into a
second-order-ish method without ever computing a Hessian: after enough
generations, Σ reflects the local shape of the fitness landscape, and
sampling along its principal axes efficiently follows ridges.

### Why it's interesting for this project

- **Embarrassingly parallel.** N independent rollouts per generation
  maps cleanly onto Brax vmap — same infrastructure we already use for
  PPO.
- **No credit assignment.** Unlike PPO, CMA-ES doesn't need to decide
  *which* action at *which* timestep caused the reward. It evaluates
  whole parameter vectors. For `urchin_v2` this is a feature: figuring
  out "panel vc_23 firing at t=0.4s is what tipped the sphere" is
  exactly the kind of attribution PPO struggles with on contact-rich,
  discontinuous dynamics.
- **Handles discontinuity gracefully.** Rewards that are piecewise
  (e.g., "got +1 if we moved a body length, else 0") are fine for
  CMA-ES and terrible for PPO.
- **Natural fit for small controllers.** A per-panel central pattern
  generator — 42 panels × {amplitude, phase, frequency, bias} ≈ 168
  parameters — is well within CMA-ES's sweet spot.

### Where it breaks

- **Curse of dimensionality.** CMA-ES's covariance is n × n. Once n
  exceeds a few thousand it becomes impractical (memory and sample
  efficiency both collapse). A 3-layer MLP with 64 hidden units and a
  42-dim action already has ~10k parameters. Use CMA-ES on compact
  parameterizations, not on neural network weights directly.
- **Sample cost.** Each generation is N full-episode rollouts. For
  long-horizon tasks this is expensive compared to PPO's
  truncated-rollout-with-bootstrap approach.
- **No closed-loop feedback.** The controllers CMA-ES evolves are
  typically open-loop (time-indexed) or have very small state feedback.
  That's a feature for gait discovery (simpler search space) but means
  you still need PPO afterward for closed-loop robustness.

### Recommended workflow: CMA-ES → PPO handoff

```
1. Parameterize a small open-loop controller for urchin_v2:
   For each of 42 panels, output target_length(t) as a
   parameterized sinusoid: A·sin(2πft + φ) + b.
   → 4 params × 42 panels = 168 parameters.

2. Run CMA-ES for 100-500 generations with N=64 samples each,
   evaluating each candidate by a full 10s episode.
   → ~6k-30k episodes, all GPU-parallel in Brax.
   → Converges to a periodic gait that produces net locomotion.

3. Distill into an MLP:
   Generate (state, action) pairs from the best CMA-ES controller
   rolled out across varied initial conditions. Train a small MLP
   by supervised regression (behavior cloning) to reproduce it.

4. Fine-tune with PPO:
   Initialize PPO's policy from that MLP. Continue training with
   the usual reward to add closed-loop robustness, handle perturbations,
   and adapt to terrain. PPO's job becomes "refine a working gait"
   instead of "discover one from scratch" — much easier.
```

### Library suggestions

- **`evosax`** (JAX-native) — ideal for this repo because it runs on GPU
  alongside Brax with no framework mismatch. Supports CMA-ES and a
  family of related ES variants (OpenES, SNES, xNES).
- **`cma`** (NumPy) — the reference implementation. CPU-only but
  well-tested. Fine if rollouts dominate cost and optimizer overhead is
  negligible.

## Part 2: Symmetry Augmentation

### The core observation

If a robot is geometrically symmetric under some transformation *g*
(e.g., rotation by 72° around the vertical axis, left-right mirror,
one of the 60 icosahedral rotations), then for any real transition
collected by the simulator:

```
(s_t, a_t, r_t, s_{t+1})
```

the *transformed* transition

```
(g · s_t,  g · a_t,  r_t,  g · s_{t+1})
```

is **also physically valid**. The simulator would have produced it, if
the robot had started in the symmetrically-equivalent initial condition
and taken the symmetrically-equivalent action.

This is free data. No extra sim rollouts required.

### How big a deal is this for `urchin_v2`?

The panel layout has **icosahedral symmetry**. The full icosahedral
rotation group I has **60 elements**; adding reflections gives Iₕ with
**120 elements**. In principle each real rollout can be augmented into
up to 60× (or 120× with reflections) training data.

Published results on quadrupeds — which have only a bilateral (order-2)
symmetry — show roughly 2× sample efficiency from symmetry
augmentation. A 30–60× effective data multiplier on `urchin_v2` should
be substantially more dramatic, though with diminishing returns
(augmented data isn't independent from the original).

### Two implementation strategies

#### Strategy A: Rollout-buffer augmentation (easy, cheap)

After each rollout, for each symmetry operation *g* in a chosen subset
of the group, synthesize an augmented transition by applying *g* to
the observation, action, and any global-frame state. Append to the
rollout buffer before the PPO update.

**What needs transforming:**

| Quantity | How *g* acts |
|----------|-------------|
| Joint positions/velocities (per-panel) | Permutation of the 42 panel indices induced by *g* |
| Panel target actions | Same permutation as joint positions |
| Base linear velocity (in world frame) | Rotate by R(*g*) |
| Base angular velocity | Rotate by R(*g*) |
| Base orientation (quaternion) | Compose with R(*g*) |
| IMU readings (body frame) | Transform according to whether IMU is fixed in base frame (rotation only affects world-frame interpretation) |
| Reward | Invariant under *g*, so no change |
| Done flag | Invariant |

**Implementation footprint:**

1. Precompute a **panel permutation table**: for each *g* in the symmetry
   group, a length-42 array `perm[g]` such that panel *i* maps to
   panel `perm[g][i]` under *g*. This comes from the urchin's geometric
   layout — run once offline and cache.
2. Precompute rotation matrices `R[g]` for each group element (3×3 each).
3. At training time, wrap each transition with a vmapped
   `augment(transition, g)` function. No changes to the learning
   algorithm itself.

Cost: a hundred-ish lines of code plus one offline permutation-table
generator. Compatible with any on-policy algorithm, including Brax PPO.

Practical note: you typically don't use the full order-60 group on every
transition. Using 4–8 random group elements per transition gives most
of the benefit without bloating memory or over-correlating the batch.

#### Strategy B: Equivariant networks (harder, more principled)

Constrain the policy architecture so that
**π(g · s) = g · π(s)** holds *by construction*, for all *g* in the
symmetry group. This is achieved with group-equivariant layers — the
generalization of conv nets' translation equivariance to arbitrary
group actions.

**Pros:**
- Inductive bias is built into the network, not added as data. Zero
  augmentation overhead at train time.
- The policy generalizes perfectly across symmetry-equivalent states
  from the first gradient step.

**Cons:**
- Requires an equivariant library (`escnn`, `e3nn`). These are PyTorch
  first; JAX ports are less mature. Integrating with Brax PPO (which
  expects JAX/Flax policies) may require writing an equivariant Flax
  MLP from scratch.
- More constrained architecture → potentially slower to train per-step
  because the effective parameter count is reduced (the constraint
  shares weights across symmetry-related units).

**Recommendation:** start with rollout augmentation. Only invest in
equivariant networks if augmentation alone plateaus and you have
evidence the policy is still learning symmetry-breaking artifacts.

### Gotchas

- **Floor asymmetry.** Even with a symmetric robot, the world breaks
  rotational symmetry around the vertical axis if there's a sloped
  floor, goal direction, or wind. Only use the subgroup that preserves
  the task: for flat-ground omnidirectional locomotion this is the full
  group; for "walk north" tasks this collapses to the stabilizer of
  the north vector (often a reflection-only subgroup).
- **Command conditioning.** If the policy takes a velocity command as
  input, the command must be transformed by *g* along with the state.
  `g · (s, cmd) → g · (s, g · cmd)`.
- **Domain randomization parameters.** If the floor friction or mass
  is part of the observation, it's a scalar — invariant under *g*.
  Easy to miss.
- **Permutation-table correctness.** Generate it programmatically from
  panel centroid positions (apply *g* to each panel's center, find
  nearest original panel). Do not hand-edit. One mis-indexed panel
  silently poisons training.

## How these plug into this project

### Where in the codebase

- **CMA-ES:** a new module `training/evolution.py` that wraps `evosax`.
  Exposes `run_cma_es(robot, env, parameterize_controller_fn, n_gens,
  pop_size) → best_params`. Called from a new `--mode evolve` or from
  within a WSL training job before PPO kicks in.
- **Symmetry augmentation:** extend `training/vectorized_env.py` (the
  MJX vmap wrapper) with an optional `symmetry_group_cfg` that, when
  set, applies augmentation at rollout-collection time. Each robot
  config in `workspace/robots/` would optionally declare its
  `symmetry_group` (element list + per-element panel permutation +
  rotation matrix).

### Where in the Claude loop

- A new Claude tool `evolve_controller` parallels `train`: Claude
  decides when an evolutionary warmup makes sense (e.g., after two
  PPO runs both plateau at low reward, suggesting exploration
  failure rather than reward-design failure).
- Symmetry augmentation is transparent to Claude — it's a
  configuration on the robot, not a tool call. Once declared in the
  robot's config, every subsequent `train` call benefits from it.

## Open questions

- Does icosahedral-symmetric augmentation degrade on non-flat terrain,
  or does only the relevant subgroup stay useful? Likely the latter —
  worth a controlled experiment.
- Is evosax's CMA-ES implementation numerically stable at pop_size=64
  with 168 parameters over 500 generations? Published benchmarks
  suggest yes, but worth sanity-checking on a known-good fitness
  landscape (e.g., Sphere, Rastrigin) before trusting it on urchin.
- At what parameter count does CMA-ES stop beating PPO's own
  exploration? Probably a few hundred — but the crossover depends on
  reward landscape shape.

## References

- Hansen, *The CMA Evolution Strategy: A Tutorial* (2023) — canonical
  CMA-ES reference
- `evosax`, Lange et al., JAX-native evolutionary strategies library
- Mittal et al., *Symmetry Considerations for Learning Task Symmetric
  Robot Policies* (2023) — quadruped symmetry augmentation
- Wang et al., *Equivariant MuZero* (2022) — equivariant networks for
  RL
- `escnn` — PyTorch-first library for E(n)-equivariant neural networks
