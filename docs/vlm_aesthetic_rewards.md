# VLM-as-Aesthetic-Reward

**Status:** Design concept, not yet implemented.
**Scope:** Extension to the existing Eureka-style reward loop (`rewards/`) and Claude tool dispatch (`claude/tool_handlers.py`).
**Prerequisite phases:** Phase 1 (robot+sim loop) and Phase 4 (MJX training) working end-to-end. Phase 5 (Eureka reward loop) is where this naturally slots in.

## TL;DR

Instead of writing a reward function in Python that scores numeric state
(velocity, joint positions, contact forces), render what the robot is
doing and hand the video clip to a vision-language model with a
natural-language criterion:

> *"On a scale of 0–10, how much does this robot look like it's breathing?"*

The VLM's score **is** the reward. The policy learns to produce behavior
that scores highly. "Breathing" is never defined in code — it's
described, and the VLM bridges language → visual perception → scalar
reward.

This is the only tractable way to train on aesthetic or affective
criteria, because "looks alive" has no closed-form expression in joint
angles.

## Why this is novel

Near-neighbor prior work and what's missing from each:

| Work | What it does | Why it's not this |
|------|--------------|-------------------|
| **Eureka** (NVIDIA 2023) | LLM writes reward *code* | Task-oriented, numeric. `rewards/` already does this. |
| **RoboCLIP, LIV, VLM-RM** (2023–24) | VLMs score task *success* | Binary/shaped task reward, not aesthetic. |
| **RLAIF** | LLM preference data for LLMs | Not embodied. |
| **Generative art + RL** | A handful of hobbyist sketches | No production pipeline. |

What's almost entirely unexplored: **using a VLM as a judge of aesthetic
or affective quality for a physical robot's motion.** Robot = generator,
VLM = critic, RL = the optimization that closes the loop. This is the
natural extension of GANs → diffusion → RLHF applied to embodied motion.

## Why this stack specifically

Three compounding reasons robo-garden is well-positioned:

1. **Claude-in-the-loop infrastructure already exists.** `claude/client.py`
   (tool dispatch), `claude/tools.py` (schemas), and `claude/tool_handlers.py`
   (dispatch wiring) are the natural host for a `score_clip` tool.
2. **MJX already renders frames.** `workspace/renders/` shows episode videos
   being produced today (e.g. `urchin_v2_dryrun.mp4`). Scoring pipeline
   consumes what's already generated.
3. **Slow-iteration tolerance is baked in.** Eureka-style reward
   refinement already tolerates API-latency in the inner loop. VLM
   scoring fits the same cadence — it doesn't violate any existing
   latency assumption.
4. **WSL + Windows split matches the cost model.** VLM calls stay
   Windows-side (where the Anthropic API key lives), training stays WSL-side.
   No new auth plumbing.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ WSL2 (GPU)                                                     │
│                                                                │
│   MJX / Brax PPO ────► episode rollout ────► render frames     │
│       ▲                                            │           │
│       │                                            ▼           │
│   policy update ◄── advantage ◄── reward ─── dense reward      │
│                                    +                           │
│                                 sparse VLM score ──┐           │
└────────────────────────────────────────────────────┼───────────┘
                                                     │ stdout JSONL
┌────────────────────────────────────────────────────┼───────────┐
│ Windows (API host)                                 ▼           │
│                                                                │
│   frames (N=8, stride=~250ms) ────► Haiku 4.5 (per episode)    │
│                                              │                 │
│                                     score ───┘                 │
│                                                                │
│   periodically: clips + scores ────► Opus 4.7 (calibration)    │
│                                                                │
│   every ~10k labeled clips ────► train small CLIP critic       │
│   learned critic eventually replaces VLM for most episodes     │
└────────────────────────────────────────────────────────────────┘
```

## Two-tier reward (the pattern that actually works)

Naive "call VLM every step" dies instantly — cost, latency, and PPO hates
sparse reward. The working structure:

### Dense physical reward (per step, free, handwritten)

Keeps the policy in the feasible manifold so the VLM ever sees something
worth judging:

- don't fall over, don't explode the simulation
- stay within bounds
- energy cost / smoothness priors as needed

This lives in the normal `rewards/<name>.py` Python file and executes in-process.

### Sparse aesthetic reward (per episode, from VLM)

One score every ~256–512 sim steps, broadcast across the episode via
standard advantage estimation. The policy learns "this episode overall
was 8/10 breath-like" and credit-assigns through PPO's returns.

## VLM sampling strategy

- **Stride frames.** 8 frames over a 1–2s clip is enough for a VLM to
  judge motion quality. Whole episodes overwhelm the model and get
  summarized away.
- **Tiered judges.**
  - **Haiku 4.5** is the everyday scorer — fast, ~$1/M input tokens.
  - **Opus 4.7** runs occasional calibration passes and end-of-epoch
    evaluation (catches Haiku drift / scoring bias).
- **Cache by frame-hash.** Early training produces near-duplicate
  rollouts. Don't pay twice for the same clip.
- **Prefer pairwise preferences over absolute scores.** "Is A more alive
  than B?" is far more consistent across VLM calls than "rate A 0–10" —
  same reason RLHF moved to preferences.

## Distill to a learned critic (the scaling move)

This is what turns the concept from "toy" into "trainable at scale":

1. Log every `(frames, VLM score)` pair during training.
2. Train a small vision encoder (CLIP-initialized, ~50M params) to
   predict the VLM score.
3. After ~10k labeled clips, the learned critic replaces the VLM for
   most episodes. VLM is only called periodically to prevent reward
   hacking and detect critic drift.
4. Iterate: as the policy evolves, keep adding fresh VLM labels so the
   critic doesn't fall behind the behavior distribution.

Without distillation: cost dominated by API calls, throughput capped by
API latency.
With distillation: the learned critic runs on GPU alongside MJX at tens
of thousands of scoring-ops per second.

## Prompt as hyperparameter

The Eureka-style refinement loop already iterates Python reward code.
With VLM rewards, it iterates **prompts** instead:

- **v1:** "Rate how alive this looks, 0–10."
- **v2:** "Rate this robot's motion for organic, unhurried, breath-like
  rhythm, 0–10. Penalize jitter and mechanical repetition."
- **v3:** same as v2, with chain-of-thought requested before the number
  (usually reduces variance).

Training stats (score variance, policy convergence behavior, qualitative
rollouts) feed back to Claude, which proposes prompt revisions. Prompt
engineering *is* reward engineering in this regime.

## Cost envelope

Back-of-envelope for a single training run, 100k episodes:

| Config | Per episode | Run total |
|--------|-------------|-----------|
| Opus 4.7, no distillation | ~$0.0015 | ~$150 |
| Haiku 4.5, no distillation | ~$0.00015 | ~$15–25 |
| Haiku + distilled critic (after 10k labels) | ~$0.00002 amortized | ~$5 |

Assumes 8 frames × ~1.5k tokens/frame + ~500 output tokens per call.
Real numbers will shift with Anthropic pricing; treat as order-of-magnitude.

## Gotchas

1. **Reward hacking is severe.** VLMs have biases — policies *will* find
   adversarial poses that score 10/10 while looking nothing like the
   intent. Mitigations:
   - Ensemble of judges (Opus + Haiku + CLIP + occasional human).
   - KL penalty against a baseline "natural motion" prior.
   - Adversarial-frame screening (flag clips that score very high but
     have low pixel-space similarity to known-good exemplars).
2. **VLM scoring is noisy.** Absolute scores drift ±1–2 points between
   calls on the same clip. Use pairwise preferences, or average over
   multiple VLM calls per clip.
3. **Cost is real.** Budget per training run, watch rate limits, prefer
   Haiku by default, distill aggressively.
4. **Clip length matters.** 1–3 second clips are judged well; 10+ second
   clips get summarized. Design episode length around this.
5. **Claude-the-critic can disagree with Claude-the-reward-author.** Log
   both and flag divergence — often reveals a vague prompt.
6. **Mode collapse on a single peak.** Policy finds one very-high-scoring
   behavior and stays there. Counter with novelty bonuses (RND-style) or
   require the VLM to compare against a buffer of past behaviors
   ("is this more interesting than what this robot did yesterday?").

## Minimal first milestone

A concrete proof-of-concept scoped to ~1 week of engineering once Phase 5
is reached:

1. Add a `score_clip` handler in `claude/tool_handlers.py` that:
   - Takes a path to an episode video (already produced by MJX render).
   - Extracts 8 stride-sampled frames.
   - Calls Haiku 4.5 with a fixed prompt.
   - Returns a float in [0, 10].
2. Extend `rewards/` to allow a `vlm_bonus` term — called once per episode
   at done-time, added to the return.
3. Pick urchin_v2 (already training). Keep all existing physical
   rewards. Add *one* VLM-scored term: "how much does this robot look
   like a breathing creature at rest."
4. Train to convergence. Inspect rollouts.
5. **Success criterion:** policy discovers a crouch-and-breathe behavior
   that was not expressible in the Python reward.

If that lands, every concept in `docs/` (or future `docs/concepts/`) —
feather displays, pendulum choirs, tensegrity mobiles — becomes a viable
canvas, and the aesthetic-reward pipeline is the reusable core.

## Open questions

- **Preference vs. absolute scoring for the initial PoC.** Preferences
  are more stable but require a buffer of comparison clips, which
  complicates the first implementation. Absolute scoring is simpler —
  start there, migrate to preferences if variance is intolerable.
- **Where does the critic distillation code live?** Plausibly
  `rewards/learned_critic/` as a sibling to hand-authored rewards.
- **Human-in-the-loop overrides.** Should a human be able to re-score a
  small fraction of clips to correct VLM drift? Adds a labeling UI but
  drastically improves alignment.
- **Multi-prompt training.** Can one policy learn to satisfy a set of
  aesthetic prompts simultaneously (conditioned policy: "now breathe,
  now dance")? Moves from fixed-reward RL to goal-conditioned RL.

## Related files when this gets built

- `claude/tool_handlers.py` — add `score_clip` handler
- `claude/tools.py` — register `score_clip` tool schema
- `rewards/` — new reward type that calls VLM at episode boundary
- `training/gym_env.py` — hook episode-done into VLM scoring
- `training/wsl_dispatch.py` — ensure episode videos surface to
  Windows-side for scoring (already partially wired)
