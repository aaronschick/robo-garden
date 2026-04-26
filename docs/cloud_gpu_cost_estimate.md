# Cloud GPU Cost Estimate — Urchin Training Runs

## Context

Databricks Free Edition is CPU-only and sandboxed, so it cannot run the Isaac Lab / Brax-MJX training pipeline. This doc estimates what realistic cloud-GPU alternatives (Lambda, RunPod, Vast.ai) would cost per training run, so we can weigh cloud dispatch against the local RTX 3070.

Intended as a pre-feature reference for a future `cloud_dispatch.py` that mirrors `src/robo_garden/training/wsl_dispatch.py` but targets a rented GPU pod.

## Baseline: what a "run" actually costs in time

Pulled from `scripts/run_urchin_v3_harden_chain.sh`, `workspace/robots/urchin_v3/scripts/train.py`, and `workspace/checkpoints/urchin_v2/*/result.json`:

| Run type | Timesteps | Wall-clock on RTX 3070 | Throughput |
|---|---|---|---|
| Smoke chain | 80k | 3–4 min | ~1.3M/hr (short, warm-up dominated) |
| Default single run | 500k | 12–23 min | ~1.5–2.5M/hr |
| 1M single run | 1M | 31–32 min | ~1.9M/hr |
| **Full harden chain** | **11M (1+2+2+2+2+2)** | **~5.5–6 hr** | ~1.9M/hr sustained |

All use `--num-envs 64`, 20s episodes, skrl 2.0 PPO in Isaac Lab. The 3070 is the measured reference.

## Cloud-GPU price points (early 2026, on-demand unless noted)

| GPU | Lambda | RunPod secure | RunPod community | Vast.ai |
|---|---|---|---|---|
| RTX 3090 / A5000 | — | ~$0.44/hr | ~$0.22/hr | $0.15–0.30/hr |
| RTX 4090 | — | ~$0.74/hr | ~$0.34/hr | $0.25–0.50/hr |
| A10 / A6000 | $0.75–0.80/hr | $0.79/hr | — | $0.50–0.70/hr |
| A100 80GB | $1.79/hr | $1.89/hr | $1.19/hr | $0.80–1.20/hr |
| H100 80GB | $2.49–3.29/hr | $3.29/hr | $1.99/hr | $1.80–2.50/hr |

Prices change weekly; treat as ±30%. Lambda is mid-tier on price but reliable; Vast is cheapest with variable reliability.

## Throughput assumptions

Isaac Lab with 64 parallel envs is bound by Isaac Sim GPU physics + PyTorch PPO, not pure tensor ops. Scaling from 3070 is sub-linear — rough multipliers vs 3070:

- RTX 4090: ~1.8×
- A100 80GB: ~2.0–2.5× (memory bandwidth wins, clocks lose; good when you push envs to 256+)
- H100: ~3× only if envs scaled up; ~1.5× at 64 envs (underutilized)

**Key lever:** increasing `--num-envs` to 256–1024 on A100/H100 jumps throughput 3–5×. The 64-env default is 3070-sized.

## Estimated cost per run

Per-run cost = wall-clock hours × $/hr. Using sustained 3070 throughput (1.9M/hr) and the multipliers above:

### At default `--num-envs=64` (no code changes)

| Run | RTX 4090 @ $0.34 | A100 @ $1.19 | H100 @ $1.99 |
|---|---|---|---|
| 500k single | ~8 min → **$0.05** | ~6 min → **$0.12** | ~5 min → **$0.16** |
| 1M single | ~17 min → **$0.10** | ~13 min → **$0.26** | ~10 min → **$0.33** |
| 11M harden chain | ~3.0 hr → **$1.02** | ~2.4 hr → **$2.86** | ~2.0 hr → **$3.98** |

### At `--num-envs=256+` (requires bumping the flag in train.py + chain script)

| Run | A100 @ $1.19 | H100 @ $1.99 |
|---|---|---|
| 11M harden chain | ~50 min → **$1.00** | ~35 min → **$1.15** |

So for **urchin-sized work**, a full harden chain is roughly **$1–4 per run** on sane hardware. A 500k exploration run is **nickels**.

## Recommendation by use case

- **Iterative dev / reward tuning (500k–1M runs, many per day):** RunPod community RTX 4090 at ~$0.34/hr. ~$1–2 per 11M chain, ~$0.10 per 1M probe.
- **One-off overnight hardening chains:** Lambda / RunPod secure A100 at $1.19–1.79/hr. Reliable, boots fast, ~$3 per chain.
- **Parallel sweeps (e.g. 8 LPF variants):** rent a single A100 80GB and run 4–8 concurrent skrl workers (each ~64 envs fits in 80GB); ~$3–5 for the whole sweep.

## Hidden costs worth naming

1. **Storage + setup time.** First-time Isaac Lab install on a fresh pod is 20–40 min and ~30GB of wheels/assets. Budget $0.50–1.00 of "wasted" time unless you build a custom image.
2. **Isaac Sim image.** Isaac Lab wants NVIDIA's container (`nvcr.io/nvidia/isaac-lab`). RunPod has official Isaac Sim templates; Vast community nodes usually don't.
3. **Data egress.** Checkpoints (~50–200MB) and videos are small; negligible at <$0.01/run.
4. **Spot / community preemption.** RunPod community and Vast can be killed mid-run. Fine for harden-chain stages (checkpointed every stage), risky for a single long run. Use secure-cloud A100 for uninterrupted long runs.

## Bottom line

For current urchin cadence, expect **~$1–4 per full harden chain** on appropriate cloud GPU, **~$0.10 per 1M-step probe**. Monthly spend for a typical dev week (20 probes + 5 chains) lands at **$10–25/month** — an order of magnitude cheaper than paid Databricks with a GPU cluster.

## Future feature: `cloud_dispatch.py`

A natural next addition to the training backend. It would mirror `src/robo_garden/training/wsl_dispatch.py`:

1. Stage job dir (`workspace/_cloud_jobs/<run_id>/job.json`) with robot XML, env MJCF, reward source, hyperparams — same schema as the WSL worker.
2. `rsync`/`scp` the staged dir to a rented pod (RunPod API / Lambda CLI to spin up).
3. `ssh` in, `uv run robo-garden --mode train --cloud-worker <remote_dir>` (new flag parallel to `--wsl-worker`).
4. Stream `__RG_PROGRESS__` JSONL over SSH stdout back to the Studio UI.
5. On completion, `scp` `result.json` + checkpoints back; optionally auto-terminate the pod.

Minimal code footprint because the worker protocol (job.json in, result.json out, progress lines over stdout) is already defined for WSL.

## Verification steps before committing to cloud

Ground-truth the numbers before building dispatch:

1. Rent one hour on RunPod community RTX 4090 (~$0.40 total).
2. Clone repo, `uv sync`, install Isaac Lab pip (or pull official image).
3. Run `scripts/smoke_urchin_v3_chain.sh` (80k timesteps). Confirm ≤4 min completion.
4. Run a single 1M-step urchin_v3 stage; note wall-clock. Multiply by 11 for chain estimate, by $/hr for cost estimate.

If smoke passes and 1M finishes in ≤20 min on 4090, these estimates hold.

## Files referenced

- `scripts/run_urchin_v3_harden_chain.sh` — chain stage timesteps
- `scripts/smoke_urchin_v3_chain.sh` — smoke budget
- `workspace/robots/urchin_v3/scripts/train.py` — defaults (`--num-envs 64`, 500k timesteps)
- `workspace/checkpoints/urchin_v2/*/result.json` — measured wall-clock baselines
- `src/robo_garden/training/wsl_dispatch.py` — dispatch pattern to mirror for cloud
- `CLAUDE.md` — hardware context (3070 8GB target, Isaac Lab cloud A100/H100 intent)
