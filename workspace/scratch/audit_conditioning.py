"""Audit how strongly a BC checkpoint conditions its action on primitive_id and
style_id. If the one-hot inputs barely affect the output, the policy has
collapsed to an averaged gait regardless of what data we feed it.

Compares phase3_oraclehalf (parent) vs phase4_iter1 (DAgger-retrained) on the
same obs batch so we can see whether iter 1 sharpened or degraded conditioning.
"""
from __future__ import annotations
import argparse
import sys
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]


class FiLMPolicy(nn.Module):
    """Inline mirror of train_bc.py FiLMPolicy."""
    def __init__(self, obs_dim, cond_dim, hidden, action_dim):
        super().__init__()
        dims = [obs_dim, *hidden]
        self.trunk = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.film = nn.ModuleList(
            nn.Linear(cond_dim, 2 * h) for h in hidden
        )
        self.head = nn.Linear(hidden[-1], action_dim)

    def forward(self, obs_scaled, cond):
        x = obs_scaled
        for layer, film in zip(self.trunk, self.film):
            gamma, beta = film(cond).chunk(2, dim=-1)
            x = torch.relu((1.0 + gamma) * layer(x) + beta)
        return self.head(x)


def build_policy(ckpt_path: Path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = blob["arch"]
    arch_type = arch.get("arch_type", "concat")
    if arch_type == "concat":
        dims = [arch["obs_dim"] + arch["cond_dim"], *arch["hidden"], arch["action_dim"]]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        net = nn.Sequential(*layers).to(device)
        sd = blob["state_dict"]
        if any(k.startswith("net.") for k in sd):
            sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
        net.load_state_dict(sd)
    elif arch_type == "film":
        net = FiLMPolicy(
            obs_dim=arch["obs_dim"], cond_dim=arch["cond_dim"],
            hidden=arch["hidden"], action_dim=arch["action_dim"],
        ).to(device)
        net.load_state_dict(blob["state_dict"])
    else:
        raise SystemExit(f"unknown arch_type={arch_type!r} in {ckpt_path}")
    net.eval()
    obs_mean = torch.from_numpy(np.asarray(blob["obs_mean"], dtype=np.float32)).to(device)
    obs_std  = torch.from_numpy(np.asarray(blob["obs_std"],  dtype=np.float32)).to(device)
    return {
        "net": net, "arch_type": arch_type,
        "obs_mean": obs_mean, "obs_std": obs_std,
        "prim_names":  list(blob["metadata"]["primitive_names"]),
        "style_names": list(blob["metadata"]["style_names"]),
        "arch": arch,
    }


def build_cond(prim_idx, style_idx, n_prim, n_style, batch, device):
    onehot = torch.zeros(batch, n_prim + n_style, device=device)
    onehot[:, prim_idx] = 1.0
    onehot[:, n_prim + style_idx] = 1.0
    return onehot


def sweep_actions(policy, obs_batch, device):
    """Return dict[(prim, style)] -> mean action across obs_batch."""
    n_prim  = len(policy["prim_names"])
    n_style = len(policy["style_names"])
    obs_n = (obs_batch - policy["obs_mean"]) / policy["obs_std"].clamp_min(1e-6)
    out = {}
    with torch.no_grad():
        for pi, si in product(range(n_prim), range(n_style)):
            cond = build_cond(pi, si, n_prim, n_style, obs_n.shape[0], device)
            if policy["arch_type"] == "film":
                a = policy["net"](obs_n, cond)
            else:
                a = policy["net"](torch.cat([obs_n, cond], dim=-1))
            out[(policy["prim_names"][pi], policy["style_names"][si])] = a.cpu().numpy()
    return out


def summarize(actions_map, label):
    prims  = sorted({p for p, _ in actions_map.keys()})
    styles = sorted({s for _, s in actions_map.keys()})
    mean_action = {k: v.mean(axis=0) for k, v in actions_map.items()}
    per_obs_var  = {k: v.std(axis=0).mean()  for k, v in actions_map.items()}
    overall_std = np.concatenate([v for v in actions_map.values()]).std()
    print(f"\n=== {label} ===")
    print(f"overall action std (across all obs, primitives, styles): {overall_std:.4f}")
    print(f"action-dim={mean_action[next(iter(mean_action))].shape[0]}")
    for p in prims:
        row = "  ".join(f"{s}={np.linalg.norm(mean_action[(p, s)]):.4f}" for s in styles)
        print(f"  prim={p:<14} ||mean_a||: {row}")

    # Primitive separation (hold style fixed, vary primitive)
    prim_sep = []
    for s in styles:
        v = np.stack([mean_action[(p, s)] for p in prims])
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
        prim_sep.append(d[np.triu_indices(len(prims), k=1)])
    prim_sep = np.concatenate(prim_sep)
    # Style separation (hold primitive fixed, vary style)
    style_sep = []
    for p in prims:
        v = np.stack([mean_action[(p, s)] for s in styles])
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
        style_sep.append(d[np.triu_indices(len(styles), k=1)])
    style_sep = np.concatenate(style_sep)

    print(f"\n  primitive-conditioning separation ||delta_a|| across primitives (style fixed):")
    print(f"    mean={prim_sep.mean():.4f}  median={np.median(prim_sep):.4f}  max={prim_sep.max():.4f}")
    print(f"  style-conditioning separation ||delta_a|| across styles (primitive fixed):")
    print(f"    mean={style_sep.mean():.4f}  median={np.median(style_sep):.4f}  max={style_sep.max():.4f}")
    print(f"  ratio prim_sep/overall_std={prim_sep.mean()/overall_std:.3f}"
          f"  style_sep/overall_std={style_sep.mean()/overall_std:.3f}")
    return {"prim_sep": prim_sep, "style_sep": style_sep, "mean_action": mean_action}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True,
                    help="comma-separated checkpoint paths")
    ap.add_argument("--dataset", type=Path,
                    default=REPO / "workspace/datasets/urchin_v3_primitives_phase3.h5")
    ap.add_argument("--n-obs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.dataset, "r") as f:
        total = f["obs"].shape[0]
        idx = np.sort(rng.choice(total, size=args.n_obs, replace=False))
        obs_np = f["obs"][idx].astype(np.float32)
    obs_batch = torch.from_numpy(obs_np).to(device)
    print(f"obs batch: {obs_batch.shape}  from {args.dataset.name}")

    for ckpt in args.ckpts.split(","):
        ckpt_path = Path(ckpt.strip())
        policy = build_policy(ckpt_path, device)
        actions = sweep_actions(policy, obs_batch, device)
        summarize(actions, ckpt_path)


if __name__ == "__main__":
    main()
