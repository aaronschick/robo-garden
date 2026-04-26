"""Audit primitive/style action separation on a RAW recorded dataset.

Sibling of `audit_conditioning.py` (which audits a trained BC checkpoint).
This one runs the same separation metric directly on ground-truth actions
from a `record_primitive_dataset.py` HDF5, so we can check whether the
data contains a differentiable signal before training BC on it.

Metric (mirroring `audit_conditioning.py::summarize`):
    mean_action[p, s] = mean of `action_sh` over all rows tagged (p, s)
    prim_sep  = mean L2 distance between {mean_action[p, s]}_p across pairs
                at fixed s, averaged over s
    style_sep = mean L2 distance between {mean_action[p, s]}_s across pairs
                at fixed p, averaged over p
    ratio     = sep / overall action std

A pass needs prim_sep/std >= 0.10 and style_sep/std >= 0.35 (Phase 4 gates).

Also splits by step_in_ep into "warmup" (first third) and "cruise" (last
third) to check the momentum-gate behavior:
  - Warmup: arc_l ≈ arc_r ≈ straight  (gate=0 at rest -> 3 identical primitives)
  - Cruise: arc_l ≠ arc_r ≠ straight  (gate saturated once rolling starts)
Plus reports mean |steering_bias| from `engine_cfg` in each window so we
can see the gate firing directly.
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np


PASS_PRIM = 0.10
PASS_STYLE = 0.35


def load_dataset(path: Path):
    with h5py.File(path, "r") as f:
        obs = f["obs"][...]
        act = f["action_sh"][...]
        cfg = f["engine_cfg"][...]
        prim_id = f["primitive_id"][...]
        style_id = f["style_id"][...]
        step = f["step_in_ep"][...]
        prim_names = [s.decode() if isinstance(s, bytes) else s
                      for s in f.attrs["primitive_names"]]
        style_names = [s.decode() if isinstance(s, bytes) else s
                       for s in f.attrs["style_names"]]
        cfg_fields = [s.decode() if isinstance(s, bytes) else s
                      for s in f.attrs["engine_cfg_fields"]]
        steps_per_ep = int(f.attrs["steps_per_ep"])
        env_dt = float(f.attrs["env_dt"])
    return {
        "obs": obs, "act": act, "cfg": cfg,
        "prim_id": prim_id, "style_id": style_id, "step": step,
        "prim_names": prim_names, "style_names": style_names,
        "cfg_fields": cfg_fields,
        "steps_per_ep": steps_per_ep, "env_dt": env_dt,
    }


def per_combo_mean_action(data, mask):
    prim_names = data["prim_names"]
    style_names = data["style_names"]
    act = data["act"][mask]
    pid = data["prim_id"][mask]
    sid = data["style_id"][mask]
    means = {}
    counts = {}
    for pi, si in product(range(len(prim_names)), range(len(style_names))):
        m = (pid == pi) & (sid == si)
        n = int(m.sum())
        if n == 0:
            continue
        means[(prim_names[pi], style_names[si])] = act[m].mean(axis=0)
        counts[(prim_names[pi], style_names[si])] = n
    return means, counts


def separation(means, prim_names, style_names):
    prim_sep = []
    for s in style_names:
        keys = [(p, s) for p in prim_names if (p, s) in means]
        if len(keys) < 2:
            continue
        v = np.stack([means[k] for k in keys])
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
        prim_sep.append(d[np.triu_indices(len(keys), k=1)])
    style_sep = []
    for p in prim_names:
        keys = [(p, s) for s in style_names if (p, s) in means]
        if len(keys) < 2:
            continue
        v = np.stack([means[k] for k in keys])
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
        style_sep.append(d[np.triu_indices(len(keys), k=1)])
    prim_sep = np.concatenate(prim_sep) if prim_sep else np.zeros(0)
    style_sep = np.concatenate(style_sep) if style_sep else np.zeros(0)
    return prim_sep, style_sep


def report_window(label, data, mask):
    prim_names = data["prim_names"]
    style_names = data["style_names"]
    means, counts = per_combo_mean_action(data, mask)
    act_window = data["act"][mask]
    overall_std = float(act_window.std()) if act_window.size else 0.0
    prim_sep, style_sep = separation(means, prim_names, style_names)
    n = int(mask.sum())

    print(f"\n=== {label}  (N={n} transitions, overall act std={overall_std:.4f}) ===")
    if overall_std == 0.0:
        print("  (no data in window)")
        return

    for p in prim_names:
        row = []
        for s in style_names:
            if (p, s) in means:
                row.append(f"{s}={np.linalg.norm(means[(p, s)]):.4f}")
            else:
                row.append(f"{s}=---")
        print(f"  prim={p:<14} ||mean_a||: {'  '.join(row)}")

    if prim_sep.size == 0 or style_sep.size == 0:
        print("  (not enough combos for separation)")
        return

    prim_ratio = prim_sep.mean() / overall_std
    style_ratio = style_sep.mean() / overall_std

    prim_ok = prim_ratio >= PASS_PRIM
    style_ok = style_ratio >= PASS_STYLE
    print(f"\n  prim_sep  mean={prim_sep.mean():.4f}  median={np.median(prim_sep):.4f}  max={prim_sep.max():.4f}")
    print(f"  style_sep mean={style_sep.mean():.4f}  median={np.median(style_sep):.4f}  max={style_sep.max():.4f}")
    print(f"  ratio prim_sep/std  = {prim_ratio:.3f}  (pass >={PASS_PRIM})  {'OK' if prim_ok else 'FAIL'}")
    print(f"  ratio style_sep/std = {style_ratio:.3f}  (pass >={PASS_STYLE})  {'OK' if style_ok else 'FAIL'}")
    return {"prim_ratio": prim_ratio, "style_ratio": style_ratio,
            "prim_ok": prim_ok, "style_ok": style_ok}


def report_gate(label, data, mask):
    """Dump mean |steering_bias| and |ang_vel| by combo in this window,
    so we can see the gate firing as designed."""
    prim_names = data["prim_names"]
    style_names = data["style_names"]
    cfg = data["cfg"][mask]
    obs = data["obs"][mask]
    pid = data["prim_id"][mask]
    sid = data["style_id"][mask]
    cfg_fields = data["cfg_fields"]
    sb_col = cfg_fields.index("steering_bias")

    ang_vel = obs[:, 3:6]  # convention: obs[:, 3:6] = ang_vel_b
    ang_vel_mag = np.linalg.norm(ang_vel, axis=-1)

    print(f"\n  -- {label} gate trace  (|steering_bias| and |ang_vel|) --")
    for pi, p in enumerate(prim_names):
        parts = []
        for si, s in enumerate(style_names):
            m = (pid == pi) & (sid == si)
            if not m.any():
                parts.append(f"{s}=---")
                continue
            sb_mean = float(np.abs(cfg[m, sb_col]).mean())
            wz_mean = float(ang_vel_mag[m].mean())
            parts.append(f"{s}: |sb|={sb_mean:.3f} |wz|={wz_mean:.3f}")
        print(f"    prim={p:<14} {'  '.join(parts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--warmup-frac", type=float, default=1.0 / 3.0)
    ap.add_argument("--cruise-frac", type=float, default=1.0 / 3.0,
                    help="Cruise window is the LAST cruise-frac of each episode.")
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    print(f"dataset: {args.dataset}")
    print(f"  rows={data['obs'].shape[0]}  obs_dim={data['obs'].shape[1]}  "
          f"act_dim={data['act'].shape[1]}")
    print(f"  primitives={data['prim_names']}")
    print(f"  styles={data['style_names']}")
    print(f"  steps_per_ep={data['steps_per_ep']}  env_dt={data['env_dt']:.4f}s")

    n_ep = data["steps_per_ep"]
    warmup_end = int(args.warmup_frac * n_ep)
    cruise_start = int((1.0 - args.cruise_frac) * n_ep)
    print(f"\n  warmup window: step_in_ep < {warmup_end}  "
          f"(~first {warmup_end * data['env_dt']:.2f}s)")
    print(f"  cruise window: step_in_ep >= {cruise_start}  "
          f"(~last {(n_ep - cruise_start) * data['env_dt']:.2f}s)")

    full_mask = np.ones(data["obs"].shape[0], dtype=bool)
    warmup_mask = data["step"] < warmup_end
    cruise_mask = data["step"] >= cruise_start

    report_window("full episode", data, full_mask)
    report_gate("full", data, full_mask)

    warm = report_window("warmup (gate ~0, arcs should collapse to straight)",
                         data, warmup_mask)
    report_gate("warmup", data, warmup_mask)

    cruise = report_window("cruise (gate saturated, arcs should differentiate)",
                           data, cruise_mask)
    report_gate("cruise", data, cruise_mask)

    print("\n=== verdict ===")
    if cruise is None:
        print("  cruise window empty — cannot conclude")
        return
    if cruise["prim_ok"] and cruise["style_ok"]:
        print(f"  CRUISE PASS: prim_ratio={cruise['prim_ratio']:.3f} >= {PASS_PRIM}, "
              f"style_ratio={cruise['style_ratio']:.3f} >= {PASS_STYLE}")
        print("  -> data-level conditioning signal is adequate; proceed to re-record Phase 3.")
    else:
        print(f"  CRUISE FAIL: prim_ratio={cruise['prim_ratio']:.3f} (>= {PASS_PRIM}?), "
              f"style_ratio={cruise['style_ratio']:.3f} (>= {PASS_STYLE}?)")
        print("  -> fix is insufficient at the data level; escalate beyond gating.")


if __name__ == "__main__":
    main()
