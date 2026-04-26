"""W6 (2026-04-25) — Latin-Hypercube sample generator for the urchin_v3
locomotion atlas.

Per the deep-research report (`docs/Urchin-deep-research-report.md`
§"Experimental Plan", Table row 4 + §"Parameter ranges to try"), the W3
explicit three-field controller exposes five primary knobs that span
the steady-rolling basin. We sample those with a Latin-Hypercube design
so every 1-D and 2-D projection is well-covered with O(N) samples
(rather than the O(N^5) a grid would need).

Outputs:
    workspace/_tasks_out/w6_atlas/lhs_samples.csv

Columns:
    sample_id, rear_push_amp, front_retract_amp, lean_phase,
    phase_velocity_hz, push_duty

Pure stdlib + numpy + (optional) scipy. No Isaac Lab. Importable on
Windows. The Isaac harness `w6_atlas_run.py` consumes this CSV.

Usage:
    uv run python workspace/scratch/w6_atlas_lhs.py
    uv run python workspace/scratch/w6_atlas_lhs.py --num-samples 16 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# Parameter ranges from the deep-research report. Keep in sync with
# `w6_atlas_run.py::PARAM_NAMES` and the README/report doc strings.
PARAM_RANGES: list[tuple[str, float, float]] = [
    ("rear_push_amp",      0.35, 0.90),
    ("front_retract_amp",  0.10, 0.80),
    ("lean_phase",         0.05, 0.17),   # in cycles, mapped to radians by run script
    ("phase_velocity_hz",  0.40, 2.50),
    ("push_duty",          0.35, 1.00),
]


def _lhs_scipy(n: int, d: int, seed: int) -> np.ndarray:
    """Latin Hypercube via scipy.stats.qmc, the preferred path."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n=n)


def _lhs_inline(n: int, d: int, seed: int) -> np.ndarray:
    """Pure-numpy LHS fallback. Stratified per-dim, randomly permuted.

    Each dimension's [0, 1] range is split into n equal bins; one sample
    is drawn uniformly from each bin; the bin order is independently
    permuted per dimension. This is a textbook standard LHS — not
    optimised for low-discrepancy (no maximin / Halton blending), but
    correct and dependency-free.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty((n, d), dtype=np.float64)
    for dim in range(d):
        # Stratified offsets in [0, 1).
        u = rng.uniform(0.0, 1.0, size=n)
        idx = rng.permutation(n)
        samples[:, dim] = (idx + u) / n
    return samples


def generate_lhs(num_samples: int, seed: int) -> np.ndarray:
    """Return an (n, 5) array of LHS unit-cube samples."""
    d = len(PARAM_RANGES)
    try:
        unit = _lhs_scipy(num_samples, d, seed)
    except Exception as exc:                                  # pragma: no cover
        print(f"[w6-lhs] scipy.stats.qmc unavailable ({exc!r}); using "
              "inline numpy LHS.", file=sys.stderr)
        unit = _lhs_inline(num_samples, d, seed)
    return unit


def scale_samples(unit: np.ndarray) -> np.ndarray:
    """Map unit-cube LHS samples to PARAM_RANGES."""
    out = np.empty_like(unit)
    for dim, (_, lo, hi) in enumerate(PARAM_RANGES):
        out[:, dim] = lo + unit[:, dim] * (hi - lo)
    return out


def write_csv(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["sample_id"] + [name for name, _, _ in PARAM_RANGES]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for sid, row in enumerate(samples):
            writer.writerow([sid] + [f"{v:.6f}" for v in row])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-samples", type=int, default=256,
                        dest="num_samples",
                        help="Number of LHS samples (default 256). "
                             "Use 16 for smoke validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="LHS RNG seed.")
    parser.add_argument(
        "--output", type=Path,
        default=Path("workspace/_tasks_out/w6_atlas/lhs_samples.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    if args.num_samples < 2:
        parser.error("--num-samples must be >= 2 for LHS")

    unit = generate_lhs(args.num_samples, args.seed)
    samples = scale_samples(unit)
    write_csv(args.output, samples)

    print(f"[w6-lhs] wrote {args.num_samples} samples × "
          f"{len(PARAM_RANGES)} dims -> {args.output.resolve()}")
    print("[w6-lhs] param ranges:")
    for name, lo, hi in PARAM_RANGES:
        print(f"            {name:22s} [{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
