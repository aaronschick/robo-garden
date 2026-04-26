"""Collate W4 substrate-id CSV outputs into substrate_baseline.json.

Called by `scripts/run_urchin_v3_substrate_id.sh`. Standalone (rather than
heredoc'd) so heredoc quoting + path-translation issues across Git Bash /
WSL bash / Linux bash all disappear — argv is just plain absolute paths.

Args:
    out_dir       output root: workspace/_tasks_out/w4_substrate_id
    git_commit    pinned git rev so downstream W6/W7 can map readings to a
                  controller snapshot
    generated_at  ISO-8601 UTC timestamp the harness recorded
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _read_csv(p: Path):
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _mean(vals):
    vs = [v for v in vals if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--generated-at", default="unknown")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)

    single_csv = out_dir / "actuator_response" / "single_panel.csv"
    paired_csv = out_dir / "actuator_response" / "paired_panel.csv"

    actuator_block = {
        "rise_time_ms": None,
        "settle_time_ms": None,
        "overshoot_pct": None,
        "peak_velocity_mps": None,
        "panel_variance": {},
        "_status": "pending wsl run",
    }
    coupling_block = {
        "opposite_pair_delay_ms": None,
        "adjacent_pair_delay_ms": None,
        "_status": "pending wsl run",
    }

    single_rows = _read_csv(single_csv)
    if single_rows:
        rises = [_f(r.get("rise_time_ms")) for r in single_rows]
        settles = [_f(r.get("settle_time_ms")) for r in single_rows]
        overshoots = [_f(r.get("overshoot_pct")) for r in single_rows]
        peaks = [_f(r.get("peak_vel_mps")) for r in single_rows]
        actuator_block = {
            "rise_time_ms": _mean(rises),
            "settle_time_ms": _mean(settles),
            "overshoot_pct": _mean(overshoots),
            "peak_velocity_mps": _mean(peaks),
            "panel_variance": {
                r["panel_label"]: {
                    "rise_time_ms": _f(r.get("rise_time_ms")),
                    "settle_time_ms": _f(r.get("settle_time_ms")),
                    "overshoot_pct": _f(r.get("overshoot_pct")),
                    "peak_vel_mps": _f(r.get("peak_vel_mps")),
                }
                for r in single_rows
            },
            "_status": "ok",
        }

    paired_rows = _read_csv(paired_csv)
    if paired_rows:
        delays = {
            r["pair_label"]: _f(r.get("coupling_delay_ms")) for r in paired_rows
        }
        coupling_block = {
            "opposite_pair_delay_ms": delays.get("opposite"),
            "adjacent_pair_delay_ms": delays.get("adjacent"),
            "_status": "ok",
        }

    coast_summary = out_dir / "friction" / "coast_down.summary.csv"
    incline_summary = out_dir / "friction" / "incline_slip.summary.csv"

    friction_block = {
        "rolling_resistance_coef": None,
        "decay_tau_s": None,
        "static_mu_s": None,
        "incline_onset_deg": None,
        "_status": "pending wsl run",
    }

    coast_rows = _read_csv(coast_summary)
    if coast_rows:
        r = coast_rows[0]
        friction_block["rolling_resistance_coef"] = _f(
            r.get("rolling_resistance_coef")
        )
        friction_block["decay_tau_s"] = _f(r.get("tau_s"))
        friction_block["_status"] = "partial"

    incline_rows = _read_csv(incline_summary)
    if incline_rows:
        r = incline_rows[0]
        friction_block["static_mu_s"] = _f(r.get("static_mu_s"))
        friction_block["incline_onset_deg"] = _f(r.get("slip_onset_deg"))
        friction_block["_status"] = "ok" if coast_rows else "partial"

    baseline = {
        "actuator": actuator_block,
        "coupling": coupling_block,
        "friction": friction_block,
        "schema_version": 1,
        "generated_at": args.generated_at,
        "git_commit": args.git_commit,
    }

    out_path = out_dir / "substrate_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"[w4-substrate] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
