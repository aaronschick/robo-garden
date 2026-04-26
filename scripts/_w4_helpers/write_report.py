"""Render workspace/_tasks_out/w4_substrate_id/report.md from the JSON baseline.

Called by `scripts/run_urchin_v3_substrate_id.sh`. Standalone (rather than
heredoc'd) so shell-quote escaping of the WSL launch command snippet inside
the report doesn't break the harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt(x, fallback: str = "(pending)") -> str:
    if x is None:
        return fallback
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


WSL_LAUNCH_BLOCK = """\
```bash
# From a Windows PowerShell:
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/aaron/Documents/repositories/robo-garden && URCHIN_RESET_MODE=canonical bash scripts/run_urchin_v3_substrate_id.sh"

# Or from inside a WSL shell:
cd /mnt/c/Users/aaron/Documents/repositories/robo-garden
URCHIN_RESET_MODE=canonical bash scripts/run_urchin_v3_substrate_id.sh
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    baseline_path = out_dir / "substrate_baseline.json"
    if not baseline_path.exists():
        print(f"[w4-substrate] ERROR: missing {baseline_path}", file=sys.stderr)
        return 1
    b = json.loads(baseline_path.read_text(encoding="utf-8"))

    a = b.get("actuator", {})
    c = b.get("coupling", {})
    f = b.get("friction", {})

    lines: list[str] = []
    lines.append("# W4 - Urchin v3 Substrate Identification\n")
    lines.append(f"_Generated: {b.get('generated_at')}_  ")
    lines.append(f"_Git commit: {b.get('git_commit')}_\n")
    lines.append(
        "Source: docs/Urchin-deep-research-report.md Table row 1-2 (HIGHEST priority)."
    )
    lines.append(
        "Consumed by: W6 (locomotion atlas), W7 (slip-aware scheduler)."
    )
    lines.append("")

    lines.append("## Actuator step response (single panel)\n")
    lines.append("| metric | mean across 4 panels |")
    lines.append("|---|---|")
    lines.append(f"| rise time (10% to 90%) ms | {fmt(a.get('rise_time_ms'))} |")
    lines.append(f"| settle time (5% band) ms | {fmt(a.get('settle_time_ms'))} |")
    lines.append(f"| overshoot % | {fmt(a.get('overshoot_pct'))} |")
    lines.append(f"| peak velocity m/s | {fmt(a.get('peak_velocity_mps'))} |")
    pv = a.get("panel_variance", {})
    if pv:
        lines.append("")
        lines.append("Per-panel detail:\n")
        lines.append("| panel | rise_ms | settle_ms | over% | peak_v_mps |")
        lines.append("|---|---|---|---|---|")
        for k, v in pv.items():
            lines.append(
                f"| {k} | {fmt(v.get('rise_time_ms'))} | "
                f"{fmt(v.get('settle_time_ms'))} | "
                f"{fmt(v.get('overshoot_pct'))} | "
                f"{fmt(v.get('peak_vel_mps'))} |"
            )

    lines.append("")
    lines.append("## Pair coupling\n")
    lines.append("| condition | inter-panel 50%-cross delay (ms) |")
    lines.append("|---|---|")
    lines.append(
        f"| opposite pair (180 deg apart) | {fmt(c.get('opposite_pair_delay_ms'))} |"
    )
    lines.append(
        f"| adjacent pair (closest neighbour) | {fmt(c.get('adjacent_pair_delay_ms'))} |"
    )

    lines.append("")
    lines.append("## Friction baseline\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| coast-down decay tau (s) | {fmt(f.get('decay_tau_s'))} |")
    lines.append(
        f"| rolling-resistance coefficient | {fmt(f.get('rolling_resistance_coef'))} |"
    )
    lines.append(f"| incline slip-onset (deg) | {fmt(f.get('incline_onset_deg'))} |")
    lines.append(f"| static mu_s = tan(onset) | {fmt(f.get('static_mu_s'))} |")

    lines.append("")
    lines.append("## Verdict\n")
    have_actuator = a.get("rise_time_ms") is not None
    have_friction = (
        f.get("rolling_resistance_coef") is not None
        or f.get("static_mu_s") is not None
    )
    if have_actuator and have_friction:
        rise = a["rise_time_ms"]
        bw_hz = (0.35 / (rise * 1e-3)) if rise and rise > 0 else None
        mu = f.get("static_mu_s") or f.get("rolling_resistance_coef")
        lines.append(
            f"actuator bandwidth ~ {fmt(bw_hz)} Hz, friction mu ~ {fmt(mu)}, "
            "ready for W6."
        )
    elif have_actuator and not have_friction:
        rise = a["rise_time_ms"]
        bw_hz = (0.35 / (rise * 1e-3)) if rise and rise > 0 else None
        lines.append(
            f"actuator bandwidth ~ {fmt(bw_hz)} Hz; friction PENDING WSL run."
        )
    elif have_friction and not have_actuator:
        mu = f.get("static_mu_s") or f.get("rolling_resistance_coef")
        lines.append(
            f"friction mu ~ {fmt(mu)}; actuator PENDING WSL run."
        )
    else:
        lines.append(
            "actuator bandwidth ~ (pending), friction mu ~ (pending). "
            "WSL run not yet executed."
        )

    lines.append("")
    lines.append("## How to (re-)populate\n")
    lines.append(WSL_LAUNCH_BLOCK)

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[w4-substrate] wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
