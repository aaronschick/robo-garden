"""W5 post-processor: collate per-condition results into report.md + verdict.json.

Inputs (produced by `tests/test_substrate/test_tipping_decomposition.py`):
  workspace/_tasks_out/w5_tipping/<condition>.json
  workspace/_tasks_out/w5_tipping/<condition>.csv          (per-step trace; ignored here)
  workspace/_tasks_out/w5_tipping/summary.csv              (one row per condition)

Outputs:
  workspace/_tasks_out/w5_tipping/report.md
  workspace/_tasks_out/w5_tipping/verdict.json   (machine-readable decision rule)
  workspace/_tasks_out/w5_tipping/net_rotation_bar.png   (matplotlib if avail)

Decision rule (deep-research §"Experimental Plan", Table row 3):
  retract_only.net_forward_rotation_rad / full_triplet.net_forward_rotation_rad
    < 0.25  -> HALT (controller hypothesis suspect)
    >= 0.25 -> PROCEED to W6 (locomotion atlas)

Usage:
  uv run python workspace/scratch/w5_postprocess.py \\
      [--input-dir workspace/_tasks_out/w5_tipping]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CONDITION_ORDER = (
    "lean_only",
    "retract_only",
    "push_only",
    "lean_retract",
    "retract_push",
    "full_triplet",
)


def _load_conditions(input_dir: Path) -> dict:
    """Load per-condition JSONs. Returns {condition: payload-dict}.

    Missing conditions are reported but do not abort -- the report and
    verdict.json record what's present. The decision rule simply requires
    both `retract_only` and `full_triplet` to be present and finite.
    """
    out = {}
    for cond in CONDITION_ORDER:
        p = input_dir / f"{cond}.json"
        if p.exists():
            try:
                out[cond] = json.loads(p.read_text())
            except Exception as e:
                print(
                    f"[w5-postprocess] WARN: failed to parse {p}: {e}",
                    file=sys.stderr,
                )
    return out


def _maybe_render_bar_chart(condition_data: dict, out_path: Path) -> bool:
    """Render a bar chart of net_forward_rotation_rad. Returns True if PNG written.

    Tries matplotlib; on ImportError or any rendering error, returns False
    and the report.md falls back to ASCII bars.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    try:
        labels = []
        values = []
        for cond in CONDITION_ORDER:
            if cond in condition_data:
                labels.append(cond)
                values.append(
                    condition_data[cond]["metrics"]["net_forward_rotation_rad"]
                )
        if not labels:
            return False
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("net_forward_rotation_rad")
        ax.set_title("W5 tipping decomposition: net forward rotation per condition")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception as e:
        print(
            f"[w5-postprocess] WARN: matplotlib bar chart failed: {e}",
            file=sys.stderr,
        )
        return False


def _ascii_bar(value: float, scale: float, width: int = 30) -> str:
    """One-line ASCII bar for `value`. `scale` is the |max|; bar width is `width`."""
    if scale <= 0:
        return ""
    n = int(round(width * abs(value) / scale))
    n = max(0, min(width, n))
    bar = "#" * n + "." * (width - n)
    sign = "-" if value < 0 else "+"
    return f"{sign}{bar}"


def _format_table(condition_data: dict) -> str:
    """Markdown table with all four metrics per condition."""
    header = (
        "| condition | tip_onset_angle_deg | tip_margin_max | "
        "contact_set_change | net_forward_rotation_rad |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for cond in CONDITION_ORDER:
        if cond not in condition_data:
            lines.append(
                f"| {cond} | _missing_ | _missing_ | _missing_ | _missing_ |"
            )
            continue
        m = condition_data[cond]["metrics"]
        lines.append(
            f"| {cond} "
            f"| {m['tip_onset_angle_deg']:.3f} "
            f"| {m['tip_margin_max']:.5f} "
            f"| {m['contact_set_change']} "
            f"| {m['net_forward_rotation_rad']:.5f} |"
        )
    return "\n".join(lines)


def _format_ascii_bars(condition_data: dict) -> str:
    """ASCII fallback chart for net_forward_rotation_rad."""
    values = {
        cond: condition_data[cond]["metrics"]["net_forward_rotation_rad"]
        for cond in CONDITION_ORDER
        if cond in condition_data
    }
    if not values:
        return "_no data_"
    scale = max((abs(v) for v in values.values()), default=1e-9) or 1e-9
    rows = []
    width = max(len(c) for c in values)
    for cond in CONDITION_ORDER:
        if cond not in values:
            continue
        v = values[cond]
        bar = _ascii_bar(v, scale)
        rows.append(f"{cond.ljust(width)}  {v:+8.5f}  {bar}")
    return "```\n" + "\n".join(rows) + "\n```"


def _evaluate_decision_rule(condition_data: dict) -> dict:
    """Compute retract_only / full_triplet ratio and verdict.

    Returns a dict with keys:
      ratio                -- float or None (None if unavailable)
      threshold            -- 0.25
      decision             -- "PROCEED" | "HALT" | "INSUFFICIENT_DATA"
      retract_only_net_rot -- float or None
      full_triplet_net_rot -- float or None
      missing_conditions   -- list of names absent from condition_data
    """
    missing = [c for c in CONDITION_ORDER if c not in condition_data]
    threshold = 0.25
    out = {
        "threshold": threshold,
        "missing_conditions": missing,
        "ratio": None,
        "retract_only_net_rot": None,
        "full_triplet_net_rot": None,
        "decision": "INSUFFICIENT_DATA",
        "rationale": "",
    }
    if "retract_only" not in condition_data or "full_triplet" not in condition_data:
        out["rationale"] = (
            "retract_only and/or full_triplet conditions missing; "
            "decision rule cannot be evaluated."
        )
        return out

    retract_net = condition_data["retract_only"]["metrics"][
        "net_forward_rotation_rad"
    ]
    full_net = condition_data["full_triplet"]["metrics"][
        "net_forward_rotation_rad"
    ]
    out["retract_only_net_rot"] = float(retract_net)
    out["full_triplet_net_rot"] = float(full_net)

    if abs(full_net) < 1e-9:
        out["rationale"] = (
            "full_triplet net_forward_rotation_rad ~= 0 (denominator is "
            "zero). The controller does not produce net forward rotation "
            "even at full amplitude -- HALT and inspect."
        )
        out["decision"] = "HALT"
        return out

    ratio = float(retract_net) / float(full_net)
    out["ratio"] = ratio
    if ratio < threshold:
        out["decision"] = "HALT"
        out["rationale"] = (
            f"retract_only contributes {ratio:.3f} of full_triplet net "
            f"forward rotation, below the {threshold:.2f} threshold. "
            "Controller hypothesis suspect; redesign before W6."
        )
    else:
        out["decision"] = "PROCEED"
        out["rationale"] = (
            f"retract_only contributes {ratio:.3f} of full_triplet net "
            f"forward rotation, at or above the {threshold:.2f} threshold. "
            "Proceed to W6 (locomotion atlas)."
        )
    return out


def _format_report(
    condition_data: dict,
    verdict: dict,
    bar_png_path: Path,
    has_png: bool,
) -> str:
    """Build the report.md text."""
    lines = [
        "# W5 — Tipping Decomposition Report",
        "",
        "**Six-condition quasi-static decomposition of the explicit "
        "rolling-engine triplet (lean / retract / push).**",
        "",
        "Source test: `tests/test_substrate/test_tipping_decomposition.py`",
        "",
        "Driver: `scripts/run_urchin_v3_tipping_decomp.sh`",
        "",
        "## Per-condition metrics",
        "",
        _format_table(condition_data),
        "",
        "## Net forward rotation per condition",
        "",
    ]
    if has_png:
        lines += [
            f"![net forward rotation bar chart]({bar_png_path.name})",
            "",
        ]
    else:
        lines += [
            _format_ascii_bars(condition_data),
            "",
        ]
    decision = verdict["decision"]
    rationale = verdict["rationale"]
    if decision == "HALT":
        verdict_md = f"**HALT — controller hypothesis suspect.** {rationale}"
    elif decision == "PROCEED":
        verdict_md = f"**PROCEED to W6.** {rationale}"
    else:
        verdict_md = f"**INSUFFICIENT DATA.** {rationale}"

    lines += [
        "## Decision rule",
        "",
        "Per `docs/Urchin-deep-research-report.md` §\"Experimental Plan\", "
        "Table row 3:",
        "",
        "> retract_only.net_forward_rotation_rad / "
        "full_triplet.net_forward_rotation_rad >= 0.25",
        "",
        f"- retract_only net_forward_rotation_rad: "
        f"{verdict['retract_only_net_rot']!r}",
        f"- full_triplet net_forward_rotation_rad: "
        f"{verdict['full_triplet_net_rot']!r}",
        f"- ratio: {verdict['ratio']!r}",
        f"- threshold: {verdict['threshold']!r}",
        "",
        verdict_md,
        "",
    ]
    if verdict.get("missing_conditions"):
        lines += [
            "## Missing conditions",
            "",
            "The following conditions did not produce a JSON artifact and "
            "are excluded from the table:",
            "",
            "".join(f"- `{c}`\n" for c in verdict["missing_conditions"]),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("workspace/_tasks_out/w5_tipping"),
        help="Directory containing per-condition JSONs and summary.csv.",
    )
    args = ap.parse_args(argv)

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        print(
            f"[w5-postprocess] ERROR: input dir does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 2

    condition_data = _load_conditions(input_dir)
    verdict = _evaluate_decision_rule(condition_data)

    bar_png = input_dir / "net_rotation_bar.png"
    has_png = _maybe_render_bar_chart(condition_data, bar_png)

    report_md = _format_report(condition_data, verdict, bar_png, has_png)
    # Force UTF-8 so em-dashes and section symbols round-trip cleanly under
    # Windows' default cp1252 locale (the post-processor may run from
    # either Linux or Windows depending on which side collates).
    (input_dir / "report.md").write_text(report_md, encoding="utf-8")
    (input_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8",
    )

    print(f"[w5-postprocess] wrote {input_dir / 'report.md'}")
    print(f"[w5-postprocess] wrote {input_dir / 'verdict.json'}")
    if has_png:
        print(f"[w5-postprocess] wrote {bar_png}")
    print(f"[w5-postprocess] decision: {verdict['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
