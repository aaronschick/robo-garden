# urchin_lite_v1 — Consolidated v1 Review (Agent C1)

Date: 2026-04-22

## Context

`urchin_lite_v1` is a simplified post-Dodec-12 iteration of the urchin
family — rolling-only, 8 linear prismatic actuators on an OCTA-8
(octahedron-with-8-triangular-panels) topology, dynamic shape-shifting
through face-normal panel extension. This doc is the smoke-pass plus
open-issues summary for the Wave B deliverables (B1 robot gen, B2 BoM,
B3 validator), prior to hand-off to training.

## Smoke results

| # | Check | Pass/Fail | Notes |
|---|---|---|---|
| 1 | URDF parse & MuJoCo load (`MjModel.from_xml_path`) | PARTIAL | Parses without error. `nbody=9, njnt=8, nq=nv=8`, all 8 panel prismatic joints present. **`nu=0` — no `<transmission>` elements in URDF, so MuJoCo compiled 0 actuators.** Also, the `base_shell` link fuses into `world` (no URDF freejoint convention), so the shell is currently welded to origin — rolling will not happen in raw-URDF load. Both issues need an MJCF wrapper (or urdf2mjcf pass) before training. |
| 2 | `mujoco.mj_step` single step | PASS | No NaN/Inf in `qpos`/`qvel`. Small settling motion from gravity (panels sag on prismatic axes, ~1e-4 m displacement) — consistent with unactuated joints relaxing. |
| 3 | `core/simulation.py::simulate` (0.5 s) | PASS | 250 steps, `diverged=False`, `stable=True`, `max_velocity=0.64 m/s`. Runs cleanly through the existing sim entry point; no changes needed to `core/simulation.py`. |
| 4 | Catalog validator (B3 extensions, 28 entries across 4 YAMLs) | PASS | All 28 entries pass `_validate_actuator_specs`. `sensata_las28_53_vca` (B1's primary) and `odrive_s1_rack_and_pinion_ref` (B2's promoted primary) both validate cleanly with B3's new linear-actuator rules. |
| 5 | Preview entry point | PASS | `src/robo_garden/building/preview.py::launch_viewer` exists (GUI only); an offscreen `mujoco.Renderer` frame was rendered to `C:\Users\aaron\Documents\repositories\robo-garden\workspace\_tasks_out\urchin_lite_v1_preview.png` (5.9 KB, 640×480 — shell sphere + 8 panel boxes visible). |
| 6 | Full test suite (`pytest tests/`) | PASS | **69 passed**, 2 unrelated SB3-checkpoint-path warnings. Matches B3's report. |

## What works

- URDF is geometrically/inertially valid and loads through MuJoCo without XML errors.
- Existing `core/simulation.py` pipeline handles the robot with zero code changes.
- All 28 actuator catalog entries (linear + rotary) pass the B3-extended validator.
- The 69-test suite is green after B3's removal of A2's bridge patch and addition of 40 new linear-actuator tests.
- SH-9 basis correction from B1 is sound: the rank-8 `(Z/2)³` character basis `{1, n_x, n_y, n_z, n_x·n_y, n_y·n_z, n_x·n_z, n_x·n_y·n_z}` is the correct octahedral-symmetric decomposition and is recorded in `urdf_meta.json`.

## What's still open

- **`nu=0` in raw URDF load (BLOCKER for training).** The URDF has no `<transmission>` elements. MuJoCo loads the prismatic joints but creates zero actuators, so `data.ctrl` has length 0 and no policy can drive the panels. Fix: add 8 `<transmission>` blocks to the URDF (`SimpleTransmission` per joint → motor) or ship an MJCF wrapper with `<motor joint="panel_N" ctrlrange="0 1" gear="..."/>` × 8. Training cannot start until this is resolved.
- **`base_shell` welded to world (BLOCKER for rolling).** URDF's first link becomes MuJoCo's world-weld by convention, so the shell cannot translate/rotate. A free-floating base needs either an MJCF wrapper that adds `<freejoint/>` to `base_shell`, or a dummy root link with a 6-DoF joint in the URDF. Check whether B1's `scripted_roll.py` / `train.py` already wrap the URDF with this — if not, add it.
- **B1/B2 stroke disagreement (ORDERING BLOCKER).** B1 built the URDF against Sensata LAS28-53 at 25 mm stroke (`limit upper=0.025`, `urdf_meta.panel_stroke_mm=25.0`). B2's BoM promoted ODrive S1 + M8325s rack-and-pinion (60 mm stroke) to **primary** because OCTA-8's sim target is 0–60 mm panel travel, demoting Sensata to backup. These are inconsistent: B1's geometry is sized for the *backup*. **Resolution path**: (a) run `build_urdf.py` at both 25 mm and 60 mm stroke, scripted-roll each, and pick the one where the shell actually makes floor contact during roll; or (b) re-derive the stroke requirement from OCTA-8 geometry (inscribed-radius vs circumscribed-radius ratio at the panel axis) to settle whether 25 mm is physically sufficient. Dollar delta: primary = ODrive $1,880 for 8 cells; Sensata backup = **not retail-priced at qty 8** (Sensata direct quote required, dev kit is $1,200–$1,500 each), so Sensata could easily exceed $10k. The cheaper build ($3,086 ODrive) is the one whose stroke B1's URDF does *not* match. User must adjudicate before any hardware is ordered.
- **B2 open flags that matter pre-first-roll.** (1) `sensata_las28_53_vca` `price_usd` and `weight_g` are `null` (flag #3, #4). (2) `stepperonline_lk60_40dl10s3_110` `price_usd` and `weight_g` are `null`, WebFetch 403 (flag #1, #2). (3) BMS backup SKU unsourced (flag #6). (4) 40 hr of mechanical CAD per ODrive cell is unbudgeted in schedule.
- **Warm-start caveat from SH basis correction.** Only 6 of 9 columns transfer cleanly from urchin_v3's SH basis to urchin_lite_v1's rank-8 character basis (`1, n_x, n_y, n_z` transfer; the three `n_i·n_j` cross-terms map to v3 `l=2` mixtures and need re-fitting; `n_x·n_y·n_z` is new l=3 and has no v3 analog). First training runs should expect some re-exploration on those 3 channels.
- **MJCF counterpart missing.** Only URDF is shipped. MuJoCo's URDF loader works, but issues #1 and #2 above (no transmissions, no freejoint) mean a proper MJCF wrapper is effectively mandatory before training. File this as a B1 follow-up.
- **Actuator delay τ=20 ms is a guess** (not measured on the chosen actuator). Relevant for Phase-1 sim2real transfer.
- **A3 gap #6 (TPU-panel contact model is rigid-convex)** carries over — the real compliant shell will deform on contact; sim uses rigid boxes today.

## Totals

- **BoM primary cost** (per B2, `build_order.md`): **$3,085.80** for one OCTA-8 unit (8× ODrive S1 + M8325s + 1 spare, Pi 5, BNO085, 4S LiPo, PETG/TPU, fasteners, dev PSU).
- **ETA to first physical roll**: **4–8 weeks** from part-order, realistic window. Critical path: ODrive S1 + M8325s kit stock (2–3 wk) + ~40 hr mechanical CAD for first rack-and-pinion cell + print + assemble + bringup. If Backup A (StepperOnline LK60) or Backup B (Sensata VCA) is chosen, push to 6–10 weeks.
- **ETA to first sim-trained policy**: B1 claims 0.5–1.5 M steps to warm-start from urchin_v3 Phase-1 checkpoint. At ~13k steps/sec effective on the RTX 3070 WSL GPU path (100k in ~8 min, per `urchin_v3_smoke_wp4_100k` reference), that's **0.5–2 hr** of GPU wall time once the two blockers (`nu=0`, `base_shell` weld) are fixed. Add a few hours for reward-shape iteration.

## Top 3 open risks

1. **Blockers #1 and #2 (`nu=0`, welded base).** The URDF as-shipped cannot be trained or rolled; these must be fixed before any other work lands. Likely a half-day of B1 rework (add 8 `<transmission>` + freejoint via MJCF wrapper), but until it's done the whole v1 thread is stalled.
2. **B1/B2 stroke disagreement (25 mm vs 60 mm).** Blocks hardware ordering and potentially invalidates either (a) B1's URDF (if 60 mm is required) or (b) B2's promotion of ODrive over Sensata (if 25 mm is sufficient). Cheap to resolve in sim (rebuild at both strokes, compare scripted-roll contact), but must be done before any $3k BoM commits.
3. **Actuator-delay τ=20 ms is a guess, not measured.** First-roll policy may twitch / oscillate if the real actuator delay differs; this is the classic sim2real failure mode for fast shape-shifting robots. Measure τ on the chosen actuator before or during first physical bringup.

---
Report file: `C:\Users\aaron\Documents\repositories\robo-garden\docs\design_review_v1\consolidated_v1.md`
