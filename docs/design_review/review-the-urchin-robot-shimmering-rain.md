# Urchin Robot — Real-World Build Feasibility Review

## Context

`urchin_v3` has cleared its Phase 1 sim smoke (approved 2026-04-21) and the curriculum plan is in place. Before more training time is sunk into a design that can't be built, we want a grounded feasibility review: can the current sim design be physically realized with a home 3D printer + off-the-shelf parts, roughly basketball-sized, for a reasonable budget? This plan captures what we already know from the codebase and breaks the remaining open questions into parallelizable research agents.

This plan file is the **consolidation target** for agent findings. It is not an implementation spec — no code changes come out of this.

---

## Current sim design (from `workspace/robots/urchin_v3/` and `scripts/build_urchin_v3_blend.py`)

| Parameter | Sim value | Real-world implication |
|---|---|---|
| Shape | Spherical shell + 42 radial prismatic panels | High-DoF soft rolling blob |
| Shell diameter | **0.34 m** (~13.4") | Already bigger than a basketball (0.24 m), half a doorway |
| Panels | 42 identical tiles, tetra-symmetric icosahedral layout | ~43 cm² per tile on a 24 cm sphere; ~85 cm² at current 34 cm |
| Panel travel | 0–60 mm (rest at 10 mm) | Stroke is ~18% of shell radius |
| Total mass | 2.52 kg (1.47 kg shell + 42 × 0.025 kg panels) | Scales ~r² with size if wall thickness constant |
| Actuator sim spec | **60 N stall, 1.5 m/s, position-controlled, k=800 N/m, ζ≈0.1** | The **big problem** — 1.5 m/s linear at 60 N doesn't exist off the shelf |
| DoF | 42 prismatic | 42 independent linear axes |
| Control basis | 9-D spherical-harmonic (l≤2) → 42-D panel targets | Actuators are slaves to a low-rank target — exploit this |
| Gait | Contact-dipole rolling (`compute_contactpush_oracle`) with gravity-aligned rear-push at ~1 Hz | Real panel duty cycle: peak ~400 mm/s, avg ~240 mm/s |

**Catalog gap (from `src/robo_garden/data/actuators/`):** only rotary actuators catalogued (Dynamixel XL/XM/XH, hobby PWM, ODrive/QDD/RMD BLDC). **No linear actuators catalogued.** TPU 95A is in `3d_printable.yaml` and is the obvious compliance material.

---

## Headline feasibility assessment

### Green (solved or easily solvable)
- **3D printing**: 42 identical panels → perfect for FDM. Whole bill-of-materials is a home-printable problem in PLA+TPU or PETG+TPU.
- **Size fit**: current 0.34 m is well under the ~0.76 m doorway constraint. Can stay at sim scale or shrink to basketball without hitting either rail.
- **IMU / control electronics**: BNO085 (fused IMU), ESP32-S3 or Teensy 4.1, SPI/I2C fan-out over CAN bus — standard hobby-robotics stack.

### Yellow (solvable but requires design choice)
- **Sensors**: rolling shell makes external perception awkward. Options exist (gyro-stabilized inner cage, 360° ToF array on hull, fisheye through a clear panel) — pick one.
- **Power**: 42 actuators pulling 60 N peak each → if all fired simultaneously, multi-kW peak. In practice the SH basis caps effective simultaneous active panels to ~10–15. A 6S 5 Ah LiPo is probably sized right but needs confirmation.

### Red (must be resolved before building)
- **42 linear actuators at 60 N / 1.5 m/s each**: no off-the-shelf part meets this. Voice coils can match velocity but not in a 42×packaging budget. Lead-screw micro-actuators (Actuonix L12-30N) are ~35× too slow. **The 1.5 m/s spec is almost certainly a sim-side safety margin** — rolling gait only needs ~0.4 m/s peak — but this must be verified by re-running a gait simulation with velocity clamped at realistic hardware limits before committing to hardware.
- **DoF reduction**: the SH-9 control basis means the policy only ever commands 9 independent modes. 42 physical actuators is likely overkill. A 12- or 20-panel mechanical design driven from a shared inner mechanism may give 90% of the behavior at 20% of the build cost. This is the single biggest design fork.

---

## User-approved constraints (2026-04-21)

- **DoF**: open to ≤20 panels. Reduced-DoF mechanical designs are in-scope and Phase 1 policy transfer is **not** a hard requirement — we can retrain.
- **Budget**: no hard ceiling. Optimize for build quality; still report costs for the record.
- **Size**: pick whichever diameter minimizes build risk, anywhere from basketball (0.24 m) to doorway-filling (~0.70 m).

These answers collapse the design space: **Agent 2 (DoF reduction) is now co-primary with Agent 1**, and size falls out of the winning mechanical topology instead of being fixed upfront.

---

## Multi-agent research breakout

Eight parallel research strands. Each is an Explore or WebSearch agent prompt that writes its findings back into the appropriate section of this plan file before we commit to hardware.

### Agent 1 — DoF-reduction mechanical design (CO-PRIMARY)
**Question**: design 2–3 candidate topologies with 8–20 physical actuators that reproduce contact-dipole rolling. Budget is open, so optimize for elegance and reliability over cost. Concepts to explore: (a) dodecahedral 12-panel variant (same prismatic-push principle, fewer cells), (b) icosahedral 20-panel variant, (c) Sphero/BB-8-style inner gimbal + internal pendulum shifting CoM (no panels at all — rolls via CoM displacement), (d) hybrid: 6–8 "big" panels around the equator + inner mass-shift, (e) cable-driven where 1 central BLDC drives N panels via spool + cam, losing independent control but massively simplifying the build.
**Deliverable**: 2–3 ranked topologies with sketches-as-text, actuator count, expected retraining effort, and which sim reward terms would need to change. **Recommend one**.

### Agent 2 — Actuator survey (CO-PRIMARY, scoped by Agent 1's output)
**Question**: for the winning topology from Agent 1, what actuators fit? Budget open, so voice-coil motors (Moticont, H2W, Akribis, LCS Motion) and custom BLDC-driven linear stages are in play — not just hobby parts. If the winning topology is gimbal-based, this reduces to "pick 2–3 brushless gimbal motors + a pendulum mass"; if it's still panel-based, it's 8–20 linear actuators with larger per-unit envelopes.
**Force/speed envelope depends on Agent 3's output** — do not commit until velocity clamp is known.
**Deliverable**: shortlist of 3 candidates with spec table, price, lead time, and control-interface notes (CAN / step-dir / analog).

### Agent 3 — Velocity-clamped gait re-simulation
**Question**: does `compute_contactpush_oracle` still roll if panel max velocity is clamped to 0.4 m/s, 0.2 m/s, 0.1 m/s? This decides how hard the actuator problem is.
**Approach**: read `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py` and `scripts/scripted_roll.py`; propose a sim config edit to rerun with each clamp; do **not** run it in this planning phase — just write the procedure.
**Deliverable**: reproducible command + expected metrics that would indicate pass/fail.

### Agent 4 — Shell geometry & collision audit
**Question**: at rest (panels at 10 mm) and fully inflated (60 mm), do neighboring panel tiles collide? The sim uses convex-hull wrappers with no self-collision — real hardware doesn't get that pass.
**Approach**: read the panel-tile mesh from `scripts/build_urchin_v3_blend.py` and the per-panel normals from `urdf_meta.json`. Compute tile edge separation as a function of panel extension.
**Deliverable**: minimum separation distance at extremes; if <5 mm, flag as a manufacturing problem and propose tile-edge chamfers or overlapping-scale design.

### Agent 5 — 3D-print BoM
**Question**: what's the full print BoM (mass, print time, filament cost, strength budget) for the 0.34 m prototype?
**Approach**: panel mesh size known (~186 vertices, ~9.6 KB OBJ each). Pick materials: TPU 95A for panel faces (compliance), PETG for panel backs (mount points for actuator), aluminum 6061 for base shell ribs if needed. Use `data/materials/3d_printable.yaml` densities.
**Deliverable**: full BoM table with per-part filament grams, total print hours on an Ender-3-class printer, material cost.

### Agent 6 — Sensor/perception package
**Question**: what does the robot need to see, and how does it see through a rolling shell?
**Core sensors (must-have)**: 9-axis IMU for orientation (BNO085), current sensing per motor channel (INA219).
**Perception options** (pick one or stack):
  - 8× VL53L5CX (8×8 ToF) distributed on shell → omnidirectional proximity field, de-rotated via IMU
  - Counter-rotating inner cage (stabilized by reaction wheel or passive heavy pendulum) with a single fisheye + Pi Zero 2 W
  - Transparent polycarbonate "window panels" (2 of the 42) with an inner-mounted camera
**Deliverable**: recommended sensor stack + wiring + data-rate estimate + which option unlocks what downstream tasks (obstacle avoid vs SLAM vs teleop).

### Agent 7 — Power + electronics architecture
**Question**: battery, motor drivers, main controller, comms. Assuming 42 actuators × Agent 1's winning actuator family.
**Deliverable**: block diagram — power rails, CAN/SPI bus topology, per-panel MCU vs central MCU trade-off, estimated continuous/peak current, battery Wh for 30 min runtime.

### Agent 8 — End-to-end cost + timeline sanity check
**Runs last**, after 1–7 report. Consolidates a total $ + hours estimate for a single prototype and flags the biggest schedule risks.

---

## Recommended agent launch order

1. **Agent 1 (DoF / topology)** and **Agent 3 (velocity-clamp gait sim)** in parallel — both are prerequisites for Agent 2.
2. **Agent 2 (actuator survey)** after Agent 1 picks a topology and Agent 3 gives the velocity envelope.
3. **Agent 4 (collision audit)** and **Agent 5 (print BoM)** after Agent 1, since both need the chosen topology.
4. **Agent 6 (sensors)** and **Agent 7 (power/electronics)** can run any time after Agent 2 — they depend on actuator count and current draw.
5. **Agent 8 (cost/timeline consolidation)** last.

---

## Agent findings (2026-04-22)

### Agent 3 — velocity-clamp procedure → `docs/design_review/velocity_clamp_procedure.md`
- **Clamp knob**: `ImplicitActuatorCfg.velocity_limit` at `workspace/robots/urchin_v3/urchin_v3/urchin_v3_cfg.py:110` (current `1.5` → clamp to `0.4`, `0.2`, `0.1`, revert after).
- **No new tooling**: existing `workspace/robots/urchin_v3/scripts/scripted_roll_video.py --mode contactpush --baseline-hz 0 --seconds 10 --episodes 1` logs `jv_max`/`jv_mean`/`speed`/`pos`. ~5 min total wall time on 3070 across all caps.
- **Pass criterion**: `|Δx| ≥ 1 m` in 10 s AND `jv_max` pinned at the cap.
- **Predictions**: 1.5 PASS (baseline), 0.4 likely PASS (~10–20% loss), 0.2 marginal (~50% progress), 0.1 likely FAIL (Weeble tilt, sub-30 cm).
- **Status**: procedure is ready for the user to execute. Agent 2 scoped on the 0.4 m/s prediction pending actual run.

### Agent 1 — topology candidates → `docs/design_review/topology.md`
- **Recommendation: Candidate A — Dodec-12** (12 pentagonal prismatic panels on a regular dodecahedron, 0.34 m shell, same 9-D SH control basis).
- **Why**: cheapest retrain (identical gait physics, light tune ~0.5–1 M steps); preserves the "shape-shifting rolling blob" identity; 12 is the smallest N where the SH-9 projection is well-conditioned; keeps Sphero-2 open as fallback via a shared inner rib cage.
- **Fallbacks ranked**: B. Sphero-2 (2 BLDCs + 1 kg pendulum, full retrain, cheapest/simplest build) — use if Agent 3 comes back saying 0.4 m/s is still too slow. C. Hybrid 16+2 (most expressive but highest complexity — not recommended).
- **Effort-limit reconciliation**: `urdf_meta.json` says `effort=15.0` per panel; `ImplicitActuatorCfg` overrides to `effort_limit=60.0` (retuned 2026-04-19 after 20 N froze the gait in a 3 cm Weeble tilt). **Agent 2 must size actuators to the 60 N working spec, not the 15 N URDF hint.**

---

## Wave 2 findings (2026-04-22)

All scoped to Dodec-12 at 0.34 m shell, 60 N working effort, 0.4 m/s velocity envelope (predicted, pending actual Agent 3 sim run).

### Agent 2 — actuators → `docs/design_review/actuator_shortlist.md`
- **Headline**: 60 N / 60 mm / 400 mm/s sits in an **off-the-shelf gap**. All hobby lead-screws (Actuonix L16, PQ12, PA-14) disqualified on velocity by 8–12×.
- **Primary pick**: Custom BLDC + rack-and-pinion (ODrive S1 + M8325s), ~$235/unit, **~$2.8k for 12**, real headroom on all 3 specs, ~40 hr mechanical design.
- **Backup**: StepperOnline LK60 ball-screw + 42 mm BLDC, ~$160/unit, ~$1.9k for 12 — hits 400 mm/s with zero margin.
- **Premium**: Sensata LAS28-53 VCM, ~$1k/unit, ~$12k for 12 — drop-in but 53 mm stroke (7 mm short) and 20 N continuous.

### Agent 4 — collision audit → `docs/design_review/collision_audit.md`
- **Verdict: PASS**, geometrically buildable with two mandatory design rules.
- Gap-per-mm coefficient **1.052** (vs 0.714 on icosa-42); gap opens 0 → 63 mm across 0–60 mm stroke.
- **Design rule 1**: 2 mm uniform edge retraction to guarantee positive clearance under ±0.6 mm FDM tolerance stack-up (costs 2.3% plate area).
- **Design rule 2**: 3-fold TPU 95A accordion gasket, 0.8 mm wall, 132 mm arc → peak wall strain 55% at s=60 mm (TPU breaks at ~500%).
- Dodec-12 actually uses **40% less TPU** and has **60 vertex pinch points vs 120** on the 42-panel baseline — the larger per-seam gap is offset by fewer seams.

### Agent 5 — print BoM → `docs/design_review/print_bom.md`
- **Filament**: $60 total (PETG 1.41 kg + TPU 95A 0.10 kg). Plastic is the cheap part of this build.
- **Print time**: ~144 printer-hours on a single Ender-3, 8–10 wall-clock days with failure overhead.
- **Strength**: Face plates have **13× safety factor** on rolling impact; rib-to-plate bolt bosses at 3 mm wall have 3× SF (**weak link** — avoid dropping below 2 mm, or use Nylon CF / AL6061 inserts there).
- **No part** exceeds the 235×235 mm bed (deliberate — 30 individual 124 mm ribs vs one monolithic cage).
- **TPU print risk**: stock Bowden Ender-3 may struggle with TPU 95A — flags direct-drive conversion (~$40) or silicone-sheet gasket fallback. Bambu / Prusa MK4 users drop to 2–3 days.

### Agent 6 — sensors → `docs/design_review/sensor_package.md`
- **Core stack**: BNO085 fused IMU + 12× INA219 (via TCA9548A mux) for per-motor current/torque + INA260 on main rail + 3× NTC thermistors → **$96**.
- **Perception primary**: Option 2 — gimbal-stabilized inner fisheye + RPi 5 (VLM-capable via cloud). Option 1 (8× VL53L5CX ToF array) kept as redundant proximity layer.
- **Rejected**: Option 3 (clear window panels) loses 2 of 12 actuators → breaks SH-9 basis. Option 4 (4 equator cameras) too many gasket penetrations for v1.
- **Compute**: RPi 5 (brain) + Teensy 4.1 (1 kHz motor controller, not the main MCU — see Agent 7).
- **Total sensor + compute + wiring**: **~$676**.
- **Conflict to reconcile**: Agent 6 needs a small inner camera gimbal; Agent 1's Dodec-12 didn't plan for one. It physically fits (~20 L interior), but requires explicit CoM balance against the battery pack to not disturb rolling.

### Agent 7 — power + electronics → `docs/design_review/power_electronics.md`
- **Battery**: 6S 2200 mAh LiPo (48.8 Wh, 320 g), ~60% headroom for the Class A (VCM) case and way more for Class B/C.
- **Bus**: CAN 2.0B @ 1 Mbps daisy-chain to 12 per-panel STM32G0 + DRV8874 driver nodes (4 wires through rib cage vs 26+ for I²C or 36 for PWM).
- **Main MCU**: Teensy 4.1 ($31) + ESP32-S3 co-processor ($6) for BLE/WiFi telemetry.
- **Electronics BoM**: **~$250** (excluding actuators).
- **Safety**: 3 A per-panel fuses, 15 A main, BMS with 3.0 V/cell cutoff, firmware action-scale taper at 3.3 V/cell, magnetic reed E-stop through shell wall (no gasket penetration), CAN-transceiver-disable watchdog → motors coast.
- **Open question for Agent 2**: confirm actuator driver interface (CAN-FD native like ODrive would replace the STM32G0 nodes → simpler).

---

## Agent 8 — consolidated assessment → `docs/design_review/consolidated_assessment.md`

**Verdict: YES, with caveats.** Dodec-12 is buildable at 0.34 m.

- **Mid base case (recommended): ~$3,160 all-in** with Class B ball-screw BLDC (StepperOnline LK60 + 42 mm BLDC, 1–2 week lead). 15% contingency included.
- **High case**: ~$4,220 with Class A custom BLDC rack-and-pinion (ODrive S1 + M8325s, 4–6 week lead, +~40 hr CAD, first-roll slips 2 weeks).
- **Premium (not recommended)**: ~$19,700 with Sensata VCMs.

**First-roll ETA: 2026-06-24** (9 weeks from 2026-04-22 baseline) under Mid case; **2026-07-08** under High case.

**Top risks**: (1) Agent 3's velocity-clamp sim **is the gating item** — if gait needs >0.4 m/s, Class B dies and we pay for Class A. Run it first. (2) Inner camera gimbal (Agent 6) shifts CoM ~40 mm — could help or fight the contact-dipole gait; demote to ToF-only for v1 if it disrupts rolling. (3) TPU vertex-pinch fatigue is paper-only — bench-test Week 4. (4) Sealed-shell thermals are untested.

**Three mandatory design rules locked in**: 2 mm edge retraction (Agent 4), 3 mm rib-boss minimum wall (Agent 5), 3-fold TPU accordion gaskets (Agent 4).

**Next action #1 (user, ~30 min)**: run Agent 3's velocity-clamp procedure at 1.5 / 0.4 / 0.2 / 0.1 m/s, record Δx and jv_max, revert `velocity_limit=1.5`. Everything downstream depends on this.

---

## Verification (how we know the research is done)

- All 8 agent sections have findings appended with dated entries (e.g. `### Agent 1 findings — 2026-04-21`).
- Agent 3's velocity-clamp result says **yes it still rolls** at ≤0.4 m/s, or we have an adapted gait that works within the new limit.
- Agent 1 has produced ≥1 actuator SKU that fits Agent 3's velocity envelope, Agent 4's geometric envelope, and the $ ceiling.
- Agent 8's cost/timeline estimate has landed in a single table at the bottom of this file.
- The user signs off on the consolidated recommendation before any hardware is ordered.
