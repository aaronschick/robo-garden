# Urchin v3 — Consolidated Feasibility Assessment (Agent 8)

Synthesis of Agents 1–7. Scope is strictly the **Dodec-12** topology
chosen by Agent 1 (`docs/design_review/topology.md` §Recommendation).

---

## 1. Top-line verdict

**Yes, with caveats.** Dodec-12 is buildable at the 0.34 m sim diameter
using twelve off-the-shelf or custom-BLDC linear actuators, an inner
rib-and-ring cage, TPU 3-fold accordion gaskets, a 6S LiPo + CAN-bus
electronics stack, and a gimbal-stabilized RPi-5 perception head — for a
**mid-case BoM of ~$3.2k and ~8–10 weeks to a first rolling trial**
(see §§2–3). The **single biggest uncertainty** is Agent 3's
velocity-clamp sim (`velocity_clamp_procedure.md`): every actuator
decision downstream of Class B/C collapses if the gait turns out to need
> 0.4 m/s per-panel velocity. Run that sim before sending any PO.

---

## 2. Unified cost table

Columns = actuator-class variants. Mid is the expected base case.
Numbers cited inline; see source docs for derivations.

| Subsystem | Low (Class C, geared-DC) | **Mid (Class B, ball-screw BLDC)** | High (Class A custom BLDC R&P) | Premium (Sensata VCMs) |
|---|---|---|---|---|
| Actuators (12×) | ~$600 (Actuonix-class, disqualified on velocity — placeholder for costing only) | **$1,900** (`actuator_shortlist.md` §Spec table, cand. C) | $2,800 (`actuator_shortlist.md` §Spec table, cand. A) | $12,000 (`actuator_shortlist.md` §Spec table, cand. B) |
| Motor drivers | $46 (12× DRV8874, `power_electronics.md` §6 Class C) | **$0** (ODrive S1 *is* the driver; already in actuator row) | $0 (ODrive S1 included) | $4,200 (12× Copley/A-M-C VCA drivers @ $350) |
| Per-panel node MCUs | $60 (12× STM32G0 @ $2.50 + passives, `power_electronics.md` §6) | **$0** (CAN-native ODrive — see §5 reconciliation) | $0 (CAN-native) | $60 (STM32G0 per node still needed for VCA supervision) |
| Main controller + co-proc | $38 (Teensy 4.1 $31.50 + ESP32-S3 $6, `power_electronics.md` §2) | **$38** | $38 | $38 |
| Power (battery + BMS + regs) | $82 (4S 2.2 Ah + Smart BMS $25 + D24V25F5 $15 + fuses/connectors, `power_electronics.md` §5 Class C) | **$95** (6S 1.5 Ah $42 + BMS $25 + buck $15 + fusing/connectors $13) | $112 (6S 2.2 Ah $55 + BMS $25 + buck $15 + connectors $17) | $112 |
| Sensors (core) | $96 (`sensor_package.md` §1 subtotal) | **$96** | $96 | $96 |
| Perception (RPi 5 + cam + gimbal) | $280 (gimbal+cam+RPi per `sensor_package.md` §3 recommendation; omit ToF halo) | **$500** (gimbal+cam $280 + ToF halo $140 + RPi $80) | $500 | $500 |
| 3D-print filament + fasteners | $60 (`print_bom.md` §1 cost totals) | **$60** | $60 | $60 |
| Structural hardware (AL6061 rings ×12, heatsets, screws) | $60 (12× $40 AL ring quote in `print_bom.md` §5 rounded down for shared stock; $20 heatsets/screws) | **$60** | $60 | $60 |
| **Subtotal** | $1,322 | **$2,749** | $3,666 | $17,126 |
| Contingency (15%) | $198 | **$412** | $550 | $2,569 |
| **TOTAL** | **~$1,520** | **~$3,160** | **~$4,220** | **~$19,700** |

Caveats on the Low column: Class C geared-DC hits only 32–51 mm/s (Actuonix L16 / PA-14P, `actuator_shortlist.md` §Disqualified outright). It is retained here only as a cost floor — if Agent 3's clamp passes at 0.1 m/s it becomes real; otherwise skip it.

**Recommended base case: Mid (~$3.2k all-in).** Adopt High if Agent 3 comes back >0.4 m/s or if the 400 mm/s ball-screw ceiling feels too tight (see §4 risk row).

---

## 3. Unified timeline

Baseline start date **2026-04-22**. Wall-clock weeks, dependencies noted.

| Wk | User | Claude | Build track | Depends on |
|----|------|--------|-------------|------------|
| 1 | Run Agent 3 velocity-clamp sim (4 caps ×10 s ≈ 5 min sim time, ~30 min incl. edits/teardown, per `velocity_clamp_procedure.md` §§2,5) | Draft CAD parametric for Dodec-12 cell + rib cage | **Place actuator PO** (4–6 wk lead on Class A; 1–2 wk on Class B per `actuator_shortlist.md` Spec table) | — |
| 2 | Review sim results; approve actuator class | Generate print G-code queues per `print_bom.md` §3 | **Print face plates (×12 @ 5.5 hr = 66 hr)** | Week-1 clamp result (to lock plate-boss geometry) |
| 3 | — | — | **Print ribs (30), brackets (12), tray, cradle ≈ 54 hr**; order AL6061 rings (SendCutSend ~5 days, `print_bom.md` §5) | Week-2 prints ok |
| 4 | — | — | **Print TPU gaskets (×12 @ 2 hr = 24 hr)** + bench-bond coupon test (`collision_audit.md` §5 item 5); assemble rib cage | Direct-drive conversion if TPU stringing (`print_bom.md` §4 item 1) |
| 5 | — | Firmware bring-up: Teensy CAN daisy-chain + IMU fusion loop on bench | **Electronics assembly**: BMS, buck, Teensy carrier, ESP32 telem link | Parts in hand |
| 6 | — | Policy retrain: re-sample SH-9 at N=12, launch light-tune (~0.5 M steps, `topology.md` §A Retraining) | **Actuators arrive** (Class B); benchtop single-cell actuation test | Weeks 1 PO + 5 electronics |
| 7 | — | Monitor retrain; render pre-hardware video | **Full-shell assembly**: panels + gaskets + actuators + equator ring | Weeks 4 + 6 |
| 8 | Smoke roll on carpet (tethered power first) | — | **First panel-actuation test under CAN control**; closed-loop position check | Week-7 assembly |
| 9 | First *rolling* trial (untethered, 30 min runtime target per `power_electronics.md` §5) | Sim-to-real gap analysis; write residual-gap reward if needed | — | Week-8 success |
| 10 | — | Retrain with randomized-contact domain randomization if sim-to-real gap visible | — | Week-9 data |

**First-roll milestone ETA: 2026-06-24** (Monday of Week 9, 9 weeks after 2026-04-22). If Class A (custom R&P) is chosen, slip by ~2 weeks for the mechanical iterations flagged in `actuator_shortlist.md` §A (~40 hr CAD + 2 prototype cycles). Earliest credible first-roll under Class A: **2026-07-08**.

---

## 4. Risk matrix

| Risk | L | I | Mitigation |
|------|---|---|------------|
| Agent 3 clamp sim says gait needs > 0.4 m/s | M | H | **Run it first.** If true, invalidates Class B (ball-screw caps at 400 mm/s with zero headroom, `actuator_shortlist.md` §C) and forces Class A (+$900) or topology fallback to Sphero-2. |
| TPU gasket fatigue in real rolling (no bench analog) | M | M | Bond-coupon test in Week 4 (`collision_audit.md` §5 item 5); silicone-sheet fallback template in `print_bom.md` §4 item 1 at $15. Cycle-life margin is 4 orders on paper (`collision_audit.md` §3) but wear at pinch-vertices is untested. |
| Actuator lead-time slip (Class A ODrive S1 stock; Sensata 6–10 wk) | M | H | Dual-source: order Class B (StepperOnline LK60, 1–2 wk) as parallel PO at $1.9k. 12× LK60 is the Week-1 safety PO if Claude flags schedule risk (`actuator_shortlist.md` §Recommendation). |
| CoM asymmetry from inner camera gimbal (not in Agent 1's topology) | M | M | See §5 reconciliation #1. Counter-balance with battery placement opposite the gimbal axis; re-measure CoM empirically in Week 7 before first roll. |
| Thermal runaway in sealed shell (RPi 5 ~7 W + 12 drivers + no forced air) | M | H | `sensor_package.md` §6 puts total electronics at ~10 W avg, 18 W peak in ~20 L sealed volume. Add 10 kΩ NTC on each DRV8874 (`power_electronics.md` §7) **plus** one on the RPi 5 CPU; firmware thermal throttle at 75 °C ambient, hard cut at 85 °C. Consider a small internal PC fan routing air across the AL6061 rings as heat spreaders if benchtop thermals exceed 70 °C. |
| Policy doesn't transfer from 42-panel to 12-panel even with SH reprojection | L | M | `topology.md` §A Retraining predicts light-tune (0.5–1 M steps); fallback is full retrain (~2–3 extra GPU-days). SH-9 basis is well-conditioned at N=12 (`topology.md` §Recommendation rationale #3). |
| E-stop unreachable mid-roll | M | H | Magnetic reed + exterior marking on the shell (`power_electronics.md` §7 E-stop) — works at any orientation. Augment with a BLE kill command from the phone app (`power_electronics.md` §8) as redundant software E-stop. Low-voltage firmware cutout at 3.2 V/cell is the third layer. |

---

## 5. Cross-agent reconciliations

1. **Gimbal + pendulum was not in Agent 1's Dodec-12 topology.** `sensor_package.md` §3 added a 2-axis BLDC gimbal carrying a ~1 kg counterweight/camera/RPi assembly to produce gravity-stable video. Agent 1's Candidate A did not reserve interior volume or a CoM slot for this — the pendulum-in-shell pattern was reserved for the *Sphero-2* fallback. **Impact:** CoM shifts off geometric center by up to ~40 mm (similar to the Sphero-2 physics in `topology.md` §B), which is also the rolling-authority direction for the contact-dipole gait. Net effect on rolling could be beneficial (extra CoM offset) or disruptive (fights the panel push direction). **Action:** add a CoM-measurement + offset-compensation step to the Week-7 assembly and feed a measured CoM offset into the sim as a fixed-shift before the Week-9 roll. If it fights the gait, demote perception to ToF-only (Option 1 in `sensor_package.md` §2, $140) for v1 and move the camera to v2.
2. **Agent 2's Class A pick uses native CAN via ODrive S1 — drops Agent 7's per-panel STM32G0 nodes.** `power_electronics.md` §6 budgeted a $2.50 STM32G0B1 + DRV8874 per panel ($60 for 12) under the assumption each actuator needed a local node. ODrive S1 speaks CAN-FD natively (`actuator_shortlist.md` §A), so for Class A and moteus/ODrive variants of Class B the per-panel node BoM is **$0**, not $60. Table in §2 reflects this. Retain STM32G0 per-panel nodes **only** for Class C (DRV8874) and Class D (Sensata VCM, which needs external driver supervision).
3. **Agent 5's boss-wall 3× SF depends on Agent 4's 2 mm retraction rule.** `print_bom.md` §2 reports SF ≈ 3× on a 3 mm PETG rib-to-plate bolt boss at 62 N impact. That assumes Agent 4's 2 mm plate-edge retraction (`collision_audit.md` §2) is honored — without it the panels bind and impose static pre-load that adds to the 62 N dynamic number, eroding SF toward 1.5×. **Rule:** no CAD iteration may thin the rib-boss wall below 3 mm **or** reduce the edge retraction below 2 mm without re-running the stress budget. If Nylon CF upgrade becomes necessary (`print_bom.md` §4 item 3), it applies to brackets/bosses only, not face plates (TPU bond is untested on nylon).
4. **Agent 3's velocity predictions are unvalidated.** Every pass/fail row in `velocity_clamp_procedure.md` §4 is a reasoned prediction, not a measurement. The entire Mid-column actuator choice depends on the 0.2–0.4 m/s row resolving as predicted. **Action:** this sim is the first-priority gating item in §6.

---

## 6. Recommended next actions

| # | Action | Owner | Duration |
|---|--------|-------|----------|
| 1 | Run Agent 3's velocity-clamp procedure at 1.5, 0.4, 0.2, 0.1 m/s caps; record Δx, jv_max per cap; revert `velocity_limit=1.5` when done | User | ~30 min incl. setup, 4 render runs, teardown |
| 2 | Based on #1 result, lock actuator class (Mid / High / fallback) and place POs on both primary *and* a Class-B hedge | User | 1 hr; 2–6 wk lead time after |
| 3 | Kick off CAD parametric for Dodec-12 cell, rib cage, and gimbal mount; honor 2 mm edge retraction + 3 mm rib-boss wall rules | Claude | 3–5 days |
| 4 | Start Week-2 print queue (face plates first) as soon as CAD is approved | User (babysit printer) | 6–10 wall-clock days |
| 5 | Coupon-test TPU-to-PETG bond (3 samples, pull to 10 N/cm) before committing to all 12 gaskets | User | 2 hr |
| 6 | SH-9 → N=12 policy retrain (light-tune, ~0.5 M steps) while hardware builds | Claude | 12–18 GPU-hr (one overnight WSL run) |
| 7 | Electronics bring-up on benchtop: BMS + buck + Teensy CAN + BNO085, verify 1 kHz loop and CAN arbitration at 10 simulated nodes | User + Claude | 2 days |
| 8 | First rolling trial on carpet, tethered power first, untethered second run; record IMU + video for sim-to-real gap analysis | User | half day |

---

## 7. Unknowns — things 8 agents couldn't answer

1. **Real-world contact-dipole rolling authority at N=12.** `topology.md` open-question #1 flags it; the velocity-clamp sim (action #1) only tells us whether the *oracle* gait tolerates the cap, not whether a *learned* policy re-projected through Y_12 does. An Agent 9 job would re-run the SH-9 contact-push oracle with N=12 *and* load the current Phase-1 checkpoint re-projected through Y_12 (per `topology.md` open-question #4) to de-risk the retrain assumption.
2. **TPU fatigue at pinch-vertices.** `collision_audit.md` §3 computes accordion-wall cycle life as >10⁷ but dismisses the 20 vertex-pinch patches with a hand-wave. A physical flex-fatigue rig (10-day cycle test at 0.5 Hz, 55 % strain) on a single vertex patch would resolve this — no simulation substitute exists for TPU wear behavior.
3. **CoM perturbation from perception gimbal vs rolling gait.** Reconciliation #1 notes the concern but the interaction is nonlinear — the pendulum acts as a low-pass CoM bias while panels push at mid-freq. Only a physical benchtop rock test (shell on a tilt rig, gimbal commanded through its range, measure net CoM shift) answers this. Flag for Week 7.
4. **Sim-to-real contact model.** MuJoCo soft contact vs. real TPU gasket + carpet has no measured correspondence for this geometry. Expected gap: 10–30 % forward-progress loss on first rolling trial, closed by domain-randomization retrain in Week 10. Unmodeled — nothing in the 8-agent sweep bounds it.
5. **Thermal steady-state in sealed shell.** §4 mitigation is plausible but untested; there's no CFD or thermal-mockup here. Resolve empirically: instrument the Week-5 benchtop electronics assembly with NTCs and run a 30 min stall-current soak before closing up the shell.
6. **E-stop magnetic-reed reliability at arbitrary roll orientations.** A reed switch is directional; the shell will present it in many attitudes. May need a 3-reed mutually-orthogonal arrangement to guarantee any-orientation trigger — not costed in §2.
