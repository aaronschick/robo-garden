# Urchin v3 — Linear Actuator Shortlist (Agent 2)

## Context

Scoped to **Candidate A (Dodec-12)** from Agent 1: 12 linear actuators, one per
pentagonal face of a 0.34 m shell. Per-unit requirements driven by the retuned
sim (`ImplicitActuatorCfg.effort_limit = 60.0`, panel travel 0–60 mm, rest
10 mm) and Agent 3's predicted 0.4 m/s velocity envelope:

- Stall ≥ 60 N, stroke ≥ 60 mm, peak velocity ≥ 0.4 m/s (= 400 mm/s)
- Position-controlled, with feedback
- Envelope ideally ≤ 70 × 30 × 30 mm, acceptable up to 100 × 40 × 40 mm
- CAN or step/dir preferred, analog PWM tolerable at N = 12
- 12 units, budget open but cost tracked

**Headline reality check.** The 400 mm/s + 60 N + 60 mm stroke combination
sits in a gap that off-the-shelf hobby and light-industrial catalogs do not
fill cleanly:

- Actuonix **L16-50-35-12** (the fastest 50 mm hobby lead-screw) tops out at
  **32 mm/s at 50 N** — ~12× too slow. The whole L-series is disqualified
  on velocity. ([RobotShop L16-50-35](https://www.robotshop.com/products/actuonix-l16-linear-actuator-351-50mm-12v-w-potentiometer-feedback), [Actuonix L16 datasheet](https://www.actuonix.com/assets/images/datasheets/ActuonixL16datasheet.pdf))
- Moticont **SDLM / GVCM** voice coils can hit 400 mm/s easily but the 60 N
  continuous-force models need housings ≥ 95 mm long for only ~25 mm stroke
  (e.g. GVCM-051 series, 28 N continuous at 19 mm stroke). Getting to 60 mm
  useful stroke pushes housings past 120 mm. ([Moticont GVCM-051-051-01](https://www.moticont.com/GVCM-051-051-01.htm))
- Nanotec **LBA60** BLDC-lead-screw hits 292 mm/s at up to 1500 N — but only
  55 mm stroke and 60 mm flange diameter × ~130 mm housing. Close, not quite.
  ([Nanotec LBA60](https://www.nanotec.com/us/en/products/11077-lba60-bldc-linear-actuator-60-mm))

So the shortlist is:
**(1)** a **custom BLDC + belt-driven rack** — the realistic primary;
**(2)** a **commercial voice-coil stage** — premium backup with fastest
dynamics; **(3)** a **ball-screw BLDC cartridge** — proven, slightly
over-envelope, lowest-risk electronics.

---

## Spec table

| Candidate | Family | Stall F | Max v | Stroke | Envelope (L×W×H) | Control | Feedback | Unit $ | 12× $ | Lead time |
|---|---|---|---|---|---|---|---|---|---|---|
| **A. Custom: ODrive S1 + M8325s gimbal + 3 mm-pitch belt + 4 mm pinion rack** | BLDC + rack-and-pinion | ~90 N (at 20:1 effective reduction via 4 mm pinion) | ~650 mm/s (7000 rpm × π × 4 mm / 60) | 60 mm (rack length) | ~75 × 40 × 35 mm (motor 40 mm Ø × 25 mm long + rack + housing) | CAN-FD via ODrive S1 | Integrated magnetic encoder on motor (16384 CPR) | ~$210 (motor+driver) + ~$25 (rack/bearings/print) = **$235** | **~$2,800** | 2–3 weeks (ODrive stock dependent) |
| **B. Sensata BEI Kimco LAS28-53 voice-coil actuator** | VCA + integrated Hall sensor | 60 N peak / 20 N continuous | 2+ m/s (field-limited; driver-dependent) | 53 mm | Ø 28 × ~95 mm (cylindrical) | Analog current command (VCA driver, e.g. Copley Xenus Plus or A-M-C AZBE) | Integrated Hall-effect position sensor | ~$650 (motor) + ~$350 (driver) = **$1,000** | **~$12,000** | 6–10 weeks (Sensata is lead-time-y) |
| **C. StepperOnline / Misumi BLDC + compact 4 mm-lead ball-screw cartridge** | BLDC + ball-screw | 200+ N | ~400 mm/s (6000 rpm × 4 mm / 60) | 60 mm | ~90 × 40 × 40 mm (~NEMA-17-class motor + screw + nut) | Step/dir or CAN (with BLDC driver e.g. MKS SERVO42D or ODrive S1) | Integrated incremental encoder on motor | ~$160 | **~$1,900** | 1–2 weeks |

All figures are for **one unit**; multiply column "12× $" for the dodec
BoM. Prices exclude wiring, PSU, central MCU (those are Agent 7's scope).

---

## Candidate details

### A. Custom BLDC + rack-and-pinion (PRIMARY)

A small brushless gimbal motor (e.g. [iFlight GBM4108H-120T](https://shop.iflight.com/) or the **ODrive M8325s** bundled with [ODrive S1](https://shop.odriverobotics.com/products/s1-and-m8325s-start-kit)) driving a 3D-printed POM/nylon rack through a 4 mm-pitch steel pinion. At 20 mm/rev linear, the ODrive-spec'd 7000 rpm no-load
translates to ~650 mm/s — comfortable headroom over the 400 mm/s target —
and the S1's peak phase current (80 A) easily delivers 90 N peak pushing
force through a ~0.34 N·m motor with a 4 mm pinion. Control is CAN-FD
from a single central MCU (STM32F4-class) to twelve ODrive S1's, which
is exactly the topology the ODrive community has shipped on quadrupeds
(e.g. MIT Mini-Cheetah derivatives). The catch is **mechanical design
time** — each cell needs a custom printed housing, rack guide bearings,
and a return spring (to keep the pinion pre-loaded against the rack).
**Compromise:** ~40 hours of CAD + 2 prototype iterations before the
first unit works. **Representative SKU:** ODrive S1 + M8325s kit, ~$200
retail. ([ODrive S1 + M8325s start kit](https://shop.odriverobotics.com/products/s1-and-m8325s-start-kit))

### B. Sensata/BEI Kimco LAS28-53 voice-coil actuator (BACKUP — premium)

Cylindrical VCA, 28 mm Ø × ~95 mm long, 53 mm stroke, 60 N peak with a
built-in Hall-effect position sensor (model `LAS28-53-000A-P01-12E`).
Velocity is essentially unlimited for our duty cycle (VCAs can hit
>2 m/s in open loop; limited only by the driver's current slew). Fits
the 100 × 40 × 40 mm acceptable envelope. **Compromises:** (1) 53 mm
stroke is 7 mm short of spec — workable if we retune `panel travel` to
0–53 mm in the sim, losing ~12% of CoM excursion; (2) **continuous**
force is only 20 N, so we must confirm the 60 N peak is a transient
demand, not steady-state (Agent 3's data suggests yes — `jv_max` pulses,
doesn't sit); (3) per-unit cost with a Copley/A-M-C current-mode driver
is ~$1,000, so 12× is $12k — a real investment. **Representative SKU:**
[Sensata LAS28-53-000A-P01-12E](https://www.sensata.com/products/motors-actuators/cylindrical-housed-linear-vca-integrated-sensor-las28-53-000a-p01-12e).
Use this if the custom build schedule slips and we need a
drop-in-with-catalog-support solution.

### C. StepperOnline/Misumi BLDC + ball-screw cartridge (BACKUP — conservative)

A 42 mm BLDC (e.g. [StepperOnline 42BLF series](https://www.omc-stepperonline.com/), 60 W, 4000–6000 rpm) coupled directly to a 60 mm-stroke 4 mm-lead ball-screw stage (Misumi LX-series or StepperOnline LK60). At 6000 rpm × 4 mm/rev ÷ 60 = **400 mm/s** — just barely meeting Agent 3's floor, with zero headroom. Stall force is 200+ N so effort is never the issue. **Compromises:** (1) envelope is ~90 mm long and 40 mm wide — it fits the *acceptable* envelope but blocks some interior volume for the Agent 6 sensor package; (2) the 400 mm/s ceiling leaves no margin if Agent 3 comes back saying "actually the transient peak is 0.5 m/s" — we'd be clamping the gait, which was the exact problem this shortlist is trying to avoid; (3) ball-screws are less backdrivable than VCAs or belts (mostly fine here since sim has passive spring stiffness, but worth noting). **Representative SKU:** StepperOnline LK60-40DL10S3 stage (~$160) + matched 42 mm BLDC. Use this if custom mechanical design is off the table and we need to order, build, and train in ≤ 4 weeks. ([StepperOnline LK60 stage](https://www.omc-stepperonline.com/lk60-series-ball-screw-driven-max-horizontal-vertical-payload-20kg-6kg-stroke-510mm-for-servo-motor-lk60-40dl10s3-510))

---

## Recommendation

**Primary: Candidate A (custom BLDC + rack-and-pinion).** It's the only
option that hits all three hard specs (60 N, 60 mm stroke, 400 mm/s)
with actual headroom, at roughly $2,800 for the twelve-cell drivetrain —
a sweet spot between the $1,900 off-the-shelf-but-marginal ball-screw
and the $12,000 premium voice-coil build. It also reuses the ODrive
ecosystem that's already in the actuator catalog (`src/robo_garden/data/actuators/bldc.yaml`), so Agent 7's
motor-driver selection collapses to "12× ODrive S1 on a single CAN-FD
bus," which is a well-understood topology. The one real cost is ~40
hours of mechanical design before the first cell is printable.

**Backup: Candidate C (BLDC + ball-screw cartridge).** Order as parallel
risk mitigation. If Candidate A's mechanical iteration blows past 4
weeks, we can fall back to twelve catalog stages, eat a 12 % stroke-axis
compromise (or downsize the shell to 0.30 m to keep 60 mm proportional),
and still be training within a month.

**Voice-coil (Candidate B) is deprioritized** — it's the "cost no object"
answer, but the 12× price tag isn't justified when the custom build
meets spec at a quarter the price. Keep it on file in case we ever want
a sub-ms-response version for high-frequency shape-shifting research
(Agent 1's Candidate C hybrid).

**Disqualified outright:**
- **Actuonix L16 / PQ12 / P16 lead-screws** — max 32 mm/s, 12× too slow.
- **Progressive Automations PA-14P** — 0.59–2 in/s = 15–51 mm/s, also 8×
  too slow.
- **Solenoids** — bang-bang, no native position control. Closed-loop
  conversion would require a full custom driver + position feedback
  per unit, at which point Candidate A is strictly better.
- **Pneumatic cylinders** — compressor + regulator + 12 proportional
  valves add >5 kg of stationary overhead plus a tether, which breaks
  the rolling-blob concept.

---

## Open questions for Agent 7 (power + electronics)

1. **Peak current draw per cell.** Candidate A's M8325s pulls up to 80 A
   phase current at peak torque. The **SH-9 control basis caps
   simultaneously-active cells to ~10** per the plan file's Yellow-risk
   note, but we need a worst-case estimate: can a 6S 5 Ah LiPo sustain
   ~600 A momentary bus current (12 × 50 A simultaneous), or do we need
   a bus-side capacitor bank?
2. **Driver selection and CAN-FD topology.** ODrive S1 is ~$130/unit at
   12× — is there a cheaper BLDC-with-encoder driver (MKS SERVO42D,
   SimpleFOC-based) that still supports CAN-FD and closed-loop position
   control at 1 kHz? This is a ~$500 swing on total BoM.
3. **Continuous vs peak force duty cycle.** Agent 3's telemetry says
   `jv_mean ≈ 0.04–0.10 m/s` and `jv_max` pulses to ~0.4 m/s. If someone
   pulls the matching `effort_mean` vs `effort_max` numbers from a
   Phase 1 rollout, we can right-size continuous-current ratings (and
   confirm whether Candidate B's 20 N continuous is actually a
   blocker).
4. **Feedback loop placement.** All three candidates have feedback at
   the motor shaft, not at the panel. On the custom rack design, is
   shaft-space position enough, or do we need a secondary linear
   encoder at the rack (e.g. AS5600 on a magnet strip) for panel-space
   control authority? This is a ~$15/unit decision.

Sources:
- [ODrive S1 + M8325s start kit](https://shop.odriverobotics.com/products/s1-and-m8325s-start-kit)
- [Sensata LAS28-53 VCA](https://www.sensata.com/products/motors-actuators/cylindrical-housed-linear-vca-integrated-sensor-las28-53-000a-p01-12e)
- [StepperOnline LK60 ball-screw stage](https://www.omc-stepperonline.com/lk60-series-ball-screw-driven-max-horizontal-vertical-payload-20kg-6kg-stroke-510mm-for-servo-motor-lk60-40dl10s3-510)
- [Nanotec LBA60 BLDC linear actuator](https://www.nanotec.com/us/en/products/11077-lba60-bldc-linear-actuator-60-mm)
- [Moticont GVCM-051 voice coil](https://www.moticont.com/GVCM-051-051-01.htm)
- [Actuonix L16 datasheet (disqualifying speed data)](https://www.actuonix.com/assets/images/datasheets/ActuonixL16datasheet.pdf)
- [Actuonix L16-50-35-12-P specs (RobotShop)](https://www.robotshop.com/products/actuonix-l16-linear-actuator-351-50mm-12v-w-potentiometer-feedback)
