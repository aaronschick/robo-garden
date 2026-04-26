# Urchin v3 — Sensor & Perception Package (Agent 6)

## Context

This doc is Agent 6 of the feasibility review driven by
`C:\Users\aaron\.claude\plans\review-the-urchin-robot-shimmering-rain.md`.
Scoped to **Candidate A (Dodec-12)** — 12 pentagonal PETG face plates on a
0.34 m shell, each riding a 60 mm prismatic actuator, total mass ~2.5 kg,
internal volume ~20 L. The shell *rolls* in the body frame, so any
outward-looking sensor rotates continuously relative to the world; every
external measurement must either (a) be de-rotated against the IMU in real
time, or (b) be produced by a gravity-stabilized inner stage. The sensor
package must deliver proprioception for control and enough external
perception to support obstacle avoidance and a near-term SLAM or
vision-language-model (VLM) loop.

## 1. Core sensor table (must-haves)

| Sensor | Purpose | Interface | Rate | Unit cost | Qty | Notes |
|---|---|---|---|---|---|---|
| **Bosch BNO085** | Body orientation quaternion + gyro + accel (IMU w/ onboard fusion) | I2C or UART (SH-2) | 400 Hz | ~$20 | 1 | Onboard sensor fusion offloads host; shell-mounted, de-rotated against body frame. Alternative: BMI270 + Mahony on host = $4 but burns MCU cycles and needs mag cal — **BNO085 wins** given our 20 L budget and short build timeline. |
| **Actuator integrated encoder** | Per-panel position feedback (0–60 mm) | depends on Agent 2 (step-dir, CAN, analog pot) | ≥200 Hz | included | 12 | Requirement handoff to Agent 2: Actuonix L16-R has built-in 10-bit pot; Iris Dynamics voice coils expose 1 µm linear encoder over CAN. Either acceptable. |
| **TI INA219** | Per-channel current + bus voltage for torque estimate & thermal protection | I2C (4 addresses/bus → needs 3 buses or a mux) | 100 Hz | ~$3 | 12 | TCA9548A 8-ch I2C mux ($2) lets one SDA/SCL pair read all 12. Gives virtual torque sensing (τ ≈ k_t · I) — removes need for discrete force sensors. |
| **TI INA260** | Main battery rail V + I (pack monitoring + brown-out guard) | I2C | 10 Hz | ~$10 | 1 | 0–36 V / ±15 A range covers a 6S LiPo. Feeds Agent 7. |
| **Hall effect thermistor** (NTC 10k) | Motor coil temperature on the 3 highest-duty actuators | analog ADC | 1 Hz | $0.50 | 3 | Only instrument the panels whose duty cycle is highest (rear-push tiles during rolling) — thermal limit is a known risk for any voice coil. |

Core stack subtotal: **~$96** + negligible wiring.

## 2. Perception options comparison

| # | Option | Hardware | Raw BW | Latency | Cost | Mech. complexity | Obstacle avoid | SLAM | Teleop | VLM |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Distributed ToF array** | 8× VL53L5CX (8×8 zones, 60° FoV, 15 Hz) on 8 of 12 faces | ~62 KB/s total | ~70 ms | ~$140 | **low** (no moving parts) | **yes** | weak (no features) | telemetry only | no |
| 2 | **Gimbal-stabilized inner fisheye** | 2-axis BLDC gimbal + 1.0 kg pendulum + Arducam IMX708 fisheye + RPi 5 | 30 MB/s @ 1080p30 | 50 ms | ~$280 | **high** (moving subassembly, CoM critical) | yes | **yes** | yes | **yes** |
| 3 | **Transparent window panels** | 2 clear PC panels replacing 2 of 12 tiles + inner fisheye | 30 MB/s | 50 ms | ~$150 | medium | partial | no | partial | partial |
| 4 | **Equatorial 4-fisheye array** | 4× OV9281 global-shutter + Jetson Orin Nano | 120 MB/s @ 400p120 ×4 | 30 ms | ~$500 | high (4 gaskets in compliant skirt) | yes | **yes** | yes | yes |

Option 3 loses 2 of 12 actuators — unacceptable given we already dropped
from 42 to 12 for mechanical simplification; each remaining actuator is
load-bearing for the SH-9 basis. **Rejected.**

Option 4 punches 4 holes through the TPU gasket skirt — each hole is a
failure mode for the rolling seal and adds manufacturing complexity that
Agent 4 already flagged for the pentagonal tiles. **Rejected** for the
prototype; revisit in v4.

## 3. Recommendation

**Primary: Option 2 (gimbal-stabilized inner fisheye) + Option 1 as a
redundant proximity layer.**

Rationale, ordered by downstream-task unlock value:
1. **VLM / Claude-in-the-loop demos need gravity-stable video.** This is
   the project's differentiator — Claude designed the robot, Claude should
   also *see through its eyes*. A de-rotated ToF field can't feed a VLM;
   a stabilized camera can.
2. **SLAM requires visual features**, which ToF arrays do not provide.
   Open3D-style ICP on ToF point clouds degrades badly below 16×16 zones.
3. **The Dodec-12 already plans an inner rib cage** (per `topology.md`
   §A Geometry and open-question #5). That cage is the obvious mount for
   a pendulum gimbal — the mechanical reuse was pre-committed.
4. **Option 1 remains cheap insurance.** 8× VL53L5CX on the face backs
   (not replacing tiles, just glued to the inner face of 8 PETG plates —
   IR transparent at 940 nm through thin PETG — verify with a coupon
   test) give an omnidirectional safety halo for collision stop even if
   the gimbal jams. $140 buys a second perception channel with
   independent failure modes.
5. **Compute headroom**. RPi 5 (8 GB) decodes 1080p30 H.264 in hardware,
   runs Claude-CLI-in-a-tunnel for VLM calls, and still has budget for
   ORB-SLAM3 at 15 Hz on the CSI fisheye.

**Fallback**: if the gimbal+pendulum subassembly conflicts with Candidate
A's inner actuator rib layout (Agent 4 to confirm), demote to Option 1
alone for v1 — proximity-only is enough for obstacle avoidance and
teleop with head-mounted camera on the operator side.

## 4. Compute platform recommendation

| Option | Fit |
|---|---|
| ESP32-S3 | Too little RAM for vision; fine as *sub-MCU* on the motor-control bus — **include as slave**, not primary. |
| Teensy 4.1 | Excellent hard-real-time motor loop (600 MHz M7, FlexCAN ×2) — **include as motor controller**, not primary. |
| **RPi 5 (8 GB)** | **Primary pick.** Runs ROS 2 Jazzy, CSI fisheye, ORB-SLAM3, Wi-Fi 6, USB-C PD. 5 W idle / 12 W active. Fits the 20 L internal volume trivially. |
| Jetson Orin Nano | Overkill and hotter (15 W sustained) unless we run local VLM inference — we don't, we call Claude API. Defer to v4. |

**Architecture**: RPi 5 (brain, vision, Wi-Fi, cloud VLM calls) → USB or
UART → Teensy 4.1 (1 kHz motor loop, IMU read, INA219 fan-out) → CAN/RS-485
→ 12 actuator drivers. This is the standard "Linux SBC + realtime MCU"
split and keeps vision latency decoupled from control latency.

## 5. Bandwidth budget

| Stream | Payload | Rate |
|---|---|---|
| BNO085 (quat+gyro+accel, 40 B) | 16 KB/s | 400 Hz |
| 12× encoder + current (24 B each) | 58 KB/s | 200 Hz |
| 12× INA219 (4 B ch ×2) | 10 KB/s | 100 Hz |
| INA260 + thermistors | <1 KB/s | 10 Hz |
| 8× VL53L5CX (128 B / frame) | 62 KB/s | 15 Hz |
| Fisheye 1080p30 H.264 (CSI) | ~4 MB/s | 30 Hz |

Control-loop side (IMU + encoders + currents + ToF) = **~145 KB/s** —
trivial for a single CAN bus at 1 Mbit and a Teensy 4.1 UART-to-Pi link.
Vision side (4 MB/s compressed) rides the RPi 5's CSI-2 lane directly,
never touches the Teensy. **Both buses have >10× headroom.**

## 6. Power budget (sensor stack only)

| Item | Idle | Active | Peak |
|---|---|---|---|
| BNO085 | 3 mA @ 3.3 V = 10 mW | 12 mW | 15 mW |
| 12× INA219 + mux | 12 mW | 12 mW | 12 mW |
| INA260 | 3 mW | 3 mW | 3 mW |
| 8× VL53L5CX (20 mA each @ 3.3 V) | 50 mW standby | **530 mW** | 660 mW |
| Thermistors (×3) | negligible | <1 mW | <1 mW |
| **Gimbal BLDCs** (2×, at hold) | 200 mW | 1.5 W | 4 W |
| Arducam IMX708 | 100 mW | 350 mW | 500 mW |
| **RPi 5 (compute)** | 3 W | 7 W | 12 W |
| **Teensy 4.1** | 100 mW | 400 mW | 500 mW |
| **Totals** | **~3.5 W** | **~10 W** | **~18 W** |

At 10 W average on a 6S 5 Ah LiPo (111 Wh usable), sensor+compute alone
draws ~9% / hour — leaves plenty of budget for the 12 actuators. Agent 7
to confirm rail integration; the gimbal BLDCs want a clean 12 V rail
(buck from 6S), and the RPi 5 needs 5 V @ 5 A via USB-C PD or a
dedicated buck. **The RPi 5 is the dominant sensor-side load by 3×.**

## 7. Open questions

**For Agent 7 (power + electronics)**:
1. Can the gimbal motors share a 12 V bus with the panel actuators, or do
   they need a separate rail to prevent motor-noise coupling into the IMU?
2. PoE-style single-cable run for RPi 5 off the main pack, or a separate
   buck-converter hat? Thermal considerations — RPi 5 wants airflow we
   don't have inside a sealed rolling shell.
3. Is a 6S 5 Ah LiPo still sized right once sensors add ~10 W baseline?
   Estimated sensor+compute energy for 30 min operation = ~5 Wh (~5% of
   pack).

**For Agent 8 (cost rollup)**:
1. Sensor BoM subtotal: core **$96** + gimbal+camera **~$280** + ToF
   array **$140** + RPi 5 **$80** + Teensy 4.1 **$30** + wiring/connectors
   **~$50** = **~$676** for the sensor + compute stack.
2. Lead times: BNO085 and VL53L5CX are Adafruit-stocked (1 week);
   Arducam IMX708 also ~1 week; RPi 5 variable but typically in stock at
   PiShop. **No long-lead items in the sensor stack.**
3. Machining: gimbal yoke is the only non-printable part — recommend
   aluminum 6061 CNC via SendCutSend (~$40, 5 business days). Otherwise
   the whole sensor mount system is FDM-printable on the Ender-3 already
   allocated to Agent 5's BoM.
