# Urchin v3 — Power & Electronics Architecture (Agent 7)

Scope: interior electronics for the DODEC-12 rolling shell (0.34 m, 2.5 kg,
12 linear actuators). Runtime target 30 min continuous rolling. Fully
sealed — no tether. Three actuator classes carried in parallel with Agent 2.

---

## 1. Block diagram

```
            ┌─────── charge port (USB-C, gasketed, pogo-pin aux) ───────┐
            │                                                            │
  +──────+  │  +──────────+   22.2V bus   +──────────────+   +─────────+ │
  | 6S   |──┴─▶|  BMS /   |───────┬──────▶|  DRV per     |──▶| VCM /   | │
  | LiPo |    | e-fuse   |       │       |  panel ×12   |   | BLDC    | │
  +──────+    +──────────+       │       +──────────────+   | DC motor| │
     │                            │                                     │
     │        +──────────+  5V    │       +──────────────+   +─────────+ │
     └───────▶| Buck 5V/3A───────┴──────▶| Teensy 4.1   |◀──│ IMU     │ │
              +──────────+               |   (main MCU) |◀──│ BNO085  │ │
                                         +──────┬───────+   +─────────+ │
                                                │                       │
                                         CAN 1 Mbps (twisted pair)      │
                                         ├─▶ panel 1 driver node        │
                                         ├─▶ panel 2 driver node        │
                                         ├─▶ ...                        │
                                         └─▶ panel 12 driver node       │
                                                │                       │
                                         +──────┴───────+   +────────+  │
                                         | ESP32-S3    │──▶│  Phone │  │
                                         |  (telem)    │   │  BLE   │  │
                                         +─────────────+   │  WiFi  │  │
                                                           └────────┘  │
                                         E-stop: hardware latch on     │
                                         BMS (magnetic reed sw on      │
                                         shell exterior) → opens bus ──┘
```

Everything above the CAN line is always powered; the 12 driver nodes are
individually gated by the BMS e-fuse on a single common 22.2 V rail.

---

## 2. Main controller

**Primary: Teensy 4.1** (PJRC, $31.50). 600 MHz Cortex-M7, 3× FlexCAN
(2.0B + CAN-FD), 7× SPI, 3× I²C, hardware float, deterministic 1 kHz
control loop. More than enough for a 200 Hz outer loop + 1 kHz inner
loop across 12 panels, IMU fusion, and SH-9 projection math. Fits on a
50×20 mm PCB.

**Co-processor: ESP32-S3-WROOM-1** ($6) on the same carrier for WiFi/BLE
telemetry — Teensy has no radio. UART bridge, 460800 baud, Teensy
publishes panel/IMU frames at 50 Hz.

Alternatives considered:
- *Raspberry Pi 5 + HAT* — overkill; 5 W idle eats ~10% of the battery
  for no gain unless Agent 6 says vision is mandatory. Deferred.
- *ESP32-S3 alone* — viable and cheap, but single CAN peripheral and
  jittery RTOS under WiFi load. Pass.
- *STM32H743 Nucleo* — equivalent to Teensy feature-wise, worse toolchain.

---

## 3. Bus topology

**Recommended: CAN 2.0B @ 1 Mbps, single twisted pair daisy-chained
through all 12 driver nodes.**

| Option | Wire count from MCU | Worst-case latency | Notes |
|---|---|---|---|
| CAN daisy-chain | 4 (CANH/CANL/GND/12V or 22V) | ~200 µs/panel frame | Standard for multi-motor robots; arbitration handles 12 nodes easily at 10% bus load. |
| Per-panel I²C | ~26 (SDA/SCL shared + 12 individual addr strap) | ~500 µs | Not robust to EMI from motor switching; clock stretching hazards at 12 slaves. |
| Per-panel PWM from main MCU | 36 (step/dir/enable × 12) | ~50 µs | No closed-loop current control without return wires; wire count blows up through the inner-frame ribs. |

CAN also future-proofs the swap to ODrive/moteus (both CAN-native) if we
upgrade actuator class in place. 1 Mbps gives ~8 kframes/s total; at
200 Hz × 12 panels × 2 frames (cmd+reply) = 4800 fps, 60% headroom.

---

## 4. Power budget (three actuator classes)

Design duty cycle: at any moment ≤4 of 12 panels at peak, ≤8 at
mid-load. "Avg W" = peak × 0.30 duty.

### Class A — Voice-coil motor (15 W peak / ch)

| Subsystem              | Cont. W | Peak W | Duty | Avg W |
|------------------------|---------|--------|------|-------|
| 12× VCM actuators      | 12.0    | 180    | 30%  | 54.0  |
| Teensy 4.1 + ESP32     | 1.2     | 1.5    | 100% | 1.2   |
| IMU + sensors (BNO085 + 12 pos encoders) | 0.8 | 1.0 | 100% | 0.8 |
| CAN transceivers (12×) | 0.6     | 0.6    | 100% | 0.6   |
| Buck converter loss (92%) | — | — | — | 4.7 |
| **Total**              |         |        |      | **~61 W** |

### Class B — Lead-screw BLDC (8 W peak / ch)

| Subsystem              | Cont. W | Peak W | Duty | Avg W |
|------------------------|---------|--------|------|-------|
| 12× lead-screw BLDC    | 6.0     | 96     | 30%  | 28.8  |
| MCU + comms + sensors  | 2.6     | 3.1    | 100% | 2.6   |
| Driver losses (90%)    | —       | —      | —    | 3.5   |
| **Total**              |         |        |      | **~35 W** |

### Class C — Geared DC linear (5 W peak / ch, 0.2 m/s)

| Subsystem              | Cont. W | Peak W | Duty | Avg W |
|------------------------|---------|--------|------|-------|
| 12× Actuonix-class     | 3.6     | 60     | 30%  | 18.0  |
| MCU + comms + sensors  | 2.6     | 3.1    | 100% | 2.6   |
| Driver losses (85%)    | —       | —      | —    | 3.6   |
| **Total**              |         |        |      | **~24 W** |

---

## 5. Battery sizing

30 min × avg W → Wh needed; add 25% margin for low-voltage cutoff and
aging.

| Class | Avg W | 30-min Wh | +25% | Recommended pack |
|---|---|---|---|---|
| A (VCM) | 61 | 30.5 | 38 | **6S 2200 mAh LiPo = 48.8 Wh, 320 g** (Turnigy Graphene 6S 2200 mAh 65C, ~$55); peak draw 180/22.2 = 8.1 A, well under 65C × 2.2 Ah = 143 A |
| B (BLDC) | 35 | 17.5 | 22 | **6S 1500 mAh LiPo = 33 Wh, 220 g** (Turnigy Graphene 6S 1500 mAh, ~$42) |
| C (DC)  | 24 | 12.0 | 15 | **4S 2200 mAh LiPo = 32.6 Wh, 235 g** (at 14.8 V nominal; drop bus voltage) |

All fit with huge margin in the ~20 L interior. Pack mass sits in the
interior tray and contributes to the pendulum-like CoM anyway.

**Chemistry call**: 6S LiPo for A/B — matches ODrive/moteus native bus
voltage (22 V is also the DRV8874 sweet spot), high discharge rate, and
light. Li-ion 18650 (Samsung 30Q 6S1P = 3 Ah, 66 Wh, 300 g, ~$60) is
safer on thermal runaway and preferred if the user runs indoor-only —
charge rate is slower (1C vs LiPo's 3C) but acceptable.

**Regulators**:
- 22.2 V → 5 V: **Pololu D24V25F5** (2.5 A, 92% eff, $15). Feeds Teensy,
  ESP32, IMU, encoders.
- 5 V → 3.3 V: Teensy's onboard LDO is fine for its MCU + low-current
  peripherals.
- If an encoder ring needs 3.3 V at >500 mA, add one AP2112K-3.3 per
  panel driver node.

---

## 6. Motor drivers

**Class A (VCM, recommended primary):**
**TI DRV8874-Q1** × 12 ($3.80 each, QFN-16). 38 V, 6 A peak, integrated
current sense (2.1 V/A typical), PWM or PH/EN interface. Pair each with
a **MCP2515 + TJA1051** CAN-bridge + ATmega328p-class slave MCU per
panel node (or, cleaner, one **STM32G0B1CBT6** per node, $2.50, which
has a built-in CAN-FD controller, 6× PWM, and 12-bit ADC for current
sensing — total per-panel driver BoM ≈ $8).

**Class B (BLDC lead-screw):**
Cleanest path: **ODrive Micro** × 12 ($140 ea = $1680) — overkill on
cost but plug-and-play CAN, integrated FOC, encoder input. Alternative:
**moteus r4.11** × 12 ($100 ea = $1200) — smaller, CAN-FD, Cheetah-style
commutation. Budget option: **SimpleFOC Mini** × 12 ($22 ea = $264,
discrete MP6540 + STM32G431) but requires firmware work.

**Class C (geared DC):**
DRV8874 × 12 identical to Class A, just never hits the high end of its
range. $46 total. Or, if cost is truly the driver, **L298N modules**
($2 ea) work but the 2 V saturation drop wastes ~15% of the budget.

---

## 7. Safety

- **Per-panel fuse**: 3 A SMD (0603) at each panel node input, clears in
  <10 ms on coil short.
- **Main bus fuse**: 15 A ANL inline with the battery pack +.
- **BMS**: integrated into the LiPo pack via **Smart BMS 6S 40 A**
  ($25) — cell balancing, over-discharge cutoff at 3.0 V/cell,
  over-current trip at 40 A, thermistor on pack.
- **Firmware thresholds**: at 3.3 V/cell (bus ≈ 19.8 V) Teensy raises a
  "low battery" telemetry flag and reduces action-scale to 0.5× to
  extend runtime. At 3.2 V/cell it commands all panels to neutral and
  opens CAN to the drivers (motors coast).
- **Thermal**: 10 kΩ NTC on each DRV8874; >85 °C → panel soft-disable.
- **Watchdog**: Teensy runs a hardware WDT at 100 ms; if the main loop
  stalls, the reset line drops the CAN-transceiver enable, which puts
  all drivers in their safe-state (high-Z outputs).
- **E-stop**: magnetic reed switch on the shell exterior (through the
  PETG wall, no gasket penetration) wired to the BMS discharge-enable
  pin. Touch a magnet to the marked spot → pack disconnects.

---

## 8. Wireless telemetry / control

- **BLE (ESP32-S3)**: phone app for start/stop, reward-eval input, live
  IMU trace. 10 m range; that's the right fit for "user watching the
  robot roll on the floor."
- **WiFi (ESP32-S3, 2.4 GHz)**: dev tether. Streams 50 Hz frames of
  `{panel_pos[12], panel_I[12], quat[4], omega[3]}` → ~100 B → 40 kbps,
  trivial. Feeds the existing Studio UI training panel unchanged.
- **Secondary charging-position RF**: not needed. USB-C presence detect
  through the charge port already tells the MCU "I'm docked." BLE
  covers any "come back here" prompts.

---

## 9. Open questions

- **Agent 6 (sensors)**: final IMU choice — is BNO085 enough (400 Hz
  fused quat), or does perception need a raw-IMU + external filter
  path at 1 kHz (ICM-42688-P)?  Also: per-panel absolute position — Hall
  strip, magnetic encoder, or open-loop step count?
- **Agent 2 (actuators)**: which driver interface do the shortlisted
  actuators expose — CAN (ODrive/moteus), step+dir, or raw PWM+encoder?
  This determines whether the per-panel STM32G0 node is needed or the
  driver speaks CAN natively.
- **Agent 8 (cost rollup)**: end-to-end electronics BoM for the
  recommended architecture below.

---

## Recommended architecture (one line)

**6S 2200 mAh LiPo → 15 A fuse → Smart BMS → 22.2 V bus → (a) Pololu
D24V25F5 buck → 5 V rail → Teensy 4.1 + ESP32-S3 + BNO085, and (b) 12×
per-panel STM32G0 + DRV8874 driver nodes on a CAN daisy-chain.** Total
electronics BoM ≈ $250 excluding actuators, 30-min runtime with ~60%
energy margin on Class A (VCM) and >100% on Classes B/C.
