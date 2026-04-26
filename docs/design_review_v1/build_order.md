# Urchin-lite v1 (OCTA-8) — Build Order / Physical BoM (Agent B2)

Date: 2026-04-22

## Context

This is the SKU-level physical bill of materials for **one** OCTA-8
urchin_lite_v1 unit: a ~0.30 m octahedral shell with 8 triangular face
panels, each driven by a single linear actuator along the face normal.
Prices are retail, USD, for qty=1 of the robot (so 8× actuators, not 12×).
Every non-verifiable price falls through to the **Open flags** section at
the bottom; every row cites either a supplier URL or a "no verified
supplier" flag in the Notes column (or the Open flags section for the
row's unit price).

**Long-lead items flagged up front** (order before anything else — drive
the critical path):

1. **Sensata BEI Kimco LAS28-53 VCAs** (if chosen as primary or backup) —
   6–10 weeks direct from Sensata, and the per-unit price is not public
   at retail. Quote required.
2. **StepperOnline LK60 ball-screw stages** — 2–4 weeks ocean freight
   from the StepperOnline catalog; the 110 mm variant price was not
   published at WebFetch time (flagged).
3. **ODrive S1 + M8325s kits** (primary) — 2–3 weeks, stock-dependent
   at shop.odriverobotics.com.
4. **Dynamixel XL-series** (none in this BoM; listed here only because
   they would be the long-lead if Candidate 2 / TET-4×2 fallback is chosen).

**Critical corrected spec** (from Agent A2, `catalog_extension.md`): the
Sensata LAS28-53-000A-P01-12E is **25 mm stroke / 266.9 N peak**, not
the 53 mm / 60 N figure in `docs/design_review/actuator_shortlist.md`.
The OCTA-8 sim is tuned for 0–60 mm panel travel; a 25 mm VCA is
**35 mm short of spec** and cannot be a drop-in. This build lists the
Sensata row only in the "backup — premium, short-stroke retune
required" slot and promotes the ODrive S1 + M8325s rack-and-pinion
reference to **primary**, which matches the topology.md Candidate 1
recommendation and the original actuator_shortlist.md Candidate A.

---

## Primary BoM

### Actuators

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| ODrive S1 + M8325s BLDC + 4 mm pinion rack (custom reference design) | 8 | 235.00 | 1,880.00 | ODrive Robotics | `ODRV-S1 + M8325S` kit | 2–3 wk | Reference design per `linear.yaml::odrive_s1_rack_and_pinion_ref`. Unit price = ODrive S1 ($149) + M8325s ($60) + printed rack + thrust bearings + belt ($25). Agent A2 flags ~40 hr mechanical CAD per first cell; subsequent are clones. CAN-FD to all 8 on one bus. [https://shop.odriverobotics.com/products/odrive-s1](https://shop.odriverobotics.com/products/odrive-s1) |
| ODrive S1 + M8325s spare actuator (full cell) | 1 | 235.00 | 235.00 | ODrive Robotics | `ODRV-S1 + M8325S` kit | 2–3 wk | One spare for field repair. Same SKU as above. |
| JST-GH 4-pin CAN-FD cable kit (0.5 m, pre-crimped) | 10 | 6.50 | 65.00 | Digi-Key / Molex | `WM15128-ND` | 1–3 d | 8 actuators + spare + 1 loss. Dispatches CAN + power pairs per ODrive wiring guide. |
| XT30 pigtails for per-cell 24 V power drop | 10 | 2.00 | 20.00 | Amazon / Digi-Key | generic `XT30-F-PIG-150` | 1–3 d | Bulk 10-pack, ~$20. |

**Actuators subtotal: $2,200.00**

### Driver electronics

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| CAN-FD transceiver USB dongle (host side) | 1 | 80.00 | 80.00 | ODrive Robotics | `CANdle` or `CANable 2.0` | 1–3 d | Needed on the Pi/Jetson end to sit on the ODrive bus at 1 Mbit. [https://canable.io/](https://canable.io/) |
| CAN-FD bus termination resistor (120 Ω) | 2 | 1.00 | 2.00 | Digi-Key | `A100933CT-ND` | 1–3 d | One each end of the 8-node bus. |
| Bus-side electrolytic capacitor 4700 µF 35 V (ODrive-recommended) | 2 | 8.00 | 16.00 | Digi-Key | `UVR1V472MHD1TN` | 1–3 d | ODrive docs recommend bulk capacitance per ~2 motors. Agent 2 open-question #1 (actuator_shortlist.md): worst-case simultaneous peak is ~10 cells × ~50 A momentary — this is the capacitor that rides out those transients. |

**Driver electronics subtotal: $98.00**

(The ODrive S1 **is** the motor driver for each M8325s. It's counted in the Actuators row above; no separate ESC line item.)

### Compute / main board

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| Raspberry Pi 5 (8 GB) | 1 | 80.00 | 80.00 | Adafruit / PiShop | `SC1112` | 1–3 d | 8-dim actor network at ≤ 200 Hz control rate is ~1% of a Pi 5 core. A Jetson Orin Nano Super is overkill unless onboard perception/vision is added. |
| Pi 5 active cooler | 1 | 5.00 | 5.00 | Adafruit | `SC1148` | 1–3 d | |
| Pi 5 27 W USB-C PSU | 1 | 12.00 | 12.00 | Adafruit | `SC1154` | 1–3 d | Bench use only. In-robot, the Pi runs off a 5 V BEC from the 4S pack (listed in Power). |
| microSD 128 GB A2 | 1 | 18.00 | 18.00 | Amazon | SanDisk `SDSQXAO-128G-GN6MA` | 1–3 d | |

**Compute subtotal: $115.00**

### IMU / sensors (minimal)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| Adafruit BNO085 9-DOF IMU with on-chip fusion | 1 | 24.95 | 24.95 | Adafruit | `4754` | 1–3 d | Built-in sensor fusion; outputs quaternion orientation. RL obs layer can drop raw accel/gyro straight in. [https://www.adafruit.com/product/4754](https://www.adafruit.com/product/4754) |
| Panel-position feedback: ODrive integrated magnetic encoder | 0 | — | 0.00 | (bundled with M8325s motor) | — | — | 16384 CPR integrated with the M8325s; panel position is shaft_angle × (4 mm pitch / 2π). No per-panel linear encoder needed for v1. |
| FSR 402 force-sensing resistor (optional contact patch) | 2 | 6.95 | 13.90 | Adafruit / SparkFun | `166` | 1–3 d | Optional — two of the eight panels get a contact-force FSR glued to the inner face. Useful RL obs. Skip if building minimal. |
| ADS1115 16-bit ADC breakout for FSR readout | 1 | 14.95 | 14.95 | Adafruit | `1085` | 1–3 d | I²C on the Pi; feeds FSR analog voltage. |

**IMU/sensors subtotal: $53.80**

### Power

Power sizing math (per actuator_shortlist.md open question #1 and topology.md
physics anchors): 8 cells × worst-case simultaneous 50 A momentary peak is
400 A bus — but the SH-9 control basis rarely drives all 8 to peak at once.
Realistic mean draw is ~10 A per active cell × ~4 simultaneously active =
40 A continuous, with short peaks to 400 A absorbed by the bus capacitor
bank (listed under Driver electronics). A 4S 5000 mAh 50C LiPo delivers
250 A continuous / 500 A burst at 14.8 V — sized for this profile.

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| 4S 5000 mAh 50C LiPo (14.8 V) | 1 | 65.00 | 65.00 | HobbyKing / Amazon | Turnigy Graphene `9067000184` | 1–3 d | Graphene-class for the high-burst C-rating. ODrive S1 accepts 14–56 V; 14.8 V nominal is in the low end, leaving current overhead. If Agent 3 telemetry says effort_mean is low (panels pulse, don't sit), we can step down to 4S 3300 mAh and save ~120 g. |
| 4S LiPo balance charger (iSDT Q6 Pro or equiv) | 1 | 80.00 | 80.00 | Amazon | iSDT `Q6-PRO-300W` | 1–3 d | 300 W bench charger. |
| BMS / over-current cutoff (40 A continuous, 4S) | 1 | 22.00 | 22.00 | Amazon | `DALY-4S-40A` | 1–3 d | Low-side N-FET cutoff. Wire in series on the pack negative. |
| 40 A slow-blow automotive fuse + inline holder | 1 | 8.00 | 8.00 | Amazon | `ATO-40A` | 1–3 d | Belt-and-suspenders with the BMS. |
| XT60 male/female connector pair (pack side) | 2 | 3.00 | 6.00 | Amazon / Digi-Key | generic `XT60H-M/F` | 1–3 d | |
| 14 AWG silicone wire (red + black, 3 m each) | 1 | 15.00 | 15.00 | Amazon | BNTECHGO `14-AWG-SOFT-SILICONE` | 1–3 d | Main DC bus from pack → distribution board. |
| Panel-wire: 22 AWG high-flex silicone (8-color, 10 m spool) | 1 | 18.00 | 18.00 | Amazon | `22AWG-HIFLEX-8C` | 1–3 d | Panel motion fatigues stiff wire; silicone high-flex mandatory per Agent 4 v3 audit. |
| Power-distribution board (4S, 40 A rated) | 1 | 15.00 | 15.00 | Amazon / RC hobby | generic `4S-PDB` | 1–3 d | Breaks the pack bus out to 8 actuator drops. |
| 5 V / 5 A UBEC for Pi 5 | 1 | 14.00 | 14.00 | Amazon / HobbyKing | `HobbyWing-UBEC-5V5A` | 1–3 d | Steps 14.8 V down to 5 V for the Pi. Never run a Pi off a cheap linear regulator off a LiPo. |
| SPST 40 A master switch (panel-mount) | 1 | 10.00 | 10.00 | Amazon | `Rotary-40A-SPST` | 1–3 d | Main kill. Also functions as the "don't leave it armed" visible indicator. |

**Power subtotal: $253.00**

### Shell / structural (3D printed)

Scaled from `docs/design_review/print_bom.md` (which covered the v3
Dodec-12 at 0.34 m, 12 pentagonal tiles) to OCTA-8 at 0.30 m with 8
equilateral triangular tiles (~250 mm edge). Triangular tiles are ~66%
the area of the v3 pentagons, so mass and print time scale down per part;
count drops from 12 → 8, so the total PETG mass roughly matches or
slightly undercuts v3's 1.41 kg. Round up to **1.5 kg PETG** / **0.1 kg
TPU 95A** total allowance.

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| PETG filament (1 kg spool) | 2 | 25.00 | 50.00 | Amazon / Prusament | Overture `PETG-BLK-1KG` | 1–3 d | 1.5 kg build + waste; 2 spools. Catalog row `materials/3d_printable.yaml::petg_standard` ($22/kg; use $25 retail). |
| TPU 95A filament (1 kg spool) | 1 | 35.00 | 35.00 | Amazon / SainSmart | SainSmart `TPU-95A-BLK-1KG` | 1–3 d | Only 100 g needed; minimum purchase is 1 kg. Catalog row `materials/3d_printable.yaml::tpu_95a`. 8 triangular gasket strips at octahedron edges. |
| Nylon CF filament (0.5 kg) — upgrade path for actuator brackets only | 0 | 35.00 | 0.00 | Amazon / Polymaker | `PolyMide-PA612-CF-500g` | 1–3 d | **Not** in baseline BoM; buy only if impact testing shows PETG brackets cracking. $0 this row. Catalog row `materials/3d_printable.yaml::nylon_cf`. |
| M3×10 socket-head screw (100-pack) | 1 | 10.00 | 10.00 | McMaster-Carr | `91290A115` | 1–2 wk | |
| M3×5 brass heatset insert (100-pack) | 1 | 12.00 | 12.00 | Amazon / McMaster | `94459A130` | 1–3 d | |
| M4×16 socket-head screw (50-pack, panel-to-rack) | 1 | 8.00 | 8.00 | McMaster-Carr | `91290A194` | 1–2 wk | |
| M4 washer (100-pack) | 1 | 6.00 | 6.00 | McMaster-Carr | `98689A113` | 1–2 wk | |
| Linear-guide miniature rod 6 mm × 120 mm (panel motion guide) | 16 | 4.50 | 72.00 | McMaster-Carr | `6061K11` | 1–2 wk | Two parallel guide rods per panel × 8 panels = 16. Prevents rack skew under side load. |
| Flanged linear bushing 6 mm ID (IGUS or equiv) | 16 | 3.50 | 56.00 | McMaster-Carr / IGUS | `6389K131` | 1–2 wk | Self-lubricating plastic bushing per rod. |
| Deep-groove ball bearing 608 (pinion thrust) | 8 | 2.50 | 20.00 | McMaster-Carr | `60355K501` | 1–2 wk | One per actuator cell for pinion-shaft thrust takeup. |
| Extension spring, 0.5 N·mm preload (rack return) | 8 | 1.50 | 12.00 | McMaster-Carr | `9044K111` (or similar lb-rated) | 1–2 wk | Keeps pinion engaged with rack; actuator_shortlist.md Candidate A notes this is mandatory. |

**Shell / structural subtotal: $281.00**

### Development accessories (one-time, not per-unit)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| 24 V / 10 A bench DC PSU | 1 | 85.00 | 85.00 | Amazon | Riden `RD6012W` or Korad `KA3010D` | 1–3 d | For tethered first-rolls before the pack is trusted. |
| USB-C oscilloscope (basic, for CAN-FD debug) | 0 | — | 0.00 | — | — | — | **Optional.** Skip unless the first build has a bus-arbitration issue. |

**Dev accessories subtotal: $85.00**

---

## Primary BoM — Category totals

| Category | Subtotal $ |
|---|---|
| Actuators | 2,200.00 |
| Driver electronics | 98.00 |
| Compute | 115.00 |
| IMU / sensors | 53.80 |
| Power | 253.00 |
| Shell / structural | 281.00 |
| Dev accessories | 85.00 |
| **Grand total (primary)** | **3,085.80** |

**3 most expensive line items (primary):**

1. 8× ODrive S1 + M8325s actuators — **$1,880**
2. 1× ODrive S1 + M8325s spare actuator — **$235**
3. Shell / structural (aggregate) — **$281** (driven mostly by 16× linear rods + 16× bushings at ~$128 combined)

---

## Backup BoM

One row per primary line item that has an alternate SKU or substitute
supplier. Where no backup is known, the row says "no backup sourced,
FLAG" and shows up in Open flags below.

### Actuators (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| **Backup A (conservative):** StepperOnline LK60-40DL10S3-110 ball-screw stage (motor separate) | 8 | TBD | TBD | StepperOnline | `LK60-40DL10S3-110` | 2–4 wk | 110 mm stroke, 196 N, 500 mm/s per `linear.yaml::stepperonline_lk60_40dl10s3_110`. Unit price **not published on public listing** (WebFetch 403). The 510 mm sibling sells for $155.52; the 110 mm variant is expected ≤ that but needs quote. FLAG. |
| + matched NEMA-23 closed-loop stepper or NEMA-24 servo for LK60 | 8 | 90.00 | 720.00 | StepperOnline | `CL57T + 23HS45-4204S` | 2–4 wk | Required companion to LK60 stage. Midpoint of $60–$130 range. |
| **Backup B (premium, short-stroke retune required):** Sensata BEI Kimco LAS28-53-000A-P01-12E VCA | 8 | TBD | TBD | Sensata BEI Kimco | `LAS28-53-000A-P01-12E` | 6–10 wk | Per `linear.yaml::sensata_las28_53_vca`: **25 mm stroke, 266.9 N peak** — 35 mm short of the 60 mm sim spec. Developer Kit DK-LAS28-53-000A-P01-12E retails ~$1,200–1,500 (1 unit, with driver+cables). Bare actuator at qty 8 needs **direct quote from Sensata**. FLAG. Requires current-mode driver (Copley Xenus / A-M-C AZBE ~$350 each, not listed here). |
| + Copley Xenus Plus XPL-230-18 VCA driver (if Backup B chosen) | 8 | 350.00 | 2,800.00 | Copley Controls | `XPL-230-18` | 4–8 wk | Only needed under Backup B. Per actuator_shortlist.md. |
| ODrive S1 + M8325s spare (backup = same SKU as primary) | 1 | 235.00 | 235.00 | ODrive Robotics | `ODRV-S1 + M8325S` | 2–3 wk | No backup for the spare — just keep one. |
| JST-GH CAN-FD cable kit (backup) | 10 | 6.50 | 65.00 | Amazon | `JST-GH-4P-PIG` (generic) | 1–3 d | Same part as primary from different vendor. |
| XT30 pigtails (backup) | 10 | 2.00 | 20.00 | HobbyKing | — | 1–3 d | Same idea, different retailer. |

**Backup actuators subtotal** (Backup A conservative path, with known prices only): **~$720 + TBD (LK60 stage ×8)** → **at $160 ea LK60 estimate = ~$2,000**. Full Backup A ~$2,785. Backup B premium is ~$5,600 + quote.

### Driver electronics (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| CANable 2.0 dongle (backup to CANdle) | 1 | 60.00 | 60.00 | Protofusion / Tindie | `CANable-2.0-Pro` | 1–3 d | Open-source CAN-FD dongle. |
| Termination resistors, bus cap — no backup needed, commodity | — | — | — | — | — | — | Same SKU works from any distributor. |

### Compute (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| NVIDIA Jetson Orin Nano Super 8 GB dev kit | 1 | 249.00 | 249.00 | SparkFun / Seeed | `900-13767-0030-000` | 1–2 wk | Overkill on raw TFLOPs but useful if onboard vision gets added later. Doubles compute row cost vs Pi 5. |
| microSD 128 GB (same as primary) | 1 | 18.00 | 18.00 | Amazon | — | 1–3 d | Same SKU. |

### IMU (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| Bosch BMI270 breakout (backup to BNO085) | 1 | 7.50 | 7.50 | SparkFun | `SEN-22397` | 1–3 d | 6-DOF only (no magnetometer); fusion must run on the Pi. ~3× cheaper but more setup effort. |
| FSR 402 — no backup; commodity | — | — | — | — | — | — | Interchangeable across SparkFun / Adafruit / Digi-Key. |

### Power (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| 4S 5000 mAh 50C LiPo — backup vendor | 1 | 70.00 | 70.00 | Amazon / Gens Ace | `Gens-Ace-Bashing-5000-4S-50C` | 1–3 d | Different cell chemistry brand; same form factor. |
| iSDT D2 charger (backup to Q6 Pro) | 1 | 60.00 | 60.00 | Amazon | `iSDT D2 200W` | 1–3 d | Cheaper, lower wattage (slower charge). |
| BMS — no backup sourced, FLAG | — | — | — | — | — | — | DALY is the usual vendor; an equivalent from BatteryHookup / Overkill Solar is workable but un-specced here. FLAG. |

### Shell / structural (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| PETG — backup spool | 2 | 22.00 | 44.00 | Hatchbox / Amazon | `Hatchbox-PETG-1KG` | 1–3 d | Different brand; same catalog row. |
| TPU 95A — backup spool | 1 | 40.00 | 40.00 | NinjaFlex / Amazon | `NinjaFlex-TPU95A-500G` | 1–3 d | Slightly softer 85A variant is **not** interchangeable; confirm 95A durometer. |
| Hardware backup (M3/M4 screws + heatsets) | — | — | — | Digi-Key / Fastenal | — | 1–3 d | Commodity; any fastener supplier. |
| Linear rod/bushing backup | 16 + 16 | 5.00 + 3.50 | 80 + 56 = 136.00 | IGUS DryLin W | IGUS `drylin-w-6mm` | 1–2 wk | IGUS direct is the commodity-grade backup to McMaster. |

### Dev accessories (backup)

| Part | Qty | Unit $ | Subtotal $ | Supplier | Part Number | Lead Time | Notes |
|---|---|---|---|---|---|---|---|
| 24 V / 10 A PSU (backup) | 1 | 75.00 | 75.00 | Amazon | Mean Well `LRS-350-24` | 1–3 d | Cheap hard-wire bench PSU; no display, but rock-solid 24 V / 14 A. |

---

## Backup BoM — approximate total (if fully substituted along "Backup A conservative" path)

| Category | Backup subtotal $ |
|---|---|
| Actuators (Backup A, with LK60 price estimated at $160 ea) | ~2,785.00 |
| Driver electronics | ~60.00 + 18.00 (same commodity) |
| Compute (Jetson path) | ~267.00 |
| IMU (BMI270 path) | ~7.50 + rest same |
| Power | ~130.00 + rest same (BMS FLAGGED) |
| Shell / structural | ~226.00 |
| Dev accessories | ~75.00 |
| **Grand total (Backup A, conservative ball-screw path)** | **~$3,550** (± LK60 quote) |
| **Grand total (Backup B, premium Sensata VCA path)** | **~$8,500+** (excluding Sensata direct quote for 8× bare VCAs) |

Backup B total is a lower bound — the eight bare LAS28-53 actuators at
qty 8 are **not priced at retail** and could easily push the Sensata
path above $12,000 per the original actuator_shortlist.md Candidate B
estimate.

---

## Lead-time breakdown

### 1–3 days (Amazon / Digi-Key / Adafruit in-stock)

- Raspberry Pi 5 + active cooler + 27 W PSU + microSD (Adafruit, PiShop)
- BNO085 IMU, FSR 402 × 2, ADS1115 (Adafruit)
- CAN-FD dongle (CANable 2.0 or CANdle)
- Bus termination resistors, bus capacitors (Digi-Key)
- 4S 5000 mAh LiPo (HobbyKing / Amazon)
- iSDT Q6 Pro charger, BMS, fuse, XT60, XT30, PDB, UBEC, master switch
- 14 AWG + 22 AWG silicone wire
- PETG × 2 spools, TPU 95A × 1 spool
- M3 heatset inserts × 100, M3×5 / M3×10 / M4×16 screws
- 24 V bench PSU
- JST-GH cable kit

### 1–2 weeks (McMaster-Carr, SparkFun)

- M3 / M4 socket-head screws, washers (McMaster)
- 6 mm × 120 mm linear rods × 16 (McMaster)
- Flanged linear bushings × 16 (McMaster / IGUS)
- 608 ball bearings × 8 (McMaster)
- Extension springs × 8 (McMaster)
- Jetson Orin Nano Super (backup compute, if chosen) — SparkFun / Seeed

### 2–4 weeks (Dynamixel-class direct-order, StepperOnline, ODrive)

- **ODrive S1 + M8325s kits × 9** (8 + 1 spare) — shop.odriverobotics.com, stock-dependent
- StepperOnline LK60-40DL10S3-110 stages × 8 (backup path only) — ocean freight from StepperOnline
- StepperOnline NEMA-23 closed-loop stepper × 8 (backup path only)

### 4–8 weeks (long lead: Sensata, Copley, etc.)

- **Sensata LAS28-53-000A-P01-12E VCAs × 8** (Backup B only) — direct quote required; lead time per Agent 2's prior review is 6–10 weeks
- **Copley Xenus Plus VCA drivers × 8** (Backup B only)

**Long-lead items to order FIRST on any path:**

- **Primary path:** the 9× ODrive S1 + M8325s kits. Place the order the same day the build is approved; everything else (fasteners, rods, print filament, Pi 5, IMU) is 1–3 days away and trivially parallelizable.
- **Backup A path:** 8× LK60 stages **and** 8× matched steppers/servos from StepperOnline the same day.
- **Backup B path:** the Sensata VCA quote is the critical path; kick it off immediately so pricing and lead-time are known before the decision is locked.

---

## Open flags (items needing manual follow-up before ordering)

1. **`linear.yaml::stepperonline_lk60_40dl10s3_110` price is `null`.** StepperOnline public listing returns 403 on WebFetch; the 510 mm sibling is $155.52 so the 110 mm is expected ≤ that, but this must be confirmed by loading the page manually (or emailing sales@omc-stepperonline.com) before Backup A is costed.
2. **`linear.yaml::stepperonline_lk60_40dl10s3_110` weight is `null`.** Needed for the mass budget in topology.md (target m ≈ 2.3 kg). Stage weight affects whether the spring/bushing sizing above is correct.
3. **`linear.yaml::sensata_las28_53_vca` price is `null`.** Requires direct quote from Sensata for qty=8 bare actuators. Developer Kit at ~$1,200–1,500 is the only public price point, and it includes driver + cables we don't need 8× of.
4. **`linear.yaml::sensata_las28_53_vca` weight is `null`.** Mass budget impact TBD.
5. **Sensata stroke mismatch vs OCTA-8 sim.** The 25 mm stroke is **35 mm short** of the 60 mm panel-travel spec. If Backup B is chosen, either (a) retune the sim to 0–25 mm panel travel and re-validate the Phase-1 contact-dipole rolling policy at the shorter stroke, or (b) add a ~2.4:1 mechanical amplifier (lever or belt) between the VCA shaft and the panel — cost and backlash TBD. Neither is free; both are project-blocking sub-tasks. FLAG.
6. **BMS backup SKU not sourced.** Only the DALY 4S-40A is specced. Alternate vendors (Overkill Solar / BatteryHookup / Seplos) exist but aren't verified for this form factor. FLAG.
7. **ODrive S1 + M8325s kit lead time is "stock-dependent".** Confirm stock at order time; if backordered, switch primary to Backup A immediately rather than wait.
8. **40 hours of mechanical CAD** per cell for the rack-and-pinion design is counted in neither $ nor time in this BoM — it's an engineering cost that shows up as project schedule, not spend. Worth calling out to the user.
9. **Spare-parts philosophy:** only 1 spare actuator is budgeted. If the build is >60 km from the nearest ODrive/McMaster warehouse or the timeline is <4 weeks total, consider upgrading to 2 spares (+$235).
10. **`print_bom.md` is v3 (Dodec-12, 0.34 m)**; this BoM scaled it by visual inspection to OCTA-8 (0.30 m, 8 triangular tiles). A proper OCTA-8 print BoM with CAD-measured volumes is still owed from whichever agent handles OCTA-8 mechanical. Mass/cost here are ±20%.
11. **Jetson vs Pi 5 compute decision** is not blocking — the Pi 5 is the primary and is more than sufficient for 8-dim policy inference at 200 Hz. The Jetson only matters if onboard vision/perception gets added; deferred.

---

## Sources / citations

- `docs/design_review_v1/topology.md` (OCTA-8 geometry, mass, physics anchors)
- `docs/design_review_v1/catalog_extension.md` (Agent A2 — Sensata correction, price flags)
- `src/robo_garden/data/actuators/linear.yaml` (canonical actuator specs)
- `docs/design_review/actuator_shortlist.md` (prior format template, Candidate A detail, lead-time buckets)
- `docs/design_review/print_bom.md` (scale reference for print mass/filament/cost)
- `src/robo_garden/data/materials/3d_printable.yaml` (PETG / TPU 95A / Nylon CF retail anchors)
- ODrive S1 + M8325s kit: [https://shop.odriverobotics.com/products/s1-and-m8325s-start-kit](https://shop.odriverobotics.com/products/s1-and-m8325s-start-kit)
- Sensata LAS28-53: [https://www.sensata.com/products/motors-actuators/cylindrical-housed-linear-vca-integrated-sensor-las28-53-000a-p01-12e](https://www.sensata.com/products/motors-actuators/cylindrical-housed-linear-vca-integrated-sensor-las28-53-000a-p01-12e) (WebFetch 403; RS Online cross-reference confirmed 25 mm / 266.9 N)
- StepperOnline LK60: [https://www.omc-stepperonline.com/lk60-series-ball-screw-driven-max-horizontal-vertical-payload-20kg-6kg-stroke-110mm-for-servo-motor-lk60-40dl10s3-110](https://www.omc-stepperonline.com/lk60-series-ball-screw-driven-max-horizontal-vertical-payload-20kg-6kg-stroke-110mm-for-servo-motor-lk60-40dl10s3-110) (WebFetch 403)
- BNO085: [https://www.adafruit.com/product/4754](https://www.adafruit.com/product/4754)
- Raspberry Pi 5: [https://www.raspberrypi.com/products/raspberry-pi-5/](https://www.raspberrypi.com/products/raspberry-pi-5/)
