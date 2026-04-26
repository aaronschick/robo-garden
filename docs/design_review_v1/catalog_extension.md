# Catalog extension — urchin_lite_v1 (Agent A2)

Date: 2026-04-22

This extends `src/robo_garden/data/actuators/` with linear actuators (new
family) and additional character-focused rotary entries requested for
urchin_lite_v1 and related small-character builds.

## Before / after entry counts

| File | Before | After | Added / Changed |
|---|---|---|---|
| `linear.yaml` | — (did not exist) | 6 | 6 new linear entries |
| `bldc_motors.yaml` | 4 | 7 | +robstride_03, +tmotor_gb36_1, +ipower_gm2804 |
| `hobby_servos.yaml` | 6 | 7 | +feetech_sts3215 |
| `dynamixel.yaml` | 8 | 8 | XL330-M288 + XL430-W250 specs corrected against vendor pages (no new rows) |
| **Total rotary in catalog** | **18** | **22** | +4 rotary (linear tracked separately; see schema flag) |

Loader smoke-test confirms 22 rotary entries load cleanly post-change. The
six linear entries are skipped by the loader at runtime — see schema flag
below.

## Added entries

### Linear (new file)

| id | Force (N) | Speed (m/s) | Stroke (mm) | Weight (g) | Price (USD) | Interface | Source |
|---|---|---|---|---|---|---|---|
| `odrive_s1_rack_and_pinion_ref` | 90 | 0.65 | 60 | ~250 | 235 | CAN-FD | Custom (ODrive S1 + M8325s) |
| `stepperonline_lk60_40dl10s3_110` | 196 | 0.50 | 110 | null | null | step/dir | StepperOnline |
| `sensata_las28_53_vca` | 267 | ~2.0 | 25 | null | null | analog current | Sensata BEI Kimco |
| `actuonix_l16_r_100mm_150_1` | 200 | 0.008 | 100 | 74 | 75 | PWM | Actuonix |
| `actuonix_pq12_63_1` | 45 | 0.015 | 20 | 15 | 80 | PWM | Actuonix |
| `firgelli_l12_r_50mm_100_1` | 42 | 0.012 | 50 | 34 | 70 | PWM | Actuonix |

### Rotary (added or corrected)

| id | File | Torque (N·m) | Speed (rpm) | Weight (g) | Price (USD) | Interface | Source |
|---|---|---|---|---|---|---|---|
| `feetech_sts3215` | `hobby_servos.yaml` | 1.91 | 50 | 55 | 21.99 | TTL bus | Feetech |
| `robstride_03` | `bldc_motors.yaml` | 60 (peak) | 195 | 880 | 269 | CAN | Robstride |
| `tmotor_gb36_1` | `bldc_motors.yaml` | 0.24 | 600 | 88 | 49.90 | 3-phase (external driver) | T-motor |
| `ipower_gm2804` | `bldc_motors.yaml` | 0.034 | 1600 | 42.3 | 21.99 | 3-phase (external driver) | iPower / iFlight |
| `dynamixel_xl330_m288` | `dynamixel.yaml` | 0.52 (was 0.42) | 103 | 18.0 (was 18.5) | 27.49 (was 24.00) | TTL | Robotis |
| `dynamixel_xl430_w250` | `dynamixel.yaml` | 1.4 (was 1.5) | 57 | 65.0 (was 57.2) | 27.50 (was 49.90) | TTL | Robotis |

Dynamixel corrections are from the current Robotis.us product pages — the
previous catalog values drifted from vendor specs. Price on XL430-W250
in particular was nearly 2x out of date (Robotis cut it from ~$50 to $27.50).

## Schema changes required (FLAG for Agent B3 — validator extension)

The `Actuator` dataclass at
`src/robo_garden/building/models.py` currently requires `torque_nm: float`
and `speed_rpm: float`, has no `force_n` / `speed_mps` / `stroke_mm`
fields, and has no `notes` field. Adding linear entries with those names
makes `Actuator(**entry)` raise `TypeError`.

Two concrete needs:

1. **Extend the `Actuator` dataclass** with optional
   `force_n: float | None = None`, `speed_mps: float | None = None`,
   `stroke_mm: float | None = None`, and optional `notes: str = ""`.
   Relax `torque_nm` and `speed_rpm` to `float | None`.

2. **Extend `building/validator.py`** to dispatch on `type`:
   - `type in {"servo", "bldc", "stepper"}` → require `torque_nm`,
     `speed_rpm`.
   - `type == "linear"` → require `force_n`, `speed_mps`, `stroke_mm`.

As a temporary bridge I patched `building/actuators.py::load_catalog` to:
- Drop any YAML keys not in the current dataclass field set (so `notes`,
  `force_n`, etc. don't break rotary entries that include them).
- Skip every entry with `type == "linear"` entirely, so `linear.yaml` is
  effectively documentation-only until B3's fix lands.

When B3 extends the model, remove the skip. The four new rotary entries
and the two Dynamixel corrections are already live in the loader output
(22 actuators load, verified).

## WebFetch / WebSearch issues during this run

These sources returned 403 or 404 against direct WebFetch and were
cross-confirmed via WebSearch snippets instead. Flagging so a human can
re-verify if BoM decisions ride on the exact value:

- **Actuonix product pages** (`actuonix.com/l16`, `actuonix.com/pq12`,
  `actuonix.com/l12`) — 404 on WebFetch across multiple slug attempts.
  Spec values confirmed via RobotShop listings + Amazon/SparkFun data
  summarized in WebSearch results.
- **StepperOnline product pages** — 403 on WebFetch. Spec values come
  from WebSearch snippets of StepperOnline's catalog text. The LK60
  110 mm variant's exact weight and retail price are not in the public
  snippet; both are marked `null` in the YAML with a FLAG in `notes`.
- **Sensata** product pages and the Developer Kit datasheet PDF — 403.
  Spec values come from the RS Online listing
  (`us.rs-online.com/product/sensata-bei-kimco/las28-53-000a-p01-12e/75053615`)
  which confirmed 25 mm stroke + 266.9 N peak force. Weight and
  per-unit price are not on the RS listing; marked `null`.
- **T-motor / iFlight store pages** — store.tmotor.com was reachable
  (GB36-1 confirmed: 50 KV, 0.24 Nm peak, 88 g, $49.90). shop.iflight.com
  was reachable for GM2804. Both clean.
- **Feetech `feetechrc.com`** — the product-level STS3215 page 404s;
  the Seeed listing was used instead.
- **Robstride** — `robstride.com` returned only the site header over
  WebFetch. Spec values confirmed via the rcdrone.top retailer listing
  and openelab.io comparison guide.

**Note on the LAS28-53 discrepancy with the prior shortlist**:
`docs/design_review/actuator_shortlist.md` lists LAS28-53 as 53 mm stroke
/ 60 N peak / 20 N continuous, citing the Sensata product page. RS Online
and DigiKey both list this exact model number (`LAS28-53-000A-P01-12E`)
as 25 mm stroke / 266.9 N peak. The "-53" in the model number refers to
housing length in mm, not stroke. The catalog entry here uses the
distributor-listed values. This does NOT invalidate Candidate A of the
Urchin v3 Dodec-12 shortlist (which remained the recommended design) but
it does mean Candidate B was analyzed on incorrect stroke assumptions
and needs a re-review before anyone sources 12 voice coils. FLAG.

## Files touched (absolute paths)

- `C:\Users\aaron\Documents\repositories\robo-garden\src\robo_garden\data\actuators\linear.yaml` (new)
- `C:\Users\aaron\Documents\repositories\robo-garden\src\robo_garden\data\actuators\bldc_motors.yaml` (extended +3)
- `C:\Users\aaron\Documents\repositories\robo-garden\src\robo_garden\data\actuators\hobby_servos.yaml` (extended +1)
- `C:\Users\aaron\Documents\repositories\robo-garden\src\robo_garden\data\actuators\dynamixel.yaml` (2 entries corrected, no new rows)
- `C:\Users\aaron\Documents\repositories\robo-garden\src\robo_garden\building\actuators.py` (defensive loader patch; see schema flag)
- `C:\Users\aaron\Documents\repositories\robo-garden\docs\design_review_v1\catalog_extension.md` (this doc)

Nothing committed. urchin_v3 workspace not touched.
