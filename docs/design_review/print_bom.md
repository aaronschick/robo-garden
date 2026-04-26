# Urchin v3 Dodec-12 — 3D-Print Bill of Materials (Agent 5)

Scope: full print list for one Dodec-12 prototype at the sim diameter of
**0.34 m**. Printer assumption: home Ender-3-class FDM (235×235×250 mm
build, bowden extruder, un-enclosed chamber, 0.4 mm nozzle). Material
densities and yields from `src/robo_garden/data/materials/3d_printable.yaml`:
PETG 1270 kg/m³ / 50 MPa, TPU 95A 1210 kg/m³ / 26 MPa, Nylon CF
1150 kg/m³ / 80 MPa.

## 1. Per-part table

Volume = as-printed plastic volume (i.e. gross × infill + shells).
Times estimated at 0.2 mm layer, 50 mm/s Ender-3 baseline, extrapolated
from ~10 cm³/hr effective extrusion rate.

| Part | Material | Qty | Vol (cm³) ea | Mass (g) ea | Print hr ea | Notes |
|---|---|---|---|---|---|---|
| Pentagonal face plate (Ø 135 mm, 5 mm ribbed) | PETG | 12 | 55 | 70 | 5.5 | Printed flat-side-down; stiffening X-ribs on interior side |
| Gasket skirt (pentagonal ring, concertina fold) | TPU 95A | 12 | 4.2 | 5.1 | 2.0 | 100% infill, 0.1 mm layer, 20 mm/s |
| Inner rib (dodec edge, 124 mm C-channel) | PETG | 30 | 5.0 | 6.3 | 0.8 | 30% gyroid; bolts to two face-plate brackets at each end |
| Actuator mount bracket (cup for linear-actuator body) | PETG | 12 | 15 | 19 | 1.5 | 100% infill, M3 heatset pockets |
| Electronics tray (180×120×25 mm dish) | PETG | 1 | 80 | 102 | 8.0 | 20% infill; bolts to 4 central ribs |
| Battery cradle (4S LiPo, 120×60×30 mm) | PETG | 1 | 40 | 51 | 4.0 | 20% infill; Velcro strap slots |
| Hemisphere cap | — | 0 | — | — | — | **N/A** — Dodec-12 is fully paneled, no caps |
| M3×5 brass heatset insert | brass | 120 | — | — | — | Not printed; $0.10 ea ≈ $12 |
| M3×10 socket-head screw | steel | 120 | — | — | — | Bulk ≈ $8 |

### Per-material subtotals

PETG:
- 12 face plates × 70 g = 840 g
- 30 ribs × 6.3 g = 189 g
- 12 brackets × 19 g = 228 g
- Electronics tray = 102 g
- Battery cradle = 51 g
- **Total PETG ≈ 1410 g**

TPU 95A:
- 12 skirts × 5.1 g = **61 g** (round up to 100 g roll allowance — TPU
  purge + priming is lossy)

Nylon CF: **0 g** in baseline build. See risk call-outs for the upgrade
path.

### Cost totals (retail filament)

- PETG: 1.41 kg × $25/kg = **$35**
- TPU 95A: 0.10 kg × $40/kg = **$4**
- Heatsets + screws: **$20**
- **Filament + fastener total: ≈ $60**

Actuators, BLDCs, electronics, and the optional AL6061 equator ring are
**not** in this BoM — those live in Agents 2/3/8 reports.

### Print-hour totals (single Ender-3)

Face plates 66 hr + skirts 24 hr + ribs 24 hr + brackets 18 hr + tray 8 hr
+ cradle 4 hr = **~144 printer-hours** = 6 wall-clock days of continuous
printing. Realistic with failures/restarts: **8–10 days**. Parallelizing
across two printers halves this; using an E3 with a larger 0.6 mm nozzle
on the structural parts (ribs, brackets) cuts ~25% more.

## 2. Strength budget — rolling-impact check

Impact spec from the brief: m = 2.5 kg, Δv = 0.5 m/s, contact time
t ≈ 20 ms on a compliant surface.

```
F_peak = m · Δv / t = 2.5 · 0.5 / 0.020 = 62.5 N
```

Assume the load is taken by **one** face plate (worst case — single-tile
strike). Pentagonal plate edge a = 124 mm, thickness h = 5 mm, treat as
a simply-supported plate with the force spread over a 30 mm patch (TPU
gasket + tile flex).

Plate bending moment (central load, span ~100 mm):
```
M ≈ F · L / 4 = 62.5 · 0.100 / 4 = 1.56 N·m
```

Section modulus for 100 mm effective width × 5 mm:
```
S = b·h² / 6 = 0.100 · 0.005² / 6 = 4.17e-7 m³
```

Peak bending stress:
```
σ = M / S = 1.56 / 4.17e-7 = 3.74 MPa
```

PETG yield = 50 MPa → **safety factor ≈ 13×**. Comfortable. With FDM
knockdown (layer adhesion ~60% of bulk), effective yield ≈ 30 MPa and
SF ≈ 8×. Still comfortable. The face plates will not be the weak link.

Weak link candidates (for Agent 4 to double-check):
- **Rib-to-plate bolt bosses**: a 62 N shear through a single M3 boss
  with 3 mm PETG wall is ~15 MPa — SF 3×, tight.
- **TPU gasket tear**: 62 N across a 1.5 mm × 30 mm TPU strip is 1.4 MPa
  — TPU 95A yields at 26 MPa, SF 19×. Fine.

## 3. Print strategy

**Layer heights**:
- 0.2 mm generic for all PETG structural parts (face plates, ribs,
  brackets, tray, cradle).
- **0.1 mm** for the TPU gasket skirts — concertina fold needs fine
  feature resolution, and TPU prints slower anyway.

**Infill**:
- Face plates: 20% gyroid + 4 top/bottom solid layers + internal X-ribs
  modeled into the CAD (not slicer infill). Gyroid beats cubic for
  impact dispersion.
- Ribs: 30% gyroid.
- Actuator brackets: **100% solid** (concentrated bolt loads).
- Electronics tray + battery cradle: 20% cubic.
- TPU skirts: **100%** (flex parts must be solid, infill voids tear).

**Orientation & supports**:
- Face plates: flat side down, ribs printing upward → **no supports**.
- Ribs: on edge, C-channel opening sideways → needs **minimal tree
  supports** inside the channel if closed-ended; open C avoids them.
- Brackets: actuator-cup opening facing up → no supports (shallow
  overhang).
- Electronics tray: dish open-side up → no supports.
- Gasket skirts: pleated side up, flat side to bed → no supports; TPU
  cannot tolerate supports (can't cleanly remove).
- Battery cradle: strap slots are 45° overhangs → no supports needed.

**Flag**: none of the baseline parts require supports. This is a
deliberate CAD-geometry choice — minimizes post-processing and
preserves dimensional accuracy on a stock Ender-3.

**Parallelization**: on a single Ender-3 at 144 printer-hours, plan a
**batch queue**: nights = 5.5 hr face plates (one at a time), days =
multi-part plates of ribs (4 per plate × ~3 hr). Gasket skirts and
brackets go in opportunistic gaps. No part exceeds the 235×235 mm bed.

## 4. Risk call-outs

1. **TPU on un-enclosed bowden (Ender-3 stock)**: marginal. TPU 95A
   prints OK on bowden if retraction ≤ 2 mm and speed ≤ 20 mm/s, but
   stringing and under-extrusion are common. **Mitigation**: direct-drive
   conversion ($40 Microswiss/Sprite clone) before printing skirts.
   **Fallback**: replace TPU skirts with silicone-sheet gaskets cut to
   a 3D-printed template — $15 for a 300×300 mm sheet, easier to seal.
2. **Dry filament**: PETG tolerates ambient humidity; TPU absolutely
   does not. Keep TPU in a dry box (<20% RH). If surface bubbles
   appear, re-dry at 55 °C for 6 hr.
3. **Nylon CF upgrade path**: if impact testing shows PETG bosses
   cracking (Agent 4 stress check), swap the **actuator brackets** and
   **rib-to-plate inserts** to Nylon CF (80 MPa yield, 6 GPa modulus).
   Adds ~$25 of filament and requires a hardened nozzle ($12) + dry box.
   Do not swap the face plates to Nylon CF — the TPU gasket adhesion
   is untested, and nylon's moisture sensitivity kills the shell seal.
4. **Build volume**: no single part exceeds 235×235×250 mm. The inner
   dodec cage is explicitly **not** printed as one piece — it is
   assembled from 30 individual 124 mm ribs bolted at the vertices.
   This is both a printability and a repairability win.
5. **Equator AL6061 ring** (Agent 1's optional reinforcement): not a
   printed part; call out to Agent 8 for machining cost. A 0.34 m dia ×
   3 mm × 20 mm aluminum ring is ~$40 water-jet cut.

## 5. Open questions

- **Agent 4 (confirmed geometry)**: final face-plate edge length and
  rib-boss wall thickness — the 62 N impact number above assumes 3 mm
  walls; if final CAD drops to 2 mm, SF goes below 2 and we need Nylon
  CF bosses or aluminum inserts.
- **Agent 7 (electronics tray volume)**: is 180×120×25 mm enough for
  the Jetson/Pi + IMU + 12 actuator drivers + wiring? If not, the tray
  mass scales roughly linearly and the print time with it.
- **Agent 8 (total cost)**: this BoM contributes ~$60 in printed
  parts; combine with actuators (Agent 2), electronics (Agent 7), and
  machined AL ring for the system total.
- **Printer assumption**: I assumed Ender-3 (235×235×250 mm,
  un-enclosed). Confirm with user — if they have a Bambu X1C or a
  Prusa MK4/XL, enclosed-chamber and faster-extrusion assumptions
  flip, total wall-clock drops to ~2–3 days, and Nylon CF becomes the
  default structural material (no dry-box retrofit needed).
