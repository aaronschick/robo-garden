# Urchin-lite v1 — Topology Candidates (Agent A1)

## Context & constraints

Fresh design pass with a tighter actuator budget (N=8) and the same core
locomotion story as urchin_v3: **dynamic shape-shifting rolling** —
rolling-only propulsion, no legs, posability = fast real-time surface
deformation. The 8 "primary" actuators *are* the posability channel;
there is no separate static-pose servo set.

Target: cleanest sim-to-real path from the urchin_v3 Phase-1 approved
oracle (contact-dipole rolling via SH-9 → 42-panel decode) while
reducing mechanical count by >4× and improving character presence.

Physics anchors (per-candidate tables refine these):
- urchin_v3 sim: R = 0.17 m (0.34 m Ø), m ≈ 2.5 kg, I ≈ (2/3)mR² ≈
  0.048 kg·m². Panel travel 0–60 mm, per-panel effort 60 N, commanded
  velocity ≤ 1.5 m/s (Agent 3 predicted 0.4 m/s is the actual need).
- SH-9 basis (l ≤ 2) is well-conditioned for N ≥ 9 point-samples.
  Dropping to N=8 is **on the edge** — 8 panels give 8 equations for
  9 unknowns, so the full SH-9 → panel map is rank-deficient by 1.
  All three candidates below have to confront this; each does so
  differently.

Actuator catalog reality check (`src/robo_garden/data/actuators/*.yaml`):
- **Rotary options are rich**: Dynamixel X-series, hobby servos
  (MG996R, DS3218, LD-27MG), and BLDC (ODrive S1 + D5065, mjbots
  qdd100, MyActuator RMD-X8).
- **Linear options are absent.** Every linear mechanism in this doc
  needs either (a) a rotary-to-linear conversion (lead-screw, rack &
  pinion, belt) built from a catalog rotary + custom mechanical, or
  (b) a new catalog entry for voice coils / ball-screw cartridges.
  Flagging for Agent A2: add a `linear_actuators.yaml` catalog
  (Sensata VCAs, Actuonix L16/PQ12 for low-speed use, StepperOnline
  LK60 ball-screw stages, ODrive + rack-and-pinion reference designs).

---

## Candidate 1 — OCTA-8: octahedral 8-face prismatic ("shape-shifting cube-corner")

### Geometry

A regular octahedron (8 equilateral-triangle faces, 6 vertices, 12
edges) inscribed in a sphere. Each triangular face is a rigid PETG
tile riding on a single prismatic actuator normal to the face.
Imagine a die with 8 triangular sides, each side breathing in and out
independently.

```
          /\                face count: 8
         /  \               vertex count: 6 (4 equatorial + N+S pole)
        /    \              edge count: 12
       /------\             each face: 1 prismatic panel
      /\      /\            inscribed-sphere radius at flat-center:
     /  \    /  \             r_face = R / sqrt(3)
    /    \  /    \
   /      \/      \         gasket pattern: 12 edges, each ~R edge length
   --------==-------        all 8 panels geometrically identical
   \      /\      /         (highest symmetry of the three candidates)
    \    /  \    /
     \  /    \  /
      \/      \/
```

### Actuator layout

- **8 linear prismatic panels**, one per face, stroke 0–60 mm.
- Rotary-to-linear: 8× ODrive S1 + M8325s + belt-rack (Agent 2's
  "Candidate A" from the prior review) or 8× StepperOnline LK60
  ball-screw cartridges (Agent 2's "Candidate C"). No new rotary
  actuator types needed — the bldc_motors.yaml line already covers it;
  **linear cartridge needs a catalog entry** (Agent A2).
- Control mapping: SH-9 → 8-panel is rank-deficient by 1. Either (a)
  drop to SH-basis order l ≤ 2 with the l=0 breathing mode *excluded*
  (leaves 8 modes = exactly 8 panels, well-posed), or (b) keep SH-9
  and accept a 1-D nullspace the policy can't drive. Option (a) is
  cleaner; the l=0 pure-breathing mode doesn't contribute to rolling
  anyway.

### Size recommendation

**0.30 m diameter.** Smaller than urchin_v3 (0.34 m) to keep triangle
edges reasonable (~250 mm at 0.30 m) and to reduce the actuator
travel-per-CoM-shift ratio. Below 0.24 m the interior volume for
electronics gets tight; above 0.34 m the triangular tiles become
floppy at 3 mm PETG.

### Physics sanity

- I ≈ (2/3) × 2.3 kg × 0.15² ≈ 0.035 kg·m² (slightly less than v3
  because smaller, similar mass).
- CoM sits at geometric center with panels balanced; rolling is
  **contact-dipole** identical to urchin_v3 — the rear-contact panel
  extrudes to push against ground, opposite face retracts to bias
  CoM. Same 9-D control conceptually.
- **Passively stable on 4 faces at a time** (an octahedron *rests on
  a face*, not a vertex — it has 8 stable static orientations). This
  is great: the robot can sit still without active balance and
  doesn't wobble when idle.
- Static-tip threshold: octahedron tips from face to face when CoM
  crosses an edge. Edge half-distance from center ≈ R/√3 ≈ 87 mm at
  R=0.15 m. Panels give ~60 mm of CoM shift at m_panel/m_total ≈
  0.025/2.3 × 8 = 8.7% → max CoM excursion ≈ 5.2 mm. **That's
  well below the 87 mm tip threshold** — rolling is NOT by static tipping;
  it is pure dynamic contact-dipole push-off exactly like urchin_v3.

### Sim-to-real risk

| Risk | Sev | Mitigation |
|---|---|---|
| 8 faces = only 8 oracle samples of SH field; coarser than v3 (42) | M | Constrain policy action to l ≤ 2 w/o l=0; reproject urchin_v3 oracle through Y_8 and verify contact-dipole still produces progress (sim smoke before committing) |
| Octahedron rests on flat face → may "park" instead of roll during gait pauses | M | Reward shaping: penalize zero-velocity on rolling episodes; domain-randomize initial orientation (vertex-up, edge-up) |
| Large triangular panels (~250 mm edge) flex under contact load | M | Rib-stiffen panels with CF strip; or go smaller (0.24 m → 200 mm edges, stiff at 3 mm PETG) |
| Linear actuator not in catalog | H | **Agent A2 must add linear_actuators.yaml** — block dependency for any real BoM |
| Gasket topology: 12 edges × 3-fold vertices = same TPU accordion pattern as v3 | L | Directly reuses urchin_v3 collision audit (Agent 4) patterns |
| 8-panel SH decode rank-1 deficient | L | Well-understood in linear algebra; pinv handles it; l=0 exclusion is clean |

### Retraining effort vs urchin_v3 Phase-1

**Medium warm-start.** Action space shrinks (8 panels vs 42, but same
9-D SH policy interface upstream — same observation layout, same
reward structure). Contact-dipole oracle reprojects through Y_8 the
same way Agent 1 described for Y_12. Estimate: **0.5–1.5 M steps**
light-tune from the Phase-1 checkpoint, reward terms unchanged except
`panel_l1_reg` and `panel_span` divisors (42 → 8).

### Expressiveness / character

- **"Puffing" gesture** — 4 alternating faces extrude while 4 retract:
  reads as *breathing* or a chest-puffing display. 8-fold symmetry
  makes this pose recognizable and cute.
- **"Crouch / pounce"** — retract the two poles, extrude the 4
  equatorial faces: robot squashes flat, then "springs" upward by
  reversing. Stronger telegraph than v3's 42-panel analog.
- **"Shiver / excited"** — 1–2 Hz alternating extrude/retract across
  random face pairs: the robot looks like a stressed-out creature.
- Fewer panels = each gesture reads clearer. The v3 42-panel
  "shimmer" is pretty but visually busy; 8 panels let a viewer *see*
  the robot thinking.

---

## Candidate 2 — TET-4×2: tetrahedral shell with doubled-axis actuators ("punchy tetra")

### Geometry

A regular tetrahedron (4 triangular faces) inscribed in a sphere, but
each face is split into two hinged half-tiles along its median — so 8
tiles, 8 actuators, but only 4 face-normals. Each half-tile rotates
about a hinge at the face edge, driven by a rotary actuator at the
hinge, effectively *flapping* outward or inward.

```
      .
     /|\
    / | \              4 triangular faces
   /  |  \             each face split by its median -> 2 flap-tiles
  / flap \             hinge runs face-edge to face-center
 /   ||   \            actuator: 1 rotary servo at hinge axis
 ---------            8 tiles total, 8 rotary actuators
 inside view: hinge    tile flap angle: -20 deg (inward) .. +40 deg
 along median line     (outward) -> surface deformation without linear stages
```

### Actuator layout

- **8 rotary servos at hinge axes.** Each servo rotates one half-tile
  about its shared-edge hinge. Surface deformation = hinge flap
  instead of panel extrude/retract.
- Catalog-native: **8× Dynamixel XL430-W250-T** or **8× DS3218**
  (hobby servo), both already in `dynamixel.yaml` / `hobby_servos.yaml`.
  No new actuator types needed. **This is the only candidate that's
  BoM-buildable from catalog today.**
- Torque requirement: half-tile mass ~0.15 kg at 100 mm arm → static
  torque ~0.15 N·m, well within XL430 (1.5 N·m stall) or DS3218
  (20 kg·cm = 2 N·m).
- Control mapping: 8 hinge angles map to a natural 8-D action space.
  Project SH-9 → tile-flap space via a map similar to urchin_v3's
  SH → panel decoder, but with *flap-angle* semantics instead of
  linear displacement.

### Size recommendation

**0.34 m diameter.** Matches urchin_v3 sim scale to preserve
policy-reuse geometry on the face-normals and to keep panel edges at
~290 mm (structurally sane for 3 mm PETG with CF rib). Character
presence is also strong at this size — it fills a dining-table corner
without being oppressive.

### Physics sanity

- I ≈ (2/3) × 2.2 kg × 0.17² ≈ 0.042 kg·m².
- Tetrahedron is **passively stable on 4 faces** (every orientation
  eventually settles face-down). Rolling mechanism is a hybrid:
  hinge-flaps on the contact face tilt the CoM (contact-dipole in a
  different parameterization), while hinge-flaps on the opposite
  face provide reaction-mass kicks.
- Tetrahedron has the *lowest* face count of any Platonic solid —
  rolling happens by **tipping over an edge** rather than the
  smooth contact-dipole slide. More punchy/chunky motion than v3.
- Each tip is a ~70° rotation (tetra dihedral angle ≈ 70.5°). That's
  a large discrete motion — reads as a confident *plop* not a
  continuous wobble.

### Sim-to-real risk

| Risk | Sev | Mitigation |
|---|---|---|
| Rolling is discrete-tip not continuous — v3 policy does NOT transfer | H | Accept full retrain; use tetra-specific reward (tip-count + heading) |
| Hinge seals / gaskets at face medians must survive large angular sweeps (60° excursion) | M | Fabric-reinforced silicone bellows at hinges, not TPU accordion |
| Large tip means CoM height oscillates ~30 mm per tip — wear on sensor mount / gimbal | M | Gimbal dampers + firmware low-pass on perception feed |
| Only 4 contact faces = anisotropic friction; rolling is lumpy on soft surfaces | M | Rubberize tile edges; accept that carpet behavior differs from hardwood |
| Catalog-only BoM (good); entry cost ~$500 for 8× XL430 vs $1,900 for linear cartridges | L (+) | Reduces BoM 4× vs Candidate 1 |
| Tetrahedron has large exposed face area; structural stiffness in PETG is the real engineering limit | M | 5 mm PETG or 3 mm nylon-CF; ribbed underside |

### Retraining effort vs urchin_v3 Phase-1

**Full retrain.** Action space semantics (angles, not displacements),
contact mechanics (discrete edge-tips, not shimmering contact-dipole),
and observation layout (8 hinge angles + 8 velocities) are all
fundamentally different from v3. Estimate: **3–8 M steps from
scratch**. Phase-1 curriculum structure (warm-start from oracle)
still applies — write a new hinge-flap oracle that tips the tetra in
a commanded direction; PPO should converge in ~2 GPU-days.

### Expressiveness / character

- **"Pounce" motion** — every tip is a discrete, deliberate-looking
  commit. Reads as an intentional creature action, not wandering
  drift. Probably the highest-character candidate on raw motion
  quality.
- **"Puff cheeks"** — all 8 flaps extrude outward at once: chunky,
  toad-like visual.
- **"Flinch"** — rapid inward retract of 2 adjacent flaps on one face
  (looks like recoiling from a touch).
- **Limitation**: between tips, the robot is just sitting on a face.
  Idle animation is quieter than v3's continuous shimmer.

---

## Candidate 3 — HEX-6+PUCK-2: hex-equator panels + internal 2-axis puck ("urchin-core + smart gimbal")

### Geometry

A pill-shape (oblate spheroid): 6 rigid panels arranged around a
hexagonal equator, 2 fixed polar caps (top/bottom). Inside: a
2-axis gimbal with a heavy internal puck (LiPo + electronics + 300 g
tungsten weight) for low-frequency CoM bias, hanging from the
internal ring.

```
  outside (hex-6 equator)         inside (gimbal + puck)
      _,--top cap--._
     /               \              __,--ring--.__
    /                 \            /   ,-puck-.   \
   |  panel  panel     |           |  |   *   |   |    2-axis BLDC gimbal
   | 1       2         |           |   '--,--'    |    0.8 kg puck on
   |  panel       p3   |           |     arm      |    0.10 m arm
   | 6           |     |           |              |
   |  p5    p4   |     |            \,__  __ __,_/
    \                 /               BLDC x 2 = 2 actuators
     \_,--bot cap--,_/             (mounted in crossed-yoke)
```

- 6 linear panels on the equator (stroke 0–50 mm each).
- 2 rotary gimbal motors (pitch + yaw of the internal puck).
- 2 fixed polar caps (no actuators, just smooth hemisphere tops).
- Total: **6 linear + 2 rotary = 8 primary actuators.**

### Actuator layout

- 6× voice-coil or ball-screw linear stages on equator (**needs
  linear_actuators.yaml**, Agent A2).
- 2× BLDC gimbal motors: **MyActuator RMD-X8** or **ODrive S1 + D5065**
  from catalog. These are overkill on torque but right-size on
  backlash (gimbal needs ≤0.1° backlash for smooth pendulum steering).
- Control mapping: two separate channels.
  - **Low-freq channel**: 2-D gimbal (pitch, yaw) biases CoM by up to
    ~30 mm — primary rolling authority.
  - **High-freq channel**: 6-panel equator shape-shift for
    "personality" gestures and terrain conformance. Project a
    reduced SH-6 basis (l ≤ 1 + l=2 m=0 mode = 6 modes) onto the 6
    panels; exact square map.
- Total action dim = 8 (2 gimbal + 6 SH-reduced panel).

### Size recommendation

**0.34 m equatorial diameter × 0.28 m polar height** (oblate pill).
The pill shape is more characterful than a sphere and gives a
natural "front/back" for the gimbal-heading to track. Polar caps
carry the perception head (top) and optional charging pad (bottom).

### Physics sanity

- m ≈ 2.5 kg total (shell 1.4 kg + puck 0.8 kg + panels/cabling 0.3 kg).
- Puck at 0.10 m arm, 0.8 kg → CoM offset at full tilt =
  0.8 × 0.10 / 2.5 = **32 mm**. That's >> the 5–10 mm contact patch,
  so **static-tipping roll works by itself** even with panels frozen.
- Rolling torque from gimbal alone: 0.8 × 9.81 × 0.10 = 0.78 N·m
  peak → α = 0.78 / 0.042 ≈ 18 rad/s² → 0 → 0.5 m/s in ~0.2 s.
  Very responsive.
- Panels add high-frequency disturbance & "character" but aren't
  load-bearing for locomotion. This is a **robust dual-channel
  design**: even if the panel subsystem fails, the gimbal still
  rolls the robot.
- Oblate pill rolls on a great-circle path (not arbitrary direction);
  steering comes from gimbal-yaw.

### Sim-to-real risk

| Risk | Sev | Mitigation |
|---|---|---|
| Dual-channel policy must coordinate bandwidth regimes | M | Phase curriculum: train gimbal alone first, freeze, then unfreeze panels as decoration |
| CoM offset from gimbal fights panel push direction | M | Precompute panel decode to *reinforce* gimbal tilt (panels extrude on the rear hemisphere during gimbal-forward commit) |
| Pill shape only rolls along one axis → can't pirouette in place | M | Use gimbal-yaw for turning while stationary (reaction from yaw-rate → shell counter-rotation) |
| 6-panel SH decode only captures l ≤ 1 — loses higher-order character modes from v3 | L | Accept; this robot's personality is in the gimbal, not the skin |
| Gimbal mass + bearings inside 0.34 m shell is tight | M | Use frameless BLDCs (ODrive + D5065 rotor-only); 3D-printed yoke |
| Linear actuators still not in catalog | H | **Agent A2 dependency** (same as Candidate 1) |

### Retraining effort vs urchin_v3 Phase-1

**Full retrain.** Action space (8-D with mixed gimbal + panel
semantics) and dynamics (pendulum-driven rolling, not
contact-dipole) are fundamentally different from v3. Expected: **2–4
M steps** from scratch. *However*, sphero-style gimbal-roll is a
well-studied MDP; PPO convergence is fast, and the final motion
quality will likely exceed v3 because rolling is no longer a sparse
contact-dipole event but a dense torque signal.

### Expressiveness / character

- **Head-tilt** — gimbal yaws toward a commanded point: the robot
  looks like it's *noticing* something. This is the killer expressive
  move no v3-style design can match.
- **"Nope" head-shake** — yaw ±30° rapidly: an obvious dissent gesture.
- **Panel "breathing"** — 6 panels cycle at 0.5 Hz while idle: robot
  reads as alive-and-breathing even when stationary.
- **"Lunge"** — gimbal commits hard while panels extrude on the rear
  hemisphere: a very telegraphed attack/forward-commit.
- **Trade-off**: panel-skin gestures are less rich than v3's 42-panel
  shimmer (fewer degrees of surface freedom), but gimbal-based head
  gestures more than compensate for audience readability.

---

## Recommendation

**Candidate 1 (OCTA-8) is the pick.**

Rationale, in priority order:

1. **Closest sim-to-real path from urchin_v3 Phase-1.** The rolling
   mechanism is identical — contact-dipole via SH-decoded panel
   extrude/retract. The policy warm-starts from the Phase-1
   checkpoint with only a light-tune (0.5–1.5 M steps vs 2–8 M from
   scratch for candidates 2 and 3). This is the single largest
   multiplier on calendar time; it dominates.
2. **Dynamic shape-shifting posability is preserved.** Candidate 2
   loses continuous shape-shift (discrete tips instead). Candidate 3
   reduces shape-shift to 6 panels at lower SH order. OCTA-8 keeps
   8 prismatic panels covering the l ≤ 2 (minus l=0) SH modes — the
   "shimmering blob" character survives.
3. **Geometric simplicity**. 8 identical cells, highest-symmetry
   topology of the three. Print / fab cost per cell is lowest;
   assembly is uniform. Each actuator cell is the same design
   repeated 8 times.
4. **Passively stable on 8 face orientations**. Idle robot sits
   quietly face-down; doesn't need active balance to not fall over.
   (Tetra has this too; sphere-ish candidates don't.)
5. **Smallest actuator count that preserves well-conditioned SH
   decode** at l ≤ 2 without the l=0 mode. Below 8, SH-9 decode goes
   rank-deficient by more than 1 and character expressiveness
   suffers noticeably.

**Trade-offs acknowledged:**

- OCTA-8 *requires* linear_actuators.yaml to be added — this is a
  blocking dependency on Agent A2. If A2 can't deliver a credible
  catalog entry, **fall back to Candidate 2 (TET-4×2)** which is
  catalog-native with existing Dynamixel / hobby-servo entries and
  builds for ~$500 in actuators vs ~$1,900+ for OCTA-8.
- OCTA-8 is 0.30 m vs v3's 0.34 m. Policy reprojection through Y_8
  on a 0.30 m shell needs one explicit verification step (project a
  recorded v3 episode through the smaller Y_8 basis + smaller
  radius; confirm contact-dipole still yields forward progress in
  sim) before committing to print/PO.
- Candidate 3 (HEX-6+PUCK-2) has the strongest raw character story
  (head-tilt gimbal is unique and audience-obvious). If the product
  criterion shifts from "sim-to-real fidelity" to "expressiveness
  first," recommend C3 instead. **Decision criterion for the user:**
  if your goal over the next 4 weeks is "first rolling trial in
  hardware with a warm-started v3 policy," pick OCTA-8; if it's
  "most compelling demo creature," pick HEX-6+PUCK-2 and accept a
  full retrain.

The three candidates are genuinely close on overall merit — pick on
the decision criterion above, not on my ranking.

---

## Flag for Agent A2

OCTA-8 (primary recommendation) and HEX-6+PUCK-2 (third candidate)
both require a **linear actuator catalog**. Requested entries:

- Voice coil: Sensata BEI Kimco LAS28-53 (60 N peak, 53 mm stroke).
- Ball-screw BLDC cartridge: StepperOnline LK60-series (200+ N,
  400 mm/s, 60 mm stroke).
- Custom rack-and-pinion reference: ODrive S1 + M8325s + 3D-printed
  rack (fastest, 60 N peak, 650 mm/s).
- Low-speed hobby: Actuonix L16-class (disqualified for main drive
  but useful for slow posing if we add a stiller-robot variant).

TET-4×2 (Candidate 2) is catalog-clean as-is and can proceed
immediately if the user prefers to de-risk the actuator-catalog path.
