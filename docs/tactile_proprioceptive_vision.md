# Tactile Proprioceptive Vision

**Status:** Design notes, not yet implemented.
**Scope:** Perception extension for shelled robots where outward-facing cameras are impractical.
**Primary motivating robot:** `urchin_v2` — 42-panel voice-coil spheriball with no stable viewport and continuous tumbling.

## TL;DR

Give the urchin "vision" without adding a camera. The panels already touch
the ground when extended; instrument them so every tile becomes a pressure
sensor. Fuse the resulting 42-channel contact map with the IMU to recover a
body-frame occupancy estimate around the whole sphere. This is how an
octopus sees — not optically, but via distributed contact across the entire
body surface.

## Why optical vision is wrong for this robot

A rolling sphere covered in 42 moving panels is hostile to every camera-like
sensor:

- **No stable viewport.** The "up" direction changes several times per
  second during rolling. Any fixed-orientation camera produces motion-blurred
  output that has to be rectified against IMU pose at the cost of latency.
- **No aperture.** The surface is ~100% panel. Cutting a window between
  panels leaves a FoV that the panels themselves obscure stochastically as
  they extend/retract. Event cameras handle the blur but not the occlusion.
- **Tumble-induced attention problem.** Even with a perfect rectified image,
  the policy has to integrate frames taken at arbitrary rotations — a
  learning burden the base PPO loop is not set up for.

Meanwhile, every panel is already in physical contact with whatever the
robot is touching (ground, walls, obstacles) at exactly the moment that
contact matters. The sensor is free if we choose to read it.

## What "tactile proprioceptive vision" means

At each control tick, produce a **42-dimensional contact vector** c ∈ ℝ⁴²,
one channel per panel, where each channel encodes how hard that panel is
pushing against the world. Combined with:

- the fixed per-panel outward-normal directions (body frame, known from
  `urdf_meta.json`),
- the current IMU-derived body-to-world rotation,

this yields a **body-frame contact field**: for each of 42 directions on
the sphere, how much reaction force is coming from that direction right
now. That field *is* a low-resolution, omnidirectional, zero-latency
depth-and-stiffness sensor, calibrated against the only thing the robot
can actually act on — surfaces it's already touching.

Mathematically: if `n_i ∈ ℝ³` is panel i's outward normal in body frame
and `f_i ∈ ℝ` is its measured pushback force, the policy sees

```
contact_field(body_frame) = { (n_i, f_i) for i = 1..42 }
```

which is the natural input for a rolling robot whose decisions are
"which panels do I push, and how hard?"

## Two sensing paths

### Path A: back-EMF / current sensing on the voice coils (cheap, indirect)

Every voice-coil panel is already a motor. When a panel is commanded to
hold a target length but the environment pushes back, the coil draws
more current to maintain position. Measuring current per channel gives
an indirect but calibrated estimate of external force.

**Pros**
- Zero mechanical additions — the sensor is the actuator.
- One ADC channel per panel (42 total) plus a current-sense resistor.
- Compatible with the existing voice-coil driver architecture.

**Cons**
- Indirect: current reflects total applied force including inertial,
  friction, and back-EMF terms. Requires a motor model to subtract the
  "what I was going to draw anyway" baseline.
- Noisy near zero contact — the signal is differential against a
  model-predicted current, so low-force contacts get lost in modeling
  error.
- Couples tightly to the control strategy: if the policy drives panels
  aggressively, the proprioceptive estimate degrades.

### Path B: per-tile strain gauges or FSRs (direct, more hardware)

Mount a force-sensing resistor or thin-film strain gauge between each
panel rod and its tile. Reads the actual tile-to-rod compression directly,
independent of what the coil is doing.

**Pros**
- Clean, direct signal. Low force = low reading, high force = high
  reading, independent of commanded trajectory.
- Decouples perception from control — the policy can read contact while
  commanding arbitrary motions.
- Same sensor serves a possible future goal of closed-loop compliance
  control (touch-softness).

**Cons**
- 42 additional sensors, wiring, and ADC channels.
- Each tile becomes a slightly more complex mechanical assembly — solder
  joint or epoxy on the rod-to-tile interface has to survive repeated
  compression.
- Adds ~1g per panel at a sphere budget where mass matters
  (already carrying ~1.47kg of shell+battery+electronics).

**Recommendation:** start with Path A in sim (free — we can read joint
torques from PhysX directly). If the policy demonstrably benefits, invest
in Path B for hardware.

## Fusing with IMU for a body-frame occupancy estimate

A single tick gives a snapshot of "which panels are being pushed right
now." Fusing across a rolling window gives something stronger:

1. **Per-tick:** record `(timestamp, body_quat, c_i)` for each contact
   reading.
2. **Rolling window (e.g., last 500ms):** for each sample, transform
   panel normal `n_i` from body frame into world frame via the recorded
   quaternion. Accumulate `f_i` into a world-frame sphere-grid histogram.
3. **Output:** a direction-indexed array `occupancy_w[θ, φ]` that
   integrates "where has the world been pushing back" over the recent past.

This gives the policy a proto-map of its immediate surroundings — not from
vision, but from the history of what it has bumped into while rolling. The
IMU turns per-tick tactile snapshots into a spatial aggregate.

For wall-avoidance and navigation around irregular terrain, this is often
more useful than a camera would be: it directly measures traversability
rather than appearance.

## What it unlocks for training

- **Obs-space extension:** add `c_i` (42-D) and optionally the fused
  occupancy map (low-res, e.g., 12 bins on the sphere) to the current
  95-D observation.
- **Reward shaping candidates:**
  - Penalise unexpected high-force contacts (bumping walls hard).
  - Reward sustained ground contact during rolling (no flying).
  - Reward gentle transitions from one support panel to the next (smooth
    gait).
- **Environment interaction:** obstacles in the env become directly
  perceivable — the policy can steer *around* them by feel instead of
  needing a separately-trained visual detector.

## Simulation vs. hardware parity

The powerful advantage here is that PhysX already computes joint
contact forces exactly. In sim the 42-D contact vector is free and
noiseless. On hardware it's Path A/B above, with noise and calibration
drift. As long as we train with realistic noise injection in sim (additive
Gaussian + multiplicative scale error + occasional dropout), the
sim-to-real gap on this modality is small compared to vision, which is
the whole point.

## Where in the codebase

- **`workspace/robots/urchin_v2/urchin_v2/urchin_env_cfg.py`** — extend
  `_get_observations` to append per-panel contact forces read from
  `robot.data.applied_torque` (or equivalent PhysX contact aggregation).
  Grow `observation_space` accordingly and gate via a cfg flag so older
  BC datasets remain loadable.
- **`workspace/robots/urchin_v2/scripts/`** — add a small script that
  visualises the contact field as a coloured sphere around a rolling
  robot in Isaac Sim, for sanity-checking.
- **Curriculum hooks:** a "contact-rich" environment variant
  (plane + a few randomly-placed vertical posts) gives the policy
  something to feel before navigation goals depend on it.

## Open questions

- How much temporal integration helps vs. instantaneous contact? 100ms
  is a useful rolling period; longer windows may smear across changes in
  terrain.
- Does the policy learn to *actively palpate* — extend unused panels
  just to probe what's nearby — if the reward encourages low-force
  exploratory contact? This is the behaviour that would most closely
  mirror biological tactile perception.
- Is 42 channels enough spatial resolution, or does the sphere need to
  get its panel count up (icosahedral refinement) before contact-based
  vision becomes reliably navigable? Worth a controlled comparison once
  the base sensing path is wired.

## References

- Prescott et al., *Scholarpedia: Active touch sensing* — biological
  background on tactile perception as active rather than passive.
- TacTip / GelSight lineage — high-resolution optical tactile sensing
  (opposite tradeoff: dense per-contact-patch vision at the cost of
  complex optics per sensor).
- Octopus arm proprioception literature — distributed contact sensing
  without centralised vision, closest biological analogue to this
  robot's constraint.
