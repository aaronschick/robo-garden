# Urchin Lite v1 — Sim-to-Real Fidelity Gaps (Agent A3)

Scope: the top 8 simulator-vs-physical-robot fidelity gaps that would hurt
first-roll transfer of a rolling-only, shape-shifting urchin_lite_v1 with
N = 8 linear actuators on a single ~0.3 m shell. All evidence is cited
against the urchin_v3 code we're inheriting, the Dodec-12 feasibility
review, and — where relevant — the shared training infra.

Severity key:
- **H** = likely to dominate first-roll failure mode; must be in from commit 1.
- **M** = will shave 10-30% off real-world performance or create
  wobbles/creep. Can be added after the first roll but before any serious
  tuning.
- **L** = real effect but either small or papered over by other gaps.

Gaps are ranked by severity (H's first), with severity summary at the end.

---

## 1. Zero actuator delay (command → force response)

**Gap.** The env commands a joint-position target with `set_joint_position_target`
on every 60 Hz env tick and Isaac Lab's `ImplicitActuatorCfg` PD responds in
the *same* 240 Hz PhysX sub-step — zero latency between "policy outputs
action" and "motor produces force". A real ODrive S1 + BLDC (Agent 2
Candidate A) has FOC loop + command latency in the 2–10 ms range; a
StepperOnline LK60 ball-screw (Candidate C) has electromechanical rise time
closer to 20–40 ms at 60 N loads; cheap DC lead-screws run 50–100 ms.

**Evidence.**
- `workspace/robots/urchin_v3/urchin_v3/urchin_v3_cfg.py:100-123` —
  `ImplicitActuatorCfg` has no `delay`, `response_time`, or first-order
  filter field. PD is evaluated directly against the commanded target.
- `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:810-812` —
  `robot.set_joint_position_target(self._smoothed_targets, ...)` is the
  only outbound control path; it hits PhysX this same frame.
- The only low-pass in the system is `target_lpf_alpha=0.30`
  (`urchin_env_cfg.py:290`), which is a policy-side smoother to prevent
  panel chatter, **not** an actuator-side delay model — it runs *before*
  the PhysX step, so the post-filter target still arrives instantly.

**Concrete sim-side fix.** Add a first-order command delay at
the env-policy boundary, *after* the LPF but *before*
`set_joint_position_target`. One-pole IIR in `_pre_physics_step`:

```python
tau_ms = cfg.actuator_delay_ms   # new field, default 20 ms, DR range [5, 40]
alpha_delay = cfg.sim.dt * cfg.decimation / (tau_ms * 1e-3)
self._delayed_targets = (
    alpha_delay * self._smoothed_targets
    + (1.0 - alpha_delay) * self._delayed_targets
)
robot.set_joint_position_target(self._delayed_targets, ...)
```

Goes in `urchin_env_cfg.py` _pre_physics_step around line 810. Randomize
`tau_ms` per-episode in `_reset_idx` over [5, 40] ms so the policy learns
to tolerate the actual envelope Agent 2's shortlist spans.

**Severity: H.** A learned rolling gait that depends on crisp
sub-200 ms contact impulses (cf. the velocity-clamp doc's "rolling cycle
needs sub-200 ms contact impulses to beat the spring's settling") will
phase-shift its push timing by 20–40 ms on the real hardware — that's
10–20% of the contact-window duration, and the published urchin_v3
iteration log at `urchin_v3_cfg.py:38-43` already notes that *under-powered*
pushes stall in a "Weeble tilt". A delay gap of this magnitude on the first
roll is the single most likely reason the robot just sits there rocking.

---

## 2. No stiction / static-vs-dynamic friction at the shell-ground contact

**Gap.** The env's ground is `TerrainImporterCfg(terrain_type="plane")` with
PhysX defaults. PhysX defaults fold static and dynamic friction into a
single coefficient (typically ~0.5) with no separate stiction breakaway —
there is no force threshold the robot has to overcome before the contact
starts sliding. Real surfaces (carpet, hardwood, tile — the three surfaces
listed as likely first-roll venues in the consolidated review §7) have
static/dynamic friction gaps of 1.3–2.5× *and* a velocity-dependent
breakaway, which is exactly what stalls a marginal rolling gait at the
start of a push.

**Evidence.**
- `workspace/robots/urchin_v3/urchin_v3/urchin_env_cfg.py:113-116` —
  `TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")`.
  No `physics_material` with `static_friction` / `dynamic_friction` split.
- `docs/design_review/consolidated_assessment.md:112` §7 item 4 flags
  "Sim-to-real contact model. MuJoCo soft contact vs. real TPU gasket +
  carpet has no measured correspondence for this geometry. Expected gap:
  10–30% forward-progress loss on first rolling trial."

**Concrete sim-side fix.** Two parts:

1. Attach an explicit `PhysicsMaterialCfg` to the ground and the shell
   panels in `UrchinSceneCfg`, with independent static/dynamic friction.
   Default: `static=0.9, dynamic=0.6`.
2. Domain-randomize per-episode in `_reset_idx`:
   `static ~ U(0.5, 1.2)`, `dynamic ~ U(0.3, 0.9)`, enforcing
   `static >= dynamic + 0.1`. This covers the carpet / hardwood / tile
   triangle directly.

Goes in `urchin_env_cfg.py` `_setup_scene` (material attach) and
`_reset_idx` (per-episode sampling).

**Severity: H.** A rolling robot that learned in a no-stiction sim will
over-command at push initiation (doesn't need to, because sim slides
immediately) and then wiggle when it encounters a real stiction floor.
Combined with gap #1 (delay), a policy will under-roll by a visible margin
on tile/hardwood; on carpet it may not roll at all.

---

## 3. Hard torque/force saturation not matched to real motor curve

**Gap.** `ImplicitActuatorCfg.effort_limit=60.0` is a hard clip — the sim
can deliver 60 N right up to the velocity limit and 60 N at zero velocity
equally. Real BLDC + screw or rack drives have a **force-vs-velocity**
envelope: continuous force falls off as RPM rises, and stall force is
available only briefly before thermal derate kicks in. Candidate B (Sensata
VCA) is explicit: "60 N peak / **20 N continuous**" — a 3× gap. Even the
primary ODrive S1 build has a phase-current-limited peak vs continuous
envelope that the sim ignores.

**Evidence.**
- `urchin_v3_cfg.py:109` — `effort_limit=60.0` as a scalar; no
  velocity-dependent curve, no continuous-vs-peak split.
- `docs/design_review/actuator_shortlist.md:44` — Candidate B row:
  "60 N peak / 20 N continuous". Candidate C row notes the ball-screw
  is at the 400 mm/s velocity ceiling, implying zero torque headroom
  there.
- `docs/design_review/consolidated_assessment.md:74` — risk row
  "Agent 3 clamp sim says gait needs > 0.4 m/s" notes ball-screw
  "caps at 400 mm/s with zero headroom" — the sim doesn't model that
  roll-off.

**Concrete sim-side fix.** Replace the scalar `effort_limit` with a
velocity-dependent cap in a custom wrapper around
`set_joint_position_target`. Simplest credible model: a piecewise-linear
F_max(v) = F_stall at |v| < v_knee, decaying linearly to F_stall * 0.33 at
|v| = velocity_limit. Applied by clipping the effort PhysX produces against
this envelope in a post-step hook (or approximated by lowering
`effort_limit` to 25 N = continuous rating and letting the policy occasionally
peak to transient 60 N via a `peak_boost` schedule — simpler but less
faithful).

Randomize `F_continuous / F_peak ratio ~ U(0.30, 0.55)` per episode to span
the Candidate-B-through-A envelope.

Goes in a new `urchin_v3_cfg.py` custom `ActuatorCfg` subclass, or a
post-physics-step clamp in `urchin_env_cfg.py`.

**Severity: H.** Our rolling gait is explicitly designed around transient
peak pushes (the retune note at `urchin_v3_cfg.py:104-108` specifically
raises `effort_limit` from 20 N to 60 N "so panels can deliver real transient
impulses"). If the real actuator can only sustain 1/3 of that for more than
~100 ms, the policy will either thermal-derate itself into stalling or the
under-built hardware will just melt through its duty cycle. Must be in sim
*before* any policy is trained for real transfer.

---

## 4. IMU bias, noise, and band-limit not modeled

**Gap.** `ImuCfg(prim_path="...")` returns Isaac's ground-truth IMU: noiseless
`lin_acc_b`, noiseless `ang_vel_b`, noiseless `projected_gravity_b`, all at
240 Hz with zero latency. A BNO085 (the sensor package's pick — see
`sensor_package.md`) has ~0.01 rad/s gyro noise floor, 0.05-0.10 m/s²
accel noise, ~1° tilt drift over minutes, and a 100 Hz data rate with
~2-5 ms delay from physical motion to I²C register. The policy reads the
IMU as three of its 137 observation dims and uses them for orientation,
so policy inference on a noisy real IMU will be biased at inference time.

**Evidence.**
- `urchin_env_cfg.py:118` — `imu = ImuCfg(prim_path="...")` with no
  `noise` or `bias` sub-config.
- `urchin_env_cfg.py:846-849` — observation dict packs `imu.lin_acc_b`,
  `imu.ang_vel_b`, `robot.data.projected_gravity_b` directly with no
  perturbation.
- `docs/design_review/consolidated_assessment.md` §7 does not call this
  out explicitly but implies it through "sim-to-real contact model"
  unknowns.

**Concrete sim-side fix.** In `_get_observations`, after fetching `imu`:

```python
# Per-episode biases (resampled in _reset_idx), per-step white noise.
gyro = imu.ang_vel_b + self._gyro_bias + sigma_gyro * torch.randn_like(imu.ang_vel_b)
acc  = imu.lin_acc_b + self._acc_bias  + sigma_acc  * torch.randn_like(imu.lin_acc_b)
grav = ... + tilt_drift_rotation  # small random rotation, resampled each ep
```

Per-episode draws in `_reset_idx`: `gyro_bias ~ N(0, 0.02 rad/s)`,
`acc_bias ~ N(0, 0.1 m/s²)`, tilt drift quaternion rotation of ~1°
random axis. Per-step noise: `sigma_gyro = 0.01 rad/s`,
`sigma_acc = 0.05 m/s²`. Optionally add a 1-step observation delay
(buffer one env tick) to model the 100 Hz rate mismatch.

Goes in `urchin_env_cfg.py` lines ~846-849.

**Severity: M.** The policy doesn't heavily depend on IMU magnitude (the
rolling_reward uses `root_ang_vel_w` and `root_lin_vel_w` from the
articulation, not the IMU sensor itself — see `urchin_env_cfg.py:862, 962`),
so noise on the IMU only degrades the policy's observation, not its reward
signal. But a tilt bias **will** rotate the `projected_gravity_b` obs by ~1°,
which propagates through the oracle decode at `urchin_env_cfg.py:773-779` —
the oracle uses body-frame gravity to decide which panels are on the
ground. A 1° tilt bias pushes the oracle's contact window by ~1 panel out
of 8 (much more impactful for N=8 than for the v3 N=42 baseline).

---

## 5. No mass / inertia randomization; CAD-predicted mass will be wrong

**Gap.** The URDF's mass and inertia tensors are whatever the CAD export
emitted, used verbatim at every reset. Real builds are historically 10-30%
off the CAD prediction (cables, fasteners, glue, solder, silicone the CAD
didn't model). For urchin_lite_v1 specifically, the consolidated review
flags a real new unknown: a gimbal + pendulum perception head that
**wasn't in Agent 1's topology** and shifts CoM off center by up to 40 mm.

**Evidence.**
- `urchin_v3_cfg.py:77-88` — `UrdfFileCfg` loads the URDF as-is; there's
  no `rigid_body_properties_range` or mass DR on the `ArticulationCfg`.
- `urchin_v3_cfg.py:89-98` — `InitialStateCfg` uses deterministic pose;
  `_reset_idx` at `urchin_env_cfg.py:1071+` resets to that deterministic
  state with only yaw jittered, no mass or inertia resampling.
- `docs/design_review/consolidated_assessment.md:76` — risk row:
  "CoM asymmetry from inner camera gimbal (not in Agent 1's topology)…
  Counter-balance with battery placement opposite the gimbal axis;
  re-measure CoM empirically in Week 7 before first roll."

**Concrete sim-side fix.** Use Isaac Lab's
`isaaclab.envs.mdp.events.randomize_rigid_body_mass` (or custom
`_reset_idx` hook since this env is a `DirectRLEnv`). Sample per episode:
- `total_mass *= U(0.85, 1.25)`
- `CoM_offset_xyz = N(0, [0.020, 0.020, 0.010]) m`
  (20 mm lateral std, 10 mm vertical std — covers the gimbal asymmetry
  concern from the reconciliation note)
- `inertia *= U(0.8, 1.2)` isotropically

Goes in `urchin_env_cfg.py` `_reset_idx` (new block before the yaw jitter).

**Severity: M.** The CoM-offset issue is specifically flagged as a first-roll
risk by the consolidated review. 20% mass error alone wouldn't sink a
rolling policy, but a 40 mm CoM offset that the policy never saw in
training will bias the rolling direction by up to 10-15° — enough to make
the robot curve instead of going straight to the goal on its first roll.

---

## 6. Contact stiffness / damping mismatch for the TPU panel shell

**Gap.** The URDF uses `collider_type="convex_hull"`
(`urchin_v3_cfg.py:86`), which gives a rigid polyhedral collider at
each panel — no contact compliance. The real urchin shell has TPU 95A
gaskets spanning 2-65 mm gaps at 0.8 mm wall thickness (see
`collision_audit.md` §3), plus the panel plates themselves are PETG/PLA
with some flex. The effective contact stiffness at ground impact is more
like a tuned spring in series with the real suspension, not a hard wall.
PhysX soft-contact defaults are "rigid with stabilization" — not compliant.

**Evidence.**
- `urchin_v3_cfg.py:86` — `collider_type="convex_hull"`, no
  `contact_offset` or `rest_offset` tuning, no soft-contact compliance.
- `docs/design_review/collision_audit.md:95-145` — TPU gasket has a
  specific strain profile (10-55%) and bonded accordion folds that store
  energy at each ground contact. None of this compliance is in sim.
- `docs/design_review/consolidated_assessment.md:112` §7 item 4 — "MuJoCo
  soft contact vs. real TPU gasket + carpet has no measured correspondence"
  (note: the project is on Isaac Sim / PhysX, not MuJoCo for urchin_v3,
  but the point transfers — PhysX soft contact isn't tuned for TPU
  either).

**Concrete sim-side fix.** Three-part:
1. Set explicit per-material `restitution=0.1, frictionCombineMode=average`
   on panel colliders.
2. Add a `soft_contact` param if Isaac Lab exposes it, else increase
   `solver_position_iteration_count` to 16 and `solver_velocity_iteration_count`
   to 4 for stability under low-stiffness DR.
3. Domain-randomize PhysX contact parameters per episode:
   `contact_offset ~ U(0.002, 0.010) m`,
   `rest_offset ~ U(0.0005, 0.003) m`.

If a more physically-grounded model is needed, a silicone skin simulation
is a future seam — `urchin_env_cfg.py:139` already stubs
`with_silicone_skin: bool = False` for exactly this.

**Severity: M.** The first roll will show up as a 10-20% energy loss per
contact that the sim never budgeted for — the robot rolls, but slower and
stops sooner between pushes. Wouldn't kill transfer, but will show as
"sim predicted we'd cover 1.5 m in 10 s, real covered 1.0 m".

---

## 7. Backlash and ball-screw deadzone on Candidate C hardware

**Gap.** The actuator shortlist's Class-B (and the StepperOnline LK60
candidate specifically) uses a ball-screw with ~0.01-0.02 mm per-cycle
backlash. At 60 mm stroke that's not individually visible, but accumulated
across push/retract cycles the commanded position drifts from the actual
position by up to 0.3-0.5 mm over a 10 s rollout. The sim's implicit PD
has zero position deadzone — command and actual position are
bit-identical until the PD's force loop acts.

**Evidence.**
- `docs/design_review/actuator_shortlist.md:88` — Candidate C row notes
  "ball-screws are less backdrivable than VCAs or belts" — implicit
  backlash but not quantified; standard cheap ball-screws run ~0.01 mm.
- `urchin_v3_cfg.py:99-123` — `ImplicitActuatorCfg` has no `backlash`,
  `deadband`, or `friction` sub-field on the joint drive. URDF joint
  damping/friction are explicitly left at 0 (`urchin_v3_cfg.py:81`).

**Concrete sim-side fix.** Add a per-joint deadband in the command path —
applied after the delay filter (gap #1) but before `set_joint_position_target`:

```python
delta = self._delayed_targets - self._actuator_state
deadband = cfg.actuator_deadband_m  # default 0.0002 m (0.2 mm)
moving = delta.abs() > deadband
self._actuator_state = torch.where(moving, self._delayed_targets, self._actuator_state)
robot.set_joint_position_target(self._actuator_state, ...)
```

Randomize `actuator_deadband_m ~ U(0.0, 0.0005)` per episode — covers
zero-backlash (VCA/rack, Candidates A/B) through cheap-ball-screw
(Candidate C).

Goes in `urchin_env_cfg.py` `_pre_physics_step`, same block as the delay
filter from gap #1.

**Severity: L.** 0.5 mm of accumulated position error on a 60 mm stroke is
<1% of travel — well under the noise floor of everything else. Matters more
for precise manipulation than for rolling. But it's free to add alongside
the delay model, and it silences one more source of sim-to-real surprise.

---

## 8. No battery sag / voltage drop → motor torque is constant forever

**Gap.** The sim's `effort_limit=60.0` holds for the full episode. Real
urchin runs a 6S LiPo (per `power_electronics.md`, summarized at
`consolidated_assessment.md:33`), and at a 12-motor build with peak bus
currents potentially in the hundreds of amps (see Candidate A's peak
current question in `actuator_shortlist.md` §Open question 1), battery
voltage sag from 25.2 V to 22.2 V over 30 minutes of running reduces peak
available motor torque by ~12%. For urchin_lite_v1 with N=8 the absolute
current will be lower, but the sag curve is the same shape. More
importantly, 30 min run time means the robot's torque ceiling is
**time-varying** — unmodeled in sim.

**Evidence.**
- `urchin_v3_cfg.py:109` — `effort_limit=60.0` is constant for the life
  of the sim. No time-varying coefficient.
- `docs/design_review/consolidated_assessment.md:79` — risk row on thermal
  and the 30 min runtime target calls out power envelope but doesn't
  propose a sim model for it.
- `docs/design_review/actuator_shortlist.md` §Open question 1 on 600 A
  momentary bus currents — not in sim.

**Concrete sim-side fix.** Add a linear voltage-sag schedule that scales
`effort_limit` over the episode:

```python
# In _pre_physics_step or as a post-physics hook:
t_frac = self.episode_length_buf / self.max_episode_length
sag = 1.0 - cfg.battery_sag_rate * t_frac  # default 0.0-0.15 over episode
effective_effort = cfg.base_effort_limit * sag
# Apply as a torque clip on the PhysX effort output.
```

Domain-randomize `battery_sag_rate ~ U(0.0, 0.15)` per episode so the
policy doesn't learn to count on the schedule.

**Severity: L.** Most first rolls will be <60 s and on freshly-charged
packs — the sag in that window is <2% and doesn't move the needle.
This matters for *endurance* testing, not first-roll transfer.
Deliberately last on the list because (a) the episode length in sim is
20 s and (b) the consolidated review's "thermal runaway in sealed shell"
already flags power as a hardware-side risk that will be resolved with
direct temperature sensors, not sim modeling.

---

## Severity summary

| # | Gap | Severity |
|---|-----|----------|
| 1 | Actuator command-to-response delay | **H** |
| 2 | Ground stiction (static > dynamic friction) | **H** |
| 3 | Force-vs-velocity motor curve + continuous/peak | **H** |
| 4 | IMU bias / noise / band-limit | **M** |
| 5 | Mass / inertia / CoM randomization | **M** |
| 6 | Contact stiffness / damping of TPU shell | **M** |
| 7 | Ball-screw backlash / actuator deadband | **L** |
| 8 | Battery voltage sag over episode | **L** |

---

## Implementation-order recommendation for Agent B1

Bake **gaps 1, 2, and 3 in from commit 1 of urchin_lite_v1.** These are the
three that directly control whether the robot rolls at all on first
contact with the floor — delay phase-shifts the push cycle, stiction
decides whether the push initiates cleanly, and the force-vs-velocity
envelope decides whether the push makes it through at the speeds the gait
wants. All three are cheap to add (each <30 lines in `_pre_physics_step` or
the actuator config), and training without them produces a policy that
can't be debugged when it fails on hardware — you can't tell if the
failure is the gait or the gap. Bake them as config fields with
DR-distribution knobs (not hard-coded values) so Agent B2/B3 can widen
the ranges if first-roll data shows bigger real-world spread.

**Gaps 4, 5, and 6 go in before any serious policy retrain cycle**
(week 2-3 of the build). IMU noise (gap 4) and mass/CoM randomization
(gap 5) tolerate being absent during the initial from-scratch training —
their impact is accuracy-degrading, not qualitative — but any policy
intended for the real robot should see them before the first transfer
attempt. Contact stiffness (gap 6) is similar: meaningful at first-roll
but second-order to gaps 1-3. The three can be added together in a single
"physical realism" DR PR.

**Gaps 7 and 8 can wait until after the first successful rolling trial.**
Backlash is swamped by every other unmodeled effect at 0.5 mm on 60 mm
stroke, and battery sag is below the noise floor for sub-60-second
rollouts. Add them in the post-first-roll domain-randomization retrain
(week 10 per the consolidated timeline) alongside whatever real-world
calibration data comes back from the physical build — at that point you
have measurements to calibrate the DR ranges against, instead of priors.
