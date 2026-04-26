# Urchin v3 — Real-World Topology Candidates (Agent 1)

## Context

This doc is part of the multi-agent feasibility review kicked off by
`C:\Users\aaron\.claude\plans\review-the-urchin-robot-shimmering-rain.md`.
The sim design is a 0.34 m spherical shell with 42 radial prismatic panels
(panel mass 0.025 kg, travel 0–60 mm, per-panel effort 15 N, commanded
velocity up to 1.5 m/s, shell 1.47 kg, total 2.52 kg). The policy
controls a **9-D spherical-harmonic basis** (l ≤ 2) that expands to the
42 panel targets — so the 42 DoF are mechanically redundant. Agent 1's
job is to propose reduced-DoF physical topologies (≤20 actuators) that
can still produce the contact-dipole rolling gait documented in the
curriculum plan.

Key physics numbers used throughout:
- Shell radius R = 0.17 m, total mass m ≈ 2.5 kg, Mg ≈ 24.5 N.
- Thin-shell moment of inertia I ≈ (2/3) m R² ≈ 0.048 kg·m².
- Rolling on PLA/carpet: μ_roll ≈ 0.02–0.05 → rolling resistance ≈ 0.5–1.2 N.
- To reach 0.5 m/s in 2 s (gentle start): τ = I·α = I·(v/R)/t ≈
  (0.048 × 2.94)/2 ≈ 0.07 N·m net torque. Easy for any of the drives below.
- Static-tip threshold for CoM-shift rolling on a shell of radius R:
  CoM must move > the contact patch half-width (~5–10 mm on TPU).
  With m_pendulum/m_total = 0.5 and pendulum arm = 0.10 m, max CoM
  offset = 50 mm — plenty of margin.

---

## Candidate A — Dodecahedral 12-panel prismatic ("DODEC-12")

### Geometry
Regular dodecahedron, 12 pentagonal faces inscribed in a 0.34 m sphere.
Each face edge ≈ 124 mm; face "flat-to-center" radius = 0.134 m. Each
pentagonal tile is a rigid PETG plate riding on a single central
prismatic actuator normal to the face, with a TPU 95A gasket skirt
bridging to its 5 neighbors (accommodates the 0–60 mm stroke without
opening a gap).

```
          __---__
        /   _,_   \           each cell:
       / .-'   '-. \          +--- PETG face plate (Ø ~135 mm)
      | |  pent. | |         /    |
      | |  tile  | |        /     | 60 mm stroke
       \ '-.___.-' /       /      |
        \   | |   /       +------- linear actuator (screw or voice coil)
         '--| |--'         |
            |_|            +------- rigid inner frame rib (PETG + AL6061 ring)
        12 such cells, 10 seams meet at each vertex (3 tiles per vertex)
```

### Actuators
- Count: **12 linear, 1 per face**. Rotary-to-linear via lead-screw stages
  (e.g. Actuonix L16 class) or short-stroke voice coils pending Agent 2/3.
- Control mapping: reuse the existing SH-9 → N-panel expansion with N=12.
  Least-squares projection matrix `B_12 = Y_12 · pinv(Y_12ᵀ·Y_12)` where
  Y_12 is the 12×9 SH basis sampled at the 12 face normals.

### Part count (rough)
12 face plates + 12 linear actuator assemblies + 1 structural inner
dodec-frame (printed in 20–30 ribs) + 12 TPU skirt panels + central
electronics tray = **~60 primary parts** plus fasteners.

### Retraining
**Light tune** (expected 0.5–1 M steps). Gait physics are identical —
same contact-dipole principle, same 9-D SH action space. Only the
panel-count N changes in the env config and the observation layout.

### Reward terms affected
`panel_l1_reg` and `panel_variance` rescale by N (divide by 12 not 42).
`panel_span` (WP5 asphericity) recomputes against 12-point panel
positions; retain but reduce the threshold. Contact-dipole oracle
target unchanged (operates in SH space).

---

## Candidate B — Gimbal + internal pendulum ("SPHERO-2")

### Geometry
Smooth sealed PETG sphere, 0.34 m diameter, wall 3 mm. Inside: a
2-axis gimbal stage driven by two BLDC gimbal motors (e.g. T-Motor
GB36-1 class) in a crossed-yoke arrangement. Hanging from the inner
gimbal ring is a **heavy pendulum puck** (LiPo pack + driver boards
epoxied to a 1.0 kg tungsten or steel weight, total ~1.2 kg) at arm
length r_p = 0.10 m from shell center.

```
     ,-""""""""-.       outer shell: smooth PETG 0.34 m Ø
    /            \      (no moving surface parts)
   |   +------+   |
   |   |  /\  |   |     inner frame: dodec ribs glued to inside of shell
   |   | /  \ |   |
   |   +-/XX\-+   |     gimbal: 2 BLDC motors, orthogonal axes
   |    /|  |\    |     pendulum: 1.0 kg puck + battery on 0.10 m arm
   |   ( | ●| )   |     rolling: gimbal tilts puck → CoM shift →
   |    \|__|/    |                shell tips + rolls
    \            /
     `-.______.-'
```

Physics: with pendulum 1.0 kg, shell 1.5 kg, total 2.5 kg, pendulum at
r_p = 0.10 m → CoM offset at full tilt = 1.0 × 0.10 / 2.5 = 40 mm.
Torque about the ground-contact point when tilted θ: τ = m_p·g·r_p·sinθ
→ 1.0 × 9.81 × 0.10 = 0.98 N·m peak. That yields angular acceleration
α = τ/I_total ≈ 0.98/0.14 ≈ 7 rad/s² → ≈1.2 m/s² linear; 0 → 0.5 m/s in
~0.4 s. Plenty.

### Actuators
- Count: **2 rotary BLDC** (yaw + pitch of the gimbal). Optional 3rd
  reaction wheel for yaw stability.
- Control mapping: the current 9-D SH action is a *bad* interface here.
  New action space: 2-D desired-tilt vector, or 3-D (tilt_x, tilt_y,
  spin_rate). Requires a new env.

### Part count
Shell hemispheres ×2 + inner dodec-rib cage (12 ribs) + 2 gimbal
mounts + 2 BLDC motors + pendulum block + battery + controller tray +
IMU = **~30 primary parts**. Lowest of the three.

### Retraining
**Full retrain.** Completely different action space and dynamics.
Luckily Sphero-style rolling is a well-studied MDP — PPO converges
fast (literature ~1–2 M steps on similar sphere-in-gimbal tasks).

### Reward terms affected
Drop `panel_*` terms entirely (no panels). Keep `forward_progress`,
`upright_cost` (reinterpreted as "gimbal inside limits"),
`action_cost`. Add a `tilt_rate_penalty` for smoothness.

---

## Candidate C — Hybrid: icosahedral 20-panel + internal pendulum ("HYBRID-20+2")

### Geometry
Regular icosahedron inscribed in 0.40 m sphere (slightly bigger than
sim to keep triangular face edges manageable ≈ 210 mm). 20 triangular
panels, each on a single prismatic actuator. Plus an inner 2-axis
gimbal-pendulum subassembly (as in Candidate B but smaller, ~0.5 kg
pendulum on 0.08 m arm) providing low-frequency CoM bias.

```
  outer: 20 triangular faces    inner: mini-gimbal + 0.5 kg puck
  ┌────── shell ──────┐         ┌──── gimbal ────┐
  │   /\    /\    /\  │         │      ╱ ╲        │
  │  /  \  /  \  /  \ │         │     ╱ ● ╲       │   2 BLDCs
  │ /____\/____\/____\│         │    ╱═════╲      │   tilt the puck
  │ 20 equilateral     │         │    pendulum     │
  │ tiles, 12 vertices │         └─────────────────┘
  └────────────────────┘
```

Logic: the pendulum handles **steady-state drive** (low-freq CoM
bias ≈ DC to 2 Hz), and the 20 prismatic panels handle
**high-frequency shape-shifting** for terrain conformance,
perturbation rejection, and the contact-dipole "push-off" at the
rear tile. Two tiers of actuation, two tiers of control bandwidth.

### Actuators
- Count: **20 linear + 2 rotary = 22**. Over user's ≤20 ceiling
  — so: drop the pendulum to 1 motor (pitch-only, ~1.5 kg ballast
  arrangement gives most of the roll authority) → **20 lin + 1 rot = 21**.
  Still over. Or use **16 linear** (icosa minus the 4 "pole" tiles,
  leave those as fixed shell caps) **+ 2 rotary = 18**. Recommended:
  the 16+2 variant.
- Control mapping: 9-D SH fits 16 panels naturally (same projection
  trick as candidate A). Pendulum adds a 2-D action channel. Total
  action dim 11.

### Part count
~90 primary parts. Most complex of the three.

### Retraining
**Full retrain.** New action channels (pendulum), new panel count,
new shell. Contact-dipole still valid in principle but the policy
needs to learn to coordinate two bandwidth regimes.

### Reward terms affected
Add a `pendulum_authority` shaping term early to bias the policy
toward using the cheap/efficient CoM-shift channel before the
panels. Adjust `panel_l1_reg` for N=16. Add
`dual_channel_smoothness` to prevent the two tiers fighting.

---

## Ranking table (1 = worst, 5 = best)

| Topology | Build complexity | Cost | Retraining cost | Closeness to current sim dynamics | Coolness |
|---|---|---|---|---|---|
| A. Dodec-12 | 3 — 12 identical cells, one-off tooling, but each cell still needs a sealed gasket | 3 — 12× linear actuators dominates BoM | **5** — same physics, same action space, just re-parametrize | **5** — identical contact-dipole gait | 4 — "shape-shifting blob" aesthetic survives |
| B. Sphero-2 | **5** — 2 motors, no moving surface, no gaskets, well-trod genre | **5** — 2 BLDCs + 1 pendulum << 12 linear actuators | 2 — full retrain, new action space | 2 — rolls by CoM shift not contact-push; different failure modes | 3 — looks like a beach ball, less "alive" |
| C. Hybrid 16+2 | 1 — hardest of the three; two subsystems + inter-system latency | 2 — 16 linear + 2 BLDC + pendulum | 1 — full retrain with dual-bandwidth coordination | 4 — keeps contact-dipole, adds pendulum | **5** — most expressive, best research value |

## Recommendation

**Proceed with Candidate A (Dodec-12).**

Rationale, in priority order:
1. **Retraining is cheapest**. Gait physics are unchanged — the
   existing SH-9 → panel mapping works for any N ≥ 9. Phase 1 approval
   transfers with a light tune, whereas B and C demand from-scratch
   training runs that each cost multiple GPU-days we don't need to
   spend.
2. **Build elegance**. 12 identical cells is the minimum that
   preserves the sim's defining *shape-shifting rolling blob*
   identity. Dropping to B turns the project into yet another
   Sphero variant; the user's research investment lives in the
   contact-dipole principle.
3. **Risk-bounded**. 12 linear actuators is the smallest N where the
   9-D SH basis is comfortably resolvable (at N=9, the projection is
   singular; N=12 gives a well-conditioned Y matrix). Hybrid 16+2 is
   also resolvable but adds a second actuation tier that doubles the
   failure surface.
4. **Keeps Candidate B open as a fallback**. If Agent 3's
   velocity-clamp sim comes back saying "no, even 0.2 m/s panel
   velocity doesn't roll," we drop to the Sphero-2 backup without
   having printed any panel tooling.

Recommended shell diameter: **0.34 m (unchanged from sim)**. Doorway
fit is comfortable (0.34 m << 0.70 m), it's the diameter Phase 1 was
approved at, and it gives 135 mm face plates — big enough for
robust panel mounting, small enough to fit the gimbal pendulum
fallback if we have to swap in Candidate B later.

## Open questions for downstream agents

1. **Agent 3 (velocity clamp)**: does the contact-dipole gait still
   roll at 0.4 / 0.2 / 0.1 m/s with N=12? The 12-panel coarser
   sampling may *increase* the needed per-panel velocity (fewer
   cells must travel further to displace the same CoM). Worth
   sweeping the clamp and N jointly.
2. **Agent 2 (actuator survey)**: for 12 actuators at ≤60 mm stroke,
   ≤20 N force, and whatever velocity Agent 3 certifies, what's the
   shortlist? Voice coils, lead-screws, or ball-screws all viable at
   this count — cost per unit can be 5–10× higher than the 42-panel
   case without blowing total BoM.
3. **Agent 4 (collision audit)**: pentagonal tiles on a dodecahedron
   have more tile-edge travel per mm of stroke than the 42-panel
   icosahedral tiling. TPU gasket design must accommodate ~30 mm of
   edge-to-edge separation swing without buckling inward on
   retraction.
4. **Policy transfer math**: confirm that re-sampling the SH basis
   at 12 instead of 42 points preserves the learned
   `compute_contactpush_oracle` trajectory in the SH-9 subspace (it
   should — the oracle lives in SH space — but verify by projecting
   a recorded episode and reprojecting through Y_12).
5. **If the Sphero-2 fallback gets activated**, we should pre-commit
   a "shared mechanical subassembly" — the dodec inner rib cage —
   that's reusable in both designs, so early structural prints are
   not wasted.
