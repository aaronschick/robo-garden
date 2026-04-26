# Non-Spherical Rolling Shapes — Future Exploration

**Author:** exploration note, 2026-04-22
**Companion to:** `urchin_ovoid_v1_design.md` (prolate spheroid — already scoped)
**Status:** idea backlog, not a committed workstream.

---

## Context

The urchin v3 body is approximately spherical — 42 prismatic panels on a round
shell, driven by a contact-dipole oracle (rear-ground push + top-front CoM
bulge). Phase 1 flat-ground rolling was video-approved 2026-04-22. The
curriculum in `urchin_v3_continued_curriculum.md` holds the geometry constant
and expands task difficulty. `urchin_ovoid_v1_design.md` breaks spherical
symmetry along *one* axis (prolate spheroid) to get short-axis-vs-long-axis
mode-switching.

This doc is the broader version of that question: **which other non-spherical
rolling geometries would teach the contact-dipole controller something the
sphere and the ovoid can't?** It's a menu of candidates with enough detail to
pick one to prototype, not a built plan for any single one.

The selection criterion is controller-shaped, not aesthetic: a shape is
interesting here iff it makes the *control problem* meaningfully different
from "pick a rolling axis on a sphere." Shapes that just look cool but reduce
to small perturbations of the sphere are skipped.

---

## Three categories, by what they do to the controller

| Category | Geometric hallmark | What changes for the dipole controller |
|---|---|---|
| **Constant-height rollers** (sphericon, oloid, polycons) | Developable surface; CoM height stays ~constant during roll; surface unfolds flat | Rolling *path* is geometry-enforced (a meander, not a line). Controller must learn to ride the natural path, or bias off it to steer. Energetics are trivial (no CoM pumping needed), so the policy is pure path-shaping. |
| **Anisotropic rollers** (ovoid already scoped; tricylinder / Steinmetz; lens / discus) | Distinct preferred axes with unequal moments of inertia | Multi-mode locomotion. Mode-switching (short-axis roll ↔ long-axis spin ↔ tumble) becomes the core skill. Ovoid covers the continuous case; tricylinder covers the discrete case (3 preferred axes). |
| **Adversarial / self-righting shapes** (Gömböc; tetrahedral monostatic shapes) | Exactly one stable equilibrium; homogeneous convex | Controller fights gravity-driven return to rest. Only interesting as a *recovery* substrate or as a stress-test for "can the policy overcome a shape that actively resists rolling." |

---

## Candidate shapes

### Tier 1 — worth prototyping after ovoid

**Oloid.** Convex hull of two equal circles of radius `r`, axes perpendicular,
centers separated by `r` (the classical oloid; at separation `r·√2` it becomes
the two-disc roller / Schatz oloid). Properties that matter:

- **Every point on the surface touches the ground during one roll cycle.**
  Uniform contact distribution — interesting for panel placement: the
  urchin's 42 panels would all get ground-contact exercise regardless of
  initial pose.
- **Rolling path is a meander**, not a straight line. The oloid wobbles
  side-to-side at ~the disc radius.
- **CoM height is near-constant** during rolling (developable surface).
- **Two distinct contact modes** — rolling on one of the two circle arcs at
  any instant, with contact transferring between circles smoothly.

What this teaches the controller: the oracle currently assumes "forward" is
well-defined from `to_goal_b`. On an oloid, the geometrically natural forward
direction meanders along a fixed curve relative to body frame. The residual
must either (a) reinforce the meander when it roughly points goal-ward, or
(b) bias off it to steer — a strictly harder problem than sphere-rolling.
Tests whether the SH residual has enough spatial bandwidth to represent a
rolling-phase-dependent contact command.

**MJCF implementation.** No primitive. Need a `mesh` asset — the oloid is
easy to generate analytically (convex hull of two circle point-sets, each
with ~64 points). Write `scripts/build_oloid_mesh.py` that emits
`workspace/robots/urchin_oloid_v1/body.obj` and a matching panel-placement
script (panels normal to the mesh surface, area-weighted Fibonacci sampling
on the convex hull — reuse the ellipsoidal-area-weighted sampler from the
ovoid design with a different area metric).

**Prolate spheroid / ovoid.** Already scoped in `urchin_ovoid_v1_design.md`.
Listed here only so this doc is a complete roadmap. Do ovoid first — it's
the minimum-viable asymmetric shape and unlocks the two-mode oracle pattern
that all harder shapes will need.

### Tier 2 — worth prototyping if Tier 1 succeeds

**Sphericon.** Two half-bicones joined at 90°, single developable surface.
Like the oloid but simpler geometry (two cone halves instead of circle-hull).
Rolling path is a meander that alternates between the two cone surfaces. The
*lesson* overlaps with the oloid — both are constant-height geometry-enforced
meander rollers — so only worth trying if the oloid reveals a specific
sub-problem (e.g. the cone's sharp edge creates a discrete contact transfer
that the oloid's smooth curve doesn't).

**n-sphericon / hexasphericon / polycons.** Sphericon generalizations (Hirsch
& Seaton, 2020) with n-fold symmetry. More lobes per rolling cycle → more
opportunities for rhythmic gait entrainment. Worth revisiting only if a
rhythmic-gait capability becomes a target in the curriculum (it currently
isn't).

**Tricylinder / Steinmetz solid** (intersection of three mutually
perpendicular cylinders). **Three discrete preferred axes** instead of
two. The discrete case of anisotropic rolling. Good follow-on to the ovoid
if the two-mode oracle works cleanly — three-mode dispatch is the next
unit of difficulty.

**Lens / discus** (oblate spheroid with aspect ratio ~0.3). The *opposite*
of the ovoid: now the short axis is the *unique* preferred rolling axis
and the long axes are isotropic in-plane. Should be easier than ovoid on
rolling, harder on steering. Low-priority unless we want an easy
anisotropic baseline.

### Tier 3 — interesting but skip unless motivated

**Gömböc.** Mono-monostatic convex homogeneous shape. Exactly one stable
and one unstable equilibrium — after any perturbation it self-rights to the
same pose. **Actively anti-locomotion.** Only interesting as:

1. A *recovery substrate*: if we want a robot that always returns to a known
   pose after a fall, the Gömböc is the geometric prior for that. But this
   is a different problem from locomotion.
2. A *stress test*: "can the policy sustain rolling on a shape that fights
   back?" A legitimate research question, but not one we need answered to
   advance the urchin curriculum.

Skip unless a specific downstream use case (self-righting after a cliff
drop, e.g.) surfaces.

**Reuleaux tetrahedron / Meissner body.** Constant-width 3D shape. Rolls
with constant *width* between parallel planes but not with constant CoM
height on a flat ground. The constant-width property is interesting for
confined-space locomotion (rolling in a pipe of matching width) but
doesn't add a new lesson for open-ground rolling.

---

## Priority order for prototyping

1. **Ovoid** — already scoped in `urchin_ovoid_v1_design.md`. Do first.
2. **Oloid** — best single lesson-per-dollar after ovoid. The
   geometry-enforced meander is a qualitatively new control problem and
   the constant-height property means we can reuse all existing reward
   weights without re-tuning energetics.
3. **Tricylinder** — discrete-axis generalization of the ovoid's
   continuous two-mode case. Only if the ovoid's two-mode oracle works
   and we want to see how far the pattern scales.
4. **Sphericon / hexasphericon / polycons** — only if the oloid reveals
   a specific sub-problem that these help isolate.
5. **Gömböc, Meissner** — only as recovery substrates or confined-space
   studies, not as main-line locomotion targets.

---

## Infrastructure reuse (what transfers, what doesn't)

What transfers unchanged across all candidates:
- Panel spring-PD pipeline, LPF, effort limits, action space (9-D SH).
- BC pretrain → PPO → scaler-refit training chain.
- All v3 reward terms except ones that assume a spherical body frame
  (`aspherity` term — compares panel positions against a sphere; would
  need a shape-aware reference for non-spherical bodies).
- `compute_contactpush_oracle` structure. Each shape gets its own oracle
  variant that decides "forward" and "top-front" in its own body frame —
  same function signature.

What each shape needs fresh:
- **Mesh generation script** (`scripts/build_<shape>_mesh.py`).
- **Panel-placement script** that samples the mesh surface with
  area-weighting (generalization of the ovoid Fibonacci sampler).
- **Outward-normal computation** (gradient-of-level-set or mesh-face
  normal, depending on whether the shape has an analytic implicit form).
- **Oracle variant**: redefine "forward" and "top-front" on the new body
  frame. For shapes with geometry-enforced rolling paths (oloid,
  sphericon), the oracle should probably *follow* the natural path when
  goal-aligned and only deflect when off-course.

What's genuinely novel per shape (not in any existing doc):
- **Oloid**: rolling-phase observation. The oloid's state-relative-to-path
  is more than just `(gravity_b, to_goal_b)` — it also depends on where
  we are in the meander cycle. Add a phase scalar to the observation,
  computed from body-frame angular position along the natural path.
- **Sphericon / polycons**: discrete contact-surface-ID observation. The
  cone edges are contact discontinuities; the policy needs to know which
  face is currently down.
- **Tricylinder**: discrete 3-mode oracle dispatch (generalize the ovoid's
  soft-sigmoid gate to a 3-way softmax).

---

## Open questions (do not attempt to answer in this doc)

- Does the 9-D SH residual basis have enough spatial bandwidth to
  represent rolling-phase-dependent contact commands on the oloid? If
  not, what's the next basis — higher-`l` SH, or something respecting
  the shape's own symmetry group?
- For geometry-enforced meanderers (oloid, sphericon), is it cheaper to
  *augment the oracle* with the natural rolling path as a closed-form
  prior, or to let the residual learn it? Memory
  `feedback_trust_physics_intuition.md` points toward closed-form first.
- How does the BC dataset generalize across shapes? The current v3 BC
  pretrain uses the contactpush oracle; on an oloid the oracle is
  different. Do we need a separate BC pretrain per shape, or does
  shape-agnostic BC from a *combined* oracle library transfer?
- Is there a single "shape-parameterized" urchin family (continuous
  interpolation sphere → ovoid → oloid) that would let the policy learn
  a shape-conditioned controller in one training run? This is a big
  architectural question — deferred.

---

## What this doc is not

- Not a committed workstream. The urchin v3 curriculum
  (`urchin_v3_continued_curriculum.md`) has priority. This is backlog.
- Not a replacement for the ovoid design. That doc is load-bearing for
  whichever of these shapes we prototype next; this doc assumes it.
- Not a physics proof. Claims like "every oloid surface point touches
  the ground" and "constant CoM height on developable rollers" are
  standard results from the rolling-surfaces literature, not derived
  here. Before any prototype commits to an MJCF, re-verify the
  geometric claim for the specific mesh that `build_*_mesh.py` emits —
  numerical meshes often break the clean analytic properties.
