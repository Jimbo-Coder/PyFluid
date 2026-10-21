# CapyrX architecture lessons

Source review: 2026-10-21. This is a focused architecture review, not independent
numerical acceptance of CapyrX or a complete audit of its branches.

The current review pin is Lucas Sanches's
[`feature/slave_evolution`, ab1b272](https://github.com/lucass-carneiro/CapyrX/tree/ab1b272291aac7963de16861f044e5f92e419d57),
dated 2026-10-13. Its interpolation implementation differs substantially from
[`main`, 4004947](https://github.com/lucass-carneiro/CapyrX/tree/4004947f7acbbdfa4744f8847e4621893375df57),
dated 2026-07-09. Conclusions below name the relevant source; default-branch
reading alone would miss the newer cache and centering checks. Source identities
and read depth are recorded in `runs/current/infrastructure_audit/cycle482/capyrx_sources.json`.
No CapyrX build or numerical run was performed.

## Useful boundaries

- **Patch description and geometry are separate from physics.**
  [`multipatch.hxx`](https://github.com/lucass-carneiro/CapyrX/blob/ab1b272291aac7963de16861f044e5f92e419d57/CapyrX_MultiPatch/src/multipatch.hxx)
  describes logical extents, cell counts, face neighbors, exterior faces and
  geometry flags. The coordinate owner selects a patch-system implementation;
  adding a map need not duplicate the equations or every reconstruction and
  Riemann combination. PyFluid already has map and patch owners: extend those
  contracts instead of adding another solver hierarchy.
- **Coordinate derivatives have their own pointwise interface.**
  [`global_derivatives.hxx`](https://github.com/lucass-carneiro/CapyrX/blob/4004947f7acbbdfa4744f8847e4621893375df57/CapyrX_GlobalDerivatives/src/global_derivatives.hxx)
  separates local first/second derivatives, inverse-map Jacobians and their
  derivatives. The scalar second-derivative transformation includes both
  terms: `f_,ij = J^a_i J^b_j f_,ab + (partial_j J^a_i) f_,a`.
  This is a useful interface model for generated derivative adapters. Tensor
  component/frame transformations remain a separate obligation; this scalar
  chain rule is not a tensor regularity prescription.
- **Geometry construction has an explicit lifecycle.**
  [`schedule.ccl`](https://github.com/lucass-carneiro/CapyrX/blob/ab1b272291aac7963de16861f044e5f92e419d57/CapyrX_MultiPatch/schedule.ccl)
  initializes patch metadata before driver startup and materializes coordinates
  and Jacobians at base-grid setup. PyFluid should likewise make setup/regrid
  ownership explicit, with no per-cell configuration lookup or topology rebuild.

## Interpolation cache lessons from the October branch

The source owner is
[`interpolate.cxx`](https://github.com/lucass-carneiro/CapyrX/blob/ab1b272291aac7963de16861f044e5f92e419d57/CapyrX_MultiPatch/src/interpolate.cxx).
The following checks are present in the implementation, not inferred solely
from its review comments:

- Lines 91--145 separate coordinate lists, deterministic component slots,
  reusable setup/distribution, scatter slices, result buffers and slaved
  interior indices. Setup and execution have different lifetimes.
- Lines 825--855 rebuild when the mesh epoch, caller's level/patch range, or
  skipped coordinate-invalid levels change. An epoch alone is insufficient:
  repair and ordinary synchronization can use different ranges without a regrid.
- Lines 895--913 exclude levels without valid coordinate data from collection.
  Lines 1120--1125 store that skipped-level set in the cache identity, so becoming
  valid invalidates the old plan even within the same epoch and range.
- Lines 1092--1118 require an actual exterior-face policy rather than silently
  allowing all outer-boundary ghosts as donor anchors.
- Lines 1177--1190 reject nonvertex groups when there are interpolation points;
  the current coordinate and scatter layout is explicitly vertex-based. It is
  not a ready-made cell/face/edge exchange implementation.

For a later PyFluid transfer plan, record mesh/map identity, active source and
destination ranges, layout/sampling/basis, donor validity and stage/time
requirements explicitly. Which properties belong to a reusable geometry plan
and which belong to each execution must follow their actual lifetimes. Keep
value validity checks separate from coordinate-plan reuse. These are proposed
PyFluid requirements, not claims that CapyrX already implements all of them.

The October source also has an optional interpolation path for slaved overlap
interiors. Its existence is not evidence of conservative finite-volume flux or
CT reconciliation. PyFluid retains unique owned evolved cells; duplicated
overlap requires a separately derived and validated conservation method.

## Architecture work before additional geometry features

1. Continue removing expensive implementation and metadata dependencies from
   public interfaces. Enforce each new boundary in the existing transitive
   ownership checker and measure affected rebuilds.
2. Preserve mathematically meaningful normalization stages in the symbolic
   source when unrestricted expansion causes repeated work. Use the existing
   ordered bindings; measure generation, resulting compilation and numerical
   behavior instead of adding a second generator or runtime formula.
3. Make point samples, finite-volume averages and staggered integrals explicit
   before exposing a centering switch or generalized transfer interface.
4. When geometry work resumes, separate map evaluation, field-basis conversion,
   interpolation planning, value transfer and conservative seam coupling.
   Reuse present owners; avoid a template product of patch, coordinate,
   centering, reconstruction and Riemann choices.

No new geometry capability is enabled by this review. The existing
[build roadmap](build_cost_roadmap.md) owns compile decisions; the
[centering inventory](baumgarte_spherical_review.md#focused-architecture-comparison-centered-spacetime)
owns the coherent spacetime layout/transfer scope.
