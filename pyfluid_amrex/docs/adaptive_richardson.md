# Adaptive Richardson AMR Policy

Production AMR uses normalized Lohner tagging. The adaptive Richardson policy
below is retained as the semiproduction design for a more selective future
tagger; it is not exposed as `amr.tag_metric` until the solver has a generic,
side-effect-safe shadow advance.

## Policy Contract

1. Evaluate Lohner on every scheduled regrid event.
2. Occasionally run one unbiased global Richardson calibration. Do not limit
   the calibration to Lohner-tagged regions: smooth-wave and shock tests show
   that spatial gating can hide the errors Richardson is meant to discover.
3. Use a fixed dimensionless Richardson discrepancy tolerance. Do not expose a
   tag fraction, tag budget, blend weight, regional extent, or calibration
   cadence as a run parameter.
4. Cache the latest Richardson map. Between calibrations, use observed
   Richardson-only misses and cheap-indicator change to trigger early checks;
   exponentially back off through longer intervals and eventually dormancy
   when Lohner already captures the relevant error.
5. A calibration event contributes Richardson tags directly. Same-event unions
   and weighted blends pay the full estimator cost without a robust accuracy
   advantage.
6. Advance the real and coarsened workspaces through the same physics,
   timestepper, source, diffusion, and boundary interfaces. Restore all solver
   state after estimating error. No physics-specific shadow evolution and no
   repair of invalid estimator output are allowed.

## Evidence

The retained four-resolution analysis used coarse-to-medium discrepancy as the
predictor and medium-to-finer discrepancy as an independent target over smooth,
shock, radial, advected, and noisy nonlinear flows. At an equal 10% selected
cell fraction, Lohner captured 54.3% of held-out error energy and Richardson
captured 71.0%; the largest gains were noisy KH (25.5% to 81.9%) and Sod (88.7%
to 97.7%).

An adaptive global-calibration replay over 36 held-out temporal samples used
nine calibrations and two early triggers. It captured 99.87% of violation error
with 0.926 worst-case recall at an estimated 1.22x runtime, compared with pure
Lohner at 99.36% and 0.591 worst recall, and Richardson every event at 8.40x.
These are policy-level estimates, not production runtime measurements.

## Promotion Gate

Before adding `richardson` or `adaptive_richardson` to the runtime option set:

- implement a reversible hierarchy workspace through the shared level advance;
- validate conservation, restart, AMR, CT, diffusion, sources, and every
  timestepper with the estimator enabled;
- measure real AMReX clustering, communication, and wall time rather than cell
  selection alone; and
- keep pure Lohner behavior and performance unchanged when it is selected.
