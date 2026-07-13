# PyFluid AMReX Code Conventions

These rules keep the production solver predictable for human and AI
maintainers. `AGENTS.md` owns workflow details; this file owns implementation
decisions.

## One Contract Per Operation

- Implement one production path for one numerical operation. Dimension,
  coordinate, AMR level, and subcycling differences are inputs to that path,
  not reasons to duplicate it.
- A selected method means that method. Do not silently replace reconstruction,
  interpolation, Riemann solving, timestepping, diffusion, or CT because a
  branch is inconvenient.
- Generalize an implementation to all intended inputs. If a combination is
  genuinely outside the design, reject it once at configuration validation and
  document the support boundary. A bug in an intended combination does not make
  that combination unsupported.

## Fail Honestly

- Do not copy neighboring values over NaNs, clamp an invalid result into a
  passing test, or add a fallback merely to keep evolution running.
- Development failures belong in `review_running_cycle.md` with the failing
  command and evidence. Fix the underlying implementation; do not add a
  temporary startup prohibition for a branch currently being repaired.
- Do not commit temporary guards, bypasses, aliases, or replacement algorithms
  whose only purpose is to quarantine an active bug. Keep the failure visible,
  reduce it to the smallest reproducer, and repair the shared operation.
- Physical floors and positivity methods are explicit numerical methods, not
  error concealment. They need diagnostics and validation.

## Ownership And Sources Of Truth

- Register each runtime option once. The input file selects supported values;
  runners do not mutate simulation parameters.
- Define each grid function once in the field schema. Evolution, checkpointing,
  output, diagnostics, and plotting consume that schema.
- Coordinate transforms, metric factors, and vector/tensor operations each have
  one shared implementation. Initial data, evolution, and diagnostics reuse it.
- Generated kernels come from one high-level equation source. Do not retain a
  handwritten production duplicate after equivalence is established.

## AMR

- Level creation, level remake, FillPatch, subcycled FillPatch, and repair seeds
  use the same selected coarse-fine transfer contract.
- Reflux, average-down, prolongation, and physical boundary filling are distinct
  operations and must not substitute for one another.
- Automatic AMR is successful when the physical metric stops refinement with
  headroom. Maximum level, minimum feature length, radial guards, and cell
  budgets are resource protection, not convergence criteria.
- Short smoke tests do not establish dynamic AMR correctness. Regridding
  changes require long moving-feature tests, conservation, causal quiescence,
  and hierarchy diagnostics.
- Regrid descendants only when their parent and child times are synchronized.
  A descendant-only regrid must preserve the active parent flux register and
  reset only rebuilt descendants; otherwise retain globally synchronized
  regridding.
- `amr.regrid_mode=synchronized` evaluates `amr.regrid_int` in accepted
  level-zero steps. `amr.regrid_mode=aligned` evaluates it in accepted steps
  on the current finest level, then rebuilds from the lowest level whose
  descendants are synchronized. These cadences have different units; compare
  them at similar measured regrid counts, not by assigning the same interval.
- AMReX level callbacks are the source of truth for whether a scheduled regrid
  changed the hierarchy. An unchanged attempt must return before rebuilding
  registers, resetting clocks or caches, averaging down, or refreshing state.
  Changed-hierarchy maintenance starts at the first modified interface and
  covers only the synchronization chain affected by that change.

## Conservation Accounting

- Evolution owns cumulative numerical boundary fluxes, explicit source
  injection, and mechanical wall work, using the same RK/subcycling weights as
  the state update. Do not reconstruct those discrete budgets in Python.
- Scalar diagnostics reduce and write the ledger with the conserved totals.
  Shared postprocessing applies policy, checks closure, and plots residuals.
- Mark a ledger unsupported when its conserved quantity has a different update
  contract, such as fluid-plus-potential self-gravity energy. Do not report a
  partial budget as a valid closure check.

## Changes

- Prefer deleting an obsolete path over preserving compatibility aliases.
- Keep patches narrow, then run the smallest relevant regression. After shared
  infrastructure changes, run the quick suite; after hot-path changes, also run
  the benchmark.
- Never weaken a test to accommodate a failure without a physical or numerical
  reason recorded in the test declaration.
