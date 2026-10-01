# Precision and experimental Metal builds

FP64 CPU remains the default and the numerical reference. Precision selects
`amrex::Real` at build time; it does not select a different physics, grid
centering, time integrator, or AMR method. Use separate build directories and
matching AMReX installations. CMake rejects a precision mismatch.

## CPU FP32

From the repository root:

```sh
uv run python pyfluid_amrex/tools/manage_amrex.py --install --variant mpi-fp32
cmake --preset full-release-fp32 -S pyfluid_amrex
uv run python pyfluid_amrex/tools/benchmark_compile.py \
  --build-dir build/pyfluid_amrex-fp32 --target pyfluid_amrex_production \
  --label fp32-production --jobs 3
```

Select `PYFLUID_MACHINE=local_fp32` for profile-aware runners. An input file
continues to contain the simulation configuration. There is no runtime
precision flag. Native FAB readers decode stored precision/byte order; HDF5
uses the dataset type. Shared output metadata records `real_mantissa_bits`
(24 or 53), including scalar tables and checkpoints.

The opt-in comparison uses the existing RHD/RMHD entropy and spacetime
gauge-wave oracles:

```sh
uv run python pyfluid_amrex/tools/validate_precision.py
```

It verifies output precision, final physical time, recovery, scalar budgets,
analytic error, convergence where resolved, and comparison with a fresh FP64
run. Its 64-epsilon conservation allowance is specific to these short,
unit-scale controls. The shared arithmetic policy is `short_run_roundoff_v1`.
It uses output mantissa bits rather than a machine/backend name. Conservation
accounting retains the existing FP64 bounds and a 64-epsilon allowance.
For integrated magnetic divergence it requires
`divb_l1*h/(V*B_ref) <= 64*epsilon` in both precisions. Here `V` is the
periodic Cartesian box volume, `B_ref` is the declared characteristic field,
and `h` is the smallest active directional spacing reported by the solver.
On AMR, the smallest spacing over saved times gives a conservative global
bound; this does not replace local coarse-fine/interface validation.
Curvilinear metrics, symmetry multipliers and masked domains need a different
volume/length-scale owner. A zero-field control needs an explicit absolute
or exact-zero contract, not division by an arbitrary magnetic floor.

The same helper serves the Alfvén, selected CT-AMR/restart, and canonical
benchmark controls. Per-run tables report raw divB, normalization, epsilon
units and effective tolerances; unrelated legacy suite budgets are unchanged.
Analytic accuracy and convergence thresholds are independent. It is not a claim
that long stellar evolutions, cold/high-Lorentz recovery, elliptic solves,
AMR, or every legacy compiled test is suitable for FP32. Keep measured
failures in the continuity cycle rather than relaxing all tests at once.

Gauge constraint norms are reported for assessment; this gate does not yet
provide an FP32 production acceptance budget for them. At N64, the resolved
wave in Cycle425 has Hamiltonian maxima about 0.0031 on CPU FP32 and 0.0125 on
Metal, versus 1.1e-7 on FP64. CPU evaluation of the saved Metal fields reproduces
the larger norm, including when those already-rounded fields are promoted to
FP64 for the diagnostic. The error is in the evolved state, with little effect
from diagnostic arithmetic. Its behavior is consistent with roundoff amplified
by derivatives; further representation/convergence work is required before
claiming FP32 numerical-relativity accuracy.

## Measured performance and accuracy boundary

The final Cycle432 canonical CPU gates pass in both precisions, including all
24 repeat-specific scalar checks. FP32/FP64 throughput ratios are 1.077, 1.121,
1.042 and 1.916 for the four cases listed below. These compare fixed work,
not equal-error solutions.

The newer multidimensional CT comparison uses PWL/HLL/SSPRK3, CFL 0.9,
64 steps, three CPU ranks or one Metal rank, and two repeats. At 64³ cells,
FP64 CPU takes about 91.1 s and FP32 CPU about 48.6 s. Metal takes 250.9 s
with a fresh application source cache and 48.5 s with reuse. All six 3D runs
pass their scalar/recovery checks. Metal's later throughput is about 19%
higher than CPU FP32, but process startup absorbs the gain in this short run.
The earlier 1D GPU advantage therefore does not generalize to every workload.

Cycle433 corrects the dimensionally inappropriate absolute divB criterion
used in the original 512² comparison. The CPU/Metal FP32 values 1.52e-5 and
1.69e-5 correspond to 0.332 and 0.368 epsilons after normalization, compared
with 0.300 epsilons for FP64. All 12 saved 2D/3D repeats pass the shared
arithmetic policy. The original four FAIL records are preserved and superseded
by a separate reassessment; no trajectory or timing changed. These fixed-work
timings now have passing scalar/recovery checks, while analytic/equal-error
performance remains a separate claim.

In the accepted 3D comparison, summed CPU rank peak RSS is 0.76 GiB for FP64
and 0.40 GiB for FP32. Metal solver peak RSS is 4.03 GiB with a fresh cache
and 3.54 GiB on reuse, while its main arena uses 260 MiB. Rank RSS sums are
not a simultaneous memory footprint and can double-count shared mappings;
GPU process RSS excludes compiler/driver services. These grids fit the 16 GiB
M4, but no memory-capacity ceiling or larger-grid crossover was measured.
Exact controls, failed budgets and cache provenance are retained in
`runs/current/infrastructure_audit/cycle432/report.md` and its adjacent tables.
The criterion correction and saved-data reassessment are in
`runs/current/infrastructure_audit/cycle433/report.md`.

The following older measurements retain their original scope.

Cycle426 compares the same executable source sets on an M4 Mac mini with
16 GiB memory. All three configurations list the same 98 unique production/tool
sources. FP32 changes scalar storage and arithmetic, with precision-dependent
recovery tolerances; it does not duplicate the physics. Each Real array uses
half the FP64 bytes, but total process memory and runtime need not halve.

The canonical CPU cases use three MPI ranks and three repeats, alternating
precision order. Inputs, grids and numerical methods are matched. All 24 runs
pass the declared scalar checks; FP32 allowances remain explicit and local.

| Case | FP64 Mcells/s | FP32 Mcells/s | FP32 / FP64 |
| --- | ---: | ---: | ---: |
| 512² hydro, PWC/LLF/Euler | 98.64 | 106.10 | 1.08 |
| 64³ hydro, PWC/LLF/Euler | 58.65 | 65.43 | 1.12 |
| 64³ hydro, WENO-Z/HLLC/RK4 | 2.603 | 2.700 | 1.04 |
| 32768-cell uniform RMHD, PWC/HLL/Euler | 2.008 | 3.726 | 1.86 |

The post-hardening Cycle428 rerun retains these results within ordinary timing
variation: FP32/FP64 ratios are 1.063, 1.100, 1.044 and 1.824, respectively.
All eight performance gates and all 24 repeat-specific scalar checks pass.
The canonical runner now retains each repeat's scalar result before overwriting
run outputs; `--allow-warn` cannot turn a solver/scalar failure into success.

These measure fixed numerical work, not equal-error performance or kernel-only
arithmetic. Recovery iteration counts can change with precision. No isolated
attribution to memory bandwidth, vectorization or root iterations is established.
No cold compile-time comparison was performed in this cycle. Separate build
profiles compile their own objects; the Metal executable is about 491 MiB,
versus 29--30 MiB for CPU, with device code embedded by the SSCP compiler.

Cycle428 corrects the fine-grid entropy error growth found in Cycle426. RK
stages now combine the input's difference from the base and the RHS before
adding the base state. At fixed time 0.2, N1024 RHD/RMHD density errors fall from
4.41e-5/4.54e-5 to 3.46e-6/4.56e-6 in FP32; FP64 retains 1.76e-6/1.77e-6.
Both FP32 cases now pass their unchanged physical and conservation budgets.
The N256/512/1024 sequence still shows an eventual FP32 roundoff plateau,
so this does not establish unlimited refinement or long-term NR accuracy.

Cycle428 establishes a Metal throughput crossover on this same 16 GiB machine
after correcting the runtime bug below. The matched RMHD entropy cases use
PWL/HLL/SSPRK3, CFL 0.9, 64 steps, three CPU ranks or one GPU rank. Later
16-step scalar windows exclude first-use work and include normal synchronization
and diagnostic costs; they are not GPU-event kernel timings.

| Cells / box size | CPU FP64 Mcells/s | CPU FP32 Mcells/s | Metal FP32 Mcells/s |
| --- | ---: | ---: | ---: |
| 32768 / 8192 (two repeats) | 0.217 | 0.416 | 0.198 |
| 262144 / 32768 (one repeat) | 0.285 | 0.550 | 1.314 |

All nine runs pass the declared scalar budgets and clean recovery. The larger
case delivers 2.39 times the CPU FP32 later-window throughput, so GPU usefulness
does not require more RAM here. This is a short throughput control, not analytic
validation at a common final physical time. Total process cost still favors
CPU: 168.78 seconds GPU versus 30.96 seconds CPU FP32 at 262144 cells. No
end-to-end crossover was demonstrated in that default-setting comparison.
The pinned runtime caches generated MSL
but still creates Metal libraries and pipelines in each process. More useful
work per launch and amortizing that startup are distinct from RAM capacity.

Cycle429 isolates a major cause of that cost: the inherited JIT guard sleeps
1000 ms after each compilation postcheck, including a pipeline archive attempt
when archives are disabled. Setting only `ACPP_METAL_JIT_POSTCHECK_MS=0` in
two otherwise unchanged 262144-cell runs reduces total process cost to
16.58/17.03 seconds; both pass the same scalar/recovery budgets. A fresh
application-source-cache run also passes in 42.97 seconds. The 20 ms guard
yield, compilation mutex, API error checks, queue deadlines and precise math
remain active. The reduction is largely explicit waiting, so the older cost
must not be described as pure compilation time.

These populated-cache controls establish an end-to-end GPU gain of about
1.84 times versus the recorded CPU FP32 reference. They do not establish
long-run backend reliability with this setting. Cycle432 subsequently adopts
it in the experimental profile after four debug-enabled fresh/reused-cache
hydro/gauge restart pairs reproduce their uninterrupted fields bitwise. `cycle429/report.md` retains the exact runtime option and cache
scope; a fresh application cache does not clear opaque OS/driver caches.

The historical Cycle426 256-cell wave took about 126 seconds and delivered
only 0.00157--0.00165 Mcells/s in later windows. Its severe small-grid overhead
should not be extrapolated to larger workloads.

The Cycle426 failures at 32768 cells were a runtime resource-declaration bug.
Metal needs explicit residency and hazard declarations for resources accessed
through nested pointers. A kernel that receives Array4 metadata can reach field
buffers absent from its direct argument list. The runtime now declares the
live USM buffers while building its pointer-translation inventory, keeping both
views consistent. This conservatively marks every live allocation read/write;
finer reachability information could reduce that overhead later.

The failure reproduces at only 256 cells when fields and metadata deliberately
occupy different allocations. The identical probe passes through 32768 cells
after the runtime patch. More RAM would not fix this bug. Corrected full wave
runs pass through 262144 cells; the historical small-case timing above is not a
GPU capacity limit. Cycle432 extends selected multidimensional/AMR controls
as described below; it does not certify the full configuration space.

Exact inputs, executable hashes, repeats, failed logs and reusable measurement
drivers are in `runs/current/infrastructure_audit/cycle426/` and `cycle428/`;
each has a compact `report.md`. The updated numerical comparison is under
`plots/current/scans/precision_and_metal/affine_precision.png`.

## Arithmetic precision contract

FP32 covers evolved fields, recovery, generated kernels, RK coefficients and
stage times, AMR thresholds, initial-data arithmetic, and physical reductions.
Floating constants use AMReX `_rt` literals. Overload selection matters too:
an integer exponent passed to `std::pow(float, int)` can select wider arithmetic.
The generator owns typed numerical catalogs; do not repair emitted headers.
Symbolic derivation and reference calculations can use exact or wider numbers;
the generated constants consumed by evolution are rounded to the selected Real.

`time_integration.hpp` owns affine stage arithmetic for both precisions and
backends. The exact RK catalog guarantees unit total state weight; accepted
fluid/spacetime states, limiter trials and CT stage fields share this rule.
General linear algebra does not implicitly acquire an affine contract.

The opt-in `pyfluid_amrex_state_update_check` CMake target checks indirect
storage with separate allocations, zero-RHS constant-state invariance over
every catalogued method, accurate summation of identical binary32 donor data,
and exact box-launch work ownership including a partial final box. Build it
in the selected precision/backend directory
and run it through `validation_common.run_mpi` (three CPU ranks or one Metal
rank). It is excluded from ordinary production builds. Analytic evolution
continues to use `validate_precision.py`, rather than a second physics oracle.

`reduction_sum.hpp` owns compensated FP32 accumulation for runtime integral
and conservation-ledger reductions. Its two Real values retain addition
residuals through AMReX's reduction tree; there is no FP64 device accumulator.
The FP64 path still uses Real directly. `diagnostic_reductions.cpp` owns the
MultiFab component reduction, avoiding a repeated kernel declaration in every
caller. This fixes false 3D conservation drift from ordinary FP32 summation;
postprocessing re-sums remain diagnostic evidence, never replacement totals.

Cold Valencia energy is generated in a rationalized form that avoids
subtracting rest-mass-scale terms at small internal energy. RMHD reuses that
hydro expression plus magnetic energy. General-EOS recovery consumes the same
symbolically generated stable internal-energy expression as the Palenzuela
source. TOV center expansion uses an actually representable enthalpy offset;
FP32 ID probes explicitly select attainable integration tolerances instead of
silently clamping the input tolerances.

Clang FP32 production compilation treats `double-promotion` and
`implicit-float-conversion` diagnostics as errors, including shared object
owners. Integer indices converting to `Real` are a separate range concern.
These checks catch implicit conversions, not arbitrary explicit casts. Keep
explicit precision conversions at reviewed boundaries. Host profiling uses
`WallTime=double`; elapsed times are formed before conversion for scalar output.
Host error formatting and Python analytic postprocessing also need not be FP32.
None of these wider host operations supplies evolved physics values.

Recovery stopping tolerances must also fit the selected arithmetic. The FP32
root bracket uses eight epsilons; the existing FP64 tolerance is unchanged.
The old 512-epsilon FP32 bracket could stop before meeting the conserved
round-trip acceptance bound. Tightening the solve fixes that mismatch without
loosening the accepted-state or physical validation budgets.

Verification has three independent parts: compile the production owners with
the precision diagnostics, inspect actual device IR, and check output metadata
and stored field types. A size assertion or an FP32 HDF5 dataset alone does not
prove FP32 arithmetic. For a fresh Metal run, `ACPP_S2_DUMP_IR_FINAL` selects an
LLVM IR output file and `ACPP_APPDB_DIR` selects a new cache. Inspect executable
types/instructions for `double`, `fpext`, and `fptrunc`, excluding quoted names,
comments and metadata. This covers the kernels exercised by that run.

## Metal dependency route

The route is `PyFluid -> AMReX SYCL -> AdaptiveCpp SSCP -> Metal`, following
the third-party Lulzx `warpx-metal` reference. PyFluid replaces WarpX at the
application layer. We retain its compiler base and selected patches, with the
local adaptations listed below. Recent kernel restructuring changes the work
inside PyFluid launches; it does not change this backend route. CPU and GPU
builds share the physics sources and generated equations.

### Current readiness boundary

Cycle451 adds native CUDA portability while retaining shared CPU/Metal physics.
The C++ symbolic printer preserves FP32 math overloads on GCC/NVCC, and common
EOS unit conversion avoids overflowing CGS intermediates. CPU32/CPU64 and
Metal32/CPU64 precision comparisons pass all 18 maintained cases; native CUDA
also passes its final 18-run comparison and all-dimensional production build.
CPU32, Metal and CUDA 3D MP5 CT-AMR pass analytic, conservation, divergence
and resource-headroom checks; CUDA MP5 checkpoint restart reproduces every
field and metadata value exactly. The separate PWL 3D six-step control reaches
its AMR cell budget on CPU64, CPU32 and CUDA, and remains a failed control
requiring an error/resource study.

The final shared-source Metal run first exposed a separate pipeline failure
in RMHD signal frequency. That factory already selected ideal/runtime EOS on
the host but discarded the selection before launching recovery. Carrying the
same selection through recovery and characteristic evaluation reduces the
ideal-gas shader from about 1.49 MB to 279 KB. Final Metal fields on all AMR
levels and the coarse control remain byte-identical to Cycle439, with exact
constraint histories. The retained failing specialization reproduces in
standalone replay; no backend equations, math modes or dependency patches
were changed. See the [Cycle451 report](../../runs/current/infrastructure_audit/cycle451/report.md)
for current CUDA instrumentation and benchmark boundaries. The matched FP32
262144-cell wave takes 39.72 s on three M4 CPU ranks, 19.89 s on Metal,
16.34 s on eight Ryzen ranks and 6.62 s on CUDA (three-repeat process medians).
These are workload-specific timings; small cheap runs remain dominated by
GPU startup, and no multi-device scaling result is established.


Cycles438--439 close the reproduced 3D CT-AMR compiler failures in the tested
configuration. Connection initialization gathers only metric first derivatives;
optional geometric tagging has separate launches; constraints omit evolution-only
derivatives and select their formulation on the host. The maintained Metal 3D
MP5/HLL CT-AMR analytic, conservation, divB and headroom checks pass. Final fields
and constraint histories match the preceding validated Metal run exactly.
Spherical Metal constraint startup also passes; spherical GPU evolution and a
fresh 3D MP5 checkpoint/restart check remain outside this evidence.

Cycle445 repeats the current 3D CT-AMR/coarse-reference control with two
independent empty application caches. Both pass with byte-identical decoded
fields and exact constraint histories against Cycle439. OS-managed Metal caches
remain intact; these checks establish application-cache repeatability for this
control, not general compiler robustness.

The retained Cycle439 intermediate combined constraint shader fails twice in
standalone replay: source-library creation succeeds, then Apple's M4 pipeline
compiler returns an internal error. The final Cartesian and reference shaders
compile successfully with the same settings. With an identical entry-point name,
changing only the retained shader's formulation flag from a runtime argument to
either constant makes it compile; restoring the dynamic choice restores failure.
This identifies a trigger in that approximately 11.8 MB shader and supports the
existing host formulation selection. It does not identify the faulty compiler
pass, establish a universal shader-size limit, or prohibit GPU branches in
general. The diagnostic variants are compile-only artifacts; production physics
and dependencies are unchanged. See the
[Cycle445 report](../../runs/current/infrastructure_audit/cycle445/report.md)
for the controlled comparison and fresh-cache numerical checks, and the
[Cycle439 report](../../runs/current/infrastructure_audit/cycle439/report.md)
for the implementation change.

Earlier Cycle432 no-postcheck startup and hydro/gauge checkpoint controls
passed, and the experimental profile selects
`ACPP_METAL_JIT_POSTCHECK_MS=0`. Actual API checks, serialization, queue deadlines
and the 20 ms compiler-service yield remain enabled.

The existing Alfvén and CT-AMR/restart tools accept selected dimensions and
profiles. The CT-AMR control also compares the diagonal entropy mode with a
matched coarse run using the existing 1.5 improvement requirement. Its smooth
hierarchy can remain unchanged; actual moving interfaces are assessed with the
existing advected-cloud refinement-centroid and density oracles. These are
separate coverage claims, not an exhaustive backend matrix.

The final moving Newtonian CT cloud reaches t=0.12 in 41 steps with mass and
energy drifts 1.19e-7 and 6.30e-8, divB L1 3.77e-6, two levels of headroom,
and refinement centroids that track the cloud. Density L2 7.42e-4 is comparable
to CPU FP64/FP32 controls 7.29e-4/7.73e-4. The existing 3D ratio-four CT/reflux
control also passes all three profiles. Its deliberately capped one-level
hierarchy is a topology control, not error-driven AMR completion. All 219
captured modules from the successful moving case use FP32 floating arithmetic.
The final 2D PWL CT-AMR checkpoint restart reproduces fields, time metadata
and reference totals exactly.

AMReX box-loop launches previously visited each cell 256 times under the
unified compiler frontend. Target-selection intrinsics and the proper SYCL
iterator restore exactly one visit; the opt-in storage/update check protects
both host whole-box and GPU partitioned iteration. Flux registers select the
actual launch region, and temporary CT edge integrals use the async arena.

Cycle437 repairs the 2D MP5/HLL limiter compilation failure by separating
candidate recovery from reconstruction/Riemann work and reusing tile-local
face fluxes. Its largest emitted limiter shader falls from 20.1 MB to 7.3 MB.
CPU64/CPU32 shock fields match their pre-change controls bitwise; Metal passes
both PWC and PWL/PWC guards with matching correction counts and differences
below three FP32 epsilons relative to CPU32. Existing 2D CT-AMR analytic,
conservation, divB and headroom gates pass on all three profiles. Extra launches
and temporary scratch are the tradeoff; no limiter throughput gain is claimed.

Optional CFL-location diagnostics now pass the shared maximum checker on Metal
and hierarchy on/off controls on all three profiles, with identical physical
scalar histories. Solid-mask host-array work remains deferred.

Longer matched single-object controls expose a further FP32 full-GR limitation.
The TOV initial-state round trip passes at about 1.7e-7 after the cold-arithmetic
fix, but evolved TOV and filled-hole runs reject invalid outer-corner ghost
metrics. Tensor-product extrapolation followed by nodal-to-cell transfer has
an absolute weight sum about 2.1e8 at one tested corner. Re-evaluating stored
FP32 donors in FP64 still gives a badly distorted metric: the stencil amplifies
information already lost in storage. This needs a conditioned boundary closure
with independent accuracy/stability validation; lower-order substitution,
clamping and bypassing the metric check are not fixes.

Matched FP64 TOV reaches t=10 with a baryon-ledger error about 5.2e-14, and the
filled-hole control reaches t=2.4M. These are bounded evolution checks. The
coarse 32-cubed TOV's surface mean thermal-pressure/cold-pressure ratio rises
from 0.0120 to 0.3658; its surface p95 reaches 1.9385. This is an EOS thermal
proxy, not temperature in kelvin, and the atmosphere's large ratio is a
separate floor effect. Surface heating remains a numerical-physics issue in
FP64 as well. The filled interior intentionally violates the constraints;
exterior Hamiltonian L2 grows from 3.36e-4 to 1.10e-3 in the recorded interval.
Neither control establishes long-time or converged single-object accuracy.

The six-step 3D PWL CT-AMR restart setup reaches the cell-budget guard in
both FP64 and FP32; a fixed-time FP64 CFL0.45 control also fails that guard.
The cause remains unresolved. The explicitly selected MP5 restart control
reproduces fields and checkpoint metadata exactly on CPU FP32. Preserve the
PWL failure as an AMR evolution limitation instead of changing its defaults
or treating a restart comparison alone as general stability evidence.

Use a Cartesian near zone with regular angular outer patches as the next
geometry direction. Retain nodal Cartesian spacetime. Pure-spherical
cell-centered spacetime and particles are deferred; adding either would not
resolve the boundary-conditioning or FP32 constraint limits above.

### Dependency and build setup

This is AMReX SYCL through AdaptiveCpp SSCP and Metal. It is experimental and
uses a separately patched dependency; it is not an upstream AMReX Metal
backend. Device arithmetic is FP32. CPU FP64 remains available in its
independent build. All three application dimensions build on the M4 GPU
profile. Runtime readiness is scoped by the controls and open limits above;
a successful executable build alone does not certify its configuration space.
This retains the pinned `warpx-metal` reference's backend route and AdaptiveCpp
base revision. Our manifest applies six patches from that reference plus eight
local compiler/runtime patches. Its AMReX26.06 integration is adapted to this
project's AMReX26.09 pin; particle/device-parser extensions are omitted. This is
an adapted implementation of the reference route, with real local maintenance
responsibility. The intermediate shader's diagnosed formulation trigger and the
passing final configuration are distinguished in the readiness boundary above.
First-use compilation and process setup remain separate costs from evolution.
The current profile removes the inherited one-second postcheck sleeps; the
Cycle429 comparison above establishes a bounded speedup with source-cache reuse.

`dependencies/metal.toml` pins the compiler, public WarpX patch source,
compatible SDK, and Apple headers. `dependencies/amrex.toml` retains ownership
of the official AMReX pin. A Metal patch-base mismatch requires a reviewed
patch update. `manage_metal.py` checks exact source identities, including
files added by patches, and refuses unexpected edits. It never executes the
upstream reset/clean scripts or replaces the CPU dependency source.

Prerequisites on this Apple Silicon profile are Homebrew `llvm@20`, `lld@20`,
`libomp`, `boost`, `ninja`, MPI and parallel HDF5, plus the macOS 26.5 SDK.
LLVM20 cannot compile the installed SDK27 math headers; set the compatible
SDK explicitly for both compiler and application builds. The setup tool
does not install system packages.

```sh
uv run python pyfluid_amrex/tools/manage_metal.py --build --jobs 2
uv run python pyfluid_amrex/tools/manage_metal.py
cmake --preset experimental-metal -S pyfluid_amrex
uv run python pyfluid_amrex/tools/benchmark_compile.py \
  --build-dir build/pyfluid_amrex-metal --target pyfluid_amrex_production \
  --label metal-production --jobs 2
```

The local profile is `PYFLUID_MACHINE=local_metal`. It selects the Metal
backend and one rank for the M4 GPU. Actual GPU access requires a host context
with device access; a sandbox reporting no devices is not solver evidence.
AMReX uses a GPU selector, so absence of a GPU must fail instead of silently
choosing CPU execution. AdaptiveCpp lacks AMReX's optional Intel UUID query;
multi-rank GPU sharing and device-count reporting require a separate audit.

The adapted AMReX patch covers compiler flags, optional Intel attributes,
base RNG compilation, supported atomics/scans, and temporary-memory lifetime.
Host/device selection uses AdaptiveCpp's target intrinsics in its unified
frontend. Testing only `__SYCL_DEVICE_ONLY__` there selects host code inside
GPU kernels; the exact box-launch ownership control protects the observed
256-fold repeated-work failure in AMReX flux-register loops.
Stream cleanup currently waits before freeing buffers because AdaptiveCpp
does not support the Intel `host_task` cleanup path. This preserves lifetime
at a possible synchronization cost. Particle and device-parser extensions
from the WarpX application are not included. Keep backend work in dependency
and launch/storage owners; do not fork the generated physics.

The local compiler patches discover unnamed LLVM aggregate types before
emitting declarations, reject device FP64 instead of silently narrowing it,
and translate LLVM binary32 classification through IEEE bit fields. The
classification control covers all ten LLVM classes, including signed zeros,
subnormals, infinities and both NaN classes. Device failure diagnostics copy a
record to the host for formatting; device varargs would promote FP32 to FP64.
Shared control-flow regions are structured once, preserving entry instructions
and PHI incoming values when a region is reused by multiple branches. Generic
Metal builtins are specialized by constant operation before optimization;
LLVM's standard scalarizer lowers optimizer-created vector loads/stores before
Metal pointer translation. Metal compilation explicitly selects safe math and
precise floating-point functions. These choices preserve arithmetic semantics; they do not promise
bitwise CPU/GPU equality or adequate accuracy for every physical problem.

Compiler or math-option changes require a fresh JIT cache for validation. Use
`ACPP_APPDB_DIR` to select a separate repo-local cache and distinguish first
kernel compilation from warmed execution when reporting timings.

Patch provenance and redistribution terms are retained in
`dependencies/patches/warpx-metal.LICENSE`. See `references.md` for the
authoritative upstream sources, `knowledge.md` for current support boundaries,
and `review_running_cycle.md` for setup evidence (Cycles424–425), kernel
restructuring (Cycles437–439), and compiler diagnosis/repeatability (Cycle445).
