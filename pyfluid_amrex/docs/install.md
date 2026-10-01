# Local AMReX Dependency

PyFluid pins its one repository-local production dependency in
`dependencies/amrex.toml`. The current release is AMReX 26.10 at the exact
source commit declared there. Homebrew continues to own CMake, MPI, compilers,
ccache, and parallel HDF5; `uv.lock` owns Python packages. Athena and CompOSE
trees are reference/data assets rather than linked native dependencies.

The default MPI-only AMReX layout is versioned and consumes one stable link:

    source:  .deps/amrex-26.10
    build:   .deps/build-amrex-26.10-mpi-lsolvers-hdf5
    install: .deps/amrex-26.10-mpi-lsolvers-hdf5
    active:  .deps/amrex-mpi-lsolvers-hdf5 -> amrex-26.10-mpi-lsolvers-hdf5

The installed library enables MPI, AMR, linear solvers, parallel HDF5, and
1D/2D/3D targets. This is the install selected by the CMake preset and local
machine profile.

An optional MPI+OpenMP companion uses the same source pin and feature contract:

    build:   .deps/build-amrex-26.10-mpi-omp-lsolvers-hdf5
    install: .deps/amrex-26.10-mpi-omp-lsolvers-hdf5
    active:  .deps/amrex-mpi-omp-lsolvers-hdf5 -> amrex-26.10-mpi-omp-lsolvers-hdf5

It is a build/machine choice only. It does not add a physics method, generated
kernel product, or runtime numerical branch.

## Check Or Update

The dependency helper checks only by default. It compares the tracked pin,
managed source commit, installed feature contract, and latest official stable
release:

    uv run python pyfluid_amrex/tools/manage_amrex.py

For a network-free local consistency check:

    uv run python pyfluid_amrex/tools/manage_amrex.py --offline

To update to the latest official stable release:

    uv run python pyfluid_amrex/tools/manage_amrex.py --update

To install or repair the OpenMP companion for the current pin:

    uv run python pyfluid_amrex/tools/manage_amrex.py --variant openmp --install

Check it without network access with `--variant openmp --offline`. After an
AMReX pin update, run the OpenMP install command once to bring the companion to
the new pin. Only the default `mpi` variant may change the shared source pin.

An explicit release may be selected with `--version YY.MM`. Update mode clones
the exact release tag, verifies its resolved source commit, builds and installs
under versioned `.deps` paths, checks MPI/HDF5/linear-solver/dimension support,
then activates the stable link and records the pin. **It builds AMReX only.**
It never configures, compiles, or tests PyFluid and never updates Homebrew or
uv packages. Build the desired PyFluid target separately using the presets
below. Do not update an installation while a build is consuming it.
Each configure/build/install command reports its own elapsed time.

Once activation succeeds, obsolete managed source/build/install trees are
retired automatically. An older companion that is still active is not obsolete;
its install and source remain until that companion is updated too. Modified or
unverified source trees and unrecognized experiments are never auto-deleted.
There is no long-term rollback archive. If recording the pin fails, activation
restores the old link before any cleanup.

To preview or explicitly perform cleanup without a build or network access:

    uv run python pyfluid_amrex/tools/manage_amrex.py --prune --dry-run
    uv run python pyfluid_amrex/tools/manage_amrex.py --prune

After a successful update, run the quick suite plus any focused tests warranted
by the upstream release notes before committing the new manifest pin. AMR/CT
interpolation and elliptic solvers are normal focused checks for AMReX updates.

## Homebrew Prerequisites

CMake 3.25 or newer is required by both AMReX 26.10 and its installed CMake
package. PyFluid's project and presets declare the same minimum.

Homebrew `hdf5-mpi` supplies parallel HDF5. Install and activate the MPI variant
before updating AMReX:

    brew unlink hdf5  # only when the serial formula is currently linked
    brew install hdf5-mpi

The optional OpenMP companion additionally requires Homebrew `libomp`. The
manager supplies AppleClang's explicit include, compile, and library discovery
settings; do not add those flags to physics targets by hand:

    brew install libomp

Compressed parallel-HDF5 plotfiles remain the default; native AMReX plotfiles
are an explicit runtime option.

## PyFluid Builds

Use the repository presets:

    cd pyfluid_amrex
    cmake --preset full-release
    cmake --build --preset full-2d

For the explicit OpenMP companion:

    cmake --preset full-release-openmp
    cmake --build --preset full-openmp-2d

The same full configuration supports every numerical method. Build the dimension
you are developing; `pyfluid_amrex_production` explicitly builds all three.
Checkpoint-comparison tools are separate explicit targets; their implementation
objects are reused from the matching dimension's solver build. Validation
scripts build those tools when needed.
Do not use feature-stripped binaries for ordinary development or validation.
Presets select the stable active AMReX link, while CMake's native
`CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS` resolves imported include/library paths to
the real versioned install. Changed paths invalidate dependent objects even
when the replacement files have older timestamps. The tracked dependency pin
is also a CMake configure dependency. No clean-first or catalog regeneration
is required just to change AMReX. After an update, configure the appropriate
preset and build its selected target; runners do this through the machine profile.

Development latency targets on this host are for one selected executable and
backend with AMReX already installed: under five minutes for a clean PyFluid
build, roughly 15--20 seconds for a small implementation edit, and about one
minute for a larger edit. These are targets, not yet demonstrated bounds for
all high-fan-out headers. Measure AMReX's own build, a full dimension/backend
matrix, cold compilation, cache-hit rebuilding, and no-op builds separately.
Normal updates run quick and changed-area checks; unrelated physics debugging
and full old/new version sweeps are separate work.

### Keeping Edit Cost Local

From the repository root, inspect header reach without compiling:

    uv run python pyfluid_amrex/tools/build_dependencies.py --header include/pyfluid_amrex/reconstruction/mp5.hpp

The quick architecture gate runs the same ownership checks. Consumer counts
combine source owners across all preprocessor branches; they are not the number
of objects in a particular configured build. System headers and generation
dependencies remain CMake's responsibility.

Generated vector operators have separate consumer leaves under
`kernels_generated/vector/` and a vector-only CMake generation stamp. Edit
their symbolic sources, not generated headers. `vector_ops.hpp` is for
aggregate checks only; production includes the exact leaves it uses. The
build-contract tests verify that a viscous formula edit changes only its leaf
and that unchanged leaves retain timestamps. This prevents an unrelated
coordinate/viscosity edit from invalidating every reconstruction kernel through
the EoS header; it does not make a genuinely changed heavy flux kernel cheap.

Profile an expensive owner with the actual configured flags, no compiler cache,
and isolated output (`--trace` supports Clang and NVCC 12.8 or newer):

    uv run python pyfluid_amrex/tools/profile_compile.py --source src/flux_kernels/mp5.cpp --dimension 3 --trace

Repeat `--source` for a sequential scan; select `--build-dir` for another
backend configured with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. NVCC traces report
elapsed spans for its individual compiler processes; these can overlap and
must not be summed. Logs, optional phase traces, and `summary.json` overwrite
under `runs/current/compile_profiles/`; temporary objects are removed. A per-owner
profile excludes generation/linking and does not estimate a parallel full build.
Run it without concurrent builds or simulations when comparing timings.
Each source has a 300 s hard deadline by default (`--timeout` changes it).
On expiry the compiler process group is terminated, the temporary object is
removed, and the scan stops with exit 124 and a timed-out summary. Begin CUDA
investigations with one owner; do not queue a full rebuild behind a probe.

At integration points, enforce an appropriate whole-target budget:

    uv run python pyfluid_amrex/tools/benchmark_compile.py --target pyfluid_amrex_3d --label local_change --budget local-edit

Budget classes are `no-op` (1 s), `local-edit` (20 s), `shared-edit` (60 s), and
`cold-selected` (300 s for one selected executable/backend). `--max-seconds`
sets a different explicit host/scope budget. `--allow-slow` records a miss
without failing the command; it never suppresses compiler errors. These flags
do not touch sources, clean the tree, bypass cache, or make a warm build cold.
The rolling compile TSV retains its schema; its stable `.latest.json` companion
also records the build directory, budget and forced-recompile counter. Preserve
that compact record with a deliberate baseline when needed.

Whole-target builds also have an independent `--timeout` of 300 s by default.
Timing goals classify a completed build as `slow`; the hard deadline actually
stops CMake and its compiler descendants and records `timeout` with exit 124.
`--allow-slow` never overrides this deadline. Choose any longer deadline
deliberately after inspecting the target's build scope and measured owner costs.

## MPI And The Agent Sandbox

MPI requires local sockets. The Codex sandbox cannot create those sockets, so a
sandbox MPI failure is an execution-environment limitation, not a solver
result. Use the approved socket-capable execution path for MPI validation.
Normal workstation jobs use three ranks; use four only for deliberate scaling
or peak-throughput measurements.

On Apple Silicon, CMake's native architecture detection also queries
`sysctl hw.optional.arm64`. The sandbox can block this query and silently
remove the generated `-arch arm64` flag, causing avoidable recompilation.
The maintained local presets/profiles explicitly pin
`CMAKE_OSX_ARCHITECTURES=arm64` so sandboxed and socket-capable reconfigures keep
the same compiler flags. Preserve that pin when deriving a local profile;
changing toolchain flags or execution context is not a compiler benchmark.
