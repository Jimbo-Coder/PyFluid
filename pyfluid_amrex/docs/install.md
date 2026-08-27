# Local AMReX Dependency

PyFluid pins its one repository-local production dependency in
`dependencies/amrex.toml`. The current release is AMReX 26.08 at the exact
source commit declared there. Homebrew continues to own CMake, MPI, compilers,
ccache, and parallel HDF5; `uv.lock` owns Python packages. Athena and CompOSE
trees are reference/data assets rather than linked native dependencies.

The default MPI-only AMReX layout is versioned and consumes one stable link:

    source:  .deps/amrex-26.08
    build:   .deps/build-amrex-26.08-mpi-lsolvers-hdf5
    install: .deps/amrex-26.08-mpi-lsolvers-hdf5
    active:  .deps/amrex-mpi-lsolvers-hdf5 -> amrex-26.08-mpi-lsolvers-hdf5

The installed library enables MPI, AMR, linear solvers, parallel HDF5, and
1D/2D/3D targets. This is the install selected by the CMake preset and local
machine profile.

An optional MPI+OpenMP companion uses the same source pin and feature contract:

    build:   .deps/build-amrex-26.08-mpi-omp-lsolvers-hdf5
    install: .deps/amrex-26.08-mpi-omp-lsolvers-hdf5
    active:  .deps/amrex-mpi-omp-lsolvers-hdf5 -> amrex-26.08-mpi-omp-lsolvers-hdf5

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
then activates the stable link and rebuilds every PyFluid production dimension.
The previous AMReX install is preserved. If the PyFluid rebuild fails, the
helper restores the former link and rebuilds against the former install before
returning failure. It never updates Homebrew or uv packages.

After a successful update, run the quick suite plus any focused tests warranted
by the upstream release notes before committing the new manifest pin. AMR/CT
interpolation and elliptic solvers are normal focused checks for AMReX updates.

## Homebrew Prerequisites

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

The same full configuration builds every numerical method. Do not use
feature-stripped binaries for ordinary development or validation. The preset
pins the stable active AMReX link directly, so reconfiguring replaces a stale
AMReX selection in an existing CMake cache.

## MPI And The Agent Sandbox

MPI requires local sockets. The Codex sandbox cannot create those sockets, so a
sandbox MPI failure is an execution-environment limitation, not a solver
result. Use the approved socket-capable execution path for MPI validation.
Normal workstation jobs use three ranks; use four only for deliberate scaling
or peak-throughput measurements.
