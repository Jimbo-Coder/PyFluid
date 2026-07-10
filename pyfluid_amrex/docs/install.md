# Local AMReX Dependency

The canonical PyFluid AMReX install is local to this repository:

    source:  .deps/amrex-25.11
    build:   .deps/build-amrex-mpi-lsolvers-hdf5
    install: .deps/amrex-mpi-lsolvers-hdf5

It is built with MPI, AMR, linear solvers, parallel HDF5, and 1D/2D/3D targets.
This is the install selected by the full-release CMake preset.

Homebrew `hdf5-mpi` supplies parallel HDF5. Compressed HDF5 plotfiles are the
default; native AMReX plotfiles remain an explicit runtime option.

Install and activate the MPI variant before configuring AMReX:

    brew unlink hdf5  # only when the serial formula is currently linked
    brew install hdf5-mpi

## Rebuild

Do not modify the AMReX source tree for PyFluid work. Reconfigure and rebuild
the dedicated dependency build:

    cmake -S .deps/amrex-25.11 -B .deps/build-amrex-mpi-lsolvers-hdf5 \
      -DCMAKE_INSTALL_PREFIX=$PWD/.deps/amrex-mpi-lsolvers-hdf5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/homebrew/bin/mpicc \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/mpicxx \
      -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/hdf5-mpi \
      -DHDF5_ROOT=/opt/homebrew/opt/hdf5-mpi \
      '-DAMReX_SPACEDIM=1;2;3' \
      -DAMReX_MPI=ON \
      -DAMReX_OMP=OFF \
      -DAMReX_FORTRAN=OFF \
      -DAMReX_PARTICLES=OFF \
      -DAMReX_LINEAR_SOLVERS=ON \
      -DAMReX_LINEAR_SOLVERS_INCFLO=ON \
      -DAMReX_LINEAR_SOLVERS_EM=ON \
      -DAMReX_HDF5=ON

    cmake --build .deps/build-amrex-mpi-lsolvers-hdf5 -j 3
    cmake --install .deps/build-amrex-mpi-lsolvers-hdf5

Verify the install contract:

    rg 'AMReX_MPI_FOUND|AMReX_LSOLVERS_FOUND|AMReX_HDF5_FOUND|AMReX_SPACEDIM' \
      .deps/amrex-mpi-lsolvers-hdf5/lib/cmake/AMReX/AMReXConfig.cmake

Expected values are MPI on, linear solvers on, HDF5 on, and space dimensions
1;2;3.

## PyFluid Builds

Use the repository presets:

    cd pyfluid_amrex
    cmake --preset full-release
    cmake --build --preset full-2d

The same full configuration builds every numerical method. Do not use
feature-stripped binaries for ordinary development or validation.
The preset pins `AMReX_DIR` directly, so reconfiguring replaces a stale AMReX
selection in an existing CMake cache.

## MPI And The Agent Sandbox

MPI requires local sockets. The Codex sandbox cannot create those sockets, so a
sandbox MPI failure is an execution-environment limitation, not a solver result.
Use the approved socket-capable execution path for MPI validation. Normal
workstation jobs use three MPI ranks; use four only for deliberate scaling or
peak-throughput measurements.
