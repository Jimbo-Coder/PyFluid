# PyFluid

PyFluid is a Newtonian finite-volume HD/MHD solver. The AMReX/C++
implementation in [`pyfluid_amrex/`](pyfluid_amrex/) is the production path;
the Python/JAX and Python/Numba implementations are retained as references and
performance baselines.

## Setup

Install Python tooling and build the AMReX executables as described in
[`pyfluid_amrex/docs/install.md`](pyfluid_amrex/docs/install.md). Normal local
runs use three MPI ranks.

## Run

Use the shared wrapper for ordinary runs:

```sh
uv run python pyfluid_amrex/tools/run_case.py \
  pyfluid_amrex/inputs/hydro_sine_2d.in
```

The wrapper selects the dimension-specific executable, writes the resolved
input and run log, validates runtime scalar diagnostics, and creates the compact
scalar report. Add `--gf-plots` only when grid-function snapshots are wanted.

Run parameters live in one input file. C++ defaults and accepted values are
defined in `pyfluid_amrex/include/pyfluid_amrex/config.hpp`.

## Validate

```sh
# Fast smoke gate
uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level quick

# Normal confidence gate after solver or infrastructure changes
uv run python pyfluid_amrex/tools/validate_amrex_suite.py --level standard

# Performance regression check
uv run python pyfluid_amrex/tools/benchmark_pyfluid.py
```

MPI needs local sockets and cannot run inside a restricted sandbox. Generated
runs, plots, checkpoints, build products, caches, and dependencies stay in
ignored repository-local directories.

See [`pyfluid_amrex/README.md`](pyfluid_amrex/README.md) for the numerical
surface, output/restart contract, validation levels, and known limitations.
