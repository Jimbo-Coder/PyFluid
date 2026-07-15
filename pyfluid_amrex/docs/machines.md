# Machine Profiles

`tools/pyfluid_machine.py` keeps compiler paths, MPI launcher syntax, environment,
and scheduler submission in one TOML profile. Simulation parameters remain only
in the input file.

Interactive `run` uses `run_case.py`, including immediate validation and reports.
Submitted jobs use the shared `machines/run.sh.in` template and execute the AMReX
binary directly. Compute nodes therefore need only the compiled executable, MPI,
the input file, and its runtime libraries; they do not need Python or `uv`.

The checked-in `machines/local.toml` reproduces the Mac mini build. Select it by
name or rely on the default:

```bash
uv run python pyfluid_amrex/tools/pyfluid_machine.py show
uv run python pyfluid_amrex/tools/pyfluid_machine.py configure
uv run python pyfluid_amrex/tools/pyfluid_machine.py build --dim 2
uv run python pyfluid_amrex/tools/pyfluid_machine.py run pyfluid_amrex/inputs/hydro_sine_2d.in --ranks 3
```

For another machine, copy the profile outside the repository or add a maintained
profile under `pyfluid_amrex/machines/`. Set `mpi_launcher` with a `{ranks}`
placeholder, provide CMake arguments under `[build]`, and set the scheduler
command/header under `[submit]`. `submit --dry-run` writes and prints the exact
batch script without launching it. `PYFLUID_MACHINE` selects a profile for all
validation tools without changing their command lines.

The profile does not choose solver physics, AMR, output cadence, or numerical
methods. Those remain in the checked-in input so a local run and a submitted run
have the same canonical `run.in`. Run `report_run.py` after a batch job when
Python postprocessing is available.
