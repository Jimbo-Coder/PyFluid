# Machine Profiles

`tools/pyfluid_machine.py` keeps compiler paths, MPI launcher syntax, environment,
and scheduler submission in one TOML profile. Simulation parameters remain only
in the input file.

Interactive `run` uses `run_case.py`, including immediate validation and reports.
Submitted jobs use the shared `machines/run.sh.in` template and execute the AMReX
binary directly. Compute nodes therefore need only the compiled executable, MPI,
the input file, and its runtime libraries; they do not need Python or `uv`.

The checked-in `machines/local.toml` reproduces the Mac mini build. Its launcher
disables Open MPI rank binding so macOS can place ranks on Apple performance
cores; Open MPI 5's default logical-core binding substantially reduces
throughput on this machine. Select it by name or rely on the default:

```bash
uv run python pyfluid_amrex/tools/pyfluid_machine.py show
uv run python pyfluid_amrex/tools/pyfluid_machine.py configure
uv run python pyfluid_amrex/tools/pyfluid_machine.py build --dim 2
uv run python pyfluid_amrex/tools/pyfluid_machine.py run pyfluid_amrex/inputs/hydro_sine_2d.in --ranks 3
```

`machines/local_openmp.toml` explicitly selects the separately managed
MPI+OpenMP AMReX companion. On this 4P+6E Mac, two ranks by five threads is the
measured cross-dimensional evolution compromise: it is within about one
percent of the 2D two-by-four optimum and tied within run variance with the
corrected 3D one-by-ten point. The optimum remains workload-dependent, and this
profile is not used for reduced initial-data solves that do not expose AMReX
tile parallelism. Install the companion first, then select the profile:

```bash
uv run python pyfluid_amrex/tools/manage_amrex.py --variant openmp --install
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine local_openmp show
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine local_openmp build --dim all
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine local_openmp run pyfluid_amrex/inputs/hydro_sine_2d.in
```

The profile leaves macOS thread placement to the operating system because the
local LLVM OpenMP runtime does not expose reliable affinity. It does not call
Grand Central Dispatch or encode Apple core IDs in PyFluid.

Apple hardware-counter attribution is intentionally separate from the machine
profile. To permit repeatable captures without retaining an authenticated root
shell, install the fixed-function root-owned wrapper once:

```bash
sudo pyfluid_amrex/tools/install_macos_powermetrics_access.sh
sudo -n /usr/local/sbin/pyfluid-powermetrics 60 \
  > /tmp/pyfluid-powermetrics.txt
```

The wrapper accepts only a bounded duration, writes only to standard output,
and fixes the `powermetrics` samplers and process-attribution options. Shell
redirection therefore opens the output as the ordinary user. Do not grant
passwordless sudo to the repository copy or to unrestricted
`/usr/bin/powermetrics`; only the installed root-owned wrapper is authorized.
Codex may still request its own app-level approval for an external command, but
the operating system will not request the account password.

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
