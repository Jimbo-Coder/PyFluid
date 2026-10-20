# Machine Profiles

`tools/pyfluid_machine.py` keeps compiler paths, MPI launcher syntax, environment,
and scheduler submission in one TOML profile. Simulation parameters remain only
in the input file.

Interactive `run` uses `run_case.py`, including immediate validation and reports.
Submitted jobs use the shared `machines/run.sh.in` template and execute the AMReX
binary directly. Compute nodes therefore need only the compiled executable, MPI,
the input file, and its runtime libraries; they do not need Python or `uv`.

The checked-in `machines/local.toml` selects the `full-release` CMake preset,
including Ninja, the compiler flags and the repository-local ccache. Its launcher
disables Open MPI rank binding so macOS can place ranks on Apple performance
cores; Open MPI 5's default logical-core binding substantially reduces
throughput on this machine. Select it by name or rely on the default:

```bash
uv run python pyfluid_amrex/tools/pyfluid_machine.py show
uv run python pyfluid_amrex/tools/pyfluid_machine.py configure
uv run python pyfluid_amrex/tools/pyfluid_machine.py build --dim 2
uv run python pyfluid_amrex/tools/pyfluid_machine.py run pyfluid_amrex/inputs/hydro_sine_2d.in --ranks 3
```

`local_dev_rmhd` selects the maintained `focused-rmhd-pwl` CMake preset in a
separate directory, inherits Ninja and the compile cache, and builds only 1D
unless `build --dim` selects another dimension. It retains PWL, LLF/HLL/HLLD,
SSPRK3, cell-centered MHD and internal correction fallbacks. CT, diffusion and
subcycling require the complete profile
or an explicitly configured additional method set. The complete `local` build
remains available for integration. The runner checks exact compiled choices
before overwriting output and saves executable identity in `build.json`.

Both profiles retain their existing build directories and executable names.
Existing Make directories require the explicit migration described in
[PyFluid builds](install.md#pyfluid-builds); configuring does not migrate or clean
them automatically. OpenMP and FP32 CPU profiles retain Make, while Metal keeps
its existing Ninja configuration. No dependency or GPU rebuild is needed just to
change the full/focused CPU executor.

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
placeholder, provide CMake arguments or `configure_preset` under `[build]`,
and set the scheduler
command/header under `[submit]`. `submit --dry-run` writes and prints the exact
batch script without launching it. `PYFLUID_MACHINE` selects a profile for all
validation tools without changing their command lines.

The profile does not choose solver physics, AMR, output cadence, or numerical
methods for a run. A deliberately focused build compiles a subset and rejects
unavailable choices. Run choices remain in the checked-in input so a local run and a submitted run
have the same canonical `run.in`. Run `report_run.py` after a batch job when
Python postprocessing is available.

## Windows workstation through WSL

The active workstation checkout is `/home/max/phys_research/PyFluid` inside
Ubuntu's ext4 disk on `E:\WSL\Ubuntu`. The older `/home/max/PyFluid` symlink
points to a Windows checkout with independent notebook edits; it is not the
build workspace. Keep Linux sources, environments and builds on the Linux
filesystem rather than under `/mnt/c` or `/mnt/e`.

`wsl_cpu` selects GCC, MPI, parallel HDF5 and FP64; `wsl_cpu_fp32` provides
the matching FP32 CPU build. `wsl_cuda` selects the same solver sources with
native AMReX CUDA, FP32 and architecture 86 for the RTX 3070 Ti. CPU profiles
use three ranks by default; the GPU profile uses one rank on the single device.
All three use the MPI launcher even for one rank: direct singleton execution
had a measured startup delay on this WSL installation. Compile jobs and runtime rank counts are
separate settings. These profiles initially use portable host code rather
than claiming a tuned Ryzen performance baseline.

Install the official CUDA toolkit without a Linux NVIDIA driver in WSL; the
Windows driver supplies GPU access. The initial CUDA 13.4 setup uses
`cuda-minimal-build-13-4`, `libcurand-dev-13-4`, `libcusparse-dev-13-4` and
`cuda-sanitizer-13-4`, plus Ubuntu's MPI/parallel-HDF5 development packages,
Ninja and ccache. Use a Linux-only `PATH` in the build shell: CMake's CUDA
library search otherwise probes inherited Windows application directories
through the slower mounted filesystem. This does not require changing global
WSL interoperation settings.
The dependency installer uses native C/C++ compilers on Linux and lets
CMake discover MPI as an explicit dependency, including its CUDA compilation
flags. It accepts standard `CC`, `CXX`, `CUDACXX` and `CUDAARCHS` environment
settings. Use ordinary host compilers for CUDA rather than MPI wrappers.

```bash
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
uv sync --locked
uv run python pyfluid_amrex/tools/manage_amrex.py --install --jobs 6
CUDAARCHS=86 uv run python pyfluid_amrex/tools/manage_amrex.py --variant cuda-fp32 --install --jobs 2
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine wsl_cpu configure
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine wsl_cpu build --dim 1
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine wsl_cuda configure
uv run python pyfluid_amrex/tools/pyfluid_machine.py --machine wsl_cuda build --dim 1
```

The dependency pin is shared with the Mac. `cuda` is the separate FP64
dependency variant; precision does not require another physics implementation.
CUDA fast math is explicitly disabled for the initial numerical comparison.
Managed CUDA builds also select `AMReX_GPU_RDC=OFF`. Each kernel's device
callees are available in its compilation unit; host modules remain separately
compiled. AMReX's target helper applies this policy to application targets.
The validated CUDA 13.4 toolchain gives template launch stubs local ownership
in this mode, resolving repeated-registration diagnostics without suppressing
Compute Sanitizer checks. CUDA 12.8/12.9 need an explicit static-stub opt-in;
those versions have not been validated here. See
[NVIDIA's linkage notes](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/).

Build configurations have separate directories and share the existing
repository-local compile cache. Cycle 451 records the earlier RDC integration:
native CUDA 1D/2D/3D production builds, all 18 maintained precision runs,
3D MP5 magnetic AMR and exact checkpoint restart pass. GCC CPU32 precision
and CPU64 Sod analytic validation also pass. Matched timing shows a useful
CUDA gain on the larger RMHD wave, with startup dominating short cheap cases.
The 3D PWL control reaches its AMR budget on both CPU precisions and CUDA; it
is retained as a failed control. Cycle 452's non-RDC candidate passes all
dimensional builds, the precision gate, fresh 3D MP5 CT-AMR, and all four
default Compute Sanitizer tools on a small 1D RMHD wave. Its exact restart
and 3D sanitizer follow-up remain incomplete. A strongly magnetized hybrid
SLy preflight fails both old/new CUDA and CPU FP32 while CPU FP64 passes;
the broader RMHD quick suite therefore does not pass. Compile cost is still
impractical. Exact evidence is in those cycles and their reports; these gates
do not certify every GRMHD configuration or multi-device scaling.

CUDA profiling on this workstation can use the Linux target collector bundled
with Windows Nsight Systems 2026.3.2, directly under
`/mnt/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-linux-x64/`.
Cycle452 verifies CUDA-only captures with `--sample=none --cpuctxsw=none`;
CPU system-wide sampling is a separate permission boundary. Nsight Compute
2026.3.0 also includes a Linux collector, but its installation path must have
no spaces, and actual counter collection currently fails with
`ERR_NVGPUCTRPERM`. Counter permission is controlled by the Windows NVIDIA
driver; installing another Linux collector alone does not resolve it.
The temporary collector copy used to establish this boundary was removed.

The CUDA profile exports `compile_commands.json`. Use the existing
`tools/profile_compile.py --trace` with an actual source owner and dimension
for isolated NVCC phase measurements, then point the installed CUDA
`ctadvisor --trace-file-path <trace.json>` at that trace. Keep uncached owner
replays, full builds, cache-hit rebuilds and no-op builds distinct. NVCC and
PTXAS have separate split-compilation controls; measure them within the total
compile-worker and memory budget before adopting profile defaults.
Both compile profiling and target benchmarking now default to a 300 s hard
timeout, with process-group cleanup. Start with one owner and retain the
configured optimized production flags. A successful no-op build establishes
dependency stability, not acceptable cold or changed-kernel compile cost.

For an HPC port, keep the shared source and AMReX execution path; choose the
site compiler, GPU architecture, MPI implementation and scheduler in a machine
profile. AMReX normally uses one rank per GPU. Verify the actual device mapping,
then run the existing precision, magnetic AMR and restart controls across ranks
before measuring scaling. WSL's two-rank, one-device control establishes only
host-staged MPI behavior. Independent devices, GPU-aware MPI and inter-node
communication require tests on that hardware; cross-backend speed ratios do not
measure strong-scaling efficiency. See the
[AMReX GPU execution model](https://amrex-codes.github.io/amrex/docs_html/GPU.html).

The workstation has 48 GiB host RAM. Its `.wslconfig` sets a 36 GiB memory
ceiling (about 35 GiB usable), leaving host headroom; WSL allocates memory on
demand. The RTX has a separate 8 GiB device memory budget. Host RAM is
useful for builds, CPU runs and staging; it does not make those 8 GiB of VRAM
larger. This WSL device supports managed allocations but reports no concurrent
managed CPU/GPU access. Preserve explicit transfer/synchronization boundaries.

Platform references: [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
and [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
