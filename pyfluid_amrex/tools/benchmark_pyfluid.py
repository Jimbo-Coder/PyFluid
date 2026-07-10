#!/usr/bin/env python3
"""Run fixed PyFluid AMReX performance regression cases.

The case list below is the single place to add or adjust canonical performance
checks. Evolution writes only run outputs under runs/current; this script's
postprocessing is just stdout parsing plus a compact TSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix
from validation_common import append_performance_history as append_common_performance_history


@dataclass(frozen=True)
class PerfCase:
   name: str
   executable: Path
   inputs: Path
   ranks: int
   overrides: tuple[str, ...]
   baseline_mcells_s: float
   baseline_label: str
   warn_fraction: float = 0.85


HYDRO2D_FAST_OVERRIDES = (
   "amr.n_cell=512 512",
   "amr.max_level=0",
   "amr.max_grid_size=64",
   "pyfluid.max_step=64",
   "pyfluid.ng=1",
   "pyfluid.eos=ideal_gas",
   "pyfluid.recon=pwc",
   "pyfluid.riemann=llf",
   "pyfluid.timestep=euler",
   "pyfluid.scalar_int=64",
   "pyfluid.plot_int=-1",
   "pyfluid.checkpoint_int=-1",
)


# Three-repeat medians from the MPI-enabled AMReX build on 2026-07-09. Earlier
# values came from a serial AMReX cache launched under mpirun and are not
# comparable.
MPI_BASELINE_LABEL = "real_mpi_2026-07-09_clean"


CASES = [
   PerfCase(
      name="hydro2d_512_pwc_llf_euler_3rank",
      executable=Path("build/pyfluid_amrex/pyfluid_amrex_2d"),
      inputs=Path("pyfluid_amrex/inputs/hydro_sine_2d.in"),
      ranks=3,
      baseline_mcells_s=109.9095765,
      baseline_label=MPI_BASELINE_LABEL,
      overrides=HYDRO2D_FAST_OVERRIDES,
   ),
   PerfCase(
      name="hydro2d_512_pwc_llf_euler_4rank",
      executable=Path("build/pyfluid_amrex/pyfluid_amrex_2d"),
      inputs=Path("pyfluid_amrex/inputs/hydro_sine_2d.in"),
      ranks=4,
      baseline_mcells_s=137.0855579,
      baseline_label=MPI_BASELINE_LABEL,
      overrides=HYDRO2D_FAST_OVERRIDES,
   ),
   PerfCase(
      name="hydro3d_64_pwc_llf_euler",
      executable=Path("build/pyfluid_amrex/pyfluid_amrex_3d"),
      inputs=Path("pyfluid_amrex/inputs/hydro_sine_3d.in"),
      ranks=3,
      baseline_mcells_s=6.373059139,
      baseline_label=MPI_BASELINE_LABEL,
      overrides=(
         "amr.n_cell=64 64 64",
         "amr.max_level=0",
         "amr.max_grid_size=32",
         "pyfluid.max_step=128",
         "pyfluid.ng=1",
         "pyfluid.eos=ideal_gas",
         "pyfluid.recon=pwc",
         "pyfluid.riemann=llf",
         "pyfluid.timestep=euler",
         "pyfluid.scalar_int=128",
         "pyfluid.plot_int=-1",
         "pyfluid.checkpoint_int=-1",
      ),
   ),
   PerfCase(
      name="hydro3d_64_wenoz_hllc_rk4",
      executable=Path("build/pyfluid_amrex/pyfluid_amrex_3d"),
      inputs=Path("pyfluid_amrex/inputs/hydro_sine_3d.in"),
      ranks=3,
      baseline_mcells_s=0.1900616783,
      baseline_label=MPI_BASELINE_LABEL,
      overrides=(
         "amr.n_cell=64 64 64",
         "amr.max_level=0",
         "amr.max_grid_size=32",
         "pyfluid.max_step=64",
         "pyfluid.ng=3",
         "pyfluid.eos=ideal_gas",
         "pyfluid.recon=wenoz",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=rk4",
         "pyfluid.scalar_int=64",
         "pyfluid.plot_int=-1",
         "pyfluid.checkpoint_int=-1",
      ),
   ),
]

DEFAULT_HISTORY = Path("runs/current/performance_history.tsv")

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")
MPI_PROCESSES_RE = re.compile(r"MPI initialized with\s+(\d+) MPI processes")


def cases_for_build_dir(build_dir: Path, only: set[str] | None) -> list[PerfCase]:
   cases = []
   known = {case.name for case in CASES}
   if only:
      unknown = sorted(only - known)
      if unknown:
         raise SystemExit(f"unknown benchmark case(s): {', '.join(unknown)}")
   for case in CASES:
      if only and case.name not in only:
         continue
      cases.append(
         PerfCase(
            name=case.name,
            executable=build_dir / case.executable.name,
            inputs=case.inputs,
            ranks=case.ranks,
            overrides=case.overrides,
            baseline_mcells_s=case.baseline_mcells_s,
            baseline_label=case.baseline_label,
            warn_fraction=case.warn_fraction,
         )
      )
   return cases


def read_profile_summary(run_dir: Path) -> dict[str, object]:
   profile_path = run_dir / "profile.tsv"
   if not profile_path.exists():
      return {
         "dominant_stage": "",
         "dominant_stage_fraction": float("nan"),
         "timestep_fraction": float("nan"),
         "flux_fraction": float("nan"),
      }

   with profile_path.open(newline="") as f:
      reader = csv.DictReader(f, delimiter="\t")
      rows = list(reader)

   fraction_key = "fraction_of_run_wall" if rows and "fraction_of_run_wall" in rows[0] else "fraction"
   fractions: dict[str, float] = {}
   for row in rows:
      stage = row.get("stage", "")
      if not stage:
         continue
      try:
         fractions[stage] = float(row.get(fraction_key, "nan"))
      except ValueError:
         fractions[stage] = float("nan")

   dominant_stage = ""
   dominant_fraction = float("nan")
   finite_fractions = {
      stage: fraction for stage, fraction in fractions.items()
      if stage and fraction == fraction
   }
   if finite_fractions:
      dominant_stage, dominant_fraction = max(finite_fractions.items(), key=lambda item: item[1])

   return {
      "dominant_stage": dominant_stage,
      "dominant_stage_fraction": dominant_fraction,
      "timestep_fraction": fractions.get("timestep", float("nan")),
      "flux_fraction": fractions.get("flux", fractions.get("fused_rhs", float("nan"))),
   }


def _run_case_once(case: PerfCase, root: Path, force: bool, label: str) -> dict[str, object]:
   run_dir = root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   command = [
      *mpi_prefix(case.ranks),
      str(case.executable),
      str(case.inputs),
      f"pyfluid.output_dir={run_dir}",
      *case.overrides,
   ]
   completed = subprocess.run(
      command,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   match = AVERAGE_RE.search(completed.stdout)
   sim_match = SIM_TIME_RE.search(completed.stdout)
   mpi_match = MPI_PROCESSES_RE.search(completed.stdout)
   mcells_s = float(match.group(1)) if match else float("nan")
   sim_time_per_s = float(sim_match.group(1)) if sim_match else float("nan")
   ratio = mcells_s / case.baseline_mcells_s
   mpi_processes = int(mpi_match.group(1)) if mpi_match else 0
   status = "pass" if completed.returncode == 0 and ratio >= case.warn_fraction else "warn"
   if completed.returncode != 0 or not match or mpi_processes != case.ranks:
      status = "fail"
   return {
      "case": case.name,
      "label": label,
      "status": status,
      "ranks": case.ranks,
      "mcells_s": mcells_s,
      "sim_time_per_s": sim_time_per_s,
      "baseline_mcells_s": case.baseline_mcells_s,
      "baseline_label": case.baseline_label,
      "ratio": ratio,
      "warn_fraction": case.warn_fraction,
      "returncode": completed.returncode,
      "execution_mode": "mpi" if mpi_processes == case.ranks else "invalid",
      "mpi_processes": mpi_processes,
      "run_dir": run_dir,
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
      **read_profile_summary(run_dir),
   }


def run_case(case: PerfCase, root: Path, force: bool, label: str, repeats: int) -> dict[str, object]:
   rows = [_run_case_once(case, root, force, label) for _ in range(repeats)]
   finite_mcells_s = [float(row["mcells_s"]) for row in rows if float(row["mcells_s"]) == float(row["mcells_s"])]
   finite_sim_time = [float(row["sim_time_per_s"]) for row in rows if float(row["sim_time_per_s"]) == float(row["sim_time_per_s"])]
   mcells_s = statistics.median(finite_mcells_s) if finite_mcells_s else float("nan")
   sim_time_per_s = statistics.median(finite_sim_time) if finite_sim_time else float("nan")
   ratio = mcells_s / case.baseline_mcells_s
   status = "pass" if ratio >= case.warn_fraction else "warn"
   if any(row["status"] == "fail" for row in rows) or not finite_mcells_s:
      status = "fail"
   profile_row = rows[-1]
   return {
      **profile_row,
      "status": status,
      "repeats": repeats,
      "mcells_s": mcells_s,
      "sim_time_per_s": sim_time_per_s,
      "ratio": ratio,
      "returncode": max(int(row["returncode"]) for row in rows),
   }


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "label",
      "status",
      "ranks",
      "mcells_s",
      "sim_time_per_s",
      "baseline_mcells_s",
      "baseline_label",
      "ratio",
      "warn_fraction",
      "returncode",
      "execution_mode",
      "mpi_processes",
      "dominant_stage",
      "dominant_stage_fraction",
      "timestep_fraction",
      "flux_fraction",
      "run_dir",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      for row in rows:
         writer.writerow(row)


def append_performance_history(rows: list[dict[str, object]], history: Path) -> None:
   append_common_performance_history(history, [{"kind": "performance_regression", **row} for row in rows])


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--build-dir", type=Path, default=Path("build/pyfluid_amrex"))
   parser.add_argument("--label", default="", help="optional label written to result and history rows")
   parser.add_argument("--repeats", type=int, default=3, help="number of repeats per case; medians are written")
   parser.add_argument("--only", action="append", default=[], choices=[case.name for case in CASES])
   parser.add_argument("--run-dir", type=Path, default=Path("runs/current/pyfluid_amrex_performance"))
   parser.add_argument("--output", type=Path, default=Path("runs/current/pyfluid_amrex_performance/results.tsv"))
   parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--allow-warn", action="store_true", help="return success even when a case is below threshold")
   args = parser.parse_args()
   if args.repeats < 1:
      raise SystemExit("--repeats must be >= 1")

   rows = []
   selected_cases = cases_for_build_dir(args.build_dir, set(args.only) if args.only else None)
   for case in selected_cases:
      row = run_case(case, args.run_dir, args.force, args.label, args.repeats)
      rows.append(row)
      print(
         f"{row['status']:>4} {case.name}: {float(row['mcells_s']):.2f} Mcells/s "
         f"sim_time/s={float(row['sim_time_per_s']):.4g} "
         f"baseline={case.baseline_mcells_s:.2f} ({case.baseline_label}) "
         f"ratio={float(row['ratio']):.2f} mpi={int(row['mpi_processes'])} "
         f"top={row['dominant_stage']}:{float(row['dominant_stage_fraction']):.2f} "
         f"dt={float(row['timestep_fraction']):.2f} flux={float(row['flux_fraction']):.2f}"
      )
      if row["status"] == "fail":
         print(row["tail"])

   write_results(rows, args.output)
   append_performance_history(rows, args.history)
   print(f"wrote {args.output}")
   print(f"appended {args.history}")
   bad = [row for row in rows if row["status"] in ("warn", "fail")]
   return 0 if args.allow_warn or not bad else 1


if __name__ == "__main__":
   raise SystemExit(main())
