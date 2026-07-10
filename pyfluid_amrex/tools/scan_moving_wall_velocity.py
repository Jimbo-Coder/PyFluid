#!/usr/bin/env python3
"""Scan translating box-piston velocity for moving-solid stress testing."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from validation_common import mpi_prefix

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "hydro_moving_wall_smush_2d.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "moving_wall_velocity_scan"
DEFAULT_RESULTS = DEFAULT_OUTPUT / "results.tsv"
DEFAULT_PLOT = ROOT / "plots" / "current" / "scans" / "moving_wall_velocity_scan_001.png"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "geometry.is_periodic=0 1",
   "amr.n_cell=256 128",
   "amr.max_level=0",
   "amr.max_grid_size=64",
   "amr.blocking_factor=16",
   "pyfluid.physics=hydro",
   "pyfluid.ct=off",
   "pyfluid.ic=uniform",
   "pyfluid.rho0=1.0",
   "pyfluid.pressure0=1.0",
   "pyfluid.v1_0=0.0",
   "pyfluid.v2_0=0.0",
   "pyfluid.v3_0=0.0",
   "pyfluid.gamma=1.4",
   "pyfluid.cfl=0.9",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hllc",
   "pyfluid.timestep=ssprk3",
   "pyfluid.bc=outflow",
   "pyfluid.bc_xlo=outflow",
   "pyfluid.bc_xhi=reflecting",
   "pyfluid.bc_ylo=periodic",
   "pyfluid.bc_yhi=periodic",
   "pyfluid.diffusion=none",
   "pyfluid.subcycling=off",
   "pyfluid.solid_geometry=boxes",
   "pyfluid.solid_motion=translate",
   "pyfluid.solid_wall_condition=slip",
   "pyfluid.solid_box_lo=-1.0 0.0",
   "pyfluid.solid_box_hi=0.0 1.0",
   "pyfluid.scalar_int=25",
   "pyfluid.plot_int=-1",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.aux_gf_int=-1",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def read_last_scalar(path: Path) -> dict[str, str]:
   with path.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   return rows[-1] if rows else {}


def run_case(
   velocity: float,
   displacement: float,
   executable: Path,
   input_file: Path,
   output_dir: Path,
   ranks: int,
   force: bool,
) -> dict[str, object]:
   case_dir = output_dir / f"vx_{velocity:g}".replace(".", "p")
   if force and case_dir.exists():
      shutil.rmtree(case_dir)
   case_dir.mkdir(parents=True, exist_ok=True)
   final_time = displacement / velocity
   command = [
      *mpi_prefix(ranks),
      str(executable),
      str(input_file),
      f"pyfluid.output_dir={repo_relative(case_dir)}",
      *BASE_OVERRIDES,
      f"pyfluid.solid_velocity_1={velocity:.17g}",
      f"pyfluid.final_time={final_time:.17g}",
      "pyfluid.max_step=5000",
   ]
   completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   average_match = AVERAGE_RE.search(completed.stdout)
   sim_time_match = SIM_TIME_RE.search(completed.stdout)
   scalar_path = case_dir / "scalars.tsv"
   final = read_last_scalar(scalar_path) if scalar_path.exists() else {}
   gamma = 1.4
   sound_speed = math.sqrt(gamma)
   return {
      "velocity": velocity,
      "mach": velocity / sound_speed,
      "target_displacement": displacement,
      "final_time": final_time,
      "returncode": completed.returncode,
      "step": final.get("step", math.nan),
      "time": final.get("time", math.nan),
      "dt": final.get("dt", math.nan),
      "mass_rel_drift": final.get("mass_rel_drift", math.nan),
      "energy_rel_drift": final.get("energy_rel_drift", math.nan),
      "mom1": final.get("mom1", math.nan),
      "min_rho": final.get("min_rho", math.nan),
      "min_pressure": final.get("min_pressure", math.nan),
      "mcells_s": float(average_match.group(1)) if average_match else math.nan,
      "sim_time_per_s": float(sim_time_match.group(1)) if sim_time_match else math.nan,
      "run_dir": repo_relative(case_dir),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def add_status(rows: list[dict[str, object]], mass_tol: float) -> None:
   for row in rows:
      failures: list[str] = []
      if int(row["returncode"]) != 0:
         failures.append(f"returncode={row['returncode']}")
      for key in ("mass_rel_drift", "energy_rel_drift", "mom1", "min_rho", "min_pressure"):
         try:
            value = float(row[key])
         except (TypeError, ValueError):
            value = math.nan
         if not math.isfinite(value):
            failures.append(f"{key}=nan")
      mass = float(row["mass_rel_drift"]) if not failures else math.nan
      if math.isfinite(mass) and mass > mass_tol:
         failures.append(f"mass_rel_drift={mass:.3e} > {mass_tol:.3e}")
      if math.isfinite(float(row["min_rho"])) and float(row["min_rho"]) <= 0.0:
         failures.append(f"min_rho={float(row['min_rho']):.3e}")
      if math.isfinite(float(row["min_pressure"])) and float(row["min_pressure"]) <= 0.0:
         failures.append(f"min_pressure={float(row['min_pressure']):.3e}")
      row["status"] = "pass" if not failures else "fail"
      row["failures"] = "; ".join(failures)


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "velocity",
      "mach",
      "status",
      "returncode",
      "target_displacement",
      "final_time",
      "step",
      "time",
      "dt",
      "mass_rel_drift",
      "energy_rel_drift",
      "mom1",
      "min_rho",
      "min_pressure",
      "mcells_s",
      "sim_time_per_s",
      "run_dir",
      "failures",
   ]
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, object]], title: str) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   ok = [row for row in rows if row["status"] == "pass"]
   velocities = np.asarray([float(row["velocity"]) for row in ok])
   mach = np.asarray([float(row["mach"]) for row in ok])
   mass = np.asarray([float(row["mass_rel_drift"]) for row in ok])
   energy = np.asarray([float(row["energy_rel_drift"]) for row in ok])
   mom1 = np.asarray([float(row["mom1"]) for row in ok])
   min_rho = np.asarray([float(row["min_rho"]) for row in ok])
   min_p = np.asarray([float(row["min_pressure"]) for row in ok])
   mcells = np.asarray([float(row["mcells_s"]) for row in ok])
   sim_time = np.asarray([float(row["sim_time_per_s"]) for row in ok])
   steps = np.asarray([float(row["step"]) for row in ok])

   fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
   fig.suptitle(title)
   axs[0, 0].loglog(velocities, np.maximum(mass, 1.0e-18), marker="o", label="mass")
   axs[0, 0].set_title("Mass Error")
   axs[0, 0].set_xlabel("wall speed")
   axs[0, 0].set_ylabel("relative error")
   axs[0, 0].legend()

   axs[0, 1].plot(velocities, energy, marker="o", label="energy")
   axs[0, 1].plot(velocities, mom1, marker="o", label="mom1")
   axs[0, 1].set_title("Wall Work / Impulse")
   axs[0, 1].set_xlabel("wall speed")
   axs[0, 1].legend()

   axs[0, 2].plot(velocities, min_rho, marker="o", label="min rho")
   axs[0, 2].plot(velocities, min_p, marker="o", label="min P")
   axs[0, 2].set_title("Floors")
   axs[0, 2].set_xlabel("wall speed")
   axs[0, 2].legend()

   axs[1, 0].plot(velocities, mach, marker="o")
   axs[1, 0].set_title("Piston Mach")
   axs[1, 0].set_xlabel("wall speed")
   axs[1, 0].set_ylabel("Mach")

   axs[1, 1].plot(velocities, steps, marker="o")
   axs[1, 1].set_title("Steps To Fixed Displacement")
   axs[1, 1].set_xlabel("wall speed")

   axs[1, 2].plot(velocities, mcells, marker="o", label="Mcells/s")
   axs2 = axs[1, 2].twinx()
   axs2.plot(velocities, sim_time, marker="s", color="tab:green", label="sim time/s")
   axs[1, 2].set_title("Performance")
   axs[1, 2].set_xlabel("wall speed")
   axs[1, 2].set_ylabel("Mcells/s")
   axs2.set_ylabel("sim time/s")
   lines, labels = axs[1, 2].get_legend_handles_labels()
   lines2, labels2 = axs2.get_legend_handles_labels()
   axs[1, 2].legend(lines + lines2, labels + labels2)

   for ax in axs.flat:
      ax.grid(True, which="major", alpha=0.25)
   fig.savefig(path, dpi=180)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
   parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--displacement", type=float, default=0.12)
   parser.add_argument("--velocities", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
   parser.add_argument("--mass-tol", type=float, default=1.0e-12)
   args = parser.parse_args()

   rows = [
      run_case(v, args.displacement, args.executable, args.input, args.output_dir, args.ranks, args.force)
      for v in args.velocities
   ]
   add_status(rows, args.mass_tol)
   write_results(args.results, rows)
   plot_results(
      args.plot,
      rows,
      f"Moving wall velocity scan: PWL/HLLC/SSPRK3, displacement={args.displacement:g}, ranks={args.ranks}",
   )
   for row in rows:
      if row["status"] == "pass":
         print(
            f"PASS vx={float(row['velocity']):g} Mach={float(row['mach']):.2f}: "
            f"mass={float(row['mass_rel_drift']):.2e}, energy={float(row['energy_rel_drift']):.2e}, "
            f"minP={float(row['min_pressure']):.3g}, {float(row['mcells_s']):.2f} Mcells/s"
         )
      else:
         print(f"FAIL vx={row['velocity']}: {row['failures']}")
         print(row["tail"])
   print(f"wrote {repo_relative(args.results)}")
   print(f"wrote {repo_relative(args.plot)}")
   return 0 if all(row["status"] == "pass" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
