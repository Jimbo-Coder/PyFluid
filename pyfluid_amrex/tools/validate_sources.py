#!/usr/bin/env python3
"""Validate AMReX source terms against closed-form uniform-flow solutions."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix
from validation_common import (
   DRIVEN_POLICY,
   append_validation_history,
   read_tsv,
   scalar_validation_row,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE_2D = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
RUN_ROOT = ROOT / "runs" / "current" / "source_validation"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"
SUMMARY = RUN_ROOT / "summary.tsv"


@dataclass(frozen=True)
class UniformAccelerationCase:
   name: str = "uniform_acceleration_2d"
   dim: int = 2
   ranks: int = 3
   nx: int = 64
   ny: int = 64
   final_time: float = 0.2
   rho0: float = 1.0
   pressure0: float = 1.0
   v1_0: float = 0.10
   v2_0: float = -0.04
   ax: float = 0.20
   ay: float = -0.15


def case_input(case: UniformAccelerationCase, run_dir: Path) -> str:
   return f"""pyfluid.run_name = {case.name}
pyfluid.output_dir = {run_dir.relative_to(ROOT)}

geometry.prob_lo = 0.0 0.0
geometry.prob_hi = 1.0 1.0
geometry.is_periodic = 1 1

amr.n_cell = {case.nx} {case.ny}
amr.max_level = 0
amr.max_grid_size = 32
amr.blocking_factor = 8

pyfluid.max_step = 100000
pyfluid.final_time = {case.final_time:.17g}
pyfluid.ng = 1
pyfluid.cfl = 0.9
pyfluid.eos = ideal_gas
pyfluid.gamma = 1.6666666666666667

pyfluid.physics = hydro
pyfluid.ic = uniform
pyfluid.rho0 = {case.rho0:.17g}
pyfluid.pressure0 = {case.pressure0:.17g}
pyfluid.v1_0 = {case.v1_0:.17g}
pyfluid.v2_0 = {case.v2_0:.17g}
pyfluid.v3_0 = 0.0

pyfluid.recon = pwc
pyfluid.riemann = llf
pyfluid.timestep = rk4
pyfluid.bc = periodic
pyfluid.diffusion = none
pyfluid.source = uniform_acceleration
pyfluid.source_a1 = {case.ax:.17g}
pyfluid.source_a2 = {case.ay:.17g}
pyfluid.source_a3 = 0.0

pyfluid.scalar_int = 8
pyfluid.validation_policy = driven
pyfluid.plot_int = -1
pyfluid.checkpoint_int = -1
pyfluid.aux_gfs = none
pyfluid.aux_gf_int = -1
pyfluid.aux_gf_compute_int = -1
"""


def run_case(case: UniformAccelerationCase, force: bool) -> tuple[Path, Path, int, float, str]:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = RUN_ROOT / "inputs" / f"{case.name}.in"
   input_path.parent.mkdir(parents=True, exist_ok=True)
   input_path.write_text(case_input(case, run_dir))
   command = [
      *mpi_prefix(case.ranks),
      str(EXECUTABLE_2D),
      str(input_path),
   ]
   env = dict(os.environ)
   env.setdefault("OMP_NUM_THREADS", "1")
   start = time.perf_counter()
   completed = subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )
   elapsed = time.perf_counter() - start
   (run_dir / "run.log").write_text(completed.stdout)
   return run_dir, input_path, completed.returncode, elapsed, completed.stdout


def analytic_errors(case: UniformAccelerationCase, run_dir: Path) -> dict[str, float]:
   rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
   if len(rows) < 2:
      raise RuntimeError(f"{run_dir / 'scalars.tsv'} does not contain initial and final scalar rows")

   initial = rows[0]
   final = rows[-1]
   mass0 = float(initial["mass"])
   mass = float(final["mass"])
   t = float(final["time"]) - float(initial["time"])
   mom0 = (
      float(initial["mom1"]),
      float(initial["mom2"]),
      float(initial["mom3"]),
   )
   mom = (
      float(final["mom1"]),
      float(final["mom2"]),
      float(final["mom3"]),
   )
   energy0 = float(initial["energy"])
   energy = float(final["energy"])
   accel = (case.ax, case.ay, 0.0)

   mom_exact = tuple(mom0[d] + mass0 * accel[d] * t for d in range(3))
   energy_exact = (
      energy0
      + sum(mom0[d] * accel[d] for d in range(3)) * t
      + 0.5 * mass0 * sum(accel[d] * accel[d] for d in range(3)) * t * t
   )

   momentum_abs_error = math.sqrt(sum((mom[d] - mom_exact[d]) ** 2 for d in range(3)))
   return {
      "mass_abs_error": abs(mass - mass0),
      "momentum_abs_error": momentum_abs_error,
      "energy_abs_error": abs(energy - energy_exact),
      "energy_rel_error": abs(energy - energy_exact) / max(abs(energy_exact), 1.0),
      "time": t,
      "mass": mass,
      "mom1": mom[0],
      "mom2": mom[1],
      "energy": energy,
      "mom1_exact": mom_exact[0],
      "mom2_exact": mom_exact[1],
      "energy_exact": energy_exact,
   }


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--mass-tol", type=float, default=1.0e-12)
   parser.add_argument("--momentum-tol", type=float, default=1.0e-10)
   parser.add_argument("--energy-tol", type=float, default=1.0e-10)
   args = parser.parse_args()

   case = UniformAccelerationCase(ranks=args.ranks)
   run_dir, input_path, returncode, elapsed, stdout = run_case(case, args.force)
   rows: list[dict[str, object]] = []
   scalar_row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=input_path,
      policy=DRIVEN_POLICY,
      mhd=False,
      dim=case.dim,
      ranks=case.ranks,
      base_input=str(input_path.relative_to(ROOT)),
      kind="analytic",
   )
   if returncode != 0:
      scalar_row["status"] = "FAIL"
      scalar_row["reason"] = f"solver returned {returncode}"
      scalar_row["returncode"] = returncode
      scalar_row["seconds"] = f"{elapsed:.6g}"
      scalar_row["tail"] = " | ".join(stdout.splitlines()[-6:])
      rows.append(scalar_row)
      write_tsv(SUMMARY, rows, ["case", "status", "reason", "mass_abs_error", "momentum_abs_error", "energy_abs_error", "avg_mcells_per_s", "avg_sim_time_per_s"])
      append_validation_history(HISTORY, rows)
      print(scalar_row["reason"])
      print(scalar_row["tail"])
      return 1

   errors = analytic_errors(case, run_dir)
   failures = []
   if errors["mass_abs_error"] > args.mass_tol:
      failures.append(f"mass_abs_error={errors['mass_abs_error']:.3e} > {args.mass_tol:.3e}")
   if errors["momentum_abs_error"] > args.momentum_tol:
      failures.append(f"momentum_abs_error={errors['momentum_abs_error']:.3e} > {args.momentum_tol:.3e}")
   if errors["energy_abs_error"] > args.energy_tol:
      failures.append(f"energy_abs_error={errors['energy_abs_error']:.3e} > {args.energy_tol:.3e}")
   if scalar_row["status"] != "PASS":
      failures.append(str(scalar_row["reason"]))

   row = {
      **scalar_row,
      **errors,
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "returncode": returncode,
      "seconds": f"{elapsed:.6g}",
      "log": str((run_dir / "run.log").relative_to(ROOT)),
      "description": "uniform periodic hydro acceleration exact solution",
   }
   rows.append(row)
   write_tsv(
      SUMMARY,
      rows,
      [
         "case",
         "status",
         "reason",
         "time",
         "mass_abs_error",
         "momentum_abs_error",
         "energy_abs_error",
         "energy_rel_error",
         "avg_mcells_per_s",
         "avg_sim_time_per_s",
         "log",
      ],
   )
   append_validation_history(HISTORY, rows)

   print(
      f"{row['status']} {case.name}: "
      f"mass_err={errors['mass_abs_error']:.3e} "
      f"mom_err={errors['momentum_abs_error']:.3e} "
      f"energy_err={errors['energy_abs_error']:.3e} "
      f"avg={float(row['avg_mcells_per_s']):.2f} Mcells/s "
      f"sim_time/s={float(row['avg_sim_time_per_s']):.4g}"
   )
   if failures:
      print("; ".join(failures))
      return 1
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
