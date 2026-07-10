#!/usr/bin/env python3
"""Validate static hydro obstacle walls on AMR.

PyFluid AMReX obstacles are cell masks: analytic geometry is sampled at cell
centers, then the numerical wall is the set of fluid-solid cell faces. These
checks exercise that discrete wall model with AMR regridding and verify that a
closed, uniform state remains static and conservative.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix
from validation_common import read_float_tsv, read_tsv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "obstacle_wall_static_validation"
DEFAULT_RESULTS = DEFAULT_OUTPUT / "results.tsv"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


@dataclass(frozen=True)
class Case:
   name: str
   geometry: str
   subcycling: str
   overrides: tuple[str, ...]


CASES = (
   Case(
      name="box_subcycling_on",
      geometry="central box",
      subcycling="on",
      overrides=(
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_box_lo=0.40 0.35",
         "pyfluid.solid_box_hi=0.60 0.65",
         "pyfluid.subcycling=on",
      ),
   ),
   Case(
      name="box_subcycling_off",
      geometry="central box",
      subcycling="off",
      overrides=(
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_box_lo=0.40 0.35",
         "pyfluid.solid_box_hi=0.60 0.65",
         "pyfluid.subcycling=off",
      ),
   ),
   Case(
      name="circle_subcycling_on",
      geometry="central circle",
      subcycling="on",
      overrides=(
         "pyfluid.solid_geometry=circle",
         "pyfluid.solid_center_x=0.50",
         "pyfluid.solid_center_y=0.50",
         "pyfluid.solid_radius=0.14",
         "pyfluid.subcycling=on",
      ),
   ),
   Case(
      name="circle_no_slip_viscous",
      geometry="central no-slip circle",
      subcycling="on",
      overrides=(
         "pyfluid.solid_geometry=circle",
         "pyfluid.solid_center_x=0.50",
         "pyfluid.solid_center_y=0.50",
         "pyfluid.solid_radius=0.14",
         "pyfluid.solid_wall_condition=no_slip",
         "pyfluid.diffusion=explicit",
         "pyfluid.viscosity=5.0e-4",
         "pyfluid.subcycling=on",
      ),
   ),
)


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "geometry.is_periodic=0 0",
   "amr.n_cell=64 64",
   "amr.max_level=2",
   "amr.ref_ratio=2 2",
   "amr.max_grid_size=32",
   "amr.blocking_factor=8",
   "amr.regrid_int=2",
   "amr.n_error_buf=2 2",
   "amr.tag_metric=none",
   "amr.total_cell_budget=180000",
   "pyfluid.physics=hydro",
   "pyfluid.ct=off",
   "pyfluid.ic=uniform",
   "pyfluid.rho0=1.0",
   "pyfluid.pressure0=1.0",
   "pyfluid.v1_0=0.0",
   "pyfluid.v2_0=0.0",
   "pyfluid.v3_0=0.0",
   "pyfluid.bc=reflecting",
   "pyfluid.bc_xlo=reflecting",
   "pyfluid.bc_xhi=reflecting",
   "pyfluid.bc_ylo=reflecting",
   "pyfluid.bc_yhi=reflecting",
   "pyfluid.final_time=0.04",
   "pyfluid.max_step=100000",
   "pyfluid.cfl=0.9",
   "pyfluid.ng=2",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hllc",
   "pyfluid.timestep=ssprk3",
   "pyfluid.diffusion=none",
   "pyfluid.subcycling_steps=spatial",
   "pyfluid.scalar_int=2",
   "pyfluid.plot_int=-1",
   "pyfluid.aux_gfs=solid_mask",
   "pyfluid.aux_gf_int=0",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.profile=on",
   "pyfluid.profile_detail=basic",
)


def repo_relative(path: Path) -> Path:
   absolute = path if path.is_absolute() else ROOT / path
   return absolute.resolve().relative_to(ROOT)


def read_last_scalar(path: Path) -> dict[str, float]:
   rows = read_float_tsv(path)
   if not rows:
      raise RuntimeError(f"{path} is empty")
   return rows[-1]


def read_max_active_level(path: Path) -> int:
   rows = read_tsv(path, require_complete=True)
   if not rows:
      return -1
   return max(int(float(row["active_finest_level"])) for row in rows)


def run_case(case: Case, executable: Path, input_file: Path, output_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = output_root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)

   command = [
      *mpi_prefix(ranks),
      str(executable),
      str(input_file),
      f"pyfluid.output_dir={repo_relative(run_dir)}",
      *BASE_OVERRIDES,
      *case.overrides,
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
   scalar_path = run_dir / "scalars.tsv"
   amr_path = run_dir / "amr_levels.tsv"
   final = read_last_scalar(scalar_path) if scalar_path.exists() else {}
   mom_abs = sum(abs(float(final.get(key, math.nan))) for key in ("mom1", "mom2", "mom3"))
   return {
      "case": case.name,
      "geometry": case.geometry,
      "subcycling": case.subcycling,
      "returncode": completed.returncode,
      "mcells_s": float(average_match.group(1)) if average_match else math.nan,
      "sim_time_per_s": float(sim_time_match.group(1)) if sim_time_match else math.nan,
      "mass_rel_drift": final.get("mass_rel_drift", math.nan),
      "energy_rel_drift": final.get("energy_rel_drift", math.nan),
      "momentum_l1": mom_abs,
      "min_rho": final.get("min_rho", math.nan),
      "min_pressure": final.get("min_pressure", math.nan),
      "active_finest_level": read_max_active_level(amr_path) if amr_path.exists() else -1,
      "run_dir": repo_relative(run_dir),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def add_status(rows: list[dict[str, object]], mass_tol: float, energy_tol: float, momentum_tol: float, floor_tol: float) -> None:
   for row in rows:
      failures: list[str] = []
      if int(row["returncode"]) != 0:
         failures.append(f"returncode={row['returncode']}")
      for key, tol in (("mass_rel_drift", mass_tol), ("energy_rel_drift", energy_tol), ("momentum_l1", momentum_tol)):
         value = float(row[key])
         if not math.isfinite(value) or value > tol:
            failures.append(f"{key}={value:.3e} > {tol:.3e}")
      for key in ("min_rho", "min_pressure"):
         value = float(row[key])
         if not math.isfinite(value) or abs(value - 1.0) > floor_tol:
            failures.append(f"{key}={value:.12g}")
      if int(row["active_finest_level"]) < 1:
         failures.append("AMR did not refine the solid-fluid interface")
      row["status"] = "pass" if not failures else "fail"
      row["failures"] = "; ".join(failures)


def write_results(rows: list[dict[str, object]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "status",
      "geometry",
      "subcycling",
      "mcells_s",
      "sim_time_per_s",
      "mass_rel_drift",
      "energy_rel_drift",
      "momentum_l1",
      "min_rho",
      "min_pressure",
      "active_finest_level",
      "run_dir",
      "failures",
      "tail",
   ]
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      for row in rows:
         writer.writerow(row)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
   parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--allow-fail", action="store_true")
   parser.add_argument("--mass-tol", type=float, default=1.0e-12)
   parser.add_argument("--energy-tol", type=float, default=1.0e-12)
   parser.add_argument("--momentum-tol", type=float, default=1.0e-11)
   parser.add_argument("--floor-tol", type=float, default=1.0e-7)
   args = parser.parse_args()

   if args.force and args.output_dir.exists():
      shutil.rmtree(args.output_dir)
   rows = [run_case(case, args.executable, args.input, args.output_dir, args.ranks, args.force) for case in CASES]
   add_status(rows, args.mass_tol, args.energy_tol, args.momentum_tol, args.floor_tol)
   write_results(rows, args.results)

   for row in rows:
      print(
         f"{str(row['status']).upper():>4} {row['case']}: "
         f"mass={float(row['mass_rel_drift']):.2e}, "
         f"energy={float(row['energy_rel_drift']):.2e}, "
         f"|mom|={float(row['momentum_l1']):.2e}, "
         f"Lmax={int(row['active_finest_level'])}, "
         f"{float(row['mcells_s']):.2f} Mcells/s"
      )
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.results)}")
   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
