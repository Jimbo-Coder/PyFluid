#!/usr/bin/env python3
"""Validate the conservative single-level translating-box piston path."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "hydro_moving_wall_smush_2d.in"
DEFAULT_OUTPUT = ROOT / "runs" / "current" / "moving_solid_validation"
DEFAULT_RESULTS = DEFAULT_OUTPUT / "results.tsv"

AVERAGE_RE = re.compile(r"average=([0-9.eE+-]+)\s+Mcells/s")
LAYOUT_AVERAGE_RE = re.compile(r"layout_average=([0-9.eE+-]+)\s+layout_Mcells/s")
SIM_TIME_RE = re.compile(r"sim_time_per_s=([0-9.eE+-]+)")


@dataclass(frozen=True)
class Case:
   name: str
   expect_success: bool
   overrides: tuple[str, ...]


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "geometry.is_periodic=0 1",
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
   "pyfluid.recon=pwc",
   "pyfluid.riemann=hllc",
   "pyfluid.timestep=euler",
   "pyfluid.bc=outflow",
   "pyfluid.bc_xlo=outflow",
   "pyfluid.bc_xhi=reflecting",
   "pyfluid.bc_ylo=periodic",
   "pyfluid.bc_yhi=periodic",
   "pyfluid.diffusion=none",
   "pyfluid.subcycling=off",
   "pyfluid.scalar_int=8",
   "pyfluid.plot_int=-1",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.aux_gf_int=-1",
)


CASES = (
   Case(
      name="small_box_piston",
      expect_success=True,
      overrides=(
         "amr.n_cell=128 64",
         "pyfluid.max_step=128",
         "pyfluid.final_time=0.04",
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_motion=translate",
         "pyfluid.solid_wall_condition=slip",
         "pyfluid.solid_box_lo=-1.0 0.0",
         "pyfluid.solid_box_hi=0.0 1.0",
         "pyfluid.solid_velocity_1=0.25",
      ),
   ),
   Case(
      name="wall_smush",
      expect_success=True,
      overrides=(
         "amr.n_cell=256 128",
         "pyfluid.max_step=320",
         "pyfluid.final_time=0.12",
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_motion=translate",
         "pyfluid.solid_wall_condition=slip",
         "pyfluid.solid_box_lo=-1.0 0.0",
         "pyfluid.solid_box_hi=0.0 1.0",
         "pyfluid.solid_velocity_1=0.5",
      ),
   ),
   Case(
      name="amr_box_piston_headroom",
      expect_success=True,
      overrides=(
         "geometry.is_periodic=0 1",
         "amr.n_cell=128 64",
         "amr.max_level=6",
         "amr.ref_ratio=2 2 2 2 2 2",
         "amr.max_grid_size=32",
         "amr.blocking_factor=8",
         "amr.regrid_int=4",
         "amr.grid_eff=0.8",
         "amr.n_error_buf=3 3",
         "amr.tag_metric=lohner",
         "amr.tag_threshold=0.12",
         "amr.solid_refine_band_cells=2.0",
         "amr.solid_curvature_error=0.0",
         "amr.min_feature_length=0.00390625",
         "amr.level_cell_budget_side=512",
         "amr.total_cell_budget=250000",
         "amr.state_interp=component-conservative",
         "pyfluid.max_step=100000",
         "pyfluid.final_time=0.12",
         "pyfluid.recon=pwc",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_motion=translate",
         "pyfluid.solid_wall_condition=slip",
         "pyfluid.solid_box_lo=-1.0 0.0",
         "pyfluid.solid_box_hi=0.0 1.0",
         "pyfluid.solid_velocity_1=0.5",
      ),
   ),
   Case(
      name="internal_box_guard",
      expect_success=False,
      overrides=(
         "amr.n_cell=128 64",
         "pyfluid.max_step=128",
         "pyfluid.final_time=0.08",
         "pyfluid.solid_geometry=boxes",
         "pyfluid.solid_motion=translate",
         "pyfluid.solid_wall_condition=slip",
         "pyfluid.solid_box_lo=0.35 0.35",
         "pyfluid.solid_box_hi=0.45 0.65",
         "pyfluid.solid_velocity_1=0.5",
         "pyfluid.bc_xlo=reflecting",
         "pyfluid.bc_xhi=reflecting",
      ),
   ),
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def read_last_scalar(path: Path) -> dict[str, str]:
   with path.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   return rows[-1] if rows else {}


def run_case(case: Case, executable: Path, input_file: Path, output_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = output_root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      input_file,
      run_dir / "run.in",
      parameters_from_assignments(
         [f"pyfluid.output_dir={repo_relative(run_dir)}", *BASE_OVERRIDES, *case.overrides]
      ),
   )
   completed = run_mpi(
      executable,
      [input_path],
      ranks=ranks,
   )
   average_match = AVERAGE_RE.search(completed.stdout)
   layout_average_match = LAYOUT_AVERAGE_RE.search(completed.stdout)
   sim_time_match = SIM_TIME_RE.search(completed.stdout)
   scalar_path = run_dir / "scalars.tsv"
   final = read_last_scalar(scalar_path) if scalar_path.exists() else {}
   tail = "\n".join(completed.stdout.splitlines()[-8:])
   return {
      "case": case.name,
      "expect_success": case.expect_success,
      "returncode": completed.returncode,
      "mcells_s": float(average_match.group(1)) if average_match else math.nan,
      "layout_mcells_s": float(layout_average_match.group(1)) if layout_average_match else math.nan,
      "sim_time_per_s": float(sim_time_match.group(1)) if sim_time_match else math.nan,
      "mass_rel_drift": final.get("mass_rel_drift", math.nan),
      "energy_rel_drift": final.get("energy_rel_drift", math.nan),
      "wall_work": final.get("wall_work", math.nan),
      "wall_work_rel_error": final.get("wall_work_rel_error", math.nan),
      "mom1": final.get("mom1", math.nan),
      "min_rho": final.get("min_rho", math.nan),
      "min_pressure": final.get("min_pressure", math.nan),
      "run_dir": repo_relative(run_dir),
      "tail": tail,
      "_stdout": completed.stdout,
   }


def add_status(rows: list[dict[str, object]], mass_tol: float, work_tol: float) -> None:
   for row in rows:
      failures: list[str] = []
      if bool(row["expect_success"]):
         if int(row["returncode"]) != 0:
            failures.append(f"returncode={row['returncode']}")
         mass = float(row["mass_rel_drift"])
         energy = float(row["energy_rel_drift"])
         wall_work = float(row["wall_work"])
         work_error = float(row["wall_work_rel_error"])
         mom1 = float(row["mom1"])
         min_rho = float(row["min_rho"])
         min_pressure = float(row["min_pressure"])
         if not math.isfinite(mass) or mass > mass_tol:
            failures.append(f"mass_rel_drift={mass:.3e} > {mass_tol:.3e}")
         if not math.isfinite(energy) or energy <= 1.0e-8:
            failures.append(f"energy_rel_drift={energy:.3e} did not show positive wall work")
         if not math.isfinite(wall_work) or wall_work <= 0.0:
            failures.append(f"wall_work={wall_work:.3e} did not increase")
         if not math.isfinite(work_error) or work_error > work_tol:
            failures.append(f"wall_work_rel_error={work_error:.3e} > {work_tol:.3e}")
         if not math.isfinite(mom1) or mom1 <= 0.0:
            failures.append(f"mom1={mom1:.3e} did not increase")
         if not math.isfinite(min_rho) or min_rho <= 0.0:
            failures.append(f"min_rho={min_rho:.3e}")
         if not math.isfinite(min_pressure) or min_pressure <= 0.0:
            failures.append(f"min_pressure={min_pressure:.3e}")
      else:
         if int(row["returncode"]) == 0:
            failures.append("unsupported moving internal box did not abort")
         if "uncovered fluid cells" not in str(row.get("_stdout", "")):
            failures.append("abort did not report uncovered-cell conservative guard")
      row["status"] = "pass" if not failures else "fail"
      row["failures"] = "; ".join(failures)


def write_results(rows: list[dict[str, object]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "status",
      "returncode",
      "mcells_s",
      "layout_mcells_s",
      "sim_time_per_s",
      "mass_rel_drift",
      "energy_rel_drift",
      "wall_work",
      "wall_work_rel_error",
      "mom1",
      "min_rho",
      "min_pressure",
      "run_dir",
      "failures",
      "tail",
   ]
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
   parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--mass-tol", type=float, default=5.0e-13)
   parser.add_argument("--work-tol", type=float, default=5.0e-12)
   args = parser.parse_args()

   rows = [run_case(case, args.executable, args.input, args.output_dir, args.ranks, args.force) for case in CASES]
   add_status(rows, args.mass_tol, args.work_tol)
   write_results(rows, args.results)
   for row in rows:
      if row["status"] == "pass":
         if bool(row["expect_success"]):
            print(
               f"PASS {row['case']}: mass={float(row['mass_rel_drift']):.2e}, "
               f"energy={float(row['energy_rel_drift']):.2e}, "
               f"work_err={float(row['wall_work_rel_error']):.2e}, "
               f"mom1={float(row['mom1']):.3e}, {float(row['mcells_s']):.2f} Mcells/s"
               f", layout={float(row['layout_mcells_s']):.2f} Mcells/s"
            )
         else:
            print(f"PASS {row['case']}: conservative uncovered-cell guard")
      else:
         print(f"FAIL {row['case']}: {row['failures']}")
         print(row["tail"])
   print(f"wrote {repo_relative(args.results)}")
   return 0 if all(row["status"] == "pass" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
