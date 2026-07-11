#!/usr/bin/env python3
"""Validate sampled solid volume-fraction and aperture auxiliary output.

This is an output-only precursor for cut-cell work. It checks that sampled
analytic volume fractions recover the area of a circle more accurately than the
binary cell-center mask, and that sampled face apertures integrate to the same
area while leaving evolution untouched.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, run_mpi
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))
from plot_amrex_slice import parse_plot_header, read_level_state  # noqa: E402
from plotfile_reader import sorted_aux_plotfiles  # noqa: E402


DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "solid_volume_fraction_validation"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "results.tsv"


@dataclass(frozen=True)
class Case:
   samples_per_dim: int


CASES = (Case(1), Case(4))
N_CELL = 64
CIRCLE_X = 0.47
CIRCLE_Y = 0.53
CIRCLE_RADIUS = 0.23


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "geometry.is_periodic=0 0",
   f"amr.n_cell={N_CELL} {N_CELL}",
   "amr.max_level=0",
   "amr.max_grid_size=32",
   "amr.blocking_factor=8",
   "pyfluid.physics=hydro",
   "pyfluid.ic=uniform",
   "pyfluid.rho0=1.0",
   "pyfluid.pressure0=1.0",
   "pyfluid.bc=reflecting",
   "pyfluid.solid_geometry=circle",
   f"pyfluid.solid_center_x={CIRCLE_X:.17g}",
   f"pyfluid.solid_center_y={CIRCLE_Y:.17g}",
   f"pyfluid.solid_radius={CIRCLE_RADIUS:.17g}",
   "pyfluid.solid_wall_condition=slip",
   "pyfluid.max_step=0",
   "pyfluid.scalar_int=0",
   "pyfluid.plot_int=-1",
   "pyfluid.aux_gfs=solid_volume_fraction,solid_aperture1,solid_aperture2",
   "pyfluid.aux_gf_int=0",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.checkpoint_int=-1",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def read_aux_field(aux_dir: Path, field_name: str) -> tuple[Any, int]:
   for plotfile in sorted_aux_plotfiles(aux_dir):
      names, ndim, _time = parse_plot_header(plotfile)
      if field_name not in names:
         continue
      state = read_level_state(plotfile, names, ndim, 0)
      return state[names.index(field_name)], ndim
   raise ValueError(f"could not find auxiliary field {field_name} under {aux_dir}")


def solid_areas_from_aux_plotfile(plotfile: Path) -> dict[str, float]:
   aux_dir = plotfile.parent
   volume_fraction, ndim = read_aux_field(aux_dir, "solid_volume_fraction")
   if ndim != 2:
      raise RuntimeError(f"expected 2D aux plotfile, got ndim={ndim}")
   cell_area = 1.0 / float(N_CELL * N_CELL)
   aperture_x, _ = read_aux_field(aux_dir, "solid_aperture1")
   aperture_y, _ = read_aux_field(aux_dir, "solid_aperture2")
   return {
      "solid_area": float((1.0 - volume_fraction).sum() * cell_area),
      "solid_area_aperture_x": float((1.0 - aperture_x).sum() * cell_area),
      "solid_area_aperture_y": float((1.0 - aperture_y).sum() * cell_area),
   }


def run_case(case: Case, executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / f"samples_{case.samples_per_dim}"
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      input_file,
      run_dir / "run.in",
      parameters_from_assignments([
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         *BASE_OVERRIDES,
         f"pyfluid.solid_volume_fraction_samples_per_dim={case.samples_per_dim}",
      ]),
      remove_parameters=("amr.ref_ratio",),
   )
   completed = run_mpi(executable, [input_path], ranks=ranks)
   plotfiles = sorted_aux_plotfiles(run_dir)
   solid_areas = solid_areas_from_aux_plotfile(plotfiles[-1]) if plotfiles else {
      "solid_area": math.nan,
      "solid_area_aperture_x": math.nan,
      "solid_area_aperture_y": math.nan,
   }
   exact_area = math.pi * CIRCLE_RADIUS * CIRCLE_RADIUS
   error = abs(solid_areas["solid_area"] - exact_area)
   aperture_error_x = abs(solid_areas["solid_area_aperture_x"] - exact_area)
   aperture_error_y = abs(solid_areas["solid_area_aperture_y"] - exact_area)
   return {
      "case": f"samples_{case.samples_per_dim}",
      "samples_per_dim": case.samples_per_dim,
      "returncode": completed.returncode,
      **solid_areas,
      "exact_area": exact_area,
      "abs_error": error,
      "abs_error_aperture_x": aperture_error_x,
      "abs_error_aperture_y": aperture_error_y,
      "run_dir": repo_relative(run_dir),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def add_status(rows: list[dict[str, object]]) -> None:
   binary_error = float(next(row["abs_error"] for row in rows if int(row["samples_per_dim"]) == 1))
   binary_aperture_error_x = float(next(row["abs_error_aperture_x"] for row in rows if int(row["samples_per_dim"]) == 1))
   binary_aperture_error_y = float(next(row["abs_error_aperture_y"] for row in rows if int(row["samples_per_dim"]) == 1))
   for row in rows:
      failures: list[str] = []
      if int(row["returncode"]) != 0:
         failures.append(f"returncode={row['returncode']}")
      error = float(row["abs_error"])
      if not math.isfinite(error):
         failures.append("missing area")
      if int(row["samples_per_dim"]) == 4:
         if error > 1.0e-4:
            failures.append(f"sampled area error={error:.3e} > 1e-4")
         if error >= binary_error:
            failures.append(f"sampled area error={error:.3e} did not improve over binary {binary_error:.3e}")
         for key in ("abs_error_aperture_x", "abs_error_aperture_y"):
            aperture_error = float(row[key])
            if aperture_error > 3.0e-4:
               failures.append(f"{key}={aperture_error:.3e} > 3e-4")
         if float(row["abs_error_aperture_x"]) >= binary_aperture_error_x:
            failures.append(
               f"sampled aperture x error={float(row['abs_error_aperture_x']):.3e} "
               f"did not improve over binary {binary_aperture_error_x:.3e}"
            )
         if float(row["abs_error_aperture_y"]) >= binary_aperture_error_y:
            failures.append(
               f"sampled aperture y error={float(row['abs_error_aperture_y']):.3e} "
               f"did not improve over binary {binary_aperture_error_y:.3e}"
            )
      row["status"] = "pass" if not failures else "fail"
      row["failures"] = "; ".join(failures)


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "status",
      "samples_per_dim",
      "solid_area",
      "solid_area_aperture_x",
      "solid_area_aperture_y",
      "exact_area",
      "abs_error",
      "abs_error_aperture_x",
      "abs_error_aperture_y",
      "run_dir",
      "failures",
      "tail",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--allow-fail", action="store_true")
   args = parser.parse_args()

   if args.force and args.run_root.exists():
      shutil.rmtree(args.run_root)
   rows = [run_case(case, args.executable, args.input, args.run_root, args.ranks, args.force) for case in CASES]
   add_status(rows)
   write_results(rows, args.results)
   for row in rows:
      print(
         f"{str(row['status']).upper():>4} {row['case']}: "
         f"area={float(row['solid_area']):.8f}, "
         f"ap_x={float(row['solid_area_aperture_x']):.8f}, "
         f"ap_y={float(row['solid_area_aperture_y']):.8f}, "
         f"exact={float(row['exact_area']):.8f}, "
         f"abs_error={float(row['abs_error']):.3e}, "
         f"ap_err=({float(row['abs_error_aperture_x']):.3e}, {float(row['abs_error_aperture_y']):.3e})"
      )
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.results)}")
   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
