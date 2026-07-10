#!/usr/bin/env python3
"""Validate analytic solid-boundary AMR tagging.

The test uses a circle small enough that the base-grid binary mask does not see
any solid cell centers. With the phi-band tag disabled it must stay coarse;
with the tag enabled AMR should refine around the analytic phi=0 boundary.
Positive curvature tolerances should stop that refinement once the geometric
sagitta error is resolved.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from validation_common import mpi_prefix
from validation_common import read_tsv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "solid_geometry_refinement_validation"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "results.tsv"


@dataclass(frozen=True)
class Case:
   name: str
   band_cells: float
   curvature_error: float
   expected_finest_level: int
   geometry_overrides: tuple[str, ...]


CIRCLE_OVERRIDES = (
   "pyfluid.solid_geometry=circle",
   "pyfluid.solid_center_x=0.5",
   "pyfluid.solid_center_y=0.5",
   "pyfluid.solid_radius=0.015",
)

TINY_BOX_OVERRIDES = (
   "pyfluid.solid_geometry=boxes",
   "pyfluid.solid_box_lo=0.49 0.49",
   "pyfluid.solid_box_hi=0.51 0.51",
)


CASES = (
   Case("band_off", 0.0, 0.0, 0, CIRCLE_OVERRIDES),
   Case("band_on", 1.5, 0.0, 2, CIRCLE_OVERRIDES),
   Case("curvature_loose", 1.5, 2.0e-2, 0, CIRCLE_OVERRIDES),
   Case("curvature_strict", 1.5, 6.0e-3, 2, CIRCLE_OVERRIDES),
   Case("flat_box_curvature", 1.5, 1.0e-12, 0, TINY_BOX_OVERRIDES),
)


BASE_OVERRIDES = (
   "geometry.prob_lo=0.0 0.0",
   "geometry.prob_hi=1.0 1.0",
   "geometry.is_periodic=0 0",
   "amr.n_cell=16 16",
   "amr.max_level=2",
   "amr.ref_ratio=2 2",
   "amr.max_grid_size=16",
   "amr.blocking_factor=4",
   "amr.regrid_int=1",
   "amr.tag_metric=none",
   "amr.n_error_buf=1 1",
   "amr.total_cell_budget=20000",
   "pyfluid.physics=hydro",
   "pyfluid.ic=uniform",
   "pyfluid.rho0=1.0",
   "pyfluid.pressure0=1.0",
   "pyfluid.v1_0=0.0",
   "pyfluid.v2_0=0.0",
   "pyfluid.bc=reflecting",
   "pyfluid.solid_motion=static",
   "pyfluid.solid_wall_condition=slip",
   "pyfluid.cfl=0.9",
   "pyfluid.recon=pwc",
   "pyfluid.riemann=hllc",
   "pyfluid.timestep=euler",
   "pyfluid.max_step=3",
   "pyfluid.final_time=-1",
   "pyfluid.scalar_int=1",
   "pyfluid.plot_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_int=-1",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.checkpoint_int=-1",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def read_last_tsv_row(path: Path) -> dict[str, str]:
   rows = read_tsv(path, require_complete=True)
   if not rows:
      raise RuntimeError(f"{path} is empty")
   return rows[-1]


def run_case(case: Case, executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   command = [
      *mpi_prefix(ranks),
      str(executable),
      str(input_file),
      f"pyfluid.output_dir={repo_relative(run_dir)}",
      *BASE_OVERRIDES,
      *case.geometry_overrides,
      f"amr.solid_refine_band_cells={case.band_cells:.17g}",
      f"amr.solid_curvature_error={case.curvature_error:.17g}",
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
   amr = read_last_tsv_row(run_dir / "amr_levels.tsv") if (run_dir / "amr_levels.tsv").exists() else {}
   scalars = read_last_tsv_row(run_dir / "scalars.tsv") if (run_dir / "scalars.tsv").exists() else {}
   finest = int(float(amr.get("active_finest_level", "-1")))
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   if finest != case.expected_finest_level:
      failures.append(f"active_finest_level={finest} expected {case.expected_finest_level}")
   if case.curvature_error > 0.0 and amr.get("solid_curvature_resolved") != "1":
      failures.append(
         "solid_curvature_resolved="
         f"{amr.get('solid_curvature_resolved', '<missing>')} expected 1"
      )
   for key in ("mass_rel_drift", "energy_rel_drift"):
      value = float(scalars.get(key, "nan"))
      if not value == value or value > 1.0e-12:
         failures.append(f"{key}={value:.3e}")
   return {
      "case": case.name,
      "status": "pass" if not failures else "fail",
      "band_cells": case.band_cells,
      "curvature_error": case.curvature_error,
      "expected_finest_level": case.expected_finest_level,
      "geometry": next(item.split("=", 1)[1] for item in case.geometry_overrides if item.startswith("pyfluid.solid_geometry=")),
      "active_finest_level": finest,
      "resolution_limited_by_max_level": amr.get("resolution_limited_by_max_level", ""),
      "solid_curvature_error": amr.get("solid_curvature_error", ""),
      "solid_curvature_tolerance": amr.get("solid_curvature_tolerance", ""),
      "solid_curvature_resolved": amr.get("solid_curvature_resolved", ""),
      "total_cells": amr.get("total_cells", ""),
      "cells_L0": amr.get("cells_L0", ""),
      "cells_L1": amr.get("cells_L1", ""),
      "cells_L2": amr.get("cells_L2", ""),
      "mass_rel_drift": scalars.get("mass_rel_drift", ""),
      "energy_rel_drift": scalars.get("energy_rel_drift", ""),
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "status",
      "band_cells",
      "curvature_error",
      "expected_finest_level",
      "geometry",
      "active_finest_level",
      "resolution_limited_by_max_level",
      "solid_curvature_error",
      "solid_curvature_tolerance",
      "solid_curvature_resolved",
      "total_cells",
      "cells_L0",
      "cells_L1",
      "cells_L2",
      "mass_rel_drift",
      "energy_rel_drift",
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
   write_results(rows, args.results)
   for row in rows:
      print(
         f"{str(row['status']).upper():>4} {row['case']}: "
         f"band={float(row['band_cells']):.2f}, "
         f"curv_tol={float(row['curvature_error']):.3g}, "
         f"geom={row['geometry']}, "
         f"finest={row['active_finest_level']}, "
         f"cells={row['total_cells']}"
      )
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.results)}")
   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
