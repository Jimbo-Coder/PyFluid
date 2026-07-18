#!/usr/bin/env python3
"""Validate pulsating solid geometry and evolution.

Pulsating circle/sphere geometry has sampled auxiliary fields, conservative
mask-remap evolution, and a non-subcycled AMR path that preserves valid fluid
totals across regrid.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import materialize_run_input, parameters_from_assignments, read_tsv, run_mpi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))
from plotfile_reader import plotfile_at_step, read_aux_field  # noqa: E402


DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "pressure_bump_2d_amr.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "pulsating_solid_geometry_validation"
DEFAULT_RESULTS = DEFAULT_RUN_ROOT / "results.tsv"

N_CELL = 96
RADIUS = 0.20
AMPLITUDE = 0.05
FREQUENCY = 1.0


@dataclass(frozen=True)
class Case:
   name: str
   phase: float


CASES = (
   Case("phase_0", 0.0),
   Case("phase_pi_over_2", 0.5 * math.pi),
   Case("phase_minus_pi_over_2", -0.5 * math.pi),
)


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
   "pyfluid.solid_motion=pulse",
   "pyfluid.solid_wall_condition=slip",
   "pyfluid.solid_center_x=0.5",
   "pyfluid.solid_center_y=0.5",
   f"pyfluid.solid_radius={RADIUS:.17g}",
   f"pyfluid.solid_radius_amplitude={AMPLITUDE:.17g}",
   f"pyfluid.solid_radius_frequency={FREQUENCY:.17g}",
   "pyfluid.solid_volume_fraction_samples_per_dim=4",
   "pyfluid.scalar_int=0",
   "pyfluid.plot_int=-1",
   "pyfluid.aux_gfs=solid_volume_fraction,solid_swept_volume,solid_aperture1,solid_aperture2,solid_phi",
   "pyfluid.aux_gf_int=0",
   "pyfluid.aux_gf_compute_int=-1",
   "pyfluid.checkpoint_int=-1",
)


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def run_with_parameters(
   executable: Path,
   base_input: Path,
   run_dir: Path,
   assignments: list[str],
   ranks: int,
   *,
   remove_parameters: tuple[str, ...] = (),
):
   input_path = materialize_run_input(
      base_input,
      run_dir / "run.in",
      parameters_from_assignments(assignments),
      remove_parameters=remove_parameters,
   )
   return run_mpi(executable, [input_path], ranks=ranks)


def solid_integrals_from_aux_plotfile(plotfile: Path) -> dict[str, float]:
   volume_fraction, _ = read_aux_field(
      plotfile.parent, "solid_volume_fraction", step=0
   )
   swept_volume, _ = read_aux_field(plotfile.parent, "solid_swept_volume", step=0)
   inv_cells = 1.0 / float(N_CELL * N_CELL)
   return {
      "solid_area": float((1.0 - volume_fraction).sum() * inv_cells),
      "swept_volume": float(swept_volume.sum() * inv_cells),
   }


def run_output_case(case: Case, executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   reference_time = -case.phase / (2.0 * math.pi * FREQUENCY) if FREQUENCY > 0.0 else 0.0
   completed = run_with_parameters(
      executable,
      input_file,
      run_dir,
      [
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         *BASE_OVERRIDES,
         "pyfluid.max_step=0",
         f"pyfluid.solid_radius_phase={case.phase:.17g}",
         f"pyfluid.solid_swept_reference_time={reference_time:.17g}",
      ],
      ranks,
      remove_parameters=("amr.ref_ratio",),
   )
   radius = RADIUS + AMPLITUDE * math.sin(case.phase)
   reference_radius = RADIUS + AMPLITUDE * math.sin(case.phase + 2.0 * math.pi * FREQUENCY * reference_time)
   exact_area = math.pi * radius * radius
   exact_swept_volume = exact_area - math.pi * reference_radius * reference_radius
   plotfile = plotfile_at_step(run_dir, 0, "auxGF")
   integrals = solid_integrals_from_aux_plotfile(plotfile) if plotfile is not None else {
      "solid_area": math.nan,
      "swept_volume": math.nan,
   }
   solid_area = integrals["solid_area"]
   swept_volume = integrals["swept_volume"]
   error = abs(solid_area - exact_area)
   swept_error = abs(swept_volume - exact_swept_volume)
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   if not math.isfinite(error) or error > 1.5e-4:
      failures.append(f"area error={error:.3e} > 1.5e-4")
   if not math.isfinite(swept_error) or swept_error > 2.0e-4:
      failures.append(f"swept error={swept_error:.3e} > 2e-4")
   return {
      "case": case.name,
      "kind": "output",
      "status": "pass" if not failures else "fail",
      "phase": case.phase,
      "radius": radius,
      "solid_area": solid_area,
      "exact_area": exact_area,
      "abs_error": error,
      "swept_volume": swept_volume,
      "exact_swept_volume": exact_swept_volume,
      "swept_abs_error": swept_error,
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def read_last_scalar_row(run_dir: Path) -> dict[str, str]:
   scalar_file = run_dir / "scalars.tsv"
   if not scalar_file.exists():
      return {}
   rows = read_tsv(scalar_file, require_complete=True)
   return rows[-1] if rows else {}


def read_max_active_level(run_dir: Path) -> int:
   levels_file = run_dir / "amr_levels.tsv"
   if not levels_file.exists():
      return 0
   rows = read_tsv(levels_file, require_complete=True)
   if not rows:
      return 0
   return max(int(row.get("active_finest_level", "0")) for row in rows)


def run_evolution_case(executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / "single_level_evolution"
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   completed = run_with_parameters(
      executable,
      input_file,
      run_dir,
      [
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         *BASE_OVERRIDES,
         "pyfluid.max_step=20",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=20",
         "pyfluid.aux_gfs=solid_mask,solid_volume_fraction,solid_swept_volume",
         "pyfluid.aux_gf_int=20",
         "pyfluid.ng=2",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.cfl=0.9",
         "pyfluid.solid_radius_phase=0.0",
         "pyfluid.solid_swept_reference_time=0.0",
      ],
      ranks,
      remove_parameters=("amr.ref_ratio",),
   )
   scalars = read_last_scalar_row(run_dir)
   mass_rel_drift = float(scalars.get("mass_rel_drift", "nan"))
   wall_work_rel_error = float(scalars.get("wall_work_rel_error", "nan"))
   min_rho = float(scalars.get("min_rho", "nan"))
   min_pressure = float(scalars.get("min_pressure", "nan"))
   plotfile = plotfile_at_step(run_dir, 20)
   aux_plotfile = plotfile_at_step(run_dir, 20, "auxGF")
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   if not math.isfinite(mass_rel_drift) or mass_rel_drift > 1.0e-12:
      failures.append(f"mass drift={mass_rel_drift:.3e} > 1e-12")
   if not math.isfinite(wall_work_rel_error) or wall_work_rel_error > 1.0e-3:
      failures.append(f"wall work rel error={wall_work_rel_error:.3e} > 1e-3")
   if not math.isfinite(min_rho) or min_rho <= 0.0:
      failures.append(f"min rho={min_rho:.3e} <= 0")
   if not math.isfinite(min_pressure) or min_pressure <= 0.0:
      failures.append(f"min pressure={min_pressure:.3e} <= 0")
   if plotfile is None:
      failures.append("missing evolved state plotfile")
   if aux_plotfile is None:
      failures.append("missing evolved aux plotfile")
   return {
      "case": "single_level_evolution",
      "kind": "evolution",
      "status": "pass" if not failures else "fail",
      "phase": 0.0,
      "radius": RADIUS,
      "solid_area": "",
      "exact_area": "",
      "abs_error": "",
      "swept_volume": "",
      "exact_swept_volume": "",
      "swept_abs_error": "",
      "mass_rel_drift": mass_rel_drift,
      "wall_work_rel_error": wall_work_rel_error,
      "min_rho": min_rho,
      "min_pressure": min_pressure,
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def run_amr_evolution_case(executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / "amr_evolution"
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   completed = run_with_parameters(
      executable,
      input_file,
      run_dir,
      [
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         *BASE_OVERRIDES,
         "amr.max_level=2",
         "amr.ref_ratio=2 2",
         "amr.regrid_int=4",
         "amr.grid_eff=0.8",
         "amr.n_error_buf=3 3",
         "amr.tag_metric=none",
         "amr.solid_refine_band_cells=2.0",
         "amr.solid_curvature_error=2.0e-5",
         "amr.level_cell_budget_side=384",
         "amr.total_cell_budget=250000",
         "amr.state_interp=conservative",
         "pyfluid.max_step=48",
         "pyfluid.scalar_int=4",
         "pyfluid.plot_int=48",
         "pyfluid.aux_gfs=solid_mask,solid_volume_fraction,solid_swept_volume",
         "pyfluid.aux_gf_int=48",
         "pyfluid.ng=2",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.cfl=0.9",
         "pyfluid.subcycling=off",
         "pyfluid.solid_radius_phase=0.0",
         "pyfluid.solid_swept_reference_time=0.0",
      ],
      ranks,
   )
   scalars = read_last_scalar_row(run_dir)
   mass_rel_drift = float(scalars.get("mass_rel_drift", "nan"))
   wall_work_rel_error = float(scalars.get("wall_work_rel_error", "nan"))
   min_rho = float(scalars.get("min_rho", "nan"))
   min_pressure = float(scalars.get("min_pressure", "nan"))
   active_finest_level = read_max_active_level(run_dir)
   plotfile = plotfile_at_step(run_dir, 48)
   aux_plotfile = plotfile_at_step(run_dir, 48, "auxGF")
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   if active_finest_level < 1:
      failures.append(f"active finest level={active_finest_level} < 1")
   if not math.isfinite(mass_rel_drift) or mass_rel_drift > 1.0e-12:
      failures.append(f"mass drift={mass_rel_drift:.3e} > 1e-12")
   if not math.isfinite(min_rho) or min_rho <= 0.0:
      failures.append(f"min rho={min_rho:.3e} <= 0")
   if not math.isfinite(min_pressure) or min_pressure <= 0.0:
      failures.append(f"min pressure={min_pressure:.3e} <= 0")
   if plotfile is None:
      failures.append("missing AMR evolved state plotfile")
   if aux_plotfile is None:
      failures.append("missing AMR evolved aux plotfile")
   return {
      "case": "amr_evolution",
      "kind": "amr_evolution",
      "status": "pass" if not failures else "fail",
      "phase": 0.0,
      "radius": RADIUS,
      "solid_area": "",
      "exact_area": "",
      "abs_error": "",
      "mass_rel_drift": mass_rel_drift,
      "wall_work_rel_error": wall_work_rel_error,
      "min_rho": min_rho,
      "min_pressure": min_pressure,
      "active_finest_level": active_finest_level,
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def run_amr_subcycling_case(executable: Path, input_file: Path, run_root: Path, ranks: int, force: bool) -> dict[str, object]:
   run_dir = run_root / "amr_subcycling"
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   completed = run_with_parameters(
      executable,
      input_file,
      run_dir,
      [
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         *BASE_OVERRIDES,
         "amr.max_level=2",
         "amr.ref_ratio=2 2",
         "amr.regrid_int=4",
         "amr.grid_eff=0.8",
         "amr.n_error_buf=3 3",
         "amr.tag_metric=none",
         "amr.solid_refine_band_cells=2.0",
         "amr.solid_curvature_error=2.0e-5",
         "amr.level_cell_budget_side=384",
         "amr.total_cell_budget=250000",
         "amr.state_interp=protected",
         "pyfluid.max_step=48",
         "pyfluid.scalar_int=4",
         "pyfluid.plot_int=48",
         "pyfluid.aux_gfs=solid_mask,solid_volume_fraction,solid_swept_volume",
         "pyfluid.aux_gf_int=48",
         "pyfluid.ng=2",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.cfl=0.9",
         "pyfluid.subcycling=on",
         "pyfluid.solid_radius_phase=0.0",
         "pyfluid.solid_swept_reference_time=0.0",
      ],
      ranks,
   )
   scalars = read_last_scalar_row(run_dir)
   mass_rel_drift = float(scalars.get("mass_rel_drift", "nan"))
   wall_work_rel_error = float(scalars.get("wall_work_rel_error", "nan"))
   min_rho = float(scalars.get("min_rho", "nan"))
   min_pressure = float(scalars.get("min_pressure", "nan"))
   active_finest_level = read_max_active_level(run_dir)
   plotfile = plotfile_at_step(run_dir, 48)
   aux_plotfile = plotfile_at_step(run_dir, 48, "auxGF")
   failures: list[str] = []
   if completed.returncode != 0:
      failures.append(f"returncode={completed.returncode}")
   if active_finest_level < 1:
      failures.append(f"active finest level={active_finest_level} < 1")
   if not math.isfinite(mass_rel_drift) or mass_rel_drift > 1.0e-10:
      failures.append(f"mass drift={mass_rel_drift:.3e} > 1e-10")
   if not math.isfinite(wall_work_rel_error) or wall_work_rel_error > 1.0e-3:
      failures.append(f"wall work rel error={wall_work_rel_error:.3e} > 1e-3")
   if not math.isfinite(min_rho) or min_rho <= 0.0:
      failures.append(f"min rho={min_rho:.3e} <= 0")
   if not math.isfinite(min_pressure) or min_pressure <= 0.0:
      failures.append(f"min pressure={min_pressure:.3e} <= 0")
   if plotfile is None:
      failures.append("missing subcycled state plotfile")
   if aux_plotfile is None:
      failures.append("missing subcycled aux plotfile")
   return {
      "case": "amr_subcycling",
      "kind": "amr_subcycling",
      "status": "pass" if not failures else "fail",
      "phase": 0.0,
      "radius": RADIUS,
      "solid_area": "",
      "exact_area": "",
      "abs_error": "",
      "mass_rel_drift": mass_rel_drift,
      "wall_work_rel_error": wall_work_rel_error,
      "min_rho": min_rho,
      "min_pressure": min_pressure,
      "active_finest_level": active_finest_level,
      "run_dir": repo_relative(run_dir),
      "failures": "; ".join(failures),
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def swept_area_row(rows: list[dict[str, object]]) -> dict[str, object]:
   output_rows = [row for row in rows if row["kind"] == "output"]
   by_case = {str(row["case"]): row for row in output_rows}
   base = by_case["phase_0"]
   expanded = by_case["phase_pi_over_2"]
   sampled_delta = float(expanded["solid_area"]) - float(base["solid_area"])
   exact_delta = float(expanded["exact_area"]) - float(base["exact_area"])
   direct_delta = float(expanded["swept_volume"])
   sampled_error = abs(sampled_delta - exact_delta)
   direct_error = abs(direct_delta - exact_delta)
   failures: list[str] = []
   if sampled_error > 2.0e-4:
      failures.append(f"sampled swept area error={sampled_error:.3e} > 2e-4")
   if direct_error > 2.0e-4:
      failures.append(f"direct swept area error={direct_error:.3e} > 2e-4")
   return {
      "case": "swept_area",
      "kind": "swept",
      "status": "pass" if not failures else "fail",
      "phase": "",
      "radius": "",
      "solid_area": sampled_delta,
      "exact_area": exact_delta,
      "abs_error": sampled_error,
      "swept_volume": direct_delta,
      "exact_swept_volume": exact_delta,
      "swept_abs_error": direct_error,
      "run_dir": "",
      "failures": "; ".join(failures),
      "tail": "",
   }


def write_results(rows: list[dict[str, object]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "case",
      "kind",
      "status",
      "phase",
      "radius",
      "solid_area",
      "exact_area",
      "abs_error",
      "swept_volume",
      "exact_swept_volume",
      "swept_abs_error",
      "mass_rel_drift",
      "wall_work_rel_error",
      "min_rho",
      "min_pressure",
      "active_finest_level",
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
   parser.add_argument("--skip-amr-evolution", action="store_true", help="skip the heavier AMR moving-wall stress rows")
   args = parser.parse_args()

   if args.force and args.run_root.exists():
      shutil.rmtree(args.run_root)
   rows = [run_output_case(case, args.executable, args.input, args.run_root, args.ranks, args.force) for case in CASES]
   rows.append(swept_area_row(rows))
   rows.append(run_evolution_case(args.executable, args.input, args.run_root, args.ranks, args.force))
   if not args.skip_amr_evolution:
      rows.append(run_amr_evolution_case(args.executable, args.input, args.run_root, args.ranks, args.force))
      rows.append(run_amr_subcycling_case(args.executable, args.input, args.run_root, args.ranks, args.force))
   write_results(rows, args.results)
   for row in rows:
      if row["kind"] == "output":
         print(
            f"{str(row['status']).upper():>4} {row['case']}: "
            f"radius={float(row['radius']):.6f}, "
            f"area={float(row['solid_area']):.8f}, "
            f"exact={float(row['exact_area']):.8f}, "
            f"abs_error={float(row['abs_error']):.3e}, "
            f"swept={float(row['swept_volume']):.8f}, "
            f"exact_swept={float(row['exact_swept_volume']):.8f}, "
            f"swept_error={float(row['swept_abs_error']):.3e}"
         )
      elif row["kind"] == "swept":
         print(
            f"{str(row['status']).upper():>4} {row['case']}: "
            f"sampled_delta={float(row['solid_area']):.8f}, "
            f"exact_delta={float(row['exact_area']):.8f}, "
            f"sampled_error={float(row['abs_error']):.3e}, "
            f"direct_delta={float(row['swept_volume']):.8f}, "
            f"direct_error={float(row['swept_abs_error']):.3e}"
         )
      elif row["kind"] == "evolution":
         print(
            f"{str(row['status']).upper():>4} {row['case']}: "
            f"mass={float(row['mass_rel_drift']):.3e}, "
            f"wall_work={float(row['wall_work_rel_error']):.3e}, "
            f"min_rho={float(row['min_rho']):.3e}, "
            f"min_P={float(row['min_pressure']):.3e}"
         )
      elif row["kind"] == "amr_evolution":
         print(
            f"{str(row['status']).upper():>4} {row['case']}: "
            f"level={int(row['active_finest_level'])}, "
            f"mass={float(row['mass_rel_drift']):.3e}, "
            f"wall_work={float(row['wall_work_rel_error']):.3e}, "
            f"min_rho={float(row['min_rho']):.3e}, "
            f"min_P={float(row['min_pressure']):.3e}"
         )
      elif row["kind"] == "amr_subcycling":
         print(
            f"{str(row['status']).upper():>4} {row['case']}: "
            f"level={int(row['active_finest_level'])}, "
            f"mass={float(row['mass_rel_drift']):.3e}, "
            f"wall_work={float(row['wall_work_rel_error']):.3e}, "
            f"min_rho={float(row['min_rho']):.3e}, "
            f"min_P={float(row['min_pressure']):.3e}"
         )
      else:
         print(f"{str(row['status']).upper():>4} {row['case']}: {row['kind']}")
      if row["status"] != "pass":
         print(f"     {row['failures']}")
   print(f"wrote {repo_relative(args.results)}")
   failed = [row for row in rows if row["status"] != "pass"]
   return 0 if args.allow_fail or not failed else 1


if __name__ == "__main__":
   raise SystemExit(main())
