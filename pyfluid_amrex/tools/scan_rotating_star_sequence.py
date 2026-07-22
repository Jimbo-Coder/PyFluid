#!/usr/bin/env python3
"""Run a compact fixed-central-density rotating-star ID sequence."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from validation_common import (
   ROOT,
   materialize_run_input,
   read_tsv,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_3d.in"
RUN_ROOT = ROOT / "runs/current/rotating_star_sequence"
SOLVE_PATTERN = re.compile(
   r"xcts initial data: matter_iterations=(?P<matter>[0-9]+) "
   r"relative_matter_update=(?P<update>[0-9.eE+-]+) "
   r"coupled_iterations=(?P<coupled>[0-9]+)"
)
SURFACE_PATTERN = re.compile(
   r"rotating star: omega=(?P<omega>[0-9.eE+-]+).* "
   r"maximum_velocity=(?P<vmax>[0-9.eE+-]+).* "
   r"equatorial_radius=(?P<equator>[0-9.eE+-]+) "
   r"polar_radius=(?P<pole>[0-9.eE+-]+) "
   r"coordinate_axis_ratio=(?P<axis_ratio>[0-9.eE+-]+) "
   r"mass_shedding_indicator=(?P<mass_shedding>[0-9.eE+-]+)"
)


def scalar_value(row: dict[str, str], name: str) -> float:
   value = row.get(name, "")
   return float(value) if value else math.nan


def relative_difference(first: float, second: float) -> float:
   if not math.isfinite(first) or not math.isfinite(second):
      return math.inf
   return abs(first - second) / max(abs(first), abs(second), 1.0e-30)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--omega", type=float, nargs="+", required=True)
   parser.add_argument("--eos-preset", default="sly")
   parser.add_argument(
      "--rotation-law",
      choices=("uniform", "j_constant", "q_power"),
      default="uniform",
   )
   parser.add_argument("--differential-rotation-length", type=float, default=10.0)
   parser.add_argument("--rotation-power-q", type=float, default=1.0)
   parser.add_argument("--cells", type=int, default=24)
   parser.add_argument("--max-grid-size", type=int, default=8)
   parser.add_argument("--continuation-steps", type=int, default=5)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=3600)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   if args.cells < 8 or args.max_grid_size < 4:
      raise SystemExit("sequence grids are too small")
   if args.continuation_steps < 1 or args.ranks < 1:
      raise SystemExit("continuation steps and MPI ranks must be positive")
   if any(not math.isfinite(omega) or omega <= 0.0 for omega in args.omega):
      raise SystemExit("sequence angular velocities must be positive and finite")
   if (
      not math.isfinite(args.differential_rotation_length)
      or args.differential_rotation_length <= 0.0
   ):
      raise SystemExit("--differential-rotation-length must be positive and finite")
   if not math.isfinite(args.rotation_power_q) or args.rotation_power_q < 1.0:
      raise SystemExit("--rotation-power-q must be finite and at least one")

   law_suffix = args.rotation_law
   if args.rotation_law != "uniform":
      law_suffix += f"_a{args.differential_rotation_length:g}"
   if args.rotation_law == "q_power":
      law_suffix += f"_q{args.rotation_power_q:g}"
   sequence_name = f"{args.eos_preset}_{law_suffix}_n{args.cells}".replace(
      ".", "p"
   )
   sequence_dir = RUN_ROOT / sequence_name
   sequence_dir.mkdir(parents=True, exist_ok=True)
   results_path = sequence_dir / "results.tsv"
   requested_omega = set(args.omega)
   rows: list[dict[str, object]] = []
   if results_path.exists():
      rows = [
         dict(row)
         for row in read_tsv(results_path)
         if float(row["omega"]) not in requested_omega
      ]
   new_rows: list[dict[str, object]] = []
   for omega in args.omega:
      case_name = f"omega{omega:.8g}".replace(".", "p")
      run_dir = sequence_dir / case_name
      if args.force and run_dir.exists():
         shutil.rmtree(run_dir)
      run_dir.mkdir(parents=True, exist_ok=True)
      input_path = materialize_run_input(
         BASE_INPUT,
         run_dir / "input.in",
         {
            "amr.n_cell": f"{args.cells} {args.cells} {args.cells}",
            "amr.max_level": "0",
            "amr.max_grid_size": str(args.max_grid_size),
            "initial_data.angular_velocity": f"{omega:.17g}",
            "initial_data.rotation_law": args.rotation_law,
            "initial_data.differential_rotation_length": str(
               args.differential_rotation_length
            ),
            "initial_data.rotation_power_q": str(args.rotation_power_q),
            "initial_data.rotation_continuation_steps": str(
               args.continuation_steps
            ),
            "pyfluid.eos": "hybrid_polytrope",
            "pyfluid.eos_preset": args.eos_preset,
            "pyfluid.max_step": "0",
            "pyfluid.plot_int": "-1",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_diagnostics": (
               "coordinate_angular_momentum,global_gr"
            ),
            "pyfluid.run_name": case_name,
            "pyfluid.output_dir": repo_relative(run_dir),
         },
      )
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      solve_match = SOLVE_PATTERN.search(completed.stdout)
      surface_match = SURFACE_PATTERN.search(completed.stdout)
      status = (
         "PASS"
         if completed.returncode == 0
         and solve_match is not None
         and surface_match is not None
         else "FAIL"
      )
      scalar_rows = read_scalar_rows(run_dir) if status == "PASS" else []
      scalar = scalar_rows[0] if scalar_rows else {}
      mass_shedding_indicator = (
         float(surface_match.group("mass_shedding"))
         if surface_match else math.nan
      )
      if status == "PASS" and mass_shedding_indicator <= 0.0:
         status = "SUPERCRITICAL"
      adm_mass_surface = scalar_value(scalar, "adm_mass_surface")
      adm_mass_volume = scalar_value(scalar, "adm_mass_volume")
      matter_angular_momentum3 = scalar_value(scalar, "angular_mom3")
      adm_angular_momentum3_volume = scalar_value(
         scalar, "adm_angular_momentum3_volume"
      )
      adm_angular_momentum3_surface = scalar_value(
         scalar, "adm_angular_momentum3_surface"
      )
      mass_surface_volume_error = relative_difference(
         adm_mass_surface, adm_mass_volume
      )
      matter_volume_angular_momentum_error = relative_difference(
         matter_angular_momentum3, adm_angular_momentum3_volume
      )
      surface_volume_angular_momentum_error = relative_difference(
         adm_angular_momentum3_surface, adm_angular_momentum3_volume
      )
      dimensionless_spin = (
         adm_angular_momentum3_volume / adm_mass_volume**2
         if math.isfinite(adm_angular_momentum3_volume)
         and math.isfinite(adm_mass_volume)
         and adm_mass_volume > 0.0
         else math.nan
      )
      if status == "PASS" and (
         mass_surface_volume_error > 0.1
         or matter_volume_angular_momentum_error > 0.02
         or surface_volume_angular_momentum_error > 0.05
         or not 0.0 < dimensionless_spin < 1.0
      ):
         status = "FAIL_PHYSICS"
      row: dict[str, object] = {
         "status": status,
         "omega": omega,
         "rotation_law": args.rotation_law,
         "differential_rotation_length": args.differential_rotation_length,
         "rotation_power_q": args.rotation_power_q,
         "cells_per_axis": args.cells,
         "continuation_steps": args.continuation_steps,
         "seconds": completed.seconds,
         "returncode": completed.returncode,
         "matter_iterations": (
            int(solve_match.group("matter")) if solve_match else ""
         ),
         "relative_matter_update": (
            float(solve_match.group("update")) if solve_match else ""
         ),
         "maximum_velocity": (
            float(surface_match.group("vmax")) if surface_match else ""
         ),
         "equatorial_radius": (
            float(surface_match.group("equator")) if surface_match else ""
         ),
         "polar_radius": (
            float(surface_match.group("pole")) if surface_match else ""
         ),
         "coordinate_axis_ratio": (
            float(surface_match.group("axis_ratio")) if surface_match else ""
         ),
         "mass_shedding_indicator": (
            mass_shedding_indicator if surface_match else ""
         ),
         "adm_mass_surface": adm_mass_surface,
         "adm_mass_volume": adm_mass_volume,
         "komar_mass_volume": scalar_value(scalar, "komar_mass_volume"),
         "angular_momentum3": matter_angular_momentum3,
         "adm_angular_momentum3_volume": adm_angular_momentum3_volume,
         "adm_angular_momentum3_surface": adm_angular_momentum3_surface,
         "dimensionless_spin": dimensionless_spin,
         "mass_surface_volume_error": mass_surface_volume_error,
         "matter_volume_angular_momentum_error": (
            matter_volume_angular_momentum_error
         ),
         "surface_volume_angular_momentum_error": (
            surface_volume_angular_momentum_error
         ),
         "run_dir": repo_relative(run_dir),
      }
      rows.append(row)
      new_rows.append(row)
      rows.sort(key=lambda result: float(result["omega"]))
      write_tsv(results_path, rows, list(row))
      print(
         f"{status} omega={omega:.8g} "
         f"chi_ms={row['mass_shedding_indicator']} "
         f"seconds={completed.seconds:.3f}"
      )

   return 0 if all(row["status"] == "PASS" for row in new_rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
