#!/usr/bin/env python3
"""Shared lightweight physics postprocessing checks for completed AMReX runs."""

from __future__ import annotations

import math
from pathlib import Path

from plotfile_reader import latest_plotfile
from validation_common import (
   find_input_file,
   has_gravity_total_energy,
   max_abs_column,
   parse_float,
   read_key_values,
   read_scalar_rows,
   read_tsv,
   write_tsv,
)


PHYSICS_POSTPROCESS_FIELDS = [
   "hook",
   "status",
   "reason",
   "metric",
   "value",
   "tolerance",
   "run_dir",
]


def active_column(rows: list[dict[str, str]], key: str) -> bool:
   return bool(rows) and key in rows[0] and any(abs(parse_float(row.get(key), 0.0)) > 0.0 for row in rows)


def param_float(params: dict[str, str], key: str, default: float) -> float:
   value = parse_float(params.get(key), default)
   return value if math.isfinite(value) else default


def passfail(value: float, tolerance: float) -> str:
   return "PASS" if math.isfinite(value) and value <= tolerance else "FAIL"


def double_mach_reflection_budget_errors(
   rows: list[dict[str, str]],
) -> tuple[float, float]:
   """Compare global changes with the analytic DMR boundary integral."""
   if not rows:
      return math.nan, math.nan

   initial = rows[0]
   final = rows[-1]
   initial_time = parse_float(initial.get("time"))
   time_value = parse_float(final.get("time"))
   elapsed_time = time_value - initial_time
   rho_post = 8.0
   pressure_post = 116.5
   velocity_1_post = 7.144709581221619
   velocity_2_post = -4.125
   shock_intersection_speed = 20.0 / math.sqrt(3.0)
   top_minus_bottom_length_at_zero = 1.0 / math.sqrt(3.0)
   domain_width = 4.0
   reflecting_segment_width = 1.0 / 6.0
   maximum_top_minus_bottom_length = domain_width - reflecting_segment_width
   saturation_time = (
      maximum_top_minus_bottom_length - top_minus_bottom_length_at_zero
   ) / shock_intersection_speed
   if time_value > saturation_time:
      # The shock has reached the outflow boundary. Its outgoing flux is
      # solution-dependent, so only the runtime finite-volume ledger remains
      # an exact global budget beyond this point.
      return math.nan, math.nan
   def integrated_boundary_factor_through(time: float) -> float:
      growing_time = min(time, saturation_time)
      return (
         top_minus_bottom_length_at_zero * growing_time
         + 0.5 * shock_intersection_speed * growing_time * growing_time
         + maximum_top_minus_bottom_length * max(time - saturation_time, 0.0)
      )

   integrated_boundary_factor = (
      integrated_boundary_factor_through(time_value)
      - integrated_boundary_factor_through(initial_time)
   )
   expected_mass_change = (
      rho_post * velocity_1_post * elapsed_time
      - rho_post * velocity_2_post * integrated_boundary_factor
   )
   energy_post = (
      pressure_post / (1.4 - 1.0)
      + 0.5 * rho_post * (velocity_1_post**2 + velocity_2_post**2)
   )
   enthalpy_density_post = energy_post + pressure_post
   expected_energy_change = (
      enthalpy_density_post * velocity_1_post * elapsed_time
      - enthalpy_density_post * velocity_2_post * integrated_boundary_factor
   )
   measured_mass_change = parse_float(final.get("mass")) - parse_float(initial.get("mass"))
   measured_energy_change = parse_float(final.get("energy")) - parse_float(initial.get("energy"))
   mass_error = abs(measured_mass_change - expected_mass_change) / max(abs(expected_mass_change), 1.0e-30)
   energy_error = abs(measured_energy_change - expected_energy_change) / max(abs(expected_energy_change), 1.0e-30)
   return mass_error, energy_error


def physics_validation_rows(run_dir: Path, *, input_file: Path | None = None) -> list[dict[str, object]]:
   run_dir = run_dir.resolve()
   rows = read_scalar_rows(run_dir)
   found_input = find_input_file(run_dir, input_file)
   params = read_key_values(found_input)
   out: list[dict[str, object]] = []

   multipole_path = run_dir / "multipoles.tsv"
   multipole_rows = read_tsv(multipole_path, require_complete=True) if multipole_path.exists() else []
   has_psi4 = any(row.get("field") == "psi4" for row in multipole_rows)
   if has_psi4:
      try:
         from postprocess_radiation import postprocess as postprocess_radiation

         postprocess_radiation(run_dir)
         status = "PASS"
         reason = ""
         value = 0.0
      except (KeyError, OSError, ValueError) as error:
         status = "FAIL"
         reason = str(error)
         value = 1.0
      out.append({
         "hook": "radiation_postprocess",
         "status": status,
         "reason": reason,
         "metric": "invalid_runtime_history",
         "value": value,
         "tolerance": 0.0,
         "run_dir": str(run_dir),
      })

   if active_column(rows, "wall_work") and "wall_work_rel_error" in (rows[0] if rows else {}):
      tol = param_float(params, "pyfluid.validation_wall_work_tol", 1.0e-5)
      value = max_abs_column(rows, "wall_work_rel_error")
      out.append({
         "hook": "wall_work",
         "status": passfail(value, tol),
         "reason": "" if passfail(value, tol) == "PASS" else f"wall_work_rel_error={value:.3e} > {tol:.3e}",
         "metric": "wall_work_rel_error",
         "value": value,
         "tolerance": tol,
         "run_dir": str(run_dir),
      })

   if has_gravity_total_energy(rows):
      tol = param_float(params, "pyfluid.validation_gravity_energy_tol", 1.0e-4)
      value = max_abs_column(rows, "gravity_total_energy_rel_drift")
      out.append({
         "hook": "gravity_total_energy",
         "status": passfail(value, tol),
         "reason": "" if passfail(value, tol) == "PASS" else f"gravity_total_energy_rel_drift={value:.3e} > {tol:.3e}",
         "metric": "gravity_total_energy_rel_drift",
         "value": value,
         "tolerance": tol,
         "run_dir": str(run_dir),
      })

   if params.get("pyfluid.boundary_profile", "") == "double_mach_reflection":
      tolerance = param_float(params, "pyfluid.validation_dmr_budget_tol", 1.0e-2)
      mass_error, energy_error = double_mach_reflection_budget_errors(rows)
      for metric, value in (
         ("analytic_mass_budget_relative_error", mass_error),
         ("analytic_energy_budget_relative_error", energy_error),
      ):
         if not math.isfinite(value):
            continue
         status = passfail(value, tolerance)
         out.append({
            "hook": "double_mach_reflection_budget",
            "status": status,
            "reason": "" if status == "PASS" else f"{metric}={value:.3e} > {tolerance:.3e}",
            "metric": metric,
            "value": value,
            "tolerance": tolerance,
            "run_dir": str(run_dir),
         })

   # This is level-3 postprocessing. Scalar validation always runs, while the
   # spatial pressure-bump checks only apply when the run actually wrote state.
   if params.get("pyfluid.ic", "") == "pressure_bump" and latest_plotfile(run_dir) is not None:
      from validate_causal_quiescence import physics_validation_row as causal_quiescence_row
      from validate_explosion_isotropy import NotApplicable, validation_rows as explosion_isotropy_rows

      causal_row = causal_quiescence_row(run_dir, params)
      if causal_row is not None:
         out.append(causal_row)
      try:
         out.extend(explosion_isotropy_rows(run_dir, input_file=found_input))
      except NotApplicable:
         pass

   return out


def write_physics_validation(path: Path, rows: list[dict[str, object]]) -> None:
   write_tsv(path, rows, PHYSICS_POSTPROCESS_FIELDS)
