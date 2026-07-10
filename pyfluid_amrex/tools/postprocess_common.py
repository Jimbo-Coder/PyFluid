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


def physics_validation_rows(run_dir: Path, *, input_file: Path | None = None) -> list[dict[str, object]]:
   run_dir = run_dir.resolve()
   rows = read_scalar_rows(run_dir)
   found_input = find_input_file(run_dir, input_file)
   params = read_key_values(found_input)
   out: list[dict[str, object]] = []

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

   # This is level-3 postprocessing. Scalar validation always runs, while the
   # spatial pressure-bump checks only apply when the run actually wrote state.
   if params.get("pyfluid.ic", "") == "pressure_bump" and latest_plotfile(run_dir) is not None:
      from validate_explosion_isotropy import NotApplicable, validation_rows as explosion_isotropy_rows

      try:
         out.extend(explosion_isotropy_rows(run_dir, input_file=found_input))
      except NotApplicable:
         pass

   return out


def write_physics_validation(path: Path, rows: list[dict[str, object]]) -> None:
   write_tsv(path, rows, PHYSICS_POSTPROCESS_FIELDS)
