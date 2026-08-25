#!/usr/bin/env python3
"""Validate and summarize one completed solve-only physical-ID run."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from validation_common import ROOT, read_key_values, repo_relative, write_tsv
from physical_id_metadata import (
   physical_id_component_labels,
   physical_id_metadata_real,
   read_physical_id_metadata,
)


FIELDS = (
   "status",
   "reason",
   "case",
   "dim",
   "cells",
   "wall_seconds",
   "source_ic",
   "spacetime_initial_data",
   "output_file",
   "artifact_bytes",
   "component_count",
   "components",
   "matter_iterations",
   "relative_matter_update",
   "coupled_iterations",
   "hamiltonian_residual",
   "lapse_residual",
   "momentum_residual",
   "maximum_density",
   "maximum_velocity",
   "equatorial_radius",
   "polar_radius",
   "mass_shedding_indicator",
   "adm_mass",
   "baryonic_mass",
   "angular_momentum",
   "dimensionless_spin",
   "rotation_law",
   "angular_velocity",
   "rotation_axis1",
   "rotation_axis2",
   "rotation_axis3",
   "rotation_center1",
   "rotation_center2",
   "rotation_center3",
   "disk_gravity",
   "disk_rotation_law",
   "disk_specific_angular_momentum",
   "disk_angular_velocity",
   "disk_differential_rotation_length",
   "disk_rotation_power_q",
   "disk_axis1",
   "disk_axis2",
   "disk_axis3",
   "disk_requested_inner_radius",
   "disk_resolved_inner_radius",
   "disk_outer_radius",
   "disk_density_maximum_radius",
   "disk_maximum_density",
   "disk_maximum_velocity",
   "disk_maximum_log_specific_enthalpy",
   "disk_baryonic_mass",
   "disk_angular_momentum",
   "magnetic_seed",
   "magnetic_seed_updates",
   "magnetic_equilibrium",
   "magnetic_equilibrium_updates",
   "magnetic_potential_update",
   "magnetic_linear_residual",
   "magnetic_equation_residual",
   "magnetic_bernoulli_residual",
   "maximum_comoving_field",
   "magnetic_divb_linf",
   "preflight_kind",
   "max_conserved_error",
   "max_primitive_error",
   "log",
)

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _match(text: str, pattern: str) -> re.Match[str] | None:
   matches = list(re.finditer(pattern, text, re.MULTILINE))
   return matches[-1] if matches else None


def _artifact_path(params: dict[str, str]) -> Path:
   value = Path(params["initial_data.output_file"])
   return value if value.is_absolute() else ROOT / value


def _artifact_metadata(
   artifact: Path, reasons: list[str]
) -> dict[str, str]:
   metadata_path = artifact / "Header.meta"
   if not metadata_path.is_file():
      return {}
   try:
      return read_physical_id_metadata(artifact)
   except (OSError, ValueError) as error:
      reasons.append(str(error))
      return {}


def _metadata_float(
   metadata: dict[str, str], key: str, fallback: float = math.nan
) -> float:
   return physical_id_metadata_real(metadata, key, fallback)


def _parameter_choice(params: dict[str, str], key: str, default: str) -> str:
   """Read a string choice from source or canonical quoted run input."""
   return params.get(key, default).strip().strip("\"'")


def initial_data_validation_row(run_dir: Path, input_file: Path) -> dict[str, object]:
   params = read_key_values(input_file)
   log_path = run_dir / "run.log"
   text = log_path.read_text(errors="replace") if log_path.exists() else ""
   n_cell = [int(value) for value in params.get("amr.n_cell", "").split()]
   artifact = _artifact_path(params)
   reasons: list[str] = []
   artifact_metadata = _artifact_metadata(artifact, reasons)
   if params.get("initial_data.solve_only", "0") not in ("1", "true", "True"):
      reasons.append("canonical input is not solve-only")
   if "completed initial-data-only solve and artifact write" not in text:
      reasons.append("missing completion marker")
   required = [artifact / "Header", artifact / "Header.meta", artifact / "physical_fluid.h5"]
   if artifact_metadata.get("has_spacetime") == "1":
      required.append(artifact / "adm_geometry.h5")
   if artifact_metadata.get("has_ct") == "1":
      required.extend(artifact / f"face_b{direction}.h5" for direction in range(1, 4))
   missing = [path.name for path in required if not path.is_file()]
   if missing:
      reasons.append("missing artifact fields: " + ",".join(missing))

   constraint = _match(
      text,
      rf"(?:ctt|cttk|xcts) initial data: matter_iterations=(\d+) "
      rf"relative_matter_update=({FLOAT}) coupled_iterations=(\d+) "
      rf"hamiltonian_residual=({FLOAT}) lapse_residual=({FLOAT}) "
      rf"momentum_residual=({FLOAT})",
   )
   rotating = _match(
      text,
      rf"rotating star:.*?maximum_density=({FLOAT}).*?maximum_velocity=({FLOAT})"
      rf".*?equatorial_radius=({FLOAT}).*?polar_radius=({FLOAT})"
      rf".*?mass_shedding_indicator=({FLOAT})",
   )
   global_diagnostic = _match(
      text,
      rf"rotating star:.*?adm_mass=({FLOAT}).*?baryonic_mass=({FLOAT})"
      rf".*?angular_momentum=({FLOAT}).*?dimensionless_spin=({FLOAT})",
   )
   equilibrium_torus = _match(
      text,
      rf"equilibrium torus: gravity=(\w+) rotation_law=(\w+) "
      rf"specific_angular_momentum=({FLOAT}) angular_velocity=({FLOAT}) "
      rf"differential_rotation_length=({FLOAT}) rotation_power_q=({FLOAT}) "
      rf"requested_inner_radius=({FLOAT}) resolved_inner_radius=({FLOAT}) "
      rf"outer_radius=({FLOAT}) density_maximum_radius=({FLOAT}) "
      rf"maximum_density=({FLOAT}) maximum_velocity=({FLOAT}) "
      rf"maximum_log_specific_enthalpy=({FLOAT}) "
      rf"baryonic_mass=({FLOAT}) angular_momentum=({FLOAT})",
   )
   preflight = _match(
      text,
      rf"Evolution preflight C2P/P2C:.*?max_conserved_error=({FLOAT}) "
      rf"max_primitive_error=({FLOAT})",
   )
   physical_id_preflight = (
      "Physical-ID preflight: finite non-ideal RMHD primitive and conserved fields"
      in text
   )
   magnetic = _match(
      text,
      rf"pressure-poloidal magnetic ID: updates=(\d+) "
      rf"maximum_pressure=({FLOAT}) normalization=({FLOAT}) "
      rf"maximum_comoving_field=({FLOAT}) divb_linf=({FLOAT})",
   )
   magnetic_equilibrium = _match(
      text,
      rf"poloidal Grad-Shafranov magnetic ID: updates=(\d+) "
      rf"relative_potential_update=({FLOAT}) linear_residual=({FLOAT}) "
      rf"equation_relative_residual=({FLOAT}) "
      rf"bernoulli_residual=({FLOAT}) "
      rf"maximum_comoving_field=({FLOAT}) divb_linf=({FLOAT})",
   )
   wall = _match(text, rf"^# wall_seconds = ({FLOAT})$")
   if preflight is None and not physical_id_preflight:
      reasons.append("missing C2P/P2C preflight")
   magnetic_seed = _parameter_choice(
      params, "initial_data.magnetic_seed", "none"
   )
   magnetic_equilibrium_name = _parameter_choice(
      params,
      "initial_data.magnetic_equilibrium", "none"
   )
   if magnetic_seed != "none":
      if magnetic is None:
         reasons.append("missing magnetic-ID summary")
      else:
         requested_field = float(
            params["initial_data.magnetic_seed_maximum_comoving_field"]
         )
         measured_field = float(magnetic.group(4))
         if not math.isclose(
            measured_field, requested_field, rel_tol=1.0e-12, abs_tol=0.0
         ):
            reasons.append("magnetic-ID normalization mismatch")
         if float(magnetic.group(5)) > 1.0e-14:
            reasons.append("magnetic-ID divergence exceeds roundoff gate")
   if magnetic_equilibrium_name != "none" \
         and float(params.get(
            "initial_data.magnetic_equilibrium_poloidal_magnetization", "0"
         )) != 0.0:
      if magnetic_equilibrium is None:
         reasons.append("missing magnetohydrostatic-ID summary")
      else:
         if float(magnetic_equilibrium.group(2)) > 1.1e-6:
            reasons.append("magnetic-potential update exceeds tolerance")
         if float(magnetic_equilibrium.group(3)) > 1.0e-12:
            reasons.append("magnetic linear residual exceeds tolerance")
         if float(magnetic_equilibrium.group(4)) > 3.0e-6:
            reasons.append("magnetic equation residual exceeds tolerance")
         if float(magnetic_equilibrium.group(5)) > 1.1e-6:
            reasons.append("magnetic Bernoulli residual exceeds tolerance")
         if float(magnetic_equilibrium.group(7)) > 1.0e-14:
            reasons.append("magnetohydrostatic-ID divergence exceeds roundoff gate")

   def value(match: re.Match[str] | None, group: int) -> float:
      return float(match.group(group)) if match is not None else math.nan

   try:
      component_labels = physical_id_component_labels(artifact_metadata)
   except ValueError as error:
      reasons.append(str(error))
      component_labels = ()
   component_count = len(component_labels)
   components = ",".join(component_labels)

   row: dict[str, object] = {
      "status": "PASS" if not reasons else "FAIL",
      "reason": "; ".join(reasons),
      "case": params.get("pyfluid.run_name", run_dir.name),
      "dim": len(n_cell),
      "cells": math.prod(n_cell) if n_cell else 0,
      "wall_seconds": value(wall, 1),
      "source_ic": artifact_metadata.get(
         "source_ic", params.get("pyfluid.ic", "")
      ),
      "spacetime_initial_data": artifact_metadata.get(
         "spacetime_initial_data", params.get("spacetime.initial_data", "")
      ),
      "output_file": repo_relative(artifact),
      "artifact_bytes": sum(
         path.stat().st_size for path in artifact.rglob("*") if path.is_file()
      ) if artifact.is_dir() else 0,
      "component_count": component_count,
      "components": components,
      "matter_iterations": int(constraint.group(1)) if constraint else 0,
      "relative_matter_update": value(constraint, 2),
      "coupled_iterations": int(constraint.group(3)) if constraint else 0,
      "hamiltonian_residual": value(constraint, 4),
      "lapse_residual": value(constraint, 5),
      "momentum_residual": value(constraint, 6),
      "maximum_density": _metadata_float(
         artifact_metadata, "maximum_density", value(rotating, 1)
      ),
      "maximum_velocity": _metadata_float(
         artifact_metadata, "maximum_velocity", value(rotating, 2)
      ),
      "equatorial_radius": _metadata_float(
         artifact_metadata, "equatorial_radius", value(rotating, 3)
      ),
      "polar_radius": _metadata_float(
         artifact_metadata, "polar_radius", value(rotating, 4)
      ),
      "mass_shedding_indicator": _metadata_float(
         artifact_metadata, "mass_shedding_indicator", value(rotating, 5)
      ),
      "adm_mass": _metadata_float(
         artifact_metadata, "adm_mass", value(global_diagnostic, 1)
      ),
      "baryonic_mass": _metadata_float(
         artifact_metadata, "baryonic_mass", value(global_diagnostic, 2)
      ),
      "angular_momentum": _metadata_float(
         artifact_metadata, "angular_momentum", value(global_diagnostic, 3)
      ),
      "dimensionless_spin": _metadata_float(
         artifact_metadata, "dimensionless_spin", value(global_diagnostic, 4)
      ),
      "rotation_law": artifact_metadata.get(
         "rotation_law", params.get("initial_data.rotation_law", "uniform")
      ),
      "angular_velocity": _metadata_float(artifact_metadata, "angular_velocity", 0.0),
      "rotation_axis1": _metadata_float(artifact_metadata, "rotation_axis1", 0.0),
      "rotation_axis2": _metadata_float(artifact_metadata, "rotation_axis2", 0.0),
      "rotation_axis3": _metadata_float(artifact_metadata, "rotation_axis3", 1.0),
      "rotation_center1": _metadata_float(artifact_metadata, "rotation_center1", 0.0),
      "rotation_center2": _metadata_float(artifact_metadata, "rotation_center2", 0.0),
      "rotation_center3": _metadata_float(artifact_metadata, "rotation_center3", 0.0),
      "disk_gravity": artifact_metadata.get("disk_gravity", ""),
      "disk_rotation_law": artifact_metadata.get("disk_rotation_law", ""),
      "disk_specific_angular_momentum": _metadata_float(
         artifact_metadata, "disk_specific_angular_momentum"
      ),
      "disk_angular_velocity": _metadata_float(
         artifact_metadata, "disk_angular_velocity"
      ),
      "disk_differential_rotation_length": _metadata_float(
         artifact_metadata, "disk_differential_rotation_length"
      ),
      "disk_rotation_power_q": _metadata_float(
         artifact_metadata, "disk_rotation_power_q"
      ),
      "disk_axis1": _metadata_float(artifact_metadata, "disk_axis1", 0.0),
      "disk_axis2": _metadata_float(artifact_metadata, "disk_axis2", 0.0),
      "disk_axis3": _metadata_float(artifact_metadata, "disk_axis3", 1.0),
      "disk_requested_inner_radius": _metadata_float(
         artifact_metadata, "disk_inner_radius", value(equilibrium_torus, 7)
      ),
      "disk_resolved_inner_radius": _metadata_float(
         artifact_metadata, "disk_resolved_inner_radius", value(equilibrium_torus, 8)
      ),
      "disk_outer_radius": _metadata_float(
         artifact_metadata, "disk_outer_radius", value(equilibrium_torus, 9)
      ),
      "disk_density_maximum_radius": _metadata_float(
         artifact_metadata, "disk_density_maximum_radius", value(equilibrium_torus, 10)
      ),
      "disk_maximum_density": _metadata_float(
         artifact_metadata, "disk_maximum_density", value(equilibrium_torus, 11)
      ),
      "disk_maximum_velocity": _metadata_float(
         artifact_metadata, "disk_maximum_velocity", value(equilibrium_torus, 12)
      ),
      "disk_maximum_log_specific_enthalpy": _metadata_float(
         artifact_metadata,
         "disk_maximum_log_specific_enthalpy",
         value(equilibrium_torus, 13),
      ),
      "disk_baryonic_mass": _metadata_float(
         artifact_metadata, "disk_baryonic_mass", value(equilibrium_torus, 14)
      ),
      "disk_angular_momentum": _metadata_float(
         artifact_metadata, "disk_angular_momentum", value(equilibrium_torus, 15)
      ),
      "magnetic_seed": artifact_metadata.get("magnetic_seed", magnetic_seed),
      "magnetic_seed_updates": int(magnetic.group(1)) if magnetic else 0,
      "magnetic_equilibrium": artifact_metadata.get(
         "magnetic_equilibrium", magnetic_equilibrium_name
      ),
      "magnetic_equilibrium_updates": int(magnetic_equilibrium.group(1))
         if magnetic_equilibrium else 0,
      "magnetic_potential_update": value(magnetic_equilibrium, 2),
      "magnetic_linear_residual": value(magnetic_equilibrium, 3),
      "magnetic_equation_residual": value(magnetic_equilibrium, 4),
      "magnetic_bernoulli_residual": value(magnetic_equilibrium, 5),
      "maximum_comoving_field": value(magnetic_equilibrium, 6)
         if magnetic_equilibrium else value(magnetic, 4),
      "magnetic_divb_linf": value(magnetic_equilibrium, 7)
         if magnetic_equilibrium else value(magnetic, 5),
      "preflight_kind": "c2p_p2c" if preflight else (
         "physical_id_finite" if physical_id_preflight else ""
      ),
      "max_conserved_error": value(preflight, 1),
      "max_primitive_error": value(preflight, 2),
      "log": repo_relative(log_path),
   }
   return row


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("run_dir", type=Path)
   parser.add_argument("--input", type=Path)
   parser.add_argument("--output", type=Path)
   args = parser.parse_args()
   run_dir = args.run_dir.resolve()
   input_file = (args.input or run_dir / "run.in").resolve()
   output = (args.output or run_dir / "initial_data_report.tsv").resolve()
   row = initial_data_validation_row(run_dir, input_file)
   write_tsv(output, [row], FIELDS)
   print(
      f"{row['status']} initial_data:{row['case']} "
      f"cells={row['cells']} wall={float(row['wall_seconds']):.3f}s "
      f"H={float(row['hamiltonian_residual']):.3e} "
      f"M={float(row['momentum_residual']):.3e} "
      f"preflight={row['preflight_kind']} "
      f"divb={float(row['magnetic_divb_linf']):.3e} "
      f"artifact={int(row['artifact_bytes'])}B"
      + (f" ({row['reason']})" if row["reason"] else "")
   )
   print(f"wrote {repo_relative(output)}")
   return 0 if row["status"] == "PASS" else 1


if __name__ == "__main__":
   raise SystemExit(main())
