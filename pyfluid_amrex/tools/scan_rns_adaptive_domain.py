#!/usr/bin/env python3
"""Scan source/exterior radial allocation for reduced spherical RNS ID."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np

from validation_common import (
   ROOT,
   materialize_run_input,
   method_catalog_names,
   read_tsv,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_spherical_id_2d.in"
RUN_ROOT = ROOT / "runs/current/rotating_star_adaptive_domain_scan"
RESULTS = RUN_ROOT / "results.tsv"
ROTATION_LAWS = method_catalog_names("rotation_law")
SOLVE_PATTERN = re.compile(
   r"xcts initial data: matter_iterations=(?P<matter>[0-9]+) "
   r"relative_matter_update=(?P<update>[0-9.eE+-]+) "
   r"coupled_iterations=(?P<coupled>[0-9]+) "
   r"hamiltonian_residual=(?P<hamiltonian>[0-9.eE+-]+) "
   r"lapse_residual=(?P<lapse>[0-9.eE+-]+) "
   r"momentum_residual=(?P<momentum>[0-9.eE+-]+)"
)
MATTER_PROGRESS_PATTERN = re.compile(
   r"matter_iteration=(?P<matter>[0-9]+) "
   r"relative_update=(?P<update>[0-9.eE+-]+)"
)
SURFACE_REJECTION = (
   "rotating-star surface ray does not contain a resolved enthalpy zero"
)


def read_metadata(artifact: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.removeprefix("#").strip()
      if "=" in line:
         key, value = line.split("=", 1)
         values[key.strip()] = value.strip()
   return values


def read_parameters(path: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   if not path.is_file():
      return values
   for raw_line in path.read_text().splitlines():
      line = raw_line.split("#", 1)[0].strip()
      if "=" in line:
         key, value = line.split("=", 1)
         values[key.strip()] = value.strip()
   return values


def case_name(
   source_factor: float,
   exterior_fraction: float,
   radial_cells: int,
   angular_cells: int,
   omega: float,
   label: str,
   rotation_law: str,
   differential_rotation_length: float,
   rotation_power_q: float,
) -> str:
   name = (
      f"r{radial_cells}_t{angular_cells}_omega{omega:.6g}_"
      f"source{source_factor:.6g}_exterior{exterior_fraction:.6g}"
      .replace(".", "p")
   )
   if rotation_law != "uniform":
      name += (
         f"_{rotation_law}_a{differential_rotation_length:.6g}"
      ).replace(".", "p")
   if rotation_law == "q_power":
      name += f"_q{rotation_power_q:.6g}".replace(".", "p")
   return f"{name}_{label}" if label else name


def scan_cases(args: argparse.Namespace) -> list[tuple[float, float]]:
   reference_source = 1.8
   reference_exterior = 1.0 / 3.0
   source_values = args.source_factor
   exterior_values = args.exterior_fraction
   if source_values is None and exterior_values is None:
      source_values = (1.4, 1.5, 1.65, 1.8)
      exterior_values = (0.125, 1.0 / 6.0, 0.25, 1.0 / 3.0, 0.5)
   cases = {
      (float(source), reference_exterior)
      for source in (source_values or ())
   }
   cases.update(
      (reference_source, float(exterior))
      for exterior in (exterior_values or ())
   )
   return sorted(cases)


def equivalent_rotation_law(
   metadata: dict[str, str],
   rotation_law: str,
   differential_rotation_length: float,
   rotation_power_q: float,
) -> bool:
   reference_law = metadata.get("rotation_law", "uniform")
   if reference_law == "uniform" or rotation_law == "uniform":
      return reference_law == rotation_law
   reference_length = float(metadata["differential_rotation_length"])
   if not math.isclose(
      reference_length,
      differential_rotation_length,
      rel_tol=0.0,
      abs_tol=1.0e-15,
   ):
      return False
   reference_q = float(metadata.get("rotation_power_q", "1.0"))
   if reference_law == rotation_law:
      return rotation_law != "q_power" or math.isclose(
         reference_q, rotation_power_q, rel_tol=0.0, abs_tol=1.0e-15
      )
   return {reference_law, rotation_law} == {"j_constant", "q_power"} \
      and math.isclose(reference_q, 1.0, rel_tol=0.0, abs_tol=1.0e-15) \
      and math.isclose(rotation_power_q, 1.0, rel_tol=0.0, abs_tol=1.0e-15)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--base-input",
      type=Path,
      default=BASE_INPUT,
      help="rotating-star input whose EoS and ID policy seed the scan",
   )
   parser.add_argument("--input-artifact", type=Path)
   parser.add_argument("--reference-artifact", type=Path)
   parser.add_argument("--continuation-targets", type=float, nargs="+")
   parser.add_argument("--omega", type=float, default=0.0329)
   parser.add_argument(
      "--rotation-law", choices=ROTATION_LAWS, default="uniform"
   )
   parser.add_argument(
      "--differential-rotation-length", type=float, default=10.0
   )
   parser.add_argument("--rotation-power-q", type=float, default=1.0)
   parser.add_argument("--radial-cells", type=int, default=256)
   parser.add_argument(
      "--angular-cells", type=int, nargs="+", default=(64,)
   )
   parser.add_argument(
      "--source-factor", type=float, nargs="+"
   )
   parser.add_argument(
      "--exterior-fraction",
      type=float,
      nargs="+",
   )
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--matter-relaxation", type=float, default=0.2)
   parser.add_argument("--matter-tolerance", type=float, default=1.0e-7)
   parser.add_argument("--matter-max-iterations", type=int, default=128)
   parser.add_argument(
      "--label",
      default="",
      help="optional safe suffix distinguishing scan provenance/controls",
   )
   parser.add_argument(
      "--expect-surface-rejection",
      action="store_true",
      help="require the resolved enthalpy-surface guard to reject every case",
   )
   parser.add_argument(
      "--save-sequence",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="persist every requested continuation target as a reusable artifact",
   )
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   base_input = args.base_input.resolve()
   if not base_input.is_file():
      raise SystemExit("--base-input must name an existing input file")
   if base_input != BASE_INPUT.resolve() and not args.label:
      raise SystemExit("a nondefault --base-input requires --label")

   input_artifact = (
      args.input_artifact.resolve() if args.input_artifact is not None else None
   )
   if input_artifact is not None \
         and not (input_artifact / "Header.meta").is_file():
      raise SystemExit("--input-artifact must name a physical-ID artifact")
   reference_artifact = (
      args.reference_artifact.resolve()
      if args.reference_artifact is not None else None
   )
   if reference_artifact is not None \
         and not (reference_artifact / "Header.meta").is_file():
      raise SystemExit("--reference-artifact must name a physical-ID artifact")
   if input_artifact is None and not args.continuation_targets:
      raise SystemExit(
         "a fresh rotating-star scan needs --continuation-targets"
      )
   if args.radial_cells < 32 or any(
      angular_cells < 8 for angular_cells in args.angular_cells
   ):
      raise SystemExit("RNS scan grid is too small")
   if not math.isfinite(args.omega) or args.omega <= 0.0:
      raise SystemExit("--omega must be positive and finite")
   if not math.isfinite(args.differential_rotation_length) \
         or args.differential_rotation_length <= 0.0:
      raise SystemExit(
         "--differential-rotation-length must be positive and finite"
      )
   if not math.isfinite(args.rotation_power_q) \
         or args.rotation_power_q < 1.0:
      raise SystemExit("--rotation-power-q must be finite and at least one")
   differential_rotation_length = (
      args.differential_rotation_length
      if args.rotation_law != "uniform" else 10.0
   )
   rotation_power_q = (
      args.rotation_power_q if args.rotation_law == "q_power" else 1.0
   )
   if args.continuation_targets and (
      any(
         not math.isfinite(target) or not 0.0 < target < args.omega
         for target in args.continuation_targets
      )
      or any(
         right <= left
         for left, right in zip(
            args.continuation_targets, args.continuation_targets[1:]
         )
      )
   ):
      raise SystemExit(
         "--continuation-targets must increase strictly between zero and omega"
      )
   if not 0.0 < args.matter_relaxation <= 1.0 \
         or not 0.0 < args.matter_tolerance < 1.0 \
         or args.matter_max_iterations < 1:
      raise SystemExit("invalid matter iteration controls")
   if args.label and re.fullmatch(r"[A-Za-z0-9_-]+", args.label) is None:
      raise SystemExit("--label may contain only letters, digits, '-' and '_'")
   cases = scan_cases(args)
   if any(not 1.0 < source <= 3.0 for source, _ in cases):
      raise SystemExit("source factors must be in (1, 3]")
   if any(not 0.125 <= exterior <= 0.5 for _, exterior in cases):
      raise SystemExit("exterior fractions must be in [0.125, 0.5]")

   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   existing = {
      (
         float(row["source_radius_factor"]),
         float(row["exterior_fraction"]),
         int(row["radial_cells"]),
         int(row["angular_cells"]),
         float(row.get("omega") or args.omega),
         row.get("label") or "",
         row.get("rotation_law") or "uniform",
         float(row.get("differential_rotation_length") or 10.0),
         float(row.get("rotation_power_q") or 1.0),
      ): row
      for row in read_tsv(RESULTS)
   } if RESULTS.is_file() else {}
   selected_keys: set[
      tuple[float, float, int, int, float, str, str, float, float]
   ] = set()
   scan_points = [
      (source_factor, exterior_fraction, angular_cells)
      for source_factor, exterior_fraction in cases
      for angular_cells in sorted(set(args.angular_cells))
   ]
   for source_factor, exterior_fraction, angular_cells in scan_points:
      key = (
         source_factor,
         exterior_fraction,
         args.radial_cells,
         angular_cells,
         args.omega,
         args.label,
         args.rotation_law,
         differential_rotation_length,
         rotation_power_q,
      )
      selected_keys.add(key)
      name = case_name(*key)
      run_dir = RUN_ROOT / name
      artifact = run_dir / "physical_id"
      if args.force and run_dir.exists():
         shutil.rmtree(run_dir)
      cached_log = (
         (run_dir / "run.log").read_text()
         if (run_dir / "run.log").is_file() else ""
      )
      has_saved_sequence = any(
         (run_dir / "initial_data_sequences").glob("**/sequence.tsv")
      )
      cached_surface_rejection = (
         args.expect_surface_rejection
         and key in existing
         and SURFACE_REJECTION in cached_log
         and (not args.save_sequence or has_saved_sequence)
      )
      if not args.force and cached_surface_rejection:
         progress_matches = list(MATTER_PROGRESS_PATTERN.finditer(cached_log))
         progress = progress_matches[-1] if progress_matches else None
         existing[key]["status"] = "EXPECTED_REJECTION"
         existing[key]["rejection"] = "resolved_enthalpy_surface"
         if progress is not None:
            existing[key]["matter_iterations"] = int(
               progress.group("matter")
            )
            existing[key]["relative_matter_update"] = float(
               progress.group("update")
            )
         print(f"CACHED EXPECTED_REJECTION {name}")
         continue
      if not args.force and key in existing and artifact.is_dir() \
            and (not args.save_sequence or has_saved_sequence):
         print(f"CACHED {name}")
         continue
      run_dir.mkdir(parents=True, exist_ok=True)
      input_path = materialize_run_input(
         base_input,
         run_dir / "input.in",
         {
            "amr.n_cell": f"{args.radial_cells} {angular_cells}",
            "amr.max_grid_size": str(min(64, args.radial_cells)),
            "initial_data.output_file": repo_relative(artifact),
            "initial_data.angular_velocity": f"{args.omega:.17g}",
            "initial_data.rotation_law": args.rotation_law,
            "initial_data.differential_rotation_length": (
               f"{differential_rotation_length:.17g}"
            ),
            "initial_data.rotation_power_q": (
               f"{rotation_power_q:.17g}"
            ),
            "initial_data.adaptive_source_radius_factor": (
               f"{source_factor:.17g}"
            ),
            "initial_data.adaptive_exterior_fraction": (
               f"{exterior_fraction:.17g}"
            ),
            "initial_data.matter_relaxation": (
               f"{args.matter_relaxation:.17g}"
            ),
            "initial_data.matter_relative_tolerance": (
               f"{args.matter_tolerance:.17g}"
            ),
            "initial_data.matter_max_iterations": str(
               args.matter_max_iterations
            ),
            "initial_data.save_sequence": "1" if args.save_sequence else "0",
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            **(
               {"initial_data.input_file": repo_relative(input_artifact)}
               if input_artifact is not None else {}
            ),
            **(
               {
                  "initial_data.rotation_continuation_targets": " ".join(
                     f"{target:.17g}" for target in args.continuation_targets
                  )
               }
               if args.continuation_targets else {}
            ),
         },
      )
      completed = run_amrex(
         dim=2,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      solve_matches = list(SOLVE_PATTERN.finditer(completed.stdout))
      solve = solve_matches[-1] if solve_matches else None
      progress_matches = list(
         MATTER_PROGRESS_PATTERN.finditer(completed.stdout)
      )
      progress = progress_matches[-1] if progress_matches else None
      matter_iterations = (
         int(solve.group("matter")) if solve is not None
         else int(progress.group("matter")) if progress is not None
         else ""
      )
      relative_matter_update = (
         float(solve.group("update")) if solve is not None
         else float(progress.group("update")) if progress is not None
         else ""
      )
      surface_rejected = (
         completed.returncode != 0 and SURFACE_REJECTION in completed.stdout
      )
      if args.expect_surface_rejection:
         if surface_rejected:
            status = "EXPECTED_REJECTION"
         elif completed.returncode == 0:
            status = "UNEXPECTED_ACCEPTANCE"
         else:
            status = "FAIL"
      else:
         status = "PASS" if completed.returncode == 0 and solve else "FAIL"
      metadata = read_metadata(artifact) if artifact.is_dir() else {}
      faces = np.fromstring(metadata.get("coordinate_faces_0_0", ""), sep=",")
      source_cells = 0
      source_radius = math.nan
      inner_dr = math.nan
      outer_dr = math.nan
      if faces.size == args.radial_cells + 1 and np.all(np.diff(faces) > 0.0):
         widths = np.diff(faces)
         tolerance = 64.0 * np.finfo(float).eps * max(1.0, widths[0])
         changed = np.flatnonzero(np.abs(widths - widths[0]) > tolerance)
         source_cells = int(changed[0]) if changed.size else widths.size
         source_radius = float(faces[source_cells])
         inner_dr = float(widths[0])
         outer_dr = float(widths[-1])
      diagnostics = {
         key_name: float(metadata.get(key_name, "nan"))
         for key_name in (
            "equatorial_radius",
            "polar_radius",
            "matter_outer_radius",
            "injection_energy",
            "equatorial_log_enthalpy_gradient",
            "polar_log_enthalpy_gradient",
            "mass_shedding_indicator",
            "adm_mass",
            "baryonic_mass",
            "angular_momentum",
            "dimensionless_spin",
         )
      }
      if not math.isfinite(diagnostics["matter_outer_radius"]):
         diagnostics["matter_outer_radius"] = max(
            diagnostics["equatorial_radius"], diagnostics["polar_radius"]
         )
      if status == "PASS" and (
         not all(math.isfinite(value) for value in diagnostics.values())
         or diagnostics["mass_shedding_indicator"] <= 0.0
      ):
         status = "FAIL_PHYSICS"
      row: dict[str, object] = {
         "kind": "initial_data",
         "case": name,
         "status": status,
         "rejection": (
            "resolved_enthalpy_surface" if surface_rejected else ""
         ),
         "label": args.label,
         "omega": args.omega,
         "rotation_law": args.rotation_law,
         "differential_rotation_length": (
            differential_rotation_length
         ),
         "rotation_power_q": rotation_power_q,
         "source_radius_factor": source_factor,
         "exterior_fraction": exterior_fraction,
         "radial_cells": args.radial_cells,
         "angular_cells": angular_cells,
         "source_cells": source_cells,
         "exterior_cells": args.radial_cells - source_cells,
         "source_radius": source_radius,
         "source_headroom": (
            source_radius / diagnostics["matter_outer_radius"]
            if diagnostics["matter_outer_radius"] > 0.0 else math.nan
         ),
         "inner_dr": inner_dr,
         "outer_dr": outer_dr,
         "matter_relaxation": args.matter_relaxation,
         "matter_tolerance": args.matter_tolerance,
         "matter_max_iterations": args.matter_max_iterations,
         "saved_sequence": int(args.save_sequence),
         "seconds": completed.seconds,
         "returncode": completed.returncode,
         "matter_iterations": matter_iterations,
         "relative_matter_update": relative_matter_update,
         "hamiltonian_residual": (
            float(solve.group("hamiltonian")) if solve else ""
         ),
         "lapse_residual": float(solve.group("lapse")) if solve else "",
         "momentum_residual": float(solve.group("momentum")) if solve else "",
         **diagnostics,
         "input_artifact": (
            repo_relative(input_artifact) if input_artifact is not None else ""
         ),
         "run_dir": repo_relative(run_dir),
         "base_input": repo_relative(base_input),
      }
      existing[key] = row
      current_rows = list(existing.values())
      current_fields = list(dict.fromkeys(
         field for current_row in current_rows for field in current_row
      ))
      write_tsv(RESULTS, current_rows, current_fields)
      print(
         f"{status} {name} source={source_cells} exterior="
         f"{args.radial_cells - source_cells} chi_ms="
         f"{diagnostics['mass_shedding_indicator']:.6g} "
         f"seconds={completed.seconds:.3f}"
      )

   rows = sorted(
      existing.values(),
      key=lambda row: (
         int(row["radial_cells"]),
         int(row["angular_cells"]),
         float(row.get("omega") or args.omega),
         row.get("label") or "",
         float(row["source_radius_factor"]),
         float(row["exterior_fraction"]),
      )
   )
   for row in rows:
      run_dir = ROOT / str(row["run_dir"])
      artifact = run_dir / "physical_id"
      if artifact.is_dir():
         metadata = read_metadata(artifact)
         row["omega"] = float(metadata.get("angular_velocity", args.omega))
         matter_outer_radius = float(
            metadata.get(
               "matter_outer_radius",
               max(
                  float(metadata.get("equatorial_radius", "nan")),
                  float(metadata.get("polar_radius", "nan")),
               ),
            )
         )
         row["matter_outer_radius"] = matter_outer_radius
         row["source_headroom"] = (
            float(row["source_radius"]) / matter_outer_radius
            if matter_outer_radius > 0.0 else math.nan
         )
      parameters = read_parameters(run_dir / "run.in")
      for output_name, parameter_name, cast in (
         ("matter_relaxation", "initial_data.matter_relaxation", float),
         ("matter_tolerance", "initial_data.matter_relative_tolerance", float),
         ("matter_max_iterations", "initial_data.matter_max_iterations", int),
      ):
         if parameter_name in parameters:
            row[output_name] = cast(parameters[parameter_name])
   if reference_artifact is not None:
      reference = read_metadata(reference_artifact)
      reference_omega = float(reference["angular_velocity"])
      if not math.isclose(reference_omega, args.omega, rel_tol=0.0, abs_tol=1e-15):
         raise RuntimeError("reference artifact angular velocity does not match")
      if not equivalent_rotation_law(
         reference,
         args.rotation_law,
         differential_rotation_length,
         rotation_power_q,
      ):
         raise RuntimeError("reference artifact rotation law does not match")
      for row in rows:
         row_key = (
            float(row["source_radius_factor"]),
            float(row["exterior_fraction"]),
            int(row["radial_cells"]),
            int(row["angular_cells"]),
            float(row.get("omega") or args.omega),
            row.get("label") or "",
            row.get("rotation_law") or "uniform",
            float(row.get("differential_rotation_length") or 10.0),
            float(row.get("rotation_power_q") or 1.0),
         )
         if row_key not in selected_keys \
               or row["status"] != "PASS" \
               or not math.isclose(
                  float(row.get("omega") or args.omega),
                  reference_omega,
                  rel_tol=0.0,
                  abs_tol=1e-15,
               ):
            continue
         row["reference_artifact"] = repo_relative(reference_artifact)
         for diagnostic in (
            "adm_mass",
            "baryonic_mass",
            "angular_momentum",
            "dimensionless_spin",
         ):
            reference_value = float(reference[diagnostic])
            row[f"{diagnostic}_relative_difference"] = abs(
               float(row[diagnostic]) - reference_value
            ) / abs(reference_value)
         for diagnostic in (
            "mass_shedding_indicator",
            "equatorial_log_enthalpy_gradient",
            "polar_log_enthalpy_gradient",
         ):
            row[f"{diagnostic}_absolute_difference"] = abs(
               float(row[diagnostic]) - float(reference[diagnostic])
            )
   if rows:
      fields = list(dict.fromkeys(key for row in rows for key in row))
      write_tsv(RESULTS, rows, fields)
   selected_rows = [existing[key] for key in selected_keys]
   required_status = (
      "EXPECTED_REJECTION" if args.expect_surface_rejection else "PASS"
   )
   return (
      0
      if selected_rows
      and all(row["status"] == required_status for row in selected_rows)
      else 1
   )


if __name__ == "__main__":
   raise SystemExit(main())
