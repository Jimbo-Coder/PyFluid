#!/usr/bin/env python3
"""Compare reduced uniform-rotation XCTS IDs with a classic RNS sequence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import tomllib

from validation_common import (
   ROOT,
   materialize_run_input,
   read_tsv,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_spherical_id_2d.in"
REFERENCE_PATH = (
   ROOT / "pyfluid_amrex/reference_data/rotating_star_uniform_n1_k100.toml"
)
RUN_ROOT = ROOT / "runs/current/rotating_star_literature"


@dataclass(frozen=True)
class ReferenceModel:
   name: str
   angular_velocity: float
   adm_mass: float
   coordinate_equatorial_radius: float
   coordinate_polar_to_equatorial_ratio: float


def read_metadata(artifact: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.removeprefix("#").strip()
      if "=" in line:
         key, value = line.split("=", 1)
         values[key.strip()] = value.strip()
   return values


def relative_error(value: float, reference: float) -> float:
   return abs(value - reference) / abs(reference)


def load_reference() -> tuple[
   dict[str, object],
   dict[str, object],
   dict[str, float],
   list[float],
   list[ReferenceModel],
]:
   with REFERENCE_PATH.open("rb") as stream:
      data = tomllib.load(stream)
   models = [ReferenceModel(**row) for row in data["model"]]
   names = [model.name for model in models]
   if len(names) != len(set(names)):
      raise ValueError("rotating-star reference model names must be unique")
   omegas = [model.angular_velocity for model in models]
   if any(not math.isfinite(omega) or omega < 0.0 for omega in omegas):
      raise ValueError(
         "reference angular velocities must be finite and nonnegative"
      )
   return (
      data["reference"],
      data["eos"],
      data["validation"],
      data["continuation"]["angular_velocities"],
      models,
   )


def common_parameters(
   eos: dict[str, object],
   run_dir: Path,
   artifact: Path,
   radial_cells: int,
   angular_cells: int,
) -> dict[str, str]:
   return {
      "amr.n_cell": f"{radial_cells} {angular_cells}",
      "amr.max_grid_size": str(min(64, radial_cells)),
      "initial_data.output_file": repo_relative(artifact),
      "initial_data.matter_relative_tolerance": "1.0e-7",
      "initial_data.matter_max_iterations": "160",
      "pyfluid.eos": str(eos["kind"]),
      "pyfluid.eos_preset": str(eos["preset"]),
      "pyfluid.polytropic_k": f"{float(eos['polytropic_k']):.17g}",
      "pyfluid.polytropic_gamma": (
         f"{float(eos['polytropic_gamma']):.17g}"
      ),
      "pyfluid.thermal_gamma": f"{float(eos['thermal_gamma']):.17g}",
      "pyfluid.tov_central_density": (
         f"{float(eos['central_rest_mass_density']):.17g}"
      ),
      "pyfluid.run_name": run_dir.name,
      "pyfluid.output_dir": repo_relative(run_dir),
   }


def run_nonrotating_probe(
   eos: dict[str, object],
   validation: dict[str, float],
   radial_cells: int,
   angular_cells: int,
   ranks: int,
   timeout: int,
) -> tuple[Path, float, int]:
   run_dir = RUN_ROOT / f"bu0_probe_r{radial_cells}_t{angular_cells}"
   artifact = run_dir / "physical_id"
   run_dir.mkdir(parents=True, exist_ok=True)
   parameters = common_parameters(
      eos, run_dir, artifact, radial_cells, angular_cells
   )
   parameters.update({
      "initial_data.angular_velocity": (
         f"{validation['nonrotating_probe_angular_velocity']:.17g}"
      ),
      "initial_data.save_sequence": "0",
   })
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      parameters,
      remove_parameters=(
         "initial_data.rotation_continuation_targets",
         "initial_data.sequence_output_dir",
      ),
   )
   completed = run_amrex(
      dim=2, input_path=input_path, ranks=ranks, timeout=timeout
   )
   (run_dir / "run.log").write_text(completed.stdout)
   return artifact, completed.seconds, completed.returncode


def run_rotating_sequence(
   eos: dict[str, object],
   continuation: list[float],
   final_omega: float,
   radial_cells: int,
   angular_cells: int,
   ranks: int,
   timeout: int,
) -> tuple[Path, float, int]:
   run_dir = RUN_ROOT / f"bu_sequence_r{radial_cells}_t{angular_cells}"
   artifact = run_dir / "physical_id"
   sequence_dir = run_dir / "sequence"
   run_dir.mkdir(parents=True, exist_ok=True)
   targets = sorted({omega for omega in continuation if 0.0 < omega < final_omega})
   parameters = common_parameters(
      eos, run_dir, artifact, radial_cells, angular_cells
   )
   parameters.update({
      "initial_data.angular_velocity": f"{final_omega:.17g}",
      "initial_data.rotation_continuation_targets": " ".join(
         f"{omega:.17g}" for omega in targets
      ),
      "initial_data.save_sequence": "1",
      "initial_data.sequence_output_dir": repo_relative(sequence_dir),
   })
   input_path = materialize_run_input(BASE_INPUT, run_dir / "input.in", parameters)
   completed = run_amrex(
      dim=2, input_path=input_path, ranks=ranks, timeout=timeout
   )
   (run_dir / "run.log").write_text(completed.stdout)
   return sequence_dir, completed.seconds, completed.returncode


def row_for_model(
   model: ReferenceModel,
   metadata: dict[str, str],
   reference: dict[str, object],
   validation: dict[str, float],
   radial_cells: int,
   angular_cells: int,
   seconds: float,
   returncode: int,
   artifact: Path,
) -> dict[str, object]:
   computed_mass = float(metadata.get("adm_mass", "nan"))
   computed_equatorial_radius = float(
      metadata.get("equatorial_radius", "nan")
   )
   computed_polar_radius = float(metadata.get("polar_radius", "nan"))
   computed_polar_to_equatorial_ratio = (
      computed_polar_radius / computed_equatorial_radius
      if computed_equatorial_radius > 0.0 else math.nan
   )
   mass_error = relative_error(computed_mass, model.adm_mass)
   radius_error = relative_error(
      computed_equatorial_radius, model.coordinate_equatorial_radius
   )
   ratio_error = abs(
      computed_polar_to_equatorial_ratio
      - model.coordinate_polar_to_equatorial_ratio
   )
   finite = all(math.isfinite(value) for value in (
      computed_mass,
      computed_equatorial_radius,
      computed_polar_radius,
      computed_polar_to_equatorial_ratio,
      mass_error,
      radius_error,
      ratio_error,
   ))
   status = "PASS" if (
      returncode == 0
      and finite
      and mass_error <= validation["adm_mass_relative_tolerance"]
      and radius_error
         <= validation["coordinate_equatorial_radius_relative_tolerance"]
      and ratio_error <= validation[
         "coordinate_polar_to_equatorial_ratio_absolute_tolerance"
      ]
   ) else "FAIL"
   return {
      "kind": "initial_data_reference",
      "case": model.name,
      "status": status,
      "reference": reference["name"],
      "reference_url": reference["url"],
      "reference_notes": reference["notes"],
      "formulation": "conformally_flat_xcts",
      "source_coordinate": "spherical_r_theta",
      "reference_angular_velocity": model.angular_velocity,
      "computed_angular_velocity": float(
         metadata.get("angular_velocity", "nan")
      ),
      "reference_adm_mass": model.adm_mass,
      "computed_adm_mass": computed_mass,
      "adm_mass_relative_error": mass_error,
      "reference_coordinate_equatorial_radius": (
         model.coordinate_equatorial_radius
      ),
      "computed_coordinate_equatorial_radius": computed_equatorial_radius,
      "coordinate_equatorial_radius_relative_error": radius_error,
      "reference_coordinate_polar_to_equatorial_ratio": (
         model.coordinate_polar_to_equatorial_ratio
      ),
      "computed_coordinate_polar_to_equatorial_ratio": (
         computed_polar_to_equatorial_ratio
      ),
      "coordinate_polar_to_equatorial_ratio_absolute_error": ratio_error,
      "mass_shedding_indicator": float(
         metadata.get("mass_shedding_indicator", "nan")
      ),
      "dimensionless_spin": float(metadata.get("dimensionless_spin", "nan")),
      "radial_cells": radial_cells,
      "angular_cells": angular_cells,
      "seconds": seconds,
      "returncode": returncode,
      "artifact": repo_relative(artifact),
   }


def print_rows(rows: list[dict[str, object]], results_path: Path) -> None:
   for row in rows:
      print(
         f"{row['status']} {row['case']} "
         f"dM={float(row['adm_mass_relative_error']):.3e} "
         f"dRe={float(row['coordinate_equatorial_radius_relative_error']):.3e} "
         "dRpRe="
         f"{float(row['coordinate_polar_to_equatorial_ratio_absolute_error']):.3e}"
      )
   print(f"WROTE {repo_relative(results_path)}")


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--radial-cells", type=int, default=512)
   parser.add_argument("--angular-cells", type=int, default=64)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=600)
   parser.add_argument("--only", nargs="+", help="reference model names")
   parser.add_argument("--list-models", action="store_true")
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   reference, eos, validation, continuation, all_models = load_reference()
   if args.list_models:
      for model in all_models:
         print(model.name)
      return 0
   known = {model.name for model in all_models}
   requested = set(args.only or known)
   unknown = sorted(requested - known)
   if unknown:
      raise SystemExit(f"unknown reference model(s): {', '.join(unknown)}")
   models = [model for model in all_models if model.name in requested]
   if args.radial_cells < 64 or args.angular_cells < 16 or args.ranks < 1:
      raise SystemExit("literature check grid/rank counts are too small")

   results_path = RUN_ROOT / "comparison.tsv"
   if results_path.is_file() and not args.force:
      cached_rows = [
         dict(row) for row in read_tsv(results_path)
         if row["case"] in requested
      ]
      expected_error_field = (
         "coordinate_polar_to_equatorial_ratio_absolute_error"
      )
      cache_matches = (
         {row["case"] for row in cached_rows} == requested
         and all(expected_error_field in row for row in cached_rows)
         and all(
            int(row["radial_cells"]) == args.radial_cells
            and int(row["angular_cells"]) == args.angular_cells
            for row in cached_rows
         )
      )
      if not cache_matches:
         raise SystemExit(
            "existing literature comparison has a different selection, "
            "grid, or schema; use --force to replace it"
         )
      print_rows(cached_rows, results_path)
      return 0 if all(row["status"] == "PASS" for row in cached_rows) else 1
   if RUN_ROOT.exists() and not args.force:
      raise SystemExit(
         "incomplete literature comparison exists; use --force to replace it"
      )
   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   rows: list[dict[str, object]] = []

   nonrotating = [model for model in models if model.angular_velocity == 0.0]
   if nonrotating:
      artifact, seconds, returncode = run_nonrotating_probe(
         eos,
         validation,
         args.radial_cells,
         args.angular_cells,
         args.ranks,
         args.timeout,
      )
      metadata = read_metadata(artifact) if artifact.is_dir() else {}
      rows.extend(
         row_for_model(
            model,
            metadata,
            reference,
            validation,
            args.radial_cells,
            args.angular_cells,
            seconds,
            returncode,
            artifact,
         )
         for model in nonrotating
      )

   rotating = [model for model in models if model.angular_velocity > 0.0]
   if rotating:
      final_omega = max(model.angular_velocity for model in rotating)
      comparison_omegas = [
         model.angular_velocity
         for model in rotating
         if model.angular_velocity < final_omega
      ]
      sequence_dir, seconds, returncode = run_rotating_sequence(
         eos,
         [*continuation, *comparison_omegas],
         final_omega,
         args.radial_cells,
         args.angular_cells,
         args.ranks,
         args.timeout,
      )
      sequence_rows = read_tsv(sequence_dir / "sequence.tsv") \
         if (sequence_dir / "sequence.tsv").is_file() else []
      for model in rotating:
         matches = [
            row for row in sequence_rows
            if math.isclose(
               float(row["angular_velocity"]),
               model.angular_velocity,
               rel_tol=0.0,
               abs_tol=1.0e-14,
            )
         ]
         artifact = sequence_dir / matches[0]["artifact"] \
            if len(matches) == 1 else sequence_dir / "missing"
         metadata = read_metadata(artifact) if artifact.is_dir() else {}
         rows.append(row_for_model(
            model,
            metadata,
            reference,
            validation,
            args.radial_cells,
            args.angular_cells,
            seconds,
            returncode if len(matches) == 1 else 1,
            artifact,
         ))

   fields = list(dict.fromkeys(field for row in rows for field in row))
   write_tsv(results_path, rows, fields)
   print_rows(rows, results_path)
   return 0 if rows and all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
