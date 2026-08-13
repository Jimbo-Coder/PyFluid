#!/usr/bin/env python3
"""Validate reduced-coordinate physical ID and 3D Cartesian handoff."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from validation_common import (
   ELLIPTIC_POLICY,
   ROOT,
   append_validation_history,
   build_cached_target,
   materialize_run_input,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_validation_row,
   write_tsv,
)


SOURCE_INPUT = ROOT / "pyfluid_amrex/inputs/tov_spherical_id_1d.in"
DIRECT_INPUT = ROOT / "pyfluid_amrex/inputs/tov_direct_id_3d.in"
MATERIALIZED_INPUT = ROOT / "pyfluid_amrex/inputs/tov_from_spherical_id_3d.in"
RUN_ROOT = ROOT / "runs/current/reduced_initial_data_validation"
RESULTS = RUN_ROOT / "reduced_initial_data.tsv"
FULL_RESULTS = RUN_ROOT / "reduced_initial_data_full.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
INTERPOLATION_CHECK = "pyfluid_amrex_physical_id_interpolation_check"
AXISYMMETRIC_ELLIPTIC_CHECK = (
   "pyfluid_amrex_initial_data_elliptic_check_2d"
)
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
PREFLIGHT_PATTERN = re.compile(
   r"Evolution preflight C2P/P2C: active_cells=(\d+) "
   r"max_conserved_error=([0-9.eE+-]+) "
   r"max_primitive_error=([0-9.eE+-]+)"
)


def run_id_case(
   name: str,
   dim: int,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
) -> tuple[Path, Path, float]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   artifact = run_dir / "physical_id"
   input_path = materialize_run_input(
      base_input,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.output_file": repo_relative(artifact),
         **overrides,
      },
   )
   log_path = run_dir / "run.log"
   seconds = 0.0
   if args.force or not artifact.is_dir():
      completed = run_amrex(
         dim=dim,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      seconds = completed.seconds
      if completed.returncode != 0:
         raise RuntimeError(
            f"{name} failed; see {repo_relative(log_path)}"
         )
   if not log_path.is_file() or not artifact.is_dir():
      raise RuntimeError(f"{name} lacks its log or physical-ID artifact")
   log_text = log_path.read_text()
   preflight = PREFLIGHT_PATTERN.findall(log_text)
   if not preflight:
      raise RuntimeError(f"{name} did not run the relativistic ID preflight")
   recovery = RECOVERY_PATTERN.findall(log_text)
   if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
      raise RuntimeError(f"{name} required relativistic recovery during ID")
   return run_dir, artifact, seconds


def read_level_components(path: Path) -> list[np.ndarray]:
   with h5py.File(path, "r") as handle:
      group = handle["level_0"]
      data = np.asarray(group["data:datatype=0"])
      offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
      boxes = np.asarray(group["boxes"])
      num_components = int(
         np.asarray(handle.attrs["num_components"]).reshape(-1)[0]
      )
      pieces: list[list[tuple[tuple[int, ...], np.ndarray]]] = [
         [] for _ in range(num_components)
      ]
      for box, begin, end in zip(boxes, offsets[:-1], offsets[1:]):
         key = tuple(int(value) for value in box)
         block = data[begin:end].reshape(num_components, -1)
         for component in range(num_components):
            pieces[component].append((key, block[component].copy()))
   return [
      np.concatenate([values for _, values in sorted(component)])
      for component in pieces
   ]


def read_cell_component(
   path: Path,
   index: tuple[int, int, int],
   component: int,
) -> float:
   with h5py.File(path, "r") as handle:
      group = handle["level_0"]
      data = np.asarray(group["data:datatype=0"])
      offsets = np.asarray(group["data:offsets=0"], dtype=np.int64)
      boxes = np.asarray(group["boxes"])
      num_components = int(
         np.asarray(handle.attrs["num_components"]).reshape(-1)[0]
      )
      if component < 0 or component >= num_components:
         raise RuntimeError("physical-ID component index is out of range")
      for box, begin, end in zip(boxes, offsets[:-1], offsets[1:]):
         lower = tuple(int(box[direction]) for direction in range(3))
         upper = tuple(int(box[direction + 3]) for direction in range(3))
         if not all(
            lower[direction] <= index[direction] <= upper[direction]
            for direction in range(3)
         ):
            continue
         extents = tuple(
            upper[direction] - lower[direction] + 1
            for direction in range(3)
         )
         points = int(np.prod(extents))
         local = index[0] - lower[0]
         local += extents[0] * (index[1] - lower[1])
         local += extents[0] * extents[1] * (index[2] - lower[2])
         value_index = int(begin) + component * points + local
         if value_index >= int(end):
            raise RuntimeError("physical-ID box/component layout is inconsistent")
         return float(data[value_index])
   raise RuntimeError(f"physical-ID cell {index} is not present")


def relative_component_error(
   candidate: list[np.ndarray],
   reference: list[np.ndarray],
   component: int,
) -> tuple[float, float]:
   difference = np.abs(candidate[component] - reference[component])
   scale = max(float(np.max(np.abs(reference[component]))), 1.0e-14)
   return float(np.max(difference) / scale), float(np.mean(difference) / scale)


def compare_artifacts(
   candidate: Path,
   reference: Path,
) -> dict[str, float]:
   fluid = read_level_components(candidate / "physical_fluid.h5")
   fluid_reference = read_level_components(reference / "physical_fluid.h5")
   adm = read_level_components(candidate / "adm_geometry.h5")
   adm_reference = read_level_components(reference / "adm_geometry.h5")
   rho_max, rho_l1 = relative_component_error(fluid, fluid_reference, 0)
   pressure_max, pressure_l1 = relative_component_error(
      fluid, fluid_reference, 4
   )
   gamma_errors = [
      relative_component_error(adm, adm_reference, component)[0]
      for component in (0, 3, 5)
   ]
   lapse_max, lapse_l1 = relative_component_error(
      adm, adm_reference, 12
   )
   inactive_absolute = max(
      float(np.max(np.abs(adm[component] - adm_reference[component])))
      for component in (1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15)
   )
   return {
      "rho_max": rho_max,
      "rho_l1": rho_l1,
      "pressure_max": pressure_max,
      "pressure_l1": pressure_l1,
      "gamma_max": max(gamma_errors),
      "lapse_max": lapse_max,
      "lapse_l1": lapse_l1,
      "inactive_absolute": inactive_absolute,
   }


def require_portable_source(artifact: Path, dimension: int) -> None:
   metadata = (artifact / "Header.meta").read_text()
   required = (
      "# schema_version=2",
      f"# amrex_spacedim={dimension}",
      "# component_basis=mesh_orthonormal",
      "# has_exact_caches=0",
   )
   if any(item not in metadata for item in required):
      raise RuntimeError("reduced source artifact lacks portable chart metadata")
   expected = {"Header", "Header.meta", "physical_fluid.h5", "adm_geometry.h5"}
   actual = {path.name for path in artifact.iterdir() if path.is_file()}
   if actual != expected:
      raise RuntimeError(
         "reduced source artifact should contain only portable physical fields: "
         f"expected={sorted(expected)} actual={sorted(actual)}"
      )


def run_evolution(
   source_artifact: Path,
   args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
   name = "tov_loaded_evolution"
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      MATERIALIZED_INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.solve_only": "0",
         "pyfluid.max_step": "3",
         "pyfluid.ng": "2",
         "pyfluid.recon": "pwl",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.scalar_int": "1",
      },
      remove_parameters=("initial_data.output_file",),
   )
   log_path = run_dir / "run.log"
   if args.force or not (run_dir / "scalars.tsv").is_file():
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(
            f"{name} failed; see {repo_relative(log_path)}"
         )
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=ELLIPTIC_POLICY,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="initial_data",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"loaded TOV evolution failed: {scalar['reason']}")
   rows = read_scalar_rows(run_dir)
   if len(rows) != 4:
      raise RuntimeError("loaded TOV evolution did not retain all four scalar rows")
   initial_mass = float(rows[0]["mass"])
   final_mass = float(rows[-1]["mass"])
   mass_drift = abs(final_mass - initial_mass) / initial_mass
   match = RECOVERY_PATTERN.search(log_path.read_text())
   if match is None or int(match.group(1)) != 0 or int(match.group(2)) != 0:
      raise RuntimeError("loaded TOV evolution required relativistic recovery")
   return {
      "kind": "initial_data",
      "case": name,
      "status": "PASS",
      "metric": "three_step_mass_drift",
      "value": mass_drift,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_errors(errors: dict[int, dict[str, float]], level: str) -> None:
   final = errors[max(errors)]
   limits = {
      "rho_max": 8.0e-5 if level == "standard" else 3.0e-5,
      "pressure_max": 8.0e-5 if level == "standard" else 3.0e-5,
      "gamma_max": 1.0e-5 if level == "standard" else 4.0e-6,
      "lapse_max": 6.0e-6 if level == "standard" else 2.0e-6,
      "inactive_absolute": 2.0e-12,
   }
   failed = {
      key: (final[key], limit)
      for key, limit in limits.items()
      if final[key] > limit
   }
   if failed:
      raise RuntimeError(f"reduced TOV handoff exceeded error limits: {failed}")
   if level == "full":
      resolutions = sorted(errors)
      for key in ("rho_max", "pressure_max", "gamma_max", "lapse_max"):
         values = [errors[resolution][key] for resolution in resolutions]
         if not all(right < left for left, right in zip(values, values[1:])):
            raise RuntimeError(
               f"reduced TOV {key} did not decrease with source resolution: "
               f"{values}"
            )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("standard", "full"), default="standard")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   interpolation_check = build_cached_target(INTERPOLATION_CHECK)
   interpolation_result = run_mpi(
      interpolation_check,
      ranks=1,
      timeout=args.timeout,
   )
   (RUN_ROOT / "interpolation_check.log").write_text(
      interpolation_result.stdout
   )
   if interpolation_result.returncode != 0:
      raise RuntimeError("portable physical-ID basis/reflection check failed")

   elliptic_check = build_cached_target(AXISYMMETRIC_ELLIPTIC_CHECK)
   elliptic_result = run_mpi(
      elliptic_check,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   (RUN_ROOT / "axisymmetric_elliptic_check.log").write_text(
      elliptic_result.stdout
   )
   if (
      elliptic_result.returncode != 0
      or "axisymmetric_spherical_poisson" not in elliptic_result.stdout
      or "axisymmetric_spherical_toroidal_vector_poisson"
         not in elliptic_result.stdout
   ):
      raise RuntimeError("axisymmetric spherical elliptic check failed")

   resolutions = (512,) if args.level == "standard" else (256, 512, 1024)
   source_artifacts: dict[int, Path] = {}
   run_seconds: dict[int, float] = {}
   for resolution in resolutions:
      _, artifact, seconds = run_id_case(
         f"source_n{resolution}",
         1,
         SOURCE_INPUT,
         {
            "amr.n_cell": str(resolution),
            "amr.max_grid_size": str(min(128, resolution)),
         },
         args,
      )
      require_portable_source(artifact, 1)
      source_artifacts[resolution] = artifact
      run_seconds[resolution] = seconds

   _, direct_artifact, _ = run_id_case(
      "direct_3d",
      3,
      DIRECT_INPUT,
      {},
      args,
   )
   errors: dict[int, dict[str, float]] = {}
   rows: list[dict[str, object]] = [
      {
         "kind": "initial_data",
         "case": "portable_id_basis_and_equatorial_reflection",
         "status": "PASS",
         "metric": "focused_cpp_check",
         "value": 0.0,
         "run_dir": repo_relative(RUN_ROOT),
      },
      {
         "kind": "initial_data",
         "case": "axisymmetric_spherical_scalar_and_toroidal_elliptic",
         "status": "PASS",
         "metric": "focused_cpp_check",
         "value": 0.0,
         "run_dir": repo_relative(RUN_ROOT),
      },
   ]
   for resolution in resolutions:
      _, materialized_artifact, seconds = run_id_case(
         f"materialized_n{resolution}_3d",
         3,
         MATERIALIZED_INPUT,
         {
            "initial_data.input_file": repo_relative(
               source_artifacts[resolution]
            ),
         },
         args,
      )
      error = compare_artifacts(materialized_artifact, direct_artifact)
      errors[resolution] = error
      rows.append({
         "kind": "initial_data",
         "case": f"tov_spherical_n{resolution}_to_cartesian_3d",
         "status": "PASS",
         "metric": "maximum_normalized_physical_error",
         "value": max(
            error["rho_max"],
            error["pressure_max"],
            error["gamma_max"],
            error["lapse_max"],
         ),
         "rho_max": error["rho_max"],
         "pressure_max": error["pressure_max"],
         "gamma_max": error["gamma_max"],
         "lapse_max": error["lapse_max"],
         "source_seconds": run_seconds[resolution],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_artifact.parent),
      })
   validate_errors(errors, args.level)

   if args.level == "full":
      finest_source = source_artifacts[max(resolutions)]
      large_target = {
         "geometry.prob_lo": "-40.0 -40.0 -40.0",
         "geometry.prob_hi": "40.0 40.0 40.0",
         "amr.n_cell": "24 24 24",
         "amr.max_grid_size": "24",
      }
      _, direct_large_artifact, _ = run_id_case(
         "direct_large_domain_3d",
         3,
         DIRECT_INPUT,
         large_target,
         args,
      )
      _, materialized_large_artifact, seconds = run_id_case(
         "materialized_large_domain_3d",
         3,
         MATERIALIZED_INPUT,
         {
            **large_target,
            "initial_data.input_file": repo_relative(finest_source),
         },
         args,
      )
      large_error = compare_artifacts(
         materialized_large_artifact, direct_large_artifact
      )
      large_limits = {
         "rho_max": 3.0e-5,
         "pressure_max": 3.0e-5,
         "gamma_max": 4.0e-6,
         "lapse_max": 2.0e-6,
         "inactive_absolute": 2.0e-12,
      }
      large_failed = {
         key: (large_error[key], limit)
         for key, limit in large_limits.items()
         if large_error[key] > limit
      }
      if large_failed:
         raise RuntimeError(
            "large-domain TOV handoff exceeded error limits: "
            f"{large_failed}"
         )
      rows.append({
         "kind": "initial_data",
         "case": "tov_spherical_n1024_to_larger_cartesian_domain",
         "status": "PASS",
         "metric": "maximum_normalized_physical_error",
         "value": max(
            large_error["rho_max"],
            large_error["pressure_max"],
            large_error["gamma_max"],
            large_error["lapse_max"],
         ),
         "rho_max": large_error["rho_max"],
         "pressure_max": large_error["pressure_max"],
         "gamma_max": large_error["gamma_max"],
         "lapse_max": large_error["lapse_max"],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_large_artifact.parent),
      })

      # The odd target has a cell exactly at the origin.  It distinguishes
      # regular radial interpolation from true outer-domain continuation.
      origin_target = {
         "geometry.prob_lo": "-15.5 -15.5 -15.5",
         "geometry.prob_hi": "15.5 15.5 15.5",
         "amr.n_cell": "31 31 31",
         "amr.max_grid_size": "31",
         "amr.blocking_factor": "1",
      }
      _, direct_origin_artifact, _ = run_id_case(
         "direct_origin_cell_3d",
         3,
         DIRECT_INPUT,
         origin_target,
         args,
      )
      _, materialized_origin_artifact, seconds = run_id_case(
         "materialized_origin_cell_3d",
         3,
         MATERIALIZED_INPUT,
         {
            **origin_target,
            "initial_data.input_file": repo_relative(finest_source),
         },
         args,
      )
      origin_error = compare_artifacts(
         materialized_origin_artifact, direct_origin_artifact
      )
      origin_index = (15, 15, 15)
      direct_central_density = read_cell_component(
         direct_origin_artifact / "physical_fluid.h5", origin_index, 0
      )
      materialized_central_density = read_cell_component(
         materialized_origin_artifact / "physical_fluid.h5", origin_index, 0
      )
      central_density_error = abs(
         materialized_central_density - direct_central_density
      ) / direct_central_density
      if central_density_error > 2.0e-5:
         raise RuntimeError(
            "regular radial-origin materialization was treated as exterior: "
            f"central_density_error={central_density_error}"
         )
      rows.append({
         "kind": "initial_data",
         "case": "tov_spherical_regular_origin_to_cartesian_cell",
         "status": "PASS",
         "metric": "central_cell_normalized_density_error",
         "value": central_density_error,
         "rho_max": origin_error["rho_max"],
         "pressure_max": origin_error["pressure_max"],
         "gamma_max": origin_error["gamma_max"],
         "lapse_max": origin_error["lapse_max"],
         "materialization_seconds": seconds,
         "run_dir": repo_relative(materialized_origin_artifact.parent),
      })

   evolution_row, scalar_row = run_evolution(
      source_artifacts[max(resolutions)], args
   )
   rows.append(evolution_row)
   result_fields = [
      "kind",
      "case",
      "status",
      "metric",
      "value",
      "rho_max",
      "pressure_max",
      "gamma_max",
      "lapse_max",
      "source_seconds",
      "materialization_seconds",
      "run_dir",
   ]
   results_path = RESULTS if args.level == "standard" else FULL_RESULTS
   write_tsv(results_path, rows, result_fields)
   append_validation_history(HISTORY, [*rows, scalar_row])
   print(f"wrote {repo_relative(results_path)}")
   for resolution in resolutions:
      error = errors[resolution]
      print(
         f"N={resolution}: rho={error['rho_max']:.6e} "
         f"pressure={error['pressure_max']:.6e} "
         f"gamma={error['gamma_max']:.6e} "
         f"lapse={error['lapse_max']:.6e}"
      )
   print("PASS reduced-coordinate TOV physical-ID handoff")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
