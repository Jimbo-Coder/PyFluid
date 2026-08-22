#!/usr/bin/env python3
"""Validate equilibrium-torus construction, coupling, persistence, and handoff."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from validation_common import (
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


TEST_FLUID_INPUT = (
   ROOT / "pyfluid_amrex/inputs/equilibrium_torus_xcts_spherical_id_2d.in"
)
COUPLED_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/equilibrium_torus_coupled_xcts_spherical_id_2d.in"
)
MATERIALIZED_INPUT = (
   ROOT / "pyfluid_amrex/inputs/equilibrium_torus_from_spherical_id_3d.in"
)
EVOLUTION_INPUT = (
   ROOT / "pyfluid_amrex/inputs/equilibrium_torus_evolution_3d.in"
)
KERR_ID_INPUT = (
   ROOT / "pyfluid_amrex/inputs/kerr_schild_equilibrium_torus_id_3d.in"
)
KERR_EVOLUTION_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/kerr_schild_equilibrium_torus_evolution_3d.in"
)
KERR_J_CONSTANT_INPUT = (
   ROOT / "pyfluid_amrex/inputs/kerr_schild_j_constant_torus_id_3d.in"
)
RNS_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_spherical_id_2d.in"
)
CARTESIAN_TOV_INPUT = (
   ROOT / "pyfluid_amrex/inputs/tov_xcts_cartesian_id_3d.in"
)
CARTESIAN_COUPLED_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/equilibrium_torus_coupled_xcts_cartesian_id_3d.in"
)
CARTESIAN_TILTED_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/tilted_torus_coupled_xcts_cartesian_id_3d.in"
)
SPHERICAL_TILTED_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/tilted_torus_from_cartesian_id_spherical_3d.in"
)
CARTESIAN_ROUNDTRIP_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/tilted_torus_from_spherical_id_cartesian_3d.in"
)
RUN_ROOT = ROOT / "runs/current/equilibrium_torus_validation"
STANDARD_RESULTS = RUN_ROOT / "equilibrium_torus.tsv"
FULL_RESULTS = RUN_ROOT / "equilibrium_torus_full.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
ELLIPTIC_CHECK = "pyfluid_amrex_initial_data_elliptic_check_2d"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TORUS_PATTERN = re.compile(
   rf"equilibrium torus: gravity=(\w+) rotation_law=(\w+) "
   rf"specific_angular_momentum=({FLOAT}) angular_velocity=({FLOAT}) "
   rf"differential_rotation_length=({FLOAT}) rotation_power_q=({FLOAT}) "
   rf"requested_inner_radius=({FLOAT}) resolved_inner_radius=({FLOAT}) "
   rf"outer_radius=({FLOAT}) density_maximum_radius=({FLOAT}) "
   rf"maximum_density=({FLOAT}) maximum_velocity=({FLOAT}) "
   rf"maximum_log_specific_enthalpy=({FLOAT}) "
   rf"baryonic_mass=({FLOAT}) angular_momentum=({FLOAT})"
)
CONSTRAINT_PATTERN = re.compile(
   rf"xcts initial data: matter_iterations=(\d+) "
   rf"relative_matter_update=({FLOAT}) coupled_iterations=(\d+) "
   rf"hamiltonian_residual=({FLOAT}) lapse_residual=({FLOAT}) "
   rf"momentum_residual=({FLOAT})"
)
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)


def read_metadata(artifact: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.removeprefix("#").strip()
      if "=" not in line:
         continue
      key, value = line.split("=", 1)
      values[key.strip()] = value.strip()
   return values


def require_clean_relativistic_run(log_text: str, name: str) -> None:
   recovery = RECOVERY_PATTERN.findall(log_text)
   if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
      raise RuntimeError(f"{name} required relativistic recovery")
   if "Evolution preflight C2P/P2C:" not in log_text:
      raise RuntimeError(f"{name} did not run the relativistic preflight")


def run_case(
   name: str,
   dim: int,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   ranks: int | None = None,
   artifact_output: bool = True,
) -> tuple[Path, Path | None, float, str]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   artifact = run_dir / "physical_id" if artifact_output else None
   parameters = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      **overrides,
   }
   if artifact is not None:
      parameters["initial_data.output_file"] = repo_relative(artifact)
   input_path = materialize_run_input(
      base_input, run_dir / "input.in", parameters
   )
   log_path = run_dir / "run.log"
   seconds = 0.0
   needs_run = args.force or not log_path.is_file()
   if artifact is not None:
      needs_run = needs_run or not artifact.is_dir()
   if needs_run:
      completed = run_amrex(
         dim=dim,
         input_path=input_path,
         ranks=ranks or args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      seconds = completed.seconds
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(log_path)}")
   log_text = log_path.read_text(errors="replace")
   require_clean_relativistic_run(log_text, name)
   return run_dir, artifact, seconds, log_text


def run_rejected_case(
   name: str,
   dim: int,
   base_input: Path,
   overrides: dict[str, str],
   expected_message: str,
   args: argparse.Namespace,
) -> tuple[Path, float]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      base_input,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "initial_data.output_file": repo_relative(run_dir / "physical_id"),
         **overrides,
      },
   )
   completed = run_amrex(
      dim=dim,
      input_path=input_path,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   log_path = run_dir / "run.log"
   log_path.write_text(completed.stdout)
   if completed.returncode == 0 or expected_message not in completed.stdout:
      raise RuntimeError(
         f"{name} did not reject with {expected_message!r}; "
         f"see {repo_relative(log_path)}"
      )
   return run_dir, completed.seconds


def torus_diagnostics(log_text: str, name: str) -> dict[str, float | str]:
   matches = TORUS_PATTERN.findall(log_text)
   if not matches:
      raise RuntimeError(f"{name} did not report equilibrium-torus diagnostics")
   values = matches[-1]
   return {
      "gravity": values[0],
      "rotation_law": values[1],
      "specific_angular_momentum": float(values[2]),
      "angular_velocity": float(values[3]),
      "differential_rotation_length": float(values[4]),
      "rotation_power_q": float(values[5]),
      "requested_inner_radius": float(values[6]),
      "resolved_inner_radius": float(values[7]),
      "outer_radius": float(values[8]),
      "density_maximum_radius": float(values[9]),
      "maximum_density": float(values[10]),
      "maximum_velocity": float(values[11]),
      "maximum_log_specific_enthalpy": float(values[12]),
      "baryonic_mass": float(values[13]),
      "angular_momentum": float(values[14]),
   }


def constraint_diagnostics(log_text: str, name: str) -> dict[str, float | int]:
   matches = CONSTRAINT_PATTERN.findall(log_text)
   if not matches:
      raise RuntimeError(f"{name} did not report XCTS diagnostics")
   values = matches[-1]
   return {
      "matter_iterations": int(values[0]),
      "relative_matter_update": float(values[1]),
      "coupled_iterations": int(values[2]),
      "hamiltonian_residual": float(values[3]),
      "lapse_residual": float(values[4]),
      "momentum_residual": float(values[5]),
   }


def validate_torus(
   diagnostics: dict[str, float | str],
   constraints: dict[str, float | int],
   *,
   gravity: str,
   massive: bool,
   rotation_law: str = "constant_l",
   minimum_mass: float | None = None,
) -> None:
   if diagnostics["gravity"] != gravity:
      raise RuntimeError(f"torus gravity={diagnostics['gravity']} != {gravity}")
   if diagnostics["rotation_law"] != rotation_law:
      raise RuntimeError(
         f"torus rotation_law={diagnostics['rotation_law']} != {rotation_law}"
      )
   if not (
      float(diagnostics["requested_inner_radius"])
      < float(diagnostics["resolved_inner_radius"])
      <= float(diagnostics["density_maximum_radius"])
      < float(diagnostics["outer_radius"])
      < 120.0
   ):
      raise RuntimeError("torus support radii are not ordered and interior")
   if not (0.0 < float(diagnostics["maximum_velocity"]) < 1.0):
      raise RuntimeError("torus velocity is not causal")
   if not (float(diagnostics["maximum_log_specific_enthalpy"]) > 0.0):
      raise RuntimeError("torus enthalpy support is empty")
   if minimum_mass is None:
      minimum_mass = 5.0e-2 if massive else 1.0e-5
   if not (float(diagnostics["baryonic_mass"]) > minimum_mass):
      raise RuntimeError(
         f"torus baryonic mass={diagnostics['baryonic_mass']} <= {minimum_mass}"
      )
   if abs(float(constraints["hamiltonian_residual"])) > 2.0e-10:
      raise RuntimeError("torus Hamiltonian residual is too large")
   if abs(float(constraints["momentum_residual"])) > 2.0e-10:
      raise RuntimeError("torus momentum residual is too large")


def torus_row(
   name: str,
   diagnostics: dict[str, float | str],
   constraints: dict[str, float | int],
   seconds: float,
   run_dir: Path,
) -> dict[str, object]:
   return {
      "kind": "initial_data",
      "case": name,
      "status": "PASS",
      "gravity": diagnostics["gravity"],
      "rotation_law": diagnostics["rotation_law"],
      "specific_angular_momentum": diagnostics["specific_angular_momentum"],
      "angular_velocity": diagnostics["angular_velocity"],
      "differential_rotation_length": diagnostics[
         "differential_rotation_length"
      ],
      "rotation_power_q": diagnostics["rotation_power_q"],
      "requested_inner_radius": diagnostics["requested_inner_radius"],
      "resolved_inner_radius": diagnostics["resolved_inner_radius"],
      "outer_radius": diagnostics["outer_radius"],
      "density_maximum_radius": diagnostics["density_maximum_radius"],
      "maximum_density": diagnostics["maximum_density"],
      "maximum_velocity": diagnostics["maximum_velocity"],
      "maximum_log_specific_enthalpy": diagnostics[
         "maximum_log_specific_enthalpy"
      ],
      "disk_baryonic_mass": diagnostics["baryonic_mass"],
      "disk_angular_momentum": diagnostics["angular_momentum"],
      **constraints,
      "seconds": seconds,
      "run_dir": repo_relative(run_dir),
   }


def require_preserved_disk_metadata(source: Path, target: Path) -> None:
   source_metadata = read_metadata(source)
   target_metadata = read_metadata(target)
   keys = (
      "disk",
      "disk_gravity",
      "disk_rotation_law",
      "disk_specific_angular_momentum",
      "disk_angular_velocity",
      "disk_differential_rotation_length",
      "disk_rotation_power_q",
      "disk_inner_radius",
      "disk_maximum_density",
      "disk_maximum_velocity",
      "disk_resolved_inner_radius",
      "disk_outer_radius",
      "disk_density_maximum_radius",
      "disk_maximum_log_specific_enthalpy",
      "disk_baryonic_mass",
      "disk_angular_momentum",
      "maximum_density",
      "maximum_velocity",
      "injection_energy",
      "equatorial_radius",
      "polar_radius",
      "equatorial_log_enthalpy_gradient",
      "polar_log_enthalpy_gradient",
      "mass_shedding_indicator",
      "adm_mass",
      "baryonic_mass",
      "angular_momentum",
      "dimensionless_spin",
      "matter_outer_radius",
   )
   mismatches = {
      key: (source_metadata.get(key), target_metadata.get(key))
      for key in keys
      if source_metadata.get(key) != target_metadata.get(key)
   }
   if mismatches:
      raise RuntimeError(f"disk metadata changed during materialization: {mismatches}")


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("standard", "full"), default="standard")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   results_path = FULL_RESULTS if args.level == "full" else STANDARD_RESULTS
   additional_scalar_rows: list[dict[str, object]] = []

   check_binary = build_cached_target(ELLIPTIC_CHECK)
   check = run_mpi(check_binary, ranks=args.ranks, timeout=args.timeout)
   (RUN_ROOT / "elliptic_check.log").write_text(check.stdout)
   if check.returncode != 0 or "constant_l_torus" not in check.stdout:
      raise RuntimeError("generated constant-l torus pointwise check failed")

   common = {
      "geometry.prob_hi": "128.0 1.5707963267948966",
      "amr.n_cell": "256 64",
      "amr.max_grid_size": "64",
      "initial_data.disk_rotation_law": "constant_l",
      "initial_data.disk_specific_angular_momentum": "5.2",
      "initial_data.disk_inner_radius": "8.0",
   }
   test_dir, test_artifact, test_seconds, test_log = run_case(
      "test_fluid_256x64",
      2,
      TEST_FLUID_INPUT,
      {**common, "initial_data.disk_gravity": "test_fluid"},
      args,
   )
   coupled_dir, coupled_artifact, coupled_seconds, coupled_log = run_case(
      "coupled_massive_256x64",
      2,
      COUPLED_INPUT,
      {**common, "initial_data.disk_gravity": "coupled"},
      args,
   )
   assert test_artifact is not None and coupled_artifact is not None
   test_torus = torus_diagnostics(test_log, "test_fluid_256x64")
   test_constraints = constraint_diagnostics(test_log, "test_fluid_256x64")
   coupled_torus = torus_diagnostics(coupled_log, "coupled_massive_256x64")
   coupled_constraints = constraint_diagnostics(
      coupled_log, "coupled_massive_256x64"
   )
   validate_torus(
      test_torus, test_constraints, gravity="test_fluid", massive=False
   )
   validate_torus(
      coupled_torus, coupled_constraints, gravity="coupled", massive=True
   )
   if int(coupled_constraints["matter_iterations"]) <= 100:
      raise RuntimeError("coupled torus did not exercise the disk fixed-point phase")

   overlap_dir, overlap_seconds = run_rejected_case(
      "test_fluid_central_overlap_rejected",
      2,
      TEST_FLUID_INPUT,
      {
         **common,
         "initial_data.disk_gravity": "test_fluid",
         "initial_data.disk_inner_radius": "4.0",
      },
      "inner radius must lie outside the central star matter support",
      args,
   )

   materialized_dir, materialized_artifact, materialized_seconds, _ = run_case(
      "coupled_massive_to_cartesian_32",
      3,
      MATERIALIZED_INPUT,
      {
         "geometry.prob_lo": "-48.0 -48.0 -48.0",
         "geometry.prob_hi": "48.0 48.0 48.0",
         "amr.n_cell": "32 32 32",
         "amr.max_grid_size": "16",
         "initial_data.input_file": repo_relative(coupled_artifact),
      },
      args,
   )
   assert materialized_artifact is not None
   require_preserved_disk_metadata(coupled_artifact, materialized_artifact)

   evolution_dir, _, evolution_seconds, _ = run_case(
      "coupled_massive_evolution_32",
      3,
      EVOLUTION_INPUT,
      {
         "geometry.prob_lo": "-48.0 -48.0 -48.0",
         "geometry.prob_hi": "48.0 48.0 48.0",
         "amr.n_cell": "32 32 32",
         "amr.max_grid_size": "16",
         "initial_data.input_file": repo_relative(materialized_artifact),
         "pyfluid.max_step": "2",
      },
      args,
      artifact_output=False,
   )
   evolution_scalar = scalar_validation_row(
      evolution_dir,
      case="coupled_massive_evolution_32",
      input_file=evolution_dir / "input.in",
      checks="dynamic_spacetime",
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="initial_data",
   )
   if evolution_scalar["status"] != "PASS":
      raise RuntimeError(
         "coupled torus evolution scalar check failed: "
         f"{evolution_scalar['reason']}"
      )
   evolution_rows = read_scalar_rows(evolution_dir)
   if int(float(evolution_rows[-1]["step"])) != 2:
      raise RuntimeError("coupled torus evolution did not complete two steps")

   analytic_constraints: dict[str, float | int] = {
      "matter_iterations": 0,
      "relative_matter_update": 0.0,
      "coupled_iterations": 0,
      "hamiltonian_residual": 0.0,
      "lapse_residual": 0.0,
      "momentum_residual": 0.0,
   }
   kerr_common = {
      "amr.n_cell": "32 32 32",
      "amr.max_grid_size": "16",
   }
   kerr_dir, kerr_artifact, kerr_seconds, kerr_log = run_case(
      "kerr_constant_l_32",
      3,
      KERR_ID_INPUT,
      kerr_common,
      args,
   )
   assert kerr_artifact is not None
   kerr_torus = torus_diagnostics(kerr_log, "kerr_constant_l_32")
   validate_torus(
      kerr_torus,
      analytic_constraints,
      gravity="test_fluid",
      massive=False,
   )

   j_dir, j_artifact, j_seconds, j_log = run_case(
      "kerr_j_constant_32",
      3,
      KERR_J_CONSTANT_INPUT,
      {},
      args,
   )
   assert j_artifact is not None
   j_torus = torus_diagnostics(j_log, "kerr_j_constant_32")
   validate_torus(
      j_torus,
      analytic_constraints,
      gravity="test_fluid",
      massive=False,
      rotation_law="j_constant",
   )

   q_dir, q_artifact, q_seconds, q_log = run_case(
      "kerr_q_power_q1_32",
      3,
      KERR_J_CONSTANT_INPUT,
      {
         "initial_data.disk_rotation_law": "q_power",
         "initial_data.disk_rotation_power_q": "1.0",
      },
      args,
   )
   assert q_artifact is not None
   q_torus = torus_diagnostics(q_log, "kerr_q_power_q1_32")
   validate_torus(
      q_torus,
      analytic_constraints,
      gravity="test_fluid",
      massive=False,
      rotation_law="q_power",
   )
   for key in (
      "resolved_inner_radius",
      "outer_radius",
      "density_maximum_radius",
      "maximum_density",
      "maximum_velocity",
      "maximum_log_specific_enthalpy",
      "baryonic_mass",
      "angular_momentum",
   ):
      if not math.isclose(
         float(j_torus[key]),
         float(q_torus[key]),
         rel_tol=2.0e-12,
         abs_tol=2.0e-13,
      ):
         raise RuntimeError(f"q=1/j-constant identity failed in {key}")

   kerr_evolution_dir, _, kerr_evolution_seconds, _ = run_case(
      "kerr_constant_l_evolution_32",
      3,
      KERR_EVOLUTION_INPUT,
      {
         **kerr_common,
         "initial_data.input_file": repo_relative(kerr_artifact),
         "pyfluid.max_step": "2",
      },
      args,
      artifact_output=False,
   )
   kerr_evolution_scalar = scalar_validation_row(
      kerr_evolution_dir,
      case="kerr_constant_l_evolution_32",
      input_file=kerr_evolution_dir / "input.in",
      checks="open",
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="initial_data",
   )
   if kerr_evolution_scalar["status"] != "PASS":
      raise RuntimeError(
         "Kerr torus evolution scalar check failed: "
         f"{kerr_evolution_scalar['reason']}"
      )
   kerr_evolution_rows = read_scalar_rows(kerr_evolution_dir)
   if int(float(kerr_evolution_rows[-1]["step"])) != 2:
      raise RuntimeError("Kerr torus evolution did not complete two steps")

   tilted_dir, tilted_seconds = run_rejected_case(
      "misaligned_kerr_torus_rejected",
      3,
      KERR_ID_INPUT,
      {
         "amr.n_cell": "16 16 16",
         "initial_data.disk_axis1": "1.0",
         "initial_data.disk_axis2": "0.0",
         "initial_data.disk_axis3": "0.0",
      },
      "tilted Kerr disks require the explicit nonequilibrium 3D owner",
      args,
   )

   rows = [
      torus_row(
         "test_fluid_256x64", test_torus, test_constraints,
         test_seconds, test_dir,
      ),
      torus_row(
         "coupled_massive_256x64", coupled_torus, coupled_constraints,
         coupled_seconds, coupled_dir,
      ),
      {
         "kind": "initial_data",
         "case": "test_fluid_central_overlap_rejected",
         "status": "PASS",
         "seconds": overlap_seconds,
         "run_dir": repo_relative(overlap_dir),
      },
      {
         "kind": "initial_data",
         "case": "coupled_massive_to_cartesian_32",
         "status": "PASS",
         "seconds": materialized_seconds,
         "run_dir": repo_relative(materialized_dir),
      },
      {
         "kind": "initial_data",
         "case": "coupled_massive_evolution_32",
         "status": "PASS",
         "seconds": evolution_seconds,
         "mass_rel_drift": evolution_scalar["mass_rel_drift"],
         "energy_rel_drift": evolution_scalar["energy_rel_drift"],
         "momentum_abs_drift": evolution_scalar["momentum_abs_drift"],
         "run_dir": repo_relative(evolution_dir),
      },
      torus_row(
         "kerr_constant_l_32",
         kerr_torus,
         analytic_constraints,
         kerr_seconds,
         kerr_dir,
      ),
      torus_row(
         "kerr_j_constant_32",
         j_torus,
         analytic_constraints,
         j_seconds,
         j_dir,
      ),
      torus_row(
         "kerr_q_power_q1_32",
         q_torus,
         analytic_constraints,
         q_seconds,
         q_dir,
      ),
      {
         "kind": "initial_data",
         "case": "kerr_constant_l_evolution_32",
         "status": "PASS",
         "seconds": kerr_evolution_seconds,
         "mass_rel_drift": kerr_evolution_scalar["mass_rel_drift"],
         "energy_rel_drift": kerr_evolution_scalar["energy_rel_drift"],
         "momentum_abs_drift": kerr_evolution_scalar[
            "momentum_abs_drift"
         ],
         "run_dir": repo_relative(kerr_evolution_dir),
      },
      {
         "kind": "initial_data",
         "case": "misaligned_kerr_torus_rejected",
         "status": "PASS",
         "seconds": tilted_seconds,
         "run_dir": repo_relative(tilted_dir),
      },
   ]

   if args.level == "full":
      rns_dir, _, rns_seconds, rns_log = run_case(
         "rotating_star_test_fluid_128x32",
         2,
         RNS_INPUT,
         {
            "geometry.prob_hi": "64.0 1.5707963267948966",
            "amr.n_cell": "128 32",
            "amr.max_grid_size": "64",
            "initial_data.disk": "equilibrium_torus",
            "initial_data.disk_gravity": "test_fluid",
            "initial_data.disk_rotation_law": "constant_l",
            "initial_data.disk_specific_angular_momentum": "5.8",
            "initial_data.disk_inner_radius": "13.0",
            "initial_data.angular_velocity": "0.01",
            "initial_data.rotation_continuation_targets": "0.005",
         },
         args,
      )
      rns_torus = torus_diagnostics(
         rns_log, "rotating_star_test_fluid_128x32"
      )
      rns_constraints = constraint_diagnostics(
         rns_log, "rotating_star_test_fluid_128x32"
      )
      validate_torus(
         rns_torus,
         rns_constraints,
         gravity="test_fluid",
         massive=False,
      )
      rows.append(torus_row(
         "rotating_star_test_fluid_128x32",
         rns_torus,
         rns_constraints,
         rns_seconds,
         rns_dir,
      ))

      coupled_rns_dir, _, coupled_rns_seconds, coupled_rns_log = run_case(
         "rotating_star_coupled_128x32",
         2,
         RNS_INPUT,
         {
            "geometry.prob_hi": "64.0 1.5707963267948966",
            "amr.n_cell": "128 32",
            "amr.max_grid_size": "64",
            "initial_data.disk": "equilibrium_torus",
            "initial_data.disk_gravity": "coupled",
            "initial_data.disk_rotation_law": "constant_l",
            "initial_data.disk_specific_angular_momentum": "5.8",
            "initial_data.disk_inner_radius": "13.0",
            "initial_data.angular_velocity": "0.01",
            "initial_data.rotation_continuation_targets": "0.005",
         },
         args,
      )
      coupled_rns_torus = torus_diagnostics(
         coupled_rns_log, "rotating_star_coupled_128x32"
      )
      coupled_rns_constraints = constraint_diagnostics(
         coupled_rns_log, "rotating_star_coupled_128x32"
      )
      validate_torus(
         coupled_rns_torus,
         coupled_rns_constraints,
         gravity="coupled",
         massive=False,
      )
      rows.append(torus_row(
         "rotating_star_coupled_128x32",
         coupled_rns_torus,
         coupled_rns_constraints,
         coupled_rns_seconds,
         coupled_rns_dir,
      ))

      (
         cartesian_tov_dir,
         cartesian_tov_artifact,
         cartesian_tov_seconds,
         cartesian_tov_log,
      ) = run_case(
            "tov_xcts_cartesian_24",
            3,
            CARTESIAN_TOV_INPUT,
            {
               "geometry.prob_lo": "-24.0 -24.0 -24.0",
               "geometry.prob_hi": "24.0 24.0 24.0",
               "amr.n_cell": "24 24 24",
            },
            args,
         )
      assert cartesian_tov_artifact is not None
      cartesian_tov_constraints = constraint_diagnostics(
         cartesian_tov_log, "tov_xcts_cartesian_24"
      )
      if (
         abs(float(cartesian_tov_constraints["hamiltonian_residual"]))
         > 2.0e-10
         or abs(float(cartesian_tov_constraints["momentum_residual"]))
         > 2.0e-10
      ):
         raise RuntimeError("3D Cartesian TOV XCTS residual is too large")
      rows.append({
         "kind": "initial_data",
         "case": "tov_xcts_cartesian_24",
         "status": "PASS",
         **cartesian_tov_constraints,
         "seconds": cartesian_tov_seconds,
         "run_dir": repo_relative(cartesian_tov_dir),
      })

      (
         reflected_tov_dir,
         reflected_tov_artifact,
         reflected_tov_seconds,
         reflected_tov_log,
      ) = run_case(
         "tov_xcts_cartesian_z_half_24x24x12",
         3,
         CARTESIAN_TOV_INPUT,
         {
            "geometry.prob_lo": "-24.0 -24.0 0.0",
            "geometry.prob_hi": "24.0 24.0 24.0",
            "amr.n_cell": "24 24 12",
            "pyfluid.bc_zlo": "symmetry",
         },
         args,
      )
      assert reflected_tov_artifact is not None
      reflected_tov_constraints = constraint_diagnostics(
         reflected_tov_log, "tov_xcts_cartesian_z_half_24x24x12"
      )
      if (
         abs(float(reflected_tov_constraints["hamiltonian_residual"]))
         > 2.0e-10
         or abs(float(reflected_tov_constraints["momentum_residual"]))
         > 2.0e-10
      ):
         raise RuntimeError("z-half 3D Cartesian TOV XCTS residual is too large")
      if not math.isclose(
         float(reflected_tov_constraints["relative_matter_update"]),
         float(cartesian_tov_constraints["relative_matter_update"]),
         rel_tol=1.0e-6,
         abs_tol=1.0e-14,
      ):
         raise RuntimeError("full/z-half Cartesian TOV fixed points differ")
      rows.append({
         "kind": "initial_data",
         "case": "tov_xcts_cartesian_z_half_24x24x12",
         "status": "PASS",
         **reflected_tov_constraints,
         "seconds": reflected_tov_seconds,
         "run_dir": repo_relative(reflected_tov_dir),
      })

      (
         cartesian_disk_dir,
         cartesian_disk_artifact,
         cartesian_disk_seconds,
         cartesian_disk_log,
      ) = run_case(
            "equilibrium_torus_coupled_xcts_cartesian_24",
            3,
            CARTESIAN_COUPLED_INPUT,
            {
               "geometry.prob_lo": "-24.0 -24.0 -24.0",
               "geometry.prob_hi": "24.0 24.0 24.0",
               "amr.n_cell": "24 24 24",
               "initial_data.input_file": repo_relative(
                  cartesian_tov_artifact
               ),
            },
            args,
         )
      assert cartesian_disk_artifact is not None
      cartesian_disk_torus = torus_diagnostics(
         cartesian_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_24",
      )
      cartesian_disk_constraints = constraint_diagnostics(
         cartesian_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_24",
      )
      validate_torus(
         cartesian_disk_torus,
         cartesian_disk_constraints,
         gravity="coupled",
         massive=False,
         minimum_mass=1.0e-12,
      )
      rows.append(torus_row(
         "equilibrium_torus_coupled_xcts_cartesian_24",
         cartesian_disk_torus,
         cartesian_disk_constraints,
         cartesian_disk_seconds,
         cartesian_disk_dir,
      ))

      (
         adaptive_disk_dir,
         adaptive_disk_artifact,
         adaptive_disk_seconds,
         adaptive_disk_log,
      ) = run_case(
         "equilibrium_torus_coupled_xcts_cartesian_amr_12",
         3,
         CARTESIAN_COUPLED_INPUT,
         {
            "geometry.prob_lo": "-24.0 -24.0 -24.0",
            "geometry.prob_hi": "24.0 24.0 24.0",
            "amr.n_cell": "12 12 12",
            "amr.max_level": "1",
            "amr.ref_ratio": "2",
            "amr.max_grid_size": "12",
            "amr.blocking_factor": "4",
            "amr.tag_metric": "none",
            "amr.initial_refine": "box",
            "amr.initial_refine_levels": "1",
            "amr.initial_refine_fixed_levels": "1",
            "amr.initial_refine_box_lo": "-18.0 -18.0 -18.0",
            "amr.initial_refine_box_hi": "18.0 18.0 18.0",
            "initial_data.input_file": repo_relative(
               cartesian_tov_artifact
            ),
         },
         args,
      )
      assert adaptive_disk_artifact is not None
      adaptive_disk_torus = torus_diagnostics(
         adaptive_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_amr_12",
      )
      adaptive_disk_constraints = constraint_diagnostics(
         adaptive_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_amr_12",
      )
      validate_torus(
         adaptive_disk_torus,
         adaptive_disk_constraints,
         gravity="coupled",
         massive=False,
         minimum_mass=1.0e-12,
      )
      if read_metadata(adaptive_disk_artifact).get("levels") != "2":
         raise RuntimeError("adaptive Cartesian torus artifact is not two-level")
      for key in (
         "resolved_inner_radius",
         "outer_radius",
         "density_maximum_radius",
         "maximum_density",
         "baryonic_mass",
         "angular_momentum",
      ):
         if not math.isclose(
            float(adaptive_disk_torus[key]),
            float(cartesian_disk_torus[key]),
            rel_tol=5.0e-7,
            abs_tol=1.0e-15,
         ):
            raise RuntimeError(f"uniform/adaptive 3D torus mismatch in {key}")
      rows.append(torus_row(
         "equilibrium_torus_coupled_xcts_cartesian_amr_12",
         adaptive_disk_torus,
         adaptive_disk_constraints,
         adaptive_disk_seconds,
         adaptive_disk_dir,
      ))

      adaptive_evolution_dir, _, adaptive_evolution_seconds, _ = run_case(
         "equilibrium_torus_coupled_xcts_cartesian_amr_to_uniform_16",
         3,
         EVOLUTION_INPUT,
         {
            "geometry.prob_lo": "-24.0 -24.0 -24.0",
            "geometry.prob_hi": "24.0 24.0 24.0",
            "amr.n_cell": "16 16 16",
            "amr.max_level": "0",
            "amr.max_grid_size": "8",
            "initial_data.input_file": repo_relative(adaptive_disk_artifact),
            "initial_data.disk_inner_radius": "10.0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
         },
         args,
         artifact_output=False,
      )
      adaptive_evolution_scalar = scalar_validation_row(
         adaptive_evolution_dir,
         case="equilibrium_torus_coupled_xcts_cartesian_amr_to_uniform_16",
         input_file=adaptive_evolution_dir / "input.in",
         checks="dynamic_spacetime",
         mhd=False,
         dim=3,
         ranks=args.ranks,
         kind="initial_data",
      )
      if adaptive_evolution_scalar["status"] != "PASS":
         raise RuntimeError(
            "adaptive Cartesian torus handoff scalar check failed: "
            f"{adaptive_evolution_scalar['reason']}"
         )
      additional_scalar_rows.append(adaptive_evolution_scalar)
      rows.append({
         "kind": "initial_data",
         "case": "equilibrium_torus_coupled_xcts_cartesian_amr_to_uniform_16",
         "status": "PASS",
         "seconds": adaptive_evolution_seconds,
         "mass_rel_drift": adaptive_evolution_scalar["mass_rel_drift"],
         "energy_rel_drift": adaptive_evolution_scalar["energy_rel_drift"],
         "momentum_abs_drift": adaptive_evolution_scalar[
            "momentum_abs_drift"
         ],
         "run_dir": repo_relative(adaptive_evolution_dir),
      })

      (
         reflected_disk_dir,
         reflected_disk_artifact,
         reflected_disk_seconds,
         reflected_disk_log,
      ) = run_case(
         "equilibrium_torus_coupled_xcts_cartesian_z_half_24x24x12",
         3,
         CARTESIAN_COUPLED_INPUT,
         {
            "geometry.prob_lo": "-24.0 -24.0 0.0",
            "geometry.prob_hi": "24.0 24.0 24.0",
            "amr.n_cell": "24 24 12",
            "initial_data.input_file": repo_relative(reflected_tov_artifact),
            "pyfluid.bc_zlo": "symmetry",
         },
         args,
      )
      assert reflected_disk_artifact is not None
      reflected_disk_torus = torus_diagnostics(
         reflected_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_z_half_24x24x12",
      )
      reflected_disk_constraints = constraint_diagnostics(
         reflected_disk_log,
         "equilibrium_torus_coupled_xcts_cartesian_z_half_24x24x12",
      )
      validate_torus(
         reflected_disk_torus,
         reflected_disk_constraints,
         gravity="coupled",
         massive=False,
         minimum_mass=1.0e-12,
      )
      for key in (
         "resolved_inner_radius",
         "outer_radius",
         "density_maximum_radius",
         "maximum_density",
         "baryonic_mass",
         "angular_momentum",
      ):
         if not math.isclose(
            float(reflected_disk_torus[key]),
            float(cartesian_disk_torus[key]),
            rel_tol=5.0e-7,
            abs_tol=1.0e-15,
         ):
            raise RuntimeError(f"full/z-half 3D torus mismatch in {key}")
      rows.append(torus_row(
         "equilibrium_torus_coupled_xcts_cartesian_z_half_24x24x12",
         reflected_disk_torus,
         reflected_disk_constraints,
         reflected_disk_seconds,
         reflected_disk_dir,
      ))

      (
         tilted_3d_dir,
         tilted_3d_artifact,
         tilted_3d_seconds,
         tilted_3d_log,
      ) = run_case(
            "tilted_torus_coupled_xcts_cartesian_24",
            3,
            CARTESIAN_TILTED_INPUT,
            {
               "geometry.prob_lo": "-24.0 -24.0 -24.0",
               "geometry.prob_hi": "24.0 24.0 24.0",
               "amr.n_cell": "24 24 24",
               "initial_data.input_file": repo_relative(
                  cartesian_tov_artifact
               ),
            },
            args,
         )
      assert tilted_3d_artifact is not None
      tilted_3d_torus = torus_diagnostics(
         tilted_3d_log, "tilted_torus_coupled_xcts_cartesian_24"
      )
      tilted_3d_constraints = constraint_diagnostics(
         tilted_3d_log, "tilted_torus_coupled_xcts_cartesian_24"
      )
      validate_torus(
         tilted_3d_torus,
         tilted_3d_constraints,
         gravity="coupled",
         massive=False,
         minimum_mass=1.0e-9,
      )
      tilted_metadata = read_metadata(tilted_3d_artifact)
      if tilted_metadata.get("disk") != "tilted_torus":
         raise RuntimeError("3D tilted-torus identity was not persisted")
      rows.append(torus_row(
         "tilted_torus_coupled_xcts_cartesian_24",
         tilted_3d_torus,
         tilted_3d_constraints,
         tilted_3d_seconds,
         tilted_3d_dir,
      ))

      spherical_dir, spherical_artifact, spherical_seconds, _ = run_case(
         "tilted_torus_to_spherical_36x24x32",
         3,
         SPHERICAL_TILTED_INPUT,
         {"initial_data.input_file": repo_relative(tilted_3d_artifact)},
         args,
      )
      assert spherical_artifact is not None
      require_preserved_disk_metadata(tilted_3d_artifact, spherical_artifact)
      rows.append({
         "kind": "initial_data",
         "case": "tilted_torus_to_spherical_36x24x32",
         "status": "PASS",
         "seconds": spherical_seconds,
         "run_dir": repo_relative(spherical_dir),
      })

      roundtrip_dir, roundtrip_artifact, roundtrip_seconds, _ = run_case(
         "tilted_torus_spherical_to_cartesian_36",
         3,
         CARTESIAN_ROUNDTRIP_INPUT,
         {"initial_data.input_file": repo_relative(spherical_artifact)},
         args,
      )
      assert roundtrip_artifact is not None
      require_preserved_disk_metadata(spherical_artifact, roundtrip_artifact)
      rows.append({
         "kind": "initial_data",
         "case": "tilted_torus_spherical_to_cartesian_36",
         "status": "PASS",
         "seconds": roundtrip_seconds,
         "run_dir": repo_relative(roundtrip_dir),
      })

      reflection_dir, reflection_seconds = run_rejected_case(
         "tilted_torus_reflection_rejected",
         3,
         CARTESIAN_TILTED_INPUT,
         {
            "geometry.prob_lo": "-24.0 -24.0 0.0",
            "geometry.prob_hi": "24.0 24.0 24.0",
            "amr.n_cell": "24 24 12",
            "initial_data.input_file": repo_relative(cartesian_tov_artifact),
            "pyfluid.bc_zlo": "symmetry",
         },
         "relative tilt requires a full 3D domain",
         args,
      )
      rows.append({
         "kind": "initial_data",
         "case": "tilted_torus_reflection_rejected",
         "status": "PASS",
         "seconds": reflection_seconds,
         "run_dir": repo_relative(reflection_dir),
      })

      curvilinear_dir, curvilinear_seconds = run_rejected_case(
         "tilted_torus_curvilinear_continuation_rejected",
         3,
         SPHERICAL_TILTED_INPUT,
         {
            "initial_data.input_file": repo_relative(spherical_artifact),
            "initial_data.angular_velocity": "0.002",
         },
         "3D curvilinear physical-ID targets may materialize a completed model",
         args,
      )
      rows.append({
         "kind": "initial_data",
         "case": "tilted_torus_curvilinear_continuation_rejected",
         "status": "PASS",
         "seconds": curvilinear_seconds,
         "run_dir": repo_relative(curvilinear_dir),
      })

      rank1_dir, rank1_artifact, rank1_seconds, rank1_log = run_case(
         "coupled_massive_256x64_rank1",
         2,
         COUPLED_INPUT,
         {**common, "initial_data.disk_gravity": "coupled"},
         args,
         ranks=1,
      )
      assert rank1_artifact is not None
      rank1_torus = torus_diagnostics(rank1_log, "coupled_massive_256x64_rank1")
      for key in (
         "maximum_density",
         "resolved_inner_radius",
         "outer_radius",
         "baryonic_mass",
         "angular_momentum",
      ):
         if not math.isclose(
            float(rank1_torus[key]),
            float(coupled_torus[key]),
            rel_tol=2.0e-12,
            abs_tol=2.0e-13,
         ):
            raise RuntimeError(f"one/three-rank torus mismatch in {key}")
      rows.append(torus_row(
         "coupled_massive_256x64_rank1",
         rank1_torus,
         constraint_diagnostics(rank1_log, "coupled_massive_256x64_rank1"),
         rank1_seconds,
         rank1_dir,
      ))

      fine_dir, _, fine_seconds, fine_log = run_case(
         "coupled_massive_512x96",
         2,
         COUPLED_INPUT,
         {
            **common,
            "amr.n_cell": "512 96",
            "initial_data.disk_gravity": "coupled",
         },
         args,
      )
      fine_torus = torus_diagnostics(fine_log, "coupled_massive_512x96")
      maximum_relative_change = max(
         abs(float(fine_torus[key]) - float(coupled_torus[key]))
         / abs(float(fine_torus[key]))
         for key in ("maximum_density", "outer_radius", "baryonic_mass")
      )
      if maximum_relative_change > 3.0e-2:
         raise RuntimeError(
            "massive torus 256x64-to-512x96 change exceeds 3 percent"
         )
      row = torus_row(
         "coupled_massive_512x96",
         fine_torus,
         constraint_diagnostics(fine_log, "coupled_massive_512x96"),
         fine_seconds,
         fine_dir,
      )
      row["resolution_relative_change"] = maximum_relative_change
      rows.append(row)

   fields = list(dict.fromkeys(key for row in rows for key in row))
   write_tsv(results_path, rows, fields)
   evolution_scalar.update({
      "suite": "equilibrium_torus",
      "step": "equilibrium_torus",
      "seconds": sum(float(row.get("seconds", 0.0)) for row in rows),
   })
   append_validation_history(HISTORY, [evolution_scalar, *additional_scalar_rows])
   print(
      "PASS equilibrium_torus: "
      f"test_mass={float(test_torus['baryonic_mass']):.6e} "
      f"coupled_mass={float(coupled_torus['baryonic_mass']):.6e} "
      f"H={float(coupled_constraints['hamiltonian_residual']):.3e} "
      f"M={float(coupled_constraints['momentum_residual']):.3e} "
      f"evolution_mass={float(evolution_scalar['mass_rel_drift']):.3e}"
   )
   print(f"wrote {repo_relative(results_path)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
