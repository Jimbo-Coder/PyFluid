#!/usr/bin/env python3
"""Validate native cold CompOSE thermodynamics, stellar ID, and handoff."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np

from validation_common import (
   DYNAMIC_SPACETIME_POLICY,
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


TABLE_ROOT = ROOT / "pyfluid_amrex/eos/compose"
TABLES = ("sly4", "apr", "dd2")
TOV_INPUT = ROOT / "pyfluid_amrex/inputs/tov_compose_sequence.in"
RNS_INPUT = (
   ROOT
   / "pyfluid_amrex/inputs/rotating_star_compose_sly4_xcts_spherical_id_2d.in"
)
MATERIALIZED_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_from_spherical_id_3d.in"
)
RUN_ROOT = ROOT / "runs/current/compose_eos_validation"
RESULTS = RUN_ROOT / "compose_eos.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
SOLAR_MASS_LENGTH_KM = 1.4766250380501247
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
RNS_PATTERN = re.compile(
   r"rotating star:.*?mass_shedding_indicator=([0-9.eE+-]+).*?"
   r"adm_mass=([0-9.eE+-]+).*?baryonic_mass=([0-9.eE+-]+).*?"
   r"angular_momentum=([0-9.eE+-]+).*?dimensionless_spin=([0-9.eE+-]+)"
)
RESULT_FIELDS = [
   "kind",
   "case",
   "status",
   "metric",
   "value",
   "median_radius_relative_error",
   "maximum_mass_relative_error",
   "mass_shedding_indicator",
   "dimensionless_spin",
   "seconds",
   "run_dir",
   "reason",
]


def read_metadata(artifact: Path) -> dict[str, str]:
   values: dict[str, str] = {}
   for raw_line in (artifact / "Header.meta").read_text().splitlines():
      line = raw_line.removeprefix("#").strip()
      if "=" in line:
         key, value = line.split("=", 1)
         values[key.strip()] = value.strip()
   return values


def require_clean_recovery(log_text: str, case: str) -> None:
   matches = RECOVERY_PATTERN.findall(log_text)
   if not matches or tuple(int(value) for value in matches[-1]) != (0, 0):
      raise RuntimeError(f"{case} required relativistic recovery")
   if "Evolution preflight C2P/P2C:" not in log_text:
      raise RuntimeError(f"{case} did not run the relativistic preflight")


def read_tov_sequence(path: Path) -> list[dict[str, str]]:
   with path.open() as stream:
      return list(csv.DictReader(
         (line for line in stream if not line.startswith("#")), delimiter="\t"
      ))


def mass_radius_errors(table: str, sequence_path: Path) -> dict[str, float]:
   rows = read_tov_sequence(sequence_path)
   mass = np.asarray([float(row["gravitational_mass"]) for row in rows])
   radius = SOLAR_MASS_LENGTH_KM * np.asarray(
      [float(row["areal_radius"]) for row in rows]
   )
   stable = np.asarray([row["stable_branch"] == "1" for row in rows])
   provider = np.loadtxt(TABLE_ROOT / table / "eos.mr")
   provider = provider[: int(np.argmax(provider[:, 1])) + 1]
   lower_mass = max(1.0, float(np.min(provider[:, 1])))
   upper_mass = min(float(np.max(provider[:, 1])), 0.98 * float(np.max(mass)))
   selected = stable & (mass >= lower_mass) & (mass <= upper_mass)
   if int(np.count_nonzero(selected)) < 12:
      raise RuntimeError(f"{table} has too little stable mass-radius overlap")
   reference_radius = np.interp(
      mass[selected], provider[:, 1], provider[:, 0]
   )
   radius_error = np.abs(radius[selected] - reference_radius) / reference_radius
   maximum_mass_error = abs(
      float(np.max(mass)) - float(np.max(provider[:, 1]))
   ) / float(np.max(provider[:, 1]))
   metrics = {
      "maximum_radius_relative_error": float(np.max(radius_error)),
      "median_radius_relative_error": float(np.median(radius_error)),
      "maximum_mass_relative_error": maximum_mass_error,
      "maximum_mass": float(np.max(mass)),
   }
   if metrics["maximum_radius_relative_error"] > 1.2e-2:
      raise RuntimeError(f"{table} mass-radius maximum error exceeded 1.2%")
   if metrics["median_radius_relative_error"] > 5.0e-3:
      raise RuntimeError(f"{table} mass-radius median error exceeded 0.5%")
   if maximum_mass_error > 1.5e-2:
      raise RuntimeError(f"{table} maximum mass error exceeded 1.5%")
   return metrics


def validate_tables(args: argparse.Namespace) -> list[dict[str, object]]:
   check = build_cached_target("pyfluid_amrex_tabulated_eos_check")
   completed = run_mpi(
      check,
      [*(TABLE_ROOT / table for table in TABLES)],
      ranks=1,
      timeout=args.timeout,
   )
   (RUN_ROOT / "tabulated_eos_check.log").write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError("native CompOSE thermodynamic/C2P check failed")

   executable = build_cached_target("pyfluid_amrex_tov_sequence")
   rows: list[dict[str, object]] = []
   for table in TABLES:
      run_dir = RUN_ROOT / f"tov_{table}"
      run_dir.mkdir(parents=True, exist_ok=True)
      input_path = materialize_run_input(
         TOV_INPUT,
         run_dir / "input.in",
         {
            "pyfluid.run_name": f"tov_compose_{table}",
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.eos_table": repo_relative(TABLE_ROOT / table),
            "tov_sequence.output_dir": repo_relative(run_dir),
         },
      )
      completed = run_mpi(
         executable, [input_path], ranks=1, timeout=args.timeout
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{table} TOV sequence failed")
      metrics = mass_radius_errors(table, run_dir / "custom.tsv")
      rows.append({
         "kind": "initial_data",
         "case": f"compose_{table}_cold_tov_mass_radius",
         "status": "PASS",
         "metric": "maximum_radius_relative_error",
         "value": metrics["maximum_radius_relative_error"],
         "median_radius_relative_error": metrics[
            "median_radius_relative_error"
         ],
         "maximum_mass_relative_error": metrics[
            "maximum_mass_relative_error"
         ],
         "seconds": completed.seconds,
         "run_dir": repo_relative(run_dir),
      })
   return rows


def run_checked_amrex(
   name: str,
   dim: int,
   input_path: Path,
   args: argparse.Namespace,
) -> tuple[Path, float, str]:
   run_dir = input_path.parent
   completed = run_amrex(
      dim=dim, input_path=input_path, ranks=args.ranks, timeout=args.timeout
   )
   (run_dir / "run.log").write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
   require_clean_recovery(completed.stdout, name)
   return run_dir, completed.seconds, completed.stdout


def validate_rns_handoff(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   build_cached_target("pyfluid_amrex_production")
   source_dir = RUN_ROOT / "rns_sly4_source"
   source_dir.mkdir(parents=True, exist_ok=True)
   source_artifact = source_dir / "physical_id"
   source_input = materialize_run_input(
      RNS_INPUT,
      source_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_source",
         "pyfluid.output_dir": repo_relative(source_dir),
         "initial_data.output_file": repo_relative(source_artifact),
      },
   )
   _, source_seconds, source_log = run_checked_amrex(
      "compose_sly4_rns_source", 2, source_input, args
   )
   diagnostics = RNS_PATTERN.findall(source_log)
   if not diagnostics:
      raise RuntimeError("CompOSE RNS solve did not report global diagnostics")
   shedding, adm_mass, baryonic_mass, angular_momentum, spin = (
      float(value) for value in diagnostics[-1]
   )
   if not (
      0.0 < shedding <= 1.1
      and adm_mass > 0.0
      and baryonic_mass > adm_mass
      and angular_momentum > 0.0
      and spin > 0.0
   ):
      raise RuntimeError("CompOSE RNS global diagnostics are not physical")
   source_metadata = read_metadata(source_artifact)
   fingerprint = source_metadata.get("eos_table_fingerprint", "")
   if (
      source_metadata.get("eos_backend") != "compose_cold_barotrope_v1"
      or len(fingerprint) != 16
   ):
      raise RuntimeError("CompOSE RNS artifact lacks its immutable EoS identity")

   relocated_table = RUN_ROOT / "relocated_sly4"
   shutil.copytree(TABLE_ROOT / "sly4", relocated_table, dirs_exist_ok=True)

   materialized_dir = RUN_ROOT / "rns_sly4_materialized"
   materialized_dir.mkdir(parents=True, exist_ok=True)
   materialized_artifact = materialized_dir / "physical_id"
   materialized_input = materialize_run_input(
      MATERIALIZED_INPUT,
      materialized_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_materialized",
         "pyfluid.output_dir": repo_relative(materialized_dir),
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.output_file": repo_relative(materialized_artifact),
         "initial_data.angular_velocity": "0.001",
         "pyfluid.eos": "compose",
         "pyfluid.eos_table": repo_relative(relocated_table),
         "pyfluid.thermal_gamma": "1.8",
         "pyfluid.pressure_floor": "1.0e-18",
      },
      remove_parameters=("pyfluid.eos_preset",),
   )
   _, materialized_seconds, _ = run_checked_amrex(
      "compose_sly4_rns_materialized", 3, materialized_input, args
   )
   target_metadata = read_metadata(materialized_artifact)
   if target_metadata.get("eos_table_fingerprint") != fingerprint:
      raise RuntimeError("RNS materialization changed the CompOSE identity")

   wrong_dir = RUN_ROOT / "rns_sly4_wrong_table"
   wrong_dir.mkdir(parents=True, exist_ok=True)
   wrong_input = materialize_run_input(
      materialized_input,
      wrong_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_wrong_table",
         "pyfluid.output_dir": repo_relative(wrong_dir),
         "initial_data.input_file": repo_relative(source_artifact),
         "initial_data.output_file": repo_relative(wrong_dir / "physical_id"),
         "pyfluid.eos_table": repo_relative(TABLE_ROOT / "apr"),
      },
   )
   rejected = run_amrex(
      dim=3, input_path=wrong_input, ranks=1, timeout=args.timeout
   )
   (wrong_dir / "run.log").write_text(rejected.stdout)
   if (
      rejected.returncode == 0
      or "metadata mismatch for 'eos_table_fingerprint'" not in rejected.stdout
   ):
      raise RuntimeError("physical-ID reload did not reject the wrong CompOSE table")

   evolution_dir = RUN_ROOT / "rns_sly4_evolution"
   evolution_dir.mkdir(parents=True, exist_ok=True)
   evolution_input = materialize_run_input(
      materialized_input,
      evolution_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_evolution",
         "pyfluid.output_dir": repo_relative(evolution_dir),
         "initial_data.input_file": repo_relative(materialized_artifact),
         "initial_data.solve_only": "0",
         "pyfluid.max_step": "3",
         "pyfluid.ng": "2",
         "pyfluid.recon": "pwl",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.scalar_int": "1",
         "pyfluid.checkpoint_int": "1",
      },
      remove_parameters=("initial_data.output_file",),
   )
   _, evolution_seconds, evolution_log = run_checked_amrex(
      "compose_sly4_rns_evolution", 3, evolution_input, args
   )
   scalar = scalar_validation_row(
      evolution_dir,
      case="compose_sly4_rns_evolution",
      input_file=evolution_dir / "run.in",
      checks=DYNAMIC_SPACETIME_POLICY,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="initial_data",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"CompOSE RNS evolution failed: {scalar['reason']}")
   scalar_rows = read_scalar_rows(evolution_dir)
   maximum_density_change = max(
      abs(float(row["max_rho_rel_change"])) for row in scalar_rows
   )
   if maximum_density_change > 2.0e-2:
      raise RuntimeError("three-step CompOSE RNS density changed by more than 2%")
   checkpoint = evolution_dir / "checkpoints/chk00001"
   checkpoint_metadata = read_metadata(checkpoint)
   if checkpoint_metadata.get("checkpoint_eos_table_fingerprint") != fingerprint:
      raise RuntimeError("checkpoint did not retain the CompOSE identity")

   restart_dir = RUN_ROOT / "rns_sly4_restart"
   restart_dir.mkdir(parents=True, exist_ok=True)
   restart_input = materialize_run_input(
      evolution_input,
      restart_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_restart",
         "pyfluid.output_dir": repo_relative(restart_dir),
         "pyfluid.restart_file": repo_relative(checkpoint),
         "pyfluid.max_step": "2",
         "pyfluid.checkpoint_int": "-1",
      },
      remove_parameters=("initial_data.input_file",),
   )
   _, restart_seconds, restart_log = run_checked_amrex(
      "compose_sly4_rns_restart", 3, restart_input, args
   )
   if "steps=1 final_step=2" not in restart_log:
      raise RuntimeError("same-table CompOSE checkpoint did not resume at step 1")

   wrong_restart_dir = RUN_ROOT / "rns_sly4_wrong_table_restart"
   wrong_restart_dir.mkdir(parents=True, exist_ok=True)
   wrong_restart_input = materialize_run_input(
      restart_input,
      wrong_restart_dir / "input.in",
      {
         "pyfluid.run_name": "compose_sly4_rns_wrong_table_restart",
         "pyfluid.output_dir": repo_relative(wrong_restart_dir),
         "pyfluid.eos_table": repo_relative(TABLE_ROOT / "apr"),
      },
   )
   rejected_restart = run_amrex(
      dim=3, input_path=wrong_restart_input, ranks=1, timeout=args.timeout
   )
   (wrong_restart_dir / "run.log").write_text(rejected_restart.stdout)
   if (
      rejected_restart.returncode == 0
      or "metadata mismatch for 'checkpoint_eos_table_fingerprint'"
         not in rejected_restart.stdout
   ):
      raise RuntimeError("checkpoint restart did not reject the wrong table")

   rows = [{
      "kind": "initial_data",
      "case": "compose_sly4_rns_reduced_id_handoff",
      "status": "PASS",
      "metric": "dimensionless_spin",
      "value": spin,
      "mass_shedding_indicator": shedding,
      "dimensionless_spin": spin,
      "seconds": source_seconds + materialized_seconds,
      "run_dir": repo_relative(source_dir),
   }, {
      "kind": "initial_data",
      "case": "compose_sly4_rns_evolution_restart_identity",
      "status": "PASS",
      "metric": "maximum_density_relative_change",
      "value": maximum_density_change,
      "seconds": evolution_seconds + restart_seconds,
      "run_dir": repo_relative(evolution_dir),
   }]
   return rows, [scalar]


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   rows = validate_tables(args)
   rns_rows, scalar_rows = validate_rns_handoff(args)
   rows.extend(rns_rows)
   write_tsv(RESULTS, rows, RESULT_FIELDS)
   append_validation_history(HISTORY, [*rows, *scalar_rows])
   for row in rows:
      print(f"PASS {row['case']}: {row['metric']}={float(row['value']):.6e}")
   print(f"wrote {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
