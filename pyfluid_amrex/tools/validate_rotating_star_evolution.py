#!/usr/bin/env python3
"""Validate Cartesian evolution of a saved reduced rotating-star artifact."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from physical_id_metadata import read_physical_id_metadata
from plotfile_reader import checkpoint_path
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


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_3d.in"
RUN_ROOT = ROOT / "runs/current/rotating_star_evolution_validation"
RESULTS = RUN_ROOT / "rotating_star_evolution.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
CHECKPOINT_COMPARE = (
   ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_3d"
)
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
PREFLIGHT_PATTERN = re.compile(
   r"Evolution preflight C2P/P2C: active_cells=(\d+) "
   r"max_conserved_error=([0-9.eE+-]+) "
   r"max_primitive_error=([0-9.eE+-]+)"
)


def relative_history_change(
   rows: list[dict[str, str]], name: str,
) -> float:
   initial = float(rows[0][name])
   scale = max(abs(initial), 1.0e-30)
   return max(abs(float(row[name]) - initial) / scale for row in rows)


def relative_difference(left: float, right: float) -> float:
   return abs(left - right) / max(abs(left), abs(right), 1.0e-30)


def parameter_label(value: str) -> str:
   return (
      f"{float(value):.6g}"
      .replace(".", "p")
      .replace("-", "m")
      .replace("+", "")
   )


def method_label(value: str) -> str:
   return re.sub(r"[^A-Za-z0-9_-]+", "-", value)


def effective_checkpoint_int(args: argparse.Namespace) -> int:
   if not args.restart_check:
      return args.checkpoint_int
   return args.steps // 2


def run_case(
   artifact: Path,
   metadata: dict[str, str],
   eos_table: Path | None,
   cells: int,
   args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
   if metadata["eos"] == "compose" and eos_table is None:
      raise RuntimeError("CompOSE evolution requires a resolved EoS table")
   duration_label = (
      f"t{args.final_time:g}" if args.final_time > 0.0
      else f"s{args.steps}"
   ).replace(".", "p")
   ko_label = f"ko{args.ko_dissipation:g}".replace(".", "p")
   control_label = "_".join(
      (
         method_label(args.recon),
         method_label(args.riemann),
         method_label(args.timestep),
         f"cfl{parameter_label(str(args.cfl))}",
         f"L{args.max_level}",
         f"sub{method_label(args.subcycling)}",
         f"chk{effective_checkpoint_int(args)}",
      )
   )
   law_label = metadata["rotation_law"]
   if law_label != "uniform":
      law_label += "_a" + parameter_label(
         metadata.get("differential_rotation_length", "10")
      )
   if metadata["rotation_law"] == "q_power":
      law_label += "_q" + parameter_label(
         metadata.get("rotation_power_q", "1")
      )
   eos_name = metadata.get("eos_preset", metadata["eos"])
   if metadata["eos"] == "compose":
      eos_name = f"compose-{eos_table.name}"
   eos_label = re.sub(r"[^A-Za-z0-9_-]+", "-", eos_name)
   omega_label = parameter_label(metadata["angular_velocity"])
   name = (
      f"{args.label}_{eos_label}_{law_label}_omega{omega_label}_"
      f"n{cells}_{duration_label}_{ko_label}_{control_label}"
   )
   run_dir = RUN_ROOT / name
   log_path = run_dir / "run.log"
   cached_scalars = (run_dir / "scalars.tsv").is_file()
   artifact_changed = (
      cached_scalars
      and (
         not log_path.is_file()
         or (artifact / "Header.meta").stat().st_mtime
            > log_path.stat().st_mtime
      )
   )
   run_needed = args.force or artifact_changed or not cached_scalars
   if run_needed and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   eos_parameters = {
      "pyfluid.eos": metadata["eos"],
      "pyfluid.thermal_gamma": metadata.get("thermal_gamma", "2.0"),
   }
   if metadata["eos"] == "compose":
      eos_parameters["pyfluid.eos_table"] = repo_relative(eos_table)
   else:
      eos_parameters["pyfluid.eos_preset"] = metadata.get(
         "eos_preset", "none"
      )
   input_parameters = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "amr.n_cell": f"{cells} {cells} {cells}",
      "amr.max_level": str(args.max_level),
      "amr.max_grid_size": str(min(args.max_grid_size, cells)),
      "amr.tag_metric": args.amr_tag_metric,
      "initial_data.input_file": repo_relative(artifact),
      "initial_data.solve_only": "0",
      "initial_data.save_sequence": "0",
      "initial_data.rotation_law": metadata["rotation_law"],
      "initial_data.angular_velocity": metadata["angular_velocity"],
      "initial_data.differential_rotation_length": metadata.get(
         "differential_rotation_length", "10.0"
      ),
      "initial_data.rotation_power_q": metadata.get(
         "rotation_power_q", "1.0"
      ),
      **eos_parameters,
      "pyfluid.tov_central_density": metadata["tov_central_density"],
      "spacetime.ko_dissipation": f"{args.ko_dissipation:.17g}",
      "pyfluid.max_step": str(args.steps),
      "pyfluid.final_time": f"{args.final_time:.17g}",
      "pyfluid.ng": str(args.ghost_cells),
      "pyfluid.cfl": f"{args.cfl:.17g}",
      "pyfluid.recon": args.recon,
      "pyfluid.riemann": args.riemann,
      "pyfluid.timestep": args.timestep,
      "pyfluid.subcycling": args.subcycling,
      "pyfluid.scalar_diagnostics": "standard,global_gr",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.checkpoint_int": str(effective_checkpoint_int(args)),
      "pyfluid.physical_check_int": "1",
   }
   if args.max_level > 0:
      input_parameters["amr.ref_ratio"] = " ".join(
         "2" for _ in range(args.max_level)
      )
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      input_parameters,
      remove_parameters=("pyfluid.eos_preset",)
      if metadata["eos"] == "compose" else (),
   )
   if run_needed:
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      log_path.write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(log_path)}")
   if not log_path.is_file():
      raise RuntimeError(f"{name} lacks its run log")

   scalar_row = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=DYNAMIC_SPACETIME_POLICY,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="stellar_evolution",
   )
   rows = read_scalar_rows(run_dir)
   if args.final_time > 0.0:
      final_time = float(rows[-1]["time"]) if rows else math.nan
      valid_length = 1 < len(rows) <= args.steps + 1
      valid_endpoint = math.isclose(
         final_time, args.final_time, rel_tol=1.0e-12, abs_tol=1.0e-12
      )
   else:
      valid_length = len(rows) == args.steps + 1
      valid_endpoint = valid_length
   if not valid_length or not valid_endpoint:
      raise RuntimeError(
         f"{name} retained {len(rows)} scalar rows through "
         f"time={rows[-1]['time'] if rows else 'none'}; expected final "
         f"time={args.final_time} within {args.steps} steps"
      )
   recovery = RECOVERY_PATTERN.findall(log_path.read_text())
   if not recovery:
      raise RuntimeError(f"{name} did not report relativistic recovery totals")
   c2p_fallbacks, atmosphere_resets = (
      int(value) for value in recovery[-1]
   )
   preflights = PREFLIGHT_PATTERN.findall(log_path.read_text())
   if not preflights:
      raise RuntimeError(f"{name} did not report its evolution preflight")
   _, conserved_error, primitive_error = preflights[-1]

   mass_drift = max(abs(float(row["mass_rel_drift"])) for row in rows)
   angular_momentum_drift = relative_history_change(rows, "angular_mom3")
   maximum_density_change = max(
      abs(float(row["max_rho_rel_change"])) for row in rows
   )
   hamiltonian_l2 = max(
      abs(float(row["z4c_hamiltonian_l2"])) for row in rows
   )
   hamiltonian_linf = max(
      abs(float(row["z4c_hamiltonian_linf"])) for row in rows
   )
   momentum_linf = max(
      abs(float(row["z4c_momentum_linf"])) for row in rows
   )
   initial = rows[0]
   adm_mass_surface = float(initial["adm_mass_surface"])
   adm_mass_volume = float(initial["adm_mass_volume"])
   komar_mass_surface = float(initial["komar_mass_surface"])
   komar_mass_volume = float(initial["komar_mass_volume"])
   matter_angular_momentum = float(initial["angular_mom3"])
   adm_angular_momentum_surface = float(
      initial["adm_angular_momentum3_surface"]
   )
   adm_angular_momentum_volume = float(
      initial["adm_angular_momentum3_volume"]
   )
   komar_surface_volume_error = relative_difference(
      komar_mass_surface, komar_mass_volume
   )
   adm_surface_volume_error = relative_difference(
      adm_mass_surface, adm_mass_volume
   )
   adm_komar_surface_error = relative_difference(
      adm_mass_surface, komar_mass_surface
   )
   matter_adm_volume_angular_momentum_error = relative_difference(
      matter_angular_momentum, adm_angular_momentum_volume
   )
   matter_adm_surface_angular_momentum_error = relative_difference(
      matter_angular_momentum, adm_angular_momentum_surface
   )
   finite_metrics = (
      mass_drift,
      angular_momentum_drift,
      maximum_density_change,
      hamiltonian_l2,
      hamiltonian_linf,
      momentum_linf,
      komar_surface_volume_error,
      adm_surface_volume_error,
      adm_komar_surface_error,
      matter_adm_volume_angular_momentum_error,
      matter_adm_surface_angular_momentum_error,
   )
   status = "PASS"
   reasons: list[str] = []
   if scalar_row["status"] != "PASS":
      reasons.append(str(scalar_row["reason"]))
   if not all(math.isfinite(value) for value in finite_metrics):
      reasons.append("a tracked evolution metric is nonfinite")
   if c2p_fallbacks != 0 or atmosphere_resets != 0:
      reasons.append(
         f"recovery events={c2p_fallbacks}/{atmosphere_resets}"
      )
   if float(conserved_error) > 1.0e-10 or float(primitive_error) > 1.0e-10:
      reasons.append(
         "materialization preflight errors exceed 1e-10: "
         f"{conserved_error}/{primitive_error}"
      )
   if komar_surface_volume_error > 2.0e-2:
      reasons.append(
         "initial Komar surface/volume disagreement exceeds 2%: "
         f"{komar_surface_volume_error:.3e}"
      )
   if matter_adm_volume_angular_momentum_error > 2.0e-2 \
         or matter_adm_surface_angular_momentum_error > 3.0e-2:
      reasons.append(
         "initial matter/ADM angular momentum disagreement is too large: "
         f"{matter_adm_volume_angular_momentum_error:.3e}/"
         f"{matter_adm_surface_angular_momentum_error:.3e}"
      )
   if reasons:
      status = "FAIL"
   result: dict[str, object] = {
      "kind": "stellar_evolution",
      "case": name,
      "status": status,
      "reason": "; ".join(reasons),
      "cells_per_axis": cells,
      "steps": len(rows) - 1,
      "final_time": float(rows[-1]["time"]),
      "step_cap": args.steps,
      "omega": float(metadata["angular_velocity"]),
      "rotation_law": metadata["rotation_law"],
      "eos": metadata["eos"],
      "eos_table_fingerprint": metadata.get("eos_table_fingerprint", ""),
      "dimensionless_spin_id": float(metadata["dimensionless_spin"]),
      "mass_shedding_indicator_id": float(
         metadata["mass_shedding_indicator"]
      ),
      "ko_dissipation": args.ko_dissipation,
      "mass_drift": mass_drift,
      "angular_momentum_drift": angular_momentum_drift,
      "maximum_density_change": maximum_density_change,
      "hamiltonian_l2": hamiltonian_l2,
      "hamiltonian_linf": hamiltonian_linf,
      "momentum_linf": momentum_linf,
      "initial_komar_surface_volume_error": komar_surface_volume_error,
      "initial_adm_surface_volume_error": adm_surface_volume_error,
      "initial_adm_komar_surface_error": adm_komar_surface_error,
      "initial_matter_adm_volume_angular_momentum_error": (
         matter_adm_volume_angular_momentum_error
      ),
      "initial_matter_adm_surface_angular_momentum_error": (
         matter_adm_surface_angular_momentum_error
      ),
      "c2p_fallbacks": c2p_fallbacks,
      "atmosphere_resets": atmosphere_resets,
      "preflight_conserved_error": float(conserved_error),
      "preflight_primitive_error": float(primitive_error),
      "run_dir": repo_relative(run_dir),
   }
   return result, scalar_row


def validate_restart(
   run_dir: Path,
   args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
   restart_step = args.steps // 2
   source_checkpoint = checkpoint_path(run_dir, restart_step)
   reference_checkpoint = checkpoint_path(run_dir, args.steps)
   if not source_checkpoint.is_dir() or not reference_checkpoint.is_dir():
      raise RuntimeError(
         "restart validation requires midpoint and final checkpoints: "
         f"{repo_relative(source_checkpoint)}, "
         f"{repo_relative(reference_checkpoint)}"
      )
   restart_dir = run_dir.with_name(run_dir.name + "_restart")
   if restart_dir.exists():
      shutil.rmtree(restart_dir)
   restart_dir.mkdir(parents=True)
   restart_name = restart_dir.name
   restart_input = materialize_run_input(
      run_dir / "run.in",
      restart_dir / "input.in",
      {
         "pyfluid.run_name": restart_name,
         "pyfluid.output_dir": repo_relative(restart_dir),
         "pyfluid.restart_file": repo_relative(source_checkpoint),
         "pyfluid.plot_int": "-1",
      },
      remove_parameters=(
         "initial_data.input_file",
         "initial_data.output_file",
         "initial_data.sequence_output_dir",
      ),
   )
   completed = run_amrex(
      dim=3,
      input_path=restart_input,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   log_path = restart_dir / "run.log"
   log_path.write_text(completed.stdout)
   if completed.returncode != 0:
      raise RuntimeError(
         f"{restart_name} failed; see {repo_relative(log_path)}"
      )
   restart_checkpoint = checkpoint_path(restart_dir, args.steps)
   if not restart_checkpoint.is_dir():
      raise RuntimeError(
         f"{restart_name} did not write its final checkpoint"
      )
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [reference_checkpoint, restart_checkpoint, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)
   summary = next(
      (
         line for line in compared.stdout.splitlines()
         if line.startswith("checkpoint max_dt_diff=")
      ),
      compared.stdout.strip(),
   )
   recovery = RECOVERY_PATTERN.findall(completed.stdout)
   if not recovery:
      raise RuntimeError(
         f"{restart_name} did not report relativistic recovery totals"
      )
   c2p_fallbacks, atmosphere_resets = (
      int(value) for value in recovery[-1]
   )
   if c2p_fallbacks != 0 or atmosphere_resets != 0:
      raise RuntimeError(
         f"{restart_name} used recovery/reset events: "
         f"{c2p_fallbacks}/{atmosphere_resets}"
      )
   scalar_row = scalar_validation_row(
      restart_dir,
      case=restart_name,
      input_file=restart_dir / "run.in",
      checks=DYNAMIC_SPACETIME_POLICY,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="restart",
   )
   result: dict[str, object] = {
      "kind": "restart",
      "case": restart_name,
      "status": scalar_row["status"],
      "reason": scalar_row["reason"],
      "metric": "checkpoint_compare",
      "value": summary,
      "restart_step": restart_step,
      "final_step": args.steps,
      "tolerance": 1.0e-12,
      "run_dir": repo_relative(restart_dir),
   }
   return result, scalar_row


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--input-artifact", type=Path, required=True)
   parser.add_argument(
      "--eos-table",
      type=Path,
      help="override the table location recorded by a CompOSE artifact",
   )
   parser.add_argument("--cells", type=int, nargs="+", default=(32, 48))
   parser.add_argument("--steps", type=int, default=8)
   parser.add_argument("--final-time", type=float, default=-1.0)
   parser.add_argument("--ko-dissipation", type=float, default=0.02)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--recon", default="mp5")
   parser.add_argument("--riemann", default="hll")
   parser.add_argument("--timestep", default="ssprk3")
   parser.add_argument("--subcycling", default="on")
   parser.add_argument("--max-level", type=int, default=0)
   parser.add_argument("--max-grid-size", type=int, default=16)
   parser.add_argument("--amr-tag-metric", default="lohner")
   parser.add_argument("--ghost-cells", type=int, default=3)
   parser.add_argument("--checkpoint-int", type=int, default=-1)
   parser.add_argument(
      "--restart-check",
      action="store_true",
      help="checkpoint halfway, restart, and compare final checkpoints",
   )
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--label", default="fast_rns")
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   artifact = args.input_artifact.resolve()
   if not (artifact / "Header.meta").is_file():
      raise SystemExit("--input-artifact must name a physical-ID artifact")
   if args.steps < 1 or args.ranks < 1 or any(cells < 16 for cells in args.cells):
      raise SystemExit("steps/ranks must be positive and cells at least 16")
   if args.max_level < 0 or args.max_grid_size < 1 or args.ghost_cells < 1:
      raise SystemExit("AMR level must be nonnegative; grid/ghost sizes positive")
   if not math.isfinite(args.cfl) or args.cfl <= 0.0:
      raise SystemExit("--cfl must be finite and positive")
   if args.checkpoint_int < -1:
      raise SystemExit("--checkpoint-int must be -1, 0, or positive")
   if args.restart_check and (
      args.final_time != -1.0 or args.steps < 2 or args.steps % 2 != 0
   ):
      raise SystemExit(
         "--restart-check requires --final-time=-1 and an even --steps >= 2"
      )
   if args.restart_check and args.checkpoint_int not in (-1, args.steps // 2):
      raise SystemExit(
         "--restart-check owns the midpoint checkpoint cadence; omit "
         "--checkpoint-int or set it to steps/2"
      )
   if not math.isfinite(args.final_time) or (
      args.final_time <= 0.0 and args.final_time != -1.0
   ):
      raise SystemExit("--final-time must be -1 or positive")
   if not math.isfinite(args.ko_dissipation) \
         or not 0.0 <= args.ko_dissipation <= 0.5:
      raise SystemExit("--ko-dissipation must be finite in [0, 0.5]")
   if re.fullmatch(r"[A-Za-z0-9_-]+", args.label) is None:
      raise SystemExit("--label may contain only letters, digits, '-' and '_'")
   metadata = read_physical_id_metadata(artifact)
   required = (
      "source_ic",
      "eos",
      "tov_central_density",
      "rotation_law",
      "angular_velocity",
      "injection_energy",
      "dimensionless_spin",
      "mass_shedding_indicator",
   )
   if metadata.get("rotation_law") in ("j_constant", "q_power"):
      required += ("differential_rotation_length",)
   if metadata.get("rotation_law") == "q_power":
      required += ("rotation_power_q",)
   if metadata.get("eos") == "compose":
      required += ("thermal_gamma", "eos_table_fingerprint")
   missing = [name for name in required if name not in metadata]
   if missing or metadata.get("source_ic") != "rotating_star":
      raise SystemExit(
         f"artifact is not complete rotating-star data; missing={missing}"
      )
   if not math.isfinite(float(metadata["injection_energy"])) \
         or float(metadata["injection_energy"]) <= 0.0:
      raise SystemExit("rotating-star artifact has invalid injection energy")
   eos_table: Path | None = None
   if metadata["eos"] == "compose":
      if not math.isfinite(float(metadata["thermal_gamma"])) \
            or float(metadata["thermal_gamma"]) <= 1.0:
         raise SystemExit("CompOSE artifact has invalid thermal_gamma")
      if len(metadata["eos_table_fingerprint"]) != 16:
         raise SystemExit("CompOSE artifact has invalid table fingerprint")
      table_value = args.eos_table or metadata.get("eos_table_source")
      if table_value is None:
         raise SystemExit(
            "CompOSE artifact has no table source; provide --eos-table"
         )
      eos_table = Path(table_value).resolve()
      if not eos_table.is_dir():
         raise SystemExit(
            "CompOSE table is unavailable; provide an existing --eos-table"
         )

   build_cached_target("pyfluid_amrex_3d")
   if args.restart_check:
      build_cached_target(CHECKPOINT_COMPARE.name)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   results: list[dict[str, object]] = []
   scalar_rows: list[dict[str, object]] = []
   for cells in sorted(set(args.cells)):
      result, scalar_row = run_case(
         artifact, metadata, eos_table, cells, args
      )
      results.append(result)
      scalar_rows.append(scalar_row)
      print(
         f"{result['status']} {result['case']} "
         f"dM={result['mass_drift']:.3e} "
         f"dJ={result['angular_momentum_drift']:.3e} "
         f"dRhoMax={result['maximum_density_change']:.3e}"
      )
      if args.restart_check:
         restart_result, restart_scalar = validate_restart(
            ROOT / str(result["run_dir"]), args
         )
         results.append(restart_result)
         scalar_rows.append(restart_scalar)
         print(
            f"{restart_result['status']} {restart_result['case']} "
            f"{restart_result['value']}"
         )
   fields = list(dict.fromkeys(key for row in results for key in row))
   write_tsv(RESULTS, results, fields)
   append_validation_history(HISTORY, [*results, *scalar_rows])
   print(f"wrote {repo_relative(RESULTS)}")
   return 0 if all(row["status"] == "PASS" for row in results) else 1


if __name__ == "__main__":
   raise SystemExit(main())
