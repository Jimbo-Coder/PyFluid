#!/usr/bin/env python3
"""Validate fixed curved-spacetime RHD geometry, sources, and equilibrium."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from plotfile_reader import (
   latest_checkpoint,
   latest_plotfile,
   level_count,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from validation_common import (
   ROOT,
   append_validation_history,
   build_cached_target,
   materialize_run_input,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_checks,
   scalar_validation_row,
   write_tsv,
)


INPUT = ROOT / "pyfluid_amrex/inputs/rhd_rindler_hse_1d.in"
RUN_ROOT = ROOT / "runs/current/fixed_spacetime_validation"
RESULTS = RUN_ROOT / "fixed_spacetime_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
CHECKPOINT_COMPARE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_1d"
FIXED_METRIC_POLICY = scalar_checks(
   "fixed_spacetime",
   energy=False,
   momentum=False,
)
FIXED_METRIC_MHD_POLICY = scalar_checks(
   "fixed_spacetime_mhd",
   energy=False,
   momentum=False,
)


def metric_values(
   metric: str,
   position: np.ndarray,
   parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
   if metric == "rindler":
      lapse = 1.0 + parameters["acceleration"] * (
         position - parameters["origin"]
      )
      return lapse, np.ones_like(position)
   radius = np.abs(position - parameters["origin"])
   psi = 1.0 + 0.5 * parameters["mass"] / radius
   return (1.0 - 0.5 * parameters["mass"] / radius) / psi, psi**6


def equilibrium_error(
   run_dir: Path,
   *,
   metric: str,
   prob_lo: float,
   prob_hi: float,
   rho0: float,
   pressure0: float,
   gamma: float,
   parameters: dict[str, float],
) -> tuple[float, dict[str, float]]:
   plotfile = latest_plotfile(run_dir, require=True)
   names, _ = parse_plot_header_1d(plotfile)
   state = read_composite_state_1d(plotfile, names)
   cells = state.shape[1]
   position = prob_lo + (np.arange(cells, dtype=float) + 0.5) * (
      prob_hi - prob_lo
   ) / cells
   lapse, sqrt_det_gamma = metric_values(metric, position, parameters)
   reference_position = np.asarray([0.5 * (prob_lo + prob_hi)])
   reference_lapse, _ = metric_values(metric, reference_position, parameters)
   reference_enthalpy = 1.0 + gamma * pressure0 / (
      (gamma - 1.0) * rho0
   )
   enthalpy = reference_enthalpy * reference_lapse[0] / lapse
   rho = rho0 * (
      (enthalpy - 1.0) / (reference_enthalpy - 1.0)
   ) ** (1.0 / (gamma - 1.0))
   pressure = pressure0 * (rho / rho0) ** gamma
   eps = pressure / ((gamma - 1.0) * rho)
   expected_dens = sqrt_det_gamma * rho
   expected_tau = sqrt_det_gamma * rho * eps
   dens_error = float(
      np.mean(np.abs(state[names.index("dens")] - expected_dens))
      / np.mean(np.abs(expected_dens))
   )
   tau_error = float(
      np.mean(np.abs(state[names.index("tau")] - expected_tau))
      / np.mean(np.abs(expected_tau))
   )
   momentum_error = float(
      np.mean(np.abs(state[names.index("mom1")])) / pressure0
   )
   return max(dens_error, tau_error, momentum_error), {
      "dens_l1": dens_error,
      "tau_l1": tau_error,
      "mom1_l1": momentum_error,
   }


def run_equilibrium(
   args: argparse.Namespace,
   *,
   metric: str,
   cells: int,
) -> tuple[dict[str, object], dict[str, object]]:
   name = f"{metric}_hse_n{cells}"
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   if metric == "rindler":
      prob_lo, prob_hi = -0.5, 0.5
      metric_parameters = {"acceleration": 0.1, "origin": 0.0}
      overrides = {
         "geometry.prob_lo": str(prob_lo),
         "geometry.prob_hi": str(prob_hi),
         "spacetime.fixed_metric": "rindler",
         "spacetime.acceleration1": "0.1",
         "spacetime.origin1": "0.0",
      }
   else:
      prob_lo, prob_hi = 2.0, 4.0
      metric_parameters = {"mass": 1.0, "origin": 0.0}
      overrides = {
         "geometry.prob_lo": str(prob_lo),
         "geometry.prob_hi": str(prob_hi),
         "spacetime.fixed_metric": "schwarzschild_isotropic",
         "spacetime.mass": "1.0",
         "spacetime.origin1": "0.0",
      }
   input_path = materialize_run_input(
      INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "amr.n_cell": str(cells),
         "amr.max_grid_size": str(cells),
         "pyfluid.final_time": "0.02",
         "pyfluid.plot_int": "1000000",
         **overrides,
      },
   )
   if args.force or latest_plotfile(run_dir) is None:
      completed = run_amrex(
         dim=1,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=FIXED_METRIC_POLICY,
      mhd=False,
      dim=1,
      ranks=args.ranks,
      kind="analytic",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   error, components = equilibrium_error(
      run_dir,
      metric=metric,
      prob_lo=prob_lo,
      prob_hi=prob_hi,
      rho0=1.0,
      pressure0=1.0,
      gamma=5.0 / 3.0,
      parameters=metric_parameters,
   )
   return {
      "kind": "convergence",
      "case": name,
      "status": "PASS",
      "metric": "fixed_metric_hse_l1",
      "value": error,
      "dens_l1": components["dens_l1"],
      "tau_l1": components["tau_l1"],
      "mom1_l1": components["mom1_l1"],
      "run_dir": repo_relative(run_dir),
   }, scalar


def run_multilevel_control(
   args: argparse.Namespace,
   name: str,
   *,
   subcycling: bool,
   restart_file: Path | None = None,
) -> tuple[Path, dict[str, object]]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   overrides = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "amr.n_cell": "16",
      "amr.max_level": "2",
      "amr.ref_ratio": "2 2",
      "amr.max_grid_size": "32",
      "amr.tag_metric": "none",
      "amr.initial_refine": "box",
      "amr.initial_refine_levels": "2",
      "amr.initial_refine_fixed_levels": "2",
      "amr.initial_refine_box_lo": "-0.5",
      "amr.initial_refine_box_hi": "0.5",
      "pyfluid.max_step": "100",
      "pyfluid.final_time": "0.05",
      "pyfluid.cfl": "0.2",
      "pyfluid.subcycling": "on" if subcycling else "off",
      "pyfluid.plot_int": "1000000",
      "pyfluid.checkpoint_int": "1",
      "pyfluid.scalar_int": "1",
   }
   if restart_file is not None:
      overrides["pyfluid.restart_file"] = repo_relative(restart_file)
   input_path = materialize_run_input(INPUT, run_dir / "input.in", overrides)
   if args.force or latest_checkpoint(run_dir) is None:
      completed = run_amrex(
         dim=1,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=FIXED_METRIC_POLICY,
      mhd=False,
      dim=1,
      ranks=args.ranks,
      kind="feature",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   plotfile = latest_plotfile(run_dir, require=True)
   if level_count(plotfile) != 3:
      raise RuntimeError(f"{name} did not retain the required three-level control")
   return run_dir, scalar


def validate_multilevel_controls(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   subcycled_dir, subcycled_scalar = run_multilevel_control(
      args, "rindler_l2_subcycled", subcycling=True
   )
   single_rate_dir, single_rate_scalar = run_multilevel_control(
      args, "rindler_l2_single_rate", subcycling=False
   )
   subcycled_plot = latest_plotfile(subcycled_dir, require=True)
   single_rate_plot = latest_plotfile(single_rate_dir, require=True)
   subcycled_names, subcycled_time = parse_plot_header_1d(subcycled_plot)
   single_rate_names, single_rate_time = parse_plot_header_1d(single_rate_plot)
   if subcycled_names != single_rate_names:
      raise RuntimeError("fixed-metric subcycling comparison changed state layout")
   subcycled_state = read_composite_state_1d(subcycled_plot, subcycled_names)
   single_rate_state = read_composite_state_1d(single_rate_plot, single_rate_names)
   scale = np.maximum(1.0, np.maximum(np.abs(subcycled_state), np.abs(single_rate_state)))
   difference = float(np.max(np.abs(subcycled_state - single_rate_state) / scale))
   if not math.isclose(subcycled_time, single_rate_time, rel_tol=0.0, abs_tol=1.0e-14):
      raise RuntimeError("fixed-metric subcycling runs did not finish at the same time")
   if difference > 5.0e-4:
      raise RuntimeError(
         f"fixed-metric subcycling difference is too large: {difference:.3e}"
      )

   restart_source = subcycled_dir / "checkpoints/chk00001"
   restarted_dir, restarted_scalar = run_multilevel_control(
      args,
      "rindler_l2_restart_from3",
      subcycling=True,
      restart_file=restart_source,
   )
   full_checkpoint = latest_checkpoint(subcycled_dir)
   restart_checkpoint = latest_checkpoint(restarted_dir)
   if full_checkpoint is None or restart_checkpoint is None:
      raise RuntimeError("fixed-metric restart validation did not write checkpoints")
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [full_checkpoint, restart_checkpoint, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)
   return [
      {
         "kind": "feature",
         "case": "rindler_l2_subcycling",
         "status": "PASS",
         "metric": "subcycling_max_relative_difference",
         "value": difference,
         "run_dir": repo_relative(subcycled_dir),
      },
      {
         "kind": "restart",
         "case": "rindler_l2_restart",
         "status": "PASS",
         "metric": "checkpoint_compare",
         "value": "exact_within_1e-12",
         "run_dir": repo_relative(restarted_dir),
      },
   ], [subcycled_scalar, single_rate_scalar, restarted_scalar]


def run_curved_rmhd_control(
   args: argparse.Namespace,
   *,
   name: str,
   dim: int,
   overrides: dict[str, str],
   expected_levels: int,
) -> tuple[dict[str, object], dict[str, object]]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      INPUT,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "pyfluid.physics": "rmhd",
         "pyfluid.ic": "uniform",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.03",
         "pyfluid.b3_0": "0.0",
         "pyfluid.final_time": "0.01",
         "pyfluid.cfl": "0.2",
         "pyfluid.plot_int": "1000000",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.scalar_int": "1",
         **overrides,
      },
   )
   if args.force or latest_plotfile(run_dir) is None:
      completed = run_amrex(
         dim=dim,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=FIXED_METRIC_MHD_POLICY,
      mhd=True,
      dim=dim,
      ranks=args.ranks,
      kind="feature",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   levels = level_count(latest_plotfile(run_dir, require=True))
   if levels != expected_levels:
      raise RuntimeError(f"{name} levels={levels}, expected {expected_levels}")
   return {
      "kind": "feature",
      "case": name,
      "status": "PASS",
      "metric": "curved_rmhd_levels",
      "value": levels,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_curved_rmhd(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   schwarzschild, schwarzschild_scalar = run_curved_rmhd_control(
      args,
      name="schwarzschild_rmhd_1d",
      dim=1,
      expected_levels=1,
      overrides={
         "geometry.prob_lo": "2.0",
         "geometry.prob_hi": "4.0",
         "spacetime.fixed_metric": "schwarzschild_isotropic",
         "spacetime.mass": "1.0",
         "spacetime.origin1": "0.0",
         "amr.n_cell": "32",
         "amr.max_level": "0",
         "amr.max_grid_size": "32",
         "pyfluid.ct": "off",
         "pyfluid.subcycling": "off",
      },
   )
   rindler_ct, rindler_ct_scalar = run_curved_rmhd_control(
      args,
      name="rindler_rmhd_2d_l2_ct_subcycled",
      dim=2,
      expected_levels=3,
      overrides={
         "geometry.prob_lo": "-0.5 -0.5",
         "geometry.prob_hi": "0.5 0.5",
         "geometry.is_periodic": "0 0",
         "spacetime.fixed_metric": "rindler",
         "spacetime.acceleration1": "0.05",
         "spacetime.acceleration2": "-0.03",
         "spacetime.origin1": "0.0",
         "spacetime.origin2": "0.0",
         "amr.n_cell": "8 8",
         "amr.max_level": "2",
         "amr.ref_ratio": "2 2",
         "amr.max_grid_size": "16",
         "amr.blocking_factor": "8",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "2",
         "amr.initial_refine_fixed_levels": "2",
         "amr.initial_refine_box_lo": "-0.5 -0.5",
         "amr.initial_refine_box_hi": "0.5 0.5",
         "pyfluid.ct": "on",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
      },
   )
   return [schwarzschild, rindler_ct], [
      schwarzschild_scalar,
      rindler_ct_scalar,
   ]


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser()
   parser.add_argument("--level", choices=("quick", "standard"), default="quick")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   if args.level == "standard":
      build_cached_target(CHECKPOINT_COMPARE.name)
   resolutions = (32, 64) if args.level == "quick" else (32, 64, 128)
   metrics = ("rindler",) if args.level == "quick" else (
      "rindler",
      "schwarzschild_isotropic",
   )
   results: list[dict[str, object]] = []
   scalar_rows: list[dict[str, object]] = []
   for metric in metrics:
      previous_error = math.nan
      metric_results = []
      for cells in resolutions:
         result, scalar = run_equilibrium(args, metric=metric, cells=cells)
         error = float(result["value"])
         result["order"] = "" if not math.isfinite(previous_error) else math.log(
            previous_error / error, 2.0
         )
         previous_error = error
         metric_results.append(result)
         scalar_rows.append(scalar)
      for result in metric_results[1:]:
         if float(result["order"]) < 0.8:
            raise RuntimeError(
               f"{metric} HSE failed to converge: "
               f"case={result['case']} order={result['order']}"
            )
      results.extend(metric_results)

   if args.level == "standard":
      controls, control_scalars = validate_multilevel_controls(args)
      results.extend(controls)
      scalar_rows.extend(control_scalars)
      rmhd_controls, rmhd_scalars = validate_curved_rmhd(args)
      results.extend(rmhd_controls)
      scalar_rows.extend(rmhd_scalars)

   fields = [
      "kind",
      "case",
      "status",
      "metric",
      "value",
      "order",
      "dens_l1",
      "tau_l1",
      "mom1_l1",
      "run_dir",
   ]
   write_tsv(RESULTS, results, fields)
   append_validation_history(HISTORY, scalar_rows)
   for result in results:
      print(
         f"PASS {result['case']} {result['metric']}={result['value']} "
         f"order={result.get('order', '')}"
      )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
