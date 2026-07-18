#!/usr/bin/env python3
"""Validate the generated vacuum Z4c evolution vertical slice."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from plotfile_reader import (
   latest_checkpoint,
   level_chunks,
   level_count,
   parse_level_cell_sizes,
   parse_plot_bounds,
   parse_plot_header,
   plotfile_step,
   read_level_state,
)
from validation_common import (
   CLOSED_POLICY,
   DYNAMIC_SPACETIME_POLICY,
   ROOT,
   SMOKE_POLICY,
   append_validation_history,
   materialize_run_input,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_validation_row,
   write_tsv,
)


INPUT = ROOT / "pyfluid_amrex/inputs/z4c_gauge_wave_1d.in"
TOV_INPUT = ROOT / "pyfluid_amrex/inputs/tov_coupled_amr_3d.in"
XCTS_TOV_INPUT = ROOT / "pyfluid_amrex/inputs/tov_xcts_3d.in"
TOV_COWLING_INPUT = ROOT / "pyfluid_amrex/inputs/tov_cowling_amr_1d.in"
CTT_INPUT = ROOT / "pyfluid_amrex/inputs/rhd_ctt_density_cloud_3d.in"
RUN_ROOT = ROOT / "runs/current/z4c_validation"
RESULTS = RUN_ROOT / "z4c_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
CHECKPOINT_COMPARE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_1d"
CONSTRAINT_COLUMNS = (
   "z4c_hamiltonian_linf",
   "z4c_momentum_linf",
   "z4c_z_linf",
   "z4c_theta_linf",
   "z4c_log_conformal_determinant_linf",
   "z4c_trace_a_tilde_linf",
)


def run_dimension_case(
   name: str,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   dim: int,
   checks: object,
   mhd: bool,
   kind: str = "analytic",
) -> tuple[Path, dict[str, object]]:
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
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
         **overrides,
      },
   )
   if args.force or latest_spacetime_plotfile(run_dir) is None:
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
      checks=checks,
      mhd=mhd,
      dim=dim,
      ranks=args.ranks,
      kind=kind,
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   return run_dir, scalar


def spacetime_plotfiles(run_dir: Path) -> list[Path]:
   files = list((run_dir / "stateGF").glob("spacetime_state*.h5"))
   return sorted(files, key=plotfile_step)


def latest_spacetime_plotfile(run_dir: Path) -> Path | None:
   files = spacetime_plotfiles(run_dir)
   return files[-1] if files else None


def latest_fluid_plotfile(run_dir: Path) -> Path | None:
   files = list((run_dir / "stateGF").glob("stateGF*.h5"))
   return max(files, key=plotfile_step) if files else None


def run_case(
   name: str,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   checks=DYNAMIC_SPACETIME_POLICY,
) -> tuple[Path, dict[str, object]]:
   return run_dimension_case(
      name,
      INPUT,
      overrides,
      args,
      dim=1,
      checks=checks,
      mhd=False,
   )


def spacetime_state(run_dir: Path) -> tuple[list[str], float, np.ndarray]:
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError(f"missing spacetime output in {repo_relative(run_dir)}")
   names, dim, time = parse_plot_header(plotfile)
   if dim != 1:
      raise RuntimeError(f"expected 1D spacetime output, got {dim}D")
   return names, time, read_level_state(plotfile, names, dim, 0)


def final_constraint_max(run_dir: Path) -> float:
   final = read_scalar_rows(run_dir)[-1]
   missing = [name for name in CONSTRAINT_COLUMNS if name not in final]
   if missing:
      raise RuntimeError(f"missing Z4c scalar diagnostics: {', '.join(missing)}")
   return max(abs(float(final[name])) for name in CONSTRAINT_COLUMNS)


def minkowski_reference(names: list[str], shape: tuple[int, ...]) -> np.ndarray:
   reference = np.zeros((len(names), *shape), dtype=float)
   for name in ("chi", "gamma_tilde11", "gamma_tilde22", "gamma_tilde33", "lapse"):
      reference[names.index(name)] = 1.0
   return reference


def validate_minkowski(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   methods = ("ssprk3", "rk4") if args.level == "quick" else (
      "euler",
      "corrector",
      "midpoint",
      "ssprk3",
      "rk4",
   )
   results = []
   scalars = []
   for method in methods:
      run_dir, scalar = run_case(
         f"minkowski_{method}",
         {
            "amr.n_cell": "16",
            "amr.max_grid_size": "16",
            "spacetime.initial_data": "minkowski",
            "spacetime.lapse_gauge": "frozen",
            "spacetime.shift_gauge": "frozen",
            "pyfluid.timestep": method,
            "pyfluid.final_time": "-1",
            "pyfluid.max_step": "2",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_int": "1",
         },
         args,
         checks=CLOSED_POLICY,
      )
      names, _time, state = spacetime_state(run_dir)
      error = float(np.max(np.abs(state - minkowski_reference(names, state.shape[1:]))))
      constraint = final_constraint_max(run_dir)
      tolerance = 256.0 * np.finfo(float).eps
      if error > tolerance or constraint > tolerance:
         raise RuntimeError(
            f"{method} Minkowski preservation failed: state={error:.3e} "
            f"constraints={constraint:.3e}"
         )
      results.append({
         "kind": "analytic",
         "case": f"minkowski_{method}",
         "status": "PASS",
         "metric": "state_linf",
         "value": error,
         "constraint_linf": constraint,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   return results, scalars


def validate_initial_data_group_policies(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   run_dir, scalar = run_case(
      "initial_data_neutral_groups",
      {
         "amr.n_cell": "16",
         "amr.max_grid_size": "16",
         "spacetime.initial_data": "gauge_wave",
         "initial_data.magnetic": "none",
         "initial_data.spacetime": "flat",
         "initial_data.lapse": "one",
         "initial_data.shift": "zero",
         "pyfluid.final_time": "-1",
         "pyfluid.max_step": "1",
         "pyfluid.plot_int": "1",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.scalar_int": "1",
      },
      args,
      checks=CLOSED_POLICY,
   )
   names, _time, state = spacetime_state(run_dir)
   error = float(np.max(np.abs(state - minkowski_reference(names, state.shape[1:]))))
   tolerance = 64.0 * np.finfo(float).eps
   if error > tolerance:
      raise RuntimeError(
         f"initial-data neutral group overrides failed: state={error:.3e}"
      )
   rhd_plotfile = latest_fluid_plotfile(run_dir)
   if rhd_plotfile is None:
      raise RuntimeError("RHD initial-data policy case did not write fluid state")
   rhd_names, _rhd_dim, _rhd_time = parse_plot_header(rhd_plotfile)
   if any(name in rhd_names for name in ("b1", "b2", "b3")):
      raise RuntimeError("RHD unexpectedly allocated magnetic state")

   rmhd_dir, rmhd_scalar = run_dimension_case(
      "initial_data_zero_magnetic_rmhd",
      INPUT,
      {
         "amr.n_cell": "16",
         "amr.max_grid_size": "16",
         "spacetime.matter_coupling": "vacuum",
         "spacetime.initial_data": "minkowski",
         "spacetime.lapse_gauge": "frozen",
         "spacetime.shift_gauge": "frozen",
         "initial_data.magnetic": "zero",
         "pyfluid.physics": "rmhd",
         "pyfluid.eos": "ideal_gas",
         "pyfluid.ic": "uniform",
         "pyfluid.b1_0": "0.3",
         "pyfluid.b2_0": "-0.2",
         "pyfluid.b3_0": "0.1",
         "pyfluid.timestep": "euler",
         "pyfluid.final_time": "-1",
         "pyfluid.max_step": "1",
         "pyfluid.plot_int": "1",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.scalar_int": "1",
      },
      args,
      dim=1,
      checks=CLOSED_POLICY,
      mhd=True,
      kind="feature",
   )
   rmhd_plotfile = latest_fluid_plotfile(rmhd_dir)
   if rmhd_plotfile is None:
      raise RuntimeError("RMHD initial-data policy case did not write fluid state")
   rmhd_names, rmhd_dim, _rmhd_time = parse_plot_header(rmhd_plotfile)
   rmhd_state = read_level_state(rmhd_plotfile, rmhd_names, rmhd_dim, 0)
   magnetic_error = max(
      float(np.max(np.abs(rmhd_state[rmhd_names.index(name)])))
      for name in ("b1", "b2", "b3")
   )
   if magnetic_error != 0.0:
      raise RuntimeError(
         f"RMHD zero-magnetic initial-data policy failed: {magnetic_error:.3e}"
      )
   return [{
      "kind": "feature",
      "case": "initial_data_neutral_groups",
      "status": "PASS",
      "metric": "state_linf",
      "value": error,
      "constraint_linf": final_constraint_max(run_dir),
      "run_dir": repo_relative(run_dir),
   }, {
      "kind": "feature",
      "case": "initial_data_zero_magnetic_rmhd",
      "status": "PASS",
      "metric": "magnetic_linf",
      "value": magnetic_error,
      "constraint_linf": final_constraint_max(rmhd_dir),
      "run_dir": repo_relative(rmhd_dir),
   }], [scalar, rmhd_scalar]


def gauge_wave_lapse_error(run_dir: Path, amplitude: float, wave_number: float) -> float:
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError(f"missing spacetime output in {repo_relative(run_dir)}")
   names, dim, time = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, dim, 0)
   lower, _upper = parse_plot_bounds(plotfile)
   spacing = parse_level_cell_sizes(plotfile)[0][0]
   x = lower[0] + np.arange(state.shape[1]) * spacing
   phase = 2.0 * math.pi * wave_number * (x - time)
   exact = np.sqrt(1.0 - amplitude * np.sin(phase))
   return float(np.sqrt(np.mean((state[names.index("lapse")] - exact) ** 2)))


def gauge_wave_lapse_level_linf(
   run_dir: Path,
   amplitude: float,
   wave_number: float,
   level: int,
) -> float:
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError(f"missing spacetime output in {repo_relative(run_dir)}")
   names, dim, time = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, dim, level)
   lower, _upper = parse_plot_bounds(plotfile)
   spacing = parse_level_cell_sizes(plotfile)[level][0]
   x = lower[0] + np.arange(state.shape[1]) * spacing
   phase = 2.0 * math.pi * wave_number * (x - time)
   exact = np.sqrt(1.0 - amplitude * np.sin(phase))
   error = np.abs(state[names.index("lapse")] - exact)
   return float(np.nanmax(error))


def validate_gauge_wave_convergence(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   resolutions = (32, 64) if args.level == "quick" else (32, 64, 128)
   amplitude = 1.0e-2
   results = []
   scalars = []
   previous_error = None
   for cells in resolutions:
      run_dir, scalar = run_case(
         f"gauge_wave_n{cells}",
         {
            "amr.n_cell": str(cells),
            "amr.max_grid_size": str(min(cells, 64)),
            "spacetime.initial_data": "gauge_wave",
            "spacetime.lapse_gauge": "harmonic",
            "spacetime.amplitude": str(amplitude),
            "spacetime.wave_number": "1.0",
            "spacetime.ko_dissipation": "0.0",
            "pyfluid.timestep": "rk4",
            "pyfluid.cfl": "0.25",
            "pyfluid.final_time": "0.2",
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "1000000",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_int": "1000000",
         },
         args,
      )
      error = gauge_wave_lapse_error(run_dir, amplitude, 1.0)
      order = "" if previous_error is None else math.log(previous_error / error, 2.0)
      results.append({
         "kind": "convergence",
         "case": f"gauge_wave_n{cells}",
         "status": "PASS",
         "metric": "lapse_l2",
         "value": error,
         "order": order,
         "constraint_linf": final_constraint_max(run_dir),
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
      previous_error = error
   if float(results[-1]["order"]) < 3.5:
      raise RuntimeError(
         f"gauge wave did not retain fourth-order convergence: {results[-1]['order']}"
      )
   return results, scalars


def linear_wave_gamma22_error(run_dir: Path, amplitude: float, wave_number: float) -> float:
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError(f"missing spacetime output in {repo_relative(run_dir)}")
   names, dim, time = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, dim, 0)
   lower, _upper = parse_plot_bounds(plotfile)
   spacing = parse_level_cell_sizes(plotfile)[0][0]
   x = lower[0] + np.arange(state.shape[1]) * spacing
   phase = 2.0 * math.pi * wave_number * (x - time)
   perturbation = amplitude * np.sin(phase)
   chi = (1.0 - perturbation**2) ** (-1.0 / 3.0)
   exact = chi * (1.0 + perturbation)
   return float(np.sqrt(np.mean((state[names.index("gamma_tilde22")] - exact) ** 2)))


def validate_linear_wave_convergence(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   amplitude = 1.0e-6
   results = []
   scalars = []
   previous_error = None
   for cells in (16, 32):
      run_dir, scalar = run_case(
         f"linear_wave_n{cells}",
         {
            "amr.n_cell": str(cells),
            "amr.max_grid_size": str(cells),
            "spacetime.initial_data": "linear_wave",
            "spacetime.lapse_gauge": "frozen",
            "spacetime.shift_gauge": "frozen",
            "spacetime.amplitude": str(amplitude),
            "spacetime.wave_number": "1.0",
            "spacetime.ko_dissipation": "0.0",
            "pyfluid.timestep": "rk4",
            "pyfluid.cfl": "0.2",
            "pyfluid.final_time": "0.2",
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "1000000",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_int": "1000000",
         },
         args,
      )
      error = linear_wave_gamma22_error(run_dir, amplitude, 1.0)
      order = "" if previous_error is None else math.log(previous_error / error, 2.0)
      results.append({
         "kind": "convergence",
         "case": f"linear_wave_n{cells}",
         "status": "PASS",
         "metric": "gamma_tilde22_l2",
         "value": error,
         "order": order,
         "constraint_linf": final_constraint_max(run_dir),
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
      previous_error = error
   if float(results[-1]["order"]) < 3.5:
      raise RuntimeError(
         f"linear wave did not retain fourth-order convergence: {results[-1]['order']}"
      )
   return results, scalars


def validate_robust_stability(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   # The standard robust-stability family scales metric noise as dx**2.
   # Hamiltonian, momentum/Z, and Theta therefore scale respectively as
   # amplitude/dx**2, amplitude/dx, and amplitude.  Comparing every evolved
   # component directly with the seed amplitude is dimensionally invalid.
   final_time = 4.0 if args.level == "quick" else 10.0
   scaled_constraints: list[np.ndarray] = []
   results = []
   scalars = []
   for cells in (32, 64):
      spacing = 1.0 / cells
      amplitude = 1.0e-10 * (64.0 / cells) ** 2
      run_dir, scalar = run_case(
         f"robust_stability_n{cells}",
         {
            "amr.n_cell": str(cells),
            "amr.max_grid_size": str(cells),
            "spacetime.initial_data": "robust_stability",
            "spacetime.lapse_gauge": "frozen",
            "spacetime.shift_gauge": "frozen",
            "spacetime.amplitude": str(amplitude),
            "spacetime.ko_dissipation": "0.1",
            "pyfluid.timestep": "rk4",
            "pyfluid.cfl": "0.1",
            "pyfluid.final_time": str(final_time),
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "1000000",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_int": str(10 * cells),
         },
         args,
      )
      rows = read_scalar_rows(run_dir)
      histories = np.asarray([
         [
            abs(float(row["z4c_hamiltonian_linf"])),
            abs(float(row["z4c_momentum_linf"])),
            abs(float(row["z4c_z_linf"])),
            abs(float(row["z4c_theta_linf"])),
         ]
         for row in rows[1:]
      ])
      if histories.shape[0] < 4 or not np.all(np.isfinite(histories)):
         raise RuntimeError(f"invalid robust-stability history at N={cells}")
      split = histories.shape[0] // 2
      early = np.max(histories[:split], axis=0)
      late = np.max(histories[split:], axis=0)
      growth = float(np.max(late / np.maximum(early, np.finfo(float).tiny)))
      if growth > 3.0:
         raise RuntimeError(
            f"robust-stability constraints grow too quickly at N={cells}: {growth:.3e}"
         )
      final = histories[-1]
      scaled = final / np.asarray([
         amplitude / spacing**2,
         amplitude / spacing,
         amplitude / spacing,
         amplitude,
      ])
      scaled_constraints.append(scaled)
      results.append({
         "kind": "stability",
         "case": f"robust_stability_n{cells}",
         "status": "PASS",
         "metric": "late_constraint_growth",
         "value": growth,
         "constraint_linf": float(np.max(final)),
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)

   resolution_ratio = np.maximum(
      scaled_constraints[0] / np.maximum(scaled_constraints[1], np.finfo(float).tiny),
      scaled_constraints[1] / np.maximum(scaled_constraints[0], np.finfo(float).tiny),
   )
   maximum_ratio = float(np.max(resolution_ratio))
   if maximum_ratio > 4.0:
      raise RuntimeError(
         "robust-stability constraints do not retain expected resolution scaling: "
         f"ratio={maximum_ratio:.3e}"
      )
   results.append({
      "kind": "stability",
      "case": "robust_stability_resolution_scaling",
      "status": "PASS",
      "metric": "maximum_scaled_constraint_ratio",
      "value": maximum_ratio,
      "constraint_linf": max(final_constraint_max(RUN_ROOT / "robust_stability_n32"),
                              final_constraint_max(RUN_ROOT / "robust_stability_n64")),
      "run_dir": repo_relative(RUN_ROOT),
   })
   return results, scalars


def compare_spacetime_files(
   left_file: Path,
   right_file: Path,
   *,
   require_same_time: bool = True,
) -> float:
   left_names, left_dim, left_time = parse_plot_header(left_file)
   right_names, right_dim, right_time = parse_plot_header(right_file)
   if left_names != right_names or left_dim != right_dim:
      raise RuntimeError("spacetime comparison changed field layout")
   if require_same_time and not math.isclose(
      left_time, right_time, rel_tol=0.0, abs_tol=1.0e-14
   ):
      raise RuntimeError("spacetime comparison did not reach the same time")
   if level_count(left_file) != level_count(right_file):
      raise RuntimeError("spacetime comparison changed hierarchy depth")
   maximum = 0.0
   for level in range(level_count(left_file)):
      left = read_level_state(left_file, left_names, left_dim, level)
      right = read_level_state(right_file, right_names, right_dim, level)
      common = np.isfinite(left) & np.isfinite(right)
      if not np.any(common):
         raise RuntimeError(f"spacetime level {level} has no comparable points")
      scale = np.maximum(1.0, np.maximum(np.abs(left[common]), np.abs(right[common])))
      maximum = max(maximum, float(np.max(np.abs(left[common] - right[common]) / scale)))
   return maximum


def compare_spacetime_outputs(left_dir: Path, right_dir: Path) -> float:
   left_file = latest_spacetime_plotfile(left_dir)
   right_file = latest_spacetime_plotfile(right_dir)
   if left_file is None or right_file is None:
      raise RuntimeError("missing spacetime output for comparison")
   return compare_spacetime_files(left_file, right_file)


def validate_tov_cowling(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   full_dir, full_scalar = run_dimension_case(
      "tov_cowling_amr_1d",
      TOV_COWLING_INPUT,
      {"pyfluid.plot_int": "2"},
      args,
      dim=1,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   rows = read_scalar_rows(full_dir)
   plotfiles = spacetime_plotfiles(full_dir)
   levels = level_count(plotfiles[-1]) if plotfiles else 0
   max_rho_change = max(abs(float(row["max_rho_rel_change"])) for row in rows)
   if len(plotfiles) < 2:
      raise RuntimeError("Cowling TOV did not write two spacetime snapshots")
   fixed_metric_change = compare_spacetime_files(
      plotfiles[0], plotfiles[-1], require_same_time=False
   )
   if levels != 4 or max_rho_change > 5.0e-4 or fixed_metric_change > 1.0e-13:
      raise RuntimeError(
         "Cowling TOV failed: "
         f"levels={levels}, max-rho change={max_rho_change:.3e}, "
         f"fixed-metric change={fixed_metric_change:.3e}"
      )

   restart_source = full_dir / "checkpoints/chk00002"
   restart_dir, restart_scalar = run_dimension_case(
      "tov_cowling_amr_1d_restart_from2",
      TOV_COWLING_INPUT,
      {
         "pyfluid.plot_int": "2",
         "pyfluid.restart_file": repo_relative(restart_source),
      },
      args,
      dim=1,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="restart",
   )
   full_checkpoint = latest_checkpoint(full_dir)
   restart_checkpoint = latest_checkpoint(restart_dir)
   if full_checkpoint is None or restart_checkpoint is None:
      raise RuntimeError("Cowling TOV restart validation is missing a checkpoint")
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [full_checkpoint, restart_checkpoint, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)

   coupled_dir, coupled_scalar = run_dimension_case(
      "tov_evolved_amr_1d_control",
      TOV_COWLING_INPUT,
      {
         "spacetime.evolution": "evolved",
         "pyfluid.plot_int": "2",
         "pyfluid.checkpoint_int": "-1",
      },
      args,
      dim=1,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   coupled_rows = read_scalar_rows(coupled_dir)
   coupled_max_rho_change = max(
      abs(float(row["max_rho_rel_change"])) for row in coupled_rows
   )
   if (
      not math.isfinite(coupled_max_rho_change)
      or coupled_max_rho_change > 2.0e-3
      or max_rho_change > 0.1 * coupled_max_rho_change
   ):
      raise RuntimeError(
         "evolved TOV control failed: "
         f"max-rho change={coupled_max_rho_change:.3e}"
      )

   return [
      {
         "kind": "feature",
         "case": "tov_cowling_amr_1d",
         "status": "PASS",
         "metric": "max_rho_rel_change",
         "value": max_rho_change,
         "constraint_linf": final_constraint_max(full_dir),
         "fixed_metric_change": fixed_metric_change,
         "levels": levels,
         "run_dir": repo_relative(full_dir),
      },
      {
         "kind": "restart",
         "case": "tov_cowling_amr_1d_restart",
         "status": "PASS",
         "metric": "checkpoint_compare",
         "value": "exact_within_1e-12",
         "levels": levels,
         "run_dir": repo_relative(restart_dir),
      },
      {
         "kind": "feature",
         "case": "tov_evolved_amr_1d_control",
         "status": "PASS",
         "metric": "max_rho_rel_change",
         "value": coupled_max_rho_change,
         "constraint_linf": final_constraint_max(coupled_dir),
         "levels": level_count(latest_spacetime_plotfile(coupled_dir)),
         "run_dir": repo_relative(coupled_dir),
      },
   ], [full_scalar, restart_scalar, coupled_scalar]


def multilevel_parameters(*, subcycling: bool) -> dict[str, str]:
   return {
      "amr.n_cell": "16",
      "amr.max_level": "3",
      "amr.ref_ratio": "2 2 2",
      "amr.max_grid_size": "16",
      "amr.blocking_factor": "8",
      "amr.tag_metric": "none",
      "amr.initial_refine": "box",
      "amr.initial_refine_levels": "3",
      "amr.initial_refine_fixed_levels": "3",
      "amr.initial_refine_box_lo": "0.25",
      "amr.initial_refine_box_hi": "0.75",
      "spacetime.initial_data": "gauge_wave",
      "spacetime.lapse_gauge": "harmonic",
      "spacetime.amplitude": "1.0e-4",
      "spacetime.ko_dissipation": "0.05",
      "pyfluid.timestep": "rk4",
      "pyfluid.cfl": "0.1",
      "pyfluid.subcycling": "on" if subcycling else "off",
      "pyfluid.subcycling_steps": "spatial",
      "pyfluid.final_time": "0.0125",
      "pyfluid.max_step": "64",
      "pyfluid.plot_int": "1",
      "pyfluid.checkpoint_int": "1",
      "pyfluid.scalar_int": "1",
   }


def validate_amr_subcycling_restart(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   subcycled_dir, subcycled_scalar = run_case(
      "amr_l3_subcycled", multilevel_parameters(subcycling=True), args
   )
   single_rate_dir, single_rate_scalar = run_case(
      "amr_l3_single_rate", multilevel_parameters(subcycling=False), args
   )
   levels = level_count(latest_spacetime_plotfile(subcycled_dir))
   if levels != 4:
      raise RuntimeError(f"Z4c AMR control reached {levels} levels, expected 4")
   difference = compare_spacetime_outputs(subcycled_dir, single_rate_dir)
   if difference > 2.0e-4:
      raise RuntimeError(f"Z4c subcycling difference is too large: {difference:.3e}")
   subcycled_error = gauge_wave_lapse_level_linf(
      subcycled_dir, 1.0e-4, 1.0, levels - 1
   )
   single_rate_error = gauge_wave_lapse_level_linf(
      single_rate_dir, 1.0e-4, 1.0, levels - 1
   )
   fixed_hierarchy_error_tolerance = 2.5e-6
   if (
      subcycled_error > fixed_hierarchy_error_tolerance
      or single_rate_error > fixed_hierarchy_error_tolerance
   ):
      raise RuntimeError(
         "Z4c AMR gauge-wave error is too large: "
         f"subcycled={subcycled_error:.3e}, single_rate={single_rate_error:.3e}"
      )

   restart_source = subcycled_dir / "checkpoints/chk00001"
   restart_dir, restart_scalar = run_case(
      "amr_l3_restart_from1",
      {
         **multilevel_parameters(subcycling=True),
         "pyfluid.restart_file": repo_relative(restart_source),
      },
      args,
   )
   full_checkpoint = latest_checkpoint(subcycled_dir)
   restart_checkpoint = latest_checkpoint(restart_dir)
   if full_checkpoint is None or restart_checkpoint is None:
      raise RuntimeError("Z4c restart validation did not write final checkpoints")
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
         "kind": "amr",
         "case": "amr_l3_subcycling",
         "status": "PASS",
         "metric": "subcycling_max_relative_difference",
         "value": difference,
         "analytic_error_subcycled": subcycled_error,
         "analytic_error_single_rate": single_rate_error,
         "levels": levels,
         "run_dir": repo_relative(subcycled_dir),
      },
      {
         "kind": "restart",
         "case": "amr_l3_restart",
         "status": "PASS",
         "metric": "checkpoint_compare",
         "value": "exact_within_1e-12",
         "levels": levels,
         "run_dir": repo_relative(restart_dir),
      },
   ], [subcycled_scalar, single_rate_scalar, restart_scalar]


def validate_coupled_amr_restart(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   parameters = {
      **multilevel_parameters(subcycling=True),
      "spacetime.matter_coupling": "coupled",
      "spacetime.initial_data": "minkowski",
      "spacetime.lapse_gauge": "one_plus_log",
      "spacetime.ko_dissipation": "0.0",
      "pyfluid.ic": "sin_wave",
      "pyfluid.rho0": "1.0e-8",
      "pyfluid.pressure0": "1.0e-10",
      "pyfluid.amplitude": "0.01",
      "pyfluid.wave_number": "1.0",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.cfl": "0.25",
      "pyfluid.final_time": "-1",
      "pyfluid.max_step": "2",
      "pyfluid.plot_int": "2",
      "pyfluid.checkpoint_int": "1",
   }
   full_dir, full_scalar = run_case(
      "coupled_amr_l3_subcycled", parameters, args
   )
   levels = level_count(latest_spacetime_plotfile(full_dir))
   if levels != 4:
      raise RuntimeError(f"coupled Z4c AMR control reached {levels} levels, expected 4")
   restart_source = full_dir / "checkpoints/chk00001"
   restart_dir, restart_scalar = run_case(
      "coupled_amr_l3_restart_from1",
      {**parameters, "pyfluid.restart_file": repo_relative(restart_source)},
      args,
   )
   full_checkpoint = latest_checkpoint(full_dir)
   restart_checkpoint = latest_checkpoint(restart_dir)
   if full_checkpoint is None or restart_checkpoint is None:
      raise RuntimeError("coupled Z4c restart validation did not write final checkpoints")
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [full_checkpoint, restart_checkpoint, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)
   return [{
      "kind": "restart",
      "case": "coupled_amr_l3_restart",
      "status": "PASS",
      "metric": "checkpoint_compare",
      "value": "exact_within_1e-12",
      "levels": levels,
      "run_dir": repo_relative(restart_dir),
   }], [full_scalar, restart_scalar]


def validate_coupled_rmhd_smoke(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   parameters = {
      **multilevel_parameters(subcycling=True),
      "spacetime.matter_coupling": "coupled",
      "spacetime.initial_data": "minkowski",
      "spacetime.lapse_gauge": "one_plus_log",
      "spacetime.ko_dissipation": "0.0",
      "pyfluid.physics": "rmhd",
      "pyfluid.eos": "ideal_gas",
      "pyfluid.ic": "sin_wave",
      "pyfluid.rho0": "1.0e-8",
      "pyfluid.pressure0": "1.0e-10",
      "pyfluid.amplitude": "0.01",
      "pyfluid.wave_number": "1.0",
      "pyfluid.v1_0": "0.1",
      "pyfluid.v2_0": "0.0",
      "pyfluid.v3_0": "0.0",
      "pyfluid.b1_0": "1.0e-5",
      "pyfluid.b2_0": "5.0e-6",
      "pyfluid.b3_0": "0.0",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hll",
      "pyfluid.ct": "off",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.cfl": "0.25",
      "pyfluid.final_time": "-1",
      "pyfluid.max_step": "2",
      "pyfluid.plot_int": "2",
      "pyfluid.checkpoint_int": "-1",
   }
   run_dir, scalar = run_dimension_case(
      "coupled_rmhd_amr_l3_subcycled",
      INPUT,
      parameters,
      args,
      dim=1,
      checks=SMOKE_POLICY,
      mhd=True,
      kind="feature",
   )
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError("coupled RMHD smoke did not write spacetime state")
   levels = level_count(plotfile)
   rows = read_scalar_rows(run_dir)
   mass_drift = max(abs(float(row["mass_rel_drift"])) for row in rows)
   constraint = final_constraint_max(run_dir)
   if levels != 4:
      raise RuntimeError(f"coupled RMHD control reached {levels} levels, expected 4")
   if mass_drift > 1.0e-10 or not math.isfinite(constraint):
      raise RuntimeError(
         "coupled RMHD smoke failed: "
         f"mass drift={mass_drift:.3e}, constraints={constraint:.3e}"
      )
   return [{
      "kind": "feature",
      "case": "coupled_rmhd_amr_l3_subcycled",
      "status": "PASS",
      "metric": "mass_rel_drift",
      "value": mass_drift,
      "constraint_linf": constraint,
      "levels": levels,
      "run_dir": repo_relative(run_dir),
   }], [scalar]


def validate_tov_initial_data(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   resolutions = (24, 32, 48)
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   previous_cells: int | None = None
   previous_l2: float | None = None
   previous_linf: float | None = None

   for cells in resolutions:
      run_dir, scalar = run_dimension_case(
         f"tov_initial_n{cells}",
         TOV_INPUT,
         {
            "amr.n_cell": f"{cells} {cells} {cells}",
            "amr.max_level": "1",
            "amr.max_grid_size": str(cells),
            "amr.tag_metric": "none",
            "pyfluid.final_time": "-1",
            "pyfluid.max_step": "0",
            "pyfluid.plot_int": "1",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_int": "1",
         },
         args,
         dim=3,
         checks=SMOKE_POLICY,
         mhd=False,
      )
      final = read_scalar_rows(run_dir)[-1]
      h_l2 = abs(float(final["z4c_hamiltonian_l2"]))
      h_linf = abs(float(final["z4c_hamiltonian_linf"]))
      if not math.isfinite(h_l2) or not math.isfinite(h_linf):
         raise RuntimeError(f"TOV N={cells} produced nonfinite constraints")
      order_l2 = ""
      order_linf = ""
      if previous_cells is not None and previous_l2 is not None and previous_linf is not None:
         denominator = math.log(cells / previous_cells)
         order_l2 = math.log(previous_l2 / h_l2) / denominator
         order_linf = math.log(previous_linf / h_linf) / denominator
      results.append({
         "kind": "convergence",
         "case": f"tov_initial_n{cells}",
         "status": "PASS",
         "metric": "hamiltonian_l2",
         "value": h_l2,
         "order": order_l2,
         "constraint_linf": h_linf,
         "hamiltonian_l2": h_l2,
         "hamiltonian_linf": h_linf,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
      previous_cells = cells
      previous_l2 = h_l2
      previous_linf = h_linf

   if any(
      float(results[index]["hamiltonian_l2"])
      >= float(results[index - 1]["hamiltonian_l2"])
      for index in range(1, len(results))
   ) or float(results[-1]["order"]) < 1.0:
      raise RuntimeError(
         "TOV Hamiltonian constraint did not show surface-limited convergence: "
         + ", ".join(
            f"N={cells}: {float(result['hamiltonian_l2']):.3e}"
            for cells, result in zip(resolutions, results)
         )
      )

   run_dir, scalar = run_dimension_case(
      "tov_coupled_amr_3d",
      TOV_INPUT,
      {},
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   rows = read_scalar_rows(run_dir)
   plotfile = latest_spacetime_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError("coupled TOV run did not write spacetime state")
   levels = level_count(plotfile)
   mass_drift = max(abs(float(row["mass_rel_drift"])) for row in rows)
   initial_h_l2 = abs(float(rows[0]["z4c_hamiltonian_l2"]))
   final_h_l2 = abs(float(rows[-1]["z4c_hamiltonian_l2"]))
   uniform_coarse_h_l2 = float(results[0]["hamiltonian_l2"])
   amr_constraint_ratio = initial_h_l2 / uniform_coarse_h_l2
   if (
      levels < 2
      or mass_drift > 1.0e-8
      or final_h_l2 > 2.0 * initial_h_l2
      or amr_constraint_ratio > 32.0
   ):
      raise RuntimeError(
         "coupled TOV AMR smoke failed: "
         f"levels={levels}, mass drift={mass_drift:.3e}, "
         f"Hamiltonian L2={initial_h_l2:.3e}->{final_h_l2:.3e}, "
         f"AMR/uniform ratio={amr_constraint_ratio:.3e}"
      )
   results.append({
      "kind": "feature",
      "case": "tov_coupled_amr_3d",
      "status": "PASS",
      "metric": "hamiltonian_l2_growth",
      "value": final_h_l2 / max(initial_h_l2, np.finfo(float).tiny),
      "constraint_linf": final_constraint_max(run_dir),
      "hamiltonian_l2": final_h_l2,
      "hamiltonian_linf": abs(float(rows[-1]["z4c_hamiltonian_linf"])),
      "amr_constraint_ratio": amr_constraint_ratio,
      "levels": levels,
      "run_dir": repo_relative(run_dir),
   })
   scalars.append(scalar)

   piecewise_dir, piecewise_scalar = run_dimension_case(
      "tov_piecewise_polytrope_n24",
      TOV_INPUT,
      {
         "amr.n_cell": "24 24 24",
         "amr.max_level": "1",
         "amr.max_grid_size": "24",
         "amr.tag_metric": "none",
         "pyfluid.polytropic_density_breaks": "6.0e-4",
         "pyfluid.polytropic_segment_gammas": "2.5",
         "pyfluid.final_time": "-1",
         "pyfluid.max_step": "2",
         "pyfluid.plot_int": "2",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.scalar_int": "1",
      },
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
   )
   piecewise_rows = read_scalar_rows(piecewise_dir)
   piecewise_initial = piecewise_rows[0]
   piecewise_final = piecewise_rows[-1]
   piecewise_h_l2 = abs(float(piecewise_final["z4c_hamiltonian_l2"]))
   piecewise_h_linf = abs(float(piecewise_final["z4c_hamiltonian_linf"]))
   piecewise_initial_h_l2 = abs(float(piecewise_initial["z4c_hamiltonian_l2"]))
   piecewise_mass_drift = max(
      abs(float(row["mass_rel_drift"])) for row in piecewise_rows
   )
   if (
      not math.isfinite(piecewise_h_l2)
      or not math.isfinite(piecewise_h_linf)
      or piecewise_mass_drift > 1.0e-8
      or piecewise_h_l2 > 2.0 * piecewise_initial_h_l2
   ):
      raise RuntimeError(
         "evolved piecewise-polytrope TOV failed: "
         f"mass drift={piecewise_mass_drift:.3e}, "
         f"Hamiltonian L2={piecewise_initial_h_l2:.3e}->{piecewise_h_l2:.3e}"
      )
   results.append({
      "kind": "feature",
      "case": "tov_piecewise_polytrope_n24",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": piecewise_h_l2,
      "constraint_linf": piecewise_h_linf,
      "hamiltonian_l2": piecewise_h_l2,
      "hamiltonian_linf": piecewise_h_linf,
      "run_dir": repo_relative(piecewise_dir),
   })
   scalars.append(piecewise_scalar)
   return results, scalars


def validate_xcts_initial_data(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   run_dir, scalar = run_dimension_case(
      "tov_xcts_3d",
      XCTS_TOV_INPUT,
      {},
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   rows = read_scalar_rows(run_dir)
   final = rows[-1]
   h_l2 = abs(float(final["z4c_hamiltonian_l2"]))
   h_linf = abs(float(final["z4c_hamiltonian_linf"]))
   momentum_linf = abs(float(final["z4c_momentum_linf"]))
   run_log = (run_dir / "run.log").read_text()
   if (
      not math.isfinite(h_l2)
      or not math.isfinite(h_linf)
      or not math.isfinite(momentum_linf)
      or "xcts initial data:" not in run_log
   ):
      raise RuntimeError(
         "staggered TOV XCTS materialization failed: "
         f"Hamiltonian L2={h_l2:.3e}, Linf={h_linf:.3e}, "
         f"momentum Linf={momentum_linf:.3e}"
      )
   return [{
      "kind": "feature",
      "case": "tov_xcts_3d",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": h_l2,
      "constraint_linf": max(h_linf, momentum_linf),
      "hamiltonian_l2": h_l2,
      "hamiltonian_linf": h_linf,
      "momentum_linf": momentum_linf,
      "run_dir": repo_relative(run_dir),
   }], [scalar]


def validate_ctt_initial_data(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   resolutions = (16, 24, 32)
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   uniform_l2: dict[int, float] = {}
   uniform_runs: dict[int, Path] = {}
   previous_cells: int | None = None
   previous_l2: float | None = None

   for cells in resolutions:
      name = f"ctt_density_cloud_n{cells}"
      run_dir, scalar = run_dimension_case(
         name,
         CTT_INPUT,
         {
            "amr.n_cell": f"{cells} {cells} {cells}",
            "amr.max_level": "0",
            "amr.max_grid_size": str(min(cells, 16)),
            "amr.tag_metric": "none",
            "pyfluid.max_step": "0",
            "pyfluid.plot_int": "0",
         },
         args,
         dim=3,
         checks=SMOKE_POLICY,
         mhd=False,
      )
      final = read_scalar_rows(run_dir)[-1]
      h_l2 = abs(float(final["z4c_hamiltonian_l2"]))
      h_linf = abs(float(final["z4c_hamiltonian_linf"]))
      if not math.isfinite(h_l2) or not math.isfinite(h_linf):
         raise RuntimeError(f"compact CTT N={cells} produced nonfinite constraints")
      order: float | str = ""
      if previous_cells is not None and previous_l2 is not None:
         order = math.log(previous_l2 / h_l2) / math.log(cells / previous_cells)
      uniform_l2[cells] = h_l2
      uniform_runs[cells] = run_dir
      results.append({
         "kind": "convergence",
         "case": name,
         "status": "PASS",
         "metric": "hamiltonian_l2",
         "value": h_l2,
         "order": order,
         "constraint_linf": h_linf,
         "hamiltonian_l2": h_l2,
         "hamiltonian_linf": h_linf,
         "levels": 1,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
      previous_cells = cells
      previous_l2 = h_l2

   overall_order = math.log(uniform_l2[resolutions[0]] / uniform_l2[resolutions[-1]]) \
      / math.log(resolutions[-1] / resolutions[0])
   if any(
      uniform_l2[cells] >= uniform_l2[resolutions[index - 1]]
      for index, cells in enumerate(resolutions[1:], start=1)
   ) or overall_order < 1.0:
      raise RuntimeError(
         "compact CTT Hamiltonian constraint did not converge: "
         + ", ".join(
            f"N={cells}: {uniform_l2[cells]:.3e}" for cells in resolutions
         )
      )

   packet_dir, packet_scalar = run_dimension_case(
      "ctt_acoustic_packet_n12",
      CTT_INPUT,
      {
         "amr.n_cell": "12 12 12",
         "amr.max_level": "0",
         "amr.max_grid_size": "6",
         "amr.tag_metric": "none",
         "pyfluid.ic": "acoustic_packet_x",
         "pyfluid.rho0": "1.0e-5",
         "pyfluid.pressure0": "1.0e-7",
         "pyfluid.amplitude": "1.0e-7",
         "pyfluid.bump_sigma": "0.75",
         "pyfluid.acoustic_transverse_sigma": "0.75",
         "pyfluid.max_step": "0",
         "pyfluid.plot_int": "0",
      },
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   packet_final = read_scalar_rows(packet_dir)[-1]
   packet_h_l2 = abs(float(packet_final["z4c_hamiltonian_l2"]))
   packet_momentum = abs(float(packet_final["z4c_momentum_linf"]))
   if not math.isfinite(packet_h_l2) or not math.isfinite(packet_momentum):
      raise RuntimeError("compact CTT acoustic packet produced nonfinite constraints")
   results.append({
      "kind": "feature",
      "case": "ctt_acoustic_packet_n12",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": packet_h_l2,
      "constraint_linf": max(
         abs(float(packet_final["z4c_hamiltonian_linf"])),
         packet_momentum,
      ),
      "hamiltonian_l2": packet_h_l2,
      "hamiltonian_linf": abs(float(packet_final["z4c_hamiltonian_linf"])),
      "momentum_linf": packet_momentum,
      "levels": 1,
      "run_dir": repo_relative(packet_dir),
   })
   scalars.append(packet_scalar)

   amr_dir, amr_scalar = run_dimension_case(
      "ctt_density_cloud_amr",
      CTT_INPUT,
      {
         "amr.n_cell": "16 16 16",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.max_grid_size": "16",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_fixed_levels": "1",
         "amr.initial_refine_box_lo": "-1.5 -1.5 -1.5",
         "amr.initial_refine_box_hi": "1.5 1.5 1.5",
         "pyfluid.max_step": "0",
         "pyfluid.plot_int": "0",
      },
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   amr_final = read_scalar_rows(amr_dir)[-1]
   amr_l2 = abs(float(amr_final["z4c_hamiltonian_l2"]))
   amr_linf = abs(float(amr_final["z4c_hamiltonian_linf"]))
   amr_plotfile = latest_spacetime_plotfile(amr_dir)
   coarse_plotfile = latest_spacetime_plotfile(uniform_runs[16])
   reference_plotfile = latest_spacetime_plotfile(uniform_runs[32])
   if amr_plotfile is None or coarse_plotfile is None or reference_plotfile is None:
      raise RuntimeError("compact CTT AMR comparison is missing initial state output")
   amr_names, amr_dim, _amr_time = parse_plot_header(amr_plotfile)
   coarse_names, coarse_dim, _coarse_time = parse_plot_header(coarse_plotfile)
   reference_names, reference_dim, _reference_time = parse_plot_header(
      reference_plotfile
   )
   if (
      amr_names != coarse_names
      or amr_names != reference_names
      or amr_dim != 3
      or coarse_dim != 3
      or reference_dim != 3
      or level_count(amr_plotfile) != 2
   ):
      raise RuntimeError("compact CTT AMR comparison changed its field/layout contract")
   chi_component = amr_names.index("chi")
   reference_state = read_level_state(reference_plotfile, reference_names, 3, 0)
   coarse_state = read_level_state(coarse_plotfile, coarse_names, 3, 0)
   fine_errors: list[float] = []
   coarse_errors: list[float] = []
   visited: set[tuple[int, int, int]] = set()
   for lo, hi, data in level_chunks(amr_plotfile, 1, 3, names=amr_names):
      for i in range(lo[0], hi[0] + 1):
         for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
               point = (i, j, k)
               if point in visited or any(index % 2 != 0 for index in point):
                  continue
               visited.add(point)
               reference = reference_state[(chi_component, *point)]
               fine_errors.append(
                  data[chi_component, i - lo[0], j - lo[1], k - lo[2]]
                  - reference
               )
               coarse_errors.append(
                  coarse_state[chi_component, i // 2, j // 2, k // 2]
                  - reference
               )
   if not fine_errors:
      raise RuntimeError("compact CTT AMR comparison found no shared sample points")
   fine_solution_l2 = float(np.sqrt(np.mean(np.square(fine_errors))))
   coarse_solution_l2 = float(np.sqrt(np.mean(np.square(coarse_errors))))
   solution_ratio = fine_solution_l2 / coarse_solution_l2
   constraint_ratio = amr_l2 / uniform_l2[16]
   if (
      not math.isfinite(amr_l2)
      or not math.isfinite(solution_ratio)
      or solution_ratio >= 0.75
      or constraint_ratio >= 4.0
   ):
      raise RuntimeError(
         "compact CTT AMR handoff failed: "
         f"solution ratio={solution_ratio:.3e}, "
         f"constraint ratio={constraint_ratio:.3e}"
      )
   results.append({
      "kind": "feature",
      "case": "ctt_density_cloud_amr",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": amr_l2,
      "constraint_linf": amr_linf,
      "hamiltonian_l2": amr_l2,
      "hamiltonian_linf": amr_linf,
      "amr_solution_ratio": solution_ratio,
      "amr_constraint_ratio": constraint_ratio,
      "levels": 2,
      "run_dir": repo_relative(amr_dir),
   })
   scalars.append(amr_scalar)
   return results, scalars


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "standard"), default="standard")
   parser.add_argument("--only", choices=("all", "ctt", "xcts"), default="all")
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=float, default=180.0)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   if not CHECKPOINT_COMPARE.exists():
      raise SystemExit("build the AMReX solver and checkpoint comparator first")
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   quick_validators = (
      (validate_ctt_initial_data,)
      if args.only == "ctt"
      else (validate_xcts_initial_data,)
      if args.only == "xcts"
      else (
         validate_minkowski,
         validate_initial_data_group_policies,
         validate_gauge_wave_convergence,
         validate_linear_wave_convergence,
         validate_robust_stability,
         validate_amr_subcycling_restart,
      )
   )
   for validator in quick_validators:
      validator_results, validator_scalars = validator(args)
      results.extend(validator_results)
      scalars.extend(validator_scalars)
   if args.level == "standard" and args.only == "all":
      for validator in (
         validate_coupled_amr_restart,
         validate_coupled_rmhd_smoke,
         validate_tov_cowling,
         validate_tov_initial_data,
         validate_xcts_initial_data,
      ):
         validator_results, validator_scalars = validator(args)
         results.extend(validator_results)
         scalars.extend(validator_scalars)

   fields = [
      "kind",
      "case",
      "status",
      "metric",
      "value",
      "order",
      "constraint_linf",
      "hamiltonian_l2",
      "hamiltonian_linf",
      "momentum_linf",
      "amr_solution_ratio",
      "amr_constraint_ratio",
      "fixed_metric_change",
      "analytic_error_subcycled",
      "analytic_error_single_rate",
      "levels",
      "run_dir",
   ]
   write_tsv(RESULTS, results, fields)
   append_validation_history(HISTORY, scalars)
   print(f"PASS z4c validation ({len(results)} checks): {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
