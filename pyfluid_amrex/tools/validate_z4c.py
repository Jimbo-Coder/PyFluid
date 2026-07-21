#!/usr/bin/env python3
"""Validate the generated vacuum Z4c evolution vertical slice."""

from __future__ import annotations

import argparse
import math
import re
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
   read_key_values,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_validation_row,
   write_tsv,
)


INPUT = ROOT / "pyfluid_amrex/inputs/z4c_gauge_wave_1d.in"
TOV_INPUT = ROOT / "pyfluid_amrex/inputs/tov_coupled_amr_3d.in"
TOV_OCTANT_INPUT = ROOT / "pyfluid_amrex/inputs/tov_cowling_octant_amr_3d.in"
CONFORMAL_TOV_INPUT = ROOT / "pyfluid_amrex/inputs/tov_xcts_3d.in"
ROTATING_STAR_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_3d.in"
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

TOV_OBSERVABLE_PATTERN = re.compile(
   r"TOV initial data: M=(?P<mass>[0-9.eE+-]+) "
   r"Mb=(?P<baryonic_mass>[0-9.eE+-]+) "
   r"Mp=(?P<proper_mass>[0-9.eE+-]+) "
   r"Mkomar=(?P<komar_mass>[0-9.eE+-]+) "
   r"R_areal=(?P<areal_radius>[0-9.eE+-]+)"
)


def parse_tov_observables(run_log: str) -> dict[str, float]:
   match = TOV_OBSERVABLE_PATTERN.search(run_log)
   if match is None:
      raise RuntimeError("run did not report the complete TOV observable set")
   return {name: float(value) for name, value in match.groupdict().items()}


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
   remove_parameters: tuple[str, ...] = (),
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
      remove_parameters=remove_parameters,
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


def validate_realistic_eos_tov(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   run_dir, scalar = run_dimension_case(
      "tov_sly_1d",
      TOV_COWLING_INPUT,
      {
         "amr.n_cell": "64",
         "amr.max_level": "3",
         "amr.max_grid_size": "64",
         "amr.tag_metric": "none",
         "amr.initial_refine_levels": "0",
         "amr.initial_refine_fixed_levels": "0",
         "pyfluid.eos_preset": "sly",
         "pyfluid.max_step": "0",
         "pyfluid.plot_int": "1",
         "pyfluid.checkpoint_int": "-1",
      },
      args,
      dim=1,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   run_log = (run_dir / "run.log").read_text()
   observables = parse_tov_observables(run_log)
   mass = observables["mass"]
   radius = observables["areal_radius"]
   if (
      not math.isfinite(mass)
      or not math.isfinite(radius)
      or mass <= 0.0
      or radius <= 0.0
   ):
      raise RuntimeError(f"SLy TOV is not physical: M={mass}, R={radius}")
   return [{
      "kind": "feature",
      "case": "tov_sly_1d",
      "status": "PASS",
      "metric": "stellar_mass",
      "value": mass,
      "stellar_radius": radius,
      "run_dir": repo_relative(run_dir),
   }], [scalar]


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


def maximum_relative_change(
   rows: list[dict[str, str]],
   field: str,
) -> float:
   values = np.asarray([float(row[field]) for row in rows])
   if not np.all(np.isfinite(values)):
      return math.inf
   scale = max(abs(float(values[0])), np.finfo(float).tiny)
   return float(np.max(np.abs(values - values[0])) / scale)


def validate_tov_evolution(
   args: argparse.Namespace,
   uniform_coarse_h_l2: float | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   full_cowling_initial: dict[str, str] | None = None
   common_overrides = {
      "pyfluid.max_step": "4",
      "pyfluid.plot_int": "2",
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.scalar_int": "1",
      "pyfluid.scalar_diagnostics": "standard,global_gr",
      "pyfluid.debug_check_finite": "1",
      "pyfluid.debug_check_physical": "1",
   }

   for evolution in ("fixed", "evolved"):
      name = f"tov_{'cowling' if evolution == 'fixed' else 'coupled'}_amr_3d"
      run_dir, scalar = run_dimension_case(
         name,
         TOV_INPUT,
         {
            **common_overrides,
            "spacetime.evolution": evolution,
         },
         args,
         dim=3,
         checks=SMOKE_POLICY,
         mhd=False,
         kind="feature",
      )
      rows = read_scalar_rows(run_dir)
      source_parameters = read_key_values(TOV_INPUT)
      canonical_parameters = read_key_values(run_dir / "run.in")
      if (
         "amr.min_feature_length" not in source_parameters
         and "amr.min_feature_length" in canonical_parameters
      ):
         raise RuntimeError(
            f"{name} canonical input converted the unset auto AMR length guard "
            "into an explicit value"
         )
      if evolution == "fixed":
         full_cowling_initial = rows[0]
      plotfiles = spacetime_plotfiles(run_dir)
      if len(plotfiles) < 2:
         raise RuntimeError(f"{name} did not write initial and final spacetime states")
      levels = level_count(plotfiles[-1])
      rest_mass_drift = max(abs(float(row["mass_rel_drift"])) for row in rows)
      external_mass_error = max(
         abs(float(row["external_mass_rel_error"])) for row in rows
      )
      max_rho_change = max(abs(float(row["max_rho_rel_change"])) for row in rows)
      adm_surface_change = maximum_relative_change(rows, "adm_mass_surface")
      adm_volume_change = maximum_relative_change(rows, "adm_mass_volume")
      initial_h_l2 = abs(float(rows[0]["z4c_hamiltonian_l2"]))
      final_h_l2 = abs(float(rows[-1]["z4c_hamiltonian_l2"]))
      fixed_metric_change = (
         compare_spacetime_files(plotfiles[0], plotfiles[-1], require_same_time=False)
         if evolution == "fixed"
         else math.nan
      )
      amr_constraint_ratio = (
         initial_h_l2 / uniform_coarse_h_l2
         if uniform_coarse_h_l2 is not None
         else math.nan
      )
      if (
         levels < 2
         or rest_mass_drift > 1.0e-8
         or external_mass_error > 1.0e-12
         or max_rho_change > 1.0e-2
         or adm_volume_change > 1.0e-3
         or final_h_l2 > max(4.0 * initial_h_l2, 1.0e-4)
         or (evolution == "fixed" and adm_surface_change > 1.0e-13)
         or (evolution == "fixed" and fixed_metric_change > 1.0e-13)
         or (
            uniform_coarse_h_l2 is not None
            and amr_constraint_ratio > 32.0
         )
      ):
         raise RuntimeError(
            f"{name} failed: levels={levels}, rest-mass drift={rest_mass_drift:.3e}, "
            f"external mass error={external_mass_error:.3e}, "
            f"max-rho change={max_rho_change:.3e}, ADM changes="
            f"({adm_surface_change:.3e}, {adm_volume_change:.3e}), "
            f"Hamiltonian L2={initial_h_l2:.3e}->{final_h_l2:.3e}, "
            f"fixed-metric change={fixed_metric_change:.3e}"
         )
      results.append({
         "kind": "feature",
         "case": name,
         "status": "PASS",
         "metric": "rest_mass_drift",
         "value": rest_mass_drift,
         "rest_mass_drift": rest_mass_drift,
         "external_mass_relative_error": external_mass_error,
         "max_rho_relative_change": max_rho_change,
         "adm_mass_surface": float(rows[-1]["adm_mass_surface"]),
         "adm_mass_volume": float(rows[-1]["adm_mass_volume"]),
         "adm_mass_surface_relative_change": adm_surface_change,
         "adm_mass_volume_relative_change": adm_volume_change,
         "constraint_linf": final_constraint_max(run_dir),
         "hamiltonian_l2": final_h_l2,
         "hamiltonian_linf": abs(float(rows[-1]["z4c_hamiltonian_linf"])),
         "amr_constraint_ratio": amr_constraint_ratio,
         "fixed_metric_change": fixed_metric_change,
         "levels": levels,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)

   if full_cowling_initial is None:
      raise RuntimeError("full-domain Cowling reference was not produced")
   octant_dir, octant_scalar = run_dimension_case(
      "tov_cowling_octant_amr_3d",
      TOV_OCTANT_INPUT,
      {
         **common_overrides,
         "pyfluid.scalar_diagnostics": (
            "standard,adm_mass_volume,proper_mass_volume,komar_mass_volume,"
            "adm_momentum_volume,adm_angular_momentum_volume"
         ),
      },
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="feature",
   )
   octant_rows = read_scalar_rows(octant_dir)
   octant_initial = octant_rows[0]
   octant_levels = level_count(spacetime_plotfiles(octant_dir)[-1])
   scalar_charges = (
      "mass",
      "adm_mass_volume",
      "proper_mass_volume",
      "komar_mass_volume",
   )
   charge_errors = {
      field: abs(float(octant_initial[field]) - float(full_cowling_initial[field]))
      / max(abs(float(full_cowling_initial[field])), np.finfo(float).tiny)
      for field in scalar_charges
   }
   odd_charge_fields = tuple(
      f"adm_{kind}{axis}_volume"
      for kind in ("momentum", "angular_momentum")
      for axis in (1, 2, 3)
   )
   odd_charge_max = max(abs(float(octant_initial[field])) for field in odd_charge_fields)
   rest_mass_drift = max(abs(float(row["mass_rel_drift"])) for row in octant_rows)
   external_mass_error = max(
      abs(float(row["external_mass_rel_error"])) for row in octant_rows
   )
   max_rho_change = max(abs(float(row["max_rho_rel_change"])) for row in octant_rows)
   maximum_charge_error = max(charge_errors.values())
   if (
      octant_levels < 2
      or maximum_charge_error > 5.0e-4
      or odd_charge_max > 1.0e-12
      or rest_mass_drift > 1.0e-8
      or external_mass_error > 1.0e-12
      or max_rho_change > 1.0e-2
   ):
      raise RuntimeError(
         "octant TOV symmetry failed: "
         f"levels={octant_levels}, charge errors={charge_errors}, "
         f"odd charge={odd_charge_max:.3e}, rest-mass drift={rest_mass_drift:.3e}, "
         f"external mass error={external_mass_error:.3e}, "
         f"max-rho change={max_rho_change:.3e}"
      )
   results.append({
      "kind": "feature",
      "case": "tov_cowling_octant_amr_3d",
      "status": "PASS",
      "metric": "full_octant_volume_charge_error",
      "value": maximum_charge_error,
      "rest_mass_drift": rest_mass_drift,
      "external_mass_relative_error": external_mass_error,
      "max_rho_relative_change": max_rho_change,
      "adm_mass_volume": float(octant_rows[-1]["adm_mass_volume"]),
      "constraint_linf": final_constraint_max(octant_dir),
      "hamiltonian_l2": abs(float(octant_rows[-1]["z4c_hamiltonian_l2"])),
      "hamiltonian_linf": abs(float(octant_rows[-1]["z4c_hamiltonian_linf"])),
      "levels": octant_levels,
      "run_dir": repo_relative(octant_dir),
   })
   scalars.append(octant_scalar)
   return results, scalars


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

   uniform_coarse_h_l2 = float(results[0]["hamiltonian_l2"])
   evolution_results, evolution_scalars = validate_tov_evolution(
      args, uniform_coarse_h_l2
   )
   results.extend(evolution_results)
   scalars.extend(evolution_scalars)

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


def validate_conformal_tov_initial_data(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   formulations = (
      (args.formulation,)
      if args.formulation is not None
      else ("ctt", "cttk", "xcts")
   )
   for formulation in formulations:
      name = f"tov_{formulation}_3d"
      global_gr_enabled = True
      run_dir, scalar = run_dimension_case(
         name,
         CONFORMAL_TOV_INPUT,
         {
            "spacetime.initial_data": formulation,
            "pyfluid.scalar_diagnostics": (
               "standard,global_gr" if global_gr_enabled else "standard"
            ),
         },
         args,
         dim=3,
         checks=SMOKE_POLICY,
         mhd=False,
         kind="feature",
      )
      final = read_scalar_rows(run_dir)[-1]
      h_l2 = abs(float(final["z4c_hamiltonian_l2"]))
      h_linf = abs(float(final["z4c_hamiltonian_linf"]))
      momentum_linf = abs(float(final["z4c_momentum_linf"]))
      run_log = (run_dir / "run.log").read_text()
      solve_match = re.search(
         rf"{formulation} initial data: matter_iterations=([0-9]+) "
         r"relative_matter_update=([0-9.eE+-]+) coupled_iterations=([0-9]+) "
         r"hamiltonian_residual=([0-9.eE+-]+) "
         r"lapse_residual=([0-9.eE+-]+) momentum_residual=([0-9.eE+-]+)",
         run_log,
      )
      elliptic_hamiltonian = math.inf
      elliptic_lapse = math.inf
      elliptic_momentum = math.inf
      relative_matter_update = math.inf
      tov_observables = parse_tov_observables(run_log)
      if solve_match is not None:
         relative_matter_update = abs(float(solve_match.group(2)))
         elliptic_hamiltonian = abs(float(solve_match.group(4)))
         elliptic_lapse = abs(float(solve_match.group(5)))
         elliptic_momentum = abs(float(solve_match.group(6)))
      adm_mass_surface = math.nan
      adm_mass_volume = math.nan
      adm_surface_volume_relative_error = math.nan
      adm_reference_relative_error = math.nan
      adm_angular_momentum_max = math.nan
      adm_momentum_max = math.nan
      proper_mass_relative_error = math.nan
      baryonic_mass_relative_error = math.nan
      binding_energy_mass_normalized_error = math.nan
      komar_volume_relative_error = math.nan
      komar_surface_volume_relative_error = math.nan
      if global_gr_enabled:
         adm_mass_surface = float(final["adm_mass_surface"])
         adm_mass_volume = float(final["adm_mass_volume"])
         adm_surface_volume_relative_error = abs(
            adm_mass_surface - adm_mass_volume
         ) / max(abs(adm_mass_surface), abs(adm_mass_volume), 1.0e-30)
         tov_mass = tov_observables["mass"]
         adm_reference_relative_error = abs(adm_mass_volume - tov_mass) / max(
            abs(tov_mass), 1.0e-30
         )
         adm_angular_momentum_max = max(
            abs(float(final[f"adm_angular_momentum{axis}_{form}"]))
            for axis in (1, 2, 3)
            for form in ("surface", "volume")
         )
         adm_momentum_max = max(
            abs(float(final[f"adm_momentum{axis}_{form}"]))
            for axis in (1, 2, 3)
            for form in ("surface", "volume")
         )
         proper_mass = float(final["proper_mass_volume"])
         baryonic_mass = float(final["mass"])
         binding_energy = float(final["binding_energy_volume"])
         reference_binding_energy = tov_observables["baryonic_mass"] - tov_mass
         proper_mass_relative_error = abs(
            proper_mass - tov_observables["proper_mass"]
         ) / max(abs(tov_observables["proper_mass"]), 1.0e-30)
         baryonic_mass_relative_error = abs(
            baryonic_mass - tov_observables["baryonic_mass"]
         ) / max(abs(tov_observables["baryonic_mass"]), 1.0e-30)
         binding_energy_mass_normalized_error = abs(
            binding_energy - reference_binding_energy
         ) / max(abs(tov_mass), 1.0e-30)
         if formulation == "xcts":
            komar_surface = float(final["komar_mass_surface"])
            komar_volume = float(final["komar_mass_volume"])
            komar_volume_relative_error = abs(
               komar_volume - tov_observables["komar_mass"]
            ) / max(abs(tov_observables["komar_mass"]), 1.0e-30)
            komar_surface_volume_relative_error = abs(
               komar_surface - komar_volume
            ) / max(abs(komar_surface), abs(komar_volume), 1.0e-30)
      if (
         not math.isfinite(h_l2)
         or not math.isfinite(h_linf)
         or not math.isfinite(momentum_linf)
         or solve_match is None
         or relative_matter_update > 1.0e-9
         or max(elliptic_hamiltonian, elliptic_lapse, elliptic_momentum) > 1.0e-10
         or (
            global_gr_enabled
            and (
               not math.isfinite(adm_mass_surface)
               or not math.isfinite(adm_mass_volume)
               or adm_surface_volume_relative_error > 1.0e-2
               or adm_angular_momentum_max > 1.0e-10
               or adm_momentum_max > 1.0e-10
               or not all(math.isfinite(value) for value in (
                  adm_reference_relative_error,
                  proper_mass_relative_error,
                  baryonic_mass_relative_error,
                  binding_energy_mass_normalized_error,
               ))
               or (
                  formulation != "cttk"
                  and (
                     adm_reference_relative_error > 6.0e-2
                     or proper_mass_relative_error > 3.0e-2
                     or baryonic_mass_relative_error > 3.0e-2
                     or binding_energy_mass_normalized_error > 5.0e-2
                  )
               )
               or (
                  formulation == "xcts"
                  and (
                     komar_volume_relative_error > 6.0e-2
                     or komar_surface_volume_relative_error > 2.0e-2
                  )
               )
            )
         )
      ):
         raise RuntimeError(
            f"staggered TOV {formulation.upper()} materialization failed: "
            f"Hamiltonian L2={h_l2:.3e}, Linf={h_linf:.3e}, "
            f"momentum Linf={momentum_linf:.3e}, "
            f"elliptic residuals=({elliptic_hamiltonian:.3e}, "
            f"{elliptic_lapse:.3e}, {elliptic_momentum:.3e}), "
            f"ADM surface/volume=({adm_mass_surface:.8g}, "
            f"{adm_mass_volume:.8g}), errors=({adm_surface_volume_relative_error:.3e}, "
            f"{adm_reference_relative_error:.3e}), "
            f"proper/baryon/binding errors=({proper_mass_relative_error:.3e}, "
            f"{baryonic_mass_relative_error:.3e}, "
            f"{binding_energy_mass_normalized_error:.3e}), "
            f"Komar errors=({komar_volume_relative_error:.3e}, "
            f"{komar_surface_volume_relative_error:.3e}), "
            f"|P_ADM|max={adm_momentum_max:.3e}, "
            f"|J_ADM|max={adm_angular_momentum_max:.3e}"
         )
      results.append({
         "kind": "feature",
         "case": name,
         "status": "PASS",
         "metric": "hamiltonian_l2",
         "value": h_l2,
         "constraint_linf": max(h_linf, momentum_linf),
         "hamiltonian_l2": h_l2,
         "hamiltonian_linf": h_linf,
         "momentum_linf": momentum_linf,
         "relative_matter_update": relative_matter_update,
         "elliptic_hamiltonian_residual": elliptic_hamiltonian,
         "elliptic_lapse_residual": elliptic_lapse,
         "elliptic_momentum_residual": elliptic_momentum,
         "adm_mass_surface": adm_mass_surface,
         "adm_mass_volume": adm_mass_volume,
         "adm_surface_volume_relative_error": adm_surface_volume_relative_error,
         "adm_reference_relative_error": adm_reference_relative_error,
         "adm_angular_momentum_max": adm_angular_momentum_max,
         "adm_momentum_max": adm_momentum_max,
         "proper_mass_relative_error": proper_mass_relative_error,
         "baryonic_mass_relative_error": baryonic_mass_relative_error,
         "binding_energy_mass_normalized_error": binding_energy_mass_normalized_error,
         "komar_volume_relative_error": komar_volume_relative_error,
         "komar_surface_volume_relative_error": komar_surface_volume_relative_error,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   if args.adm_convergence:
      convergence_results, convergence_scalars = validate_adm_mass_convergence(
         args
      )
      results.extend(convergence_results)
      scalars.extend(convergence_scalars)
   return results, scalars


def validate_rotating_star_initial_data(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   common = {
      "amr.max_level": "0",
      "pyfluid.scalar_diagnostics": "standard",
   }
   zero_spin_common = {**common, "pyfluid.max_step": "0"}
   tov_dir, tov_scalar = run_dimension_case(
      "rotating_star_zero_spin_tov_control",
      ROTATING_STAR_INPUT,
      {
         **zero_spin_common,
         "pyfluid.ic": "tov",
         "initial_data.angular_velocity": "0.0",
      },
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="analytic",
   )
   zero_dir, zero_scalar = run_dimension_case(
      "rotating_star_zero_spin",
      ROTATING_STAR_INPUT,
      {**zero_spin_common, "initial_data.angular_velocity": "0.0"},
      args,
      dim=3,
      checks=SMOKE_POLICY,
      mhd=False,
      kind="analytic",
   )
   zero_tov = read_scalar_rows(tov_dir)[-1]
   zero_rotating = read_scalar_rows(zero_dir)[-1]
   equivalence_columns = (
      "mass", "mom1", "mom2", "mom3",
      "angular_mom1", "angular_mom2", "angular_mom3", "energy",
      "min_rho", "max_rho", "min_pressure", "min_lapse",
      "max_lorentz_factor", "z4c_hamiltonian_l2",
      "z4c_hamiltonian_linf", "z4c_momentum_linf",
   )
   zero_spin_error = max(
      abs(float(zero_rotating[name]) - float(zero_tov[name]))
      / max(abs(float(zero_tov[name])), 1.0)
      for name in equivalence_columns
   )
   if zero_spin_error > 1.0e-13:
      raise RuntimeError(
         "zero-spin rotating_star does not reproduce its direct TOV control: "
         f"maximum normalized scalar error={zero_spin_error:.3e}"
      )

   rotating_dir, rotating_scalar = run_dimension_case(
      "rotating_star_uniform_small_spin",
      ROTATING_STAR_INPUT,
      common,
      args,
      dim=3,
      checks=DYNAMIC_SPACETIME_POLICY,
      mhd=False,
      kind="feature",
   )
   rotating = read_scalar_rows(rotating_dir)[-1]
   run_log = (rotating_dir / "run.log").read_text()
   solve_match = re.search(
      r"xcts initial data: matter_iterations=([0-9]+) "
      r"relative_matter_update=([0-9.eE+-]+) coupled_iterations=([0-9]+) "
      r"hamiltonian_residual=([0-9.eE+-]+) "
      r"lapse_residual=([0-9.eE+-]+) momentum_residual=([0-9.eE+-]+)",
      run_log,
   )
   rotation_match = re.search(
      r"rotating star: omega=([0-9.eE+-]+) "
      r"maximum_density=([0-9.eE+-]+) "
      r"maximum_velocity=([0-9.eE+-]+) "
      r"injection_energy=([0-9.eE+-]+) "
      r"boundary_density_excess=([0-9.eE+-]+)",
      run_log,
   )
   if solve_match is None or rotation_match is None:
      raise RuntimeError("rotating-star run did not report its coupled solve")
   relative_update = abs(float(solve_match.group(2)))
   elliptic_residual = max(abs(float(solve_match.group(index))) for index in (4, 5, 6))
   maximum_velocity = float(rotation_match.group(3))
   boundary_density_excess = float(rotation_match.group(5))
   angular_momentum3 = float(rotating["angular_mom3"])
   transverse_momentum = max(
      abs(float(rotating[name]))
      for name in ("mom1", "mom2", "mom3", "angular_mom1", "angular_mom2")
   )
   hamiltonian_l2 = abs(float(rotating["z4c_hamiltonian_l2"]))
   hamiltonian_linf = abs(float(rotating["z4c_hamiltonian_linf"]))
   momentum_linf = abs(float(rotating["z4c_momentum_linf"]))
   rest_mass_drift = abs(float(rotating["mass_rel_drift"]))
   max_rho_relative_change = abs(float(rotating["max_rho_rel_change"]))
   if (
      relative_update > 1.0e-5
      or elliptic_residual > 1.0e-10
      or maximum_velocity <= 0.0
      or angular_momentum3 <= 0.0
      or transverse_momentum > 1.0e-10
      or boundary_density_excess != 0.0
      or not all(math.isfinite(value) for value in (
         hamiltonian_l2, hamiltonian_linf, momentum_linf,
      ))
      or hamiltonian_l2 > 1.0e-3
      or max(hamiltonian_linf, momentum_linf) > 1.0e-2
   ):
      raise RuntimeError(
         "uniform rotating-star initialization failed: "
         f"relative update={relative_update:.3e}, "
         f"elliptic residual={elliptic_residual:.3e}, "
         f"vmax={maximum_velocity:.3e}, J3={angular_momentum3:.3e}, "
         f"transverse momentum={transverse_momentum:.3e}, "
         f"boundary excess={boundary_density_excess:.3e}, "
         f"constraints=({hamiltonian_l2:.3e}, {hamiltonian_linf:.3e}, "
         f"{momentum_linf:.3e})"
      )
   return ([
      {
         "kind": "analytic",
         "case": "rotating_star_zero_spin_tov_equivalence",
         "status": "PASS",
         "metric": "zero_spin_relative_error",
         "value": zero_spin_error,
         "zero_spin_relative_error": zero_spin_error,
         "run_dir": repo_relative(zero_dir),
      },
      {
         "kind": "feature",
         "case": "rotating_star_uniform_small_spin",
         "status": "PASS",
         "metric": "relative_matter_update",
         "value": relative_update,
         "relative_matter_update": relative_update,
         "elliptic_residual": elliptic_residual,
         "maximum_velocity": maximum_velocity,
         "angular_momentum3": angular_momentum3,
         "hamiltonian_l2": hamiltonian_l2,
         "hamiltonian_linf": hamiltonian_linf,
         "momentum_linf": momentum_linf,
         "rest_mass_drift": rest_mass_drift,
         "max_rho_relative_change": max_rho_relative_change,
         "run_dir": repo_relative(rotating_dir),
      },
   ], [tov_scalar, zero_scalar, rotating_scalar])


def validate_adm_mass_convergence(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   reference_errors: list[float] = []
   surface_volume_errors: list[float] = []
   for base_cells in (16, 24, 32):
      name = f"tov_ctt_adm_n{base_cells}"
      run_dir, scalar = run_dimension_case(
         name,
         CONFORMAL_TOV_INPUT,
         {
            "amr.n_cell": f"{base_cells} {base_cells} {base_cells}",
            "amr.max_level": "0",
            "amr.initial_refine": "none",
            "spacetime.initial_data": "ctt",
            "pyfluid.scalar_diagnostics": "standard,global_gr",
         },
         args,
         dim=3,
         checks=SMOKE_POLICY,
         mhd=False,
         kind="analytic",
         remove_parameters=("amr.ref_ratio",),
      )
      final = read_scalar_rows(run_dir)[-1]
      run_log = (run_dir / "run.log").read_text()
      tov_mass = parse_tov_observables(run_log)["mass"]
      surface_mass = float(final["adm_mass_surface"])
      volume_mass = float(final["adm_mass_volume"])
      surface_volume_error = abs(surface_mass - volume_mass) / max(
         abs(surface_mass), abs(volume_mass), 1.0e-30
      )
      reference_error = abs(volume_mass - tov_mass) / max(abs(tov_mass), 1.0e-30)
      if (
         not math.isfinite(surface_mass)
         or not math.isfinite(volume_mass)
         or surface_volume_error > 2.0e-2
      ):
         raise RuntimeError(
            f"ADM mass quadratures disagree at n={base_cells}: "
            f"surface={surface_mass:.8g}, volume={volume_mass:.8g}, "
            f"relative error={surface_volume_error:.3e}"
         )
      reference_errors.append(reference_error)
      surface_volume_errors.append(surface_volume_error)
      results.append({
         "kind": "analytic",
         "case": name,
         "status": "PASS",
         "metric": "adm_reference_relative_error",
         "value": reference_error,
         "adm_mass_surface": surface_mass,
         "adm_mass_volume": volume_mass,
         "hierarchy": "uniform",
         "adm_surface_volume_relative_error": surface_volume_error,
         "adm_reference_relative_error": reference_error,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   if reference_errors[-1] >= 3.0e-2:
      raise RuntimeError(
         "constraint-assisted ADM mass is not plausible against the TOV mass: "
         + ", ".join(f"{error:.3e}" for error in reference_errors)
      )
   if not all(
      finer < coarser
      for coarser, finer in zip(
         surface_volume_errors, surface_volume_errors[1:]
      )
   ):
      raise RuntimeError(
         "ADM surface-volume disagreement does not decrease with resolution: "
         + ", ".join(f"{error:.3e}" for error in surface_volume_errors)
      )
   return results, scalars


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

   partial_amr_runs: dict[int, tuple[Path, dict[str, object], dict[str, str]]] = {}
   for cells in (16, 32):
      run_dir, scalar = run_dimension_case(
         f"ctt_density_cloud_amr_n{cells}",
         CTT_INPUT,
         {
            "amr.n_cell": f"{cells} {cells} {cells}",
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
      partial_amr_runs[cells] = (run_dir, scalar, read_scalar_rows(run_dir)[-1])

   amr_dir, amr_scalar, amr_final = partial_amr_runs[16]
   amr_l2 = abs(float(amr_final["z4c_hamiltonian_l2"]))
   amr_linf = abs(float(amr_final["z4c_hamiltonian_linf"]))
   amr_interior_l2 = abs(float(amr_final["z4c_hamiltonian_interior_l2"]))
   amr_cf_l2 = abs(float(amr_final["z4c_hamiltonian_cf_l2"]))
   amr_cf_linf = abs(float(amr_final["z4c_hamiltonian_cf_linf"]))
   refined_amr_dir, refined_amr_scalar, refined_amr_final = partial_amr_runs[32]
   refined_amr_l2 = abs(float(refined_amr_final["z4c_hamiltonian_l2"]))
   refined_amr_interior_l2 = abs(
      float(refined_amr_final["z4c_hamiltonian_interior_l2"])
   )
   refined_amr_cf_l2 = abs(float(refined_amr_final["z4c_hamiltonian_cf_l2"]))
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
   fine_values: dict[tuple[int, int, int], float] = {}
   for lo, hi, data in level_chunks(amr_plotfile, 1, 3, names=amr_names):
      for i in range(lo[0], hi[0] + 1):
         for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
               point = (i, j, k)
               fine_values[point] = float(
                  data[chi_component, i - lo[0], j - lo[1], k - lo[2]]
               )

   fine_errors: list[float] = []
   coarse_errors: list[float] = []
   for point, fine_value in fine_values.items():
      if all(index % 2 == 0 for index in point):
         continue
      # Exclude the two-point fourth-order coarse-fine stencil buffer from
      # this local-accuracy comparison. The interface contribution remains
      # visible in the dedicated Hamiltonian CF diagnostics.
      if not all(
         tuple(
            coordinate + (offset if axis == direction else 0)
            for axis, coordinate in enumerate(point)
         ) in fine_values
         for direction in range(3)
         for offset in (-2, 2)
      ):
         continue
      reference = reference_state[(chi_component, *point)]
      coarse_value = 0.0
      for di in (0, 1):
         wi = 1.0 - 0.5 * (point[0] % 2) if di == 0 else 0.5 * (point[0] % 2)
         for dj in (0, 1):
            wj = 1.0 - 0.5 * (point[1] % 2) if dj == 0 else 0.5 * (point[1] % 2)
            for dk in (0, 1):
               wk = 1.0 - 0.5 * (point[2] % 2) if dk == 0 else 0.5 * (point[2] % 2)
               coarse_value += wi * wj * wk * coarse_state[
                  (
                     chi_component,
                     point[0] // 2 + di,
                     point[1] // 2 + dj,
                     point[2] // 2 + dk,
                  )
               ]
      fine_errors.append(fine_value - reference)
      coarse_errors.append(coarse_value - reference)
   if not fine_errors:
      raise RuntimeError("compact CTT AMR comparison found no shared sample points")
   fine_solution_l2 = float(np.sqrt(np.mean(np.square(fine_errors))))
   coarse_solution_l2 = float(np.sqrt(np.mean(np.square(coarse_errors))))
   solution_ratio = fine_solution_l2 / coarse_solution_l2
   constraint_ratio = amr_interior_l2 / uniform_l2[16]
   if (
      not math.isfinite(amr_l2)
      or not math.isfinite(amr_interior_l2)
      or not math.isfinite(amr_cf_l2)
      or not math.isfinite(refined_amr_l2)
      or not math.isfinite(refined_amr_interior_l2)
      or not math.isfinite(refined_amr_cf_l2)
      or not math.isfinite(solution_ratio)
      or solution_ratio >= 0.75
      or constraint_ratio >= 4.0
      or refined_amr_l2 >= 0.75 * amr_l2
      or refined_amr_interior_l2 >= 0.75 * amr_interior_l2
   ):
      raise RuntimeError(
         "compact CTT AMR handoff failed: "
         f"solution ratio={solution_ratio:.3e}, "
         f"constraint ratio={constraint_ratio:.3e}, "
         f"composite convergence ratio={refined_amr_l2 / amr_l2:.3e}, "
         f"interior convergence ratio={refined_amr_interior_l2 / amr_interior_l2:.3e}"
      )
   results.append({
      "kind": "feature",
      "case": "ctt_density_cloud_amr_n16",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": amr_l2,
      "constraint_linf": amr_linf,
      "hamiltonian_l2": amr_l2,
      "hamiltonian_linf": amr_linf,
      "hamiltonian_interior_l2": amr_interior_l2,
      "hamiltonian_cf_l2": amr_cf_l2,
      "hamiltonian_cf_linf": amr_cf_linf,
      "amr_solution_ratio": solution_ratio,
      "amr_constraint_ratio": constraint_ratio,
      "levels": 2,
      "run_dir": repo_relative(amr_dir),
   })
   scalars.append(amr_scalar)
   results.append({
      "kind": "convergence",
      "case": "ctt_density_cloud_amr_n32",
      "status": "PASS",
      "metric": "hamiltonian_l2",
      "value": refined_amr_l2,
      "order": math.log(amr_l2 / refined_amr_l2) / math.log(2.0),
      "constraint_linf": abs(float(refined_amr_final["z4c_hamiltonian_linf"])),
      "hamiltonian_l2": refined_amr_l2,
      "hamiltonian_linf": abs(float(refined_amr_final["z4c_hamiltonian_linf"])),
      "hamiltonian_interior_l2": refined_amr_interior_l2,
      "hamiltonian_cf_l2": refined_amr_cf_l2,
      "hamiltonian_cf_linf": abs(
         float(refined_amr_final["z4c_hamiltonian_cf_linf"])
      ),
      "amr_cf_convergence_ratio": refined_amr_cf_l2 / amr_cf_l2,
      "levels": 2,
      "run_dir": repo_relative(refined_amr_dir),
   })
   scalars.append(refined_amr_scalar)
   return results, scalars


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "standard"), default="standard")
   parser.add_argument(
      "--only",
      choices=(
         "all", "ctt", "conformal_tov", "realistic_eos", "rotating_star",
         "tov_evolution",
      ),
      default="all",
   )
   parser.add_argument("--formulation", choices=("ctt", "cttk", "xcts"))
   parser.add_argument(
      "--adm-convergence",
      action="store_true",
      help="run the opt-in three-resolution CTT ADM mass convergence check",
   )
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=float, default=240.0)
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
      else (validate_conformal_tov_initial_data,)
      if args.only == "conformal_tov"
      else (validate_realistic_eos_tov,)
      if args.only == "realistic_eos"
      else (validate_rotating_star_initial_data,)
      if args.only == "rotating_star"
      else (validate_tov_evolution,)
      if args.only == "tov_evolution"
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
         validate_realistic_eos_tov,
         validate_tov_initial_data,
         validate_conformal_tov_initial_data,
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
      "relative_matter_update",
      "zero_spin_relative_error",
      "elliptic_residual",
      "maximum_velocity",
      "angular_momentum3",
      "elliptic_hamiltonian_residual",
      "elliptic_lapse_residual",
      "elliptic_momentum_residual",
      "hamiltonian_l2",
      "hamiltonian_linf",
      "momentum_linf",
      "adm_mass_surface",
      "adm_mass_volume",
      "adm_mass_surface_relative_change",
      "adm_mass_volume_relative_change",
      "adm_surface_volume_relative_error",
      "adm_reference_relative_error",
      "adm_momentum_max",
      "adm_angular_momentum_max",
      "proper_mass_relative_error",
      "baryonic_mass_relative_error",
      "binding_energy_mass_normalized_error",
      "komar_volume_relative_error",
      "komar_surface_volume_relative_error",
      "amr_solution_ratio",
      "amr_constraint_ratio",
      "fixed_metric_change",
      "rest_mass_drift",
      "external_mass_relative_error",
      "max_rho_relative_change",
      "analytic_error_subcycled",
      "analytic_error_single_rate",
      "levels",
      "stellar_radius",
      "run_dir",
   ]
   write_tsv(RESULTS, results, fields)
   append_validation_history(HISTORY, scalars)
   print(f"PASS z4c validation ({len(results)} checks): {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
