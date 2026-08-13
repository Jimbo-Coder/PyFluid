#!/usr/bin/env python3
"""Validate the fixed-Minkowski ideal-RMHD vertical slice."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np

from exact_reference import cell_average_tabulated_mapped
from plotfile_reader import (
   latest_checkpoint,
   latest_plotfile,
   level_count,
   parse_plot_header,
   parse_plot_header_1d,
   parse_plot_bounds,
   read_composite_state_1d,
   read_level_state,
)
from relativistic_validation import entropy_density_error
from validation_common import (
   ROOT,
   amrex_executable,
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


INPUTS = ROOT / "pyfluid_amrex/inputs"
RUN_ROOT = ROOT / "runs/current/rmhd_validation"
RESULTS = RUN_ROOT / "rmhd_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
CHECKPOINT_COMPARE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_2d"
CT_AMR_DIVB_TOLERANCE = 1.0e-12
RMHD_COMPONENT_NAMES = (
   "dens",
   "mom1",
   "mom2",
   "mom3",
   "tau",
   "b1",
   "b2",
   "b3",
)
COMPONENT_RECONSTRUCTION_METHODS = (
   "pwc",
   "pwl",
   "wenoz",
   "mp5",
   "teno5",
   "parabolic",
   "ppm4",
   "ppmx",
   "ppm3",
)
EXACT_BALSARA_ROOT = ROOT / "exact_balsara"
EXACT_BALSARA_CASES = (
   {
      "case": 1,
      "time": 0.4,
      "gamma": 2.0,
      "left": (1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0),
      "right": (0.125, 0.1, 0.0, 0.0, 0.0, 0.5, -1.0, 0.0),
   },
   {
      "case": 2,
      "time": 0.4,
      "gamma": 5.0 / 3.0,
      "left": (1.0, 30.0, 0.0, 0.0, 0.0, 5.0, 6.0, 6.0),
      "right": (1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.7, 0.7),
   },
   {
      "case": 3,
      "time": 0.4,
      "gamma": 5.0 / 3.0,
      "left": (1.0, 1000.0, 0.0, 0.0, 0.0, 10.0, 7.0, 7.0),
      "right": (1.0, 0.1, 0.0, 0.0, 0.0, 10.0, 0.7, 0.7),
   },
   {
      "case": 4,
      "time": 0.4,
      "gamma": 5.0 / 3.0,
      "left": (1.0, 0.1, 0.999, 0.0, 0.0, 10.0, 7.0, 7.0),
      "right": (1.0, 0.1, -0.999, 0.0, 0.0, 10.0, -7.0, -7.0),
   },
   {
      "case": 5,
      "time": 0.55,
      "gamma": 5.0 / 3.0,
      "left": (1.08, 0.95, 0.4, 0.3, 0.2, 2.0, 0.3, 0.3),
      "right": (1.0, 1.0, -0.45, -0.2, 0.2, 2.0, -0.7, 0.5),
   },
)


def run_case(
   name: str,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   dim: int = 1,
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
      mhd=True,
      dim=dim,
      ranks=args.ranks,
      kind="reference",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   return run_dir, scalar


def _relativistic_recovery_summary(run_dir: Path) -> str:
   for line in reversed((run_dir / "run.log").read_text().splitlines()):
      if line.startswith("Relativistic recovery "):
         return line
   raise RuntimeError(
      f"{repo_relative(run_dir)} did not report relativistic recovery counts"
   )


def relativistic_recovery_counts(run_dir: Path) -> tuple[int, int]:
   match = re.search(
      r"c2p_fallbacks=(\d+) atmosphere_resets=(\d+)",
      _relativistic_recovery_summary(run_dir),
   )
   if match is None:
      raise RuntimeError(
         f"{repo_relative(run_dir)} did not report relativistic recovery counts"
      )
   return tuple(int(value) for value in match.groups())


def rmhd_candidate_correction_counts(run_dir: Path) -> tuple[int, int]:
   match = re.search(
      r"rmhd_pwl_corrections=(\d+) "
      r"rmhd_pwc_corrections=(\d+)",
      _relativistic_recovery_summary(run_dir),
   )
   if match is None:
      return 0, 0
   return tuple(int(value) for value in match.groups())


def rmhd_riemann_hll_fallback_count(run_dir: Path) -> int:
   match = re.search(
      r"rmhd_riemann_hll_fallbacks=(\d+)",
      _relativistic_recovery_summary(run_dir),
   )
   if match is None:
      return 0
   return int(match.group(1))


def validate_uniform(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   cases = (
      ("uniform_1d", INPUTS / "rmhd_uniform_1d.in", 1),
      ("uniform_2d_ct", INPUTS / "rmhd_uniform_2d_ct.in", 2),
   )
   results = []
   scalars = []
   for name, input_path, dim in cases:
      run_dir, scalar = run_case(
         name,
         input_path,
         {"pyfluid.plot_int": "1000000"},
         args,
         dim=dim,
      )
      plotfile = latest_plotfile(run_dir, require=True)
      names, output_dim, _ = parse_plot_header(plotfile)
      state = read_level_state(plotfile, names, output_dim, 0)
      spread = float(np.max(np.ptp(state.reshape(state.shape[0], -1), axis=1)))
      if spread > 64.0 * np.finfo(float).eps:
         raise RuntimeError(f"{name} free stream changed across cells: spread={spread:.3e}")
      results.append({
         "kind": "uniform",
         "case": name,
         "status": "PASS",
         "metric": "max_component_spread",
         "value": spread,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   return results, scalars


def validate_component_reconstruction_methods(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results = []
   scalars = []
   for method in COMPONENT_RECONSTRUCTION_METHODS:
      run_dir, scalar = run_case(
         f"recon_{method}",
         INPUTS / "rmhd_entropy_wave_1d.in",
         {
            "amr.n_cell": "32",
            "amr.max_grid_size": "32",
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.ng": "2" if method in {"pwc", "pwl"} else "3",
            "pyfluid.recon": method,
            "pyfluid.rmhd_candidate_guard": "pwc",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "4",
         },
         args,
      )
      fallback_count, reset_count = relativistic_recovery_counts(run_dir)
      pwl_count, pwc_count = rmhd_candidate_correction_counts(run_dir)
      if (
         fallback_count != 0
         or reset_count != 0
         or pwl_count != 0
         or pwc_count != 0
      ):
         raise RuntimeError(
            f"{method} smooth wave triggered a robustness intervention: "
            f"fallbacks={fallback_count} atmosphere_resets={reset_count} "
            f"pwl_corrections={pwl_count} pwc_corrections={pwc_count}"
         )
      results.append({
         "kind": "reconstruction",
         "case": f"component_{method}",
         "status": "PASS",
         "metric": "recovery_repairs",
         "value": 0,
         "rmhd_pwl_corrections": pwl_count,
         "rmhd_pwc_corrections": pwc_count,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   return results, scalars


def validate_entropy_convergence(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   resolutions = (32, 64) if args.level == "quick" else (32, 64, 128, 256)
   results = []
   scalars = []
   previous_error = None
   for cells in resolutions:
      run_dir, scalar = run_case(
         f"entropy_n{cells}",
         INPUTS / "rmhd_entropy_wave_1d.in",
         {
            "amr.n_cell": str(cells),
            "amr.max_grid_size": str(min(cells, 128)),
            "pyfluid.final_time": "0.2",
            "pyfluid.plot_int": "1000000",
         },
         args,
      )
      error = entropy_density_error(run_dir)
      order = "" if previous_error is None else math.log(previous_error / error, 2.0)
      results.append({
         "kind": "convergence",
         "case": f"entropy_n{cells}",
         "status": "PASS",
         "metric": "normalized_dens_l1",
         "value": error,
         "order": order,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
      previous_error = error
   order_indices = sorted({1, len(results) - 1})
   for index in order_indices:
      order = float(results[index]["order"])
      if order < 1.8:
         raise RuntimeError(
            "RMHD entropy wave did not retain second-order convergence: "
            f"n={resolutions[index]} order={order:.3f}"
         )
   return results, scalars


def validate_shock(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
   run_dir, scalar = run_case(
      "balsara1_reference",
      INPUTS / "rmhd_balsara1_1d.in",
      {"pyfluid.plot_int": "1000000"},
      args,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   names, _ = parse_plot_header_1d(plotfile)
   state = read_composite_state_1d(plotfile, names)
   normal_field_spread = float(np.ptp(state[names.index("b1")]))
   if normal_field_spread > 1.0e-12:
      raise RuntimeError(
         f"RMHD shock changed the normal magnetic field: spread={normal_field_spread:.3e}"
      )
   return {
      "kind": "reference",
      "case": "balsara1_reference",
      "status": "PASS",
      "metric": "normal_field_spread",
      "value": normal_field_spread,
      "run_dir": repo_relative(run_dir),
   }, scalar


def read_last_exact_balsara_profile(path: Path) -> np.ndarray:
   """Read the final nonlinear iteration from Giacomazzo's solution.dat."""
   blocks: list[list[list[float]]] = []
   current: list[list[float]] = []
   for line in path.read_text().splitlines():
      if line.lstrip().startswith("#iteration"):
         if current:
            blocks.append(current)
         current = []
      elif line.strip() and not line.lstrip().startswith("#"):
         current.append([float(value) for value in line.split()])
   if current:
      blocks.append(current)
   if not blocks or any(len(row) != 8 for row in blocks[-1]):
      raise RuntimeError(f"invalid exact RMHD profile: {path}")
   profile = np.asarray(blocks[-1], dtype=float)
   while np.any(np.diff(profile[:, 0]) < 0.0):
      reversal = int(np.flatnonzero(np.diff(profile[:, 0]) < 0.0)[0])
      lower_coordinate = profile[reversal + 1, 0]
      redundant = reversal
      while (
         redundant >= 0
         and profile[redundant, 0] > lower_coordinate
         and np.allclose(
            profile[redundant, 1:],
            profile[reversal + 1, 1:],
            rtol=64.0 * np.finfo(float).eps,
            atol=64.0 * np.finfo(float).eps,
         )
      ):
         redundant -= 1
      first_redundant = redundant + 1
      if first_redundant > reversal:
         raise RuntimeError(
            f"nontrivial out-of-order exact RMHD profile: {path}"
         )
      profile = np.delete(
         profile,
         np.s_[first_redundant : reversal + 1],
         axis=0,
      )
   return profile


def exact_rmhd_conserved(
   profile: np.ndarray,
   *,
   normal_magnetic: float,
   gamma: float,
) -> np.ndarray:
   """Convert exact primitive RMHD output to PyFluid's flat Valencia state."""
   rho = profile[:, 1]
   total_pressure = profile[:, 2]
   velocity = profile[:, 3:6]
   magnetic = np.column_stack((
      np.full(profile.shape[0], normal_magnetic),
      profile[:, 6],
      profile[:, 7],
   ))
   velocity_squared = np.sum(velocity * velocity, axis=1)
   if np.any(velocity_squared >= 1.0):
      raise RuntimeError("exact RMHD reference contains superluminal states")
   lorentz = 1.0 / np.sqrt(1.0 - velocity_squared)
   magnetic_squared = np.sum(magnetic * magnetic, axis=1)
   magnetic_velocity = np.sum(magnetic * velocity, axis=1)
   comoving_magnetic_squared = (
      magnetic_squared / (lorentz * lorentz) + magnetic_velocity**2
   )
   pressure = total_pressure - 0.5 * comoving_magnetic_squared
   if np.any(pressure <= 0.0):
      raise RuntimeError("exact RMHD reference has nonpositive gas pressure")
   enthalpy = 1.0 + gamma / (gamma - 1.0) * pressure / rho
   z_value = rho * enthalpy * lorentz**2
   dens = rho * lorentz
   momentum = (
      (z_value + magnetic_squared)[:, np.newaxis] * velocity
      - magnetic_velocity[:, np.newaxis] * magnetic
   )
   tau = (
      z_value
      + magnetic_squared
      - pressure
      - 0.5 * comoving_magnetic_squared
      - dens
   )
   return np.vstack((
      dens,
      momentum[:, 0],
      momentum[:, 1],
      momentum[:, 2],
      tau,
      magnetic[:, 0],
      magnetic[:, 1],
      magnetic[:, 2],
   ))


def exact_rmhd_cell_averages(
   profile: np.ndarray,
   *,
   normal_magnetic: float,
   gamma: float,
   cell_edges: np.ndarray,
) -> np.ndarray:
   """Average the exact primitive profile after its nonlinear Valencia map."""

   def transform(primitive: np.ndarray) -> np.ndarray:
      samples = primitive.shape[-1]
      mapped_profile = np.empty((samples, 8))
      mapped_profile[:, 0] = 0.0
      mapped_profile[:, 1:] = primitive.T
      return exact_rmhd_conserved(
         mapped_profile,
         normal_magnetic=normal_magnetic,
         gamma=gamma,
      )

   return cell_average_tabulated_mapped(
      profile[:, 0],
      profile[:, 1:].T,
      cell_edges,
      transform,
   )


def exact_balsara_comparison(
   case: int,
   spec: dict[str, object],
   run_dir: Path,
) -> tuple[
   np.ndarray,
   np.ndarray,
   np.ndarray,
   float,
   float,
   float,
]:
   """Compare conserved FV data with the exact Balsara cell averages."""
   plotfile = latest_plotfile(run_dir, require=True)
   names, output_time = parse_plot_header_1d(plotfile)
   if not math.isclose(
      output_time,
      float(spec["time"]),
      rel_tol=0.0,
      abs_tol=1.0e-12,
   ):
      raise RuntimeError(
         f"Balsara {case} stopped at t={output_time}, expected {spec['time']}"
      )
   state = read_composite_state_1d(plotfile, names)
   numerical = np.vstack([
      state[names.index(name)] for name in RMHD_COMPONENT_NAMES
   ])
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   exact_state = exact_rmhd_conserved(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
   )
   prob_lo, prob_hi = parse_plot_bounds(plotfile)
   spacing = (prob_hi[0] - prob_lo[0]) / numerical.shape[1]
   centers = prob_lo[0] + (np.arange(numerical.shape[1]) + 0.5) * spacing
   cell_edges = np.linspace(
      prob_lo[0],
      prob_hi[0],
      numerical.shape[1] + 1,
   )
   averaged_exact = exact_rmhd_cell_averages(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
      cell_edges=cell_edges,
   )
   scales = np.maximum(
      np.ptp(exact_state, axis=1),
      np.max(np.abs(exact_state), axis=1),
   )
   scales = np.maximum(scales, 1.0e-14)
   component_l1 = np.mean(np.abs(numerical - averaged_exact), axis=1) / scales

   exact_min = np.min(exact_state, axis=1)
   exact_max = np.max(exact_state, axis=1)
   lower_overshoot = np.maximum(exact_min[:, np.newaxis] - numerical, 0.0)
   upper_overshoot = np.maximum(numerical - exact_max[:, np.newaxis], 0.0)
   maximum_overshoot = float(
      np.max(np.maximum(lower_overshoot, upper_overshoot) / scales[:, np.newaxis])
   )

   numerical_tv = np.sum(np.abs(np.diff(numerical, axis=1)), axis=1)
   exact_tv = np.sum(np.abs(np.diff(averaged_exact, axis=1)), axis=1)
   active = exact_tv > 1.0e-12 * scales
   maximum_tv_ratio = float(np.max(numerical_tv[active] / exact_tv[active]))
   return (
      centers,
      numerical,
      averaged_exact,
      float(np.max(component_l1)),
      maximum_overshoot,
      maximum_tv_ratio,
   )


def validate_exact_balsara(
   args: argparse.Namespace,
   *,
   riemann_method: str = "hll",
   candidate_riemann_method: str = "hll",
   cases: tuple[dict[str, object], ...] = EXACT_BALSARA_CASES,
   cells: int = 512,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   component_names = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
   for spec in cases:
      case = int(spec["case"])
      left = spec["left"]
      right = spec["right"]
      overrides = {
         "geometry.prob_lo": "-1.0",
         "geometry.prob_hi": "1.0",
         "amr.n_cell": str(cells),
         "amr.max_grid_size": "128",
         "pyfluid.gamma": repr(spec["gamma"]),
         "pyfluid.final_time": repr(spec["time"]),
         "pyfluid.max_step": "10000",
         "pyfluid.ng": "3",
         "pyfluid.cfl": "0.9",
         "pyfluid.recon": "mp5",
         "pyfluid.riemann": riemann_method,
         "pyfluid.rmhd_candidate_guard": "pwc",
         "pyfluid.rmhd_candidate_riemann": candidate_riemann_method,
         "pyfluid.scalar_int": "64",
         "pyfluid.plot_int": "1000000",
      }
      for side, state in (("left", left), ("right", right)):
         for name, value in zip(
            ("rho", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
            state,
            strict=True,
         ):
            overrides[f"pyfluid.{name}_{side}"] = repr(value)
      run_name = f"balsara{case}_exact"
      if riemann_method != "hll":
         run_name = f"balsara{case}_{riemann_method}_exact"
      if candidate_riemann_method != "hll":
         run_name += f"_guard_{candidate_riemann_method}"
      run_dir, scalar = run_case(
         run_name,
         INPUTS / "rmhd_balsara1_1d.in",
         overrides,
         args,
      )
      plotfile = latest_plotfile(run_dir, require=True)
      names, output_time = parse_plot_header_1d(plotfile)
      if not math.isclose(output_time, float(spec["time"]), rel_tol=0.0, abs_tol=1.0e-12):
         raise RuntimeError(
            f"Balsara {case} stopped at t={output_time}, expected {spec['time']}"
         )
      state = read_composite_state_1d(plotfile, names)
      numerical = np.vstack([state[names.index(name)] for name in component_names])
      exact_profile = read_last_exact_balsara_profile(
         EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
      )
      exact_state = exact_rmhd_conserved(
         exact_profile,
         normal_magnetic=float(left[5]),
         gamma=float(spec["gamma"]),
      )
      prob_lo, prob_hi = parse_plot_bounds(plotfile)
      cell_edges = np.linspace(
         prob_lo[0],
         prob_hi[0],
         numerical.shape[1] + 1,
      )
      averaged_exact = exact_rmhd_cell_averages(
         exact_profile,
         normal_magnetic=float(left[5]),
         gamma=float(spec["gamma"]),
         cell_edges=cell_edges,
      )
      scales = np.maximum(
         np.ptp(exact_state, axis=1),
         np.max(np.abs(exact_state), axis=1),
      )
      scales = np.maximum(scales, 1.0e-14)
      component_errors = np.mean(
         np.abs(numerical - averaged_exact), axis=1
      ) / scales
      maximum_error = float(np.max(component_errors))
      if maximum_error > 0.03:
         details = ", ".join(
            f"{name}={error:.3e}"
            for name, error in zip(component_names, component_errors, strict=True)
         )
         raise RuntimeError(
            f"Balsara {case} exact-profile error={maximum_error:.3e}: {details}"
         )
      normal_field_spread = float(np.ptp(numerical[component_names.index("b1")]))
      if normal_field_spread > 1.0e-12:
         raise RuntimeError(
            f"Balsara {case} changed normal magnetic field: "
            f"spread={normal_field_spread:.3e}"
         )
      fallback_count, reset_count = relativistic_recovery_counts(run_dir)
      pwl_count, pwc_count = rmhd_candidate_correction_counts(run_dir)
      riemann_hll_fallbacks = rmhd_riemann_hll_fallback_count(run_dir)
      if fallback_count != 0 or reset_count != 0:
         raise RuntimeError(
            f"Balsara {case} required a post-update recovery repair: "
            f"fallbacks={fallback_count} atmosphere_resets={reset_count}"
         )
      results.append({
         "kind": "exact",
         "case": run_name,
         "status": "PASS",
         "metric": "maximum_normalized_component_l1",
         "value": maximum_error,
         "c2p_fallbacks": fallback_count,
         "atmosphere_resets": reset_count,
         "rmhd_pwl_corrections": pwl_count,
         "rmhd_pwc_corrections": pwc_count,
         "rmhd_riemann_hll_fallbacks": riemann_hll_fallbacks,
         "run_dir": repo_relative(run_dir),
      })
      scalars.append(scalar)
   return results, scalars


def validate_shock_ct_reference(
   args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
   resolutions = (128, 256, 512, 1024)
   profiles: dict[int, np.ndarray] = {}
   names: list[str] | None = None
   scalars = []
   last_run_dir = RUN_ROOT
   for cells in resolutions:
      run_dir, scalar = run_case(
         f"balsara1_2d_ct_n{cells}",
         INPUTS / "rmhd_balsara1_1d.in",
         {
            "geometry.prob_lo": "-0.5 0.0",
            "geometry.prob_hi": "0.5 1.0",
            "geometry.is_periodic": "0 1",
            "amr.n_cell": f"{cells} 8",
            "amr.max_grid_size": str(min(cells, 128)),
            "amr.blocking_factor": "8",
            "pyfluid.bc_xlo": "outflow",
            "pyfluid.bc_xhi": "outflow",
            "pyfluid.bc_ylo": "periodic",
            "pyfluid.bc_yhi": "periodic",
            "pyfluid.ct": "on",
            "pyfluid.plot_int": "1000000",
         },
         args,
         dim=2,
      )
      plotfile = latest_plotfile(run_dir, require=True)
      current_names, output_dim, _ = parse_plot_header(plotfile)
      state = read_level_state(plotfile, current_names, output_dim, 0)
      transverse_spread = float(np.max(np.ptp(state, axis=2)))
      if transverse_spread > 1024.0 * np.finfo(float).eps:
         raise RuntimeError(
            f"RMHD CT shock lost transverse uniformity at N={cells}: "
            f"spread={transverse_spread:.3e}"
         )
      if names is not None and current_names != names:
         raise RuntimeError("RMHD CT refinement cases use different state schemas")
      names = current_names
      profiles[cells] = np.mean(state, axis=2)
      scalars.append(scalar)
      last_run_dir = run_dir

   reference = profiles[resolutions[-1]]
   errors = []
   for cells in resolutions[:-1]:
      factor = resolutions[-1] // cells
      averaged_reference = reference.reshape(reference.shape[0], cells, factor).mean(axis=2)
      component_scale = np.maximum(np.ptp(averaged_reference, axis=1), 1.0)
      error = float(np.mean(
         np.abs(profiles[cells] - averaged_reference) / component_scale[:, np.newaxis]
      ))
      errors.append(error)
   improvements = [errors[index] / errors[index + 1] for index in range(len(errors) - 1)]
   if min(improvements) < 1.2:
      raise RuntimeError(
         f"RMHD CT shock did not improve toward the N=1024 reference: "
         f"errors={errors} improvements={improvements}"
      )
   return {
      "kind": "reference_ct",
      "case": "balsara1_2d_ct_refinement",
      "status": "PASS",
      "metric": "minimum_error_improvement",
      "value": min(improvements),
      "run_dir": repo_relative(last_run_dir),
   }, scalars


def validate_amr(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
   base_dir, base_scalar = run_case(
      "entropy_uniform_n16",
      INPUTS / "rmhd_entropy_wave_1d.in",
      {
         "amr.n_cell": "16",
         "amr.max_grid_size": "16",
         "pyfluid.final_time": "0.2",
         "pyfluid.plot_int": "1000000",
         "pyfluid.rmhd_candidate_guard": "pwc",
      },
      args,
   )
   amr_dir, amr_scalar = run_case(
      "entropy_amr_subcycling",
      INPUTS / "rmhd_entropy_wave_1d_amr.in",
      {
         "pyfluid.plot_int": "1000000",
         "pyfluid.rmhd_candidate_guard": "pwc",
      },
      args,
   )
   plotfile = latest_plotfile(amr_dir, require=True)
   names, _ = parse_plot_header_1d(plotfile)
   target_cells = read_composite_state_1d(plotfile, names).shape[1]
   base_error = entropy_density_error(base_dir, target_cells)
   amr_error = entropy_density_error(amr_dir, target_cells)
   levels = level_count(plotfile)
   improvement = base_error / amr_error
   if levels < 3 or improvement < 1.5:
      raise RuntimeError(
         f"RMHD AMR did not resolve the entropy mode: levels={levels} "
         f"improvement={improvement:.3f}"
      )
   return {
      "kind": "amr",
      "case": "entropy_amr_subcycling",
      "status": "PASS",
      "metric": "error_improvement_over_n16",
      "value": improvement,
      "levels": levels,
      "run_dir": repo_relative(amr_dir),
   }, [base_scalar, amr_scalar]


def validate_ct_amr(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
   run_dir, scalar = run_case(
      "entropy_2d_amr_ct_subcycling",
      INPUTS / "rmhd_entropy_wave_2d_amr_ct.in",
      {
         "pyfluid.ng": "3",
         "pyfluid.recon": "mp5",
         "pyfluid.rmhd_candidate_guard": "pwc",
         "pyfluid.plot_int": "1000000",
      },
      args,
      dim=2,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   levels = level_count(plotfile)
   if levels < 3 or float(scalar["divb_l1"]) > CT_AMR_DIVB_TOLERANCE:
      raise RuntimeError(
         f"RMHD CT AMR control failed: levels={levels} divb={scalar['divb_l1']}"
      )
   return {
      "kind": "amr_ct",
      "case": "entropy_2d_amr_ct_subcycling",
      "status": "PASS",
      "metric": "divb_l1",
      "value": scalar["divb_l1"],
      "levels": levels,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_restart(args: argparse.Namespace) -> dict[str, object]:
   common = {
      "pyfluid.max_step": "6",
      "pyfluid.final_time": "-1",
      "pyfluid.plot_int": "6",
      "pyfluid.checkpoint_int": "3",
      "pyfluid.scalar_int": "1",
   }
   full_dir, _ = run_case(
      "restart_full",
      INPUTS / "rmhd_entropy_wave_2d_amr_ct.in",
      common,
      args,
      dim=2,
   )
   checkpoint = full_dir / "checkpoints/chk00003"
   restart_dir, _ = run_case(
      "restart_from3",
      INPUTS / "rmhd_entropy_wave_2d_amr_ct.in",
      {**common, "pyfluid.restart_file": repo_relative(checkpoint)},
      args,
      dim=2,
   )
   full_final = latest_checkpoint(full_dir)
   restart_final = latest_checkpoint(restart_dir)
   if full_final is None or restart_final is None:
      raise RuntimeError("RMHD restart validation did not produce final checkpoints")
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [full_final, restart_final, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)
   summary = next(
      line for line in compared.stdout.splitlines() if line.startswith("checkpoint max_dt_diff=")
   )
   return {
      "kind": "restart",
      "case": "amr_ct_subcycling_restart",
      "status": "PASS",
      "metric": "checkpoint_compare",
      "value": summary,
      "run_dir": repo_relative(restart_dir),
   }


def validate_physical_id(
   args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
   writer_dir = RUN_ROOT / "physical_id_writer"
   artifact = writer_dir / "physical_id"
   common = {
      "pyfluid.max_step": "1",
      "pyfluid.final_time": "-1",
      "pyfluid.plot_int": "1",
      "pyfluid.scalar_int": "1",
      "pyfluid.checkpoint_int": "-1",
   }
   writer_dir, writer_scalar = run_case(
      "physical_id_writer",
      INPUTS / "rmhd_uniform_2d_ct.in",
      {
         **common,
         "initial_data.output_file": repo_relative(artifact),
      },
      args,
      dim=2,
   )
   reader_dir, reader_scalar = run_case(
      "physical_id_reader",
      INPUTS / "rmhd_uniform_2d_ct.in",
      {
         **common,
         "initial_data.input_file": repo_relative(artifact),
      },
      args,
      dim=2,
   )
   required_fields = {
      "Header",
      "Header.meta",
      "physical_fluid.h5",
      "adm_geometry.h5",
      "evolution_fluid_cache.h5",
      "evolution_spacetime_cache.h5",
      "face_b1.h5",
      "face_b2.h5",
   }
   missing = sorted(
      name for name in required_fields if not (artifact / name).is_file()
   )
   if missing:
      raise RuntimeError(
         "RMHD physical-ID artifact is incomplete: " + ", ".join(missing)
      )
   writer_rows = read_scalar_rows(writer_dir)
   reader_rows = read_scalar_rows(reader_dir)
   ignored = {"mcells_per_s", "layout_mcells_per_s", "sim_time_per_s"}
   comparable_writer = [
      {key: value for key, value in row.items() if key not in ignored}
      for row in writer_rows
   ]
   comparable_reader = [
      {key: value for key, value in row.items() if key not in ignored}
      for row in reader_rows
   ]
   if comparable_writer != comparable_reader:
      raise RuntimeError(
         "RMHD physical-ID direct evolution differs from fresh initialization"
      )
   writer_log = (writer_dir / "run.log").read_text()
   reader_log = (reader_dir / "run.log").read_text()
   if "wrote physical initial data:" not in writer_log:
      raise RuntimeError("RMHD physical-ID writer did not report its artifact")
   if "loaded physical initial data:" not in reader_log:
      raise RuntimeError("RMHD physical-ID reader did not report direct loading")
   materialized_scalars: list[dict[str, object]] = []
   writer_initial = writer_rows[0]
   extensive = ("mass", "mom1", "mom2", "mom3", "energy", "rmhd_electromagnetic_energy")
   materialized_cases = (
      ("physical_id_coarse", {"amr.n_cell": "8 8", "amr.max_grid_size": "8"}, 1.0),
      (
         "physical_id_fine_larger",
         {
            "amr.n_cell": "40 40",
            "amr.max_grid_size": "20",
            "geometry.prob_lo": "-0.5 -0.5",
            "geometry.prob_hi": "1.5 1.5",
         },
         4.0,
      ),
   )
   for case_name, grid_options, volume in materialized_cases:
      materialized_dir, materialized_scalar = run_case(
         case_name,
         INPUTS / "rmhd_uniform_2d_ct.in",
         {
            **common,
            **grid_options,
            "initial_data.input_file": repo_relative(artifact),
         },
         args,
         dim=2,
      )
      materialized_rows = read_scalar_rows(materialized_dir)
      initial = materialized_rows[0]
      for key in extensive:
         if not math.isclose(
            float(initial[key]) / volume,
            float(writer_initial[key]),
            rel_tol=2.0e-12,
            abs_tol=2.0e-13,
         ):
            raise RuntimeError(
               f"RMHD physical-ID {case_name} changed uniform {key} density"
            )
      if float(initial["divb_linf"]) > 1.0e-13:
         raise RuntimeError(
            f"RMHD physical-ID {case_name} violated the CT divergence contract"
         )
      materialized_log = (materialized_dir / "run.log").read_text()
      if "materialized physical initial data onto target hierarchy" not in materialized_log:
         raise RuntimeError(
            f"RMHD physical-ID {case_name} did not use cross-grid materialization"
         )
      materialized_scalars.append(materialized_scalar)
   return {
      "kind": "physical_id",
      "case": "rmhd_ct_direct_and_cross_grid_round_trip",
      "status": "PASS",
      "metric": "exact_rows_and_materialized_layouts",
      "value": f"{len(writer_rows)};2",
      "run_dir": repo_relative(reader_dir),
   }, [writer_scalar, reader_scalar, *materialized_scalars]


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "standard"), default="standard")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if args.level == "standard":
      build_cached_target(CHECKPOINT_COMPARE.name)
   if not amrex_executable(1).exists():
      raise SystemExit("build the AMReX solvers first")

   rows: list[dict[str, object]] = []
   scalar_rows: list[dict[str, object]] = []
   uniform, scalars = validate_uniform(args)
   rows.extend(uniform)
   scalar_rows.extend(scalars)
   reconstruction, scalars = validate_component_reconstruction_methods(args)
   rows.extend(reconstruction)
   scalar_rows.extend(scalars)
   convergence, scalars = validate_entropy_convergence(args)
   rows.extend(convergence)
   scalar_rows.extend(scalars)
   if args.level == "standard":
      shock, scalar = validate_shock(args)
      rows.append(shock)
      scalar_rows.append(scalar)
      exact, scalars = validate_exact_balsara(args)
      rows.extend(exact)
      scalar_rows.extend(scalars)
      ct_reference, scalars = validate_shock_ct_reference(args)
      rows.append(ct_reference)
      scalar_rows.extend(scalars)
      amr, scalars = validate_amr(args)
      rows.append(amr)
      scalar_rows.extend(scalars)
      ct_amr, scalar = validate_ct_amr(args)
      rows.append(ct_amr)
      scalar_rows.append(scalar)
      rows.append(validate_restart(args))
      physical_id, scalars = validate_physical_id(args)
      rows.append(physical_id)
      scalar_rows.extend(scalars)

   write_tsv(
      RESULTS,
      rows,
      [
         "kind",
         "case",
         "status",
         "metric",
         "value",
         "order",
         "levels",
         "c2p_fallbacks",
         "atmosphere_resets",
         "rmhd_pwl_corrections",
         "rmhd_pwc_corrections",
         "rmhd_riemann_hll_fallbacks",
         "run_dir",
      ],
   )
   append_validation_history(HISTORY, scalar_rows)
   for row in rows:
      detail = f" order={row.get('order')}" if row.get("order", "") != "" else ""
      print(f"PASS {row['kind']} {row['case']}: {row['metric']}={row['value']}{detail}")
   print(f"wrote {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
