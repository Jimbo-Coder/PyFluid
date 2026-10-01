#!/usr/bin/env python3
"""Validate reference-metric GRHD balance, conservation and fallback."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))

from plotfile_reader import (  # noqa: E402
   latest_plotfile,
   level_index_boxes,
   parse_plot_bounds,
   parse_plot_header,
   read_level_state,
)
from validation_common import (  # noqa: E402
   append_validation_history,
   build_cached_target,
   materialize_run_input,
   read_key_values,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   scalar_checks,
   scalar_validation_row,
   write_tsv,
)


INPUTS = ROOT / "pyfluid_amrex" / "inputs"
RUN_ROOT = ROOT / "runs" / "current" / "rhd_reference_metric"
RESULTS = RUN_ROOT / "validation.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"

CASES = (
   (1, "rhd_spherical_uniform_static_1d.in", True),
   (2, "rhd_spherical_uniform_static_2d.in", True),
   (1, "rhd_spherical_radial_expansion_1d.in", False),
   (2, "rhd_spherical_radial_expansion_2d.in", False),
   (1, "rhd_spherical_schwarzschild_hse_1d.in", True),
   (2, "rhd_spherical_schwarzschild_hse_2d.in", True),
)

REJECTIONS = (
   (
      "amr_unvalidated",
      {"amr.max_level": "1"},
      "coarse-fine well-balanced momentum reflux is not yet validated",
   ),
   (
      "rmhd_unimplemented",
      {
         "pyfluid.physics": "rmhd",
         "initial_data.magnetic": "zero",
      },
      "relativistic-fluid runtime evolution supports a single Cartesian patch",
   ),
   (
      "unsupported_curved_metric",
      {
         "spacetime.fixed_metric": "rindler",
         "spacetime.acceleration1": "0.1",
      },
      "relativistic-fluid runtime evolution supports a single Cartesian patch",
   ),
   (
      "displaced_schwarzschild",
      {
         "spacetime.fixed_metric": "schwarzschild_isotropic",
         "spacetime.mass": "0.5",
         "spacetime.origin1": "0.1",
      },
      "requires spacetime.origin1/2/3 = 0",
   ),
   (
      "schwarzschild_horizon_crossing",
      {
         "spacetime.fixed_metric": "schwarzschild_isotropic",
         "spacetime.mass": "3.0",
      },
      "strictly outside r = M/2",
   ),
   (
      "stored_metric_unimplemented",
      {
         "spacetime.initial_data": "xcts",
      },
      "relativistic-fluid runtime evolution supports a single Cartesian patch",
   ),
)


def output_dir(input_path: Path) -> Path:
   configured = read_key_values(input_path)["pyfluid.output_dir"]
   return ROOT / configured


def run_case(dim: int, input_path: Path, force: bool, ranks: int, timeout: int) -> Path:
   run_dir = output_dir(input_path)
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   if not (run_dir / "scalars.tsv").is_file():
      result = run_amrex(
         dim=dim,
         input_path=input_path,
         ranks=ranks,
         timeout=timeout,
      )
      run_dir.mkdir(parents=True, exist_ok=True)
      (run_dir / "run.log").write_text(result.stdout)
      if result.returncode != 0:
         tail = "\n".join(result.stdout.splitlines()[-12:])
         raise RuntimeError(f"{input_path.name} failed:\n{tail}")
   return run_dir


def compare_radial_projection(
   run_1d: Path,
   run_2d: Path,
   *,
   label: str,
   inputs: str,
) -> dict[str, object]:
   plot_1d = latest_plotfile(run_1d, require=True)
   plot_2d = latest_plotfile(run_2d, require=True)
   assert plot_1d is not None and plot_2d is not None
   names_1d, dim_1d, time_1d = parse_plot_header(plot_1d)
   names_2d, dim_2d, time_2d = parse_plot_header(plot_2d)
   if names_1d != names_2d or dim_1d != 1 or dim_2d != 2:
      raise RuntimeError("reference-metric projection outputs have incompatible layouts")
   state_1d = read_level_state(plot_1d, names_1d, dim_1d, 0)
   state_2d = read_level_state(plot_2d, names_2d, dim_2d, 0)
   if state_2d.shape[1] != state_1d.shape[1]:
      raise RuntimeError("reference-metric projection radial grids do not match")
   time_error = abs(time_2d - time_1d)
   theta_spread = float(np.max(np.ptp(state_2d, axis=2)))
   projection_error = float(np.max(np.abs(state_2d - state_1d[:, :, None])))
   tolerance = 1.0e-12
   passed = max(time_error, theta_spread, projection_error) <= tolerance
   return {
      "kind": "reference_metric_rhd_projection",
      "case": label,
      "label": label,
      "status": "PASS" if passed else "FAIL",
      "reason": "" if passed else "dimensionally reduced GRHD states disagree",
      "dim": "1,2",
      "input": inputs,
      "run_dir": repo_relative(RUN_ROOT),
      "time": time_1d,
      "time_error": time_error,
      "theta_spread": theta_spread,
      "projection_linf": projection_error,
      "tolerance": tolerance,
   }


def validate_curved_scalar_metric_contract(
   run_1d: Path,
   run_2d: Path,
) -> dict[str, object]:
   rows_1d = read_scalar_rows(run_1d)
   rows_2d = read_scalar_rows(run_2d)
   if not rows_1d or not rows_2d:
      raise RuntimeError("curved reference-metric scalar output is empty")
   final_1d = rows_1d[-1]
   final_2d = rows_2d[-1]
   has_z4c_constraints = any(
      name.startswith("z4c_") for name in (*final_1d, *final_2d)
   )
   inner_cell_radius = 1.0 + 0.5 * (3.0 - 1.0) / 64.0
   compactness = 0.5 / (2.0 * inner_cell_radius)
   exact_min_lapse = (1.0 - compactness) / (1.0 + compactness)
   lapse_1d = float(final_1d["min_lapse"])
   lapse_2d = float(final_2d["min_lapse"])
   lapse_error = max(
      abs(lapse_1d - exact_min_lapse),
      abs(lapse_2d - exact_min_lapse),
      abs(lapse_1d - lapse_2d),
   )
   passed = not has_z4c_constraints and lapse_error <= 1.0e-14
   return {
      "kind": "reference_metric_rhd_diagnostic",
      "case": "spherical_schwarzschild_scalar_metric_contract",
      "label": "spherical_schwarzschild_scalar_metric_contract",
      "status": "PASS" if passed else "FAIL",
      "reason": "" if passed else (
         "curved scalar diagnostics used stored Cartesian Z4c semantics"
      ),
      "dim": "1,2",
      "input": (
         "rhd_spherical_schwarzschild_hse_1d.in,"
         "rhd_spherical_schwarzschild_hse_2d.in"
      ),
      "run_dir": repo_relative(RUN_ROOT),
      "z4c_constraints_present": int(has_z4c_constraints),
      "min_lapse_1d": lapse_1d,
      "min_lapse_2d": lapse_2d,
      "exact_min_lapse": exact_min_lapse,
      "min_lapse_error": lapse_error,
   }


def validate_rejections(ranks: int, timeout: int) -> list[dict[str, object]]:
   rows: list[dict[str, object]] = []
   base_input = INPUTS / "rhd_spherical_uniform_static_1d.in"
   rejection_inputs = RUN_ROOT / "rejection_inputs"
   for name, parameters, expected in REJECTIONS:
      input_path = materialize_run_input(
         base_input,
         rejection_inputs / f"{name}.in",
         {
            **parameters,
            "pyfluid.run_name": f"rhd_reference_metric_reject_{name}",
            "pyfluid.output_dir": repo_relative(RUN_ROOT / "rejections" / name),
         },
      )
      result = run_amrex(
         dim=1,
         input_path=input_path,
         ranks=ranks,
         timeout=timeout,
      )
      passed = result.returncode != 0 and expected in result.stdout
      rows.append({
         "kind": "reference_metric_rhd_rejection",
         "case": name,
         "label": name,
         "status": "PASS" if passed else "FAIL",
         "reason": "" if passed else "unsupported reference-metric branch was not rejected as specified",
         "dim": 1,
         "input": repo_relative(input_path),
         "ranks": ranks,
      })
   return rows


def schwarzschild_hse_error(run_dir: Path) -> float:
   plotfile = latest_plotfile(run_dir, require=True)
   assert plotfile is not None
   names, dim, _ = parse_plot_header(plotfile)
   if dim != 1 or names != ["dens", "mom1", "mom2", "mom3", "tau"]:
      raise RuntimeError("unexpected Schwarzschild HSE output layout")
   state = read_level_state(plotfile, names, dim, 0)
   lower, upper = parse_plot_bounds(plotfile)
   cells = state.shape[1]
   radius = lower[0] + (np.arange(cells) + 0.5) * (
      upper[0] - lower[0]
   ) / cells
   mass = 0.5
   gamma = 5.0 / 3.0
   rho0 = 1.0
   pressure0 = 0.1

   def lapse(r: np.ndarray | float) -> np.ndarray | float:
      compactness = mass / (2.0 * r)
      return (1.0 - compactness) / (1.0 + compactness)

   conformal_factor = 1.0 + mass / (2.0 * radius)
   sqrt_det_gamma = conformal_factor**6
   reference_enthalpy = 1.0 + gamma * pressure0 / ((gamma - 1.0) * rho0)
   enthalpy = reference_enthalpy * lapse(2.0) / lapse(radius)
   density = rho0 * (
      (enthalpy - 1.0) / (reference_enthalpy - 1.0)
   ) ** (1.0 / (gamma - 1.0))
   pressure = pressure0 * density**gamma
   exact = np.vstack((
      sqrt_det_gamma * density,
      np.zeros_like(radius),
      np.zeros_like(radius),
      np.zeros_like(radius),
      sqrt_det_gamma * pressure / (gamma - 1.0),
   ))
   # The generic reflecting boundary is not an analytic HSE ghost fill. Use
   # one resolution-independent physical shell so refinement does not move the
   # comparison toward that boundary.  The quarter-unit margins are four
   # coarse cells and exceed the one-step numerical domain of dependence.
   interior = (radius >= lower[0] + 0.25) & (radius <= upper[0] - 0.25)
   if not np.any(interior):
      raise RuntimeError("Schwarzschild HSE comparison shell contains no cells")
   scale = max(1.0, float(np.max(np.abs(exact[:, interior]))))
   return float(np.max(np.abs(state[:, interior] - exact[:, interior])) / scale)


def validate_schwarzschild_hse_convergence(
   force: bool,
   ranks: int,
   timeout: int,
) -> list[dict[str, object]]:
   base_input = INPUTS / "rhd_spherical_schwarzschild_hse_1d.in"
   input_root = RUN_ROOT / "convergence_inputs"
   errors: list[float] = []
   rows: list[dict[str, object]] = []
   for cells in (32, 64, 128):
      input_path = materialize_run_input(
         base_input,
         input_root / f"schwarzschild_hse_n{cells}.in",
         {
            "amr.n_cell": str(cells),
            "pyfluid.run_name": f"rhd_spherical_schwarzschild_hse_n{cells}",
            "pyfluid.output_dir": repo_relative(
               RUN_ROOT / "convergence" / f"n{cells}"
            ),
         },
      )
      run_dir = run_case(1, input_path, force, ranks, timeout)
      errors.append(schwarzschild_hse_error(run_dir))
      rows.append(scalar_validation_row(
         run_dir,
         case=f"schwarzschild_hse_n{cells}",
         input_file=input_path,
         checks=scalar_checks(
            "reference_metric_rhd_curved_convergence",
            energy=False,
            momentum=False,
            divb=False,
            mass_tol=1.0e-12,
         ),
         mhd=False,
         dim=1,
         ranks=ranks,
         base_input=base_input.name,
         kind="reference_metric_rhd_convergence_scalar",
      ))
   orders = [
      float(np.log(errors[index] / errors[index + 1]) / np.log(2.0))
      for index in range(len(errors) - 1)
   ]
   passed = all(np.isfinite(errors)) and all(
      errors[index + 1] < errors[index] for index in range(len(errors) - 1)
   ) and min(orders) >= 1.8
   rows.append({
      "kind": "reference_metric_rhd_convergence",
      "case": "spherical_schwarzschild_hse_convergence",
      "label": "spherical_schwarzschild_hse_convergence",
      "status": "PASS" if passed else "FAIL",
      "reason": "" if passed else "curved HSE interior error did not converge at second-order scale",
      "dim": 1,
      "input": base_input.name,
      "run_dir": repo_relative(RUN_ROOT / "convergence"),
      "ranks": ranks,
      "error_n32": errors[0],
      "error_n64": errors[1],
      "error_n128": errors[2],
      "order_32_64": orders[0],
      "order_64_128": orders[1],
   })
   return rows


def validate_spherical_admissibility(
   force: bool, ranks: int, timeout: int,
) -> list[dict[str, object]]:
   """Exercise shared-face fallback, not just a smooth curved flux operator."""
   rows = []
   for evolution in ("fixed", "evolved"):
      name = f"spherical_vacuum_{evolution}"
      parameters = {
         "mesh.coordinate": "spherical",
         "mesh.axes": "r",
         "geometry.coord_sys": "2",
         "geometry.prob_lo": "1",
         "geometry.prob_hi": "2",
         "pyfluid.shock_x0": "1.5",
         "amr.n_cell": "256",
         "spacetime.evolution": evolution,
         "pyfluid.debug_check_physical": "1",
         "pyfluid.scalar_int": "1",
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(RUN_ROOT / name),
      }
      if evolution == "evolved":
         parameters.update({
            "spacetime.formulation": "fccz4",
            "spacetime.initial_data": "minkowski",
            "spacetime.matter_coupling": "vacuum",
            "spacetime.lapse_gauge": "frozen",
            "spacetime.shift_gauge": "frozen",
            "spacetime.ko_dissipation": "0",
            "pyfluid.ng": "3",
         })
      input_path = materialize_run_input(
         INPUTS / "rhd_vacuum_expansion_1d.in",
         RUN_ROOT / "fallback_inputs" / f"{name}.in", parameters,
      )
      run_dir = run_case(1, input_path, force, ranks, timeout)
      # The disturbance cannot reach the boundaries by t=0.1. Conserved mass
      # must remain constant; energy is checked for the fixed Minkowski path.
      # The shared scalar checker also enforces the runtime boundary ledger.
      checks = scalar_checks(
         "spherical_admissibility", energy=evolution == "fixed",
         momentum=False, divb=False, mass_tol=1.0e-12, energy_tol=1.0e-12,
      )
      row = scalar_validation_row(
         run_dir, case=name, input_file=input_path, checks=checks,
         mhd=False, dim=1, ranks=ranks, kind="reference_metric_rhd_fallback",
      )
      counts = [int(value) for value in re.findall(
         r"RHD admissibility limiter: level=\d+ limited_cells=(\d+)",
         (run_dir / "run.log").read_text(),
      )]
      row["limited_cell_stages"] = sum(counts)
      row["limiter_events"] = len(counts)
      if not counts:
         row["status"] = "FAIL"
         row["reason"] = "control did not exercise admissibility fallback"
      rows.append(row)
   return rows


def validate_spherical_reflux(
   force: bool, ranks: int, timeout: int,
) -> dict[str, object]:
   """Check Valencia flux measures on a partial, fixed spherical hierarchy.

   A radial shell interface exercises geometric area weighting in full 3D.
   This is an isolated reflux control, not an adaptive-accuracy certificate.
   Reflecting boundaries close the baryon budget; local spherical momentum
   components are not conserved global Cartesian momenta.
   """
   name = "spherical_3d_reflux"
   final_time = 0.02
   cells = (32, 8, 16)
   parameters = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(RUN_ROOT / name),
      "mesh.coordinate": "spherical",
      "mesh.axes": "r theta phi",
      "geometry.coord_sys": "0",
      "geometry.prob_lo": "1 0.7 0",
      "geometry.prob_hi": f"2 2.1 {2 * math.pi}",
      "geometry.is_periodic": "0 0 1",
      "amr.n_cell": " ".join(map(str, cells)),
      "amr.max_grid_size": "16",
      "amr.blocking_factor": "4",
      "amr.max_level": "1",
      "amr.ref_ratio": "2",
      "amr.tag_metric": "none",
      "amr.initial_refine": "box",
      "amr.initial_refine_levels": "1",
      "amr.initial_refine_fixed_levels": "1",
      "amr.initial_refine_box_lo": "1.4 0.7 0",
      "amr.initial_refine_box_hi": f"1.6 2.1 {2 * math.pi}",
      "amr.n_error_buf": "1",
      "amr.total_cell_budget": "100000",
      "amr.regrid_int": "0",
      "amr.state_interp": "component-conservative",
      "spacetime.evolution": "evolved",
      "spacetime.formulation": "fccz4",
      "spacetime.initial_data": "minkowski",
      "spacetime.matter_coupling": "vacuum",
      "spacetime.lapse_gauge": "frozen",
      "spacetime.shift_gauge": "frozen",
      "spacetime.ko_dissipation": "0",
      "pyfluid.ic": "uniform",
      "pyfluid.v1_0": "0.1",
      "pyfluid.bc": "reflecting",
      "pyfluid.bc_zlo": "periodic",
      "pyfluid.bc_zhi": "periodic",
      "pyfluid.ng": "3",
      "pyfluid.subcycling": "on",
      "pyfluid.final_time": str(final_time),
      "pyfluid.max_step": "100000",
      "pyfluid.cfl": "0.25",
      "pyfluid.plot_int": "1000000",
      "pyfluid.scalar_int": "1",
      "pyfluid.debug_check_physical": "1",
   }
   input_path = materialize_run_input(
      INPUTS / "rhd_entropy_wave_1d.in",
      RUN_ROOT / "reflux_inputs" / f"{name}.in", parameters,
   )
   run_dir = run_case(3, input_path, force, ranks, timeout)
   row = scalar_validation_row(
      run_dir, case=name, input_file=input_path,
      checks=scalar_checks(
         "spherical_closed_reflux", momentum=False, energy=False,
         mass_tol=1.0e-12,
      ),
      mhd=False, dim=3, ranks=ranks, kind="reference_metric_rhd_reflux",
   )
   plotfile = latest_plotfile(run_dir, require=True)
   _, dim, output_time = parse_plot_header(plotfile)
   if dim != 3 or abs(output_time - final_time) > 1.0e-12:
      raise RuntimeError("spherical reflux control lacks its final 3D snapshot")
   fine_cells = sum(
      math.prod(upper - lower + 1 for lower, upper in zip(lo, hi))
      for lo, hi in level_index_boxes(plotfile, 1, dim)
   )
   row["fine_coverage_fraction"] = fine_cells / (8 * math.prod(cells))
   if not 0 < row["fine_coverage_fraction"] < 1:
      row["status"] = "FAIL"
      row["reason"] = "reflux control requires a partial fine hierarchy"
   return row


def main() -> int:
   parser = argparse.ArgumentParser()
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   args = parser.parse_args()

   write_tsv(RESULTS, [{"status": "INCOMPLETE"}], ["status"])
   build_cached_target("pyfluid_amrex_1d")
   build_cached_target("pyfluid_amrex_2d")
   build_cached_target("pyfluid_amrex_3d")
   scalar_rows: list[dict[str, object]] = []
   run_dirs: dict[str, Path] = {}
   for dim, filename, static in CASES:
      input_path = INPUTS / filename
      run_dir = run_case(dim, input_path, args.force, args.ranks, args.timeout)
      run_dirs[filename] = run_dir
      curved = "schwarzschild" in filename
      checks = scalar_checks(
         "reference_metric_rhd_curved_static" if curved
         else ("reference_metric_rhd_static" if static
               else "reference_metric_rhd_dynamic"),
         energy=not curved,
         momentum=static and not curved,
         divb=False,
         mass_tol=1.0e-12,
         energy_tol=1.0e-12,
         momentum_tol=1.0e-12,
      )
      scalar_rows.append(scalar_validation_row(
         run_dir,
         case=input_path.stem,
         input_file=input_path,
         checks=checks,
         mhd=False,
         dim=dim,
         ranks=args.ranks,
         base_input=filename,
         kind="reference_metric_rhd_scalar",
      ))

   projections = [
      compare_radial_projection(
         run_dirs["rhd_spherical_radial_expansion_1d.in"],
         run_dirs["rhd_spherical_radial_expansion_2d.in"],
         label="spherical_minkowski_r_to_rtheta",
         inputs=(
            "rhd_spherical_radial_expansion_1d.in,"
            "rhd_spherical_radial_expansion_2d.in"
         ),
      ),
      compare_radial_projection(
         run_dirs["rhd_spherical_schwarzschild_hse_1d.in"],
         run_dirs["rhd_spherical_schwarzschild_hse_2d.in"],
         label="spherical_schwarzschild_r_to_rtheta",
         inputs=(
            "rhd_spherical_schwarzschild_hse_1d.in,"
            "rhd_spherical_schwarzschild_hse_2d.in"
         ),
      ),
   ]
   convergence_rows = validate_schwarzschild_hse_convergence(
      args.force, args.ranks, args.timeout
   )
   diagnostic_rows = [validate_curved_scalar_metric_contract(
      run_dirs["rhd_spherical_schwarzschild_hse_1d.in"],
      run_dirs["rhd_spherical_schwarzschild_hse_2d.in"],
   )]
   fallback_rows = validate_spherical_admissibility(
      args.force, args.ranks, args.timeout
   )
   reflux_row = validate_spherical_reflux(args.force, args.ranks, args.timeout)
   rejection_rows = validate_rejections(args.ranks, args.timeout)
   rows = [
      *scalar_rows,
      *projections,
      *convergence_rows,
      *diagnostic_rows,
      *rejection_rows,
      *fallback_rows,
      reflux_row,
   ]
   fields = sorted({key for row in rows for key in row})
   write_tsv(RESULTS, rows, fields)
   append_validation_history(HISTORY, rows)
   for row in rows:
      print(
         f"{row['status']} {row['case']}"
         + (f": {row['reason']}" if row.get("reason") else "")
      )
   for projection in projections:
      print(
         f"{projection['label']}: "
         f"Linf={projection['projection_linf']:.3e} "
         f"theta_spread={projection['theta_spread']:.3e}"
      )
   return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
