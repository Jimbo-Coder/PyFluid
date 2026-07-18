#!/usr/bin/env python3
"""Validate the fixed-Minkowski ideal-RMHD vertical slice."""

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
   parse_plot_header,
   parse_plot_header_1d,
   read_composite_state_1d,
   read_level_state,
)
from relativistic_validation import entropy_density_error
from validation_common import (
   ROOT,
   amrex_executable,
   append_validation_history,
   materialize_run_input,
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
   if float(results[-1]["order"]) < 1.8:
      raise RuntimeError(
         f"RMHD entropy wave did not retain second-order convergence: {results[-1]['order']}"
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
      },
      args,
   )
   amr_dir, amr_scalar = run_case(
      "entropy_amr_subcycling",
      INPUTS / "rmhd_entropy_wave_1d_amr.in",
      {"pyfluid.plot_int": "1000000"},
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
      {"pyfluid.plot_int": "1000000"},
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


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "standard"), default="standard")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if not amrex_executable(1).exists() or not CHECKPOINT_COMPARE.exists():
      raise SystemExit("build the AMReX solvers and 2D checkpoint comparator first")

   rows: list[dict[str, object]] = []
   scalar_rows: list[dict[str, object]] = []
   uniform, scalars = validate_uniform(args)
   rows.extend(uniform)
   scalar_rows.extend(scalars)
   convergence, scalars = validate_entropy_convergence(args)
   rows.extend(convergence)
   scalar_rows.extend(scalars)
   if args.level == "standard":
      shock, scalar = validate_shock(args)
      rows.append(shock)
      scalar_rows.append(scalar)
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

   write_tsv(
      RESULTS,
      rows,
      ["kind", "case", "status", "metric", "value", "order", "levels", "run_dir"],
   )
   append_validation_history(HISTORY, scalar_rows)
   for row in rows:
      detail = f" order={row.get('order')}" if row.get("order", "") != "" else ""
      print(f"PASS {row['kind']} {row['case']}: {row['metric']}={row['value']}{detail}")
   print(f"wrote {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
