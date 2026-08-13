#!/usr/bin/env python3
"""Validate the fixed-Minkowski ideal RHD vertical slice."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import math
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
   CLOSED_POLICY,
   OPEN_POLICY,
   ROOT,
   amrex_executable,
   append_validation_history,
   build_cached_target,
   materialize_run_input,
   repo_relative,
   run_amrex,
   run_mpi,
   scalar_validation_row,
   write_tsv,
)


INPUTS = ROOT / "pyfluid_amrex/inputs"
RUN_ROOT = ROOT / "runs/current/rhd_validation"
RESULTS = RUN_ROOT / "rhd_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
CHECKPOINT_COMPARE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_1d"


def run_case(
   name: str,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   checks=CLOSED_POLICY,
   dim: int = 1,
) -> tuple[Path, dict[str, object]]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   parameters = {
      "pyfluid.run_name": name,
      "pyfluid.output_dir": repo_relative(run_dir),
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.aux_gf_int": "-1",
      **overrides,
   }
   input_path = materialize_run_input(base_input, run_dir / "input.in", parameters)
   if args.force or latest_plotfile(run_dir) is None:
      completed = run_amrex(dim=dim, input_path=input_path, ranks=args.ranks, timeout=args.timeout)
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=checks,
      mhd=False,
      dim=dim,
      ranks=args.ranks,
      kind="analytic" if checks == CLOSED_POLICY else "reference",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   return run_dir, scalar


def validate_uniform(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
   run_dir, scalar = run_case(
      "uniform",
      INPUTS / "rhd_uniform_1d.in",
      {"pyfluid.plot_int": "1000000"},
      args,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   names, _ = parse_plot_header_1d(plotfile)
   state = read_composite_state_1d(plotfile, names)
   spread = float(np.max(np.ptp(state, axis=1)))
   if spread > 64.0 * np.finfo(float).eps:
      raise RuntimeError(f"RHD free stream changed across cells: spread={spread:.3e}")
   return {
      "kind": "uniform",
      "case": "uniform",
      "status": "PASS",
      "metric": "max_component_spread",
      "value": spread,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_multidimensional_uniform(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   results = []
   scalars = []
   dimension_parameters = {
      2: {
         "geometry.prob_lo": "0.0 0.0",
         "geometry.prob_hi": "1.0 1.0",
         "geometry.is_periodic": "1 1",
         "amr.n_cell": "16 16",
         "amr.max_grid_size": "16",
      },
      3: {
         "geometry.prob_lo": "0.0 0.0 0.0",
         "geometry.prob_hi": "1.0 1.0 1.0",
         "geometry.is_periodic": "1 1 1",
         "amr.n_cell": "8 8 8",
         "amr.max_grid_size": "8",
      },
   }
   for dim, geometry in dimension_parameters.items():
      run_dir, scalar = run_case(
         f"uniform_{dim}d",
         INPUTS / "rhd_uniform_1d.in",
         {
            **geometry,
            "pyfluid.v1_0": "0.2",
            "pyfluid.v2_0": "-0.1",
            "pyfluid.v3_0": "0.05",
            "pyfluid.plot_int": "1000000",
         },
         args,
         dim=dim,
      )
      plotfile = latest_plotfile(run_dir, require=True)
      names, output_dim, _ = parse_plot_header(plotfile)
      if output_dim != dim:
         raise RuntimeError(f"{dim}D RHD run wrote a {output_dim}D plotfile")
      state = read_level_state(plotfile, names, dim, 0)
      spread = float(np.max(np.ptp(state.reshape(state.shape[0], -1), axis=1)))
      if spread > 64.0 * np.finfo(float).eps:
         raise RuntimeError(f"{dim}D RHD free stream changed across cells: spread={spread:.3e}")
      results.append({
         "kind": "uniform",
         "case": f"uniform_{dim}d",
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
         INPUTS / "rhd_entropy_wave_1d.in",
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
            "RHD entropy wave did not retain second-order convergence: "
            f"n={resolutions[index]} order={order:.3f}"
         )
   return results, scalars


def validate_shock(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
   run_dir, scalar = run_case(
      "sod_reference",
      INPUTS / "rhd_sod_x_1d.in",
      {"pyfluid.plot_int": "1000000"},
      args,
      checks=OPEN_POLICY,
   )
   return {
      "kind": "reference",
      "case": "sod_reference",
      "status": "PASS",
      "metric": "completed_physical",
      "value": 1,
      "run_dir": repo_relative(run_dir),
   }, scalar


def validate_amr(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
   base_dir, base_scalar = run_case(
      "entropy_uniform_n16",
      INPUTS / "rhd_entropy_wave_1d.in",
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
      INPUTS / "rhd_entropy_wave_1d_amr.in",
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
         f"RHD AMR did not provide the expected resolved hierarchy: levels={levels} improvement={improvement:.3f}"
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


def validate_restart(args: argparse.Namespace) -> dict[str, object]:
   common = {
      "pyfluid.max_step": "6",
      "pyfluid.final_time": "-1",
      "pyfluid.plot_int": "6",
      "pyfluid.checkpoint_int": "3",
      "pyfluid.scalar_int": "1",
   }
   full_dir, _ = run_case(
      "restart_full", INPUTS / "rhd_entropy_wave_1d_amr.in", common, args
   )
   checkpoint = full_dir / "checkpoints/chk00003"
   restart_dir, _ = run_case(
      "restart_from3",
      INPUTS / "rhd_entropy_wave_1d_amr.in",
      {**common, "pyfluid.restart_file": repo_relative(checkpoint)},
      args,
   )
   full_final = latest_checkpoint(full_dir)
   restart_final = latest_checkpoint(restart_dir)
   if full_final is None or restart_final is None:
      raise RuntimeError("RHD restart validation did not produce final checkpoints")
   compared = run_mpi(
      CHECKPOINT_COMPARE,
      [full_final, restart_final, "1e-12"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   if compared.returncode != 0:
      raise RuntimeError(compared.stdout)
   summary = next(
      line
      for line in compared.stdout.splitlines()
      if line.startswith("checkpoint max_dt_diff=")
   )
   return {
      "kind": "restart",
      "case": "amr_subcycling_restart",
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
   if args.level == "standard":
      build_cached_target(CHECKPOINT_COMPARE.name)
   if not amrex_executable(1).exists():
      raise SystemExit("build the AMReX 1D solver first")

   rows = []
   scalar_rows = []
   uniform, scalar = validate_uniform(args)
   rows.append(uniform)
   scalar_rows.append(scalar)
   convergence, scalars = validate_entropy_convergence(args)
   rows.extend(convergence)
   scalar_rows.extend(scalars)
   if args.level == "standard":
      multidimensional, scalars = validate_multidimensional_uniform(args)
      rows.extend(multidimensional)
      scalar_rows.extend(scalars)
      shock, scalar = validate_shock(args)
      rows.append(shock)
      scalar_rows.append(scalar)
      amr, scalars = validate_amr(args)
      rows.append(amr)
      scalar_rows.extend(scalars)
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
