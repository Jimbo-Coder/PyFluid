#!/usr/bin/env python3
"""Validate Cartesian AMR on the Woodward-Colella double-Mach reflection."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_amrex_boxed_movie import composite_slice
from plotfile_reader import (
   latest_plotfile,
   level_index_boxes,
   parse_level_shapes,
   parse_plot_bounds,
   parse_plot_header,
)
from validation_common import (
   ROOT,
   materialize_run_input,
   parse_float,
   read_tsv,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/double_mach_reflection_2d_amr.in"
RUN_ROOT = ROOT / "runs/current/double_mach_reflection_validation"
SUMMARY = RUN_ROOT / "summary.tsv"


@dataclass(frozen=True)
class Case:
   name: str
   n_cell: str
   max_level: int
   subcycling: str


QUICK_CASES = (
   Case("amr_subcycling", "64 16", 3, "on"),
)

REFERENCE_CASES = (
   Case("coarse_uniform", "64 16", 0, "off"),
   Case("amr_single_rate", "64 16", 2, "off"),
   Case("amr_subcycling", "64 16", 2, "on"),
   Case("equivalent_uniform", "256 64", 0, "off"),
   Case("fine_reference", "512 128", 0, "off"),
)


def run_case(case: Case, ranks: int, force: bool, final_time: float) -> dict[str, object]:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   if case.max_level > 0:
      ratios = " ".join("2" for _ in range(case.max_level))
      buffers = " ".join("2" for _ in range(case.max_level))
   else:
      ratios = "2"
      buffers = "2"
   input_file = materialize_run_input(
      BASE_INPUT,
      run_dir / "run.in",
      {
         "pyfluid.run_name": f"double_mach_reflection_{case.name}",
         "pyfluid.output_dir": str(run_dir),
         "pyfluid.final_time": f"{final_time:.17g}",
         "pyfluid.max_step": "100000",
         "pyfluid.scalar_int": "1",
         "pyfluid.plot_int": "16",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.subcycling": case.subcycling,
         "amr.n_cell": case.n_cell,
         "amr.max_level": str(case.max_level),
         "amr.ref_ratio": ratios,
         "amr.n_error_buf": buffers,
         "amr.max_grid_size": "32",
         "amr.blocking_factor": "8",
         "amr.total_cell_budget": "400000",
      },
      remove_parameters=("amr.ref_ratio",) if case.max_level == 0 else (),
   )
   result = run_amrex(dim=2, input_path=input_file, ranks=ranks, timeout=600)
   log_path = run_dir / "run.log"
   log_path.write_text("$ " + " ".join(result.command) + "\n" + result.stdout)
   scalar_row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=input_file,
      checks="open",
      dim=2,
      ranks=ranks,
      base_input=repo_relative(BASE_INPUT),
      kind="reference",
   )
   if result.returncode != 0:
      scalar_row["status"] = "FAIL"
      scalar_row["reason"] = f"solver return code {result.returncode}"
   scalar_rows = read_tsv(run_dir / "scalars.tsv")
   level_rows = read_tsv(run_dir / "amr_levels.tsv")
   level_steps: dict[int, int] = {}
   cell_updates = 0
   for level_row in level_rows:
      level = 0
      while f"steps_L{level}" in level_row:
         steps = int(float(level_row[f"steps_L{level}"]))
         cells = int(float(level_row[f"cells_L{level}"]))
         cell_updates += cells * (steps - level_steps.get(level, 0))
         level_steps[level] = steps
         level += 1
   mass_budget_error = math.nan
   energy_budget_error = math.nan
   if scalar_rows:
      initial_mass = parse_float(scalar_rows[0].get("mass"))
      final_mass = parse_float(scalar_rows[-1].get("mass"))
      initial_energy = parse_float(scalar_rows[0].get("energy"))
      final_energy = parse_float(scalar_rows[-1].get("energy"))
      time_value = parse_float(scalar_rows[-1].get("time"))
      rho_post = 8.0
      pressure_post = 116.5
      u_post = 7.144709581221619
      v_post = -4.125
      shock_intersection_speed = 20.0 / math.sqrt(3.0)
      top_minus_bottom_length_at_zero = 1.0 / math.sqrt(3.0)
      integrated_boundary_factor = (
         top_minus_bottom_length_at_zero * time_value
         + 0.5 * shock_intersection_speed * time_value * time_value
      )
      expected_mass_change = (
         rho_post * u_post * time_value
         - rho_post * v_post * integrated_boundary_factor
      )
      energy_post = (
         pressure_post / (1.4 - 1.0)
         + 0.5 * rho_post * (u_post * u_post + v_post * v_post)
      )
      enthalpy_density_post = energy_post + pressure_post
      expected_energy_change = (
         enthalpy_density_post * u_post * time_value
         - enthalpy_density_post * v_post * integrated_boundary_factor
      )
      measured_mass_change = final_mass - initial_mass
      measured_energy_change = final_energy - initial_energy
      mass_budget_error = abs(measured_mass_change - expected_mass_change) / max(abs(expected_mass_change), 1.0e-30)
      energy_budget_error = abs(measured_energy_change - expected_energy_change) / max(abs(expected_energy_change), 1.0e-30)
      if mass_budget_error > 0.05:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"DMR mass-budget relative error {mass_budget_error:.3e} exceeds 5e-2"
         )
      if energy_budget_error > 0.05:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"DMR energy-budget relative error {energy_budget_error:.3e} exceeds 5e-2"
         )
   scalar_row.update({
      "suite": "double_mach_reflection",
      "step": case.name,
      "returncode": result.returncode,
      "seconds": result.seconds,
      "log": repo_relative(log_path),
      "tail": " | ".join(result.stdout.splitlines()[-6:]),
      "cell_updates": cell_updates,
      "mass_budget_relative_error": mass_budget_error,
      "energy_budget_relative_error": energy_budget_error,
   })
   if case.max_level > 0 and result.returncode == 0:
      front_level = incident_shock_finest_level(latest_plotfile(run_dir, require=True))
      scalar_row["incident_shock_finest_level"] = front_level
      if front_level < case.max_level:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"incident shock is covered only through L{front_level}; expected L{case.max_level}"
         )
   return scalar_row


def incident_shock_finest_level(plotfile: Path) -> int:
   _names, _ndim, time_value = parse_plot_header(plotfile)
   problo, probhi = parse_plot_bounds(plotfile)
   level_shapes = parse_level_shapes(plotfile)
   finest = -1
   for level, shape in enumerate(level_shapes):
      for box_lo, box_hi in level_index_boxes(plotfile, level, 2):
         xlo = problo[0] + box_lo[0] * (probhi[0] - problo[0]) / shape[0]
         xhi = problo[0] + (box_hi[0] + 1) * (probhi[0] - problo[0]) / shape[0]
         ylo = problo[1] + box_lo[1] * (probhi[1] - problo[1]) / shape[1]
         yhi = problo[1] + (box_hi[1] + 1) * (probhi[1] - problo[1]) / shape[1]
         shock_xlo = 1.0 / 6.0 + (20.0 * time_value + ylo) / math.sqrt(3.0)
         shock_xhi = 1.0 / 6.0 + (20.0 * time_value + yhi) / math.sqrt(3.0)
         if max(xlo, min(shock_xlo, shock_xhi)) <= min(xhi, max(shock_xlo, shock_xhi)):
            finest = level
            break
   return finest


def sampled_field(run_dir: Path, component: str) -> np.ndarray:
   plotfile = latest_plotfile(run_dir, require=True)
   image, _boxes, _time, _ndim, _names, _xlim, _ylim = composite_slice(
      plotfile,
      component,
      1.4,
      2,
      0.5,
      False,
      None,
      512,
      (0.0, 4.0),
      (0.0, 1.0),
   )
   return image


def relative_l1(candidate: np.ndarray, reference: np.ndarray) -> float:
   valid = np.isfinite(candidate) & np.isfinite(reference)
   if not np.any(valid):
      return math.inf
   scale = max(float(np.mean(np.abs(reference[valid]))), 1.0e-30)
   return float(np.mean(np.abs(candidate[valid] - reference[valid]))) / scale


def add_reference_metrics(rows: list[dict[str, object]]) -> list[str]:
   failures: list[str] = []
   by_name = {str(row["case"]): row for row in rows}
   fields = ("rho", "pressure")
   references = {field: sampled_field(RUN_ROOT / "fine_reference", field) for field in fields}
   errors: dict[tuple[str, str], float] = {}
   for name in ("coarse_uniform", "amr_single_rate", "amr_subcycling", "equivalent_uniform"):
      for field in fields:
         error = relative_l1(sampled_field(RUN_ROOT / name, field), references[field])
         errors[(name, field)] = error
         by_name[name][f"{field}_relative_l1"] = error

   for field in fields:
      coarse_error = errors[("coarse_uniform", field)]
      amr_error = errors[("amr_single_rate", field)]
      subcycled_error = errors[("amr_subcycling", field)]
      equivalent_error = errors[("equivalent_uniform", field)]
      if not amr_error < coarse_error:
         failures.append(
            f"{field}: AMR error {amr_error:.3e} did not improve coarse error {coarse_error:.3e}"
         )
      if subcycled_error > 1.5 * amr_error:
         failures.append(
            f"{field}: subcycled error {subcycled_error:.3e} exceeds 1.5x single-rate {amr_error:.3e}"
         )
      if subcycled_error > 1.5 * equivalent_error:
         failures.append(
            f"{field}: AMR error {subcycled_error:.3e} exceeds 1.5x equivalent-uniform {equivalent_error:.3e}"
         )

   amr_levels = read_tsv(RUN_ROOT / "amr_subcycling" / "amr_levels.tsv")
   active_levels = [int(float(row.get("active_finest_level", "0"))) for row in amr_levels]
   if not active_levels or max(active_levels) < 2:
      failures.append("adaptive run did not exercise both configured fine levels")
   by_name["amr_subcycling"]["peak_active_level"] = max(active_levels, default=0)

   coarse_work = parse_float(by_name["coarse_uniform"].get("cell_updates"), 0.0)
   amr_work = parse_float(by_name["amr_subcycling"].get("cell_updates"), 0.0)
   equivalent_work = parse_float(by_name["equivalent_uniform"].get("cell_updates"), 0.0)
   by_name["amr_subcycling"]["work_vs_equivalent_uniform"] = (
      amr_work / equivalent_work if equivalent_work > 0.0 else math.nan
   )
   by_name["amr_subcycling"]["work_vs_coarse"] = amr_work / coarse_work if coarse_work > 0.0 else math.nan
   return failures


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--level", choices=("quick", "reference"), default="quick")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--final-time", type=float)
   args = parser.parse_args()

   cases = QUICK_CASES if args.level == "quick" else REFERENCE_CASES
   final_time = args.final_time if args.final_time is not None else (0.03 if args.level == "quick" else 0.01)
   rows = [run_case(case, args.ranks, args.force, final_time) for case in cases]
   failures = [f"{row['case']}: {row['reason']}" for row in rows if row["status"] != "PASS"]
   if args.level == "reference" and not failures:
      failures.extend(add_reference_metrics(rows))
   if failures:
      for row in rows:
         if row["status"] == "PASS":
            row["status"] = "FAIL"
            row["reason"] = "; ".join(failures)

   fields = sorted({key for row in rows for key in row})
   write_tsv(SUMMARY, rows, fields)
   for row in rows:
      print(
         f"{row['case']}: {row['status']} t={row.get('time', '')} "
         f"L={row.get('amr_active_finest_level', '')} "
         f"Mcells/s={parse_float(row.get('avg_mcells_per_s')):.3f} "
         f"sim/s={parse_float(row.get('avg_sim_time_per_s')):.3e}"
      )
   if failures:
      raise SystemExit("FAIL: " + "; ".join(failures))
   print(f"PASS: {repo_relative(SUMMARY)}")


if __name__ == "__main__":
   main()
