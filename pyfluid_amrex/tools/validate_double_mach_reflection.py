#!/usr/bin/env python3
"""Validate Cartesian AMR on the Woodward-Colella double-Mach reflection."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from amr_plot_data import HierarchySlice
from postprocess_common import double_mach_reflection_budget_errors
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
   recon: str = "pwl"
   ng: int = 2
   recon_fine: str = ""
   recon_fine_start_level: int = -1


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

RECONSTRUCTION_CASES = (
   Case("recon_pwl", "64 16", 2, "on", "pwl", 2),
   Case("recon_wenoz", "64 16", 2, "on", "wenoz", 3),
   Case("recon_pwl_char_hybrid", "64 16", 2, "on", "pwl", 3, "wenoz-char", 2),
   Case("recon_wenoz_char", "64 16", 2, "on", "wenoz-char", 3),
   Case("recon_mp5", "64 16", 2, "on", "mp5", 3),
   Case("recon_mp5_char", "64 16", 2, "on", "mp5-char", 3),
   Case("recon_teno5", "64 16", 2, "on", "teno5", 3),
   Case("recon_teno5_char", "64 16", 2, "on", "teno5-char", 3),
   Case("recon_fine_reference", "512 128", 0, "off", "wenoz-char", 3),
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
   overrides = {
      "pyfluid.run_name": f"double_mach_reflection_{case.name}",
      "pyfluid.output_dir": str(run_dir),
      "pyfluid.final_time": f"{final_time:.17g}",
      "pyfluid.max_step": "100000",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "16",
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.subcycling": case.subcycling,
      "pyfluid.recon": case.recon,
      "pyfluid.ng": str(case.ng),
      "amr.n_cell": case.n_cell,
      "amr.max_level": str(case.max_level),
      "amr.ref_ratio": ratios,
      "amr.n_error_buf": buffers,
      "amr.max_grid_size": "32",
      "amr.blocking_factor": "8",
      "amr.total_cell_budget": "400000",
   }
   if case.recon_fine:
      overrides["pyfluid.recon_fine"] = case.recon_fine
      overrides["pyfluid.recon_fine_start_level"] = str(case.recon_fine_start_level)
   remove_parameters = []
   if case.max_level == 0:
      remove_parameters.append("amr.ref_ratio")
   if not case.recon_fine:
      remove_parameters.extend(("pyfluid.recon_fine", "pyfluid.recon_fine_start_level"))
   input_file = materialize_run_input(
      BASE_INPUT,
      run_dir / "run.in",
      overrides,
      remove_parameters=tuple(remove_parameters),
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
      mass_budget_error, energy_budget_error = double_mach_reflection_budget_errors(scalar_rows)
      if mass_budget_error > 0.01:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"DMR mass-budget relative error {mass_budget_error:.3e} exceeds 1e-2"
         )
      if energy_budget_error > 0.01:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"DMR energy-budget relative error {energy_budget_error:.3e} exceeds 1e-2"
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
      final_plotfile = latest_plotfile(run_dir, require=True)
      front_level = incident_shock_finest_level(final_plotfile)
      scalar_row["incident_shock_finest_level"] = front_level
      if front_level < case.max_level:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            f"incident shock is covered only through L{front_level}; expected L{case.max_level}"
         )
      finest_view_error, crop_view_error = hierarchy_view_errors(final_plotfile)
      scalar_row["hierarchy_finest_view_max_error"] = finest_view_error
      scalar_row["hierarchy_crop_view_max_error"] = crop_view_error
      if max(finest_view_error, crop_view_error) > 1.0e-13:
         scalar_row["status"] = "FAIL"
         scalar_row["reason"] = (
            "hierarchy renderer changed native finest data or cropped sampling: "
            f"finest={finest_view_error:.3e}, crop={crop_view_error:.3e}"
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


def hierarchy_view_errors(plotfile: Path) -> tuple[float, float]:
   view = HierarchySlice(plotfile, "rho", 1.4)
   tiles = list(view.tiles())
   tile_levels = {tile.level for tile in tiles}
   expected_levels = set(range(view.num_levels))
   if tile_levels != expected_levels:
      raise ValueError(
         f"hierarchy view exposed levels {sorted(tile_levels)}, expected {sorted(expected_levels)}"
      )

   finest_shape = view.level_shapes[-1]
   full = view.uniform(finest_shape[0], method="nearest")
   finest_error = 0.0
   for tile in tiles:
      if tile.level != view.num_levels - 1:
         continue
      rendered = full.values[tile.lo[1] : tile.hi[1] + 1, tile.lo[0] : tile.hi[0] + 1]
      finest_error = max(finest_error, float(np.max(np.abs(rendered - tile.values))))

   crop_index_lo = finest_shape[0] // 4
   crop_index_hi = 3 * finest_shape[0] // 4
   crop_lo = view.xlim[0] + (view.xlim[1] - view.xlim[0]) * crop_index_lo / finest_shape[0]
   crop_hi = view.xlim[0] + (view.xlim[1] - view.xlim[0]) * crop_index_hi / finest_shape[0]
   crop = HierarchySlice(
      plotfile,
      "rho",
      1.4,
      xlim=(crop_lo, crop_hi),
      ylim=view.ylim,
   ).uniform(max(crop_index_hi - crop_index_lo, finest_shape[1]), method="nearest")
   expected_crop = full.values[:, crop_index_lo:crop_index_hi]
   if crop.values.shape != expected_crop.shape:
      raise ValueError(
         f"cropped hierarchy shape {crop.values.shape} does not match expected {expected_crop.shape}"
      )
   crop_error = float(np.max(np.abs(crop.values - expected_crop)))

   for method in ("linear", "average"):
      sampled = view.uniform(257, method=method).values
      if not np.all(np.isfinite(sampled)):
         raise ValueError(f"{method} hierarchy view contains nonfinite pixels")
   return finest_error, crop_error


def sampled_field(
   run_dir: Path,
   component: str,
   *,
   max_pixels: int | None = None,
) -> np.ndarray:
   plotfile = latest_plotfile(run_dir, require=True)
   view = HierarchySlice(plotfile, component, 1.4)
   target_pixels = max_pixels or view.level_shapes[-1][view.dim0]
   return view.uniform(target_pixels, method="average").values


def finest_x_resolution(run_dir: Path) -> int:
   plotfile = latest_plotfile(run_dir, require=True)
   return parse_level_shapes(plotfile)[-1][0]


def relative_l1(candidate: np.ndarray, reference: np.ndarray) -> float:
   valid = np.isfinite(candidate) & np.isfinite(reference)
   if not np.any(valid):
      return math.inf
   scale = max(float(np.mean(np.abs(reference[valid]))), 1.0e-30)
   return float(np.mean(np.abs(candidate[valid] - reference[valid]))) / scale


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
   valid = np.isfinite(candidate) & np.isfinite(reference)
   if not np.any(valid):
      return math.inf
   numerator = float(np.sqrt(np.mean((candidate[valid] - reference[valid]) ** 2)))
   denominator = max(float(np.sqrt(np.mean(reference[valid] ** 2))), 1.0e-30)
   return numerator / denominator


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
   dy = 1.0 / field.shape[0]
   dx = 4.0 / field.shape[1]
   grad_y, grad_x = np.gradient(field, dy, dx, edge_order=1)
   return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def checkerboard_error(candidate: np.ndarray, reference: np.ndarray) -> float:
   error = candidate - reference
   mixed_difference = error[1:, 1:] - error[:-1, 1:] - error[1:, :-1] + error[:-1, :-1]
   scale = max(float(np.mean(np.abs(reference))), 1.0e-30)
   return float(np.mean(np.abs(mixed_difference))) / scale


def add_reference_metrics(rows: list[dict[str, object]]) -> list[str]:
   failures: list[str] = []
   by_name = {str(row["case"]): row for row in rows}
   fields = ("rho", "pressure")
   errors: dict[tuple[str, str], float] = {}
   for name in ("coarse_uniform", "amr_single_rate", "amr_subcycling", "equivalent_uniform"):
      target_pixels = finest_x_resolution(RUN_ROOT / name)
      for field in fields:
         candidate = sampled_field(RUN_ROOT / name, field, max_pixels=target_pixels)
         reference = sampled_field(RUN_ROOT / "fine_reference", field, max_pixels=target_pixels)
         error = relative_l1(candidate, reference)
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


def plot_reconstruction_comparison(
   images: dict[str, np.ndarray],
   reference: np.ndarray,
   output: Path,
) -> None:
   methods = ("recon_wenoz", "recon_wenoz_char", "recon_mp5", "recon_teno5_char")
   labels = ("WENO-Z component", "WENO-Z characteristic", "MP5 component", "TENO5 characteristic")
   errors = [np.abs(images[name] - reference) for name in methods]
   positive_errors = np.concatenate([error[error > 0.0] for error in errors])
   error_min = max(float(np.percentile(positive_errors, 1.0)), 1.0e-12)
   error_max = max(float(np.percentile(positive_errors, 99.5)), 10.0 * error_min)

   output.parent.mkdir(parents=True, exist_ok=True)
   fig, axes = plt.subplots(2, 4, figsize=(19, 6.8), constrained_layout=True)
   figure_image = None
   error_image = None
   for column, (name, label, error) in enumerate(zip(methods, labels, errors, strict=True)):
      figure_image = axes[0, column].imshow(
         images[name],
         origin="lower",
         extent=(0.0, 4.0, 0.0, 1.0),
         aspect="auto",
         interpolation="none",
         vmin=float(np.nanpercentile(reference, 0.5)),
         vmax=float(np.nanpercentile(reference, 99.5)),
         cmap="viridis",
      )
      axes[0, column].set_title(label)
      error_image = axes[1, column].imshow(
         np.maximum(error, error_min),
         origin="lower",
         extent=(0.0, 4.0, 0.0, 1.0),
         aspect="auto",
         interpolation="none",
         norm=LogNorm(vmin=error_min, vmax=error_max),
         cmap="magma",
      )
      axes[1, column].set_title("absolute density error")
      axes[1, column].set_xlabel("x")
      axes[0, column].set_ylabel("y")
      axes[1, column].set_ylabel("y")
   ny, nx = reference.shape
   fig.suptitle(
      f"Double Mach reflection: matched AMR reconstruction comparison "
      f"({nx}x{ny} finite-volume cell averages)"
   )
   fig.colorbar(figure_image, ax=axes[0, :], label="density", shrink=0.85)
   fig.colorbar(error_image, ax=axes[1, :], label="absolute error", shrink=0.85)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def add_reconstruction_metrics(rows: list[dict[str, object]]) -> tuple[list[str], float, float]:
   failures: list[str] = []
   by_name = {str(row["case"]): row for row in rows}
   method_names = (
      "recon_pwl",
      "recon_wenoz",
      "recon_pwl_char_hybrid",
      "recon_wenoz_char",
      "recon_mp5",
      "recon_mp5_char",
      "recon_teno5",
      "recon_teno5_char",
   )
   reference_dir = RUN_ROOT / "recon_fine_reference"
   target_pixels = min(finest_x_resolution(RUN_ROOT / name) for name in method_names)
   reference_fields = {
      field: sampled_field(reference_dir, field, max_pixels=target_pixels)
      for field in ("rho", "pressure", "v1", "v2")
   }
   candidate_fields = {
      name: {
         field: sampled_field(RUN_ROOT / name, field, max_pixels=target_pixels)
         for field in reference_fields
      }
      for name in method_names
   }
   reference_vorticity = (
      np.gradient(reference_fields["v2"], 4.0 / reference_fields["v2"].shape[1], axis=1)
      - np.gradient(reference_fields["v1"], 1.0 / reference_fields["v1"].shape[0], axis=0)
   )
   aggregate: dict[str, list[float]] = {name: [] for name in method_names}
   for name in method_names:
      row = by_name[name]
      for field in ("rho", "pressure"):
         candidate = candidate_fields[name][field]
         reference = reference_fields[field]
         l1 = relative_l1(candidate, reference)
         l2 = relative_l2(candidate, reference)
         gradient_l1 = relative_l1(gradient_magnitude(candidate), gradient_magnitude(reference))
         row[f"{field}_relative_l1"] = l1
         row[f"{field}_relative_l2"] = l2
         row[f"{field}_gradient_relative_l1"] = gradient_l1
         aggregate[name].extend((l1, l2, gradient_l1))
      candidate_vorticity = (
         np.gradient(candidate_fields[name]["v2"], 4.0 / reference_fields["v2"].shape[1], axis=1)
         - np.gradient(candidate_fields[name]["v1"], 1.0 / reference_fields["v1"].shape[0], axis=0)
      )
      vorticity_error = relative_l1(candidate_vorticity, reference_vorticity)
      row["vorticity_relative_l1"] = vorticity_error
      row["rho_checkerboard_error"] = checkerboard_error(
         candidate_fields[name]["rho"], reference_fields["rho"]
      )
      aggregate[name].append(vorticity_error)

   component_score = float(np.mean(aggregate["recon_wenoz"]))
   characteristic_score = float(np.mean(aggregate["recon_wenoz_char"]))
   hybrid_score = float(np.mean(aggregate["recon_pwl_char_hybrid"]))
   characteristic_ratio = characteristic_score / max(component_score, 1.0e-30)
   hybrid_ratio = hybrid_score / max(component_score, 1.0e-30)
   by_name["recon_wenoz_char"]["aggregate_error_vs_component"] = characteristic_ratio
   by_name["recon_pwl_char_hybrid"]["aggregate_error_vs_component"] = hybrid_ratio
   for name in ("recon_mp5", "recon_mp5_char", "recon_teno5", "recon_teno5_char"):
      by_name[name]["aggregate_error_vs_wenoz_component"] = (
         float(np.mean(aggregate[name])) / max(component_score, 1.0e-30)
      )
   plot_reconstruction_comparison(
      {name: candidate_fields[name]["rho"] for name in method_names},
      reference_fields["rho"],
      RUN_ROOT / "reconstruction_comparison.png",
   )
   return failures, characteristic_ratio, hybrid_ratio


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--level", choices=("quick", "reference", "reconstruction"), default="quick")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--final-time", type=float)
   args = parser.parse_args()

   if args.level == "quick":
      cases = QUICK_CASES
   elif args.level == "reference":
      cases = REFERENCE_CASES
   else:
      cases = RECONSTRUCTION_CASES
   default_final_time = {"quick": 0.03, "reference": 0.01, "reconstruction": 0.03}[args.level]
   final_time = args.final_time if args.final_time is not None else default_final_time
   rows = [run_case(case, args.ranks, args.force, final_time) for case in cases]
   failures = [f"{row['case']}: {row['reason']}" for row in rows if row["status"] != "PASS"]
   if args.level == "reference" and not failures:
      failures.extend(add_reference_metrics(rows))
   characteristic_ratio = math.nan
   hybrid_ratio = math.nan
   if args.level == "reconstruction" and not failures:
      reconstruction_failures, characteristic_ratio, hybrid_ratio = add_reconstruction_metrics(rows)
      failures.extend(reconstruction_failures)
   if failures:
      for row in rows:
         if row["status"] == "PASS":
            row["status"] = "FAIL"
            row["reason"] = "; ".join(failures)

   fields = sorted({key for row in rows for key in row})
   summary_path = {
      "quick": SUMMARY,
      "reference": RUN_ROOT / "reference_summary.tsv",
      "reconstruction": RUN_ROOT / "reconstruction_summary.tsv",
   }[args.level]
   write_tsv(summary_path, rows, fields)
   for row in rows:
      print(
         f"{row['case']}: {row['status']} t={row.get('time', '')} "
         f"L={row.get('amr_active_finest_level', '')} "
         f"Mcells/s={parse_float(row.get('avg_mcells_per_s')):.3f} "
         f"sim/s={parse_float(row.get('avg_sim_time_per_s')):.3e}"
      )
   if failures:
      raise SystemExit("FAIL: " + "; ".join(failures))
   if args.level == "reconstruction":
      decision = "material improvement" if characteristic_ratio <= 0.9 else "no material improvement"
      print(
         f"characteristic/component aggregate error ratio={characteristic_ratio:.4f}: {decision}"
      )
      print(f"level-hybrid/component aggregate error ratio={hybrid_ratio:.4f}")
   print(f"PASS: {repo_relative(summary_path)}")


if __name__ == "__main__":
   main()
