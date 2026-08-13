#!/usr/bin/env python3
"""Compare HLL/HLLC/HLLD RMHD profiles in flat and curved spacetime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotfile_reader import (
   latest_plotfile,
   parse_plot_bounds,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from validate_rmhd import (
   EXACT_BALSARA_CASES,
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   exact_rmhd_cell_averages,
   read_last_exact_balsara_profile,
   relativistic_recovery_counts,
   rmhd_candidate_correction_counts,
   rmhd_riemann_hll_fallback_count,
   run_case,
)
from validation_common import ROOT, repo_relative, write_tsv


PLOT_ROOT = ROOT / "plots/current/scans"
FLAT_PLOT = PLOT_ROOT / "rmhd_riemann_flat_profiles.png"
CURVED_PLOT = PLOT_ROOT / "rmhd_riemann_schwarzschild_profiles.png"
RESULTS = RUN_ROOT / "rmhd_riemann_profile_comparison.tsv"
PAYLOAD = RUN_ROOT / "rmhd_riemann_profile_comparison.json"
STATE_NAMES = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
METHODS = ("hll", "hllc", "hlld")
METHOD_LABELS = {"hll": "HLL", "hllc": "HLLC", "hlld": "HLLD"}
METHOD_COLORS = {"hll": "#777777", "hllc": "#0072B2", "hlld": "#D55E00"}


def artifact_paths(scope: str) -> tuple[Path, Path]:
   if scope == "all":
      return RESULTS, PAYLOAD
   return (
      RESULTS.with_name(f"{RESULTS.stem}_{scope}{RESULTS.suffix}"),
      PAYLOAD.with_name(f"{PAYLOAD.stem}_{scope}{PAYLOAD.suffix}"),
   )


def centers_from_plotfile(plotfile: Path, cells: int) -> np.ndarray:
   prob_lo, prob_hi = parse_plot_bounds(plotfile)
   spacing = (prob_hi[0] - prob_lo[0]) / cells
   return prob_lo[0] + (np.arange(cells) + 0.5) * spacing


def read_state(run_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
   plotfile = latest_plotfile(run_dir, require=True)
   names, time = parse_plot_header_1d(plotfile)
   state = read_composite_state_1d(plotfile, names)
   ordered = np.vstack([state[names.index(name)] for name in STATE_NAMES])
   return centers_from_plotfile(plotfile, ordered.shape[1]), ordered, time


def exact_average(case: int, plotfile: Path, cells: int) -> np.ndarray:
   spec = next(
      item for item in EXACT_BALSARA_CASES if int(item["case"]) == case
   )
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   prob_lo, prob_hi = parse_plot_bounds(plotfile)
   return exact_rmhd_cell_averages(
      profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
      cell_edges=np.linspace(prob_lo[0], prob_hi[0], cells + 1),
   )


def normalized_l1(numerical: np.ndarray, reference: np.ndarray) -> float:
   scale = max(float(np.ptp(reference)), float(np.max(np.abs(reference))), 1.0e-14)
   return float(np.mean(np.abs(numerical - reference)) / scale)


def profile_payload(
   *,
   title: str,
   component: str,
   coordinate: np.ndarray,
   reference: np.ndarray,
   profiles: dict[str, np.ndarray],
   reference_label: str,
   method_references: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
   result = {
      "title": title,
      "component": component,
      "reference_label": reference_label,
      "x": np.round(coordinate, 10).tolist(),
      "reference": np.round(reference, 10).tolist(),
      "profiles": {
         method: np.round(profile, 10).tolist()
         for method, profile in profiles.items()
      },
   }
   if method_references is not None:
      result["method_references"] = {
         method: np.round(reference_profile, 10).tolist()
         for method, reference_profile in method_references.items()
      }
   return result


def flat_profiles() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   cases = (
      (1, "dens", r"$D$"),
      (3, "dens", r"$D$"),
      (4, "mom2", r"$S_2$"),
   )
   figure, axes = plt.subplots(
      len(cases), 2, figsize=(11.5, 8.2), constrained_layout=True
   )
   rows: list[dict[str, object]] = []
   payload: list[dict[str, object]] = []
   for row_index, (case, component, label) in enumerate(cases):
      run_dirs = {
         "hll": RUN_ROOT / f"balsara{case}_exact",
         "hllc": RUN_ROOT / f"balsara{case}_hllc_exact",
         "hlld": RUN_ROOT / f"balsara{case}_hlld_exact",
      }
      profiles: dict[str, np.ndarray] = {}
      coordinate = np.empty(0)
      output_time = 0.0
      for method, run_dir in run_dirs.items():
         method_coordinate, state, method_time = read_state(run_dir)
         if coordinate.size and not np.array_equal(coordinate, method_coordinate):
            raise RuntimeError(f"Balsara {case} method grids differ")
         coordinate = method_coordinate
         output_time = method_time
         profiles[method] = state[STATE_NAMES.index(component)]
      hll_plotfile = latest_plotfile(run_dirs["hll"], require=True)
      reference_state = exact_average(case, hll_plotfile, coordinate.size)
      reference = reference_state[STATE_NAMES.index(component)]

      profile_axis = axes[row_index, 0]
      error_axis = axes[row_index, 1]
      profile_axis.step(
         coordinate,
         reference,
         where="mid",
         color="black",
         linewidth=1.4,
         label="exact FV average",
      )
      for method in METHODS:
         error = np.abs(profiles[method] - reference)
         profile_axis.plot(
            coordinate,
            profiles[method],
            color=METHOD_COLORS[method],
            linewidth=1.05,
            label=METHOD_LABELS[method],
         )
         error_axis.semilogy(
            coordinate,
            np.maximum(error, 1.0e-16),
            color=METHOD_COLORS[method],
            linewidth=1.05,
            label=(
               f"{METHOD_LABELS[method]} "
               f"L1={normalized_l1(profiles[method], reference):.3e}"
            ),
         )
         rows.append({
            "geometry": "minkowski",
            "case": f"balsara{case}",
            "component": component,
            "method": method,
            "cells": coordinate.size,
            "reference": "exact_fv_average",
            "normalized_l1": normalized_l1(profiles[method], reference),
         })
      profile_axis.set_title(f"Balsara {case}, {label}, t={output_time:g}")
      error_axis.set_title(f"Balsara {case}, absolute {label} error")
      profile_axis.set_ylabel(label)
      error_axis.set_ylabel(r"$|U-U_{exact}|$")
      profile_axis.grid(alpha=0.22)
      error_axis.grid(alpha=0.22, which="both")
      if row_index == len(cases) - 1:
         profile_axis.set_xlabel("x")
         error_axis.set_xlabel("x")
      if row_index == 0:
         profile_axis.legend(frameon=False, ncol=2)
      error_axis.legend(frameon=False)
      payload.append(profile_payload(
         title=f"Flat SRMHD Balsara {case}: {component}",
         component=component,
         coordinate=coordinate,
         reference=reference,
         profiles=profiles,
         reference_label="exact FV average",
      ))
   figure.suptitle(
      "Flat SRMHD Riemann comparison — N=512, MP5/PWC guard, SSPRK3, CFL 0.9"
   )
   FLAT_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(FLAT_PLOT, dpi=180)
   plt.close(figure)
   return rows, payload


def run_curved_profile(
   args: argparse.Namespace,
   *,
   method: str,
   cells: int,
) -> Path:
   name = f"schwarzschild_balsara1_{method}_n{cells}"
   run_dir, _ = run_case(
      name,
      ROOT / "pyfluid_amrex/inputs/rmhd_balsara1_1d.in",
      {
         "geometry.prob_lo": "2.0",
         "geometry.prob_hi": "4.0",
         "amr.n_cell": str(cells),
         "amr.max_grid_size": "128",
         "spacetime.fixed_metric": "schwarzschild_isotropic",
         "spacetime.mass": "1.0",
         "spacetime.origin1": "0.0",
         "pyfluid.shock_x0": "3.0",
         "pyfluid.final_time": "0.15",
         "pyfluid.ng": "3",
         "pyfluid.cfl": "0.9",
         "pyfluid.recon": "mp5",
         "pyfluid.riemann": method,
         "pyfluid.rmhd_candidate_guard": "pwc",
         "pyfluid.rmhd_candidate_riemann": "hll",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.validation_policy": "smoke",
         "pyfluid.scalar_int": "64",
         "pyfluid.plot_int": "1000000",
      },
      args,
   )
   c2p_fallbacks, atmosphere_resets = relativistic_recovery_counts(run_dir)
   if c2p_fallbacks or atmosphere_resets:
      raise RuntimeError(
         f"{name} used recovery/reset={c2p_fallbacks}/{atmosphere_resets}"
      )
   return run_dir


def curved_profiles(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   coarse_cells = 512
   reference_cells = 2048
   run_dirs = {
      method: run_curved_profile(args, method=method, cells=coarse_cells)
      for method in METHODS
   }
   reference_dirs = {
      method: run_curved_profile(args, method=method, cells=reference_cells)
      for method in METHODS
   }
   factor = reference_cells // coarse_cells
   if factor * coarse_cells != reference_cells:
      raise RuntimeError("curved-profile reference ratio must be integral")
   references: dict[str, np.ndarray] = {}
   coordinate = np.empty(0)
   reference_time = 0.0
   for method, reference_dir in reference_dirs.items():
      reference_coordinate, fine_state, method_time = read_state(reference_dir)
      method_coordinate = reference_coordinate.reshape(
         coarse_cells, factor
      ).mean(axis=1)
      if coordinate.size and not np.allclose(
         method_coordinate, coordinate, rtol=0.0, atol=1.0e-13
      ):
         raise RuntimeError("curved-profile fine-reference grids differ")
      if coordinate.size and abs(method_time - reference_time) > 1.0e-12:
         raise RuntimeError("curved-profile fine-reference times differ")
      coordinate = method_coordinate
      reference_time = method_time
      references[method] = fine_state.reshape(
         fine_state.shape[0], coarse_cells, factor
      ).mean(axis=2)

   states: dict[str, np.ndarray] = {}
   for method, run_dir in run_dirs.items():
      method_coordinate, state, method_time = read_state(run_dir)
      if not np.allclose(method_coordinate, coordinate, rtol=0.0, atol=1.0e-13):
         raise RuntimeError("curved-profile method/reference grids differ")
      if abs(method_time - reference_time) > 1.0e-12:
         raise RuntimeError("curved-profile method/reference times differ")
      states[method] = state

   figure, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), constrained_layout=True)
   rows: list[dict[str, object]] = []
   payload: list[dict[str, object]] = []
   for row_index, (component, label) in enumerate((("dens", r"$D$"), ("b2", r"$B_2$"))):
      component_index = STATE_NAMES.index(component)
      component_reference = references["hlld"][component_index]
      component_profiles = {
         method: states[method][component_index] for method in METHODS
      }
      component_references = {
         method: references[method][component_index] for method in METHODS
      }
      profile_axis = axes[row_index, 0]
      error_axis = axes[row_index, 1]
      profile_axis.step(
         coordinate,
         component_reference,
         where="mid",
         color="black",
         linewidth=1.4,
         label="N=2048 HLLD reference",
      )
      for method in METHODS:
         method_profile = component_profiles[method]
         method_reference = component_references[method]
         error = np.abs(method_profile - method_reference)
         error_value = normalized_l1(method_profile, method_reference)
         profile_axis.plot(
            coordinate,
            method_profile,
            color=METHOD_COLORS[method],
            linewidth=1.05,
            label=METHOD_LABELS[method],
         )
         error_axis.semilogy(
            coordinate,
            np.maximum(error, 1.0e-16),
            color=METHOD_COLORS[method],
            linewidth=1.05,
            label=f"{METHOD_LABELS[method]} L1={error_value:.3e}",
         )
         pwl_count, pwc_count = rmhd_candidate_correction_counts(run_dirs[method])
         rows.append({
            "geometry": "schwarzschild_isotropic",
            "case": "balsara1",
            "component": component,
            "method": method,
            "cells": coarse_cells,
            "reference": f"{method}_n2048_self",
            "normalized_l1": error_value,
            "rmhd_pwl_corrections": pwl_count,
            "rmhd_pwc_corrections": pwc_count,
            "rmhd_riemann_hll_fallbacks": rmhd_riemann_hll_fallback_count(
               run_dirs[method]
            ),
         })
      profile_axis.set_title(f"Schwarzschild GRMHD {label}, t={reference_time:g}")
      error_axis.set_title(f"Absolute {label} error vs each method's N=2048 run")
      profile_axis.set_ylabel(label)
      error_axis.set_ylabel(r"$|U-U_{ref}|$")
      profile_axis.grid(alpha=0.22)
      error_axis.grid(alpha=0.22, which="both")
      if row_index == 1:
         profile_axis.set_xlabel("isotropic radius r")
         error_axis.set_xlabel("isotropic radius r")
      if row_index == 0:
         profile_axis.legend(frameon=False, ncol=2)
      error_axis.legend(frameon=False)
      payload.append(profile_payload(
         title=f"Fixed Schwarzschild GRMHD: {component}",
         component=component,
         coordinate=coordinate,
         reference=component_reference,
         profiles=component_profiles,
         reference_label="N=2048 HLLD reference",
         method_references=component_references,
      ))
   figure.suptitle(
      "Curved-background GRMHD Riemann comparison — N=512, MP5/PWC guard, "
      "SSPRK3, CFL 0.9"
   )
   CURVED_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(CURVED_PLOT, dpi=180)
   plt.close(figure)
   return rows, payload


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--only", choices=("all", "flat", "curved"), default="all")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=240)
   args = parser.parse_args()
   results_path, payload_path = artifact_paths(args.only)

   rows: list[dict[str, object]] = []
   payload: dict[str, list[dict[str, object]]] = {}
   if args.only in ("all", "flat"):
      flat_rows, flat_payload = flat_profiles()
      rows.extend(flat_rows)
      payload["flat"] = flat_payload
      print(f"wrote {repo_relative(FLAT_PLOT)}")
   if args.only in ("all", "curved"):
      curved_rows, curved_payload = curved_profiles(args)
      rows.extend(curved_rows)
      payload["curved"] = curved_payload
      print(f"wrote {repo_relative(CURVED_PLOT)}")
   write_tsv(
      results_path,
      rows,
      [
         "geometry",
         "case",
         "component",
         "method",
         "cells",
         "reference",
         "normalized_l1",
         "rmhd_pwl_corrections",
         "rmhd_pwc_corrections",
         "rmhd_riemann_hll_fallbacks",
      ],
   )
   payload_path.write_text(json.dumps(payload, separators=(",", ":")))
   print(f"wrote {repo_relative(results_path)}")
   print(f"wrote {repo_relative(payload_path)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
