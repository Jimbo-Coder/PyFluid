#!/usr/bin/env python3
"""Compare production reconstruction methods on exact SRMHD Balsara tests."""

from __future__ import annotations

import argparse
import math
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
   INPUTS,
   RUN_ROOT,
   exact_rmhd_cell_averages,
   exact_rmhd_conserved,
   read_last_exact_balsara_profile,
   relativistic_recovery_counts,
   run_case,
)
from validation_common import ROOT, read_scalar_rows, repo_relative, write_tsv


METHODS = ("pwc", "pwl", "wenoz", "mp5", "teno5", "ppm4", "ppmx", "ppm3")
METHOD_CHOICES = METHODS + (
   "parabolic",
   "wenoz-char",
   "mp5-char",
   "teno5-char",
   "parabolic-char",
   "ppm4-char",
   "ppmx-char",
   "ppm3-char",
)
COMPONENT_NAMES = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
PLOT_COMPONENTS = (
   ("dens", r"$D$"),
   ("mom1", r"$S_1$"),
   ("tau", r"$\tau$"),
   ("b2", r"$B_2$"),
)
RESULTS = RUN_ROOT / "balsara_reconstruction_scan.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_reconstruction_scan.png"
CONVERGENCE_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara_reconstruction_convergence.png"
)
PWL_BALSARA4_RESOLUTION_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara4_pwl_resolution.png"
)
WENOZ_FAILURE_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara4_wenoz_failure.png"
)


def exact_comparison(
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
      state[names.index(name)] for name in COMPONENT_NAMES
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


def run_scan(
   args: argparse.Namespace,
) -> tuple[
   list[dict[str, object]],
   dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]],
]:
   specs = {
      int(spec["case"]): spec
      for spec in EXACT_BALSARA_CASES
      if int(spec["case"]) in args.cases
   }
   rows: list[dict[str, object]] = []
   profiles: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
   for cells in args.cells:
      for case in args.cases:
         spec = specs[case]
         left = spec["left"]
         right = spec["right"]
         for method in args.methods:
            overrides = {
               "geometry.prob_lo": "-1.0",
               "geometry.prob_hi": "1.0",
               "amr.n_cell": str(cells),
               "amr.max_grid_size": "128",
               "pyfluid.gamma": repr(spec["gamma"]),
               "pyfluid.final_time": repr(spec["time"]),
               "pyfluid.max_step": "10000",
               "pyfluid.ng": "2" if method in {"pwc", "pwl"} else "3",
               "pyfluid.cfl": "0.9",
               "pyfluid.recon": method,
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
            run_dir, scalar = run_case(
               f"recon_scan_balsara{case}_{method}_n{cells}",
               INPUTS / "rmhd_balsara1_1d.in",
               overrides,
               args,
            )
            (
               centers,
               numerical,
               _,
               maximum_l1,
               maximum_overshoot,
               maximum_tv_ratio,
            ) = exact_comparison(case, spec, run_dir)
            fallback_count, reset_count = relativistic_recovery_counts(run_dir)
            scalar_rows = read_scalar_rows(run_dir)
            rows.append({
               "case": case,
               "method": method,
               "cells": cells,
               "maximum_normalized_component_l1": maximum_l1,
               "maximum_normalized_global_overshoot": maximum_overshoot,
               "maximum_component_tv_ratio": maximum_tv_ratio,
               "c2p_fallbacks": fallback_count,
               "atmosphere_resets": reset_count,
               "mass_rel_drift": scalar["mass_rel_drift"],
               "energy_rel_drift": scalar["energy_rel_drift"],
               "mcells_per_s": scalar_rows[-1]["mcells_per_s"],
               "run_dir": repo_relative(run_dir),
            })
            profiles[(case, method, cells)] = (centers, numerical)
            print(
               f"Balsara {case} {method} N={cells}: L1={maximum_l1:.6e} "
               f"overshoot={maximum_overshoot:.6e} TV={maximum_tv_ratio:.4f} "
               f"resets={reset_count}"
            )
   return rows, profiles


def plot_scan(
   args: argparse.Namespace,
   profiles: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]],
) -> None:
   plot_cells = max(args.cells)
   specs = {int(spec["case"]): spec for spec in EXACT_BALSARA_CASES}
   plot_cases = [case for case in args.cases if case in {1, 4}]
   if not plot_cases:
      plot_cases = args.cases[:2]
   figure, axes = plt.subplots(
      len(plot_cases),
      len(PLOT_COMPONENTS),
      figsize=(13.5, 3.0 * len(plot_cases)),
      sharex=True,
      constrained_layout=True,
      squeeze=False,
   )
   colors = {
      "pwc": "#999999",
      "pwl": "#009E73",
      "wenoz": "#CC79A7",
      "mp5": "#0072B2",
      "teno5": "#D55E00",
      "parabolic": "#E69F00",
      "ppm4": "#56B4E9",
      "ppmx": "#000000",
      "ppm3": "#F0E442",
      "wenoz-char": "#CC79A7",
      "mp5-char": "#0072B2",
      "teno5-char": "#D55E00",
      "parabolic-char": "#E69F00",
      "ppm4-char": "#56B4E9",
      "ppmx-char": "#000000",
      "ppm3-char": "#F0E442",
   }
   line_styles = {
      "pwc": "-",
      "pwl": "-.",
      "wenoz": "--",
      "mp5": "-",
      "teno5": ":",
      "parabolic": "--",
      "ppm4": "-",
      "ppmx": "-.",
      "ppm3": "--",
      "wenoz-char": ":",
      "mp5-char": ":",
      "teno5-char": "--",
      "parabolic-char": ":",
      "ppm4-char": "--",
      "ppmx-char": ":",
      "ppm3-char": ":",
   }
   display_names = {
      "pwc": "PWC",
      "pwl": "PWL",
      "wenoz": "WENO-Z",
      "mp5": "MP5",
      "teno5": "TENO5",
      "parabolic": "parabolic",
      "ppm4": "PPM4",
      "ppmx": "PPMX",
      "ppm3": "PPM3",
      "wenoz-char": "WENO-Z char",
      "mp5-char": "MP5 char",
      "teno5-char": "TENO5 char",
      "parabolic-char": "parabolic char",
      "ppm4-char": "PPM4 char",
      "ppmx-char": "PPMX char",
      "ppm3-char": "PPM3 char",
   }
   for row, case in enumerate(plot_cases):
      spec = specs[case]
      exact_profile = read_last_exact_balsara_profile(
         EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
      )
      exact_state = exact_rmhd_conserved(
         exact_profile,
         normal_magnetic=float(spec["left"][5]),
         gamma=float(spec["gamma"]),
      )
      reference_centers = profiles[(case, args.methods[0], plot_cells)][0]
      spacing = reference_centers[1] - reference_centers[0]
      cell_edges = np.linspace(
         reference_centers[0] - 0.5 * spacing,
         reference_centers[-1] + 0.5 * spacing,
         reference_centers.size + 1,
      )
      exact_average = exact_rmhd_cell_averages(
         exact_profile,
         normal_magnetic=float(spec["left"][5]),
         gamma=float(spec["gamma"]),
         cell_edges=cell_edges,
      )
      for column, (name, label) in enumerate(PLOT_COMPONENTS):
         component = COMPONENT_NAMES.index(name)
         axis = axes[row, column]
         axis.plot(
            exact_profile[:, 0],
            exact_state[component],
            color="0.65",
            linewidth=1.0,
            label="exact point profile",
            zorder=9,
         )
         axis.step(
            reference_centers,
            exact_average[component],
            where="mid",
            color="black",
            linewidth=1.6,
            label="exact FV average",
            zorder=10,
         )
         plotted_methods = (
            tuple(method for method in args.methods if method != "wenoz")
            if case == 4
            else args.methods
         )
         for method in plotted_methods:
            centers, numerical = profiles[(case, method, plot_cells)]
            axis.plot(
               centers,
               numerical[component],
               color=colors[method],
               linestyle=line_styles[method],
               linewidth=1.25,
               alpha=0.95,
               label=display_names[method],
            )
         axis.set_title(f"Balsara {case}: {label}")
         axis.grid(alpha=0.22)
         if column == 0:
            axis.set_ylabel(f"t = {float(spec['time']):g}")
         if row == len(plot_cases) - 1:
            axis.set_xlabel("x")
   axes[0, 0].legend(
      frameon=False,
      ncol=2,
      fontsize=8,
      loc="best",
   )
   figure.suptitle(
      f"SRMHD Valencia conserved comparison — "
      f"N={plot_cells}, HLL/SSPRK3, CFL 0.9; "
      "Balsara 4 WENO-Z is off scale"
   )
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)

   if 4 not in args.cases or "wenoz" not in args.methods:
      return
   case = 4
   spec = specs[case]
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / "solution_b4.dat"
   )
   exact_state = exact_rmhd_conserved(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
   )
   centers, numerical = profiles[(case, "wenoz", plot_cells)]
   spacing = centers[1] - centers[0]
   exact_average = exact_rmhd_cell_averages(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
      cell_edges=np.linspace(
         centers[0] - 0.5 * spacing,
         centers[-1] + 0.5 * spacing,
         centers.size + 1,
      ),
   )
   failure_figure, failure_axes = plt.subplots(
      1,
      len(PLOT_COMPONENTS),
      figsize=(13.5, 3.2),
      constrained_layout=True,
   )
   for column, (name, label) in enumerate(PLOT_COMPONENTS):
      component = COMPONENT_NAMES.index(name)
      axis = failure_axes[column]
      axis.plot(
         exact_profile[:, 0],
         exact_state[component],
         color="0.65",
         linewidth=1.0,
         label="exact point profile",
      )
      axis.step(
         centers,
         exact_average[component],
         where="mid",
         color="black",
         linewidth=1.6,
         label="exact FV average",
      )
      axis.plot(
         centers,
         numerical[component],
         color=colors["wenoz"],
         linestyle=line_styles["wenoz"],
         linewidth=1.25,
         label="WENO-Z",
      )
      axis.set_title(label)
      axis.set_xlabel("x")
      axis.grid(alpha=0.22)
   failure_axes[0].set_ylabel(f"t = {float(spec['time']):g}")
   failure_axes[0].legend(frameon=False)
   failure_figure.suptitle(
      f"Balsara 4 WENO-Z off-scale failure — N={plot_cells}, CFL 0.9"
   )
   WENOZ_FAILURE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   failure_figure.savefig(WENOZ_FAILURE_PLOT, dpi=180)
   plt.close(failure_figure)


def plot_convergence(
   args: argparse.Namespace,
   rows: list[dict[str, object]],
) -> None:
   if len(args.cells) < 2:
      return
   columns = min(3, len(args.cases))
   num_plot_rows = math.ceil(len(args.cases) / columns)
   figure, axes = plt.subplots(
      num_plot_rows,
      columns,
      figsize=(4.5 * columns, 3.6 * num_plot_rows),
      constrained_layout=True,
      squeeze=False,
   )
   colors = {
      "pwc": "#999999",
      "pwl": "#009E73",
      "wenoz": "#CC79A7",
      "mp5": "#0072B2",
      "teno5": "#D55E00",
      "parabolic": "#E69F00",
      "ppm4": "#56B4E9",
      "ppmx": "#000000",
      "ppm3": "#F0E442",
      "wenoz-char": "#CC79A7",
      "mp5-char": "#0072B2",
      "teno5-char": "#D55E00",
      "parabolic-char": "#E69F00",
      "ppm4-char": "#56B4E9",
      "ppmx-char": "#000000",
      "ppm3-char": "#F0E442",
   }
   display_names = {
      "pwc": "PWC",
      "pwl": "PWL-MC",
      "wenoz": "WENO-Z",
      "mp5": "MP5",
      "teno5": "TENO5",
      "parabolic": "parabolic",
      "ppm4": "PPM4",
      "ppmx": "PPMX",
      "ppm3": "PPM3",
      "wenoz-char": "WENO-Z char",
      "mp5-char": "MP5 char",
      "teno5-char": "TENO5 char",
      "parabolic-char": "parabolic char",
      "ppm4-char": "PPM4 char",
      "ppmx-char": "PPMX char",
      "ppm3-char": "PPM3 char",
   }
   for index, case in enumerate(args.cases):
      axis = axes[index // columns, index % columns]
      for method in args.methods:
         method_rows = sorted(
            (
               row for row in rows
               if int(row["case"]) == case and row["method"] == method
            ),
            key=lambda row: int(row["cells"]),
         )
         axis.loglog(
            [int(row["cells"]) for row in method_rows],
            [float(row["maximum_normalized_component_l1"]) for row in method_rows],
            marker="o",
            color=colors[method],
            label=display_names[method],
         )
      axis.set_title(f"Balsara {case}")
      axis.set_xlabel("cells")
      axis.set_ylabel("maximum normalized component L1")
      axis.grid(which="both", alpha=0.22)
   for index in range(len(args.cases), num_plot_rows * columns):
      axes[index // columns, index % columns].set_visible(False)
   axes[0, 0].legend(frameon=False, fontsize=8)
   figure.suptitle(
      "Exact cell-averaged SRMHD Balsara resolution scan — "
      "HLL/SSPRK3, CFL 0.9"
   )
   CONVERGENCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(CONVERGENCE_PLOT, dpi=180)
   plt.close(figure)


def plot_pwl_balsara4_resolution(
   args: argparse.Namespace,
   profiles: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]],
) -> None:
   if 4 not in args.cases or "pwl" not in args.methods or len(args.cells) < 2:
      return
   spec = next(
      spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) == 4
   )
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / "solution_b4.dat"
   )
   exact_state = exact_rmhd_conserved(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
   )
   reference_centers = profiles[(4, "pwl", max(args.cells))][0]
   spacing = reference_centers[1] - reference_centers[0]
   exact_average = exact_rmhd_cell_averages(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
      cell_edges=np.linspace(
         reference_centers[0] - 0.5 * spacing,
         reference_centers[-1] + 0.5 * spacing,
         reference_centers.size + 1,
      ),
   )
   figure, axes = plt.subplots(
      1,
      len(PLOT_COMPONENTS),
      figsize=(13.5, 3.2),
      constrained_layout=True,
   )
   colors = plt.get_cmap("viridis")(
      np.linspace(0.15, 0.85, len(args.cells))
   )
   for column, (name, label) in enumerate(PLOT_COMPONENTS):
      component = COMPONENT_NAMES.index(name)
      axis = axes[column]
      axis.plot(
         exact_profile[:, 0],
         exact_state[component],
         color="0.65",
         linewidth=1.0,
         label="exact point profile",
         zorder=10,
      )
      axis.step(
         reference_centers,
         exact_average[component],
         where="mid",
         color="black",
         linewidth=1.5,
         label=f"exact FV average (N={max(args.cells)})",
         zorder=11,
      )
      for color, cells in zip(colors, args.cells, strict=True):
         centers, numerical = profiles[(4, "pwl", cells)]
         axis.plot(
            centers,
            numerical[component],
            color=color,
            linewidth=1.0,
            label=f"N={cells}",
         )
      axis.set_title(label)
      axis.set_xlabel("x")
      axis.grid(alpha=0.22)
   axes[0].set_ylabel("t = 0.4")
   axes[0].legend(frameon=False, fontsize=8)
   figure.suptitle(
      "Balsara 4 PWL-MC resolution sequence — HLL/SSPRK3, CFL 0.9"
   )
   PWL_BALSARA4_RESOLUTION_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PWL_BALSARA4_RESOLUTION_PLOT, dpi=180)
   plt.close(figure)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--case",
      dest="cases",
      action="append",
      type=int,
      choices=tuple(range(1, 6)),
      help="Balsara case to scan; repeatable (default: all five)",
   )
   parser.add_argument(
      "--cells",
      action="append",
      type=int,
      help="resolution to scan; repeatable (default: 512)",
   )
   parser.add_argument(
      "--method",
      dest="methods",
      action="append",
      choices=METHOD_CHOICES,
      help="reconstruction method to scan; repeatable (default: all)",
   )
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if args.cases is None:
      args.cases = [1, 2, 3, 4, 5]
   if args.cells is None:
      args.cells = [512]
   args.cells = sorted(set(args.cells))
   if args.methods is None:
      args.methods = list(METHODS)
   rows, profiles = run_scan(args)
   write_tsv(
      RESULTS,
      rows,
      [
         "case",
         "method",
         "cells",
         "maximum_normalized_component_l1",
         "maximum_normalized_global_overshoot",
         "maximum_component_tv_ratio",
         "c2p_fallbacks",
         "atmosphere_resets",
         "mass_rel_drift",
         "energy_rel_drift",
         "mcells_per_s",
         "run_dir",
      ],
   )
   plot_scan(args, profiles)
   plot_convergence(args, rows)
   plot_pwl_balsara4_resolution(args, profiles)
   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(PLOT)}")
   if len(args.cells) > 1:
      print(f"wrote {repo_relative(CONVERGENCE_PLOT)}")
   if 4 in args.cases and "pwl" in args.methods and len(args.cells) > 1:
      print(f"wrote {repo_relative(PWL_BALSARA4_RESOLUTION_PLOT)}")
   if 4 in args.cases and "wenoz" in args.methods:
      print(f"wrote {repo_relative(WENOZ_FAILURE_PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
