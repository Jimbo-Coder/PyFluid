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
   exact_rmhd_conserved,
   read_last_exact_balsara_profile,
   relativistic_recovery_counts,
   run_case,
)
from validation_common import ROOT, read_scalar_rows, repo_relative, write_tsv


METHODS = ("pwc", "pwl", "wenoz", "mp5", "teno5")
COMPONENT_NAMES = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
PLOT_COMPONENTS = (
   ("dens", r"$D$"),
   ("mom1", r"$S_1$"),
   ("tau", r"$\tau$"),
   ("b2", r"$B_2$"),
)
RESULTS = RUN_ROOT / "balsara_reconstruction_scan.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_reconstruction_scan.png"
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
   sampled_exact = np.vstack([
      np.interp(centers, exact_profile[:, 0], exact_state[component])
      for component in range(exact_state.shape[0])
   ])
   scales = np.maximum(
      np.ptp(sampled_exact, axis=1),
      np.max(np.abs(sampled_exact), axis=1),
   )
   scales = np.maximum(scales, 1.0e-14)
   component_l1 = np.mean(np.abs(numerical - sampled_exact), axis=1) / scales

   exact_min = np.min(sampled_exact, axis=1)
   exact_max = np.max(sampled_exact, axis=1)
   lower_overshoot = np.maximum(exact_min[:, np.newaxis] - numerical, 0.0)
   upper_overshoot = np.maximum(numerical - exact_max[:, np.newaxis], 0.0)
   maximum_overshoot = float(
      np.max(np.maximum(lower_overshoot, upper_overshoot) / scales[:, np.newaxis])
   )

   numerical_tv = np.sum(np.abs(np.diff(numerical, axis=1)), axis=1)
   exact_tv = np.sum(np.abs(np.diff(sampled_exact, axis=1)), axis=1)
   active = exact_tv > 1.0e-12 * scales
   maximum_tv_ratio = float(np.max(numerical_tv[active] / exact_tv[active]))
   return (
      centers,
      numerical,
      exact_profile[:, 0],
      float(np.max(component_l1)),
      maximum_overshoot,
      maximum_tv_ratio,
   )


def run_scan(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
   specs = {
      int(spec["case"]): spec
      for spec in EXACT_BALSARA_CASES
      if int(spec["case"]) in args.cases
   }
   rows: list[dict[str, object]] = []
   profiles: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
   for case in args.cases:
      spec = specs[case]
      left = spec["left"]
      right = spec["right"]
      for method in METHODS:
         overrides = {
            "geometry.prob_lo": "-1.0",
            "geometry.prob_hi": "1.0",
            "amr.n_cell": str(args.cells),
            "amr.max_grid_size": "128",
            "pyfluid.gamma": repr(spec["gamma"]),
            "pyfluid.final_time": repr(spec["time"]),
            "pyfluid.max_step": "10000",
            "pyfluid.ng": "3" if method in {"wenoz", "mp5", "teno5"} else "2",
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
            f"recon_scan_balsara{case}_{method}_n{args.cells}",
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
            "cells": args.cells,
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
         profiles[(case, method)] = (centers, numerical)
         print(
            f"Balsara {case} {method}: L1={maximum_l1:.6e} "
            f"overshoot={maximum_overshoot:.6e} TV={maximum_tv_ratio:.4f} "
            f"resets={reset_count}"
         )
   return rows, profiles


def plot_scan(
   args: argparse.Namespace,
   profiles: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
) -> None:
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
      for column, (name, label) in enumerate(PLOT_COMPONENTS):
         component = COMPONENT_NAMES.index(name)
         axis = axes[row, column]
         axis.plot(
            exact_profile[:, 0],
            exact_state[component],
            color="black",
            linewidth=1.8,
            label="exact",
            zorder=10,
         )
         plotted_methods = (
            tuple(method for method in METHODS if method != "wenoz")
            if case == 4
            else METHODS
         )
         for method in plotted_methods:
            centers, numerical = profiles[(case, method)]
            axis.plot(
               centers,
               numerical[component],
               color=colors[method],
               linewidth=0.85,
               alpha=0.9,
               label=method.upper(),
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
      f"SRMHD reconstruction comparison — N={args.cells}, HLL/SSPRK3, CFL 0.9; "
      "Balsara 4 WENO-Z is off scale"
   )
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)

   if 4 not in args.cases:
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
   centers, numerical = profiles[(case, "wenoz")]
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
         color="black",
         linewidth=1.8,
         label="exact",
      )
      axis.plot(
         centers,
         numerical[component],
         color=colors["wenoz"],
         linewidth=0.9,
         label="WENO-Z",
      )
      axis.set_title(label)
      axis.set_xlabel("x")
      axis.grid(alpha=0.22)
   failure_axes[0].set_ylabel(f"t = {float(spec['time']):g}")
   failure_axes[0].legend(frameon=False)
   failure_figure.suptitle(
      f"Balsara 4 WENO-Z off-scale failure — N={args.cells}, CFL 0.9"
   )
   WENOZ_FAILURE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   failure_figure.savefig(WENOZ_FAILURE_PLOT, dpi=180)
   plt.close(failure_figure)


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
   parser.add_argument("--cells", type=int, default=512)
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()
   if args.cases is None:
      args.cases = [1, 2, 3, 4, 5]
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
   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(PLOT)}")
   if 4 in args.cases:
      print(f"wrote {repo_relative(WENOZ_FAILURE_PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
