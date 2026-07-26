#!/usr/bin/env python3
"""Plot exact Balsara profiles and uniform-grid CT refinement evidence."""

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
   parse_plot_header,
   parse_plot_header_1d,
   read_composite_state_1d,
   read_level_state,
)
from validate_rmhd import (
   EXACT_BALSARA_CASES,
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   exact_rmhd_conserved,
   read_last_exact_balsara_profile,
)
from validation_common import ROOT, repo_relative


PLOT_ROOT = ROOT / "plots/current/scans"
EXACT_PLOT = PLOT_ROOT / "rmhd_balsara_exact_profiles.png"
CONVERGENCE_PLOT = PLOT_ROOT / "rmhd_balsara_ct_uniform_convergence.png"
COMPONENTS = (
   ("dens", r"$D$"),
   ("mom1", r"$S_1$"),
   ("tau", r"$\tau$"),
   ("b2", r"$B_2$"),
)


def centers_from_plotfile(plotfile: Path, cells: int) -> np.ndarray:
   prob_lo, prob_hi = parse_plot_bounds(plotfile)
   spacing = (prob_hi[0] - prob_lo[0]) / cells
   return prob_lo[0] + (np.arange(cells) + 0.5) * spacing


def plot_exact_profiles() -> None:
   cases = (1, 4)
   specs = {int(spec["case"]): spec for spec in EXACT_BALSARA_CASES}
   figure, axes = plt.subplots(
      len(cases),
      len(COMPONENTS),
      figsize=(13.0, 5.8),
      sharex=True,
      constrained_layout=True,
   )
   for row, case in enumerate(cases):
      spec = specs[case]
      run_dir = RUN_ROOT / f"balsara{case}_exact"
      plotfile = latest_plotfile(run_dir, require=True)
      names, output_time = parse_plot_header_1d(plotfile)
      numerical_state = read_composite_state_1d(plotfile, names)
      centers = centers_from_plotfile(plotfile, numerical_state.shape[1])
      exact_profile = read_last_exact_balsara_profile(
         EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
      )
      exact_state = exact_rmhd_conserved(
         exact_profile,
         normal_magnetic=float(spec["left"][5]),
         gamma=float(spec["gamma"]),
      )
      exact_names = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
      for column, (name, label) in enumerate(COMPONENTS):
         axis = axes[row, column]
         axis.plot(
            exact_profile[:, 0],
            exact_state[exact_names.index(name)],
            color="black",
            linewidth=1.6,
            label="exact",
         )
         axis.plot(
            centers,
            numerical_state[names.index(name)],
            color="#0072B2",
            linewidth=1.0,
            label="PyFluid MP5",
         )
         axis.set_title(f"Balsara {case}: {label}")
         axis.grid(alpha=0.25)
         if column == 0:
            axis.set_ylabel(f"t = {output_time:g}")
         if row == len(cases) - 1:
            axis.set_xlabel("x")
   axes[0, 0].legend(frameon=False)
   figure.suptitle(
      "Exact 1D SRMHD Riemann profiles — N=512, MP5/HLL, CFL 0.9"
   )
   EXACT_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(EXACT_PLOT, dpi=180)
   plt.close(figure)


def uniform_ct_errors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   resolutions = np.asarray((128, 256, 512, 1024))
   profiles: dict[int, np.ndarray] = {}
   for cells in resolutions:
      run_dir = RUN_ROOT / f"balsara1_2d_ct_n{cells}"
      plotfile = latest_plotfile(run_dir, require=True)
      names, output_dim, _ = parse_plot_header(plotfile)
      state = read_level_state(plotfile, names, output_dim, 0)
      profiles[int(cells)] = np.mean(state, axis=2)
   reference = profiles[1024]
   errors = []
   for cells in resolutions[:-1]:
      factor = 1024 // int(cells)
      averaged_reference = reference.reshape(
         reference.shape[0], int(cells), factor
      ).mean(axis=2)
      scale = np.maximum(np.ptp(averaged_reference, axis=1), 1.0)
      errors.append(float(np.mean(
         np.abs(profiles[int(cells)] - averaged_reference)
         / scale[:, np.newaxis]
      )))
   errors_array = np.asarray(errors)
   rates = np.log(errors_array[:-1] / errors_array[1:]) / math.log(2.0)
   return resolutions[:-1], errors_array, rates


def plot_uniform_ct_convergence() -> None:
   resolutions, errors, rates = uniform_ct_errors()
   figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
   axis.loglog(
      resolutions,
      errors,
      "o-",
      color="#D55E00",
      linewidth=1.8,
      label="error vs N=1024 reference",
   )
   first_order = errors[0] * resolutions[0] / resolutions
   axis.loglog(
      resolutions,
      first_order,
      "--",
      color="0.35",
      label=r"$N^{-1}$ guide",
   )
   for index, rate in enumerate(rates):
      x_value = math.sqrt(resolutions[index] * resolutions[index + 1])
      y_value = math.sqrt(errors[index] * errors[index + 1])
      axis.annotate(
         f"p={rate:.2f}",
         (x_value, y_value),
         xytext=(0, 8),
         textcoords="offset points",
         ha="center",
      )
   axis.set_xlabel("x resolution N")
   axis.set_ylabel("mean normalized conserved-state L1")
   axis.set_title(
      "Balsara 1: uniform-grid 2D CT refinement\n"
      "(numerical N=1024 reference; this is not an AMR test)"
   )
   axis.grid(which="both", alpha=0.25)
   axis.legend(frameon=False)
   CONVERGENCE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(CONVERGENCE_PLOT, dpi=180)
   plt.close(figure)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--only",
      choices=("all", "exact", "convergence"),
      default="all",
   )
   args = parser.parse_args()
   if args.only in ("all", "exact"):
      plot_exact_profiles()
      print(f"wrote {repo_relative(EXACT_PLOT)}")
   if args.only in ("all", "convergence"):
      plot_uniform_ct_convergence()
      print(f"wrote {repo_relative(CONVERGENCE_PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
