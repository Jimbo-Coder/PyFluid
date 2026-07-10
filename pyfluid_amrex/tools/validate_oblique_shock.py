#!/usr/bin/env python3
"""Measure an oblique-shock angle for a hydro wedge AMR run."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from validation_common import mpi_prefix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_boxed_movie import composite_slice, parse_plot_bounds
from plotfile_reader import latest_plotfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_2d"
DEFAULT_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "oblique_shock_wedge_2d_obstacle.in"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "current" / "oblique_shock_wedge_validation"
DEFAULT_PLOT = ROOT / "plots" / "current" / "IC" / "oblique_shock_wedge_2d_obstacle" / "oblique_shock_angle_001.png"


def repo_relative(path: Path) -> Path:
   absolute = path if path.is_absolute() else ROOT / path
   return absolute.resolve().relative_to(ROOT)


def theta_beta_m_residual(beta: float, theta: float, mach: float, gamma: float) -> float:
   numerator = 2.0 / math.tan(beta) * (mach * mach * math.sin(beta) ** 2 - 1.0)
   denominator = mach * mach * (gamma + math.cos(2.0 * beta)) + 2.0
   return math.tan(theta) - numerator / denominator


def weak_oblique_shock_angle(theta_deg: float, mach: float, gamma: float) -> float:
   theta = math.radians(theta_deg)
   beta_min = math.asin(1.0 / mach) + 1.0e-10
   beta_max = 0.5 * math.pi - 1.0e-10
   samples = np.linspace(beta_min, beta_max, 4096)
   values = np.array([theta_beta_m_residual(float(beta), theta, mach, gamma) for beta in samples])
   crossings = np.where(values[:-1] * values[1:] <= 0.0)[0]
   if crossings.size == 0:
      raise ValueError(f"no attached oblique shock solution for M={mach}, theta={theta_deg}, gamma={gamma}")
   lo = float(samples[crossings[0]])
   hi = float(samples[crossings[0] + 1])
   flo = theta_beta_m_residual(lo, theta, mach, gamma)
   for _ in range(80):
      mid = 0.5 * (lo + hi)
      fmid = theta_beta_m_residual(mid, theta, mach, gamma)
      if flo * fmid <= 0.0:
         hi = mid
      else:
         lo = mid
         flo = fmid
   return math.degrees(0.5 * (lo + hi))


def run_solver(executable: Path, input_file: Path, output_dir: Path, ranks: int, force: bool) -> None:
   if force and output_dir.exists():
      import shutil

      shutil.rmtree(output_dir)
   command = [
      *mpi_prefix(ranks),
      str(executable),
      str(input_file),
      f"pyfluid.output_dir={repo_relative(output_dir)}",
   ]
   subprocess.run(command, cwd=ROOT, check=True)


def measure_shock_angle(
   plotfile: Path,
   mask_plotfile: Path,
   gamma: float,
   theta_deg: float,
   wedge_x0: float,
   wedge_y0: float,
   max_pixels: int,
   fit_x_min_offset: float,
   fit_x_max_offset: float,
   wall_clearance: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], float, float]:
   problo, probhi = parse_plot_bounds(plotfile)
   xlim = (problo[0], probhi[0])
   ylim = (problo[1], probhi[1])
   pressure, _boxes, _time, _ndim, _names, _xlim, _ylim = composite_slice(
      plotfile, "pressure", gamma, 2, 0.5, False, None, max_pixels, xlim, ylim
   )
   mask, _mask_boxes, _mask_time, _mask_ndim, _mask_names, _mask_xlim, _mask_ylim = composite_slice(
      mask_plotfile, "solid_mask", gamma, 2, 0.5, False, None, max_pixels, xlim, ylim
   )
   ny, nx = pressure.shape
   xs = np.linspace(xlim[0], xlim[1], nx)
   ys = np.linspace(ylim[0], ylim[1], ny)
   dx = (xlim[1] - xlim[0]) / max(nx - 1, 1)
   dy = (ylim[1] - ylim[0]) / max(ny - 1, 1)
   grad_y, grad_x = np.gradient(pressure, dy, dx)
   grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
   xx, yy = np.meshgrid(xs, ys)
   wall_y = wedge_y0 + math.tan(math.radians(theta_deg)) * (xx - wedge_x0)
   fit_x_min = wedge_x0 + fit_x_min_offset
   fit_x_max = min(xlim[1] - 0.20, wedge_x0 + fit_x_max_offset)
   roi = (
      (mask < 0.5)
      & (xx > fit_x_min)
      & (xx < fit_x_max)
      & (yy > wall_y + wall_clearance)
      & (yy < ylim[1] - 0.12)
      & np.isfinite(pressure)
   )
   x_points = []
   y_points = []
   g_points = []
   nbins = 42
   for lo, hi in zip(np.linspace(fit_x_min, fit_x_max, nbins)[:-1], np.linspace(fit_x_min, fit_x_max, nbins)[1:]):
      bin_mask = roi & (xx >= lo) & (xx < hi)
      if not np.any(bin_mask):
         continue
      flat_indices = np.argwhere(bin_mask)
      strengths = grad[bin_mask]
      index = flat_indices[int(np.argmax(strengths))]
      if grad[index[0], index[1]] <= np.percentile(grad[roi], 70.0):
         continue
      x_points.append(xx[index[0], index[1]])
      y_points.append(yy[index[0], index[1]])
      g_points.append(grad[index[0], index[1]])
   points = np.column_stack([x_points, y_points]) if x_points else np.empty((0, 2))
   if len(points) < 8:
      raise ValueError("could not identify enough shock-ridge points")

   weights = np.asarray(g_points)
   coeff = np.polyfit(points[:, 0], points[:, 1], 1, w=weights / np.max(weights))
   measured_beta = math.degrees(math.atan(coeff[0]))
   return measured_beta, pressure, mask, points, xlim, ylim, float(coeff[0]), float(coeff[1])


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
   parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
   parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-run", action="store_true")
   parser.add_argument("--mach", type=float, default=3.0)
   parser.add_argument("--gamma", type=float, default=1.4)
   parser.add_argument("--theta-deg", type=float, default=10.0)
   parser.add_argument("--wedge-x0", type=float, default=0.35)
   parser.add_argument("--wedge-y0", type=float, default=0.0)
   parser.add_argument("--max-pixels", type=int, default=1500)
   parser.add_argument("--angle-tol-deg", type=float, default=4.0)
   parser.add_argument("--fit-x-min-offset", type=float, default=0.50)
   parser.add_argument("--fit-x-max-offset", type=float, default=1.00)
   parser.add_argument("--wall-clearance", type=float, default=0.05)
   args = parser.parse_args()

   if not args.skip_run:
      run_solver(args.executable, args.input, args.output_dir, args.ranks, args.force)
   plotfile = latest_plotfile(args.output_dir)
   if plotfile is None:
      raise ValueError(f"no plotfiles found in {args.output_dir}")
   mask_plotfile = latest_plotfile(args.output_dir, "auxGF")
   if mask_plotfile is None:
      raise ValueError(f"no aux plotfiles found in {args.output_dir}")
   measured_beta, pressure, mask, points, xlim, ylim, slope, intercept = measure_shock_angle(
      plotfile,
      mask_plotfile,
      args.gamma,
      args.theta_deg,
      args.wedge_x0,
      args.wedge_y0,
      args.max_pixels,
      args.fit_x_min_offset,
      args.fit_x_max_offset,
      args.wall_clearance,
   )
   analytic_beta = weak_oblique_shock_angle(args.theta_deg, args.mach, args.gamma)
   error = measured_beta - analytic_beta
   analytic_slope = math.tan(math.radians(analytic_beta))
   wedge_slope = math.tan(math.radians(args.theta_deg))
   wedge_intercept = args.wedge_y0 - wedge_slope * args.wedge_x0
   analytic_offset_intercept = float(np.mean(points[:, 1] - analytic_slope * points[:, 0]))
   measured_origin_x = (wedge_intercept - intercept) / (slope - wedge_slope)
   measured_origin_y = wedge_slope * measured_origin_x + wedge_intercept
   origin_shift = math.hypot(measured_origin_x - args.wedge_x0, measured_origin_y - args.wedge_y0)

   args.plot_output.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(8.0, 5.3), constrained_layout=True)
   shown = pressure.copy()
   shown[mask > 0.5] = np.nan
   cmap = plt.get_cmap("viridis").copy()
   cmap.set_bad("#151515")
   im = ax.imshow(shown, origin="lower", extent=(xlim[0], xlim[1], ylim[0], ylim[1]), cmap=cmap, aspect="equal")
   xline = np.linspace(args.wedge_x0, min(xlim[1], args.wedge_x0 + 1.25), 200)
   fit_xline = np.linspace(float(points[:, 0].min()), float(points[:, 0].max()), 160)
   ax.plot(
      xline,
      args.wedge_y0 + analytic_slope * (xline - args.wedge_x0),
      color="white",
      linestyle=":",
      lw=1.6,
      alpha=0.65,
      label="analytic from mathematical vertex",
   )
   ax.plot(
      fit_xline,
      analytic_slope * fit_xline + analytic_offset_intercept,
      "w--",
      lw=2.0,
      label=f"analytic beta={analytic_beta:.2f} deg, free offset",
   )
   ax.plot(fit_xline, slope * fit_xline + intercept, color="#ff4d6d", lw=2.0, label=f"measured beta={measured_beta:.2f} deg")
   ax.scatter(points[:, 0], points[:, 1], s=12, c="#ffb703", edgecolors="black", linewidths=0.25, label="fit points")
   ax.set_xlim(0.0, 1.55)
   ax.set_ylim(0.0, 0.8)
   ax.set_xlabel("x")
   ax.set_ylabel("y")
   ax.set_title(
      f"Oblique shock validation: wedge, Mach {args.mach:g}, gamma={args.gamma:g}\n"
      f"theta={args.theta_deg:.1f} deg, measured-analytic={error:+.2f} deg, fitted origin shift={origin_shift:.3f}"
   )
   ax.legend(loc="upper right")
   fig.colorbar(im, ax=ax, label="pressure")
   fig.savefig(args.plot_output, dpi=180)
   plt.close(fig)

   print(
      "oblique shock validation: "
      f"theta={args.theta_deg:.3f} deg, M={args.mach:.3f}, gamma={args.gamma:.3f}, "
      f"analytic_beta={analytic_beta:.6f} deg, measured_beta={measured_beta:.6f} deg, "
      f"error={error:+.6f} deg, fitted_origin=({measured_origin_x:.6f}, {measured_origin_y:.6f}), "
      f"origin_shift={origin_shift:.6f}"
   )
   print(f"wrote {args.plot_output}")
   if abs(error) > args.angle_tol_deg:
      raise SystemExit(f"FAIL: angle error {error:+.3f} deg exceeds tolerance {args.angle_tol_deg:.3f} deg")
   print("PASS")


if __name__ == "__main__":
   main()
