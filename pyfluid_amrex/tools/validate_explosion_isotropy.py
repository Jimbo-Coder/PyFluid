#!/usr/bin/env python3
"""Check locality and angular symmetry of a centered Cartesian pressure bump."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from plot_amrex_boxed_movie import component_values, level_chunks, level_count, parse_plot_bounds
from plotfile_reader import latest_plotfile
from plot_amrex_slice import parse_plot_header
from validation_common import ROOT, read_key_values, repo_relative, write_tsv


FIELDS = [
   "hook",
   "status",
   "reason",
   "metric",
   "value",
   "tolerance",
   "run_dir",
]


class NotApplicable(RuntimeError):
   """Raised when a run is not a clean centered explosion diagnostic."""


def param_float(params: dict[str, str], key: str, default: float) -> float:
   raw = params.get(key)
   if raw is None:
      return default
   try:
      value = float(raw)
   except ValueError:
      return default
   return value if math.isfinite(value) else default


def ref_ratio(params: dict[str, str]) -> int:
   raw = params.get("amr.ref_ratio", "2").replace(",", " ").split()
   return int(raw[0]) if raw else 2


def run_is_supported(params: dict[str, str], plotfile: Path) -> None:
   names, ndim, _time_value = parse_plot_header(plotfile)
   if ndim != 2:
      raise NotApplicable("explosion isotropy check currently expects a 2D plotfile")
   if "pressure" not in names and not {"rho", "mom1", "mom2", "mom3", "energy"}.issubset(names):
      raise NotApplicable("plotfile does not contain enough fields to reconstruct pressure")
   if params.get("pyfluid.ic", "") != "pressure_bump":
      raise NotApplicable("run is not pyfluid.ic = pressure_bump")
   if params.get("pyfluid.physics", "hydro") != "hydro":
      raise NotApplicable("magnetized pressure bumps are not isotropic acoustic checks")
   if params.get("mesh.coordinate", "cartesian") != "cartesian":
      raise NotApplicable("explosion isotropy check currently expects mesh.coordinate = cartesian")
   if params.get("mesh.patch_system", "single") != "single":
      raise NotApplicable("explosion isotropy check currently expects mesh.patch_system = single")
   for key in ("pyfluid.v1_0", "pyfluid.v2_0", "pyfluid.v3_0"):
      if abs(param_float(params, key, 0.0)) > 1.0e-14:
         raise NotApplicable("background flow is nonzero")


def base_shape(plotfile: Path, ndim: int) -> tuple[int, ...]:
   chunks = level_chunks(plotfile, 0, ndim)
   return tuple(max(chunk[1][d] for chunk in chunks) + 1 for d in range(ndim))


def composite_component(
   plotfile: Path,
   component: str,
   gamma: float,
   ratio: int,
) -> tuple[np.ndarray, float, tuple[float, float], tuple[float, float]]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 2:
      raise NotApplicable("composite explosion diagnostic currently supports 2D only")
   problo, probhi = parse_plot_bounds(plotfile)
   nlevels = level_count(plotfile)
   coarse_shape = base_shape(plotfile, ndim)
   finest_factor = ratio ** (nlevels - 1)
   finest_shape = tuple(size * finest_factor for size in coarse_shape)
   image = np.full(finest_shape, np.nan)

   for lev in range(nlevels):
      scale = ratio ** (nlevels - 1 - lev)
      for lo, hi, data in level_chunks(plotfile, lev, ndim):
         values = component_values(data, names, component, gamma)
         tile = values
         if scale > 1:
            tile = np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)
         slices = tuple(slice(lo[d] * scale, (hi[d] + 1) * scale) for d in range(ndim))
         image[slices] = tile

   if np.isnan(image).any():
      raise RuntimeError(f"{plotfile} did not cover the full Cartesian composite grid")
   return image, time_value, tuple(problo[:2]), tuple(probhi[:2])


def pressure_background(params: dict[str, str]) -> float:
   return param_float(params, "pyfluid.pressure0", 1.0)


def center_from_params(params: dict[str, str], problo: tuple[float, float], probhi: tuple[float, float]) -> tuple[float, float]:
   return (
      param_float(params, "pyfluid.bump_x0", 0.5 * (problo[0] + probhi[0])),
      param_float(params, "pyfluid.bump_y0", 0.5 * (problo[1] + probhi[1])),
   )


def require_centered(
   center: tuple[float, float],
   problo: tuple[float, float],
   probhi: tuple[float, float],
) -> None:
   dx = probhi[0] - problo[0]
   dy = probhi[1] - problo[1]
   mid = (0.5 * (problo[0] + probhi[0]), 0.5 * (problo[1] + probhi[1]))
   if abs(center[0] - mid[0]) > 1.0e-8 * max(dx, 1.0) or abs(center[1] - mid[1]) > 1.0e-8 * max(dy, 1.0):
      raise NotApplicable("pressure bump is not centered in the domain")


def expected_acoustic_radius(
   params: dict[str, str],
   time_value: float,
) -> float:
   gamma = param_float(params, "pyfluid.gamma", 5.0 / 3.0)
   rho0 = param_float(params, "pyfluid.rho0", 1.0)
   pressure0 = pressure_background(params)
   sigma = param_float(params, "pyfluid.bump_sigma", 0.08)
   sound_speed = math.sqrt(max(gamma * pressure0 / rho0, 0.0))
   safety = param_float(params, "pyfluid.validation_explosion_cone_safety", 1.35)
   width_factor = param_float(params, "pyfluid.validation_explosion_initial_widths", 4.0)
   return safety * sound_speed * max(time_value, 0.0) + width_factor * sigma


def front_statistics(
   delta: np.ndarray,
   problo: tuple[float, float],
   probhi: tuple[float, float],
   center: tuple[float, float],
   threshold_rel: float,
   sectors: int,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
   nx, ny = delta.shape
   x = problo[0] + (np.arange(nx, dtype=float) + 0.5) * (probhi[0] - problo[0]) / nx
   y = problo[1] + (np.arange(ny, dtype=float) + 0.5) * (probhi[1] - problo[1]) / ny
   xx, yy = np.meshgrid(x, y, indexing="ij")
   radius = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
   angle = np.mod(np.arctan2(yy - center[1], xx - center[0]), 2.0 * np.pi)
   abs_delta = np.abs(delta)
   threshold = max(float(np.nanmax(abs_delta)) * threshold_rel, 1.0e-14)
   changed = abs_delta >= threshold
   if not np.any(changed):
      raise RuntimeError("explosion diagnostic found no disturbed cells")

   front = np.full(sectors, np.nan)
   sector_index = np.minimum((angle * sectors / (2.0 * np.pi)).astype(int), sectors - 1)
   for sector in range(sectors):
      mask = changed & (sector_index == sector)
      if np.any(mask):
         front[sector] = float(np.nanmax(radius[mask]))
   valid = np.isfinite(front)
   if np.count_nonzero(valid) < max(8, sectors // 2):
      raise RuntimeError("too few angular sectors contain disturbed cells")
   front_valid = front[valid]
   mean = float(np.mean(front_valid))
   std = float(np.std(front_valid))
   cv = std / max(mean, 1.0e-300)
   return (
      mean,
      cv,
      float(np.min(front_valid)),
      float(np.max(front_valid)),
      threshold,
      radius,
      abs_delta,
   )


def passfail(value: float, tolerance: float) -> str:
   return "PASS" if math.isfinite(value) and value <= tolerance else "FAIL"


def validation_rows(run_dir: Path, input_file: Path | None = None) -> list[dict[str, object]]:
   run_dir = run_dir.resolve()
   input_path = input_file or run_dir / "run.in"
   params = read_key_values(input_path)
   plotfile = latest_plotfile(run_dir, require=True)
   run_is_supported(params, plotfile)
   gamma = param_float(params, "pyfluid.gamma", 5.0 / 3.0)
   pressure, time_value, problo, probhi = composite_component(plotfile, "pressure", gamma, ref_ratio(params))
   center = center_from_params(params, problo, probhi)
   require_centered(center, problo, probhi)

   expected_radius = expected_acoustic_radius(params, time_value)
   boundary_distance = min(center[0] - problo[0], probhi[0] - center[0], center[1] - problo[1], probhi[1] - center[1])
   if expected_radius >= 0.9 * boundary_distance:
      raise NotApplicable("expected acoustic cone reaches the boundary")

   delta = pressure - pressure_background(params)
   threshold_rel = param_float(params, "pyfluid.validation_explosion_front_threshold", 2.0e-3)
   sectors = max(8, int(param_float(params, "pyfluid.validation_explosion_sectors", 32)))
   front_mean, front_cv, _front_min, front_max, _threshold, radius, abs_delta = front_statistics(
      delta,
      problo,
      probhi,
      center,
      threshold_rel,
      sectors,
   )
   total_signal = float(np.sum(abs_delta))
   outside_fraction = 0.0 if total_signal == 0.0 else float(np.sum(abs_delta[radius > expected_radius]) / total_signal)
   dx_finest = min((probhi[0] - problo[0]) / delta.shape[0], (probhi[1] - problo[1]) / delta.shape[1])
   causal_excess = max(0.0, front_max - expected_radius) / max(expected_radius, dx_finest)

   front_tol = param_float(params, "pyfluid.validation_explosion_front_cv_tol", 8.0e-2)
   outside_tol = param_float(params, "pyfluid.validation_explosion_outside_fraction_tol", 3.0e-2)
   causal_tol = param_float(params, "pyfluid.validation_explosion_causal_excess_tol", 5.0e-2)
   rows = [
      {
         "hook": "explosion_isotropy",
         "status": passfail(front_cv, front_tol),
         "reason": "" if front_cv <= front_tol else f"front_radius_cv={front_cv:.3e} > {front_tol:.3e}",
         "metric": "front_radius_cv",
         "value": front_cv,
         "tolerance": front_tol,
         "run_dir": str(run_dir),
      },
      {
         "hook": "explosion_locality",
         "status": passfail(outside_fraction, outside_tol),
         "reason": "" if outside_fraction <= outside_tol else f"outside_cone_fraction={outside_fraction:.3e} > {outside_tol:.3e}",
         "metric": "outside_cone_fraction",
         "value": outside_fraction,
         "tolerance": outside_tol,
         "run_dir": str(run_dir),
      },
      {
         "hook": "explosion_locality",
         "status": passfail(causal_excess, causal_tol),
         "reason": "" if causal_excess <= causal_tol else f"causal_excess={causal_excess:.3e} > {causal_tol:.3e}",
         "metric": "causal_excess",
         "value": causal_excess,
         "tolerance": causal_tol,
         "run_dir": str(run_dir),
      },
   ]
   rows[0]["reason"] = rows[0]["reason"] or f"front_radius_mean={front_mean:.6e}"
   return rows


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("run_dir", type=Path)
   parser.add_argument("--input", type=Path)
   parser.add_argument("--output", type=Path)
   args = parser.parse_args()

   run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
   input_file = args.input
   if input_file is not None and not input_file.is_absolute():
      input_file = ROOT / input_file
   output = args.output or run_dir / "explosion_isotropy.tsv"
   if not output.is_absolute():
      output = ROOT / output

   try:
      rows = validation_rows(run_dir, input_file=input_file)
   except NotApplicable as exc:
      print(f"SKIP explosion_isotropy: {exc}")
      return 0
   write_tsv(output, rows, FIELDS)
   failed = [row for row in rows if row["status"] != "PASS"]
   for row in rows:
      print(
         f"{row['status']} {row['hook']} {row['metric']}="
         f"{float(row['value']):.3e} tol={float(row['tolerance']):.3e}"
      )
   print(f"wrote {repo_relative(output)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
