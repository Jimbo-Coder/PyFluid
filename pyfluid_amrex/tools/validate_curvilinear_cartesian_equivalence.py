#!/usr/bin/env python3
"""Compare Cartesian xy and cylindrical r-phi evolution for smooth problems.

The two runs use the same physical pressure bump.  The comparison samples the
AMR composite solutions at the same physical points on a cylindrical sampling
grid and writes pointwise error norms.  This is intended to catch coordinate,
origin, and AMR hierarchy artifacts rather than to replace normal convergence
tests.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_slice import full_shape, level_chunks, level_count, parse_plot_bounds, parse_plot_header
from plotfile_reader import latest_plotfile, plotfile_step, sorted_plotfiles
from validation_common import (
   ROOT,
   materialize_run_input,
   parameters_from_assignments,
   read_key_values,
   read_tsv,
   scalar_validation_row,
)


RUN_CASE = ROOT / "pyfluid_amrex/tools/run_case.py"
BASE_INPUT = ROOT / "pyfluid_amrex/inputs/pressure_bump_2d_amr.in"
OUT_ROOT = ROOT / "runs/current/curvilinear_equivalence"


@dataclass(frozen=True)
class FieldSet:
   x: np.ndarray
   y: np.ndarray
   rho: np.ndarray
   pressure: np.ndarray
   v1: np.ndarray
   v2: np.ndarray
   level: np.ndarray


@dataclass(frozen=True)
class CompositeState:
   plotfile: Path
   names: list[str]
   ndim: int
   problo: tuple[float, ...]
   probhi: tuple[float, ...]
   base_shape: tuple[int, ...]
   chunks: list[list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]]]


def run_case(name: str, overrides: list[str], ranks: int, timeout: int | None, policy: str) -> Path:
   run_dir = OUT_ROOT / name
   if run_dir.exists():
      shutil.rmtree(run_dir)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "run.in",
      parameters_from_assignments(
         [
            *overrides,
            f"pyfluid.output_dir={run_dir.relative_to(ROOT)}",
            f"pyfluid.run_name={name}",
         ]
      ),
   )
   command = [
      sys.executable,
      str(RUN_CASE),
      str(input_path),
      "--ranks",
      str(ranks),
      "--keep-existing",
      "--skip-plots",
      "--policy",
      policy,
   ]
   completed = subprocess.run(command, cwd=ROOT, text=True, check=False, timeout=timeout)
   if completed.returncode != 0:
      raise RuntimeError(f"{name} failed with return code {completed.returncode}; see {run_dir / 'run.log'}")
   return run_dir


def centers(lo: float, hi: float, n: int) -> np.ndarray:
   dx = (hi - lo) / n
   return lo + (np.arange(n, dtype=float) + 0.5) * dx


def read_composite_state(run_dir: Path, plotfile: Path | None = None) -> CompositeState:
   plotfile = latest_plotfile(run_dir, require=True) if plotfile is None else plotfile
   names, ndim, _time = parse_plot_header(plotfile)
   if ndim != 2:
      raise RuntimeError(f"{plotfile} is not 2D")
   problo, probhi = parse_plot_bounds(plotfile)
   nlevels = level_count(plotfile)
   chunks = [level_chunks(plotfile, lev, ndim) for lev in range(nlevels)]
   return CompositeState(
      plotfile=plotfile,
      names=names,
      ndim=ndim,
      problo=problo,
      probhi=probhi,
      base_shape=full_shape(chunks[0], ndim),
      chunks=chunks,
   )


def sample_state_components(state: CompositeState, q1: np.ndarray, q2: np.ndarray, components: list[str], ref_ratio: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
   if q1.shape != q2.shape:
      raise ValueError("sample coordinate arrays must have matching shapes")
   missing = [name for name in components if name not in state.names]
   if missing:
      raise ValueError(f"{state.plotfile} is missing components: {', '.join(missing)}")

   values = {name: np.full(q1.shape, np.nan, dtype=float) for name in components}
   level_used = np.full(q1.shape, -1, dtype=int)
   valid = (
      (q1 >= state.problo[0])
      & (q1 < state.probhi[0])
      & (q2 >= state.problo[1])
      & (q2 < state.probhi[1])
   )
   span0 = state.probhi[0] - state.problo[0]
   span1 = state.probhi[1] - state.problo[1]
   component_indices = {name: state.names.index(name) for name in components}

   for lev, chunks in enumerate(state.chunks):
      level_shape = tuple(state.base_shape[d] * (ref_ratio**lev) for d in range(state.ndim))
      i = np.floor((q1 - state.problo[0]) * level_shape[0] / span0).astype(int)
      j = np.floor((q2 - state.problo[1]) * level_shape[1] / span1).astype(int)
      i = np.clip(i, 0, level_shape[0] - 1)
      j = np.clip(j, 0, level_shape[1] - 1)
      for lo, hi, data in chunks:
         inside = valid & (i >= lo[0]) & (i <= hi[0]) & (j >= lo[1]) & (j <= hi[1])
         if not np.any(inside):
            continue
         local_i = i[inside] - lo[0]
         local_j = j[inside] - lo[1]
         for name, component in component_indices.items():
            values[name][inside] = data[component, local_i, local_j]
         level_used[inside] = lev

   if np.any(valid & (level_used < 0)):
      missing_count = int(np.count_nonzero(valid & (level_used < 0)))
      raise RuntimeError(f"{state.plotfile} did not cover {missing_count} requested sample points")
   return values, level_used


def bilinear_sample_state_components(
   state: CompositeState,
   q1: np.ndarray,
   q2: np.ndarray,
   components: list[str],
   ref_ratio: int,
   sample_level: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
   level_shape = tuple(state.base_shape[d] * (ref_ratio**sample_level) for d in range(state.ndim))
   dx0 = (state.probhi[0] - state.problo[0]) / level_shape[0]
   dx1 = (state.probhi[1] - state.problo[1]) / level_shape[1]
   f0 = (q1 - (state.problo[0] + 0.5 * dx0)) / dx0
   f1 = (q2 - (state.problo[1] + 0.5 * dx1)) / dx1
   i0 = np.clip(np.floor(f0).astype(int), 0, level_shape[0] - 2)
   j0 = np.clip(np.floor(f1).astype(int), 0, level_shape[1] - 2)
   t0 = np.clip(f0 - i0, 0.0, 1.0)
   t1 = np.clip(f1 - j0, 0.0, 1.0)
   q00 = state.problo[0] + (i0 + 0.5) * dx0
   q10 = state.problo[0] + (i0 + 1.5) * dx0
   q01 = state.problo[1] + (j0 + 0.5) * dx1
   q11 = state.problo[1] + (j0 + 1.5) * dx1

   v00, l00 = sample_state_components(state, q00, q01, components, ref_ratio)
   v10, l10 = sample_state_components(state, q10, q01, components, ref_ratio)
   v01, l01 = sample_state_components(state, q00, q11, components, ref_ratio)
   v11, l11 = sample_state_components(state, q10, q11, components, ref_ratio)
   values: dict[str, np.ndarray] = {}
   for name in components:
      values[name] = (
         (1.0 - t0) * (1.0 - t1) * v00[name]
         + t0 * (1.0 - t1) * v10[name]
         + (1.0 - t0) * t1 * v01[name]
         + t0 * t1 * v11[name]
      )
   level_used = np.maximum.reduce([l00, l10, l01, l11])
   return values, level_used


def read_sampled_fields(
   run_dir: Path,
   gamma: float,
   coordinate: str,
   q1: np.ndarray,
   q2: np.ndarray,
   ref_ratio: int,
   *,
   interpolate: bool,
   sample_level: int,
) -> FieldSet:
   state = read_composite_state(run_dir)
   components = ["rho", "mom1", "mom2", "mom3", "energy"]
   if interpolate:
      raw, level_used = bilinear_sample_state_components(state, q1, q2, components, ref_ratio, sample_level)
   else:
      raw, level_used = sample_state_components(state, q1, q2, components, ref_ratio)
   rho = raw["rho"]
   u1 = raw["mom1"] / rho
   u2 = raw["mom2"] / rho
   kinetic = 0.5 * (raw["mom1"] * raw["mom1"] + raw["mom2"] * raw["mom2"] + raw["mom3"] * raw["mom3"]) / rho
   pressure = (raw["energy"] - kinetic) * (gamma - 1.0)
   if coordinate == "cylindrical":
      x = q1 * np.cos(q2)
      y = q1 * np.sin(q2)
      v1 = u1 * np.cos(q2) - u2 * np.sin(q2)
      v2 = u1 * np.sin(q2) + u2 * np.cos(q2)
   else:
      x = q1
      y = q2
      v1 = u1
      v2 = u2
   return FieldSet(x=x, y=y, rho=rho, pressure=pressure, v1=v1, v2=v2, level=level_used)


def compare_fields(
   cart: FieldSet,
   cyl: FieldSet,
   compare_radius: float,
   pressure0: float,
   sample_level: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
   radius = np.sqrt(cyl.x * cyl.x + cyl.y * cyl.y)
   mask = radius <= compare_radius
   feature = mask & (np.abs(cyl.pressure - pressure0) > 1.0e-4 * max(pressure0, 1.0))
   if np.count_nonzero(feature) < 8:
      feature = mask
   common_finest = mask & (cart.level >= sample_level) & (cyl.level >= sample_level)

   rows: list[dict[str, object]] = []
   pressure_error = np.full_like(cyl.pressure, np.nan)
   for name in ("rho", "pressure", "v1", "v2"):
      cart_values = getattr(cart, name)
      cyl_values = getattr(cyl, name)
      diff = cyl_values - cart_values
      if name == "pressure":
         pressure_error = np.where(mask, diff, np.nan)
      regions = [("all", mask), ("feature", feature), ("common_finest", common_finest)]
      for label, selected in regions:
         vals = diff[selected]
         ref = np.maximum(np.abs(cyl_values[selected]), np.abs(cart_values[selected]))
         scale = max(float(np.max(ref)) if ref.size else 1.0, 1.0)
         rows.append({
            "field": name,
            "region": label,
            "count": int(vals.size),
            "l2_abs": float(math.sqrt(float(np.mean(vals * vals)))) if vals.size else math.nan,
            "linf_abs": float(np.max(np.abs(vals))) if vals.size else math.nan,
            "l2_rel": float(math.sqrt(float(np.mean(vals * vals))) / scale) if vals.size else math.nan,
            "linf_rel": float(np.max(np.abs(vals)) / scale) if vals.size else math.nan,
         })
   return rows, pressure_error


def write_error_plot(path: Path, cyl: FieldSet, pressure_error: np.ndarray) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
   finite = np.isfinite(pressure_error)
   limit = np.nanmax(np.abs(pressure_error[finite])) if np.any(finite) else 1.0
   limit = max(limit, 1.0e-30)
   image = ax.tripcolor(
      cyl.x[finite].ravel(),
      cyl.y[finite].ravel(),
      pressure_error[finite].ravel(),
      shading="gouraud",
      cmap="coolwarm",
      vmin=-limit,
      vmax=limit,
   )
   ax.set_aspect("equal")
   ax.set_xlabel("x")
   ax.set_ylabel("y")
   ax.set_title("cylindrical r-phi minus Cartesian xy pressure")
   xmin = float(np.min(cyl.x[finite]))
   xmax = float(np.max(cyl.x[finite]))
   ymin = float(np.min(cyl.y[finite]))
   ymax = float(np.max(cyl.y[finite]))
   span = max(xmax - xmin, ymax - ymin, 1.0e-12)
   margin = 0.04 * span
   ax.set_xlim(xmin - margin, xmax + margin)
   ax.set_ylim(ymin - margin, ymax + margin)
   fig.colorbar(image, ax=ax, label="pressure error")
   fig.savefig(path, dpi=170)
   plt.close(fig)


def write_acoustic_packet_profile(
   case_dir: Path,
   cart_dir: Path,
   cyl_dir: Path,
   *,
   gamma: float,
   ref_ratio: int,
   sample_level: int,
   compare_radius: float,
   center_x: float,
   center_y: float,
   sigma: float,
   transverse_sigma: float,
   amplitude: float,
   rho0: float,
   pressure0: float,
) -> None:
   x = np.linspace(-compare_radius, compare_radius, 801)
   y = np.full_like(x, center_y)
   cart = read_sampled_fields(
      cart_dir, gamma, "cartesian", x, y, ref_ratio, interpolate=False, sample_level=sample_level
   )
   radius = np.hypot(x, y)
   phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)
   cyl = read_sampled_fields(
      cyl_dir, gamma, "cylindrical", radius, phi, ref_ratio, interpolate=False, sample_level=sample_level
   )

   cart_plotfile = latest_plotfile(cart_dir, require=True)
   cyl_plotfile = latest_plotfile(cyl_dir, require=True)
   _cart_names, _cart_ndim, cart_time = parse_plot_header(cart_plotfile)
   _cyl_names, _cyl_ndim, cyl_time = parse_plot_header(cyl_plotfile)
   if not math.isclose(cart_time, cyl_time, rel_tol=0.0, abs_tol=1.0e-12):
      raise RuntimeError(f"profile times differ: Cartesian {cart_time:.17g}, cylindrical {cyl_time:.17g}")

   sound_speed = math.sqrt(gamma * pressure0 / rho0)
   transverse = np.exp(-0.5 * ((y - center_y) / transverse_sigma) ** 2)
   initial = amplitude * np.exp(-0.5 * ((x - center_x) / sigma) ** 2) * transverse
   translated = amplitude * np.exp(-0.5 * ((x - center_x - sound_speed * cart_time) / sigma) ** 2) * transverse
   cart_delta = cart.rho - rho0
   cyl_delta = cyl.rho - rho0

   fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
   ax.plot(x, initial, color="0.45", linestyle="--", linewidth=1.5, label="initial")
   ax.plot(x, translated, color="black", linestyle=":", linewidth=1.8, label="ideal 1D translation", zorder=4)
   ax.plot(x, cart_delta, color="tab:blue", linewidth=2.2, label="Cartesian")
   ax.plot(x, cyl_delta, color="tab:red", linewidth=1.8, marker="o", markevery=32, markersize=3.2, label="cylindrical r-phi", zorder=3)
   ax.set_xlabel("x")
   ax.set_ylabel(r"$\rho-\rho_0$")
   ax.set_title(f"right-going acoustic packet at t={cart_time:.3f}")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   fig.savefig(case_dir / "acoustic_packet_centerline.png", dpi=180)
   plt.close(fig)

   initial_peak = float(np.max(np.abs(initial)))
   cart_peak = float(np.max(np.abs(cart_delta)))
   cyl_peak = float(np.max(np.abs(cyl_delta)))
   write_tsv(case_dir / "acoustic_packet_profile.tsv", [{
      "time": cart_time,
      "sound_speed": sound_speed,
      "initial_peak": initial_peak,
      "cartesian_peak": cart_peak,
      "cylindrical_peak": cyl_peak,
      "cartesian_retained_fraction": cart_peak / initial_peak,
      "cylindrical_retained_fraction": cyl_peak / initial_peak,
      "cylindrical_to_cartesian_peak": cyl_peak / cart_peak,
      "cartesian_l2_vs_ideal_1d": float(np.sqrt(np.mean((cart_delta - translated) ** 2))),
      "cylindrical_l2_vs_ideal_1d": float(np.sqrt(np.mean((cyl_delta - translated) ** 2))),
      "cylindrical_l2_vs_cartesian": float(np.sqrt(np.mean((cyl_delta - cart_delta) ** 2))),
   }])


def write_entropy_packet_profile(
   case_dir: Path,
   cart_dir: Path,
   cyl_dir: Path,
   *,
   gamma: float,
   ref_ratio: int,
   sample_level: int,
   compare_radius: float,
   center_x: float,
   center_y: float,
   sigma: float,
   amplitude: float,
   advection_speed: float,
   rho0: float,
) -> None:
   x = np.linspace(-compare_radius, compare_radius, 801)
   y = np.full_like(x, center_y)
   cart = read_sampled_fields(
      cart_dir, gamma, "cartesian", x, y, ref_ratio, interpolate=False, sample_level=sample_level
   )
   radius = np.hypot(x, y)
   phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)
   cyl = read_sampled_fields(
      cyl_dir, gamma, "cylindrical", radius, phi, ref_ratio, interpolate=False, sample_level=sample_level
   )

   cart_plotfile = latest_plotfile(cart_dir, require=True)
   cyl_plotfile = latest_plotfile(cyl_dir, require=True)
   _cart_names, _cart_ndim, cart_time = parse_plot_header(cart_plotfile)
   _cyl_names, _cyl_ndim, cyl_time = parse_plot_header(cyl_plotfile)
   if not math.isclose(cart_time, cyl_time, rel_tol=0.0, abs_tol=1.0e-12):
      raise RuntimeError(f"profile times differ: Cartesian {cart_time:.17g}, cylindrical {cyl_time:.17g}")

   initial = amplitude * np.exp(-0.5 * ((x - center_x) / sigma) ** 2)
   exact = amplitude * np.exp(-0.5 * ((x - center_x - advection_speed * cart_time) / sigma) ** 2)
   cart_delta = cart.rho - rho0
   cyl_delta = cyl.rho - rho0

   fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
   ax.plot(x, initial, color="0.45", linestyle="--", linewidth=1.5, label="initial")
   ax.plot(x, exact, color="black", linestyle=":", linewidth=2.0, label="exact translation", zorder=4)
   ax.plot(x, cart_delta, color="tab:blue", linewidth=2.2, label="Cartesian")
   ax.plot(
      x,
      cyl_delta,
      color="tab:red",
      linewidth=1.8,
      marker="o",
      markevery=32,
      markersize=3.2,
      label="cylindrical r-phi",
      zorder=3,
   )
   ax.set_xlabel("x")
   ax.set_ylabel(r"$\rho-\rho_0$")
   ax.set_title(f"advected Gaussian entropy packet at t={cart_time:.3f}")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   fig.savefig(case_dir / "entropy_packet_centerline.png", dpi=180)
   plt.close(fig)

   write_tsv(case_dir / "entropy_packet_profile.tsv", [{
      "time": cart_time,
      "advection_speed": advection_speed,
      "exact_center_x": center_x + advection_speed * cart_time,
      "initial_peak": float(np.max(initial)),
      "cartesian_peak": float(np.max(cart_delta)),
      "cylindrical_peak": float(np.max(cyl_delta)),
      "cartesian_l2_vs_exact": float(np.sqrt(np.mean((cart_delta - exact) ** 2))),
      "cylindrical_l2_vs_exact": float(np.sqrt(np.mean((cyl_delta - exact) ** 2))),
      "cylindrical_l2_vs_cartesian": float(np.sqrt(np.mean((cyl_delta - cart_delta) ** 2))),
      "cartesian_linf_vs_exact": float(np.max(np.abs(cart_delta - exact))),
      "cylindrical_linf_vs_exact": float(np.max(np.abs(cyl_delta - exact))),
   }])


def read_amr_summary(run_dir: Path) -> dict[str, object]:
   rows = read_tsv(run_dir / "amr_levels.tsv")
   if not rows:
      return {
         "active_finest_level": 0,
         "max_level": 0,
         "finest_headroom": 0,
         "total_budget_fraction": "",
         "peak_level_budget_fraction": "",
      }
   final = rows[-1]
   active_level = int(float(final.get("active_finest_level", "0")))
   return {
      "active_finest_level": active_level,
      "max_level": int(float(final.get("max_level", "0"))),
      "finest_headroom": int(float(final.get("finest_headroom", "0"))),
      "total_budget_fraction": final.get("total_budget_fraction", ""),
      "peak_level_budget_fraction": final.get("peak_level_budget_fraction", ""),
      "terminal_decision": final.get(f"refinement_decision_L{active_level}", ""),
   }


def polar_center(x: float, y: float) -> tuple[float, float]:
   radius = math.hypot(x, y)
   if radius == 0.0:
      return 0.0, 0.0
   angle = math.atan2(y, x)
   if angle < 0.0:
      angle += 2.0 * math.pi
   return radius, angle


def label_float(value: float) -> str:
   text = f"{value:.6g}"
   return text.replace("-", "m").replace(".", "p")


def acoustic_packet_coverage_rows(
   run_dir: Path,
   coordinate: str,
   *,
   center_x: float,
   center_y: float,
   sigma: float,
   transverse_sigma: float,
   sound_speed: float,
   compare_radius: float,
   ref_ratio: int,
) -> list[dict[str, object]]:
   rows: list[dict[str, object]] = []
   y_extent = min(2.0 * transverse_sigma, 0.5 * compare_radius)
   y = np.linspace(center_y - y_extent, center_y + y_extent, 129)
   for plotfile in sorted_plotfiles(run_dir):
      state = read_composite_state(run_dir, plotfile)
      _names, _ndim, time_value = parse_plot_header(plotfile)
      packet_center = center_x + sound_speed * time_value
      x = np.concatenate((
         np.full_like(y, packet_center - sigma),
         np.full_like(y, packet_center + sigma),
      ))
      yy = np.concatenate((y, y))
      inside = x * x + yy * yy < compare_radius * compare_radius
      x = x[inside]
      yy = yy[inside]
      if coordinate == "cylindrical":
         q1 = np.hypot(x, yy)
         q2 = np.mod(np.arctan2(yy, x), 2.0 * math.pi)
      else:
         q1 = x
         q2 = yy
      _values, level_used = sample_state_components(state, q1, q2, ["rho"], ref_ratio)
      active_level = len(state.chunks) - 1
      covered = level_used >= active_level
      refined = level_used > 0
      rows.append({
         "step": int(plotfile.name[-8:-3]) if plotfile.name[-8:-3].isdigit() else "",
         "time": time_value,
         "active_finest_level": active_level,
         "gradient_samples": int(level_used.size),
         "gradient_refined_fraction": float(np.mean(refined)) if refined.size else math.nan,
         "gradient_finest_fraction": float(np.mean(covered)) if covered.size else math.nan,
         "gradient_mean_level": float(np.mean(level_used)) if level_used.size else math.nan,
      })
   return rows


def amr_interest_score_rows(
   run_dir: Path,
   coordinate: str,
   *,
   ref_ratio: int,
) -> list[dict[str, object]]:
   """Measure how efficiently AMR depth follows the solver's own tag metric."""
   auxiliary_by_step = {
      plotfile_step(path): path
      for path in sorted_plotfiles(run_dir, prefix="auxGF", require=True)
   }
   rows: list[dict[str, object]] = []
   for state_path in sorted_plotfiles(run_dir, require=True):
      step = plotfile_step(state_path)
      if step not in auxiliary_by_step:
         raise RuntimeError(f"missing amr_tag_metric output at step {step} in {run_dir}")
      state = read_composite_state(run_dir, state_path)
      auxiliary = read_composite_state(run_dir, auxiliary_by_step[step])
      if state.base_shape != auxiliary.base_shape or len(state.chunks) != len(auxiliary.chunks):
         raise RuntimeError(f"state/aux hierarchy mismatch at step {step} in {run_dir}")

      active_level = len(state.chunks) - 1
      sample_shape = tuple(min(n * (ref_ratio**active_level), 512) for n in state.base_shape)
      q1 = centers(state.problo[0], state.probhi[0], sample_shape[0])
      q2 = centers(state.problo[1], state.probhi[1], sample_shape[1])
      qq1, qq2 = np.meshgrid(q1, q2, indexing="ij")
      metric_values, metric_level = sample_state_components(
         auxiliary, qq1, qq2, ["amr_tag_metric"], ref_ratio
      )
      _state_values, state_level = sample_state_components(state, qq1, qq2, ["rho"], ref_ratio)
      if not np.array_equal(metric_level, state_level):
         raise RuntimeError(f"state/aux ownership mismatch at step {step} in {run_dir}")

      metric = np.maximum(metric_values["amr_tag_metric"], 0.0)
      volume_weight = qq1 if coordinate == "cylindrical" else np.ones_like(qq1)
      interest_weight = metric * volume_weight
      interest_total = float(np.sum(interest_weight))
      physical_volume_total = float(np.sum(volume_weight))
      refined = state_level > 0
      if active_level > 0:
         depth = (ref_ratio**state_level - 1.0) / (ref_ratio**active_level - 1.0)
      else:
         depth = np.zeros_like(metric)

      coverage = (
         float(np.sum(interest_weight[refined])) / interest_total
         if interest_total > 0.0 else 1.0
      )
      normalized_depth = (
         float(np.sum(interest_weight * depth)) / interest_total
         if interest_total > 0.0 else 0.0
      )
      refined_volume_fraction = (
         float(np.sum(volume_weight[refined])) / physical_volume_total
         if physical_volume_total > 0.0 else math.nan
      )
      concentration = (
         coverage / refined_volume_fraction
         if refined_volume_fraction > 0.0 else math.inf
      )
      _names, _ndim, time_value = parse_plot_header(state_path)
      rows.append({
         "step": step,
         "time": time_value,
         "active_finest_level": active_level,
         "metric_weighted_coverage": coverage,
         "metric_weighted_depth": normalized_depth,
         "amr_interest_score": coverage * normalized_depth,
         "refined_volume_fraction": refined_volume_fraction,
         "interest_concentration": concentration,
         "max_tag_metric": float(np.max(metric)),
      })
   return rows


def write_amr_interest_plot(path: Path, rows: list[dict[str, object]], coordinate: str) -> None:
   time = np.asarray([float(row["time"]) for row in rows])
   fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)
   axes[0].plot(time, [float(row["metric_weighted_coverage"]) for row in rows], "o-", label="coverage")
   axes[0].plot(time, [float(row["metric_weighted_depth"]) for row in rows], "s-", label="depth")
   axes[0].plot(time, [float(row["amr_interest_score"]) for row in rows], "^-", label="score")
   axes[0].set_ylabel("Metric-weighted fraction")
   axes[0].set_ylim(-0.03, 1.03)
   axes[0].legend()
   axes[1].plot(
      time,
      [float(row["refined_volume_fraction"]) for row in rows],
      "o-",
      label="refined physical volume",
   )
   axes[1].set_xlabel("Time")
   axes[1].set_ylabel("Volume fraction")
   axes[1].set_ylim(bottom=0.0)
   axes[1].legend()
   for axis in axes:
      axis.grid(True, which="major", alpha=0.25)
   fig.suptitle(f"AMR Interest Coverage | {coordinate}")
   fig.savefig(path, dpi=180)
   plt.close(fig)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fields = list(rows[0].keys()) if rows else []
   with path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def scalar_status(run_dir: Path, case: str, policy: str) -> dict[str, object]:
   row = scalar_validation_row(run_dir, case=case, input_file=run_dir / "run.in", policy=policy, mhd=False, dim=2, ranks="")
   return row


def require_common_finest_norms(
   rows: list[dict[str, object]],
   case: str,
   *,
   pressure_l2_max: float,
   rho_l2_max: float,
   velocity_l2_max: float,
   min_count: int,
) -> None:
   by_field = {
      str(row["field"]): row
      for row in rows
      if row.get("region") == "common_finest"
   }
   missing = [field for field in ("rho", "pressure", "v1", "v2") if field not in by_field]
   if missing:
      raise SystemExit(f"{case} missing common-finest rows for {', '.join(missing)}")

   failures: list[str] = []
   for field, row in by_field.items():
      count = int(row["count"])
      l2 = float(row["l2_abs"])
      if count < min_count:
         failures.append(f"{field} common_finest count {count} < {min_count}")
         continue
      if field == "pressure":
         limit = pressure_l2_max
      elif field == "rho":
         limit = rho_l2_max
      else:
         limit = velocity_l2_max
      if not math.isfinite(l2) or l2 > limit:
         failures.append(f"{field} common_finest L2 {l2:.6e} > {limit:.6e}")

   if failures:
      detail = "; ".join(failures)
      raise SystemExit(f"{case} curvilinear/Cartesian AMR equivalence failed: {detail}")


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ic", choices=("pressure_bump", "acoustic_packet_x", "density_cloud"), default="pressure_bump")
   parser.add_argument("--nr", nargs="+", type=int, default=[48], help="Cylindrical radial resolutions.")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--radius", type=float, default=1.5)
   parser.add_argument("--compare-radius", type=float, default=0.75)
   parser.add_argument("--final-time", type=float, default=0.012)
   parser.add_argument("--max-step", type=int, default=100000)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--cart-cfl", type=float, help="Override Cartesian CFL for timestep-matching diagnostics.")
   parser.add_argument("--cyl-cfl", type=float, help="Override cylindrical CFL for timestep-matching diagnostics.")
   parser.add_argument("--angular-factor", type=int, default=4, help="Base phi cells per radial cell.")
   parser.add_argument("--nphi", type=int, help="Explicit cylindrical base phi cells; overrides --angular-factor.")
   parser.add_argument("--amr-max-level", type=int, default=0)
   parser.add_argument("--min-active-level", type=int, help="Required final active AMR level; defaults to --amr-max-level.")
   parser.add_argument("--ref-ratio", type=int, default=2)
   parser.add_argument("--tag-threshold", type=float, default=0.02)
   parser.add_argument("--min-feature-length", type=float, default=1.0e-6)
   parser.add_argument("--n-error-buf", type=int, default=4)
   parser.add_argument("--regrid-int", type=int, default=2)
   parser.add_argument("--blocking-factor", type=int, default=8)
   parser.add_argument("--max-grid-size", type=int, default=64)
   parser.add_argument("--state-interp", default="quartic-conservative")
   parser.add_argument("--subcycling", choices=("off", "on"), default="off")
   parser.add_argument("--outer-bc", choices=("reflecting", "outflow"), default="reflecting")
   parser.add_argument("--plot-int", type=int, default=100000)
   parser.add_argument("--scalar-int", type=int, default=1)
   parser.add_argument("--amplitude", type=float, default=0.20)
   parser.add_argument("--bump-sigma", type=float, default=0.18)
   parser.add_argument("--transverse-sigma", type=float, default=0.14)
   parser.add_argument("--advection-speed", type=float, default=1.0)
   parser.add_argument("--min-headroom", type=int, default=2)
   parser.add_argument("--cart-metric-divergence", action="store_true", help="Route Cartesian RHS divergence through the metric face-integral path.")
   parser.add_argument("--center-refine-radius", type=float, default=0.0, help="Optional initial refinement seed radius around the origin.")
   parser.add_argument("--bump-x0", type=float, default=0.0, help="Physical Cartesian x-center of the pressure bump.")
   parser.add_argument("--bump-y0", type=float, default=0.0, help="Physical Cartesian y-center of the pressure bump.")
   parser.add_argument("--max-pressure-l2-common-finest", type=float, default=2.0e-4)
   parser.add_argument("--max-rho-l2-common-finest", type=float, default=1.0e-5)
   parser.add_argument("--max-velocity-l2-common-finest", type=float, default=5.0e-5)
   parser.add_argument("--min-common-finest-points", type=int, default=8)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   if args.angular_factor < 1:
      raise SystemExit("--angular-factor must be positive")
   if args.nphi is not None and args.nphi < 4:
      raise SystemExit("--nphi must be at least 4")
   if args.amr_max_level < 0:
      raise SystemExit("--amr-max-level must be non-negative")
   if args.ref_ratio != 2:
      raise SystemExit("this comparison currently assumes AMReX ref_ratio=2")
   if args.blocking_factor < 1 or args.max_grid_size < args.blocking_factor:
      raise SystemExit("--blocking-factor must be positive and no larger than --max-grid-size")
   min_active_level = (1 if args.amr_max_level > 0 else 0) if args.min_active_level is None else args.min_active_level

   all_rows: list[dict[str, object]] = []
   scalar_policy = "smoke" if args.max_step == 0 else ("open" if args.outer_bc == "outflow" else "wall")
   plot_int = 0 if args.max_step == 0 else args.plot_int
   cyl_bump_r0, cyl_bump_phi0 = polar_center(args.bump_x0, args.bump_y0)
   cart_cfl = args.cfl if args.cart_cfl is None else args.cart_cfl
   cyl_cfl = args.cfl if args.cyl_cfl is None else args.cyl_cfl
   for nr in args.nr:
      nphi = args.nphi if args.nphi is not None else args.angular_factor * nr
      ncart = 2 * nr
      case_suffix = "_init" if args.max_step == 0 else ""
      if args.regrid_int != 2:
         case_suffix += f"_rg{args.regrid_int}"
      if args.state_interp != "quartic-conservative":
         case_suffix += f"_{args.state_interp.replace('-', '_')}"
      if args.cart_metric_divergence:
         case_suffix += "_cartmetricdiv"
      if args.cart_cfl is not None:
         case_suffix += f"_cartcfl{label_float(cart_cfl)}"
      if args.cyl_cfl is not None:
         case_suffix += f"_cylcfl{label_float(cyl_cfl)}"
      if args.bump_x0 != 0.0 or args.bump_y0 != 0.0:
         case_suffix += f"_x{label_float(args.bump_x0)}_y{label_float(args.bump_y0)}"
      if args.ic == "acoustic_packet_x":
         case_suffix += f"_ts{label_float(args.transverse_sigma)}"
      if args.ic == "density_cloud":
         case_suffix += f"_vx{label_float(args.advection_speed)}"
      if args.center_refine_radius > 0.0:
         case_suffix += f"_seed{label_float(args.center_refine_radius)}"
      if args.tag_threshold != 0.02:
         case_suffix += f"_tag{label_float(args.tag_threshold)}"
      if args.n_error_buf != 4:
         case_suffix += f"_buf{args.n_error_buf}"
      if args.blocking_factor != 8:
         case_suffix += f"_bf{args.blocking_factor}"
      case_suffix += f"_subcycle_{args.subcycling}"
      case_suffix += f"_{args.outer_bc}"
      case = f"{args.ic}_cart_vs_cyl_rphi_nr{nr:04d}_nphi{nphi:04d}_l{args.amr_max_level}{case_suffix}"
      case_dir = OUT_ROOT / case
      if case_dir.exists() and not args.force:
         raise SystemExit(f"{case_dir} exists; pass --force to overwrite")
      if case_dir.exists():
         shutil.rmtree(case_dir)

      ref_ratio_values = " ".join(str(args.ref_ratio) for _ in range(args.amr_max_level))
      amr_overrides = [
         f"amr.max_level={args.amr_max_level}",
         f"amr.blocking_factor={args.blocking_factor}",
         f"amr.max_grid_size={args.max_grid_size}",
         f"amr.regrid_int={args.regrid_int}",
         "amr.grid_eff=0.85",
         f"amr.state_interp={args.state_interp}",
         f"amr.tag_threshold={args.tag_threshold:.17g}",
         f"amr.min_feature_length={args.min_feature_length:.17g}",
         "amr.total_cell_budget=1000000",
      ]
      if args.amr_max_level > 0:
         amr_overrides.extend([
            f"amr.ref_ratio={ref_ratio_values}",
            "amr.n_error_buf=" + " ".join(str(args.n_error_buf) for _ in range(args.amr_max_level)),
         ])
      common = [
         "pyfluid.physics=hydro",
         f"pyfluid.ic={args.ic}",
         "pyfluid.rho0=1.0",
         "pyfluid.pressure0=1.0",
         f"pyfluid.amplitude={args.amplitude:.17g}",
         f"pyfluid.bump_sigma={args.bump_sigma:.17g}",
         f"pyfluid.acoustic_transverse_sigma={args.transverse_sigma:.17g}",
         f"pyfluid.v1_0={args.advection_speed if args.ic == 'density_cloud' else 0.0:.17g}",
         "pyfluid.v2_0=0.0",
         "pyfluid.v3_0=0.0",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.final_time={:.17g}".format(args.final_time),
         f"pyfluid.max_step={args.max_step}",
         "pyfluid.ng=2",
         f"pyfluid.scalar_int={args.scalar_int}",
         f"pyfluid.plot_int={plot_int}",
         "pyfluid.checkpoint_int=-1",
         "pyfluid.physical_check_int=1",
         "pyfluid.debug_check_finite=1",
         f"pyfluid.subcycling={args.subcycling}",
         "pyfluid.diffusion=none",
         "pyfluid.gravity=none",
         "pyfluid.source=none",
         f"pyfluid.aux_gfs={'amr_tag_metric' if args.ic in ('acoustic_packet_x', 'density_cloud') else 'none'}",
         f"pyfluid.aux_gf_compute_int={plot_int if args.ic in ('acoustic_packet_x', 'density_cloud') else -1}",
         f"pyfluid.aux_gf_int={plot_int if args.ic in ('acoustic_packet_x', 'density_cloud') else -1}",
         *amr_overrides,
      ]
      cart_amr = []
      cyl_amr = []
      if args.center_refine_radius > 0.0 and args.amr_max_level > 0:
         cart_amr = [
            "amr.initial_refine=box",
            f"amr.initial_refine_levels={args.amr_max_level}",
            "amr.initial_refine_fixed_levels=0",
            f"amr.initial_refine_box_lo={-args.center_refine_radius:.17g} {-args.center_refine_radius:.17g}",
            f"amr.initial_refine_box_hi={args.center_refine_radius:.17g} {args.center_refine_radius:.17g}",
         ]
         cyl_amr = [
            "amr.initial_refine=box",
            f"amr.initial_refine_levels={args.amr_max_level}",
            "amr.initial_refine_fixed_levels=0",
            "amr.initial_refine_box_lo=0.0 0.0",
            f"amr.initial_refine_box_hi={args.center_refine_radius:.17g} {2.0 * math.pi:.17g}",
         ]
      cart_name = f"{case}_cartesian"
      cart_dir = run_case(
         cart_name,
         [
            *common,
            "mesh.coordinate=cartesian",
            f"mesh.metric_divergence={'always' if args.cart_metric_divergence else 'auto'}",
            "mesh.axes=x y",
            f"geometry.prob_lo={-args.radius:.17g} {-args.radius:.17g}",
            f"geometry.prob_hi={args.radius:.17g} {args.radius:.17g}",
            "geometry.is_periodic=0 0",
            f"amr.n_cell={ncart} {ncart}",
            *cart_amr,
            f"pyfluid.bc={args.outer_bc}",
            f"pyfluid.cfl={cart_cfl:.17g}",
            f"pyfluid.bump_x0={args.bump_x0:.17g}",
            f"pyfluid.bump_y0={args.bump_y0:.17g}",
         ],
         args.ranks,
         args.timeout,
         scalar_policy,
      )
      cyl_name = f"{case}_cylindrical"
      cyl_dir = run_case(
         cyl_name,
         [
            *common,
            "mesh.coordinate=cylindrical",
            "mesh.axes=r phi",
            f"geometry.prob_lo=0.0 0.0",
            f"geometry.prob_hi={args.radius:.17g} {2.0 * math.pi:.17g}",
            "geometry.is_periodic=0 1",
            f"amr.n_cell={nr} {nphi}",
            *cyl_amr,
            "pyfluid.bc=reflecting",
            "pyfluid.bc_xlo=origin",
            f"pyfluid.bc_xhi={args.outer_bc}",
            "pyfluid.bc_ylo=periodic",
            "pyfluid.bc_yhi=periodic",
            f"pyfluid.cfl={cyl_cfl:.17g}",
            f"pyfluid.bump_x0={(args.bump_x0 if args.ic == 'acoustic_packet_x' else cyl_bump_r0):.17g}",
            f"pyfluid.bump_y0={(args.bump_y0 if args.ic == 'acoustic_packet_x' else cyl_bump_phi0):.17g}",
         ],
         args.ranks,
         args.timeout,
         scalar_policy,
      )

      cart_amr_summary = read_amr_summary(cart_dir)
      cyl_amr_summary = read_amr_summary(cyl_dir)
      if cart_amr_summary["active_finest_level"] < min_active_level or cyl_amr_summary["active_finest_level"] < min_active_level:
         raise SystemExit(
            f"{case} did not activate requested AMR depth: "
            f"cart L{cart_amr_summary['active_finest_level']}, "
            f"cyl L{cyl_amr_summary['active_finest_level']}, "
            f"required L{min_active_level}"
         )
      for label, summary in (("cart", cart_amr_summary), ("cyl", cyl_amr_summary)):
         headroom = args.amr_max_level - int(summary["active_finest_level"])
         if args.amr_max_level > 0 and headroom < args.min_headroom:
            raise SystemExit(f"{case} {label} hierarchy retained only {headroom} level(s) of headroom")
         if args.amr_max_level > 0 and summary.get("terminal_decision") != "smooth":
            raise SystemExit(
               f"{case} {label} hierarchy stopped on {summary.get('terminal_decision')!r}, not 'smooth'"
            )
      if args.ic in ("acoustic_packet_x", "density_cloud"):
         for coordinate, run_dir in (("cartesian", cart_dir), ("cylindrical", cyl_dir)):
            interest_rows = amr_interest_score_rows(run_dir, coordinate, ref_ratio=args.ref_ratio)
            write_tsv(case_dir / f"amr_interest_score_{coordinate}.tsv", interest_rows)
            write_amr_interest_plot(
               case_dir / f"amr_interest_score_{coordinate}.png",
               interest_rows,
               coordinate,
            )
      gamma = float(read_key_values(cyl_dir / "run.in").get("pyfluid.gamma", "1.6666666666666667"))
      sample_level = min(cart_amr_summary["active_finest_level"], cyl_amr_summary["active_finest_level"])
      r = centers(0.0, args.radius, nr * (args.ref_ratio**sample_level))
      phi = centers(0.0, 2.0 * math.pi, nphi * (args.ref_ratio**sample_level))
      rr, pp = np.meshgrid(r, phi, indexing="ij")
      cyl = read_sampled_fields(
         cyl_dir, gamma, "cylindrical", rr, pp, args.ref_ratio, interpolate=False, sample_level=sample_level
      )
      cart = read_sampled_fields(
         cart_dir, gamma, "cartesian", cyl.x, cyl.y, args.ref_ratio, interpolate=True, sample_level=sample_level
      )
      comparison_rows, pressure_error = compare_fields(cart, cyl, args.compare_radius, pressure0=1.0, sample_level=sample_level)
      require_common_finest_norms(
         comparison_rows,
         case,
         pressure_l2_max=args.max_pressure_l2_common_finest,
         rho_l2_max=args.max_rho_l2_common_finest,
         velocity_l2_max=args.max_velocity_l2_common_finest,
         min_count=args.min_common_finest_points,
      )
      for row in comparison_rows:
         row.update({
            "case": case,
            "nr": nr,
            "nphi": nphi,
            "ncart": ncart,
            "angular_factor": args.angular_factor,
            "amr_max_level": args.amr_max_level,
            "sample_level": sample_level,
            "cart_active_finest_level": cart_amr_summary["active_finest_level"],
            "cyl_active_finest_level": cyl_amr_summary["active_finest_level"],
            "cart_total_budget_fraction": cart_amr_summary["total_budget_fraction"],
            "cyl_total_budget_fraction": cyl_amr_summary["total_budget_fraction"],
            "cart_cfl": cart_cfl,
            "cyl_cfl": cyl_cfl,
            "time": args.final_time,
         })
      all_rows.extend(comparison_rows)

      write_tsv(case_dir / "equivalence.tsv", comparison_rows)
      write_error_plot(case_dir / "pressure_error.png", cyl, pressure_error)
      if args.ic == "density_cloud":
         write_entropy_packet_profile(
            case_dir,
            cart_dir,
            cyl_dir,
            gamma=gamma,
            ref_ratio=args.ref_ratio,
            sample_level=sample_level,
            compare_radius=args.compare_radius,
            center_x=args.bump_x0,
            center_y=args.bump_y0,
            sigma=args.bump_sigma,
            amplitude=args.amplitude,
            advection_speed=args.advection_speed,
            rho0=1.0,
         )
      if args.ic == "acoustic_packet_x":
         sound_speed = math.sqrt((5.0 / 3.0) * 1.0 / 1.0)
         write_acoustic_packet_profile(
            case_dir,
            cart_dir,
            cyl_dir,
            gamma=gamma,
            ref_ratio=args.ref_ratio,
            sample_level=sample_level,
            compare_radius=args.compare_radius,
            center_x=args.bump_x0,
            center_y=args.bump_y0,
            sigma=args.bump_sigma,
            transverse_sigma=args.transverse_sigma,
            amplitude=args.amplitude,
            rho0=1.0,
            pressure0=1.0,
         )
         cart_coverage = acoustic_packet_coverage_rows(
            cart_dir,
            "cartesian",
            center_x=args.bump_x0,
            center_y=args.bump_y0,
            sigma=args.bump_sigma,
            transverse_sigma=args.transverse_sigma,
            sound_speed=sound_speed,
            compare_radius=args.compare_radius,
            ref_ratio=args.ref_ratio,
         )
         cyl_coverage = acoustic_packet_coverage_rows(
            cyl_dir,
            "cylindrical",
            center_x=args.bump_x0,
            center_y=args.bump_y0,
            sigma=args.bump_sigma,
            transverse_sigma=args.transverse_sigma,
            sound_speed=sound_speed,
            compare_radius=args.compare_radius,
            ref_ratio=args.ref_ratio,
         )
         for row in cart_coverage:
            row["coordinate"] = "cartesian"
         for row in cyl_coverage:
            row["coordinate"] = "cylindrical"
         coverage_rows = cart_coverage + cyl_coverage
         write_tsv(case_dir / "amr_gradient_coverage.tsv", coverage_rows)
         evolved_coverage = [
            float(row["gradient_refined_fraction"])
            for row in coverage_rows
            if (float(row["time"]) > 0.0 or args.max_step == 0)
            and int(row["active_finest_level"]) > 0
         ]
         if not evolved_coverage or min(evolved_coverage) < 0.80:
            worst = min(evolved_coverage, default=math.nan)
            raise SystemExit(
               f"{case} AMR failed to refine acoustic gradient strips above the base level: "
               f"coverage={worst:.6g} < 0.8"
            )
      scalar_rows = [scalar_status(cart_dir, cart_name, scalar_policy), scalar_status(cyl_dir, cyl_name, scalar_policy)]
      write_tsv(case_dir / "scalar_checks.tsv", scalar_rows)
      if any(row["status"] != "PASS" for row in scalar_rows):
         raise SystemExit(f"{case} scalar sanity failed; see {case_dir / 'scalar_checks.tsv'}")

   write_tsv(OUT_ROOT / "cartesian_cylindrical_equivalence.tsv", all_rows)
   print(f"PASS curvilinear_cartesian_equivalence rows={len(all_rows)} out={OUT_ROOT}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
