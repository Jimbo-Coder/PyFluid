#!/usr/bin/env python3
"""Compare Cartesian xy and cylindrical r-phi evolution for one smooth problem.

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
from plotfile_reader import latest_plotfile
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


def read_composite_state(run_dir: Path) -> CompositeState:
   plotfile = latest_plotfile(run_dir, require=True)
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
   return {
      "active_finest_level": int(float(final.get("active_finest_level", "0"))),
      "max_level": int(float(final.get("max_level", "0"))),
      "finest_headroom": int(float(final.get("finest_headroom", "0"))),
      "total_budget_fraction": final.get("total_budget_fraction", ""),
      "peak_level_budget_fraction": final.get("peak_level_budget_fraction", ""),
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
   parser.add_argument("--amr-max-level", type=int, default=0)
   parser.add_argument("--min-active-level", type=int, help="Required final active AMR level; defaults to --amr-max-level.")
   parser.add_argument("--ref-ratio", type=int, default=2)
   parser.add_argument("--tag-threshold", type=float, default=0.02)
   parser.add_argument("--n-error-buf", type=int, default=4)
   parser.add_argument("--regrid-int", type=int, default=2)
   parser.add_argument("--state-interp", default="quartic-conservative")
   parser.add_argument("--cart-metric-regrid", action="store_true", help="Route Cartesian valid-cell regrid transfer through the metric-density path.")
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
   if args.amr_max_level < 0:
      raise SystemExit("--amr-max-level must be non-negative")
   if args.ref_ratio != 2:
      raise SystemExit("this comparison currently assumes AMReX ref_ratio=2")
   min_active_level = args.amr_max_level if args.min_active_level is None else args.min_active_level

   all_rows: list[dict[str, object]] = []
   scalar_policy = "smoke" if args.max_step == 0 else "wall"
   plot_int = 0 if args.max_step == 0 else 100000
   cyl_bump_r0, cyl_bump_phi0 = polar_center(args.bump_x0, args.bump_y0)
   cart_cfl = args.cfl if args.cart_cfl is None else args.cart_cfl
   cyl_cfl = args.cfl if args.cyl_cfl is None else args.cyl_cfl
   for nr in args.nr:
      nphi = args.angular_factor * nr
      ncart = 2 * nr
      case_suffix = "_init" if args.max_step == 0 else ""
      if args.regrid_int != 2:
         case_suffix += f"_rg{args.regrid_int}"
      if args.state_interp != "quartic-conservative":
         case_suffix += f"_{args.state_interp.replace('-', '_')}"
      if args.cart_metric_regrid:
         case_suffix += "_cartmetric"
      if args.cart_metric_divergence:
         case_suffix += "_cartmetricdiv"
      if args.cart_cfl is not None:
         case_suffix += f"_cartcfl{label_float(cart_cfl)}"
      if args.cyl_cfl is not None:
         case_suffix += f"_cylcfl{label_float(cyl_cfl)}"
      if args.bump_x0 != 0.0 or args.bump_y0 != 0.0:
         case_suffix += f"_x{label_float(args.bump_x0)}_y{label_float(args.bump_y0)}"
      if args.center_refine_radius > 0.0:
         case_suffix += f"_seed{label_float(args.center_refine_radius)}"
      if args.tag_threshold != 0.02:
         case_suffix += f"_tag{label_float(args.tag_threshold)}"
      if args.n_error_buf != 4:
         case_suffix += f"_buf{args.n_error_buf}"
      case = f"pressure_bump_cart_vs_cyl_rphi_nr{nr:04d}_phi{args.angular_factor}_l{args.amr_max_level}{case_suffix}"
      case_dir = OUT_ROOT / case
      if case_dir.exists() and not args.force:
         raise SystemExit(f"{case_dir} exists; pass --force to overwrite")

      ref_ratio_values = " ".join(str(args.ref_ratio) for _ in range(args.amr_max_level))
      amr_overrides = [
         f"amr.max_level={args.amr_max_level}",
         "amr.blocking_factor=8",
         "amr.max_grid_size=64",
         f"amr.regrid_int={args.regrid_int}",
         "amr.grid_eff=0.85",
         f"amr.state_interp={args.state_interp}",
         f"amr.tag_threshold={args.tag_threshold:.17g}",
         "amr.total_cell_budget=1000000",
      ]
      if args.amr_max_level > 0:
         amr_overrides.extend([
            f"amr.ref_ratio={ref_ratio_values}",
            "amr.n_error_buf=" + " ".join(str(args.n_error_buf) for _ in range(args.amr_max_level)),
         ])
      common = [
         "pyfluid.physics=hydro",
         "pyfluid.ic=pressure_bump",
         "pyfluid.rho0=1.0",
         "pyfluid.pressure0=1.0",
         "pyfluid.amplitude=0.20",
         "pyfluid.bump_sigma=0.18",
         "pyfluid.v1_0=0.0",
         "pyfluid.v2_0=0.0",
         "pyfluid.v3_0=0.0",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.final_time={:.17g}".format(args.final_time),
         f"pyfluid.max_step={args.max_step}",
         "pyfluid.ng=2",
         "pyfluid.scalar_int=100000",
         f"pyfluid.plot_int={plot_int}",
         "pyfluid.checkpoint_int=-1",
         "pyfluid.physical_check_int=1",
         "pyfluid.subcycling=off",
         "pyfluid.diffusion=none",
         "pyfluid.gravity=none",
         "pyfluid.source=none",
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
            f"amr.metric_regrid={'always' if args.cart_metric_regrid else 'auto'}",
            "mesh.axes=x y",
            f"geometry.prob_lo={-args.radius:.17g} {-args.radius:.17g}",
            f"geometry.prob_hi={args.radius:.17g} {args.radius:.17g}",
            "geometry.is_periodic=0 0",
            f"amr.n_cell={ncart} {ncart}",
            *cart_amr,
            "pyfluid.bc=reflecting",
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
            "pyfluid.bc_xhi=reflecting",
            "pyfluid.bc_ylo=periodic",
            "pyfluid.bc_yhi=periodic",
            f"pyfluid.cfl={cyl_cfl:.17g}",
            f"pyfluid.bump_x0={cyl_bump_r0:.17g}",
            f"pyfluid.bump_y0={cyl_bump_phi0:.17g}",
         ],
         args.ranks,
         args.timeout,
         scalar_policy,
      )

      gamma = float(read_key_values(cyl_dir / "run.in").get("pyfluid.gamma", "1.6666666666666667"))
      sample_level = args.amr_max_level
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
      cart_amr_summary = read_amr_summary(cart_dir)
      cyl_amr_summary = read_amr_summary(cyl_dir)
      if cart_amr_summary["active_finest_level"] < min_active_level or cyl_amr_summary["active_finest_level"] < min_active_level:
         raise SystemExit(
            f"{case} did not activate requested AMR depth: "
            f"cart L{cart_amr_summary['active_finest_level']}, "
            f"cyl L{cyl_amr_summary['active_finest_level']}, "
            f"required L{min_active_level}"
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
      scalar_rows = [scalar_status(cart_dir, cart_name, scalar_policy), scalar_status(cyl_dir, cyl_name, scalar_policy)]
      write_tsv(case_dir / "scalar_checks.tsv", scalar_rows)
      if any(row["status"] != "PASS" for row in scalar_rows):
         raise SystemExit(f"{case} scalar sanity failed; see {case_dir / 'scalar_checks.tsv'}")

   write_tsv(OUT_ROOT / "cartesian_cylindrical_equivalence.tsv", all_rows)
   print(f"PASS curvilinear_cartesian_equivalence rows={len(all_rows)} out={OUT_ROOT}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
