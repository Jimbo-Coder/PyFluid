#!/usr/bin/env python3
"""Crawl checks for mixed Cartesian-curvilinear patch evolution.

The first mixed-patch evolution gate should answer three narrow questions:

1. A localized isotropic perturbation remains as isotropic as the same
   Cartesian calculation before it reaches a patch interface.
2. Closed-run scalar budgets stay at roundoff.
3. The static outer shell remains quiet before causal contact.

This is intentionally a small HD/PWC/LLF/Euler gate. It is not the final
seam-crossing validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import (  # noqa: E402
   build_visual_patches,
   box_intersection,
   component_values,
   level_chunks,
   parse_key_value_file,
   physical_patch_edges,
)
from plot_amrex_slice import parse_plot_header  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402
from run_case import merged_params, write_resolved_input, write_run_log  # noqa: E402
from validation_common import (  # noqa: E402
   CLOSED_POLICY,
   ROOT,
   check_scalars,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/mixed_thornburg04_pressure_bump_2d.in"
RUN_ROOT = ROOT / "runs/current/mixed_multipatch/core_equivalence"
SUMMARY = RUN_ROOT / "summary.tsv"
COMPONENTS = ("rho", "mom1", "mom2", "mom3", "energy", "pressure")
SUMMARY_FIELDS = [
   "case",
   "status",
   "reason",
   "run_dir",
   "max_linf",
   "max_rel_linf",
   "pressure_symmetry_linf",
   "momentum_symmetry_linf",
   "outer_pressure_dev",
   "outer_rho_dev",
   "outer_momentum",
   "radial_pressure_std_over_peak",
]


def safe_clean(path: Path) -> None:
   resolved = path.resolve()
   runs_root = (ROOT / "runs").resolve()
   if not resolved.is_relative_to(runs_root) or resolved == runs_root:
      raise ValueError(f"refusing to clean non-run output directory: {resolved}")
   if resolved.exists():
      shutil.rmtree(resolved)


def run_case(name: str, overrides: list[str], *, ranks: int, timeout: int | None) -> tuple[Path, dict[str, object]]:
   run_dir = RUN_ROOT / name
   safe_clean(run_dir)
   full_overrides = [
      *overrides,
      f"pyfluid.run_name={name}",
      f"pyfluid.output_dir={repo_relative(run_dir)}",
   ]
   result = run_amrex(dim=2, input_path=BASE_INPUT, ranks=ranks, overrides=full_overrides, timeout=timeout)
   write_run_log(run_dir, result.command, result.stdout, result.seconds)
   params = merged_params(BASE_INPUT, full_overrides)
   input_copy = run_dir / "run.in"
   write_resolved_input(input_copy, params)
   if result.returncode != 0:
      tail = "\n".join(result.stdout.strip().splitlines()[-10:])
      return run_dir, {
         "case": name,
         "status": "FAIL",
         "reason": tail,
         "run_dir": repo_relative(run_dir),
         "seconds": result.seconds,
      }

   row = scalar_validation_row(
      run_dir,
      case=name,
      input_file=input_copy,
      checks=CLOSED_POLICY,
      mhd=False,
      dim=2,
      ranks=ranks,
      base_input=repo_relative(BASE_INPUT),
      kind="mixed_patch_crawl",
   )
   row["seconds"] = result.seconds
   return run_dir, row


def run_reconstruction_smokes(*, ranks: int, timeout: int | None) -> list[dict[str, object]]:
   """Exercise mapped-patch seam flux correction with non-default reconstruction."""
   cases = [
      ("mixed_thornburg04_recon_pwl_smoke", "pwl", "2"),
      ("mixed_thornburg04_recon_wenoz_smoke", "wenoz", "3"),
      ("mixed_thornburg04_recon_wenoz_char_smoke", "wenoz-char", "3"),
   ]
   rows = []
   for name, recon, ng in cases:
      _run_dir, row = run_case(
         name,
         [
            "pyfluid.max_step=4",
            "pyfluid.final_time=-1.0",
            "pyfluid.scalar_int=1",
            "pyfluid.plot_int=-1",
            "pyfluid.checkpoint_int=-1",
            "pyfluid.physical_check_int=1",
            "pyfluid.recon=" + recon,
            "pyfluid.ng=" + ng,
            "pyfluid.riemann=hllc",
            "pyfluid.timestep=ssprk3",
            "pyfluid.bc=reflecting",
            "pyfluid.subcycling=off",
            "amr.max_level=0",
         ],
         ranks=ranks,
         timeout=timeout,
      )
      row["case"] = name
      rows.append(row)
   return rows


def read_level0_state(plotfile: Path) -> tuple[list[str], float, np.ndarray]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 2:
      raise ValueError(f"{plotfile} is {ndim}D; expected 2D")
   chunks = level_chunks(plotfile, 0, ndim)
   shape = (
      max(hi[0] for _lo, hi, _data in chunks) + 1,
      max(hi[1] for _lo, hi, _data in chunks) + 1,
   )
   state = np.full((len(names), *shape), np.nan)
   for lo, hi, data in chunks:
      state[:, lo[0] : hi[0] + 1, lo[1] : hi[1] + 1] = data
   return names, time_value, state


def compare_core(
   mixed_run: Path,
   cartesian_run: Path,
   *,
   tolerance: float,
   gamma: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
   mixed_names, mixed_time, mixed_state = read_level0_state(latest_plotfile(mixed_run, require=True))
   cart_names, cart_time, cart_state = read_level0_state(latest_plotfile(cartesian_run, require=True))
   if mixed_names != cart_names:
      raise ValueError("mixed and Cartesian plotfiles have different component names")
   if cart_state.shape[1:] != (16, 16):
      raise ValueError(f"Cartesian reference shape is {cart_state.shape[1:]}, expected 16x16")

   mixed_core = mixed_state[:, :16, :16]
   if not np.all(np.isfinite(mixed_core)):
      raise ValueError(f"{latest_plotfile(mixed_run, require=True)} did not fill the central owned 16x16 patch")
   if not np.all(np.isfinite(cart_state)):
      raise ValueError(
         f"{latest_plotfile(cartesian_run, require=True)} did not fill the Cartesian 16x16 reference"
      )
   component_rows: list[dict[str, object]] = []
   failures = []
   for component in COMPONENTS:
      mixed_values = component_values(mixed_core, mixed_names, component, gamma)
      cart_values = component_values(cart_state, cart_names, component, gamma)
      diff = mixed_values - cart_values
      linf = float(np.max(np.abs(diff)))
      l2 = float(np.sqrt(np.mean(diff * diff)))
      scale = max(float(np.max(np.abs(cart_values))), 1.0e-30)
      rel_linf = linf / scale
      rel_l2 = l2 / scale
      status = "PASS" if linf <= tolerance else "FAIL"
      if status != "PASS":
         failures.append(f"{component} linf={linf:.3e}")
      component_rows.append(
         {
            "component": component,
            "linf": linf,
            "l2": l2,
            "rel_linf": rel_linf,
            "rel_l2": rel_l2,
            "status": status,
         }
      )

   time_error = abs(mixed_time - cart_time)
   if time_error > 1.0e-14:
      failures.append(f"time mismatch {time_error:.3e}")
   row = {
      "case": "central_core_vs_single_cartesian",
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "mixed_time": mixed_time,
      "cartesian_time": cart_time,
      "max_linf": max(float(item["linf"]) for item in component_rows),
      "max_rel_linf": max(float(item["rel_linf"]) for item in component_rows),
      "run_dir": repo_relative(RUN_ROOT),
   }
   return row, component_rows


def central_symmetry_row(mixed_run: Path, *, tolerance: float, gamma: float) -> dict[str, object]:
   names, _time_value, state = read_level0_state(latest_plotfile(mixed_run, require=True))
   core = state[:, :16, :16]
   pressure = component_values(core, names, "pressure", gamma)
   mom1 = component_values(core, names, "mom1", gamma)
   mom2 = component_values(core, names, "mom2", gamma)
   pressure_defects = (
      np.max(np.abs(pressure - pressure[::-1, :])),
      np.max(np.abs(pressure - pressure[:, ::-1])),
      np.max(np.abs(pressure - pressure.T)),
   )
   momentum_defects = (
      np.max(np.abs(mom1 + mom1[::-1, :])),
      np.max(np.abs(mom2 + mom2[:, ::-1])),
      np.max(np.abs(mom1 - mom2.T)),
   )
   pressure_linf = float(max(pressure_defects))
   momentum_linf = float(max(momentum_defects))
   failures = []
   if pressure_linf > tolerance:
      failures.append(f"pressure symmetry={pressure_linf:.3e}")
   if momentum_linf > tolerance:
      failures.append(f"momentum symmetry={momentum_linf:.3e}")
   return {
      "case": "central_core_isotropy_symmetry",
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "pressure_symmetry_linf": pressure_linf,
      "momentum_symmetry_linf": momentum_linf,
      "run_dir": repo_relative(mixed_run),
   }


def physical_samples(run_dir: Path, component: str, *, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   plotfile = latest_plotfile(run_dir, require=True)
   names, ndim, _time_value = parse_plot_header(plotfile)
   params = parse_key_value_file(run_dir / "run.in")
   patches = build_visual_patches(params, ndim, refinement=1)
   chunks = level_chunks(plotfile, 0, ndim)
   x_parts = []
   y_parts = []
   value_parts = []
   for lo, hi, data in chunks:
      values = component_values(data, names, component, gamma)
      for patch in patches:
         intersection = box_intersection(lo, hi, patch)
         if intersection is None:
            continue
         inter_lo, inter_hi = intersection
         local_x = slice(inter_lo[0] - lo[0], inter_hi[0] - lo[0] + 1)
         local_y = slice(inter_lo[1] - lo[1], inter_hi[1] - lo[1] + 1)
         edge_x, edge_y = physical_patch_edges(patch, inter_lo, inter_hi)
         center_x = 0.25 * (edge_x[:-1, :-1] + edge_x[1:, :-1] + edge_x[:-1, 1:] + edge_x[1:, 1:])
         center_y = 0.25 * (edge_y[:-1, :-1] + edge_y[1:, :-1] + edge_y[:-1, 1:] + edge_y[1:, 1:])
         tile = values[local_x, local_y].T
         x_parts.append(center_x.ravel())
         y_parts.append(center_y.ravel())
         value_parts.append(tile.ravel())
   if not value_parts:
      raise ValueError(f"no physical samples found for {component}")
   return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(value_parts)


def radial_scatter(radius: np.ndarray, values: np.ndarray, *, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   edges = np.linspace(float(np.min(radius)), float(np.max(radius)), nbins + 1)
   centers = 0.5 * (edges[:-1] + edges[1:])
   means = np.full(nbins, np.nan)
   stds = np.full(nbins, np.nan)
   for index in range(nbins):
      if index == nbins - 1:
         mask = (radius >= edges[index]) & (radius <= edges[index + 1])
      else:
         mask = (radius >= edges[index]) & (radius < edges[index + 1])
      if np.any(mask):
         means[index] = float(np.mean(values[mask]))
         stds[index] = float(np.std(values[mask]))
   return centers, means, stds


def mixed_isotropy_and_quietness(
   mixed_run: Path,
   *,
   pressure0: float,
   rho0: float,
   outer_radius: float,
   quiet_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
   x, y, pressure = physical_samples(mixed_run, "pressure", gamma=5.0 / 3.0)
   _x, _y, rho = physical_samples(mixed_run, "rho", gamma=5.0 / 3.0)
   _x, _y, mom1 = physical_samples(mixed_run, "mom1", gamma=5.0 / 3.0)
   _x, _y, mom2 = physical_samples(mixed_run, "mom2", gamma=5.0 / 3.0)
   radius = np.sqrt(x * x + y * y)
   pressure_perturbation = pressure - pressure0
   centers, means, stds = radial_scatter(radius, pressure_perturbation, nbins=24)
   active = np.isfinite(stds)
   peak = max(float(np.max(np.abs(pressure_perturbation))), 1.0e-30)
   max_radial_std = float(np.max(stds[active])) if np.any(active) else math.nan
   radial_std_over_peak = max_radial_std / peak if math.isfinite(max_radial_std) else math.nan

   quiet = radius >= outer_radius
   quiet_pressure = float(np.max(np.abs(pressure[quiet] - pressure0))) if np.any(quiet) else math.nan
   quiet_rho = float(np.max(np.abs(rho[quiet] - rho0))) if np.any(quiet) else math.nan
   quiet_momentum = float(np.max(np.sqrt(mom1[quiet] * mom1[quiet] + mom2[quiet] * mom2[quiet]))) if np.any(quiet) else math.nan
   failures = []
   if not math.isfinite(quiet_pressure) or quiet_pressure > quiet_tolerance:
      failures.append(f"outer pressure dev={quiet_pressure:.3e}")
   if not math.isfinite(quiet_rho) or quiet_rho > quiet_tolerance:
      failures.append(f"outer rho dev={quiet_rho:.3e}")
   if not math.isfinite(quiet_momentum) or quiet_momentum > quiet_tolerance:
      failures.append(f"outer momentum={quiet_momentum:.3e}")

   details = [
      {
         "radius": float(center),
         "pressure_perturbation_mean": float(mean),
         "pressure_perturbation_std": float(std),
      }
      for center, mean, std in zip(centers, means, stds)
      if math.isfinite(std)
   ]
   row = {
      "case": "mixed_isotropy_and_outer_quietness",
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "max_radial_pressure_std": max_radial_std,
      "radial_pressure_std_over_peak": radial_std_over_peak,
      "outer_pressure_dev": quiet_pressure,
      "outer_rho_dev": quiet_rho,
      "outer_momentum": quiet_momentum,
      "run_dir": repo_relative(mixed_run),
   }
   return row, details


def plot_checks(
   mixed_run: Path,
   cartesian_run: Path,
   component_rows: list[dict[str, object]],
   radial_rows: list[dict[str, object]],
) -> Path:
   plot_dir = mixed_run / "plots"
   plot_dir.mkdir(parents=True, exist_ok=True)
   output = plot_dir / "mixed_patch_crawl_checks.png"
   fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)

   components = [str(row["component"]) for row in component_rows]
   linf = [float(row["linf"]) for row in component_rows]
   plotted_linf = [max(value, 1.0e-18) for value in linf]
   axes[0].bar(components, plotted_linf, color="#4a6f8a")
   axes[0].set_yscale("log")
   axes[0].axhline(1.0e-12, color="0.35", linestyle=":", linewidth=1.2, label="1e-12 check")
   axes[0].set_ylabel("Linf mixed core - Cartesian")
   axes[0].set_title("Central patch equivalence")
   axes[0].tick_params(axis="x", rotation=35)
   axes[0].legend(frameon=False, fontsize=8)
   if max(linf) == 0.0:
      axes[0].text(0.5, 0.82, "all differences exactly 0", transform=axes[0].transAxes, ha="center")

   radii = [float(row["radius"]) for row in radial_rows]
   means = [float(row["pressure_perturbation_mean"]) for row in radial_rows]
   stds = [float(row["pressure_perturbation_std"]) for row in radial_rows]
   axes[1].plot(radii, means, marker="o", markersize=3, label="radial mean")
   axes[1].plot(radii, stds, marker="s", markersize=3, linestyle=":", label="radial std")
   axes[1].set_yscale("symlog", linthresh=1.0e-16)
   axes[1].set_xlabel("physical radius")
   axes[1].set_ylabel("pressure perturbation")
   axes[1].set_title("Mixed-patch radial profile")
   axes[1].legend(frameon=False, fontsize=8)
   fig.suptitle("Mixed Thornburg04 crawl: HD PWC LLF Euler, CFL 0.9, one clamped step")
   fig.savefig(output, dpi=180)
   plt.close(fig)
   return output


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   if not rows:
      return
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=90)
   parser.add_argument("--tolerance", type=float, default=1.0e-12)
   parser.add_argument("--quiet-tolerance", type=float, default=1.0e-12)
   parser.add_argument("--force", action="store_true", help="Accepted for validation-suite consistency.")
   args = parser.parse_args()

   common = [
      "pyfluid.max_step=100",
      "pyfluid.final_time=0.03",
      "pyfluid.scalar_int=1",
      "pyfluid.plot_int=100",
      "pyfluid.checkpoint_int=-1",
      "pyfluid.physical_check_int=1",
      "pyfluid.amplitude=1.0e-3",
      "pyfluid.bump_sigma=0.12",
      "pyfluid.bump_x0=0.0",
      "pyfluid.bump_y0=0.0",
      "pyfluid.bump_z0=0.0",
      "pyfluid.recon=pwc",
      "pyfluid.riemann=llf",
      "pyfluid.timestep=euler",
      "pyfluid.bc=reflecting",
      "pyfluid.subcycling=off",
      "amr.max_level=0",
   ]
   mixed_run, mixed_scalar_row = run_case(
      "mixed_thornburg04_core_probe",
      common,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   cart_run, cart_scalar_row = run_case(
      "single_cartesian_core_probe",
      [
         *common,
         "mesh.patch_system=single",
         "geometry.prob_lo=-1.0 -1.0",
         "geometry.prob_hi=1.0 1.0",
         "geometry.is_periodic=0 0",
         "amr.n_cell=16 16",
         "amr.max_grid_size=16",
         "amr.blocking_factor=4",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )

   rows: list[dict[str, object]] = [
      {"case": mixed_scalar_row["case"], "status": mixed_scalar_row["status"], "reason": mixed_scalar_row["reason"], "run_dir": mixed_scalar_row["run_dir"]},
      {"case": cart_scalar_row["case"], "status": cart_scalar_row["status"], "reason": cart_scalar_row["reason"], "run_dir": cart_scalar_row["run_dir"]},
   ]
   rows.extend(run_reconstruction_smokes(ranks=args.ranks, timeout=args.timeout))
   if mixed_scalar_row["status"] == "PASS" and cart_scalar_row["status"] == "PASS":
      core_row, component_rows = compare_core(mixed_run, cart_run, tolerance=args.tolerance, gamma=5.0 / 3.0)
      symmetry_row = central_symmetry_row(mixed_run, tolerance=args.tolerance, gamma=5.0 / 3.0)
      quiet_row, radial_rows = mixed_isotropy_and_quietness(
         mixed_run,
         pressure0=1.0,
         rho0=1.0,
         outer_radius=1.2,
         quiet_tolerance=args.quiet_tolerance,
      )
      rows.extend([core_row, symmetry_row, quiet_row])
      write_detail(RUN_ROOT / "component_errors.tsv", component_rows)
      write_detail(RUN_ROOT / "radial_pressure_scatter.tsv", radial_rows)
      plot_path = plot_checks(mixed_run, cart_run, component_rows, radial_rows)
      print(f"wrote {repo_relative(plot_path)}")

   write_tsv(SUMMARY, rows, SUMMARY_FIELDS)
   for row in rows:
      reason = f"\t{row['reason']}" if row.get("reason") else ""
      print(f"{row['status']}\t{row['case']}{reason}")
   print(f"summary: {repo_relative(SUMMARY)}")

   # Keep the scalar sanity checks explicit in the script even though
   # scalar_validation_row already recorded the detailed report row.
   scalar_failures = []
   scalar_failures.extend(check_scalars(read_scalar_rows(mixed_run), performance=True))
   scalar_failures.extend(check_scalars(read_scalar_rows(cart_run), performance=True))
   return 0 if all(row["status"] == "PASS" for row in rows) and not scalar_failures else 1


if __name__ == "__main__":
   raise SystemExit(main())
