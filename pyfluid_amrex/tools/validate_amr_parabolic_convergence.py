#!/usr/bin/env python3
"""Check explicit/implicit viscous convergence across a fixed AMR interface."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotfile_reader import (
   latest_plotfile,
   level_chunks,
   level_count,
   level_index_boxes,
   parse_plot_bounds,
   parse_plot_header,
)
from validation_common import (
   DIFFUSIVE_POLICY,
   ROOT,
   append_validation_history,
   check_scalar_rows,
   materialize_run_input,
   read_scalar_rows,
   repo_relative,
   run_mpi,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/hydro_shear_wave_x_1d.in"
EXE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_1d"
RUN_ROOT = ROOT / "runs/current/amr_parabolic_convergence"
OUTPUT_ROOT = ROOT / "plots/current/scans"
RESULTS = OUTPUT_ROOT / "amr_parabolic_convergence.tsv"
PLOT = OUTPUT_ROOT / "amr_parabolic_convergence_001.png"
HISTORY = ROOT / "runs/current/validation_history.tsv"
RESOLUTIONS = (32, 64, 128, 256)
METHODS = ("explicit", "implicit")


def write_input(method: str, resolution: int, run_dir: Path, args: argparse.Namespace) -> Path:
   return materialize_run_input(
      BASE_INPUT,
      run_dir / "run.in",
      {
         "pyfluid.run_name": run_dir.name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "amr.n_cell": str(resolution),
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.max_grid_size": "64",
         "amr.blocking_factor": "8",
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "2",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_fixed_levels": "1",
         "amr.initial_refine_box_lo": "0.25",
         "amr.initial_refine_box_hi": "0.75",
         "amr.state_interp": "quartic-conservative",
         "pyfluid.final_time": f"{args.final_time:.17g}",
         "pyfluid.cfl": f"{args.cfl:.17g}",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
         "pyfluid.diffusion": method,
         "pyfluid.viscosity": f"{args.viscosity:.17g}",
         "pyfluid.diffusion_tolerance": "1.0e-12",
         "pyfluid.physical_check_int": "1",
         "pyfluid.scalar_int": "1",
         "pyfluid.plot_int": "1",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
      },
   )


def composite_error(plotfile: Path, amplitude: float, wave_number: float, viscosity: float) -> tuple[float, float]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 1 or level_count(plotfile) != 2:
      raise ValueError(f"expected a two-level 1D hierarchy, got ndim={ndim}, levels={level_count(plotfile)}")
   index = {name: component for component, name in enumerate(names)}
   problo, probhi = parse_plot_bounds(plotfile)
   domain_length = probhi[0] - problo[0]
   wave_k = 2.0 * math.pi * wave_number / domain_length
   exact_amplitude = amplitude * math.exp(-viscosity * wave_k * wave_k * time_value)

   coarse_size = max(hi[0] for _lo, hi, _data in level_chunks(plotfile, 0, ndim)) + 1
   covered = np.zeros(coarse_size, dtype=bool)
   for lo, hi in level_index_boxes(plotfile, 1, ndim):
      covered[lo[0] // 2 : (hi[0] + 1) // 2] = True

   weighted_square_error = 0.0
   measured_mode = 0.0
   total_volume = 0.0
   for level in range(2):
      level_size = coarse_size * (2**level)
      dx = domain_length / level_size
      for lo, hi, state in level_chunks(plotfile, level, ndim, names=names):
         rho = state[index["rho"]]
         velocity = state[index["mom2"]] / rho
         cell_indices = np.arange(lo[0], hi[0] + 1)
         active = np.ones(cell_indices.size, dtype=bool)
         if level == 0:
            active &= ~covered[cell_indices]
         x = problo[0] + (cell_indices + 0.5) * dx
         mode = np.sin(wave_k * (x - problo[0]))
         exact = exact_amplitude * mode
         weighted_square_error += float(np.sum((velocity[active] - exact[active]) ** 2)) * dx
         measured_mode += float(np.sum(velocity[active] * mode[active])) * dx
         total_volume += float(np.count_nonzero(active)) * dx

   l2_error = math.sqrt(weighted_square_error / total_volume)
   measured_amplitude = 2.0 * measured_mode / total_volume
   return l2_error, abs(measured_amplitude - exact_amplitude)


def run_case(method: str, resolution: int, args: argparse.Namespace) -> dict[str, object]:
   run_dir = RUN_ROOT / f"{method}_n{resolution}"
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   if latest_plotfile(run_dir) is None:
      completed = run_mpi(EXE, [write_input(method, resolution, run_dir, args)], ranks=args.ranks)
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{method} N={resolution} failed; see {run_dir / 'run.log'}")
   plotfile = latest_plotfile(run_dir, require=True)
   l2_error, amplitude_error = composite_error(plotfile, args.amplitude, args.wave_number, args.viscosity)
   scalar_rows = read_scalar_rows(run_dir)
   failures = check_scalar_rows(scalar_rows, DIFFUSIVE_POLICY, mhd=False)
   if failures:
      raise RuntimeError(f"{method} N={resolution}: {'; '.join(failures)}")
   return {
      "method": method,
      "resolution": resolution,
      "finest_dx": 0.5 / resolution,
      "l2_error": l2_error,
      "amplitude_error": amplitude_error,
      "expected_order": 2.0 if method == "explicit" else 1.0,
      "order": math.nan,
      "run_dir": repo_relative(run_dir),
   }


def add_orders(rows: list[dict[str, object]]) -> None:
   for method in METHODS:
      method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["resolution"]))
      for previous, current in zip(method_rows, method_rows[1:]):
         current["order"] = math.log(float(previous["l2_error"]) / float(current["l2_error"]), 2.0)


def plot_results(rows: list[dict[str, object]]) -> None:
   fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
   for method, marker in (("explicit", "o"), ("implicit", "s")):
      selected = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["finest_dx"]), reverse=True)
      dx = np.asarray([float(row["finest_dx"]) for row in selected])
      error = np.asarray([float(row["l2_error"]) for row in selected])
      ax.loglog(dx, error, marker=marker, linewidth=2.0, label=method)
   reference_rows = sorted((row for row in rows if row["method"] == "explicit"), key=lambda row: float(row["finest_dx"]), reverse=True)
   dx_ref = np.asarray([float(row["finest_dx"]) for row in reference_rows])
   anchor = float(reference_rows[0]["l2_error"])
   ax.loglog(dx_ref, anchor * (dx_ref / dx_ref[0]) ** 2, ":", color="black", linewidth=1.7, label=r"$\Delta x^2$")
   implicit_rows = sorted((row for row in rows if row["method"] == "implicit"), key=lambda row: float(row["finest_dx"]), reverse=True)
   implicit_anchor = float(implicit_rows[0]["l2_error"])
   ax.loglog(
      dx_ref,
      implicit_anchor * (dx_ref / dx_ref[0]),
      "--",
      color="0.35",
      linewidth=1.5,
      label=r"$\Delta x$",
   )
   ax.set_xlabel("finest AMR cell width")
   ax.set_ylabel("absolute composite L2 velocity error")
   ax.grid(True, which="both", alpha=0.25)
   ax.legend()
   ax.set_title("AMR parabolic convergence across a fixed coarse-fine interface")
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(PLOT, dpi=180)
   plt.close(fig)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser()
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--final-time", type=float, default=0.05)
   parser.add_argument("--viscosity", type=float, default=0.01)
   parser.add_argument("--amplitude", type=float, default=1.0e-3)
   parser.add_argument("--wave-number", type=float, default=1.0)
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   rows = [run_case(method, resolution, args) for method in METHODS for resolution in RESOLUTIONS]
   add_orders(rows)
   failures = []
   for method in METHODS:
      orders = [float(row["order"]) for row in rows if row["method"] == method and math.isfinite(float(row["order"]))]
      errors = [float(row["l2_error"]) for row in rows if row["method"] == method]
      if any(fine >= coarse for coarse, fine in zip(errors, errors[1:])):
         failures.append(f"{method} composite error is not monotonically decreasing")
      minimum_order = 1.5 if method == "explicit" else 0.8
      if not orders or orders[-1] < minimum_order:
         failures.append(
            f"{method} finest order {orders[-1] if orders else math.nan:.3f} < {minimum_order:.1f}"
         )
   write_tsv(
      RESULTS,
      rows,
      ("method", "resolution", "finest_dx", "l2_error", "amplitude_error", "expected_order", "order", "run_dir"),
   )
   plot_results(rows)
   summaries = []
   for method in METHODS:
      method_rows = [row for row in rows if row["method"] == method]
      summaries.append({
         "kind": "amr",
         "suite": "amr_parabolic_convergence",
         "step": "amr_parabolic_convergence",
         "case": method,
         "label": f"AMR {method} viscosity convergence",
         "status": "FAIL" if any(item.startswith(method) for item in failures) else "PASS",
         "reason": "; ".join(item for item in failures if item.startswith(method)),
         "policy": DIFFUSIVE_POLICY.name,
         "ranks": args.ranks,
         "description": "two-level shear-wave diffusion across fixed coarse-fine interfaces",
      })
      print(f"{summaries[-1]['status']} {method}: orders=" + ", ".join(f"{float(row['order']):.3f}" for row in method_rows[1:]))
   append_validation_history(HISTORY, summaries)
   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(PLOT)}")
   if failures:
      raise SystemExit("; ".join(failures))


if __name__ == "__main__":
   main()
