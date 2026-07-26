#!/usr/bin/env python3
"""Measure smooth spatial convergence of high-order reconstruction methods."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plotfile_reader import latest_plotfile, parse_plot_header_1d, read_level0_state_1d
from validation_common import (
   CLOSED_POLICY,
   ROOT,
   append_validation_history,
   materialize_run_input,
   parameters_from_assignments,
   repo_relative,
   run_amrex,
   scalar_validation_row,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_1d.in"
RUN_ROOT = ROOT / "runs/current/reconstruction_convergence"
RESULTS = RUN_ROOT / "reconstruction_convergence.tsv"
PLOT = ROOT / "plots/current/scans/reconstruction_convergence_001.png"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class Method:
   name: str
   label: str
   minimum_order: float


METHODS = (
   Method("wenoz", "WENO-Z component", 4.0),
   Method("wenoz-char", "WENO-Z characteristic", 4.0),
   Method("mp5", "MP5 component", 4.0),
   Method("mp5-char", "MP5 characteristic", 4.0),
   Method("teno5", "TENO5 component", 4.0),
   Method("teno5-char", "TENO5 characteristic", 4.0),
   Method("parabolic", "parabolic component", 3.5),
   Method("parabolic-char", "parabolic characteristic", 3.5),
   Method("ppm4", "PPM4 component", 2.0),
   Method("ppm4-char", "PPM4 characteristic", 2.0),
   Method("ppmx", "PPMX component", 3.5),
   Method("ppmx-char", "PPMX characteristic", 3.5),
)


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def run_point(method: Method, nx: int, args: argparse.Namespace) -> dict[str, object]:
   run_dir = RUN_ROOT / method.name.replace("-", "_") / f"nx{nx:04d}"
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      parameters_from_assignments((
         f"pyfluid.run_name=reconstruction_convergence_{method.name}_nx{nx}",
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         f"amr.n_cell={nx}",
         f"amr.max_grid_size={min(nx, 256)}",
         "pyfluid.max_step=1000000",
         f"pyfluid.final_time={args.final_time:.17g}",
         "pyfluid.ng=3",
         f"pyfluid.cfl={args.cfl:.17g}",
         f"pyfluid.rho0={args.rho0:.17g}",
         f"pyfluid.pressure0={args.pressure0:.17g}",
         f"pyfluid.amplitude={args.amplitude:.17g}",
         f"pyfluid.v1_0={args.velocity:.17g}",
         "pyfluid.wave_number=1",
         f"pyfluid.recon={method.name}",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=rk4",
         "pyfluid.scalar_int=32",
         "pyfluid.plot_int=1000000",
         "pyfluid.checkpoint_int=-1",
      )),
   )
   if latest_plotfile(run_dir) is None:
      result = run_amrex(dim=1, input_path=input_path, ranks=args.ranks)
      (run_dir / "run.log").write_text(result.stdout)
      if result.returncode != 0:
         raise RuntimeError(f"{method.name} nx={nx} failed; see {run_dir / 'run.log'}")

   plotfile = latest_plotfile(run_dir, require=True)
   names, time_value = parse_plot_header_1d(plotfile)
   state = read_level0_state_1d(plotfile, names)
   rho = state[names.index("rho")]
   x = (np.arange(nx, dtype=float) + 0.5) / nx
   exact = args.rho0 + args.amplitude * np.sin(2.0 * math.pi * (x - args.velocity * time_value))
   error = (rho - exact) / args.amplitude
   scalar = scalar_validation_row(
      run_dir,
      case=f"reconstruction_convergence_{method.name}_nx{nx}",
      input_file=run_dir / "run.in",
      checks=CLOSED_POLICY,
      mhd=False,
      dim=1,
      ranks=args.ranks,
      kind="analytic",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{method.name} nx={nx} scalar validation failed: {scalar['reason']}")
   return {
      "method": method.name,
      "label": method.label,
      "nx": nx,
      "dx": 1.0 / nx,
      "l1": float(np.mean(np.abs(error))),
      "l2": float(np.sqrt(np.mean(error * error))),
      "linf": float(np.max(np.abs(error))),
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "mcells_s": scalar["avg_mcells_per_s"],
      "sim_time_per_s": scalar["avg_sim_time_per_s"],
      "run_dir": repo_relative(run_dir),
      "validation": scalar,
   }


def add_orders(rows: list[dict[str, object]]) -> None:
   for method in METHODS:
      selected = [row for row in rows if row["method"] == method.name]
      selected.sort(key=lambda row: int(row["nx"]))
      previous = None
      for row in selected:
         row["order_l1"] = "" if previous is None else math.log(
            float(previous["l1"]) / float(row["l1"]), 2.0
         )
         previous = row


def plot_results(rows: list[dict[str, object]], output: Path) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
   for method in METHODS:
      selected = sorted((row for row in rows if row["method"] == method.name), key=lambda row: int(row["nx"]))
      dx = np.asarray([float(row["dx"]) for row in selected])
      error = np.asarray([float(row["l1"]) for row in selected])
      axes[0].loglog(dx, error, marker="o", linestyle=":" if method.name.endswith("char") else "-", label=method.label)
   reference_rows = sorted((row for row in rows if row["method"] == "wenoz"), key=lambda row: int(row["nx"]))
   dx = np.asarray([float(row["dx"]) for row in reference_rows])
   reference = float(reference_rows[0]["l1"]) * (dx / dx[0]) ** 5
   axes[0].loglog(dx, reference, color="black", linestyle="--", label=r"$\Delta x^5$")
   finest = [row for row in rows if int(row["nx"]) == max(int(item["nx"]) for item in rows)]
   labels = [str(row["label"]) for row in finest]
   axes[1].plot(labels, [float(row["mcells_s"]) for row in finest], marker="o", label="Mcells/s")
   rate_axis = axes[1].twinx()
   rate_axis.plot(labels, [float(row["sim_time_per_s"]) for row in finest], marker="s", color="tab:red", label="sim time / s")
   axes[0].set(title="Smooth contact-wave convergence", xlabel=r"$\Delta x$", ylabel="normalized L1 error")
   axes[0].invert_xaxis()
   axes[1].set(title="Finest-grid throughput", ylabel="Mcells/s")
   rate_axis.set_ylabel("simulation time / real time", color="tab:red")
   axes[1].tick_params(axis="x", rotation=25)
   for axis in axes:
      axis.grid(alpha=0.25)
   axes[0].legend(fontsize=8)
   fig.suptitle("Exact periodic contact wave: RK4/HLLC")
   fig.savefig(output, dpi=180)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128, 256])
   parser.add_argument("--final-time", type=float, default=0.1)
   parser.add_argument("--cfl", type=float, default=0.05)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument(
      "--method",
      dest="method_names",
      action="append",
      choices=tuple(method.name for method in METHODS),
      help="reconstruction method; repeatable (default: all)",
   )
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   args = parser.parse_args()
   args.rho0 = 1.0
   args.pressure0 = 1.0
   args.amplitude = 0.1
   args.velocity = 1.0
   if min(args.resolutions) < 16 or args.final_time <= 0.0 or args.cfl <= 0.0:
      raise SystemExit("resolutions must be at least 16; final time and CFL must be positive")

   selected_names = set(args.method_names or (method.name for method in METHODS))
   selected_methods = tuple(
      method for method in METHODS if method.name in selected_names
   )
   rows = [
      run_point(method, nx, args)
      for method in selected_methods
      for nx in args.resolutions
   ]
   add_orders(rows)
   serializable = [{key: value for key, value in row.items() if key != "validation"} for row in rows]
   write_tsv(serializable, RESULTS)
   if not args.skip_plot:
      plot_results(rows, PLOT)
   append_validation_history(HISTORY, [row["validation"] for row in rows])
   for method in selected_methods:
      selected = sorted((row for row in rows if row["method"] == method.name), key=lambda row: int(row["nx"]))
      orders = [float(row["order_l1"]) for row in selected[1:]]
      if min(orders[-2:]) < method.minimum_order:
         raise RuntimeError(f"{method.label} did not retain high-order smooth convergence: {orders}")
      print(f"{method.label}: orders={','.join(f'{order:.2f}' for order in orders)} finest_L1={selected[-1]['l1']:.3e}")
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
