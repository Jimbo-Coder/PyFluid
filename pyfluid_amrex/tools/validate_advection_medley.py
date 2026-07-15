#!/usr/bin/env python3
"""Compare reconstruction methods on one exact periodic contact-advection medley."""

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


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/advection_medley_1d.in"
RUN_ROOT = ROOT / "runs/current/advection_medley"
RESULTS = RUN_ROOT / "reconstruction_comparison.tsv"
PLOT = ROOT / "plots/current/scans/advection_medley_reconstruction_002.png"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class Method:
   name: str
   label: str
   recon: str


METHODS = (
   Method("pwc", "PWC", "pwc"),
   Method("pwl", "PWL-MC", "pwl"),
   Method("wenoz", "WENO-Z component", "wenoz"),
   Method("wenoz_char", "WENO-Z characteristic", "wenoz-char"),
   Method("mp5", "MP5 component", "mp5"),
   Method("mp5_char", "MP5 characteristic", "mp5-char"),
   Method("teno5", "TENO5 component", "teno5"),
   Method("teno5_char", "TENO5 characteristic", "teno5-char"),
)

FEATURES = ("sinusoid", "square", "triangle", "gaussian")


def medley_profile(phase: np.ndarray) -> np.ndarray:
   phase = np.mod(phase, 1.0)
   segment = np.minimum((4.0 * phase).astype(int), 3)
   local = 4.0 * phase - segment
   profile = np.zeros_like(phase)

   mask = segment == 0
   profile[mask] = np.sin(np.pi * local[mask]) ** 2
   mask = segment == 1
   profile[mask] = ((local[mask] >= 0.2) & (local[mask] <= 0.8)).astype(float)
   mask = segment == 2
   profile[mask] = 1.0 - np.abs(2.0 * local[mask] - 1.0)
   mask = segment == 3
   sigma = 0.12
   edge = math.exp(-0.5 * (0.5 / sigma) ** 2)
   z = (local[mask] - 0.5) / sigma
   profile[mask] = np.maximum((np.exp(-0.5 * z * z) - edge) / (1.0 - edge), 0.0)
   return profile


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fields = list(rows[0])
   with path.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def run_method(method: Method, args: argparse.Namespace) -> dict[str, object]:
   run_dir = RUN_ROOT / method.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      parameters_from_assignments((
         f"pyfluid.run_name=advection_medley_{method.name}",
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         f"amr.n_cell={args.nx}",
         f"pyfluid.final_time={args.periods}",
         f"pyfluid.recon={method.recon}",
         "pyfluid.timestep=rk4",
      )),
   )
   if latest_plotfile(run_dir) is None:
      result = run_amrex(dim=1, input_path=input_path, ranks=args.ranks)
      (run_dir / "run.log").write_text(result.stdout)
      if result.returncode != 0:
         raise RuntimeError(f"{method.name} failed; see {run_dir / 'run.log'}")

   plotfile = latest_plotfile(run_dir, require=True)
   names, time_value = parse_plot_header_1d(plotfile)
   state = read_level0_state_1d(plotfile, names)
   rho = state[names.index("rho")]
   x = (np.arange(rho.size, dtype=float) + 0.5) / rho.size
   exact = args.rho0 + args.amplitude * medley_profile(x - args.velocity * time_value)
   normalized_error = (rho - exact) / args.amplitude
   phase = np.mod(x - args.velocity * time_value, 1.0)
   segment = np.minimum((4.0 * phase).astype(int), 3)
   feature_l1 = {
      f"{name}_l1": float(np.mean(np.abs(normalized_error[segment == index])))
      for index, name in enumerate(FEATURES)
   }
   scalar = scalar_validation_row(
      run_dir,
      case=f"advection_medley_{method.name}",
      input_file=run_dir / "run.in",
      checks=CLOSED_POLICY,
      mhd=False,
      dim=1,
      ranks=args.ranks,
      kind="analytic",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{method.name} scalar validation failed: {scalar['reason']}")
   return {
      "method": method.name,
      "label": method.label,
      "nx": args.nx,
      "periods": args.periods,
      "time": time_value,
      "l1": float(np.mean(np.abs(normalized_error))),
      "l2": float(np.sqrt(np.mean(normalized_error**2))),
      "linf": float(np.max(np.abs(normalized_error))),
      **feature_l1,
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "mcells_s": scalar["avg_mcells_per_s"],
      "sim_time_per_s": scalar["avg_sim_time_per_s"],
      "run_dir": repo_relative(run_dir),
      "plotfile": repo_relative(plotfile),
      "x": x,
      "rho": rho,
      "exact": exact,
      "error": np.abs(normalized_error),
      "validation": scalar,
   }


def plot_results(rows: list[dict[str, object]], output: Path) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
   reference = rows[0]
   axes[0, 0].plot(reference["x"], reference["exact"], color="black", lw=2.0, label="exact")
   for row in rows:
      linestyle = ":" if row["method"].endswith("_char") else "-"
      zorder = 4 if row["method"].endswith("_char") else 2
      axes[0, 0].plot(row["x"], row["rho"], lw=1.1, ls=linestyle, zorder=zorder, label=row["label"])
      axes[0, 1].semilogy(
         row["x"], np.maximum(row["error"], 1.0e-16), lw=1.1, ls=linestyle, zorder=zorder, label=row["label"]
      )

   positions = np.arange(len(FEATURES), dtype=float)
   width = 0.8 / len(rows)
   for index, row in enumerate(rows):
      values = [row[f"{feature}_l1"] for feature in FEATURES]
      axes[1, 0].bar(positions + (index - 0.5 * (len(rows) - 1)) * width, values, width=width, label=row["label"])
   labels = [row["label"] for row in rows]
   axes[1, 1].plot(labels, [row["mcells_s"] for row in rows], marker="o", label="Mcells/s")
   rate_axis = axes[1, 1].twinx()
   rate_axis.plot(labels, [row["sim_time_per_s"] for row in rows], marker="s", color="tab:red", label="sim time / s")

   axes[0, 0].set(title="Density after periodic advection", xlabel="x", ylabel=r"$\rho$")
   axes[0, 1].set(title="Pointwise normalized error", xlabel="x", ylabel=r"$|\rho-\rho_{exact}|/A$")
   axes[1, 0].set(title="Feature-wise L1 error", ylabel="normalized L1", xticks=positions, xticklabels=FEATURES, yscale="log")
   axes[1, 1].set(title="Runtime throughput", ylabel="Mcells/s")
   rate_axis.set_ylabel("simulation time / real time", color="tab:red")
   axes[1, 1].tick_params(axis="x", rotation=20)
   for axis in axes.flat:
      axis.grid(alpha=0.25)
   axes[0, 0].legend(fontsize=8, ncol=2)
   axes[0, 1].legend(fontsize=8)
   axes[1, 0].legend(fontsize=8)
   fig.suptitle(f"Periodic contact-advection medley: N={rows[0]['nx']}, periods={rows[0]['periods']}, RK4/HLLC, CFL=0.9")
   fig.savefig(output, dpi=180)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--nx", type=int, default=256)
   parser.add_argument("--periods", type=float, default=2.0)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   args = parser.parse_args()
   args.rho0 = 1.0
   args.amplitude = 0.5
   args.velocity = 1.0
   if args.nx < 32 or args.periods <= 0.0:
      raise SystemExit("--nx must be at least 32 and --periods must be positive")

   rows = [run_method(method, args) for method in METHODS]
   write_tsv([{key: value for key, value in row.items() if key not in {"x", "rho", "exact", "error", "validation"}} for row in rows], RESULTS)
   if not args.skip_plot:
      plot_results(rows, PLOT)
   append_validation_history(HISTORY, [row["validation"] for row in rows])
   for row in rows:
      print(
         f"{row['label']}: L1={row['l1']:.3e} L2={row['l2']:.3e} "
         f"square={row['square_l1']:.3e} gaussian={row['gaussian_l1']:.3e} "
         f"{row['mcells_s']:.2f} Mcells/s sim_time/s={row['sim_time_per_s']:.3e}"
      )
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
