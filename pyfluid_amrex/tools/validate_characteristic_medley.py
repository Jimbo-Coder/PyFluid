#!/usr/bin/env python3
"""Compare reconstruction methods on a periodic mixed-characteristic medley."""

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


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/characteristic_medley_1d.in"
RUN_ROOT = ROOT / "runs/current/characteristic_medley"
RESULTS = RUN_ROOT / "reconstruction_comparison.tsv"
PLOT = ROOT / "plots/current/scans/characteristic_medley_reconstruction_002.png"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class Method:
   name: str
   label: str
   recon: str
   resolution_factor: int = 1


METHODS = (
   Method("pwl", "PWL-MC", "pwl"),
   Method("wenoz", "WENO-Z component", "wenoz"),
   Method("wenoz_char", "WENO-Z characteristic", "wenoz-char"),
   Method("mp5", "MP5 component", "mp5"),
   Method("mp5_char", "MP5 characteristic", "mp5-char"),
   Method("teno5", "TENO5 component", "teno5"),
   Method("teno5_char", "TENO5 characteristic", "teno5-char"),
)
REFERENCE = Method("fine_reference", "fine characteristic reference", "wenoz-char", 8)
WAVE_NAMES = ("sound_minus", "entropy", "sound_plus")


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
   with path.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def run_method(method: Method, args: argparse.Namespace) -> dict[str, object]:
   run_dir = RUN_ROOT / method.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   nx = args.nx * method.resolution_factor
   sound_speed = math.sqrt(args.gamma * args.pressure0 / args.rho0)
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      parameters_from_assignments((
         f"pyfluid.run_name=characteristic_medley_{method.name}",
         f"pyfluid.output_dir={repo_relative(run_dir)}",
         f"amr.n_cell={nx}",
         f"amr.max_grid_size={min(nx, 256)}",
         f"pyfluid.final_time={args.periods / sound_speed:.17g}",
         f"pyfluid.gamma={args.gamma:.17g}",
         f"pyfluid.rho0={args.rho0:.17g}",
         f"pyfluid.pressure0={args.pressure0:.17g}",
         f"pyfluid.amplitude={args.amplitude:.17g}",
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
   scalar = scalar_validation_row(
      run_dir,
      case=f"characteristic_medley_{method.name}",
      input_file=run_dir / "run.in",
      checks=CLOSED_POLICY,
      mhd=False,
      dim=1,
      ranks=args.ranks,
      kind="reference",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{method.name} scalar validation failed: {scalar['reason']}")
   return {
      "method": method.name,
      "label": method.label,
      "nx": nx,
      "time": time_value,
      "state": state,
      "names": names,
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "mcells_s": scalar["avg_mcells_per_s"],
      "sim_time_per_s": scalar["avg_sim_time_per_s"],
      "run_dir": repo_relative(run_dir),
      "plotfile": repo_relative(plotfile),
      "validation": scalar,
   }


def coarsen_state(state: np.ndarray, target_nx: int) -> np.ndarray:
   if state.shape[1] == target_nx:
      return state
   if state.shape[1] % target_nx != 0:
      raise ValueError(f"reference resolution {state.shape[1]} is not divisible by target {target_nx}")
   ratio = state.shape[1] // target_nx
   return state.reshape(state.shape[0], target_nx, ratio).mean(axis=2)


def characteristic_waves(
   state: np.ndarray,
   names: list[str],
   *,
   rho0: float,
   pressure0: float,
   gamma: float,
) -> dict[str, np.ndarray]:
   rho = state[names.index("rho")]
   mom1 = state[names.index("mom1")]
   mom2 = state[names.index("mom2")]
   mom3 = state[names.index("mom3")]
   energy = state[names.index("energy")]
   velocity = mom1 / rho
   kinetic = 0.5 * (mom1 * mom1 + mom2 * mom2 + mom3 * mom3) / rho
   pressure = (gamma - 1.0) * (energy - kinetic)
   sound_speed = math.sqrt(gamma * pressure0 / rho0)
   pressure_wave = (pressure - pressure0) / (sound_speed * sound_speed)
   velocity_wave = rho0 * velocity / sound_speed
   return {
      "sound_minus": 0.5 * (pressure_wave - velocity_wave),
      "entropy": rho - rho0 - pressure_wave,
      "sound_plus": 0.5 * (pressure_wave + velocity_wave),
   }


def add_error_metrics(rows: list[dict[str, object]], reference: dict[str, object], args: argparse.Namespace) -> None:
   reference_state = coarsen_state(reference["state"], args.nx)
   reference_waves = characteristic_waves(
      reference_state,
      reference["names"],
      rho0=args.rho0,
      pressure0=args.pressure0,
      gamma=args.gamma,
   )
   x = (np.arange(args.nx, dtype=float) + 0.5) / args.nx
   initial_waves = {
      "sound_minus": args.amplitude * medley_profile(x),
      "entropy": args.amplitude * medley_profile(x + 1.0 / 3.0),
      "sound_plus": args.amplitude * medley_profile(x + 2.0 / 3.0),
   }
   reference["waves"] = reference_waves
   reference["linear_return_l1"] = float(np.mean([
      np.mean(np.abs(reference_waves[name] - initial_waves[name])) / args.amplitude
      for name in WAVE_NAMES
   ]))
   for row in rows:
      waves = characteristic_waves(
         row["state"],
         row["names"],
         rho0=args.rho0,
         pressure0=args.pressure0,
         gamma=args.gamma,
      )
      errors = []
      for name in WAVE_NAMES:
         error = float(np.mean(np.abs(waves[name] - reference_waves[name])) / args.amplitude)
         row[f"{name}_l1"] = error
         errors.append(error)
      row["aggregate_l1"] = float(np.mean(errors))
      row["waves"] = waves


def plot_results(rows: list[dict[str, object]], reference: dict[str, object], output: Path, args: argparse.Namespace) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   x = (np.arange(args.nx, dtype=float) + 0.5) / args.nx
   fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
   for axis, wave_name in zip(axes.flat[:3], WAVE_NAMES, strict=True):
      axis.plot(x, reference["waves"][wave_name], color="black", lw=2.0, label="fine reference")
      for row in rows:
         style = ":" if row["method"].endswith("_char") else "-"
         axis.plot(x, row["waves"][wave_name], lw=1.1, ls=style, label=row["label"])
      axis.set(title=wave_name.replace("_", " "), xlabel="x", ylabel="amplitude")
      axis.grid(alpha=0.25)
   labels = [row["label"] for row in rows]
   axes[1, 1].semilogy(labels, [row["aggregate_l1"] for row in rows], marker="o", label="reference L1")
   speed_axis = axes[1, 1].twinx()
   speed_axis.plot(labels, [row["sim_time_per_s"] for row in rows], marker="s", color="tab:red", label="sim time / s")
   axes[1, 1].set(title="Accuracy and throughput", ylabel="normalized aggregate L1")
   speed_axis.set_ylabel("simulation time / real time", color="tab:red")
   axes[1, 1].tick_params(axis="x", rotation=20)
   axes[0, 0].legend(fontsize=8)
   fig.suptitle(
      f"Periodic mixed-characteristic medley: N={args.nx}, crossings={args.periods:g}, "
      f"A={args.amplitude:g}, RK4/HLLC"
   )
   fig.savefig(output, dpi=180)
   plt.close(fig)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--nx", type=int, default=256)
   parser.add_argument("--periods", type=float, default=1.0)
   parser.add_argument("--amplitude", type=float, default=0.1)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   args = parser.parse_args()
   args.rho0 = 1.0
   args.pressure0 = 1.0
   args.gamma = 5.0 / 3.0
   if args.nx < 32 or args.periods <= 0.0 or args.amplitude <= 0.0:
      raise SystemExit("--nx must be at least 32; --periods and --amplitude must be positive")

   rows = [run_method(method, args) for method in METHODS]
   reference = run_method(REFERENCE, args)
   add_error_metrics(rows, reference, args)
   serializable = []
   for row in rows:
      serializable.append({
         key: value for key, value in row.items()
         if key not in {"state", "names", "waves", "validation"}
      })
   write_tsv(serializable, RESULTS)
   if not args.skip_plot:
      plot_results(rows, reference, PLOT, args)
   append_validation_history(HISTORY, [row["validation"] for row in (*rows, reference)])
   by_method = {row["method"]: row for row in rows}
   if by_method["wenoz"]["aggregate_l1"] >= by_method["pwl"]["aggregate_l1"]:
      raise RuntimeError("component WENO-Z did not improve on PWL-MC")
   if by_method["wenoz_char"]["aggregate_l1"] >= 0.95 * by_method["wenoz"]["aggregate_l1"]:
      raise RuntimeError("characteristic WENO-Z did not materially improve the coupled medley")
   for row in rows:
      print(
         f"{row['label']}: L1={row['aggregate_l1']:.3e} "
         f"{row['mcells_s']:.2f} Mcells/s sim_time/s={row['sim_time_per_s']:.3e}"
      )
   print(f"fine-reference nonlinear return departure={reference['linear_return_l1']:.3e}")
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
