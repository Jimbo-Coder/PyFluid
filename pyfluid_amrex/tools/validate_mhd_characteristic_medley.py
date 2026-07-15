#!/usr/bin/env python3
"""Compare reconstruction methods on a periodic mixed-mode ideal-MHD medley."""

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
   read_key_values,
   read_tsv,
   repo_relative,
   run_amrex,
   scalar_validation_row,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/mhd_characteristic_medley_1d.in"
RUN_ROOT = ROOT / "runs/current/mhd_characteristic_medley"
RESULTS = RUN_ROOT / "reconstruction_comparison.tsv"
AMR_RESULTS = RUN_ROOT / "amr_subcycling_comparison.tsv"
PLOT = ROOT / "plots/current/scans/mhd_characteristic_medley_reconstruction_001.png"
HISTORY = ROOT / "runs/current/validation_history.tsv"
FIELD_NAMES = ("rho", "v1", "v2", "v3", "pressure", "b2", "b3")


@dataclass(frozen=True)
class Method:
   name: str
   label: str
   recon: str
   resolution_factor: int = 1
   max_level: int = 0
   subcycling: str = "off"


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
AMR_METHODS = (
   Method("mp5_amr_single_rate", "MP5 AMR single-rate", "mp5", max_level=3),
   Method("mp5_amr_subcycling", "MP5 AMR subcycling", "mp5", max_level=3, subcycling="on"),
)


def write_tsv(rows: list[dict[str, object]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fields = sorted({key for row in rows for key in row})
   with path.open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def run_method(method: Method, args: argparse.Namespace) -> dict[str, object]:
   run_dir = RUN_ROOT / method.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   base_nx = args.amr_nx if method.max_level else args.nx
   nx = base_nx * method.resolution_factor
   assignments = [
      f"pyfluid.run_name=mhd_characteristic_medley_{method.name}",
      f"pyfluid.output_dir={repo_relative(run_dir)}",
      f"amr.n_cell={nx}",
      f"amr.max_grid_size={min(nx, 256)}",
      f"amr.blocking_factor={min(8, nx)}",
      f"amr.max_level={method.max_level}",
      f"pyfluid.final_time={args.final_time:.17g}",
      f"pyfluid.amplitude={args.amplitude:.17g}",
      f"pyfluid.recon={method.recon}",
      f"pyfluid.subcycling={method.subcycling}",
      "pyfluid.timestep=rk4",
   ]
   if method.max_level:
      assignments.extend((
         "amr.ref_ratio=2 2 2",
         "amr.n_error_buf=2 2 2",
         "amr.regrid_int=2",
         "amr.regrid_mode=aligned",
         "amr.tag_metric=lohner",
         "amr.min_feature_length=1.0e-6",
         "amr.total_cell_budget=20000",
         "pyfluid.subcycling_steps=local_cfl",
      ))
   input_path = materialize_run_input(
      BASE_INPUT,
      run_dir / "input.in",
      parameters_from_assignments(tuple(assignments)),
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
      case=f"mhd_characteristic_medley_{method.name}",
      input_file=run_dir / "run.in",
      checks=CLOSED_POLICY,
      mhd=True,
      dim=1,
      ranks=args.ranks,
      kind="reference",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{method.name} scalar validation failed: {scalar['reason']}")
   peak_level = 0
   amr_rows = read_tsv(run_dir / "amr_levels.tsv")
   if amr_rows:
      peak_level = max(int(float(row.get("active_finest_level", "0"))) for row in amr_rows)
   return {
      "method": method.name,
      "label": method.label,
      "nx": nx,
      "max_level": method.max_level,
      "peak_level": peak_level,
      "subcycling": method.subcycling,
      "time": time_value,
      "state": state,
      "names": names,
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "divb_l1": scalar.get("divb_l1", math.nan),
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


def primitive_fields(state: np.ndarray, names: list[str], gamma: float) -> dict[str, np.ndarray]:
   rho = state[names.index("rho")]
   momenta = np.stack([state[names.index(f"mom{axis}")] for axis in (1, 2, 3)])
   magnetic = np.stack([state[names.index(f"b{axis}")] for axis in (1, 2, 3)])
   velocity = momenta / rho
   kinetic = 0.5 * np.sum(momenta * momenta, axis=0) / rho
   magnetic_energy = 0.5 * np.sum(magnetic * magnetic, axis=0)
   pressure = (gamma - 1.0) * (state[names.index("energy")] - kinetic - magnetic_energy)
   return {
      "rho": rho,
      "v1": velocity[0],
      "v2": velocity[1],
      "v3": velocity[2],
      "pressure": pressure,
      "b2": magnetic[1],
      "b3": magnetic[2],
   }


def add_error_metrics(
   rows: list[dict[str, object]],
   reference: dict[str, object],
   args: argparse.Namespace,
) -> None:
   reference_state = coarsen_state(reference["state"], args.nx)
   reference_fields = primitive_fields(reference_state, reference["names"], args.gamma)
   reference["fields"] = reference_fields
   backgrounds = {
      "rho": args.rho0,
      "v1": 0.0,
      "v2": 0.0,
      "v3": 0.0,
      "pressure": args.pressure0,
      "b2": args.b2_0,
      "b3": args.b3_0,
   }
   scales = {
      name: max(float(np.mean(np.abs(reference_fields[name] - backgrounds[name]))), args.amplitude * 1.0e-4)
      for name in FIELD_NAMES
   }
   for row in rows:
      fields = primitive_fields(row["state"], row["names"], args.gamma)
      row["fields"] = fields
      errors = []
      for name in FIELD_NAMES:
         error = float(np.mean(np.abs(fields[name] - reference_fields[name])) / scales[name])
         row[f"{name}_normalized_l1"] = error
         errors.append(error)
      row["aggregate_l1"] = float(np.mean(errors))


def compare_amr_runs(rows: list[dict[str, object]], args: argparse.Namespace) -> float:
   if any(int(row["peak_level"]) < 2 for row in rows):
      raise RuntimeError("MHD medley AMR comparison did not exercise at least two fine levels")
   states = [row["state"] for row in rows]
   scale = args.amplitude
   agreement = float(np.mean(np.abs(states[1] - states[0])) / scale)
   for row in rows:
      row["single_rate_level0_relative_l1"] = agreement
   if agreement > 2.0e-3:
      raise RuntimeError(f"MHD medley AMR subcycling disagreement {agreement:.3e} exceeds 2e-3")
   return agreement


def plot_results(
   rows: list[dict[str, object]],
   reference: dict[str, object],
   output: Path,
   args: argparse.Namespace,
) -> None:
   import matplotlib

   matplotlib.use("Agg")
   import matplotlib.pyplot as plt

   output.parent.mkdir(parents=True, exist_ok=True)
   x = (np.arange(args.nx, dtype=float) + 0.5) / args.nx
   fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
   for axis, field_name in zip(axes.flat[:7], FIELD_NAMES, strict=True):
      axis.plot(x, reference["fields"][field_name], color="black", lw=2.0, label="fine reference")
      for row in rows:
         style = ":" if row["method"].endswith("_char") else "-"
         axis.plot(x, row["fields"][field_name], lw=1.0, ls=style, label=row["label"])
      axis.set(title=field_name, xlabel="x")
      axis.grid(alpha=0.25)
   accuracy_axis = axes.flat[7]
   for row in rows:
      accuracy_axis.scatter(row["sim_time_per_s"], row["aggregate_l1"], s=38)
      accuracy_axis.annotate(row["label"], (row["sim_time_per_s"], row["aggregate_l1"]), fontsize=7)
   accuracy_axis.set(
      title="Accuracy versus throughput",
      xlabel="simulation time / real time",
      ylabel="normalized aggregate L1",
      yscale="log",
   )
   accuracy_axis.grid(alpha=0.25)
   axes.flat[0].legend(fontsize=7)
   fig.suptitle(
      f"Periodic mixed-mode ideal-MHD medley: N={args.nx}, t={args.final_time:g}, "
      f"A={args.amplitude:g}, RK4/HLLD"
   )
   fig.savefig(output, dpi=180)
   plt.close(fig)


def serializable_row(row: dict[str, object]) -> dict[str, object]:
   return {
      key: value for key, value in row.items()
      if key not in {"state", "names", "fields", "validation"}
   }


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--nx", type=int, default=256)
   parser.add_argument("--amr-nx", type=int, default=64)
   parser.add_argument("--final-time", type=float, default=0.15)
   parser.add_argument("--amplitude", type=float, default=0.025)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--include-amr", action="store_true")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   args = parser.parse_args()
   base_parameters = read_key_values(BASE_INPUT)
   args.gamma = float(base_parameters["pyfluid.gamma"])
   args.rho0 = float(base_parameters["pyfluid.rho0"])
   args.pressure0 = float(base_parameters["pyfluid.pressure0"])
   args.b2_0 = float(base_parameters["pyfluid.b2_0"])
   args.b3_0 = float(base_parameters["pyfluid.b3_0"])
   if args.nx < 32 or args.amr_nx < 16 or args.final_time <= 0.0 or args.amplitude <= 0.0:
      raise SystemExit("resolutions must be at least 32/16 and time/amplitude must be positive")

   rows = [run_method(method, args) for method in METHODS]
   reference = run_method(REFERENCE, args)
   add_error_metrics(rows, reference, args)
   write_tsv([serializable_row(row) for row in rows], RESULTS)
   if not args.skip_plot:
      plot_results(rows, reference, PLOT, args)

   history_rows = [row["validation"] for row in (*rows, reference)]
   if args.include_amr:
      amr_rows = [run_method(method, args) for method in AMR_METHODS]
      agreement = compare_amr_runs(amr_rows, args)
      write_tsv([serializable_row(row) for row in amr_rows], AMR_RESULTS)
      history_rows.extend(row["validation"] for row in amr_rows)
      print(f"AMR subcycling/single-rate level-0 agreement={agreement:.3e}")
   append_validation_history(HISTORY, history_rows)

   by_method = {row["method"]: row for row in rows}
   if by_method["mp5"]["aggregate_l1"] >= by_method["wenoz"]["aggregate_l1"]:
      raise RuntimeError("component MP5 did not improve on component WENO-Z in the MHD medley")
   for row in rows:
      print(
         f"{row['label']}: L1={row['aggregate_l1']:.3e} "
         f"{row['mcells_s']:.2f} Mcells/s sim_time/s={row['sim_time_per_s']:.3e}"
      )
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
