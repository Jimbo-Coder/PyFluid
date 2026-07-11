#!/usr/bin/env python3
"""Measure the sharp 1D hydro CFL edge with a high-frequency acoustic mode."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from validation_common import run_mpi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_slice import parse_plot_header, read_level_state
from plotfile_reader import latest_plotfile


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "pyfluid_amrex"
BASE_INPUT = ROOT / "pyfluid_amrex" / "inputs" / "acoustic_wave_1d.in"
DEFAULT_RUN_ROOT = ROOT / "runs" / "current" / "hydro_cfl_edge"
DEFAULT_PLOT_DIR = ROOT / "plots" / "current" / "scans"
TIMESTEPPERS = ("euler", "corrector", "midpoint", "ssprk3", "rk4")
AMR_MODES = ("uniform", "amr_nosub", "amr_spatial", "amr_local_cfl")


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   if not existing:
      return directory / f"{stem}_001.png"
   try:
      index = int(existing[-1].stem.rsplit("_", 1)[-1]) + 1
   except ValueError:
      index = 1
   return directory / f"{stem}_{index:03d}.png"


def parse_cfl_values(text: str) -> list[float]:
   values = [float(raw.strip()) for raw in text.split(",") if raw.strip()]
   if not values:
      raise ValueError("at least one CFL value is required")
   return values


def parse_timesteps(text: str) -> list[str]:
   values = [raw.strip() for raw in text.split(",") if raw.strip()]
   if not values:
      raise ValueError("at least one timestepper is required")
   unknown = sorted(set(values) - set(TIMESTEPPERS))
   if unknown:
      raise ValueError(f"unknown timestepper(s): {', '.join(unknown)}")
   return values


def parse_amr_modes(text: str) -> list[str]:
   values = [raw.strip() for raw in text.split(",") if raw.strip()]
   if not values:
      raise ValueError("at least one AMR mode is required")
   unknown = sorted(set(values) - set(AMR_MODES))
   if unknown:
      raise ValueError(f"unknown AMR mode(s): {', '.join(unknown)}")
   return values


def rk_stability_polynomial(timestep: str, z: complex) -> complex:
   if timestep == "euler":
      return 1.0 + z
   if timestep in ("corrector", "midpoint"):
      return 1.0 + z + 0.5 * z * z
   if timestep == "ssprk3":
      return 1.0 + z + 0.5 * z * z + z * z * z / 6.0
   if timestep == "rk4":
      return 1.0 + z + 0.5 * z * z + z * z * z / 6.0 + z * z * z * z / 24.0
   raise ValueError(f"unsupported timestepper {timestep}")


def nyquist_upwind_growth_per_step(timestep: str, cfl: float) -> float:
   # PWC/LLF at the Nyquist mode has scalar upwind eigenvalue z = -2 CFL.
   return abs(rk_stability_polynomial(timestep, complex(-2.0 * cfl, 0.0)))


def repeated_growth(growth_per_step: float, steps: int) -> float:
   if growth_per_step == 0.0:
      return 0.0
   log_growth = steps * math.log(growth_per_step)
   if log_growth > 700.0:
      return 1.0e300
   if log_growth < -745.0:
      return 0.0
   return math.exp(log_growth)


def write_input(
   timestep: str,
   amr_mode: str,
   cfl: float,
   run_root: Path,
   nx: int,
   wave_number: float,
   steps: int,
   amplitude: float,
   gamma: float,
   force: bool,
) -> tuple[Path, Path]:
   tag = f"hydro_acoustic_nyquist_{amr_mode}_pwc_llf_{timestep}_cfl{cfl:.8g}".replace(".", "p")
   output_dir = run_root / tag
   if force and output_dir.exists():
      shutil.rmtree(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   input_dir = run_root / "inputs"
   input_dir.mkdir(parents=True, exist_ok=True)

   amr_overrides = {
      "amr.max_level": "0",
      "pyfluid.subcycling": "off",
   }
   if amr_mode != "uniform":
      amr_overrides = {
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "-1",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_fixed_levels": "1",
         "amr.initial_refine_box_lo": "0.0",
         "amr.initial_refine_box_hi": "1.0",
         "pyfluid.subcycling": "off",
      }
      if amr_mode == "amr_spatial":
         amr_overrides["pyfluid.subcycling"] = "on"
         amr_overrides["pyfluid.subcycling_steps"] = "spatial"
      elif amr_mode == "amr_local_cfl":
         amr_overrides["pyfluid.subcycling"] = "on"
         amr_overrides["pyfluid.subcycling_steps"] = "local_cfl"

   overrides = {
      "pyfluid.run_name": tag,
      "pyfluid.output_dir": str(output_dir.relative_to(ROOT)),
      "geometry.prob_lo": "0.0",
      "geometry.prob_hi": "1.0",
      "geometry.is_periodic": "1",
      "amr.n_cell": str(nx),
      "amr.max_grid_size": "64",
      "amr.blocking_factor": "8",
      "pyfluid.max_step": str(steps),
      "pyfluid.final_time": "-1.0",
      "pyfluid.ng": "1",
      "pyfluid.cfl": f"{cfl:.17g}",
      "pyfluid.gamma": f"{gamma:.17g}",
      "pyfluid.eos": "ideal_gas",
      "pyfluid.physics": "hydro",
      "pyfluid.ct": "off",
      "pyfluid.ic": "acoustic_wave_x",
      "pyfluid.rho0": "1.0",
      "pyfluid.pressure0": "1.0",
      "pyfluid.amplitude": f"{amplitude:.17g}",
      "pyfluid.wave_number": f"{wave_number:.17g}",
      "pyfluid.v1_0": "0.0",
      "pyfluid.v2_0": "0.0",
      "pyfluid.v3_0": "0.0",
      "pyfluid.recon": "pwc",
      "pyfluid.riemann": "llf",
      "pyfluid.timestep": timestep,
      "pyfluid.bc": "periodic",
      "pyfluid.diffusion": "none",
      "pyfluid.scalar_int": str(steps),
      "pyfluid.plot_int": str(steps),
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
   }
   overrides.update(amr_overrides)
   lines = BASE_INPUT.read_text().splitlines()
   lines.append("")
   lines.append("# scan_hydro_cfl_edge.py overrides")
   for key, value in overrides.items():
      lines.append(f"{key} = {value}")
   input_path = input_dir / f"{tag}.in"
   input_path.write_text("\n".join(lines) + "\n")
   return input_path, output_dir


def finest_level(plotfile: Path) -> int:
   levels = []
   for path in plotfile.iterdir():
      if path.is_dir() and path.name.startswith("Level_"):
         try:
            levels.append(int(path.name.removeprefix("Level_")))
         except ValueError:
            continue
   return max(levels, default=0)


def density_mode_l2(plotfile: Path) -> tuple[float, float, int]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 1:
      raise ValueError(f"expected 1D plotfile, got {ndim}D")
   level = finest_level(plotfile)
   state = read_level_state(plotfile, names, ndim, level)
   rho = state[names.index("rho")]
   perturbation = rho - np.mean(rho)
   return float(np.sqrt(np.mean(perturbation * perturbation))), time_value, level


def initial_density_l2(nx: int, wave_number: float, amplitude: float) -> float:
   x = (np.arange(nx, dtype=float) + 0.5) / nx
   perturbation = amplitude * np.sin(2.0 * np.pi * wave_number * x)
   perturbation -= np.mean(perturbation)
   return float(np.sqrt(np.mean(perturbation * perturbation)))


def scalar_last(output_dir: Path) -> dict[str, str]:
   scalar_file = output_dir / "scalars.tsv"
   if not scalar_file.exists():
      return {}
   with scalar_file.open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   return rows[-1] if rows else {}


def run_case(
   timestep: str,
   amr_mode: str,
   cfl: float,
   args: argparse.Namespace,
) -> dict[str, str]:
   effective_nx = args.nx * (2 if amr_mode != "uniform" else 1)
   wave_number = 0.5 * effective_nx
   initial_l2 = initial_density_l2(effective_nx, wave_number, args.amplitude)
   input_path, output_dir = write_input(
      timestep,
      amr_mode,
      cfl,
      args.run_root,
      args.nx,
      wave_number,
      args.steps,
      args.amplitude,
      args.gamma,
      args.force,
   )
   exe = BUILD / "pyfluid_amrex_1d"
   proc = run_mpi(
      exe,
      [input_path],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   last = scalar_last(output_dir)
   plotfile = latest_plotfile(output_dir)
   final_l2 = math.nan
   time_value = math.nan
   level = -1
   if plotfile is not None:
      final_l2, time_value, level = density_mode_l2(plotfile)
   ok = proc.returncode == 0 and math.isfinite(final_l2)
   reason = "" if ok else "\n".join(proc.stdout.splitlines()[-5:]).replace("\t", " ").replace("\x00", "")
   theory_step = nyquist_upwind_growth_per_step(timestep, cfl)
   theory_ratio = repeated_growth(theory_step, args.steps)
   return {
      "timestep": timestep,
      "amr_mode": amr_mode,
      "cfl": f"{cfl:.10g}",
      "status": "PASS" if ok else "FAIL",
      "reason": reason,
      "steps": last.get("step", ""),
      "time": f"{time_value:.17g}" if math.isfinite(time_value) else "",
      "level": str(level) if level >= 0 else "",
      "effective_nx": str(effective_nx),
      "initial_l2": f"{initial_l2:.17g}",
      "final_l2": f"{final_l2:.17g}" if math.isfinite(final_l2) else "",
      "growth_ratio": f"{final_l2 / initial_l2:.17g}" if ok else "",
      "theory_step_growth": f"{theory_step:.17g}",
      "theory_nyquist_ratio": f"{theory_ratio:.17g}",
      "min_rho": last.get("min_rho", ""),
      "min_pressure": last.get("min_pressure", ""),
      "avg_mcells_per_s": last.get("mcells_per_s", ""),
   }


def write_results(rows: list[dict[str, str]], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   fields = [
      "timestep",
      "amr_mode",
      "cfl",
      "status",
      "reason",
      "steps",
      "time",
      "level",
      "effective_nx",
      "initial_l2",
      "final_l2",
      "growth_ratio",
      "theory_step_growth",
      "theory_nyquist_ratio",
      "min_rho",
      "min_pressure",
      "avg_mcells_per_s",
   ]
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      writer.writerows(rows)


def plot_results(rows: list[dict[str, str]], output: Path, args: argparse.Namespace) -> None:
   fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
   labels = []
   for row in rows:
      label = f"{row['timestep']} {row['amr_mode']}"
      if label not in labels:
         labels.append(label)
   for label in labels:
      case_rows = sorted(
         (row for row in rows if f"{row['timestep']} {row['amr_mode']}" == label),
         key=lambda row: float(row["cfl"]),
      )
      cfl = np.array([float(row["cfl"]) for row in case_rows])
      measured = np.array(
         [float(row["growth_ratio"]) if row["growth_ratio"] else np.nan for row in case_rows],
         dtype=float,
      )
      theory = np.array([float(row["theory_nyquist_ratio"]) for row in case_rows], dtype=float)
      finite = np.isfinite(measured)
      theory_finite = np.isfinite(theory)
      if finite.any():
         ax.semilogy(
            cfl[finite],
            measured[finite],
            "o-",
            label=f"{label} sim",
         )
      if not finite.all():
         cfl_unstable = cfl[~finite]
         max_y = np.nanmax(measured[finite]) if finite.any() else 1.0
         yvals = np.full_like(cfl_unstable, 0.8 * max_y if max_y > 0 else 1.0)
         ax.semilogy(cfl_unstable, yvals, "x", color="0.35", label=f"{label} sim (non-finite)")
      if theory_finite.any():
         ax.semilogy(
            cfl[theory_finite],
            theory[theory_finite],
            "--",
            linewidth=1.2,
            alpha=0.7,
            label=f"{case_rows[0]['timestep']} RK ref",
         )
   ax.axhline(1.0, color="0.4", linewidth=1.0)
   ax.axvline(1.0, color="0.2", linestyle=":", linewidth=1.4)
   ax.set_xlabel("CFL")
   ax.set_ylabel("amplitude growth after run")
   ax.set_title(
      "1D hydro CFL edge: acoustic Nyquist mode, PWC/LLF, "
      f"N={args.nx}, steps={args.steps}"
   )
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--timesteps", default=",".join(TIMESTEPPERS))
   parser.add_argument("--amr-modes", default="uniform")
   parser.add_argument("--cfl-values", default="0.8,0.9,0.98,0.999,1.0,1.0001,1.001,1.02,1.05,1.1,1.2,1.3,1.4")
   parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
   parser.add_argument("--results", type=Path)
   parser.add_argument("--plot", type=Path)
   parser.add_argument("--nx", type=int, default=128)
   parser.add_argument("--steps", type=int, default=5000)
   parser.add_argument("--amplitude", type=float, default=1.0e-8)
   parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   args.run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
   results_path = args.results if args.results is not None else args.run_root / "hydro_cfl_edge.tsv"
   results_path = results_path if results_path.is_absolute() else ROOT / results_path
   plot_path = args.plot if args.plot is not None else next_output_path(DEFAULT_PLOT_DIR, "hydro_cfl_edge")
   plot_path = plot_path if plot_path.is_absolute() else ROOT / plot_path

   rows = [
      run_case(timestep, amr_mode, cfl, args)
      for timestep in parse_timesteps(args.timesteps)
      for amr_mode in parse_amr_modes(args.amr_modes)
      for cfl in parse_cfl_values(args.cfl_values)
   ]
   write_results(rows, results_path)
   plot_results(rows, plot_path, args)
   print(f"wrote {results_path.relative_to(ROOT)}")
   print(f"wrote {plot_path.relative_to(ROOT)}")
   for row in rows:
      print(
         f"{row['timestep']} {row['amr_mode']} CFL={row['cfl']} "
         f"{row['status']} growth={row['growth_ratio']}"
      )


if __name__ == "__main__":
   main()
