#!/usr/bin/env python3
"""Validate AMReX EOS modes with a periodic acoustic wave."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotfile_reader import latest_plotfile as discover_latest_plotfile
from plot_amrex_1d_profile import parse_plot_header, read_level0_state_1d
from validation_common import mpi_prefix
from validation_common import read_float_tsv


@dataclass(frozen=True)
class EOSCase:
   name: str
   label: str
   gamma: float
   polytropic_k: float
   polytropic_gamma: float
   thermal_gamma: float
   pressure0: float


@dataclass(frozen=True)
class ScanPoint:
   case: EOSCase
   nx: int
   final_time: float
   run_dir: Path


CASES = [
   EOSCase("ideal_gas", "ideal gas", 5.0 / 3.0, 1.0, 5.0 / 3.0, 5.0 / 3.0, 1.0),
   EOSCase("polytrope", "polytrope", 5.0 / 3.0, 1.0, 2.0, 5.0 / 3.0, 1.0),
   EOSCase("hybrid_polytrope", "hybrid polytrope", 5.0 / 3.0, 0.2, 2.0, 1.5, 1.0),
]


def base_pressure(case: EOSCase, rho0: float) -> float:
   if case.name == "polytrope":
      return case.polytropic_k * rho0**case.polytropic_gamma
   return case.pressure0


def sound_speed(case: EOSCase, rho0: float, pressure0: float) -> float:
   if case.name == "polytrope":
      return math.sqrt(case.polytropic_gamma * pressure0 / rho0)
   if case.name == "hybrid_polytrope":
      cold = case.polytropic_k * rho0**case.polytropic_gamma
      thermal = pressure0 - cold
      return math.sqrt((case.polytropic_gamma * cold + case.thermal_gamma * thermal) / rho0)
   return math.sqrt(case.gamma * pressure0 / rho0)


def pressure_from_state(case: EOSCase, rho: np.ndarray, vx: np.ndarray, energy: np.ndarray) -> np.ndarray:
   if case.name == "polytrope":
      return case.polytropic_k * rho**case.polytropic_gamma
   internal = energy - 0.5 * rho * vx * vx
   if case.name == "hybrid_polytrope":
      cold_pressure = case.polytropic_k * rho**case.polytropic_gamma
      cold_energy = cold_pressure / (case.polytropic_gamma - 1.0)
      return cold_pressure + (case.thermal_gamma - 1.0) * (internal - cold_energy)
   return (case.gamma - 1.0) * internal


def exact_wave(
   x: np.ndarray,
   time: float,
   case: EOSCase,
   rho0: float,
   amplitude: float,
   wave_number: float,
   v1_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   pressure0 = base_pressure(case, rho0)
   cs = sound_speed(case, rho0, pressure0)
   x0 = (x - (v1_0 + cs) * time) % 1.0
   drho = amplitude * np.sin(2.0 * np.pi * wave_number * x0)
   rho = rho0 + drho
   pressure = pressure0 + cs * cs * drho
   vx = v1_0 + cs * drho / rho0
   return rho, pressure, vx


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   used = []
   for path in existing:
      suffix = path.stem.removeprefix(f"{stem}_")
      if suffix.isdigit():
         used.append(int(suffix))
   index = max(used, default=0) + 1
   return directory / f"{stem}_{index:03d}.png"


def build_scan(args: argparse.Namespace) -> list[ScanPoint]:
   scan = []
   for case in CASES:
      pressure0 = base_pressure(case, args.rho0)
      period = 1.0 / (args.v1_0 + sound_speed(case, args.rho0, pressure0))
      final_time = args.final_time if args.final_time > 0.0 else period
      for nx in args.resolutions:
         run_dir = args.run_dir / case.name / f"nx{nx:04d}"
         scan.append(ScanPoint(case=case, nx=nx, final_time=final_time, run_dir=run_dir))
   return scan


def write_scan_plan(scan: list[ScanPoint], output: Path, args: argparse.Namespace) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "eos",
      "nx",
      "final_time",
      "sound_speed",
      "period",
      "cfl",
      "amplitude",
      "run_dir",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, keys, delimiter="\t")
      writer.writeheader()
      for point in scan:
         pressure0 = base_pressure(point.case, args.rho0)
         cs = sound_speed(point.case, args.rho0, pressure0)
         writer.writerow(
            {
               "eos": point.case.name,
               "nx": point.nx,
               "final_time": point.final_time,
               "sound_speed": cs,
               "period": 1.0 / (args.v1_0 + cs),
               "cfl": args.cfl,
               "amplitude": args.amplitude,
               "run_dir": point.run_dir,
            }
         )


def evolution_command(args: argparse.Namespace, point: ScanPoint) -> list[str]:
   command = [
      *mpi_prefix(args.ranks),
      str(args.executable),
      str(args.input),
      f"pyfluid.output_dir={point.run_dir}",
      f"amr.n_cell={point.nx}",
      f"pyfluid.eos={point.case.name}",
      f"pyfluid.gamma={point.case.gamma:.17g}",
      f"pyfluid.polytropic_k={point.case.polytropic_k:.17g}",
      f"pyfluid.polytropic_gamma={point.case.polytropic_gamma:.17g}",
      f"pyfluid.thermal_gamma={point.case.thermal_gamma:.17g}",
      f"pyfluid.pressure0={point.case.pressure0:.17g}",
      f"pyfluid.final_time={point.final_time:.17g}",
      f"pyfluid.cfl={args.cfl:.17g}",
      f"pyfluid.amplitude={args.amplitude:.17g}",
      f"pyfluid.rho0={args.rho0:.17g}",
      f"pyfluid.v1_0={args.v1_0:.17g}",
      f"pyfluid.wave_number={args.wave_number:.17g}",
      "pyfluid.scalar_int=8",
      "pyfluid.plot_int=100000",
      "pyfluid.checkpoint_int=-1",
   ]
   return command


def run_evolution(args: argparse.Namespace, scan: list[ScanPoint]) -> None:
   for point in scan:
      pressure0 = base_pressure(point.case, args.rho0)
      period = 1.0 / (args.v1_0 + sound_speed(point.case, args.rho0, pressure0))
      print(
         f"evolve {point.case.name} nx={point.nx} final_time={point.final_time:.8e} period={period:.8e}",
         flush=True,
      )
      if args.force and point.run_dir.exists():
         shutil.rmtree(point.run_dir)
      point.run_dir.mkdir(parents=True, exist_ok=True)
      if (point.run_dir / "stateGF").exists():
         continue
      command = evolution_command(args, point)
      subprocess.run(command, check=True)


def collect_result(args: argparse.Namespace, point: ScanPoint) -> dict[str, float | str | Path]:
   plotfile = discover_latest_plotfile(point.run_dir)
   if plotfile is None:
      raise FileNotFoundError(f"no plotfiles found under {point.run_dir}")
   names, time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   index = {name: idx for idx, name in enumerate(names)}
   rho = state[index["rho"]]
   vx = state[index["mom1"]] / rho
   energy = state[index["energy"]]
   pressure = pressure_from_state(point.case, rho, vx, energy)
   dx = 1.0 / point.nx
   x = (np.arange(point.nx, dtype=float) + 0.5) * dx
   rho_exact, pressure_exact, vx_exact = exact_wave(
      x, time_value, point.case, args.rho0, args.amplitude, args.wave_number, args.v1_0
   )
   pressure0 = base_pressure(point.case, args.rho0)
   cs = sound_speed(point.case, args.rho0, pressure0)
   errors = {
      "rho_error": float(np.sqrt(np.mean((rho - rho_exact) ** 2)) / args.amplitude),
      "pressure_error": float(np.sqrt(np.mean((pressure - pressure_exact) ** 2)) / (cs * cs * args.amplitude)),
      "vx_error": float(np.sqrt(np.mean((vx - vx_exact) ** 2)) / (cs * args.amplitude / args.rho0)),
   }
   scalars = read_final_scalars(point.run_dir / "scalars.tsv")
   return {
      "eos": point.case.name,
      "label": point.case.label,
      "nx": point.nx,
      "dx": dx,
      "time": time_value,
      "sound_speed": cs,
      "run_dir": point.run_dir,
      "plotfile": plotfile,
      **errors,
      **scalars,
   }


def read_final_scalars(path: Path) -> dict[str, float]:
   rows = read_float_tsv(path)
   if not rows:
      return {}
   final = rows[-1]
   keep = ["mass_rel_drift", "energy_rel_drift", "mom1", "dt", "mcells_per_s", "sim_time_per_s"]
   return {key: float(final[key]) for key in keep if key in final}


def read_scalar_rows(path: Path) -> list[dict[str, float]]:
   return read_float_tsv(path)


def convergence_slope(dx: np.ndarray, err: np.ndarray) -> float:
   if len(dx) < 2:
      return float("nan")
   fit_count = min(3, len(dx))
   return float(np.polyfit(np.log(dx[:fit_count]), np.log(err[:fit_count]), 1)[0])


def add_reference(ax: plt.Axes, dx: np.ndarray, err: np.ndarray, power: int) -> None:
   ref = err[0] * (dx / dx[0]) ** power
   ax.loglog(dx, ref, color="0.25", linestyle="--", linewidth=1.4)
   mid = len(dx) // 2
   ax.text(dx[mid], ref[mid] * 1.15, f"dx^{power}", color="0.2", fontsize=9)


def plot_convergence(results: list[dict[str, float | str | Path]], output: Path, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
   fig.suptitle(
      "AMReX EOS acoustic_wave_x convergence: "
      f"WENO-Z/HLLC/RK4, CFL={args.cfl:g}, BC=periodic, hydro, amp={args.amplitude:g}"
   )
   fields = [("rho_error", "rho"), ("pressure_error", "P"), ("vx_error", "vx")]
   for ax, (key, label) in zip(axs, fields, strict=True):
      for case in CASES:
         rows = [row for row in results if row["eos"] == case.name]
         dx = np.asarray([float(row["dx"]) for row in rows])
         err = np.asarray([float(row[key]) for row in rows])
         order = convergence_slope(dx, err)
         ax.loglog(dx, err, marker="o", linewidth=2.0, label=f"{case.label} p={order:.2f}")
      first_rows = [row for row in results if row["eos"] == CASES[0].name]
      add_reference(
         ax,
         np.asarray([float(row["dx"]) for row in first_rows]),
         np.asarray([float(row[key]) for row in first_rows]),
         4,
      )
      ax.set_title(f"{label} normalized L2")
      ax.set_xlabel("dx")
      ax.set_ylabel("normalized L2 error")
      ax.grid(True, which="major", alpha=0.25)
      ax.invert_xaxis()
      ax.legend(fontsize=8)
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def plot_profiles(results: list[dict[str, float | str | Path]], output: Path, nx: int, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(3, 3, figsize=(15, 9), sharex=True, constrained_layout=True)
   fig.suptitle(
      "AMReX EOS acoustic_wave_x profiles: "
      f"WENO-Z/HLLC/RK4, CFL={args.cfl:g}, BC=periodic, hydro, amp={args.amplitude:g}"
   )
   for col, case in enumerate(CASES):
      row = next(item for item in results if item["eos"] == case.name and int(item["nx"]) == nx)
      names, time_value = parse_plot_header(Path(row["plotfile"]))
      state = read_level0_state_1d(Path(row["plotfile"]), names)
      index = {name: idx for idx, name in enumerate(names)}
      rho = state[index["rho"]]
      vx = state[index["mom1"]] / rho
      pressure = pressure_from_state(case, rho, vx, state[index["energy"]])
      pressure0 = base_pressure(case, args.rho0)
      dx = 1.0 / nx
      x = (np.arange(nx, dtype=float) + 0.5) * dx
      exact = exact_wave(x, time_value, case, args.rho0, args.amplitude, args.wave_number, args.v1_0)
      series = [
         (rho - args.rho0, exact[0] - args.rho0, "rho - rho0"),
         (pressure - pressure0, exact[1] - pressure0, "P - P0"),
         (vx - args.v1_0, exact[2] - args.v1_0, "vx - v1_0"),
      ]
      for ax, (num, ref, ylabel) in zip(axs[:, col], series, strict=True):
         ax.plot(x, num, color="tab:blue", marker="o", markersize=2.4, linewidth=1.1, label="num")
         ax.plot(x, ref, color="black", linestyle=":", linewidth=2.0, label="exact")
         ax.set_ylabel(ylabel)
         ax.grid(True, which="major", alpha=0.25)
         if ax is axs[0, col]:
            ax.set_title(f"{case.label}, nx={nx}, t={time_value:.4f}")
         if ax is axs[-1, col]:
            ax.set_xlabel("x")
         ax.legend(fontsize=8)
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def plot_scalars(results: list[dict[str, float | str | Path]], output: Path, nx: int, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
   fig.suptitle(
      "AMReX EOS acoustic_wave_x scalars: "
      f"WENO-Z/HLLC/RK4, CFL={args.cfl:g}, BC=periodic, hydro, nx={nx}"
   )
   for case in CASES:
      result = next(row for row in results if row["eos"] == case.name and int(row["nx"]) == nx)
      rows = read_scalar_rows(Path(result["run_dir"]) / "scalars.tsv")
      t = np.asarray([row["time"] for row in rows])
      markers = {"marker": "o", "markersize": 3.0} if len(rows) <= 16 else {}
      mass = np.maximum([row["mass_rel_drift"] for row in rows], 1.0e-18)
      energy = np.maximum([row["energy_rel_drift"] for row in rows], 1.0e-18)
      axs[0, 0].semilogy(t, mass, linewidth=2.0, label=f"{case.label} mass", **markers)
      axs[0, 0].semilogy(t, energy, linestyle="--", linewidth=1.6, label=f"{case.label} energy", **markers)
      axs[0, 1].plot(t, [row["mom1"] for row in rows], linewidth=2.0, label=case.label, **markers)
      axs[1, 0].plot(t, [row["dt"] for row in rows], linewidth=2.0, label=case.label, **markers)
      axs[1, 1].plot(t, [row["mcells_per_s"] for row in rows], linewidth=2.0, label=f"{case.label} Mcells/s", **markers)
      if "sim_time_per_s" in rows[0]:
         axs[1, 1].plot(t, [row["sim_time_per_s"] for row in rows], linestyle="--", linewidth=1.6, label=f"{case.label} sim time/s", **markers)

   axs[0, 0].set_title("Conserved Scalars")
   axs[0, 0].set_ylabel("relative drift")
   axs[0, 1].set_title("Total Momentum")
   axs[0, 1].set_ylabel("mom1")
   axs[1, 0].set_title("Timestep")
   axs[1, 0].set_ylabel("dt")
   axs[1, 1].set_title("Performance")
   axs[1, 1].set_ylabel("Mcells/s, sim time/s")
   for ax in axs.flat:
      ax.set_xlabel("time")
      ax.grid(True, which="major", alpha=0.25)
      ax.legend(fontsize=8)
      try:
         ax.ticklabel_format(useOffset=False)
      except AttributeError:
         pass
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def write_tsv(results: list[dict[str, float | str | Path]], output: Path) -> None:
   output.parent.mkdir(parents=True, exist_ok=True)
   keys = [
      "eos",
      "nx",
      "dx",
      "time",
      "sound_speed",
      "rho_error",
      "pressure_error",
      "vx_error",
      "mass_rel_drift",
      "energy_rel_drift",
      "mom1",
      "dt",
      "mcells_per_s",
      "sim_time_per_s",
      "run_dir",
   ]
   with output.open("w", newline="") as f:
      writer = csv.DictWriter(f, keys, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      for row in results:
         writer.writerow(row)


def postprocess_scan(args: argparse.Namespace, scan: list[ScanPoint]) -> None:
   results = [collect_result(args, point) for point in scan]

   if not args.skip_plots:
      scan_dir = Path("plots/current/scans")
      ic_dir = Path("plots/current/IC/acoustic_wave_x")
      convergence_path = next_output_path(scan_dir, "eos_acoustic_convergence")
      profile_path = next_output_path(ic_dir, "eos_acoustic_profiles")
      scalar_path = next_output_path(ic_dir, "eos_acoustic_scalars")
      plot_convergence(results, convergence_path, args)
      plot_profiles(results, profile_path, max(args.resolutions), args)
      plot_scalars(results, scalar_path, max(args.resolutions), args)
      print(f"wrote {convergence_path}")
      print(f"wrote {profile_path}")
      print(f"wrote {scalar_path}")
   write_tsv(results, args.run_dir / "eos_acoustic_convergence.tsv")
   print(f"wrote {args.run_dir / 'eos_acoustic_convergence.tsv'}")


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--input", type=Path, default=Path("pyfluid_amrex/inputs/acoustic_wave_1d.in"))
   parser.add_argument("--executable", type=Path, default=Path("build/pyfluid_amrex/pyfluid_amrex_1d"))
   parser.add_argument("--run-dir", type=Path, default=Path("runs/current/eos_acoustic_validation"))
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64, 128, 256])
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--rho0", type=float, default=1.0)
   parser.add_argument("--v1_0", type=float, default=0.0)
   parser.add_argument("--amplitude", type=float, default=1.0e-6)
   parser.add_argument("--wave-number", type=float, default=1.0)
   parser.add_argument("--final-time", type=float, default=0.1, help="use <=0 to evolve one acoustic period per EOS")
   parser.add_argument("--stage", choices=["all", "evolve", "postprocess"], default="all")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plots", action="store_true", help="write convergence TSV only")
   args = parser.parse_args()

   scan = build_scan(args)
   write_scan_plan(scan, args.run_dir / "eos_acoustic_scan.tsv", args)
   if args.stage in ("all", "evolve"):
      run_evolution(args, scan)
   if args.stage in ("all", "postprocess"):
      postprocess_scan(args, scan)


if __name__ == "__main__":
   main()
