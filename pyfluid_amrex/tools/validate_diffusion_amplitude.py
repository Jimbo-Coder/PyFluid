#!/usr/bin/env python3
"""Validate explicit/implicit diffusion with analytic Fourier-mode decay."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_1d_profile import parse_plot_header as parse_plot_header_1d
from plot_amrex_1d_profile import read_level0_state_1d
from plot_amrex_slice import parse_plot_header as parse_plot_header_nd
from plot_amrex_slice import read_level_state
from plotfile_reader import sorted_plotfiles
from validation_common import mpi_prefix
from validation_common import (
   DIFFUSIVE_POLICY,
   ROOT,
   append_validation_history,
   average_positive,
   check_scalars,
   parse_float,
   read_scalar_rows,
   repo_relative,
   write_tsv,
)
from pyfluid.exact_solutions import diffusion_wave_amplitude_exact


HISTORY = ROOT / "runs/current/validation_history.tsv"
RUN_ROOT = ROOT / "runs/current/diffusion_amplitude_validation"
PLOT_ROOT = ROOT / "plots/current/IC/diffusion_wave"
RESULTS = RUN_ROOT / "diffusion_amplitude.tsv"


@dataclass(frozen=True)
class DiffusionCase:
   name: str
   label: str
   dim: int
   input_file: Path
   method: str
   physics: str
   coefficient_name: str
   amplitude_kind: str
   use_viscosity: bool
   use_resistivity: bool
   mhd: bool
   ct: bool


CASES = [
   DiffusionCase(
      name="hydro_viscosity_explicit",
      label="hydro explicit viscosity",
      dim=1,
      input_file=ROOT / "pyfluid_amrex/inputs/hydro_shear_wave_x_1d.in",
      method="explicit",
      physics="hydro",
      coefficient_name="nu",
      amplitude_kind="hydro_shear",
      use_viscosity=True,
      use_resistivity=False,
      mhd=False,
      ct=False,
   ),
   DiffusionCase(
      name="hydro_viscosity_implicit",
      label="hydro implicit viscosity",
      dim=1,
      input_file=ROOT / "pyfluid_amrex/inputs/hydro_shear_wave_x_1d.in",
      method="implicit",
      physics="hydro",
      coefficient_name="nu",
      amplitude_kind="hydro_shear",
      use_viscosity=True,
      use_resistivity=False,
      mhd=False,
      ct=False,
   ),
   DiffusionCase(
      name="mhd_resistivity_explicit_ct",
      label="MHD explicit resistivity CT",
      dim=2,
      input_file=ROOT / "pyfluid_amrex/inputs/resistive_wave_x_2d_ct.in",
      method="explicit",
      physics="mhd",
      coefficient_name="eta_m",
      amplitude_kind="mhd_resistive",
      use_viscosity=False,
      use_resistivity=True,
      mhd=True,
      ct=True,
   ),
   DiffusionCase(
      name="mhd_alfven_combined_explicit_ct",
      label="MHD explicit viscosity+resistivity CT",
      dim=2,
      input_file=ROOT / "pyfluid_amrex/inputs/circular_alfven_x_2d_ct.in",
      method="explicit",
      physics="mhd",
      coefficient_name="0.5*(nu+eta_m)",
      amplitude_kind="mhd_alfven_combined",
      use_viscosity=True,
      use_resistivity=True,
      mhd=True,
      ct=True,
   ),
]


def executable(dim: int) -> Path:
   return ROOT / "build/pyfluid_amrex" / f"pyfluid_amrex_{dim}d"


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   indices = []
   for path in directory.glob(f"{stem}_*.png"):
      suffix = path.stem.removeprefix(f"{stem}_")
      if suffix.isdigit():
         indices.append(int(suffix))
   return directory / f"{stem}_{max(indices, default=0) + 1:03d}.png"


def plotfiles(run_dir: Path) -> list[Path]:
   files = sorted_plotfiles(run_dir)
   if not files:
      raise FileNotFoundError(f"no plotfiles found under {run_dir / 'stateGF'}")
   return files


def centered_x(ncell: int, xlo: float, xhi: float) -> np.ndarray:
   dx = (xhi - xlo) / ncell
   return xlo + (np.arange(ncell, dtype=float) + 0.5) * dx


def hydro_shear_amplitude(plotfile: Path, wave_number: float, xlo: float, xhi: float) -> tuple[float, float]:
   names, time_value = parse_plot_header_1d(plotfile)
   state = read_level0_state_1d(plotfile, names)
   index = {name: idx for idx, name in enumerate(names)}
   rho = state[index["rho"]]
   vy = state[index["mom2"]] / rho
   x = centered_x(vy.size, xlo, xhi)
   mode = np.sin(2.0 * np.pi * wave_number * x)
   return time_value, float(2.0 * np.mean(vy * mode))


def mhd_resistive_amplitude(plotfile: Path, wave_number: float, xlo: float, xhi: float) -> tuple[float, float]:
   names, ndim, time_value = parse_plot_header_nd(plotfile)
   if ndim != 2:
      raise ValueError(f"MHD resistive amplitude validator expects a 2D plotfile, got ndim={ndim}")
   state = read_level_state(plotfile, names, ndim, 0)
   index = {name: idx for idx, name in enumerate(names)}
   by = state[index["b2"]]
   bz = state[index["b3"]]
   x = centered_x(by.shape[0], xlo, xhi)[:, None]
   kx = 2.0 * np.pi * wave_number * x
   by_coeff = 2.0 * float(np.mean(by * np.sin(kx)))
   bz_coeff = 2.0 * float(np.mean(bz * np.cos(kx)))
   return time_value, 0.5 * (abs(by_coeff) + abs(bz_coeff))


def fourier_amplitude(values: np.ndarray, wave_number: float, xlo: float, xhi: float) -> float:
   x = centered_x(values.shape[0], xlo, xhi)
   shape = (values.shape[0],) + (1,) * (values.ndim - 1)
   kx = 2.0 * np.pi * wave_number * x.reshape(shape)
   cos_coeff = 2.0 * float(np.mean(values * np.cos(kx)))
   sin_coeff = 2.0 * float(np.mean(values * np.sin(kx)))
   return math.hypot(cos_coeff, sin_coeff)


def mhd_alfven_combined_amplitude(
   plotfile: Path,
   wave_number: float,
   xlo: float,
   xhi: float,
   b1_0: float,
   rho0: float,
) -> tuple[float, float]:
   names, ndim, time_value = parse_plot_header_nd(plotfile)
   if ndim != 2:
      raise ValueError(f"MHD Alfvén damping validator expects a 2D plotfile, got ndim={ndim}")
   state = read_level_state(plotfile, names, ndim, 0)
   index = {name: idx for idx, name in enumerate(names)}
   rho = state[index["rho"]]
   by = state[index["b2"]]
   bz = state[index["b3"]]
   vy = state[index["mom2"]] / rho
   vz = state[index["mom3"]] / rho
   magnetic_amplitude = 0.5 * (
      fourier_amplitude(by, wave_number, xlo, xhi)
      + fourier_amplitude(bz, wave_number, xlo, xhi)
   )
   ca = abs(b1_0) / math.sqrt(rho0)
   velocity_factor = abs(b1_0) / max(rho0 * ca, 1.0e-300)
   velocity_amplitude = 0.5 * (
      fourier_amplitude(vy, wave_number, xlo, xhi)
      + fourier_amplitude(vz, wave_number, xlo, xhi)
   ) / max(velocity_factor, 1.0e-300)
   return time_value, 0.5 * (magnetic_amplitude + velocity_amplitude)


def damping_coefficient(case: DiffusionCase, args: argparse.Namespace) -> float:
   if case.amplitude_kind == "mhd_alfven_combined":
      return 0.5 * (args.viscosity + args.resistivity)
   if case.use_resistivity:
      return args.resistivity
   return args.viscosity


def coefficient_label(name: str) -> str:
   if name == "0.5*(nu+eta_m)":
      return "0.5(nu+eta_m)"
   return name


def collect_amplitudes(case: DiffusionCase, run_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
   rows = []
   coefficient = damping_coefficient(case, args)
   for plotfile in plotfiles(run_dir):
      if case.amplitude_kind == "mhd_resistive":
         time_value, amplitude = mhd_resistive_amplitude(plotfile, args.wave_number, args.xlo, args.xhi)
      elif case.amplitude_kind == "mhd_alfven_combined":
         time_value, amplitude = mhd_alfven_combined_amplitude(
            plotfile,
            args.wave_number,
            args.xlo,
            args.xhi,
            args.b1_0,
            args.rho0,
         )
      else:
         time_value, amplitude = hydro_shear_amplitude(plotfile, args.wave_number, args.xlo, args.xhi)
      exact = float(diffusion_wave_amplitude_exact(args.amplitude, coefficient, args.wave_number, time_value))
      rows.append(
         {
            "case": case.name,
            "label": case.label,
            "method": case.method,
            "physics": case.physics,
            "coefficient": coefficient,
            "coefficient_name": case.coefficient_name,
            "viscosity": args.viscosity if case.use_viscosity else 0.0,
            "resistivity": args.resistivity if case.use_resistivity else 0.0,
            "amplitude_kind": case.amplitude_kind,
            "time": time_value,
            "amplitude": amplitude,
            "exact_amplitude": exact,
            "relative_error": abs(amplitude - exact) / max(abs(exact), 1.0e-300),
            "run_dir": repo_relative(run_dir),
         }
      )
   return rows


def evolution_command(case: DiffusionCase, run_dir: Path, args: argparse.Namespace) -> list[str]:
   viscosity = args.viscosity if case.use_viscosity else 0.0
   resistivity = args.resistivity if case.use_resistivity else 0.0
   command = [
      *mpi_prefix(args.ranks),
      str(executable(case.dim)),
      str(case.input_file),
      f"pyfluid.output_dir={run_dir}",
      f"pyfluid.diffusion={case.method}",
      f"pyfluid.cfl={args.cfl:.17g}",
      f"pyfluid.final_time={args.final_time:.17g}",
      f"pyfluid.max_step={args.max_step}",
      f"pyfluid.rho0={args.rho0:.17g}",
      f"pyfluid.b1_0={args.b1_0:.17g}",
      f"pyfluid.amplitude={args.amplitude:.17g}",
      f"pyfluid.alfven_amplitude={args.amplitude:.17g}",
      f"pyfluid.wave_number={args.wave_number:.17g}",
      f"pyfluid.viscosity={viscosity:.17g}",
      f"pyfluid.resistivity={resistivity:.17g}",
      f"pyfluid.diffusion_cfl={args.diffusion_cfl:.17g}",
      f"pyfluid.diffusion_tolerance={args.diffusion_tolerance:.17g}",
      f"pyfluid.diffusion_max_iterations={args.diffusion_max_iterations}",
      f"pyfluid.diffusion_sor_omega={args.diffusion_sor_omega:.17g}",
      f"pyfluid.scalar_int={args.scalar_int}",
      f"pyfluid.plot_int={args.plot_int}",
      "pyfluid.checkpoint_int=-1",
      "pyfluid.aux_gfs=none",
      "pyfluid.aux_gf_compute_int=-1",
      "pyfluid.aux_gf_int=-1",
   ]
   if case.dim == 1:
      command.append(f"amr.n_cell={args.hydro_nx}")
   else:
      command.append(f"amr.n_cell={args.mhd_nx} {args.mhd_ny}")
   return command


def run_case(case: DiffusionCase, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
   run_dir = RUN_ROOT / case.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   if not (run_dir / "stateGF").exists():
      command = evolution_command(case, run_dir, args)
      env = dict(os.environ)
      env.setdefault("OMP_NUM_THREADS", "1")
      completed = subprocess.run(
         command,
         cwd=ROOT,
         env=env,
         text=True,
         stdout=subprocess.PIPE,
         stderr=subprocess.STDOUT,
         check=False,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(f"{case.name} failed with return code {completed.returncode}; see {run_dir / 'run.log'}")

   amplitude_rows = collect_amplitudes(case, run_dir, args)
   scalar_rows = read_scalar_rows(run_dir)
   scalar_failures = check_scalars(scalar_rows, mhd=case.mhd, energy=False)
   max_rel = max(float(row["relative_error"]) for row in amplitude_rows)
   if max_rel > args.rel_tol:
      scalar_failures.append(f"max amplitude relative error {max_rel:.3e} > {args.rel_tol:.3e}")

   final_scalars = scalar_rows[-1] if scalar_rows else {}
   final_amplitude = amplitude_rows[-1]
   summary = {
      "kind": "analytic",
      "suite": "diffusion_amplitude",
      "step": "diffusion_amplitude",
      "case": case.name,
      "label": case.label,
      "status": "PASS" if not scalar_failures else "FAIL",
      "reason": "; ".join(scalar_failures),
      "returncode": 0 if not scalar_failures else 1,
      "dim": case.dim,
      "input": repo_relative(case.input_file),
      "policy": DIFFUSIVE_POLICY.name,
      "ranks": args.ranks,
      "cfl": args.cfl,
      "steps": final_scalars.get("step", ""),
      "time": final_amplitude["time"],
      "min_rho": final_scalars.get("min_rho", ""),
      "min_pressure": final_scalars.get("min_pressure", ""),
      "mass_rel_drift": final_scalars.get("mass_rel_drift", ""),
      "energy_rel_drift": final_scalars.get("energy_rel_drift", ""),
      "momentum_abs_drift": "",
      "divb_l1": final_scalars.get("divb_l1", ""),
      "avg_mcells_per_s": average_positive(scalar_rows, "mcells_per_s"),
      "avg_sim_time_per_s": average_positive(scalar_rows, "sim_time_per_s"),
      "diffusion_residual": final_scalars.get("diffusion_residual", ""),
      "diffusion_iterations": final_scalars.get("diffusion_iterations", ""),
      "log": repo_relative(run_dir / "run.log"),
      "description": (
         f"{case.label}: measured Fourier amplitude vs "
         f"A0 exp(-{case.coefficient_name} k^2 t)"
      ),
   }
   return amplitude_rows, summary


def plot_amplitude(rows: list[dict[str, object]], output: Path, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
   by_case: dict[str, list[dict[str, object]]] = {}
   for row in rows:
      by_case.setdefault(str(row["case"]), []).append(row)

   for case in CASES:
      case_rows = sorted(by_case.get(case.name, []), key=lambda row: float(row["time"]))
      if not case_rows:
         continue
      t = np.asarray([float(row["time"]) for row in case_rows])
      amp = np.asarray([float(row["amplitude"]) for row in case_rows])
      exact = np.asarray([float(row["exact_amplitude"]) for row in case_rows])
      rel = np.asarray([float(row["relative_error"]) for row in case_rows])
      axs[0].plot(t, amp, marker="o", markersize=4, linewidth=1.9, label=case.label)
      axs[1].semilogy(t, np.maximum(rel, 1.0e-16), marker="o", markersize=4, linewidth=1.9, label=case.label)

   t_ref = np.linspace(0.0, args.final_time, 200)
   reference_laws = sorted(
      {(str(row["coefficient_name"]), float(row["coefficient"])) for row in rows},
      key=lambda item: (item[0], item[1]),
   )
   reference_styles = [":", "--", "-."]
   reference_colors = ["black", "0.25", "0.45"]
   for idx, (coefficient_name, coefficient) in enumerate(reference_laws):
      exact_ref = diffusion_wave_amplitude_exact(args.amplitude, coefficient, args.wave_number, t_ref)
      label = f"exact {coefficient_label(coefficient_name)}={coefficient:g}"
      axs[0].plot(
         t_ref,
         exact_ref,
         color=reference_colors[idx % len(reference_colors)],
         linestyle=reference_styles[idx % len(reference_styles)],
         linewidth=2.2,
         label=label,
      )
   axs[0].set_ylabel("mode amplitude")
   axs[0].set_xlabel("time")
   axs[0].grid(True, which="major", alpha=0.25)
   axs[0].legend(fontsize=8)

   axs[1].set_ylabel("relative amplitude error")
   axs[1].set_xlabel("time")
   axs[1].grid(True, which="major", alpha=0.25)
   axs[1].legend(fontsize=8)

   fig.suptitle(
      "Diffusion amplitude validation: hydro shear, MHD resistive wave, "
      f"damped Alfvén; explicit/implicit, CFL={args.cfl:g}, ranks={args.ranks}"
   )
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser()
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--final-time", type=float, default=0.2)
   parser.add_argument("--max-step", type=int, default=100000)
   parser.add_argument("--hydro-nx", type=int, default=256)
   parser.add_argument("--mhd-nx", type=int, default=96)
   parser.add_argument("--mhd-ny", type=int, default=24)
   parser.add_argument("--rho0", type=float, default=1.0)
   parser.add_argument("--b1_0", type=float, default=1.0)
   parser.add_argument("--amplitude", type=float, default=1.0e-3)
   parser.add_argument("--wave-number", type=float, default=1.0)
   parser.add_argument("--viscosity", type=float, default=1.0e-2)
   parser.add_argument("--resistivity", type=float, default=1.0e-2)
   parser.add_argument("--diffusion-cfl", type=float, default=0.9)
   parser.add_argument("--diffusion-tolerance", type=float, default=1.0e-10)
   parser.add_argument("--diffusion-max-iterations", type=int, default=300)
   parser.add_argument("--diffusion-sor-omega", type=float, default=1.5)
   parser.add_argument("--scalar-int", type=int, default=8)
   parser.add_argument("--plot-int", type=int, default=8)
   parser.add_argument("--rel-tol", type=float, default=2.0e-2)
   parser.add_argument("--skip-plot", action="store_true")
   parser.add_argument("--xlo", type=float, default=0.0)
   parser.add_argument("--xhi", type=float, default=1.0)
   return parser.parse_args()


def main() -> None:
   args = parse_args()
   all_rows: list[dict[str, object]] = []
   summary_rows: list[dict[str, object]] = []
   for case in CASES:
      rows, summary = run_case(case, args)
      all_rows.extend(rows)
      summary_rows.append(summary)

   fields = [
      "case",
      "label",
      "method",
      "physics",
      "coefficient_name",
      "coefficient",
      "viscosity",
      "resistivity",
      "amplitude_kind",
      "time",
      "amplitude",
      "exact_amplitude",
      "relative_error",
      "run_dir",
   ]
   write_tsv(RESULTS, all_rows, fields)
   append_validation_history(HISTORY, summary_rows)

   plot_path = None
   if not args.skip_plot:
      plot_path = next_output_path(PLOT_ROOT, "diffusion_amplitude_explicit_implicit")
      plot_amplitude(all_rows, plot_path, args)

   failed = [row for row in summary_rows if row["status"] != "PASS"]
   for row in summary_rows:
      case_rows = [item for item in all_rows if item["case"] == row["case"]]
      max_rel = max(float(item["relative_error"]) for item in case_rows)
      final = max(case_rows, key=lambda item: float(item["time"]))
      residual = parse_float(row.get("diffusion_residual"), 0.0)
      iterations = parse_float(row.get("diffusion_iterations"), 0.0)
      print(
         f"{row['status']} {row['case']}: final A={float(final['amplitude']):.8e}, "
         f"exact={float(final['exact_amplitude']):.8e}, max rel err={max_rel:.3e}, "
         f"diff residual={residual:.3e}, iterations={iterations:.1f}",
         flush=True,
      )
   print(f"wrote {repo_relative(RESULTS)}")
   if plot_path is not None:
      print(f"wrote {repo_relative(plot_path)}")
   if failed:
      raise SystemExit(1)


if __name__ == "__main__":
   main()
