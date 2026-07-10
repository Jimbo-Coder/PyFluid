#!/usr/bin/env python3
"""Validate AMReX Newtonian gravity Poisson solves against analytic fields."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_1d_profile import parse_plot_header, read_level0_state_1d  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402
from validation_common import mpi_prefix
from validation_common import append_validation_history, read_tsv, scalar_checks, scalar_validation_row, write_tsv  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE_1D = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_1d"
RUN_ROOT = ROOT / "runs" / "current" / "gravity_validation"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"


@dataclass(frozen=True)
class GravityCase:
   name: str = "poisson_density_sine_1d"
   ranks: int = 1
   nx: int = 128
   max_step: int = 0
   final_time: float = -1.0
   ng: int = 1
   cfl: float = 0.9
   bc: str = "periodic"
   physics: str = "hydro"
   ct: str = "off"
   ic: str = "sin_wave"
   eos: str = "ideal_gas"
   gamma: float = 5.0 / 3.0
   polytropic_k: float = 1.0
   polytropic_gamma: float = 5.0 / 3.0
   thermal_gamma: float = 5.0 / 3.0
   gravity: str = "poisson"
   gravity_coupling: str = "athena"
   gravity_fixed_potential: str = "uniform_acceleration"
   check: str = "poisson_fields"
   rho0: float = 1.0
   pressure0: float = 1.0
   amplitude: float = 1.0e-3
   gravity_constant: float = 1.0
   gravity_a1: float = 0.0
   gravity_omega: float = 0.0
   b1_0: float = 0.0
   b2_0: float = 0.0
   b3_0: float = 0.0
   recon: str = "pwc"
   riemann: str = "llf"
   timestep: str = "euler"
   plot_int: int = -1
   aux_gfs: str = "phi,grad_phi"
   aux_gf_int: int = 0
   aux_gf_compute_int: int = -1


def case_input(case: GravityCase, run_dir: Path) -> str:
   periodic = "1" if case.bc == "periodic" else "0"
   return f"""pyfluid.run_name = {case.name}
pyfluid.output_dir = {run_dir.relative_to(ROOT)}

geometry.prob_lo = 0.0
geometry.prob_hi = 1.0
geometry.is_periodic = {periodic}

amr.n_cell = {case.nx}
amr.max_level = 0
amr.max_grid_size = 64
amr.blocking_factor = 8

pyfluid.max_step = {case.max_step}
pyfluid.final_time = {case.final_time:.17g}
pyfluid.ng = {case.ng}
pyfluid.cfl = {case.cfl:.17g}
pyfluid.eos = {case.eos}
pyfluid.gamma = {case.gamma:.17g}
pyfluid.polytropic_k = {case.polytropic_k:.17g}
pyfluid.polytropic_gamma = {case.polytropic_gamma:.17g}
pyfluid.thermal_gamma = {case.thermal_gamma:.17g}

pyfluid.physics = {case.physics}
pyfluid.ct = {case.ct}
pyfluid.ic = {case.ic}
pyfluid.rho0 = {case.rho0:.17g}
pyfluid.pressure0 = {case.pressure0:.17g}
pyfluid.amplitude = {case.amplitude:.17g}
pyfluid.bump_x0 = 0.5
pyfluid.v1_0 = 0.0
pyfluid.v2_0 = 0.0
pyfluid.v3_0 = 0.0
pyfluid.b1_0 = {case.b1_0:.17g}
pyfluid.b2_0 = {case.b2_0:.17g}
pyfluid.b3_0 = {case.b3_0:.17g}

pyfluid.recon = {case.recon}
pyfluid.riemann = {case.riemann}
pyfluid.timestep = {case.timestep}
pyfluid.bc = {case.bc}
pyfluid.inflow_rho = {case.rho0:.17g}
pyfluid.inflow_pressure = {case.pressure0:.17g}
pyfluid.inflow_v1 = 0.0
pyfluid.inflow_v2 = 0.0
pyfluid.inflow_v3 = 0.0
pyfluid.diffusion = none
pyfluid.source = none
pyfluid.gravity = {case.gravity}
pyfluid.gravity_coupling = {case.gravity_coupling}
pyfluid.gravity_fixed_potential = {case.gravity_fixed_potential}
pyfluid.gravity_constant = {case.gravity_constant:.17g}
pyfluid.gravity_a1 = {case.gravity_a1:.17g}
pyfluid.gravity_omega = {case.gravity_omega:.17g}
pyfluid.gravity_center_x = 0.5
pyfluid.gravity_center_y = 0.5
pyfluid.gravity_center_z = 0.5
pyfluid.gravity_tolerance = 1.0e-12
pyfluid.gravity_max_iterations = 200
pyfluid.gravity_max_fmg_iterations = 20
pyfluid.gravity_max_order = 2

pyfluid.scalar_int = 1
pyfluid.validation_policy = elliptic
pyfluid.plot_int = {case.plot_int}
pyfluid.checkpoint_int = -1
pyfluid.aux_gfs = {case.aux_gfs}
pyfluid.aux_gf_int = {case.aux_gf_int}
pyfluid.aux_gf_compute_int = {case.aux_gf_compute_int}
"""


def run_case(case: GravityCase, force: bool) -> tuple[Path, Path, int, float, str]:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = RUN_ROOT / "inputs" / f"{case.name}.in"
   input_path.parent.mkdir(parents=True, exist_ok=True)
   input_path.write_text(case_input(case, run_dir))
   if case.ranks == 1:
      command = [str(EXECUTABLE_1D), str(input_path)]
   else:
      command = [
         *mpi_prefix(case.ranks),
         str(EXECUTABLE_1D),
         str(input_path),
      ]
   env = dict(os.environ)
   env.setdefault("OMP_NUM_THREADS", "1")
   start = time.perf_counter()
   completed = subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )
   elapsed = time.perf_counter() - start
   (run_dir / "run.log").write_text(completed.stdout)
   return run_dir, input_path, completed.returncode, elapsed, completed.stdout


def latest_aux_plotfile(run_dir: Path) -> Path:
   plotfile = latest_plotfile(run_dir, "auxGF")
   if plotfile is None:
      raise RuntimeError(f"missing aux plotfile under {run_dir / 'auxGF'}")
   return plotfile


def latest_state_plotfile(run_dir: Path) -> Path:
   plotfile = latest_plotfile(run_dir)
   if plotfile is None:
      raise RuntimeError(f"missing state plotfile under {run_dir / 'stateGF'}")
   return plotfile


def hse_polytrope_profile(case: GravityCase, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   x = (np.arange(nx, dtype=float) + 0.5) / nx
   gamma = case.gamma if case.eos == "ideal_gas" else case.polytropic_gamma
   k_poly = case.pressure0 / (case.rho0**gamma) if case.eos == "ideal_gas" else case.polytropic_k
   rho_power = case.rho0 ** (gamma - 1.0) + (gamma - 1.0) * case.gravity_a1 * (x - 0.5) / (gamma * k_poly)
   if np.any(rho_power <= 0.0):
      raise RuntimeError(f"{case.name} exact HSE profile has non-positive density")
   rho = rho_power ** (1.0 / (gamma - 1.0))
   pressure = k_poly * rho**gamma
   return x, rho, pressure


def pressure_from_state(case: GravityCase, names: list[str], state: np.ndarray) -> np.ndarray:
   name_to_index = {name: index for index, name in enumerate(names)}
   rho = state[name_to_index["rho"]]
   if case.eos == "polytrope":
      return case.polytropic_k * rho**case.polytropic_gamma
   momx = state[name_to_index["mom1"]]
   momy = state[name_to_index["mom2"]]
   momz = state[name_to_index["mom3"]]
   energy = state[name_to_index["energy"]]
   kinetic = 0.5 * (momx * momx + momy * momy + momz * momz) / rho
   magnetic = 0.0
   if "b1" in name_to_index:
      bx = state[name_to_index["b1"]]
      by = state[name_to_index["b2"]]
      bz = state[name_to_index["b3"]]
      magnetic = 0.5 * (bx * bx + by * by + bz * bz)
   if case.eos == "hybrid_polytrope":
      cold_pressure = case.polytropic_k * rho**case.polytropic_gamma
      cold_energy = cold_pressure / (case.polytropic_gamma - 1.0)
      return cold_pressure + (case.thermal_gamma - 1.0) * (energy - kinetic - magnetic - cold_energy)
   return (case.gamma - 1.0) * (energy - kinetic - magnetic)


def hse_polytrope_error(case: GravityCase, run_dir: Path, *, plot: bool) -> dict[str, float | str]:
   plotfile = latest_state_plotfile(run_dir)
   names, time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   name_to_index = {name: index for index, name in enumerate(names)}
   rho = state[name_to_index["rho"]]
   pressure = pressure_from_state(case, names, state)
   momx = state[name_to_index["mom1"]]
   momy = state[name_to_index["mom2"]]
   momz = state[name_to_index["mom3"]]
   vx = momx / rho
   vy = momy / rho
   vz = momz / rho
   speed = np.sqrt(vx * vx + vy * vy + vz * vz)
   gamma = case.gamma if case.eos == "ideal_gas" else case.polytropic_gamma
   sound_speed = np.sqrt(gamma * pressure / rho)
   x, exact_rho, exact_pressure = hse_polytrope_profile(case, rho.size)
   rho_scale = max(float(np.sqrt(np.mean(exact_rho * exact_rho))), 1.0e-30)
   pressure_scale = max(float(np.sqrt(np.mean(exact_pressure * exact_pressure))), 1.0e-30)
   scalar_rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
   final = scalar_rows[-1] if scalar_rows else {}
   errors: dict[str, float | str] = {
      "hse_max_mach": float(np.max(speed / sound_speed)),
      "hse_max_abs_vx": float(np.max(np.abs(vx))),
      "hse_rho_rel_l2": float(np.sqrt(np.mean((rho - exact_rho) ** 2)) / rho_scale),
      "hse_pressure_rel_l2": float(np.sqrt(np.mean((pressure - exact_pressure) ** 2)) / pressure_scale),
      "gravity_total_energy_rel_drift": abs(float(final.get("gravity_total_energy_rel_drift", "nan"))),
      "profile_time": float(time_value),
   }
   if plot:
      output = next_plot_path(ROOT / "plots" / "current" / "IC" / "hse_polytrope", "hse_polytrope_x_profile")
      plot_hse_profile(output, case, x, rho, pressure, vx, exact_rho, exact_pressure, time_value)
      errors["plot"] = str(output.relative_to(ROOT))
   return errors


def next_plot_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   existing = sorted(directory.glob(f"{stem}_*.png"))
   next_index = 1
   if existing:
      try:
         next_index = max(int(path.stem.rsplit("_", 1)[1]) for path in existing) + 1
      except (IndexError, ValueError):
         next_index = len(existing) + 1
   return directory / f"{stem}_{next_index:03d}.png"


def plot_hse_profile(
   output: Path,
   case: GravityCase,
   x: np.ndarray,
   rho: np.ndarray,
   pressure: np.ndarray,
   vx: np.ndarray,
   exact_rho: np.ndarray,
   exact_pressure: np.ndarray,
   time_value: float,
) -> None:
   import matplotlib.pyplot as plt

   fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
   fig.suptitle(
      f"HSE polytrope x: fixed gravity, {case.recon.upper()}/{case.riemann.upper()}/{case.timestep.upper()}, "
      f"CFL={case.cfl:g}, BC={case.bc}, eos={case.eos}, g_x={case.gravity_a1:g}, nx={x.size}, t={time_value:.5f}",
      fontsize=12,
   )
   for ax, numerical, exact, ylabel in (
      (axs[0], rho, exact_rho, "rho"),
      (axs[1], pressure, exact_pressure, "P"),
   ):
      ax.plot(x, numerical, color="tab:blue", marker="o", markersize=2.3, linewidth=1.1, label="num")
      ax.plot(x, exact, color="black", linestyle=":", linewidth=2.0, label="static HSE")
      ax.set_ylabel(ylabel)
      ax.grid(True, which="major", alpha=0.25)
      ax.legend()
   axs[2].plot(x, vx, color="tab:blue", marker="o", markersize=2.3, linewidth=1.1)
   axs[2].axhline(0.0, color="black", linestyle=":", linewidth=1.6)
   axs[2].set_ylabel("vx")
   axs[2].set_xlabel("x")
   axs[2].grid(True, which="major", alpha=0.25)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def analytic_phi_error(case: GravityCase, run_dir: Path) -> dict[str, float]:
   plotfile = latest_aux_plotfile(run_dir)
   names, _time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   name_to_index = {name: index for index, name in enumerate(names)}
   if "phi" not in name_to_index:
      raise RuntimeError(f"aux plotfile {plotfile} does not contain phi")
   if "grad_phi1" not in name_to_index:
      raise RuntimeError(f"aux plotfile {plotfile} does not contain grad_phi1")
   phi = state[name_to_index["phi"]]
   grad_phi1 = state[name_to_index["grad_phi1"]]
   nx = phi.size
   x = (np.arange(nx, dtype=float) + 0.5) / nx
   k = 2.0 * math.pi
   exact = -(4.0 * math.pi * case.gravity_constant * case.amplitude / (k * k)) * np.sin(k * x)
   exact_grad = -(4.0 * math.pi * case.gravity_constant * case.amplitude / k) * np.cos(k * x)
   phi = phi - np.mean(phi)
   exact = exact - np.mean(exact)
   diff = phi - exact
   grad_diff = grad_phi1 - exact_grad
   scale = max(float(np.sqrt(np.mean(exact * exact))), 1.0e-30)
   grad_scale = max(float(np.sqrt(np.mean(exact_grad * exact_grad))), 1.0e-30)
   return {
      "phi_rel_l2": float(np.sqrt(np.mean(diff * diff)) / scale),
      "phi_abs_linf": float(np.max(np.abs(diff))),
      "grad_phi_rel_l2": float(np.sqrt(np.mean(grad_diff * grad_diff)) / grad_scale),
      "grad_phi_abs_linf": float(np.max(np.abs(grad_diff))),
      "phi_num_l2": float(np.sqrt(np.mean(phi * phi))),
      "phi_exact_l2": float(np.sqrt(np.mean(exact * exact))),
   }


def phi_finiteness(run_dir: Path) -> dict[str, float]:
   plotfile = latest_aux_plotfile(run_dir)
   names, _time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   name_to_index = {name: index for index, name in enumerate(names)}
   if "phi" not in name_to_index:
      raise RuntimeError(f"aux plotfile {plotfile} does not contain phi")
   phi = state[name_to_index["phi"]]
   finite = np.isfinite(phi)
   return {
      "phi_finite_fraction": float(np.mean(finite)),
      "phi_abs_linf": float(np.max(np.abs(phi[finite]))) if np.any(finite) else float("inf"),
      "phi_num_l2": float(np.sqrt(np.mean(phi[finite] * phi[finite]))) if np.any(finite) else float("inf"),
   }


def fixed_uniform_error(case: GravityCase, run_dir: Path) -> dict[str, float]:
   plotfile = latest_aux_plotfile(run_dir)
   names, _time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   name_to_index = {name: index for index, name in enumerate(names)}
   phi = state[name_to_index["phi"]]
   grad_phi1 = state[name_to_index["grad_phi1"]]
   nx = phi.size
   x = (np.arange(nx, dtype=float) + 0.5) / nx
   exact_phi = -case.gravity_a1 * x
   exact_grad = np.full_like(x, -case.gravity_a1)
   scalar_rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
   if not scalar_rows:
      raise RuntimeError(f"{run_dir / 'scalars.tsv'} missing scalar rows")
   scalar = scalar_rows[-1]
   potential_energy = float(scalar["gravity_potential_energy"])
   expected_potential_energy = -0.5 * case.rho0 * case.gravity_a1
   total_energy = float(scalar["gravity_total_energy"])
   fluid_energy = float(scalar["energy"])
   return {
      "fixed_phi_abs_linf": float(np.max(np.abs(phi - exact_phi))),
      "fixed_grad_phi_abs_linf": float(np.max(np.abs(grad_phi1 - exact_grad))),
      "gravity_potential_energy_abs_error": abs(potential_energy - expected_potential_energy),
      "gravity_total_energy_closure": abs(total_energy - fluid_energy - potential_energy),
   }


def fixed_uniform_evolution_error(run_dir: Path) -> dict[str, float]:
   scalar_rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
   if not scalar_rows:
      raise RuntimeError(f"{run_dir / 'scalars.tsv'} missing scalar rows")
   final = scalar_rows[-1]
   return {
      "gravity_total_energy_rel_drift": abs(float(final["gravity_total_energy_rel_drift"])),
      "mass_rel_drift": abs(float(final["mass_rel_drift"])),
      "energy_rel_drift": abs(float(final["energy_rel_drift"])),
   }


def validate_case(case: GravityCase, args: argparse.Namespace) -> tuple[dict[str, object], int]:
   run_dir, input_path, returncode, elapsed, stdout = run_case(case, args.force)
   elliptic_policy = scalar_checks("elliptic", performance=False)
   gravity_evolution_policy = scalar_checks(
      "gravity_evolution",
      energy=False,
      momentum=False,
      performance=False,
      mass_tol=args.mass_drift_tol,
   )
   hse_policy = scalar_checks(
      "hse",
      energy=False,
      momentum=False,
      performance=False,
      mass_tol=args.mass_drift_tol,
   )
   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=input_path,
      policy=hse_policy if case.check == "hse_polytrope" else (
         gravity_evolution_policy if case.check == "fixed_uniform_evolution" else elliptic_policy
      ),
      mhd=case.physics == "mhd",
      dim=1,
      ranks=case.ranks,
      base_input=str(input_path.relative_to(ROOT)),
      kind="analytic" if case.bc == "periodic" else "feature",
   )
   row["bc"] = case.bc
   row["returncode"] = returncode
   row["seconds"] = f"{elapsed:.6g}"
   if returncode != 0:
      row["status"] = "FAIL"
      row["reason"] = f"solver returned {returncode}"
      row["tail"] = " | ".join(stdout.splitlines()[-6:])
      return row, 1

   failures = []
   if case.gravity == "poisson" and case.bc == "periodic":
      errors = analytic_phi_error(case, run_dir)
      if errors["phi_rel_l2"] > args.phi_rel_l2_tol:
         failures.append(f"phi_rel_l2={errors['phi_rel_l2']:.3e} > {args.phi_rel_l2_tol:.3e}")
      if errors["phi_abs_linf"] > args.phi_abs_linf_tol:
         failures.append(f"phi_abs_linf={errors['phi_abs_linf']:.3e} > {args.phi_abs_linf_tol:.3e}")
      if errors["grad_phi_rel_l2"] > args.grad_phi_rel_l2_tol:
         failures.append(f"grad_phi_rel_l2={errors['grad_phi_rel_l2']:.3e} > {args.grad_phi_rel_l2_tol:.3e}")
      row.update(errors)
   elif case.check == "fixed_uniform_fields":
      errors = fixed_uniform_error(case, run_dir)
      if errors["fixed_phi_abs_linf"] > args.fixed_phi_abs_linf_tol:
         failures.append(f"fixed_phi_abs_linf={errors['fixed_phi_abs_linf']:.3e} > {args.fixed_phi_abs_linf_tol:.3e}")
      if errors["fixed_grad_phi_abs_linf"] > args.fixed_grad_phi_abs_linf_tol:
         failures.append(f"fixed_grad_phi_abs_linf={errors['fixed_grad_phi_abs_linf']:.3e} > {args.fixed_grad_phi_abs_linf_tol:.3e}")
      if errors["gravity_potential_energy_abs_error"] > args.gravity_energy_tol:
         failures.append(
            f"gravity_potential_energy_abs_error={errors['gravity_potential_energy_abs_error']:.3e} "
            f"> {args.gravity_energy_tol:.3e}"
         )
      if errors["gravity_total_energy_closure"] > args.gravity_energy_tol:
         failures.append(f"gravity_total_energy_closure={errors['gravity_total_energy_closure']:.3e} > {args.gravity_energy_tol:.3e}")
      row.update(errors)
   elif case.check == "fixed_uniform_evolution":
      errors = fixed_uniform_evolution_error(run_dir)
      if errors["gravity_total_energy_rel_drift"] > args.gravity_total_energy_drift_tol:
         failures.append(
            f"gravity_total_energy_rel_drift={errors['gravity_total_energy_rel_drift']:.3e} "
            f"> {args.gravity_total_energy_drift_tol:.3e}"
         )
      if errors["mass_rel_drift"] > args.mass_drift_tol:
         failures.append(f"mass_rel_drift={errors['mass_rel_drift']:.3e} > {args.mass_drift_tol:.3e}")
      row.update(errors)
   elif case.check == "hse_polytrope":
      errors = hse_polytrope_error(case, run_dir, plot=args.plots)
      if errors["hse_max_mach"] > args.hse_max_mach_tol:
         failures.append(f"hse_max_mach={errors['hse_max_mach']:.3e} > {args.hse_max_mach_tol:.3e}")
      if errors["hse_rho_rel_l2"] > args.hse_profile_rel_l2_tol:
         failures.append(f"hse_rho_rel_l2={errors['hse_rho_rel_l2']:.3e} > {args.hse_profile_rel_l2_tol:.3e}")
      if errors["hse_pressure_rel_l2"] > args.hse_profile_rel_l2_tol:
         failures.append(
            f"hse_pressure_rel_l2={errors['hse_pressure_rel_l2']:.3e} > {args.hse_profile_rel_l2_tol:.3e}"
         )
      if errors["gravity_total_energy_rel_drift"] > args.hse_gravity_total_energy_drift_tol:
         failures.append(
            f"gravity_total_energy_rel_drift={errors['gravity_total_energy_rel_drift']:.3e} "
            f"> {args.hse_gravity_total_energy_drift_tol:.3e}"
         )
      row.update(errors)
   else:
      errors = phi_finiteness(run_dir)
      if errors["phi_finite_fraction"] < 1.0:
         failures.append(f"phi finite fraction={errors['phi_finite_fraction']:.6g}")
      row.update(errors)

   if row["status"] != "PASS":
      failures.append(str(row["reason"]))
   if failures:
      row["status"] = "FAIL"
      row["reason"] = "; ".join(failures)
   elif case.gravity == "poisson" and case.bc == "periodic":
      row["status"] = "PASS"
      row["reason"] = (
         f"analytic phi_rel_l2={row['phi_rel_l2']:.3e}, "
         f"phi_abs_linf={row['phi_abs_linf']:.3e}, "
         f"grad_phi_rel_l2={row['grad_phi_rel_l2']:.3e}"
      )
   elif case.check == "fixed_uniform_fields":
      row["status"] = "PASS"
      row["reason"] = (
         f"fixed phi_abs={row['fixed_phi_abs_linf']:.3e}, "
         f"grad_abs={row['fixed_grad_phi_abs_linf']:.3e}, "
         f"Egrav_abs={row['gravity_potential_energy_abs_error']:.3e}"
      )
   elif case.check == "fixed_uniform_evolution":
      row["status"] = "PASS"
      row["reason"] = f"gravity_total_energy_rel_drift={row['gravity_total_energy_rel_drift']:.3e}"
   elif case.check == "hse_polytrope":
      row["status"] = "PASS"
      row["reason"] = (
         f"HSE max Mach={row['hse_max_mach']:.3e}, "
         f"rho_rel_l2={row['hse_rho_rel_l2']:.3e}, "
         f"P_rel_l2={row['hse_pressure_rel_l2']:.3e}"
      )
   else:
      row["status"] = "PASS"
      row["reason"] = "homogeneous-Neumann gravity BC smoke passed with finite phi"
   return row, 0 if row["status"] == "PASS" else 1


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--nx", type=int, default=128)
   parser.add_argument("--skip-bc-smoke", action="store_true")
   parser.add_argument("--phi-rel-l2-tol", type=float, default=2.0e-3)
   parser.add_argument("--phi-abs-linf-tol", type=float, default=1.0e-6)
   parser.add_argument("--grad-phi-rel-l2-tol", type=float, default=3.0e-3)
   parser.add_argument("--fixed-phi-abs-linf-tol", type=float, default=1.0e-14)
   parser.add_argument("--fixed-grad-phi-abs-linf-tol", type=float, default=1.0e-14)
   parser.add_argument("--gravity-energy-tol", type=float, default=1.0e-13)
   parser.add_argument("--gravity-total-energy-drift-tol", type=float, default=2.0e-8)
   parser.add_argument("--mass-drift-tol", type=float, default=1.0e-12)
   parser.add_argument("--hse-max-mach-tol", type=float, default=2.0e-3)
   parser.add_argument("--hse-profile-rel-l2-tol", type=float, default=5.0e-4)
   parser.add_argument("--hse-gravity-total-energy-drift-tol", type=float, default=5.0e-7)
   parser.add_argument("--plots", action="store_true", help="write human-facing gravity validation plots")
   args = parser.parse_args()

   cases = [GravityCase(ranks=args.ranks, nx=args.nx)]
   cases.append(GravityCase(
      name="poisson_density_sine_1d_ssprk3",
      ranks=args.ranks,
      nx=args.nx,
      max_step=4,
      final_time=-1.0,
      timestep="ssprk3",
      gravity="poisson",
      check="poisson_fields",
      bc="periodic",
   ))
   cases.append(GravityCase(
      name="poisson_density_sine_1d_rk4",
      ranks=args.ranks,
      nx=args.nx,
      max_step=4,
      final_time=-1.0,
      timestep="rk4",
      gravity="poisson",
      check="poisson_fields",
      bc="periodic",
   ))
   cases.append(GravityCase(
      name="fixed_uniform_gravity_1d",
      ranks=args.ranks,
      nx=args.nx,
      bc="reflecting",
      gravity="fixed",
      check="fixed_uniform_fields",
      rho0=1.0,
      amplitude=0.0,
      gravity_a1=0.1,
   ))
   cases.append(GravityCase(
      name="fixed_uniform_gravity_evolved_1d",
      ranks=args.ranks,
      nx=args.nx,
      max_step=8,
      final_time=0.02,
      bc="reflecting",
      gravity="fixed",
      check="fixed_uniform_evolution",
      rho0=1.0,
      amplitude=0.0,
      gravity_a1=0.1,
      aux_gfs="none",
      aux_gf_int=-1,
      aux_gf_compute_int=-1,
   ))
   cases.append(GravityCase(
      name="fixed_uniform_gravity_evolved_1d_rk4",
      ranks=args.ranks,
      nx=args.nx,
      max_step=8,
      final_time=0.02,
      bc="reflecting",
      gravity="fixed",
      check="fixed_uniform_evolution",
      rho0=1.0,
      amplitude=0.0,
      gravity_a1=0.1,
      timestep="rk4",
      aux_gfs="none",
      aux_gf_int=-1,
      aux_gf_compute_int=-1,
   ))
   cases.append(GravityCase(
      name="hse_polytrope_x_1d",
      ranks=args.ranks,
      nx=args.nx,
      max_step=16,
      ng=2,
      bc="reflecting",
      ic="hse_polytrope_x",
      eos="polytrope",
      polytropic_k=1.0,
      polytropic_gamma=5.0 / 3.0,
      rho0=1.0,
      pressure0=1.0,
      gravity="fixed",
      check="hse_polytrope",
      gravity_a1=0.1,
      recon="pwl",
      riemann="hllc",
      timestep="ssprk3",
      plot_int=16,
      aux_gfs="none",
      aux_gf_int=-1,
      aux_gf_compute_int=-1,
   ))
   if not args.skip_bc_smoke:
      for bc in ("outflow", "reflecting", "symmetry", "inflow", "diode"):
         cases.append(GravityCase(name=f"poisson_density_sine_1d_bc_{bc}", ranks=args.ranks, nx=args.nx, bc=bc))

   rows = []
   status = 0
   for case in cases:
      row, case_status = validate_case(case, args)
      rows.append(row)
      status = max(status, case_status)
   write_tsv(
      SUMMARY,
      rows,
      [
         "case",
         "status",
         "bc",
         "reason",
         "returncode",
         "seconds",
         "phi_rel_l2",
         "phi_abs_linf",
         "grad_phi_rel_l2",
         "grad_phi_abs_linf",
         "fixed_phi_abs_linf",
         "fixed_grad_phi_abs_linf",
         "gravity_potential_energy_abs_error",
         "gravity_total_energy_closure",
         "gravity_total_energy_rel_drift",
         "mass_rel_drift",
         "energy_rel_drift",
         "hse_max_mach",
         "hse_max_abs_vx",
         "hse_rho_rel_l2",
         "hse_pressure_rel_l2",
         "profile_time",
         "plot",
         "phi_num_l2",
         "phi_exact_l2",
         "phi_finite_fraction",
      ],
   )
   append_validation_history(HISTORY, rows)
   for row in rows:
      print(f"{row['case']}: {row['status']} {row['reason']}")
   return status


if __name__ == "__main__":
   raise SystemExit(main())
