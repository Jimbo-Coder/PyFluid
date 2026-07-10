#!/usr/bin/env python3
"""Validate HD/MHD characteristic and self-gravity dispersion relations."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_amrex_1d_profile import parse_plot_header, read_level0_state_1d  # noqa: E402
from plotfile_reader import latest_plotfile
from pyfluid.dispersion import (  # noqa: E402
   hydro_acoustic_frequencies,
   ideal_mhd_frequencies,
   jeans_wave_exact,
   linear_wave_x_exact,
   sound_speed_ideal_gas,
)
from validation_common import mpi_prefix
from validation_common import (  # noqa: E402
   CLOSED_POLICY,
   append_validation_history,
   average_positive,
   max_abs_column,
   read_scalar_rows,
   scalar_checks,
   scalar_validation_row,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE_1D = ROOT / "build" / "pyfluid_amrex" / "pyfluid_amrex_1d"
RUN_ROOT = ROOT / "runs/current/dispersion_validation"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
PLOT_ROOT = ROOT / "plots/current/scans"


@dataclass(frozen=True)
class DispersionCase:
   name: str
   label: str
   physics: str
   ic: str
   family: str
   compare_field: str
   gravity: str = "none"
   gravity_constant: float = 0.0
   nx: int = 256
   final_time: float = 0.08
   amplitude: float = 1.0e-6
   wave_number: float = 1.0
   rho0: float = 1.0
   pressure0: float = 1.0
   gamma: float = 5.0 / 3.0
   v1_0: float = 0.0
   v2_0: float = 0.0
   v3_0: float = 0.0
   b1_0: float = 0.0
   b2_0: float = 0.0
   b3_0: float = 0.0
   recon: str = "wenoz-char"
   riemann: str = "hllc"
   timestep: str = "rk4"
   ng: int = 3
   extra_modes: tuple[int, ...] = ()


def base_case_name(name: str) -> str:
   parts = name.rsplit("_m", 1)
   if len(parts) == 2 and parts[1].isdigit():
      return parts[0]
   return name


HD_VX0 = 0.2
MHD_VX0 = 0.1
MHD_FIELD = {"b1_0": 1.0, "b2_0": 0.45, "b3_0": 0.2}

CASES = [
   DispersionCase("hd_acoustic_plus", "HD acoustic +", "hydro", "linear_wave_x", "acoustic_plus", "rho", v1_0=HD_VX0),
   DispersionCase("hd_acoustic_minus", "HD acoustic -", "hydro", "linear_wave_x", "acoustic_minus", "rho", v1_0=HD_VX0),
   DispersionCase("hd_entropy", "HD entropy", "hydro", "linear_wave_x", "entropy", "rho", v1_0=HD_VX0),
   DispersionCase("hd_shear_y", "HD shear y", "hydro", "linear_wave_x", "shear_y", "vy", v1_0=HD_VX0),
   DispersionCase("hd_shear_z", "HD shear z", "hydro", "linear_wave_x", "shear_z", "vz", v1_0=HD_VX0),
   DispersionCase(
      "mhd_fast_plus",
      "MHD fast +",
      "mhd",
      "linear_wave_x",
      "fast_plus",
      "vx",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_fast_minus",
      "MHD fast -",
      "mhd",
      "linear_wave_x",
      "fast_minus",
      "vx",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_alfven_plus",
      "MHD Alfvén +",
      "mhd",
      "linear_wave_x",
      "alfven_plus",
      "b2",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_alfven_minus",
      "MHD Alfvén -",
      "mhd",
      "linear_wave_x",
      "alfven_minus",
      "b2",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_slow_plus",
      "MHD slow +",
      "mhd",
      "linear_wave_x",
      "slow_plus",
      "vx",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_slow_minus",
      "MHD slow -",
      "mhd",
      "linear_wave_x",
      "slow_minus",
      "vx",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "mhd_entropy",
      "MHD entropy",
      "mhd",
      "linear_wave_x",
      "entropy",
      "rho",
      v1_0=MHD_VX0,
      riemann="hlld",
      **MHD_FIELD,
   ),
   DispersionCase(
      "jeans_stable_g1",
      "Jeans stable G=1",
      "hydro",
      "jeans_wave_x",
      "jeans_stable",
      "rho",
      gravity="poisson",
      gravity_constant=1.0,
      final_time=0.1,
   ),
   DispersionCase(
      "jeans_dense_rho10",
      "Jeans dense rho=10",
      "hydro",
      "jeans_wave_x",
      "jeans_unstable",
      "rho",
      gravity="poisson",
      gravity_constant=1.0,
      rho0=10.0,
      final_time=0.1,
      amplitude=1.0e-7,
      extra_modes=(5,),
   ),
]


def build_cases(args: argparse.Namespace) -> list[DispersionCase]:
   cases: list[DispersionCase] = []
   modes = sorted(set(args.rate_modes))
   for case in CASES:
      case_modes = sorted(set(modes).union(case.extra_modes))
      for mode in case_modes:
         if mode == 1:
            cases.append(replace(case, wave_number=float(mode)))
         else:
            cases.append(replace(case, name=f"{case.name}_m{mode}", wave_number=float(mode)))
   return cases


def next_output_path(directory: Path, stem: str) -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   used = []
   for path in directory.glob(f"{stem}_*.png"):
      suffix = path.stem.removeprefix(f"{stem}_")
      if suffix.isdigit():
         used.append(int(suffix))
   return directory / f"{stem}_{max(used, default=0) + 1:03d}.png"


def case_input(case: DispersionCase, run_dir: Path, args: argparse.Namespace) -> str:
   gravity_block = "pyfluid.gravity = none\n"
   if case.gravity == "poisson":
      gravity_block = f"""pyfluid.gravity = poisson
pyfluid.gravity_constant = {case.gravity_constant:.17g}
pyfluid.gravity_tolerance = 1.0e-12
pyfluid.gravity_max_iterations = 200
pyfluid.gravity_max_fmg_iterations = 20
pyfluid.gravity_max_order = 2
"""
   return f"""pyfluid.run_name = {case.name}
pyfluid.output_dir = {run_dir.relative_to(ROOT)}

geometry.prob_lo = 0.0
geometry.prob_hi = 1.0
geometry.is_periodic = 1

amr.n_cell = {case.nx}
amr.max_level = 0
amr.max_grid_size = 64
amr.blocking_factor = 8

pyfluid.max_step = 100000
pyfluid.final_time = {case.final_time:.17g}
pyfluid.ng = {case.ng}
pyfluid.cfl = {args.cfl:.17g}
pyfluid.gamma = {case.gamma:.17g}
pyfluid.eos = ideal_gas

pyfluid.physics = {case.physics}
pyfluid.ct = off
pyfluid.ic = {case.ic}
pyfluid.wave_family = {case.family}
pyfluid.rho0 = {case.rho0:.17g}
pyfluid.pressure0 = {case.pressure0:.17g}
pyfluid.amplitude = {case.amplitude:.17g}
pyfluid.alfven_amplitude = {case.amplitude:.17g}
pyfluid.wave_number = {case.wave_number:.17g}
pyfluid.v1_0 = {case.v1_0:.17g}
pyfluid.v2_0 = {case.v2_0:.17g}
pyfluid.v3_0 = {case.v3_0:.17g}
pyfluid.b1_0 = {case.b1_0:.17g}
pyfluid.b2_0 = {case.b2_0:.17g}
pyfluid.b3_0 = {case.b3_0:.17g}

pyfluid.recon = {case.recon}
pyfluid.riemann = {case.riemann}
pyfluid.timestep = {case.timestep}
pyfluid.bc = periodic

pyfluid.diffusion = none
pyfluid.source = none
{gravity_block}
pyfluid.scalar_int = 8
pyfluid.plot_int = 100000
pyfluid.checkpoint_int = -1
pyfluid.aux_gfs = none
pyfluid.aux_gf_int = -1
pyfluid.aux_gf_compute_int = -1
"""


def run_case(case: DispersionCase, args: argparse.Namespace) -> tuple[Path, Path, int, float, str]:
   run_root = getattr(args, "run_root", RUN_ROOT)
   run_dir = run_root / case.name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = RUN_ROOT / "inputs" / f"{case.name}.in"
   input_path.parent.mkdir(parents=True, exist_ok=True)
   input_path.write_text(case_input(case, run_dir, args))
   command = [*mpi_prefix(args.ranks), str(EXECUTABLE_1D), str(input_path)]
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


def primitive_from_state(case: DispersionCase, names: list[str], state: np.ndarray) -> dict[str, np.ndarray]:
   index = {name: n for n, name in enumerate(names)}
   rho = state[index["rho"]]
   vx = state[index["mom1"]] / rho
   vy = state[index["mom2"]] / rho
   vz = state[index["mom3"]] / rho
   energy = state[index["energy"]]
   result = {"rho": rho, "vx": vx, "vy": vy, "vz": vz}
   if "b1" in index:
      bx = state[index["b1"]]
      by = state[index["b2"]]
      bz = state[index["b3"]]
      pressure = (case.gamma - 1.0) * (
         energy - 0.5 * rho * (vx * vx + vy * vy + vz * vz) - 0.5 * (bx * bx + by * by + bz * bz)
      )
      result.update({"pressure": pressure, "b1": bx, "b2": by, "b3": bz})
   else:
      pressure = (case.gamma - 1.0) * (energy - 0.5 * rho * (vx * vx + vy * vy + vz * vz))
      result["pressure"] = pressure
   return result


def project_sin_cos(values: np.ndarray, x: np.ndarray, wave_number: float) -> tuple[float, float]:
   phase = 2.0 * math.pi * wave_number * x
   return 2.0 * float(np.mean(values * np.sin(phase))), 2.0 * float(np.mean(values * np.cos(phase)))


def fourier_coefficient(values: np.ndarray, x: np.ndarray, wave_number: float) -> complex:
   sin_coeff, cos_coeff = project_sin_cos(values, x, wave_number)
   return complex(sin_coeff, -cos_coeff)


def measured_phase_rate(
   values: np.ndarray,
   initial_values: np.ndarray,
   x: np.ndarray,
   wave_number: float,
   time_value: float,
   expected_rate: float,
) -> tuple[float, float]:
   final_coeff = fourier_coefficient(values, x, wave_number)
   initial_coeff = fourier_coefficient(initial_values, x, wave_number)
   if abs(initial_coeff) <= 0.0:
      raise RuntimeError("cannot measure phase of a zero-amplitude initial projection")
   phase = math.atan2((final_coeff / initial_coeff).imag, (final_coeff / initial_coeff).real)
   expected_phase = expected_rate * time_value
   while phase - expected_phase > math.pi:
      phase -= 2.0 * math.pi
   while phase - expected_phase < -math.pi:
      phase += 2.0 * math.pi
   return phase / time_value, abs(final_coeff)


def measured_growth_rate(values: np.ndarray, x: np.ndarray, wave_number: float, amplitude: float, time_value: float) -> tuple[float, float]:
   sin_coeff, cos_coeff = project_sin_cos(values, x, wave_number)
   measured_amplitude = math.sqrt(sin_coeff * sin_coeff + cos_coeff * cos_coeff)
   return math.log(max(measured_amplitude, 1.0e-300) / amplitude) / time_value, measured_amplitude


def base_value(case: DispersionCase, field: str) -> float:
   return {
      "rho": case.rho0,
      "vx": case.v1_0,
      "vy": case.v2_0,
      "vz": case.v3_0,
      "pressure": case.pressure0,
      "b1": case.b1_0,
      "b2": case.b2_0,
      "b3": case.b3_0,
   }[field]


def normalized_field_error(case: DispersionCase, num: dict[str, np.ndarray], exact: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
   fields = ["rho", "vx", "vy", "vz", "pressure"]
   if case.physics == "mhd":
      fields += ["b1", "b2", "b3"]
   errors = {}
   for field in fields:
      scale = max(float(np.sqrt(np.mean((exact[field] - base_value(case, field)) ** 2))), case.amplitude, 1.0e-30)
      errors[field] = float(np.sqrt(np.mean((num[field] - exact[field]) ** 2)) / scale)
   return max(errors.values()), errors


def collect_result(case: DispersionCase, run_dir: Path) -> dict[str, object]:
   plotfile = latest_plotfile(run_dir, require=True)
   names, time_value = parse_plot_header(plotfile)
   state = read_level0_state_1d(plotfile, names)
   ncell = state.shape[1]
   x = (np.arange(ncell, dtype=float) + 0.5) / ncell
   num = primitive_from_state(case, names, state)

   if case.ic == "jeans_wave_x":
      exact_rho, exact_pressure, exact_vx, mode = jeans_wave_exact(
         x,
         time_value,
         case.gamma,
         case.rho0,
         case.pressure0,
         case.amplitude,
         case.wave_number,
         case.gravity_constant,
      )
      exact = {
         "rho": exact_rho,
         "pressure": exact_pressure,
         "vx": exact_vx,
         "vy": np.full_like(x, case.v2_0),
         "vz": np.full_like(x, case.v3_0),
      }
      initial_rho, initial_pressure, initial_vx, _initial_mode = jeans_wave_exact(
         x,
         0.0,
         case.gamma,
         case.rho0,
         case.pressure0,
         case.amplitude,
         case.wave_number,
         case.gravity_constant,
      )
      initial = {
         "rho": initial_rho,
         "pressure": initial_pressure,
         "vx": initial_vx,
         "vy": np.full_like(x, case.v2_0),
         "vz": np.full_like(x, case.v3_0),
      }
      field_error, component_errors = normalized_field_error(case, num, exact)
      if mode.branch == "stable":
         theory_rate = mode.omega
         measured_rate, measured_amplitude = measured_phase_rate(
            num["rho"] - case.rho0,
            initial["rho"] - case.rho0,
            x,
            case.wave_number,
            time_value,
            theory_rate,
         )
         rate_name = "omega"
      else:
         theory_rate = mode.growth_rate
         measured_rate, measured_amplitude = measured_growth_rate(
            num["rho"] - case.rho0, x, case.wave_number, case.amplitude, time_value
         )
         rate_name = "growth"
   else:
      exact, theory_rate = linear_wave_x_exact(
         x,
         time_value,
         physics=case.physics,
         family=case.family,
         gamma=case.gamma,
         rho0=case.rho0,
         pressure0=case.pressure0,
         amplitude=case.amplitude,
         wave_number=case.wave_number,
         v1_0=case.v1_0,
         v2_0=case.v2_0,
         v3_0=case.v3_0,
         b1_0=case.b1_0,
         b2_0=case.b2_0,
         b3_0=case.b3_0,
      )
      initial, _initial_rate = linear_wave_x_exact(
         x,
         0.0,
         physics=case.physics,
         family=case.family,
         gamma=case.gamma,
         rho0=case.rho0,
         pressure0=case.pressure0,
         amplitude=case.amplitude,
         wave_number=case.wave_number,
         v1_0=case.v1_0,
         v2_0=case.v2_0,
         v3_0=case.v3_0,
         b1_0=case.b1_0,
         b2_0=case.b2_0,
         b3_0=case.b3_0,
      )
      field_error, component_errors = normalized_field_error(case, num, exact)
      measured_rate, measured_amplitude = measured_phase_rate(
         num[case.compare_field] - base_value(case, case.compare_field),
         initial[case.compare_field] - base_value(case, case.compare_field),
         x,
         case.wave_number,
         time_value,
         theory_rate,
      )
      rate_name = "omega"

   rate_abs_error = abs(measured_rate - theory_rate)
   rate_rel_error = rate_abs_error / max(abs(theory_rate), 1.0)
   scalars = read_scalar_rows(run_dir)
   return {
      "case": case.name,
      "base_case": base_case_name(case.name),
      "label": case.label,
      "physics": case.physics,
      "family": case.family,
      "gravity": case.gravity,
      "gravity_constant": case.gravity_constant,
      "nx": ncell,
      "time": time_value,
      "mode_number": case.wave_number,
      "theory_rate": theory_rate,
      "measured_rate": measured_rate,
      "rate_abs_error": rate_abs_error,
      "rate_rel_error": rate_rel_error,
      "rate_name": rate_name,
      "measured_amplitude": measured_amplitude,
      "field_error": field_error,
      "component_errors": component_errors,
      "avg_mcells_per_s": average_positive(scalars, "mcells_per_s"),
      "avg_sim_time_per_s": average_positive(scalars, "sim_time_per_s"),
      "mass_rel_drift": max_abs_column(scalars, "mass_rel_drift"),
      "energy_rel_drift": max_abs_column(scalars, "energy_rel_drift"),
      "gravity_total_energy_rel_drift": max_abs_column(scalars, "gravity_total_energy_rel_drift"),
      "plotfile": plotfile,
      "run_dir": run_dir,
      "profile": {
         "x": x,
         "primary": num[case.compare_field] - base_value(case, case.compare_field),
         "exact": exact[case.compare_field] - base_value(case, case.compare_field),
         "initial": initial[case.compare_field] - base_value(case, case.compare_field),
         "ylabel": f"{case.compare_field} perturbation",
      },
   }


def validation_policy(case: DispersionCase):
   if case.gravity == "poisson":
      return scalar_checks("self_gravity_dispersion", energy=False, momentum=True, performance=True, mass_tol=1.0e-10)
   return CLOSED_POLICY


def validate_case(case: DispersionCase, args: argparse.Namespace) -> tuple[dict[str, object], int]:
   run_dir, input_path, returncode, elapsed, stdout = run_case(case, args)
   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=input_path,
      policy=validation_policy(case),
      mhd=case.physics == "mhd",
      dim=1,
      ranks=args.ranks,
      base_input=str(input_path.relative_to(ROOT)),
      kind="dispersion",
   )
   row["returncode"] = returncode
   row["seconds"] = f"{elapsed:.6g}"
   if returncode != 0:
      row["status"] = "FAIL"
      row["reason"] = f"solver returned {returncode}"
      row["tail"] = " | ".join(stdout.splitlines()[-6:])
      return row, 1

   result = collect_result(case, run_dir)
   row.update({key: result[key] for key in (
      "base_case",
      "mode_number",
      "theory_rate",
      "measured_rate",
      "rate_abs_error",
      "rate_rel_error",
      "field_error",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "gravity_total_energy_rel_drift",
   )})
   failures = []
   if row["status"] != "PASS":
      failures.append(str(row["reason"]))
   if float(result["field_error"]) > args.field_error_tol:
      failures.append(f"field_error={result['field_error']:.3e} > {args.field_error_tol:.3e}")
   if float(result["rate_abs_error"]) > args.rate_abs_error_tol and float(result["rate_rel_error"]) > args.rate_rel_error_tol:
      failures.append(
         f"rate_abs_error={result['rate_abs_error']:.3e}, rate_rel_error={result['rate_rel_error']:.3e}"
      )
   if failures:
      row["status"] = "FAIL"
      row["reason"] = "; ".join(failures)
   else:
      row["status"] = "PASS"
      row["reason"] = (
         f"{result['rate_name']} num/theory={result['measured_rate']:.6g}/{result['theory_rate']:.6g}, "
         f"field_error={result['field_error']:.3e}"
      )
   return row, 0 if row["status"] == "PASS" else 1


def plot_results(results: list[dict[str, object]], output: Path, args: argparse.Namespace) -> None:
   fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
   fig.suptitle("AMReX characteristic dispersion validation: WENO-Z char/RK4, CFL=0.9, BC=periodic, nx=256")

   rho0 = 1.0
   pressure0 = 1.0
   gamma = 5.0 / 3.0
   cs = sound_speed_ideal_gas(gamma, pressure0, rho0)
   dense_jeans = next(case for case in CASES if case.name == "jeans_dense_rho10")
   dense_cs = sound_speed_ideal_gas(dense_jeans.gamma, dense_jeans.pressure0, dense_jeans.rho0)
   dense_cutoff = math.sqrt(4.0 * math.pi * dense_jeans.gravity_constant * dense_jeans.rho0) / (2.0 * math.pi * dense_cs)
   mode = np.linspace(0.15, max(4.0, dense_cutoff * 1.08, max(float(row["mode_number"]) for row in results)), 300)
   k = 2.0 * math.pi * mode
   hd_minus, hd_plus = hydro_acoustic_frequencies(k, cs, HD_VX0)
   jeans_omega2 = cs * cs * k * k - 4.0 * math.pi * rho0
   dense_jeans_omega2 = (
      dense_cs * dense_cs * k * k - 4.0 * math.pi * dense_jeans.gravity_constant * dense_jeans.rho0
   )
   jeans_stable_g1 = np.where(jeans_omega2 >= 0.0, np.sqrt(np.clip(jeans_omega2, 0.0, None)), np.nan)
   dense_jeans_growth = np.where(dense_jeans_omega2 < 0.0, np.sqrt(np.clip(-dense_jeans_omega2, 0.0, None)), np.nan)
   dense_jeans_stable = np.where(dense_jeans_omega2 >= 0.0, np.sqrt(np.clip(dense_jeans_omega2, 0.0, None)), np.nan)
   mhd = ideal_mhd_frequencies(k, rho0, pressure0, gamma, (1.0, 0.45, 0.2), (1.0, 0.0, 0.0), MHD_VX0)

   ax = axs[0, 0]
   branch_specs = [
      ("hd_acoustic_plus", "HD acoustic +", hd_plus, None, "o"),
      ("hd_acoustic_minus", "HD acoustic -", hd_minus, None, "o"),
      ("jeans_stable_g1", "Jeans stable G=1", jeans_stable_g1, None, "P"),
      ("jeans_dense_rho10", "Jeans growth rho=10 (m<mJ)", dense_jeans_growth, "growth", "D"),
      ("jeans_dense_rho10", "Jeans stable rho=10 (m>mJ)", dense_jeans_stable, "omega", "s"),
      ("mhd_fast_plus", "MHD fast +", mhd["fast+"], None, "o"),
      ("mhd_alfven_plus", "MHD Alfvén +", mhd["alfven+"], None, "o"),
      ("mhd_slow_plus", "MHD slow +", mhd["slow+"], None, "o"),
   ]
   colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
   for idx, (base_case, label, analytic, rate_filter, marker) in enumerate(branch_specs):
      color = colors[idx % len(colors)]
      is_jeans = label.startswith("Jeans")
      ax.plot(mode, analytic, color=color, linewidth=2.6 if is_jeans else 2.2, alpha=0.42, label=label)
      rows = sorted(
         [
            row for row in results
            if row["base_case"] == base_case and (rate_filter is None or row["rate_name"] == rate_filter)
         ],
         key=lambda row: float(row["mode_number"]),
      )
      if rows:
         ax.plot(
            [float(row["mode_number"]) for row in rows],
            [float(row["measured_rate"]) for row in rows],
            color=color,
            linestyle=":",
            marker=marker,
            markersize=7.0 if is_jeans else 5.2,
            markeredgecolor="black",
            markeredgewidth=0.75 if is_jeans else 0.45,
            linewidth=1.8 if is_jeans else 1.4,
            label="_nolegend_",
            zorder=10 if is_jeans else 5,
         )
         if label.startswith("Jeans") and "rho=10" in label:
            for row in rows:
               ax.annotate(
                  f"m={int(float(row['mode_number']))}",
                  xy=(float(row["mode_number"]), float(row["measured_rate"])),
                  xytext=(4, 5),
                  textcoords="offset points",
                  fontsize=7,
                  color=color,
                  zorder=11,
               )
   ax.axvline(dense_cutoff, color="0.25", linestyle="--", linewidth=1.2, alpha=0.8)
   ax.annotate(
      f"rho=10, G=1 cutoff mJ={dense_cutoff:.2f}",
      xy=(dense_cutoff, 0.0),
      xytext=(6, 10),
      textcoords="offset points",
      rotation=90,
      fontsize=7,
      color="0.25",
      va="bottom",
   )
   ax.set_title("Dispersion branches: solid exact, dotted+markers num")
   ax.set_xlabel("mode number")
   ax.set_ylabel("rate")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend(fontsize=6.4, framealpha=0.85)

   profile_cases = [
      ("hd_entropy", axs[0, 1]),
      ("mhd_fast_plus", axs[0, 2]),
      ("mhd_slow_plus", axs[1, 0]),
      ("jeans_dense_rho10", axs[1, 1]),
   ]
   for case_name, ax in profile_cases:
      row = next(item for item in results if item["case"] == case_name)
      profile = row["profile"]
      x = profile["x"]
      ax.plot(x, profile["primary"], color="tab:blue", marker="o", markersize=2.1, linewidth=1.0, label="num")
      ax.plot(x, profile["exact"], color="black", linestyle=":", linewidth=2.0, label="exact")
      ax.plot(x, profile["initial"], color="0.45", linestyle="--", linewidth=1.4, label="initial")
      ax.set_title(f"{row['label']}: rate rel err={row['rate_rel_error']:.2e}")
      ax.set_xlabel("x")
      ax.set_ylabel(profile["ylabel"])
      ax.grid(True, which="major", alpha=0.25)
      ax.legend(fontsize=8)

   ax = axs[1, 2]
   summary_rows = []
   for base_case in dict.fromkeys(str(row["base_case"]) for row in results):
      rows = [row for row in results if row["base_case"] == base_case]
      first = rows[0]
      summary_rows.append({
         "label": first["label"],
         "rate_abs_error": max(float(row["rate_abs_error"]) for row in rows),
         "field_error": max(float(row["field_error"]) for row in rows),
      })
   labels = [str(row["label"]) for row in summary_rows]
   xloc = np.arange(len(labels))
   rate_error = np.asarray([max(float(row["rate_abs_error"]), 1.0e-16) for row in summary_rows])
   field_error = np.asarray([max(float(row["field_error"]), 1.0e-16) for row in summary_rows])
   ax.semilogy(xloc, rate_error, marker="o", linewidth=1.8, label="rate abs error")
   ax.semilogy(xloc, field_error, marker="s", linewidth=1.8, label="field L2 error")
   ax.set_title("All branches")
   ax.set_xticks(xloc)
   ax.set_xticklabels(labels, rotation=35, ha="right")
   ax.grid(True, which="major", axis="y", alpha=0.25)
   ax.legend(fontsize=8)

   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)


def write_summary(rows: list[dict[str, object]]) -> None:
   write_tsv(
      SUMMARY,
      rows,
      [
         "case",
         "base_case",
         "status",
         "reason",
         "returncode",
         "seconds",
         "theory_rate",
         "measured_rate",
         "rate_abs_error",
         "rate_rel_error",
         "field_error",
         "mode_number",
         "mass_rel_drift",
         "energy_rel_drift",
         "gravity_total_energy_rel_drift",
         "avg_mcells_per_s",
         "avg_sim_time_per_s",
         "run_dir",
      ],
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--cfl", type=float, default=0.9)
   parser.add_argument("--field-error-tol", type=float, default=2.0e-2)
   parser.add_argument("--rate-abs-error-tol", type=float, default=2.0e-2)
   parser.add_argument("--rate-rel-error-tol", type=float, default=2.0e-2)
   parser.add_argument("--rate-modes", type=int, nargs="+", default=[1, 2, 3, 4])
   parser.add_argument("--skip-plots", action="store_true")
   args = parser.parse_args()

   rows = []
   status = 0
   cases = build_cases(args)
   for case in cases:
      row, case_status = validate_case(case, args)
      rows.append(row)
      status = max(status, case_status)
      print(f"{row['case']}: {row['status']} {row['reason']}", flush=True)

   write_summary(rows)
   append_validation_history(HISTORY, rows)

   if status == 0 and not args.skip_plots:
      results = [collect_result(case, RUN_ROOT / case.name) for case in cases]
      plot_path = next_output_path(PLOT_ROOT, "dispersion_relations")
      plot_results(results, plot_path, args)
      print(f"wrote {plot_path.relative_to(ROOT)}")
   print(f"wrote {SUMMARY.relative_to(ROOT)}")
   return status


if __name__ == "__main__":
   raise SystemExit(main())
