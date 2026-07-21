#!/usr/bin/env python3
import argparse
import csv
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, literally

sys.path.insert(0, str(ROOT))

from pyfluid.exact_solutions import circular_alfven_wave_exact, resistive_mhd_wave_exact, sod_conserved_integrals, sod_exact_solution
from pyfluid_numba import (
   BC_OUTFLOW,
   BC_PERIODIC,
   BC_REFLECTIVE,
   CT_FLUX_TOTH,
   CT_OFF,
   DEFAULT_CFL,
   DEFAULT_THREADS,
   DIFFUSION_EXPLICIT,
   DIFFUSION_IMPLICIT,
   DIFFUSION_NONE,
   AcousticWaveConfig,
   CircularAlfvenWaveConfig,
   EntropyWaveConfig,
   HydroCharacteristicWaveConfig,
   HydroRunConfig,
   HYDRO_MODE_ENTROPY,
   HYDRO_MODE_SHEAR,
   HYDRO_MODE_SHEAR_Z,
   HYDRO_MODE_SOUND_MINUS,
   HYDRO_MODE_SOUND_PLUS,
   MHDCharacteristicWaveConfig,
   MHD_MODE_ALFVEN_MINUS,
   MHD_MODE_ALFVEN_PLUS,
   MHD_MODE_ENTROPY,
   MHD_MODE_FAST_PLUS,
   MHD_MODE_FAST_MINUS,
   MHD_MODE_SLOW_MINUS,
   MHD_MODE_SLOW_PLUS,
   MHDRunConfig,
   ResistiveWaveConfig,
   LIMITER_MC,
   RECON_PWC,
   RECON_PLM,
   RECON_WENOZ,
   RECON_WENOZ_CHAR,
   RECON_VARS_CONSERVED,
   RECON_VARS_PRIMITIVE,
   RIEMANN_HLL,
   RIEMANN_HLLD,
   RIEMANN_HLLE,
   RIEMANN_HLLC,
   RIEMANN_LLF,
   TIME_CORRECTOR,
   TIME_EULER,
   TIME_MIDPOINT,
   TIME_RK4,
   SUPPORTED_HYDRO_INITIAL_CONDITIONS,
   TIME_SSPRK3,
   bc_tuple_name,
   ct_method_name,
   diffusion_method_name,
   hydro_initial_condition_name,
   hydro_mode_name,
   limiter_name,
   mhd_mode_name,
   recon_name,
   recon_vars_name,
   riemann_name,
   run_acoustic_wave_case,
   run_circular_alfven_wave_case,
   run_entropy_wave_case,
   run_hydro_characteristic_wave_case,
   run_hydro_case,
   run_mhd_characteristic_wave_case,
   run_mhd_case,
   timestep_name,
   validate_circular_alfven_options,
   validate_hydro_characteristic_mode,
)
from pyfluid.runtime import configure_numba_threads
from pyfluid_numba.boundaries import apply_boundaries
from pyfluid_numba.diagnostics import conserved_totals, normalized_divb, scalar_diagnostics as numba_scalar_diagnostics
from pyfluid_numba.diffusion import implicit_diffusion_update
from pyfluid_numba.eigensystem import (
   hydro_characteristic_eigenvalues,
   hydro_primitive_left_eigenvectors,
   hydro_primitive_right_eigenvectors,
   mhd_primitive_reduced_eigenvalues,
   mhd_primitive_reduced_jacobian,
   mhd_primitive_reduced_left_eigenvectors,
   mhd_primitive_reduced_right_eigenvectors,
   mhd_mode_norm,
)
from pyfluid_numba.hydro2d import evolve_circular_alfven_wave_case, evolve_resistive_wave_case, make_workspace, throughput_metrics
from pyfluid_numba.initial_data import (
   circular_alfven_wave_initial_state,
   hydro_characteristic_wave_initial_state,
   mhd_characteristic_wave_initial_state,
   mhd_ic_initial_state,
   resistive_wave_initial_state,
)
from pyfluid_numba.options import (
   AXIS_X,
   ENERGY,
   BX,
   BY,
   BZ,
   MOMX,
   MOMY,
   MOMZ,
   NVAR_HYDRO,
   NVAR_MHD,
   RHO,
   require_reached_final_time,
   resolve_diffusion_coefficients,
   SIDE_LOWER,
   SIDE_UPPER,
   validate_bc,
   validate_common_numeric_options,
)
from pyfluid_numba.physics import pressure_state_mhd
from pyfluid_numba.reconstruction import (
   mhd_characteristic_to_primitive,
   mhd_characteristics_at_offset,
   reconstruct_hydro_characteristic_cached,
   reconstruct_primitive_cached,
)
from pyfluid_numba.run_configs import (
   HYDRO_DIRECT_RECONSTRUCTION_SCAN,
   HYDRO_METHOD_SCAN,
   HYDRO_SPATIAL_RECONSTRUCTION_SCAN,
   MHD_METHOD_SCAN,
)
from pyfluid_numba.timestepping import evolve_steps_with_diagnostics, evolve_to_time


OUTDIR = ROOT / "plots" / "current"
RUNOUTDIR = ROOT / "runs" / "current"
SCANDIR = OUTDIR / "scans"


def require_close_to_roundoff(name, value, tolerance):
   if abs(value) > tolerance:
      raise AssertionError(f"{name}={value:.3e} exceeds {tolerance:.3e}")


def has_numeric_value(value):
   return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def require_optional_close_to_roundoff(name, value, tolerance):
   if has_numeric_value(value):
      require_close_to_roundoff(name, float(value), tolerance)


def require_finite_row(name, row):
   for key, value in row.items():
      if isinstance(value, (int, float)) and not np.isfinite(value):
         raise AssertionError(f"{name} produced non-finite {key}={value}")


def data_output_path(outpath):
   outpath = pathlib.Path(outpath)
   try:
      relative = outpath.relative_to(OUTDIR)
   except ValueError:
      relative = outpath.name
   return RUNOUTDIR / relative


def write_csv(rows, outpath):
   data_path = data_output_path(outpath)
   data_path.parent.mkdir(parents=True, exist_ok=True)
   with data_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
      writer.writeheader()
      writer.writerows(rows)


def next_mobile_plot_path(outpath):
   outpath = pathlib.Path(outpath)
   for version in range(1, 10000):
      candidate = outpath.with_name(f"{outpath.stem}_mobile_v{version:03d}{outpath.suffix}")
      if not candidate.exists():
         return candidate
   raise RuntimeError(f"too many mobile plot versions for {outpath}")


def save_plot(fig, outpath, dpi, **kwargs):
   outpath = pathlib.Path(outpath)
   outpath.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(outpath, dpi=dpi, **kwargs)
   mobile_path = next_mobile_plot_path(outpath)
   fig.savefig(mobile_path, dpi=dpi, **kwargs)
   return mobile_path


def reference_power_law(dx, error, order):
   coarse = int(np.argmax(dx))
   return error[coarse] * (dx / dx[coarse]) ** order


def write_plot_index():
   text = """# Current PyFluid Plots

Images live here. Tabular run outputs live under `runs/current/` with matching relative names.
Plot writers also save numbered `_mobile_vNNN` copies to avoid stale image caching in the mobile app.

## Scans

- `scans/hydro_method_convergence.png`: hydro method scans; smooth wave for time integration, exact Sod shock for Riemann/reconstruction.
- `scans/hydro_direct_reconstruction_convergence.png`: direct smooth primitive interface reconstruction scan, isolated from time integration.
- `scans/hydro_spatial_reconstruction_convergence.png`: pure hydro smooth 1D characteristic wave spatial reconstruction scan.
- `scans/mhd_method_convergence.png`: MHD method scans; smooth fast wave for time integration, Balsara.1A shock for Riemann/reconstruction.
- `scans/mhd_characteristic_convergence.png`: all seven smooth MHD characteristic modes against analytic wave propagation.

## Per-IC Runs

- `IC/circular_alfven/circular_alfven_validation.png`: nonlinear circular Alfven exact-wave convergence and final overlay.
- `IC/circular_alfven/circular_alfven_matrix.png`: compact nonlinear MHD method matrix for circular Alfven wave.
- `IC/field_loop/field_loop_advection.png`: field-loop advection CT stress test.

Other ordinary validation plots should go under `IC/<ic_name>/`.
"""
   OUTDIR.mkdir(parents=True, exist_ok=True)
   (OUTDIR / "README.md").write_text(text)


def plot_ct_divb_history(histories, outpath, title):
   fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
   for label, rows in histories:
      time_values = np.array([row["time"] for row in rows], dtype=float)
      divb = log_floor([row["divb"] for row in rows])
      ax.semilogy(time_values, divb, linewidth=2.4, label=label, **marker_kwargs(time_values))
   ax.set_xlabel("time")
   ax.set_ylabel("normalized divB")
   ax.set_title(title)
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.legend(fontsize=10)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def relative_drift(value, initial):
   return abs(value - initial) / max(abs(initial), 1.0)


def log_floor(values, floor=1.0e-18):
   values = np.asarray(values, dtype=float)
   return np.maximum(values, floor)


def config_label(config):
   physics = "magnetized" if isinstance(config, MHDRunConfig) else "non-magnetized"
   diffusion = f"diffusion {diffusion_method_name(getattr(config, 'diffusion_method', DIFFUSION_NONE))}"
   ct = ct_method_name(getattr(config, "ct_method", CT_OFF))
   return (
      f"IC={hydro_initial_condition_name(config.ic)} ({config.ic}), {physics}, {diffusion}, {ct}\n"
      f"{recon_name(config.recon).upper()} {recon_vars_name(config.recon_vars)}/"
      f"{limiter_name(config.limiter).upper()}/{riemann_name(config.riemann).upper()}/"
      f"{timestep_name(config.timestep).upper()}, BC={bc_tuple_name(config.bc)}, {config.nx}x{config.ny}, CFL={config.cfl:g}"
   )


def marker_kwargs(values):
   return {"marker": "o", "markersize": 4.0} if len(values) <= 40 else {}


def momentum_drift(scalars, scalars0):
   return float(
      np.sqrt(
         (scalars["momx"] - scalars0["momx"]) ** 2
         + (scalars["momy"] - scalars0["momy"]) ** 2
         + (scalars["momz"] - scalars0["momz"]) ** 2
      )
   )


def state_scalar_diagnostics(Uvec, nx, ny, ng, dx, dy, gamma):
   return numba_scalar_diagnostics(Uvec, nx, ny, ng, dx, dy, gamma)


def append_scalar_diagnostics_row(
   rows,
   step,
   time_value,
   dt,
   elapsed,
   cell_updates,
   scalars,
   scalars0,
   diffusion_error=0.0,
   diffusion_steps=0,
):
   mcells_per_second = 0.0
   if elapsed > 0.0:
      mcells_per_second = cell_updates / elapsed / 1.0e6
   rows.append(
      {
         "step": step,
         "time": time_value,
         "dt": dt,
         "elapsed": elapsed,
         "cell_updates": cell_updates,
         "million_cell_updates_per_second": mcells_per_second,
         "diffusion_error": diffusion_error,
         "diffusion_steps": diffusion_steps,
         "mass": scalars["mass"],
         "energy": scalars["energy"],
         "momx": scalars["momx"],
         "momy": scalars["momy"],
         "momz": scalars["momz"],
         "momentum_norm": scalars["momentum_norm"],
         "internal_energy": scalars["internal_energy"],
         "kinetic_energy": scalars["kinetic_energy"],
         "magnetic_energy": scalars["magnetic_energy"],
         "min_rho": scalars["min_rho"],
         "min_pressure": scalars["min_pressure"],
         "mass_rel_drift": relative_drift(scalars["mass"], scalars0["mass"]),
         "energy_rel_drift": relative_drift(scalars["energy"], scalars0["energy"]),
         "momentum_drift": momentum_drift(scalars, scalars0),
         "divb": scalars["divb"],
      }
   )


def collect_mhd_diagnostics(config):
   validate_common_numeric_options(config, mhd=True)
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   viscosity, resistivity = resolve_diffusion_coefficients(config, dx)

   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   evolve_steps_with_diagnostics(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 1,
      config.ct_method, config.diffusion_method, viscosity, resistivity, config.diffusion_cfl,
      config.diffusion_tolerance, config.diffusion_max_steps, config.diffusion_sor_omega
   )

   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   scalars0 = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
   rows = []
   time_value = 0.0
   append_scalar_diagnostics_row(
      rows,
      0,
      time_value,
      0.0,
      0.0,
      0.0,
      scalars0,
      scalars0,
      0.0,
      0,
   )

   for step in range(1, config.steps + 1):
      start = time.time()
      dt, _, _, diffusion_error, _, diffusion_steps = evolve_steps_with_diagnostics(
         Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
         bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 1,
         config.ct_method, config.diffusion_method, viscosity, resistivity, config.diffusion_cfl,
         config.diffusion_tolerance, config.diffusion_max_steps, config.diffusion_sor_omega
      )
      elapsed = time.time() - start
      time_value += dt
      scalars = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
      append_scalar_diagnostics_row(
         rows,
         step,
         time_value,
         dt,
         elapsed,
         float(config.nx * config.ny),
         scalars,
         scalars0,
         diffusion_error,
         diffusion_steps,
      )
   return rows, Uvec


def collect_circular_alfven_diagnostics(config, nsamples=32):
   validate_circular_alfven_options(config)
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny

   warm = circular_alfven_wave_initial_state(config)
   warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt = make_workspace(
      warm, config.nx, config.ny
   )
   evolve_to_time(
      warm, warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt,
      dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
      config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
      min(config.final_time, 2.0 * dx), 16, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )

   Uvec = circular_alfven_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   scalars0 = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
   rows = []
   append_scalar_diagnostics_row(rows, 0, 0.0, 0.0, 0.0, 0.0, scalars0, scalars0)

   time_value = 0.0
   total_steps = 0
   for target_time in np.linspace(0.0, config.final_time, nsamples + 1)[1:]:
      delta = float(target_time - time_value)
      if delta <= 1.0e-14:
         continue
      start = time.time()
      reached, steps, _, _ = evolve_to_time(
         Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT,
         dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
         config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
         delta, 20000, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
      )
      elapsed = time.time() - start
      time_value += reached
      total_steps += steps
      scalars = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
      append_scalar_diagnostics_row(
         rows,
         total_steps,
         time_value,
         reached / max(steps, 1),
         elapsed,
         float(config.nx * config.ny * steps),
         scalars,
         scalars0,
      )
   return rows, Uvec


def collect_mhd_to_time_diagnostics(config, final_time, nsamples=32):
   validate_common_numeric_options(config, mhd=True)
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny

   warm = mhd_ic_initial_state(config)
   warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt = make_workspace(
      warm, config.nx, config.ny
   )
   evolve_to_time(
      warm, warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt,
      dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
      config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
      min(final_time, 2.0 * dx), 16, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )

   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   scalars0 = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
   rows = []
   append_scalar_diagnostics_row(rows, 0, 0.0, 0.0, 0.0, 0.0, scalars0, scalars0)

   time_value = 0.0
   total_steps = 0
   for target_time in np.linspace(0.0, final_time, nsamples + 1)[1:]:
      delta = float(target_time - time_value)
      if delta <= 1.0e-14:
         continue
      start = time.time()
      reached, steps, _, _ = evolve_to_time(
         Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT,
         dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
         config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
         delta, 20000, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
      )
      elapsed = time.time() - start
      time_value += reached
      total_steps += steps
      scalars = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
      append_scalar_diagnostics_row(
         rows,
         total_steps,
         time_value,
         reached / max(steps, 1),
         elapsed,
         float(config.nx * config.ny * steps),
         scalars,
         scalars0,
      )
   require_reached_final_time(time_value, final_time, total_steps, 20000 * nsamples)
   return rows, Uvec


def primitive_field_arrays(Uvec, nx, ny, ng, gamma):
   interior = (slice(ng, ng + nx), slice(ng, ng + ny))
   rho = np.asarray(Uvec[RHO][interior])
   vx = np.asarray(Uvec[MOMX][interior]) / rho
   vy = np.asarray(Uvec[MOMY][interior]) / rho
   vz = np.asarray(Uvec[MOMZ][interior]) / rho
   kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
   magnetic = 0.0
   fields = {
      "rho": rho,
      "vx": vx,
      "vy": vy,
   }
   if Uvec.shape[0] > NVAR_HYDRO:
      bx = np.asarray(Uvec[BX][interior])
      by = np.asarray(Uvec[BY][interior])
      bz = np.asarray(Uvec[BZ][interior])
      magnetic = 0.5 * (bx * bx + by * by + bz * bz)
      fields.update({"Bx": bx, "By": by, "Bz": bz})
   fields["P"] = (gamma - 1.0) * (np.asarray(Uvec[ENERGY][interior]) - kinetic - magnetic)
   return fields


def magnetic_strength(Uvec, nx, ny, ng):
   interior = (slice(ng, ng + nx), slice(ng, ng + ny))
   bx = np.asarray(Uvec[BX][interior])
   by = np.asarray(Uvec[BY][interior])
   bz = np.asarray(Uvec[BZ][interior])
   return np.sqrt(bx * bx + by * by + bz * bz)


def plot_final_heatmap(Uvec, nx, ny, ng, gamma, outpath, title):
   fields = primitive_field_arrays(Uvec, nx, ny, ng, gamma)
   if Uvec.shape[0] > NVAR_HYDRO:
      specs = (("rho", fields["rho"]), ("P", fields["P"]), ("Bx", fields["Bx"]), ("By", fields["By"]))
   else:
      specs = (("rho", fields["rho"]), ("P", fields["P"]), ("vx", fields["vx"]), ("vy", fields["vy"]))
   fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.8), constrained_layout=True)
   for ax, (label, data) in zip(axs.ravel(), specs):
      im = ax.imshow(data.T, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), aspect="auto")
      fig.colorbar(im, ax=ax, shrink=0.88)
      ax.set_title(label)
      ax.set_xlabel("x")
      ax.set_ylabel("y")
   fig.suptitle(title)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def plot_field_loop_advection(initial_state, final_states, histories, config, final_time, outpath):
   fig, axs = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
   initial_b = magnetic_strength(initial_state, config.nx, config.ny, config.ng)
   final_flux = magnetic_strength(final_states["flux-Toth"], config.nx, config.ny, config.ng)
   final_no_ct = magnetic_strength(final_states["no CT"], config.nx, config.ny, config.ng)

   for ax, (label, data) in zip(
      axs[0],
      (("initial |B|", initial_b), ("flux-Toth final |B|", final_flux), ("no CT final |B|", final_no_ct)),
   ):
      vmax = max(float(np.max(data)), 1.0e-30)
      im = ax.imshow(data.T, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), vmin=0.0, vmax=vmax, cmap="magma")
      fig.colorbar(im, ax=ax, shrink=0.82)
      ax.set_title(f"{label}, max={vmax:.2e}")
      ax.set_xlabel("x")
      ax.set_ylabel("y")

   for label, rows in histories.items():
      time_values = np.array([row["time"] for row in rows], dtype=float)
      markers = marker_kwargs(time_values)
      axs[1, 0].semilogy(time_values, log_floor([row["divb"] for row in rows]), linewidth=2.2, label=label, **markers)
      magnetic0 = max(abs(rows[0]["magnetic_energy"]), 1.0e-300)
      axs[1, 1].semilogy(
         time_values,
         [row["magnetic_energy"] / magnetic0 for row in rows],
         linewidth=2.2,
         label=label,
         **markers,
      )
      axs[1, 2].semilogy(
         time_values,
         log_floor([row["energy_rel_drift"] for row in rows]),
         linewidth=2.2,
         label=label,
         **markers,
      )

   axs[1, 0].set_title("Magnetic constraint")
   axs[1, 0].set_ylabel("normalized divB")
   axs[1, 1].set_title("Magnetic energy")
   axs[1, 1].set_ylabel("E_mag / E_mag(0)")
   axs[1, 2].set_title("Total energy drift")
   axs[1, 2].set_ylabel("relative drift")
   for ax in axs[1]:
      ax.set_xlabel("time")
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
      ax.legend(fontsize=9)

   fig.suptitle(
      f"Field-loop advection: WENO-Z char/HLLD/RK4, BC=periodic, CFL={config.cfl:g}, "
      f"{config.nx}x{config.ny}, t={final_time:g}",
      fontsize=12,
   )
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def plot_scalar_diagnostics(rows, outpath, title, mhd=False):
   time_values = np.array([row["time"] for row in rows], dtype=float)
   step_values = np.array([row["step"] for row in rows], dtype=int)
   active = step_values > 0
   fig, axs = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)
   axs = axs.ravel()
   markers = marker_kwargs(time_values)
   active_markers = marker_kwargs(time_values[active])

   axs[0].semilogy(time_values, log_floor([row["mass_rel_drift"] for row in rows]), label="mass", linewidth=2.2, **markers)
   axs[0].semilogy(time_values, log_floor([row["energy_rel_drift"] for row in rows]), label="energy", linewidth=2.2, **markers)
   axs[0].set_title("Conserved Scalars")
   axs[0].set_ylabel("relative drift")
   axs[0].legend(fontsize=9)

   axs[1].semilogy(time_values, log_floor([row["momentum_drift"] for row in rows]), label="|dP|", linewidth=2.2, **markers)
   axs[1].set_title("Total Momentum Drift")
   axs[1].set_ylabel("absolute drift")

   axs[2].plot(time_values, [row["min_rho"] for row in rows], label="min rho", linewidth=2.2, **markers)
   axs[2].plot(time_values, [row["min_pressure"] for row in rows], label="min P", linewidth=2.2, **markers)
   axs[2].set_title("Primitive Floors")
   axs[2].legend(fontsize=9)

   axs[3].plot(time_values, [row["internal_energy"] for row in rows], label="internal", linewidth=2.0, **markers)
   axs[3].plot(time_values, [row["kinetic_energy"] for row in rows], label="kinetic", linewidth=2.0, **markers)
   if mhd:
      axs[3].plot(time_values, [row["magnetic_energy"] for row in rows], label="magnetic", linewidth=2.0, **markers)
   axs[3].plot(time_values, [row["energy"] for row in rows], label="total", color="black", linewidth=1.4, **markers)
   axs[3].set_title("Energy Components")
   axs[3].legend(fontsize=8)

   axs[4].plot(time_values[active], [row["dt"] for row in rows if row["step"] > 0], linewidth=2.2, **active_markers)
   axs[4].set_title("Timestep")
   axs[4].set_ylabel("dt")
   axs[4].ticklabel_format(axis="y", style="plain", useOffset=False)

   if mhd:
      axs[5].semilogy(time_values, log_floor([row["divb"] for row in rows]), linewidth=2.2, label="divB", **markers)
      axs[5].set_title("Magnetic Constraint")
      axs[5].set_ylabel("normalized divB")
   else:
      axs[5].axis("off")

   for ax in axs:
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
      ax.set_xlabel("time")
      ax.tick_params(labelsize=10)
   fig.suptitle(title)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def plot_diffusion_diagnostics(rows, outpath, title):
   active_rows = [row for row in rows if row["step"] > 0]
   time_values = np.array([row["time"] for row in active_rows], dtype=float)
   fig, ax_error = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
   error_line = ax_error.semilogy(
      time_values,
      log_floor([row["diffusion_error"] for row in active_rows]),
      color="#2878b5",
      linewidth=2.4,
      marker="o",
      markersize=4.0,
      label="diffusion error",
   )
   ax_steps = ax_error.twinx()
   step_line = ax_steps.plot(
      time_values,
      [row["diffusion_steps"] for row in active_rows],
      color="#d66b2a",
      linewidth=2.1,
      marker="s",
      markersize=4.0,
      label="implicit steps",
   )
   ax_error.set_title(title)
   ax_error.set_xlabel("time")
   ax_error.set_ylabel("operator residual")
   ax_steps.set_ylabel("implicit steps")
   ax_error.grid(True, which="major", alpha=0.25)
   ax_error.grid(False, which="minor")
   lines = error_line + step_line
   ax_error.legend(lines, [line.get_label() for line in lines], loc="best")
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def resistive_wave_profiles(Uvec, config):
   interior = (slice(config.ng, config.ng + config.nx), slice(config.ng, config.ng + config.ny))
   x = (np.arange(config.nx) + 0.5) / config.nx
   by_num = np.mean(np.asarray(Uvec[BY][interior]), axis=1)
   bz_num = np.mean(np.asarray(Uvec[BZ][interior]), axis=1)
   rho = np.asarray(Uvec[RHO][interior])
   vx = np.asarray(Uvec[MOMX][interior]) / rho
   vy = np.asarray(Uvec[MOMY][interior]) / rho
   vz = np.asarray(Uvec[MOMZ][interior]) / rho
   kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
   magnetic = 0.5 * (
      np.asarray(Uvec[BX][interior]) ** 2
      + np.asarray(Uvec[BY][interior]) ** 2
      + np.asarray(Uvec[BZ][interior]) ** 2
   )
   pressure_num = np.mean((config.gamma - 1.0) * (np.asarray(Uvec[ENERGY][interior]) - kinetic - magnetic), axis=1)
   by_exact, bz_exact, pressure_exact = resistive_mhd_wave_exact(
      x,
      config.final_time,
      config.gamma,
      config.pressure0,
      config.amplitude,
      config.resistivity,
      config.wave_number,
   )
   return x, by_num, bz_num, pressure_num, by_exact, bz_exact, pressure_exact


def resistive_wave_mode_amplitude(Uvec, config):
   interior = (slice(config.ng, config.ng + config.nx), slice(config.ng, config.ng + config.ny))
   x = (np.arange(config.nx) + 0.5) / config.nx
   k = 2.0 * np.pi * config.wave_number
   sin_kx = np.sin(k * x)
   cos_kx = np.cos(k * x)
   by = np.mean(np.asarray(Uvec[BY][interior]), axis=1)
   bz = np.mean(np.asarray(Uvec[BZ][interior]), axis=1)
   by_amp = np.hypot(2.0 * np.mean(by * sin_kx), 2.0 * np.mean(by * cos_kx))
   bz_amp = np.hypot(2.0 * np.mean(bz * sin_kx), 2.0 * np.mean(bz * cos_kx))
   return float(0.5 * (by_amp + bz_amp))


def exact_resistive_wave_amplitude(config, time_value):
   k = 2.0 * np.pi * config.wave_number
   return float(config.amplitude * np.exp(-config.resistivity * k * k * time_value))


def resistive_wave_label(config):
   k = 2.0 * np.pi * config.wave_number
   decay_rate = config.resistivity * k * k
   return f"eta_m={config.resistivity:g} (magnetic diffusivity), n={config.wave_number}, k=2*pi*n={k:.4g}, eta_m*k^2={decay_rate:.4g}"


def resistive_curve_label(config):
   return f"eta_m = {config.resistivity:g}"


def stamp_figure(fig):
   fig.text(0.995, 0.005, f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ha="right", va="bottom", fontsize=8)


def collect_resistive_wave_history(config):
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   dt = config.final_time / config.steps

   warm = resistive_wave_initial_state(config)
   warm_tmp = warm.copy()
   warm_rhs = np.zeros_like(warm)
   implicit_diffusion_update(
      warm,
      warm_tmp,
      warm_rhs,
      dx,
      dy,
      dt,
      config.ng,
      bc[0],
      bc[1],
      bc[2],
      bc[3],
      0.0,
      config.resistivity,
      config.diffusion_tolerance,
      config.diffusion_max_steps,
      config.diffusion_sor_omega,
   )

   Uvec = resistive_wave_initial_state(config)
   Utmp = Uvec.copy()
   rhs = np.zeros_like(Uvec)
   history = []

   def append_history(step, time_value, diffusion_error, diffusion_steps):
      amplitude = resistive_wave_mode_amplitude(Uvec, config)
      exact = exact_resistive_wave_amplitude(config, time_value)
      history.append(
         {
            "step": step,
            "time": time_value,
            "amplitude": amplitude,
            "exact_amplitude": exact,
            "relative_error": abs(amplitude - exact) / max(abs(exact), 1.0e-300),
            "diffusion_error": diffusion_error,
            "diffusion_steps": diffusion_steps,
         }
      )

   append_history(0, 0.0, 0.0, 0)
   diffusion_error = 0.0
   diffusion_steps_max = 0
   diffusion_steps_total = 0
   start = time.time()
   for step in range(1, config.steps + 1):
      diffusion_error, diffusion_steps = implicit_diffusion_update(
         Uvec,
         Utmp,
         rhs,
         dx,
         dy,
         dt,
         config.ng,
         bc[0],
         bc[1],
         bc[2],
         bc[3],
         0.0,
         config.resistivity,
         config.diffusion_tolerance,
         config.diffusion_max_steps,
         config.diffusion_sor_omega,
      )
      diffusion_steps_total += diffusion_steps
      diffusion_steps_max = max(diffusion_steps_max, diffusion_steps)
      append_history(step, step * dt, diffusion_error, diffusion_steps)
   elapsed = time.time() - start

   row = {
      "backend": "numba_mhd2d_resistive_wave",
      "nx": config.nx,
      "ny": config.ny,
      "nvar": Uvec.shape[0],
      "dtype": str(Uvec.dtype),
      "steps": config.steps,
      "threads": config.threads,
      "bc": str(bc),
      "time": config.final_time,
      "dt_min": dt,
      "dt_max": dt,
      "elapsed": elapsed,
      "million_cell_updates_per_second": config.nx * config.ny * config.steps / elapsed / 1.0e6,
      "resistivity": config.resistivity,
      "diffusion_sor_omega": config.diffusion_sor_omega,
      "diffusion_error": diffusion_error,
      "diffusion_steps_max": diffusion_steps_max,
      "diffusion_steps_total": diffusion_steps_total,
   }
   return row, Uvec, history


def plot_resistive_mhd_exact(config, row, Uvec, outpath):
   x, by_num, bz_num, pressure_num, by_exact, bz_exact, pressure_exact = resistive_wave_profiles(Uvec, config)
   fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
   fields = (
      ("By", by_num, by_exact),
      ("Bz", bz_num, bz_exact),
      ("P - P0", pressure_num - config.pressure0, pressure_exact - config.pressure0),
   )
   for ax, (label, numerical, exact) in zip(axs, fields):
      ax.plot(
         x,
         numerical,
         color="#2878b5",
         linewidth=2.4,
         marker="o",
         markersize=3.8,
         markevery=4,
         label="num.",
         zorder=2,
      )
      ax.plot(x, exact, color="black", linestyle=":", linewidth=2.2, label=resistive_curve_label(config), zorder=3)
      ax.set_ylabel(label)
      ax.grid(True, which="major", alpha=0.25)
      ax.tick_params(labelsize=10)
   fig.suptitle(
      "Resistive MHD sinusoid: pure implicit diffusion\n"
      f"{resistive_wave_label(config)}\n"
      f"t={config.final_time:g}, {config.nx}x{config.ny}, dt={row['dt_min']:.3e}, max sweeps={row['diffusion_steps_max']}"
   )
   axs[-1].set_xlabel("x")
   axs[0].legend(loc="best", fontsize=11)
   stamp_figure(fig)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def plot_resistive_mhd_amplitude(config, row, history, outpath):
   time_values = np.array([item["time"] for item in history], dtype=float)
   amplitude = np.array([item["amplitude"] for item in history], dtype=float)
   exact = np.array([item["exact_amplitude"] for item in history], dtype=float)
   rel_error = np.array([item["relative_error"] for item in history], dtype=float)
   fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True)
   markers = marker_kwargs(time_values)
   axs[0].plot(time_values, amplitude, color="#2878b5", linewidth=2.4, label="num.", **markers)
   axs[0].plot(time_values, exact, color="black", linestyle=":", linewidth=2.3, label=resistive_curve_label(config))
   axs[0].set_ylabel("wave amplitude")
   axs[0].legend(loc="best", fontsize=11)
   axs[0].grid(True, which="major", alpha=0.25)

   axs[1].semilogy(time_values, log_floor(rel_error), color="#d66b2a", linewidth=2.2, **markers)
   axs[1].set_xlabel("time")
   axs[1].set_ylabel("relative error")
   axs[1].grid(True, which="major", alpha=0.25)
   axs[1].grid(False, which="minor")
   fig.suptitle(
      "Resistive MHD amplitude decay\n"
      f"{resistive_wave_label(config)}\n"
      f"{config.nx}x{config.ny}, dt={row['dt_min']:.3e}, max sweeps={row['diffusion_steps_max']}"
   )
   stamp_figure(fig)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def explicit_diffusion_step_budget(config, diffusion_cfl):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   eta_m = config.resistivity
   if eta_m <= 0.0:
      raise ValueError("explicit magnetic diffusion benchmark requires eta_m > 0")
   dt_limit = diffusion_cfl / (2.0 * eta_m * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
   steps = max(1, int(np.ceil(config.final_time / dt_limit)))
   return steps, config.final_time / steps, dt_limit


def resistive_diffusion_benchmark_row(method, row, Uvec, config, explicit_steps):
   amplitude = resistive_wave_mode_amplitude(Uvec, config)
   exact = exact_resistive_wave_amplitude(config, config.final_time)
   dt = row["dt"] if "dt" in row else row["dt_min"]
   steps = row["steps"]
   output = {
      "method": method,
      "nx": config.nx,
      "ny": config.ny,
      "threads": row.get("numba_threads", row.get("threads", config.threads)),
      "final_time": config.final_time,
      "eta_m": config.resistivity,
      "wave_number": config.wave_number,
      "steps": steps,
      "dt": dt,
      "explicit_dt_limit": row.get("explicit_dt_limit", ""),
      "elapsed": row["elapsed"],
      "million_cell_updates_per_second": row["million_cell_updates_per_second"],
      "amplitude": amplitude,
      "exact_amplitude": exact,
      "amplitude_relative_error": abs(amplitude - exact) / max(abs(exact), 1.0e-300),
      "timestep_gain_vs_explicit": explicit_steps / steps,
      "diffusion_error": row.get("diffusion_error", 0.0),
      "diffusion_steps_max": row.get("diffusion_steps_max", 0),
      "diffusion_steps_total": row.get("diffusion_steps_total", 0),
      "diffusion_sor_omega": row.get("diffusion_sor_omega", ""),
   }
   return output


def plot_resistive_diffusion_timestep_benchmark(rows, outpath):
   labels = [row["method"] for row in rows]
   x = np.arange(len(rows))
   elapsed = np.array([row["elapsed"] for row in rows], dtype=float)
   steps = np.array([row["steps"] for row in rows], dtype=float)
   fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)

   axs[0].bar(x, elapsed, color=("#2878b5", "#d66b2a"))
   axs[0].set_xticks(x)
   axs[0].set_xticklabels(labels)
   axs[0].set_ylabel("wall time [s]")
   axs[0].grid(True, axis="y", alpha=0.25)
   axs[0].set_ylim(0.0, float(np.max(elapsed)) * 1.25)
   for xpos, value in zip(x, elapsed):
      axs[0].text(xpos, value * 1.03, f"{value:.3f}s", ha="center", va="bottom", fontsize=10)

   axs[1].bar(x, steps, color=("#2878b5", "#d66b2a"))
   axs[1].set_xticks(x)
   axs[1].set_xticklabels(labels)
   axs[1].set_ylabel("physical timesteps")
   axs[1].grid(True, axis="y", alpha=0.25)
   axs[1].set_ylim(0.0, float(np.max(steps)) * 1.3)
   for xpos, row in zip(x, rows):
      text = f"{int(row['steps'])} steps\ndt={row['dt']:.2e}"
      if row["diffusion_steps_total"]:
         text += f"\n{int(row['diffusion_steps_total'])} SOR sweeps"
      axs[1].text(xpos, row["steps"] * 1.03, text, ha="center", va="bottom", fontsize=9)

   ref = rows[0]
   implicit_rows = [row for row in rows if row["diffusion_steps_total"]]
   solver_text = ""
   if implicit_rows:
      solver_text = f", SOR omega={implicit_rows[0]['diffusion_sor_omega']:g}"
   fig.suptitle(
      "Resistive MHD magnetic diffusion: explicit vs implicit\n"
      f"sinusoidal B, magnetic diffusivity eta_m={ref['eta_m']:g}, mode n={ref['wave_number']}, "
      f"t={ref['final_time']:g}, {int(ref['nx'])}x{int(ref['ny'])}{solver_text}"
   )
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def observed_order(dx, error):
   dx = np.asarray(dx, dtype=float)
   error = np.asarray(error, dtype=float)
   mask = np.isfinite(dx) & np.isfinite(error) & (dx > 0.0) & (error > 0.0)
   if np.count_nonzero(mask) < 2:
      return np.nan
   return float(np.polyfit(np.log(dx[mask]), np.log(error[mask]), 1)[0])


def add_observed_orders(rows):
   for scan in sorted({row["scan"] for row in rows}):
      for method in sorted({row["method"] for row in rows if row["scan"] == scan}):
         group = [row for row in rows if row["scan"] == scan and row["method"] == method]
         order = observed_order([row["x"] for row in group], [row["error_l2"] for row in group])
         for row in group:
            row["observed_order"] = order
   return rows


@njit
def smooth_reconstruction_profile(x, amplitude):
   phase = 2.0 * np.pi * x
   rho = 1.0 + amplitude * (0.45 * np.sin(phase) + 0.10 * np.cos(2.0 * phase))
   vx = 0.20 + amplitude * (0.20 * np.cos(phase) + 0.05 * np.sin(2.0 * phase))
   vy = -0.10 + amplitude * (0.15 * np.sin(phase + 0.3))
   vz = 0.05 + amplitude * (0.12 * np.cos(phase - 0.4))
   pressure = 1.0 + amplitude * (0.35 * np.sin(phase - 0.2) + 0.08 * np.cos(2.0 * phase + 0.1))
   return rho, vx, vy, vz, pressure


def smooth_reconstruction_profile_py(x, amplitude):
   phase = 2.0 * np.pi * x
   return (
      1.0 + amplitude * (0.45 * np.sin(phase) + 0.10 * np.cos(2.0 * phase)),
      0.20 + amplitude * (0.20 * np.cos(phase) + 0.05 * np.sin(2.0 * phase)),
      -0.10 + amplitude * (0.15 * np.sin(phase + 0.3)),
      0.05 + amplitude * (0.12 * np.cos(phase - 0.4)),
      1.0 + amplitude * (0.35 * np.sin(phase - 0.2) + 0.08 * np.cos(2.0 * phase + 0.1)),
   )


def average_sin_mode(left, right, frequency, phase):
   wave_number = 2.0 * np.pi * frequency
   return (np.cos(wave_number * left + phase) - np.cos(wave_number * right + phase)) / (wave_number * (right - left))


def average_cos_mode(left, right, frequency, phase):
   wave_number = 2.0 * np.pi * frequency
   return (np.sin(wave_number * right + phase) - np.sin(wave_number * left + phase)) / (wave_number * (right - left))


def smooth_reconstruction_profile_cell_average_py(cell, nx, amplitude):
   left = cell / nx
   right = (cell + 1) / nx
   return (
      1.0 + amplitude * (0.45 * average_sin_mode(left, right, 1.0, 0.0) + 0.10 * average_cos_mode(left, right, 2.0, 0.0)),
      0.20 + amplitude * (0.20 * average_cos_mode(left, right, 1.0, 0.0) + 0.05 * average_sin_mode(left, right, 2.0, 0.0)),
      -0.10 + amplitude * (0.15 * average_sin_mode(left, right, 1.0, 0.3)),
      0.05 + amplitude * (0.12 * average_cos_mode(left, right, 1.0, -0.4)),
      1.0
      + amplitude
      * (0.35 * average_sin_mode(left, right, 1.0, -0.2) + 0.08 * average_cos_mode(left, right, 2.0, 0.1)),
   )


def smooth_reconstruction_primitive_cache(nx, ng, amplitude):
   ny = 1
   Vcache = np.zeros((NVAR_HYDRO, nx + 2 * ng, ny + 2 * ng), dtype=np.float64)
   for i in range(nx + 2 * ng):
      values = smooth_reconstruction_profile_cell_average_py(i - ng, nx, amplitude)
      for j in range(ny + 2 * ng):
         for q, value in enumerate(values):
            Vcache[q, i, j] = value
   return Vcache


@njit
def direct_hydro_reconstruction_error(Vcache, nx, ng, gamma, amplitude, recon, limiter):
   recon = literally(recon)
   limiter = literally(limiter)
   error2 = 0.0
   count = 0
   j = ng
   for ii in range(nx):
      i = ng - 1 + ii
      x_face = ii / nx
      erho, evx, evy, evz, ep = smooth_reconstruction_profile(x_face, amplitude)
      if recon == RECON_WENOZ_CHAR:
         (
            lrho,
            lvx,
            lvy,
            lvz,
            lp,
            rrho,
            rvx,
            rvy,
            rvz,
            rp,
         ) = reconstruct_hydro_characteristic_cached(Vcache, i, j, gamma, AXIS_X)
      else:
         lrho = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         lvx = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         lvy = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         lvz = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         lp = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         rrho = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         rvx = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         rvy = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         rvz = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         rp = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_X, SIDE_UPPER)

      error2 += (lrho - erho) * (lrho - erho)
      error2 += (lvx - evx) * (lvx - evx)
      error2 += (lvy - evy) * (lvy - evy)
      error2 += (lvz - evz) * (lvz - evz)
      error2 += (lp - ep) * (lp - ep)
      error2 += (rrho - erho) * (rrho - erho)
      error2 += (rvx - evx) * (rvx - evx)
      error2 += (rvy - evy) * (rvy - evy)
      error2 += (rvz - evz) * (rvz - evz)
      error2 += (rp - ep) * (rp - ep)
      count += 10
   return np.sqrt(error2 / count) / amplitude


def primitive_state_l2_against_reference(Uvec, Uref, config):
   ng = config.ng
   sx = slice(ng, ng + config.nx)
   sy = slice(ng, ng + config.ny)
   rho = Uvec[RHO, sx, sy]
   rho_ref = Uref[RHO, sx, sy]
   vx = Uvec[MOMX, sx, sy] / rho
   vy = Uvec[MOMY, sx, sy] / rho
   vz = Uvec[MOMZ, sx, sy] / rho
   vx_ref = Uref[MOMX, sx, sy] / rho_ref
   vy_ref = Uref[MOMY, sx, sy] / rho_ref
   vz_ref = Uref[MOMZ, sx, sy] / rho_ref
   pressure = (config.gamma - 1.0) * (
      Uvec[ENERGY, sx, sy] - 0.5 * (Uvec[MOMX, sx, sy] ** 2 + Uvec[MOMY, sx, sy] ** 2 + Uvec[MOMZ, sx, sy] ** 2) / rho
   )
   pressure_ref = (config.gamma - 1.0) * (
      Uref[ENERGY, sx, sy] - 0.5 * (Uref[MOMX, sx, sy] ** 2 + Uref[MOMY, sx, sy] ** 2 + Uref[MOMZ, sx, sy] ** 2) / rho_ref
   )
   normal = np.array((config.kx, config.ky), dtype=float)
   normal /= np.linalg.norm(normal)
   mode_vector = hydro_primitive_right_eigenvectors(
      config.rho0, config.pressure0, config.gamma, normal[0], normal[1]
   )[:, config.mode]
   mode_norm = np.linalg.norm(mode_vector)
   error2 = (
      (rho - rho_ref) ** 2
      + (vx - vx_ref) ** 2
      + (vy - vy_ref) ** 2
      + (vz - vz_ref) ** 2
      + (pressure - pressure_ref) ** 2
   )
   return float(np.sqrt(np.mean(error2)) / (config.amplitude * mode_norm))


def mhd_primitive_state_l2_against_reference(Uvec, Uref, config):
   ng = config.ng
   sx = slice(ng, ng + config.nx)
   sy = slice(ng, ng + config.ny)
   rho = Uvec[RHO, sx, sy]
   rho_ref = Uref[RHO, sx, sy]
   vx = Uvec[MOMX, sx, sy] / rho
   vy = Uvec[MOMY, sx, sy] / rho
   vz = Uvec[MOMZ, sx, sy] / rho
   vx_ref = Uref[MOMX, sx, sy] / rho_ref
   vy_ref = Uref[MOMY, sx, sy] / rho_ref
   vz_ref = Uref[MOMZ, sx, sy] / rho_ref
   bx = Uvec[BX, sx, sy]
   by = Uvec[BY, sx, sy]
   bz = Uvec[BZ, sx, sy]
   bx_ref = Uref[BX, sx, sy]
   by_ref = Uref[BY, sx, sy]
   bz_ref = Uref[BZ, sx, sy]
   pressure = (config.gamma - 1.0) * (
      Uvec[ENERGY, sx, sy]
      - 0.5 * (Uvec[MOMX, sx, sy] ** 2 + Uvec[MOMY, sx, sy] ** 2 + Uvec[MOMZ, sx, sy] ** 2) / rho
      - 0.5 * (bx * bx + by * by + bz * bz)
   )
   pressure_ref = (config.gamma - 1.0) * (
      Uref[ENERGY, sx, sy]
      - 0.5 * (Uref[MOMX, sx, sy] ** 2 + Uref[MOMY, sx, sy] ** 2 + Uref[MOMZ, sx, sy] ** 2) / rho_ref
      - 0.5 * (bx_ref * bx_ref + by_ref * by_ref + bz_ref * bz_ref)
   )
   mode_norm = mhd_mode_norm(
      config.rho0, config.pressure0, config.bx0, config.by0, config.bz0, config.gamma, config.kx, config.ky, config.mode
   )
   error2 = (
      (rho - rho_ref) ** 2
      + (vx - vx_ref) ** 2
      + (vy - vy_ref) ** 2
      + (vz - vz_ref) ** 2
      + (pressure - pressure_ref) ** 2
      + (bx - bx_ref) ** 2
      + (by - by_ref) ** 2
      + (bz - bz_ref) ** 2
   )
   return float(np.sqrt(np.mean(error2)) / (config.amplitude * mode_norm))


def evolve_characteristic_state(config):
   if not isinstance(config, HydroCharacteristicWaveConfig):
      config = HydroCharacteristicWaveConfig(**config)
   validate_common_numeric_options(config, mhd=False)
   validate_hydro_characteristic_mode(config.mode)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = 200000
   Uvec = hydro_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.final_time, max_steps,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, config.final_time, steps, max_steps)
   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   row = {
      "nx": config.nx,
      "ny": config.ny,
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "dt_avg": time_value / steps,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
   }
   return Uvec, row


def evolve_mhd_characteristic_state(config):
   if not isinstance(config, MHDCharacteristicWaveConfig):
      config = MHDCharacteristicWaveConfig(**config)
   validate_common_numeric_options(config, mhd=True)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = 200000
   Uvec = mhd_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb0 = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)
   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.final_time, max_steps,
      config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, config.final_time, steps, max_steps)
   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)
   row = {
      "nx": config.nx,
      "ny": config.ny,
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "dt_avg": time_value / steps,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      "divb_initial": divb0,
      "divb_final": divb,
      "divb_delta": divb - divb0,
   }
   return Uvec, row


def characteristic_scan_row(scan, method, test_name, reference, x_label, x_value, config, row, error_l2):
   return {
      "scan": scan,
      "method": method,
      "test": test_name,
      "reference": reference,
      "x_label": x_label,
      "x": x_value,
      "nx": row["nx"],
      "ny": row["ny"],
      "dx": 1.0 / row["nx"],
      "dt": row.get("dt_avg", row.get("time", 0.0) / row["steps"]),
      "cfl": config.cfl,
      "final_time": config.final_time,
      "steps": row["steps"],
      "mode": hydro_mode_name(config.mode),
      "amplitude": config.amplitude,
      "recon": recon_name(config.recon),
      "recon_vars": recon_vars_name(config.recon_vars),
      "riemann": riemann_name(config.riemann),
      "timestep": timestep_name(config.timestep),
      "error_metric": "L2",
      "error_l2": error_l2,
      "slope_label": "p",
      "rho_l2": row.get("rho_l2", ""),
      "pressure_l2": row.get("pressure_l2", ""),
      "velocity_l2": row.get("velocity_l2", ""),
      "mass_rel_drift": row["mass_rel_drift"],
      "energy_rel_drift": row["energy_rel_drift"],
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "observed_order": np.nan,
   }


def mhd_characteristic_scan_row(scan, method, test_name, reference, x_label, x_value, config, row, error_l2):
   return {
      "scan": scan,
      "method": method,
      "test": test_name,
      "reference": reference,
      "x_label": x_label,
      "x": x_value,
      "nx": row["nx"],
      "ny": row["ny"],
      "dx": 1.0 / row["nx"],
      "dt": row.get("dt_avg", row.get("time", 0.0) / row["steps"]),
      "cfl": config.cfl,
      "final_time": config.final_time,
      "steps": row["steps"],
      "mode": mhd_mode_name(config.mode),
      "amplitude": config.amplitude,
      "recon": recon_name(config.recon),
      "recon_vars": recon_vars_name(config.recon_vars),
      "riemann": riemann_name(config.riemann),
      "timestep": timestep_name(config.timestep),
      "ct_method": config.ct_method,
      "ct_method_name": ct_method_name(config.ct_method),
      "error_metric": "L2",
      "error_l2": error_l2,
      "slope_label": "p",
      "rho_l2": row.get("rho_l2", ""),
      "pressure_l2": row.get("pressure_l2", ""),
      "velocity_l2": row.get("velocity_l2", ""),
      "magnetic_l2": row.get("magnetic_l2", ""),
      "mass_rel_drift": row["mass_rel_drift"],
      "energy_rel_drift": row["energy_rel_drift"],
      "divb_initial": row.get("divb_initial", ""),
      "divb_final": row.get("divb_final", ""),
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "observed_order": np.nan,
   }


def analytic_characteristic_row(scan, method, config):
   row = run_hydro_characteristic_wave_case(config)
   return characteristic_scan_row(
      scan,
      method,
      "sound+ characteristic",
      "analytic characteristic wave",
      "dx",
      1.0 / row["nx"],
      config,
      row,
      row["mode_l2"],
   )


def analytic_mhd_characteristic_row(scan, method, config):
   row = run_mhd_characteristic_wave_case(config)
   return mhd_characteristic_scan_row(
      scan,
      method,
      "fast+ MHD characteristic",
      "analytic MHD characteristic wave",
      "dx",
      1.0 / row["nx"],
      config,
      row,
      row["mode_l2"],
   )


def hydro_discontinuity_scan_row(scan, method, config, row, error, rho_error, pressure_error, velocity_error, reference):
   return {
      "scan": scan,
      "method": method,
      "test": "Sod shock tube",
      "reference": reference,
      "x_label": "dx",
      "x": 1.0 / row["nx"],
      "nx": row["nx"],
      "ny": row["ny"],
      "dx": 1.0 / row["nx"],
      "dt": row["time"] / row["steps"],
      "cfl": config["cfl"],
      "final_time": row["time"],
      "steps": row["steps"],
      "mode": "",
      "amplitude": "",
      "recon": recon_name(config["recon"]),
      "recon_vars": recon_vars_name(config["recon_vars"]),
      "riemann": riemann_name(config["riemann"]),
      "timestep": timestep_name(config["timestep"]),
      "error_metric": "mean abs primitive profile error",
      "error_l2": error,
      "slope_label": "s",
      "rho_l2": rho_error,
      "pressure_l2": pressure_error,
      "velocity_l2": velocity_error,
      "mass_rel_drift": "",
      "energy_rel_drift": "",
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "observed_order": np.nan,
   }


def mhd_discontinuity_scan_row(scan, method, config, row, error, component_errors, reference):
   return {
      "scan": scan,
      "method": method,
      "test": "Balsara.1A MHD discontinuity",
      "reference": reference,
      "x_label": "dx",
      "x": 1.0 / row["nx"],
      "nx": row["nx"],
      "ny": row["ny"],
      "dx": 1.0 / row["nx"],
      "dt": row["time"] / row["steps"],
      "cfl": config.cfl,
      "final_time": row["time"],
      "steps": row["steps"],
      "mode": "",
      "amplitude": "",
      "recon": recon_name(config.recon),
      "recon_vars": recon_vars_name(config.recon_vars),
      "riemann": riemann_name(config.riemann),
      "timestep": timestep_name(config.timestep),
      "ct_method": config.ct_method,
      "ct_method_name": ct_method_name(config.ct_method),
      "error_metric": "mean primitive profile error vs high-res HLLD",
      "error_l2": error,
      "slope_label": "s",
      "rho_l2": component_errors["rho"],
      "pressure_l2": component_errors["P"],
      "velocity_l2": max(component_errors["vx"], component_errors["vy"]),
      "magnetic_l2": max(component_errors["By"], component_errors["Bz"]),
      "mass_rel_drift": row["mass_rel_drift"],
      "energy_rel_drift": row["energy_rel_drift"],
      "divb_initial": row["divb_initial"],
      "divb_final": row["divb_final"],
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "observed_order": np.nan,
   }


def draw_scan(ax, rows, scan, title, reference_specs):
   scan_rows = [row for row in rows if row["scan"] == scan]
   metric = scan_rows[0].get("error_metric", "L2")
   methods = []
   for row in scan_rows:
      if row["method"] not in methods:
         methods.append(row["method"])

   method_data = {}
   for method in methods:
      group = sorted([row for row in scan_rows if row["method"] == method], key=lambda row: row["x"], reverse=True)
      x = np.array([row["x"] for row in group], dtype=float)
      error = np.array([row["error_l2"] for row in group], dtype=float)
      order = group[0]["observed_order"]
      slope_label = group[0].get("slope_label", "p")
      method_data[method] = (x, error)
      label = f"{method} {slope_label}={order:.2f}" if np.isfinite(order) else method
      ax.loglog(x, error, marker="o", linewidth=2.4, markersize=7.5, label=label)

   for method, order, label in reference_specs:
      if method not in method_data:
         continue
      x, error = method_data[method]
      reference = reference_power_law(x, error, order)
      ax.loglog(x, reference, color="black", linestyle="--", linewidth=2.0, alpha=0.75)
      text_index = len(x) // 2
      ax.annotate(
         label,
         xy=(x[text_index], reference[text_index]),
         xytext=(5, 0),
         textcoords="offset points",
         fontsize=10,
         color="black",
         va="center",
         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
      )

   ax.set_xlabel(scan_rows[0]["x_label"])
   ax.set_ylabel(f"{metric} error" if "error" not in metric.lower() else metric)
   ax.set_title(title)
   ax.invert_xaxis()
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.tick_params(labelsize=10)
   ax.legend(fontsize=10)


def sod_method_error_row(scan_config, scan, method, nx, recon, riemann):
   gamma = scan_config["gamma"]
   ny = scan_config["ny"]
   ng = scan_config["ng"]
   left = scan_config["left_state"]
   right = scan_config["right_state"]
   result = run_sod_method(
      nx,
      ny,
      ng,
      gamma,
      scan_config["final_time"],
      scan_config["cfl"],
      left,
      right,
      recon,
      riemann=riemann,
      timestep=scan_config["timestep"],
      recon_vars=scan_config["recon_vars"],
      limiter=scan_config["limiter"],
   )
   x = (np.arange(nx) + 0.5) / nx
   rho_exact, pressure_exact, velocity_exact = sod_exact_solution(x, result["time"], gamma, left, right)
   rho_error = float(np.mean(np.abs(result["rho"] - rho_exact)))
   pressure_error = float(np.mean(np.abs(result["pressure"] - pressure_exact)))
   velocity_error = float(np.mean(np.abs(result["velocity"] - velocity_exact)))
   error = (rho_error + pressure_error + velocity_error) / 3.0
   return hydro_discontinuity_scan_row(
      scan,
      method,
      {
         "cfl": scan_config["cfl"],
         "recon": recon,
         "riemann": riemann,
         "timestep": scan_config["timestep"],
         "recon_vars": scan_config["recon_vars"],
      },
      result,
      error,
      rho_error,
      pressure_error,
      velocity_error,
      "analytic Newtonian hydro Riemann solution",
   )


def evolve_mhd_discontinuity_profile(config, final_time):
   validate_common_numeric_options(config, mhd=True)
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb0 = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)
   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT,
      dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3],
      config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
      final_time, 20000, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, final_time, steps, 20000)
   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)
   profile = mhd_profile_1d(Uvec, config)
   row = {
      "nx": config.nx,
      "ny": config.ny,
      "time": time_value,
      "steps": steps,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": relative_drift(totals["mass"], totals0["mass"]),
      "energy_rel_drift": relative_drift(totals["energy"], totals0["energy"]),
      "divb_initial": divb0,
      "divb_final": divb,
   }
   return profile, row


def mhd_profile_reference_error(profile, reference_profile, x, x_ref):
   fields = ("rho", "P", "vx", "vy", "By", "Bz")
   errors = {}
   for field in fields:
      reference_values = np.interp(x, x_ref, reference_profile[field])
      errors[field] = float(np.mean(np.abs(profile[field] - reference_values)))
   return sum(errors.values()) / len(errors), errors


def make_mhd_discontinuity_config(scan_config, nx, riemann, recon):
   return MHDRunConfig(
      nx=nx,
      ny=scan_config["ny"],
      steps=1,
      threads=DEFAULT_THREADS,
      ng=scan_config["ng"],
      gamma=scan_config["gamma"],
      cfl=scan_config["cfl"],
      bc=scan_config["bc"],
      recon=recon,
      recon_vars=scan_config["recon_vars"],
      limiter=scan_config["limiter"],
      riemann=riemann,
      timestep=scan_config["timestep"],
      ic=scan_config["ic"],
      ct_method=scan_config["ct_method"],
      diffusion_method=DIFFUSION_NONE,
   )


def method_scan_rows():
   time_scan = HYDRO_METHOD_SCAN["time"]
   discontinuity_scan = HYDRO_METHOD_SCAN["discontinuity"]
   rows = []

   reference_config = HydroCharacteristicWaveConfig(
      nx=time_scan["nx"],
      ny=time_scan["ny"],
      ng=time_scan["ng"],
      final_time=time_scan["final_time"],
      mode=HYDRO_MODE_SOUND_PLUS,
      amplitude=time_scan["amplitude"],
      cfl=time_scan["reference_cfl"],
      recon=time_scan["recon"],
      recon_vars=time_scan["recon_vars"],
      limiter=time_scan["limiter"],
      riemann=time_scan["riemann"],
      timestep=time_scan["reference_timestep"],
   )
   reference_state, reference_row = evolve_characteristic_state(reference_config)
   for method, timestep in time_scan["methods"]:
      for cfl in time_scan["cfl_values"]:
         config = HydroCharacteristicWaveConfig(
            nx=time_scan["nx"],
            ny=time_scan["ny"],
            ng=time_scan["ng"],
            final_time=time_scan["final_time"],
            mode=HYDRO_MODE_SOUND_PLUS,
            amplitude=time_scan["amplitude"],
            cfl=cfl,
            recon=time_scan["recon"],
            recon_vars=time_scan["recon_vars"],
            limiter=time_scan["limiter"],
            riemann=time_scan["riemann"],
            timestep=timestep,
         )
         state, row = evolve_characteristic_state(config)
         error = primitive_state_l2_against_reference(state, reference_state, config)
         rows.append(
            characteristic_scan_row(
               "time_integrator",
               method,
               "sound+ characteristic",
               f"same-grid RK4 reference, CFL={reference_config.cfl:g}, steps={reference_row['steps']}",
               "dt",
               row["dt_avg"],
               config,
               row,
               error,
            )
         )

   for method, recon in discontinuity_scan["reconstruction_methods"]:
      for n in discontinuity_scan["nx_values"]:
         rows.append(
            sod_method_error_row(
               discontinuity_scan,
               "reconstruction",
               method,
               n,
               recon,
               discontinuity_scan["reconstruction_riemann"],
            )
         )

   for method, riemann in discontinuity_scan["riemann_methods"]:
      for n in discontinuity_scan["nx_values"]:
         rows.append(
            sod_method_error_row(
               discontinuity_scan,
               "riemann_solver",
               method,
               n,
               discontinuity_scan["riemann_recon"],
               riemann,
            )
         )

   return add_observed_orders(rows)


def add_local_orders(rows):
   for scan in sorted({row["scan"] for row in rows}):
      for method in sorted({row["method"] for row in rows if row["scan"] == scan}):
         group = sorted(
            [row for row in rows if row["scan"] == scan and row["method"] == method],
            key=lambda row: row["x"],
            reverse=True,
         )
         for i, row in enumerate(group):
            if i + 1 >= len(group):
               row["local_order_to_next"] = np.nan
               continue
            next_row = group[i + 1]
            row["local_order_to_next"] = np.log(row["error_l2"] / next_row["error_l2"]) / np.log(row["x"] / next_row["x"])
   return rows


def hydro_spatial_reconstruction_rows():
   scan = HYDRO_SPATIAL_RECONSTRUCTION_SCAN
   rows = []
   for method, recon in scan["methods"]:
      for n in scan["nx_values"]:
         config = HydroCharacteristicWaveConfig(
            nx=n,
            ny=scan["ny"],
            ng=scan["ng"],
            final_time=scan["final_time"],
            mode=HYDRO_MODE_SOUND_PLUS,
            amplitude=scan["amplitude"],
            cfl=scan["cfl"],
            recon=recon,
            recon_vars=scan["recon_vars"],
            limiter=scan["limiter"],
            riemann=scan["riemann"],
            timestep=scan["timestep"],
            vx0=scan["vx0"],
            vy0=scan["vy0"],
            kx=scan["kx"],
            ky=scan["ky"],
         )
         row = run_hydro_characteristic_wave_case(config)
         require_finite_row(f"hydro_spatial_reconstruction_{method}_{n}", row)
         rows.append(
            characteristic_scan_row(
               "hydro_spatial_reconstruction",
               method,
               scan["name"],
               "analytic linear characteristic wave",
               "dx",
               1.0 / row["nx"],
               config,
               row,
               row["mode_l2"],
            )
         )
   return add_local_orders(add_observed_orders(rows))


def hydro_direct_reconstruction_rows():
   scan = HYDRO_DIRECT_RECONSTRUCTION_SCAN
   rows = []
   for method, recon, expected_order in scan["methods"]:
      for n in scan["nx_values"]:
         Vcache = smooth_reconstruction_primitive_cache(n, scan["ng"], scan["amplitude"])
         start = time.time()
         error = direct_hydro_reconstruction_error(
            Vcache,
            n,
            scan["ng"],
            scan["gamma"],
            scan["amplitude"],
            recon,
            scan["limiter"],
         )
         elapsed = time.time() - start
         rows.append(
            {
               "scan": "hydro_direct_reconstruction",
               "method": method,
               "test": scan["name"],
               "reference": "analytic primitive value at cell interface",
               "x_label": "dx",
               "x": 1.0 / n,
               "nx": n,
               "ny": 1,
               "dx": 1.0 / n,
               "gamma": scan["gamma"],
               "amplitude": scan["amplitude"],
               "recon": recon_name(recon),
               "limiter": limiter_name(scan["limiter"]),
               "expected_order": expected_order,
               "error_metric": "primitive interface L2",
               "error_l2": error,
               "elapsed": elapsed,
               "observed_order": np.nan,
               "local_order_to_next": np.nan,
            }
         )
   return add_local_orders(add_observed_orders(rows))


def mhd_method_scan_rows():
   time_scan = MHD_METHOD_SCAN["time"]
   discontinuity_scan = MHD_METHOD_SCAN["discontinuity"]
   rows = []

   reference_config = MHDCharacteristicWaveConfig(
      nx=time_scan["nx"],
      ny=time_scan["ny"],
      ng=time_scan["ng"],
      final_time=time_scan["final_time"],
      mode=MHD_MODE_FAST_PLUS,
      amplitude=time_scan["amplitude"],
      cfl=time_scan["reference_cfl"],
      recon=time_scan["recon"],
      recon_vars=time_scan["recon_vars"],
      limiter=time_scan["limiter"],
      riemann=time_scan["riemann"],
      timestep=time_scan["reference_timestep"],
      ct_method=time_scan["ct_method"],
   )
   reference_state, reference_row = evolve_mhd_characteristic_state(reference_config)
   for method, timestep in time_scan["methods"]:
      for cfl in time_scan["cfl_values"]:
         config = MHDCharacteristicWaveConfig(
            nx=time_scan["nx"],
            ny=time_scan["ny"],
            ng=time_scan["ng"],
            final_time=time_scan["final_time"],
            mode=MHD_MODE_FAST_PLUS,
            amplitude=time_scan["amplitude"],
            cfl=cfl,
            recon=time_scan["recon"],
            recon_vars=time_scan["recon_vars"],
            limiter=time_scan["limiter"],
            riemann=time_scan["riemann"],
            timestep=timestep,
            ct_method=time_scan["ct_method"],
         )
         state, row = evolve_mhd_characteristic_state(config)
         error = mhd_primitive_state_l2_against_reference(state, reference_state, config)
         rows.append(
            mhd_characteristic_scan_row(
               "time_integrator",
               method,
               "fast+ MHD characteristic",
               f"same-grid RK4 reference, CFL={reference_config.cfl:g}, steps={reference_row['steps']}",
               "dt",
               row["dt_avg"],
               config,
               row,
               error,
            )
         )

   final_time = discontinuity_scan["final_time"]
   reference_n = discontinuity_scan["reference_nx"]
   reference_config = make_mhd_discontinuity_config(
      discontinuity_scan,
      reference_n,
      riemann=discontinuity_scan["reference_riemann"],
      recon=discontinuity_scan["reference_recon"],
   )
   reference_profile, _ = evolve_mhd_discontinuity_profile(reference_config, final_time)
   x_ref = (np.arange(reference_n) + 0.5) / reference_n

   for method, riemann in discontinuity_scan["riemann_methods"]:
      for n in discontinuity_scan["nx_values"]:
         config = make_mhd_discontinuity_config(discontinuity_scan, n, riemann=riemann, recon=discontinuity_scan["riemann_recon"])
         profile, row = evolve_mhd_discontinuity_profile(config, final_time)
         x = (np.arange(n) + 0.5) / n
         error, component_errors = mhd_profile_reference_error(profile, reference_profile, x, x_ref)
         rows.append(
            mhd_discontinuity_scan_row(
               "riemann_solver",
               method,
               config,
               row,
               error,
               component_errors,
               f"{reference_n}x{discontinuity_scan['ny']} HLLD/WENO-Z char reference",
            )
         )

   for method, recon in discontinuity_scan["reconstruction_methods"]:
      for n in discontinuity_scan["nx_values"]:
         config = make_mhd_discontinuity_config(
            discontinuity_scan,
            n,
            riemann=discontinuity_scan["reconstruction_riemann"],
            recon=recon,
         )
         profile, row = evolve_mhd_discontinuity_profile(config, final_time)
         x = (np.arange(n) + 0.5) / n
         error, component_errors = mhd_profile_reference_error(profile, reference_profile, x, x_ref)
         rows.append(
            mhd_discontinuity_scan_row(
               "reconstruction",
               method,
               config,
               row,
               error,
               component_errors,
               f"{reference_n}x{discontinuity_scan['ny']} HLLD/WENO-Z char reference",
            )
         )

   return add_observed_orders(rows)


def short_recon_label(recon):
   return recon_name(recon).upper()


def plot_method_scans(rows):
   time_scan = HYDRO_METHOD_SCAN["time"]
   discontinuity_scan = HYDRO_METHOD_SCAN["discontinuity"]
   fig, axs = plt.subplots(1, 3, figsize=(18.5, 6.0), constrained_layout=True)
   draw_scan(
      axs[0],
      rows,
      "time_integrator",
      f"Time integrator\nsmooth sound+ wave, {time_scan['nx']}x{time_scan['ny']}\n"
      f"{short_recon_label(time_scan['recon'])}/{riemann_name(time_scan['riemann']).upper()}, "
      f"{timestep_name(time_scan['reference_timestep']).upper()} ref",
      (("Euler", 1, "dt"), ("midpoint", 2, "dt^2"), ("SSPRK3", 3, "dt^3"), ("RK4", 4, "dt^4")),
   )
   draw_scan(
      axs[1],
      rows,
      "riemann_solver",
      f"Riemann solver\n{discontinuity_scan['name']}\n"
      f"{short_recon_label(discontinuity_scan['riemann_recon'])}/"
      f"{timestep_name(discontinuity_scan['timestep']).upper()}, CFL={discontinuity_scan['cfl']:g}",
      (("HLLC", 1, "dx"),),
   )
   draw_scan(
      axs[2],
      rows,
      "reconstruction",
      f"Reconstruction\n{discontinuity_scan['name']}\n"
      f"{riemann_name(discontinuity_scan['reconstruction_riemann']).upper()}/"
      f"{timestep_name(discontinuity_scan['timestep']).upper()}, CFL={discontinuity_scan['cfl']:g}",
      (("WENO-Z char", 1, "dx"),),
   )
   fig.suptitle(
      f"Numba hydro method scans: smooth wave for time, {discontinuity_scan['name']} for Riemann/reconstruction, "
      f"threads={DEFAULT_THREADS}, float64",
      fontsize=12,
   )
   SCANDIR.mkdir(parents=True, exist_ok=True)
   save_plot(fig, SCANDIR / "hydro_method_convergence.png", dpi=170)
   plt.close(fig)
   write_plot_index()


def plot_hydro_direct_reconstruction(rows):
   scan = HYDRO_DIRECT_RECONSTRUCTION_SCAN
   fig, ax = plt.subplots(figsize=(9.8, 6.4), constrained_layout=True)
   method_data = {}
   styles = {
      "PWC": {"marker": "o", "linestyle": "-"},
      "PWL MC": {"marker": "o", "linestyle": "-"},
      "WENO-Z": {"marker": "o", "linestyle": "-"},
      "WENO-Z char": {"marker": "s", "linestyle": ":"},
   }
   for method, _, expected_order in scan["methods"]:
      group = sorted([row for row in rows if row["method"] == method], key=lambda row: row["x"], reverse=True)
      x = np.array([row["x"] for row in group], dtype=float)
      error = np.array([row["error_l2"] for row in group], dtype=float)
      method_data[method] = (x, error, expected_order)
      order = group[0]["observed_order"]
      ax.loglog(x, error, linewidth=2.4, markersize=7.0, label=f"{method} p={order:.2f}", **styles[method])

   reference_methods = (
      ("PWC", 1.0, "dx"),
      ("PWL MC", 2.0, "dx^2"),
      ("WENO-Z", 5.0, "dx^5"),
   )
   for method, expected_order, label in reference_methods:
      x, error, _ = method_data[method]
      reference = reference_power_law(x, error, expected_order)
      ax.loglog(x, reference, color="black", linestyle="--", linewidth=1.8, alpha=0.65)
      text_index = min(1, len(x) - 1)
      ax.annotate(
         label,
         xy=(x[text_index], reference[text_index]),
         xytext=(5, 0),
         textcoords="offset points",
         fontsize=10,
         va="center",
         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
      )

   roundoff_guide = np.finfo(np.float64).eps / scan["amplitude"]
   ax.axhline(roundoff_guide, color="0.35", linestyle=":", linewidth=1.8)
   ax.text(
      0.97,
      0.06,
      "eps64 / amp guide",
      transform=ax.transAxes,
      fontsize=9.5,
      color="0.25",
      ha="right",
      va="center",
      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
   )

   x_weno, error_weno, order_weno = method_data["WENO-Z"]
   coarse = int(np.argmax(x_weno))
   coefficient = error_weno[coarse] / (x_weno[coarse] ** order_weno)
   x_cross = (roundoff_guide / coefficient) ** (1.0 / order_weno)
   if np.min(x_weno) / 4.0 <= x_cross <= np.max(x_weno):
      ax.axvline(x_cross, color="0.35", linestyle=":", linewidth=1.8)
      ax.annotate(
         "WENO hits eps guide",
         xy=(x_cross, roundoff_guide),
         xytext=(5, 14),
         textcoords="offset points",
         fontsize=9.5,
         color="0.25",
         va="bottom",
         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
      )

   local_lines = []
   for method in ("PWL MC", "WENO-Z", "WENO-Z char"):
      group = sorted([row for row in rows if row["method"] == method], key=lambda row: row["x"], reverse=True)
      local = [row["local_order_to_next"] for row in group[:-1] if np.isfinite(row["local_order_to_next"])]
      local_lines.append(f"{method} local p: " + ", ".join(f"{value:.2f}" for value in local))
   ax.text(
      0.03,
      0.04,
      "\n".join(local_lines),
      transform=ax.transAxes,
      fontsize=9.5,
      bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86, "pad": 4.0},
   )

   ax.set_xlabel("dx")
   ax.set_ylabel("primitive interface L2 error")
   ax.set_title(
      f"Hydro direct reconstruction scan: {scan['name']}\n"
      f"primitive variables, MC limiter, amp={scan['amplitude']:g}, BC=periodic, float64"
   )
   ax.invert_xaxis()
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.legend(fontsize=10)
   SCANDIR.mkdir(parents=True, exist_ok=True)
   save_plot(fig, SCANDIR / "hydro_direct_reconstruction_convergence.png", dpi=170)
   plt.close(fig)
   write_plot_index()


def plot_hydro_spatial_reconstruction(rows):
   scan = HYDRO_SPATIAL_RECONSTRUCTION_SCAN
   fig, ax = plt.subplots(figsize=(9.5, 6.4), constrained_layout=True)
   method_data = {}
   line_styles = {
      "PWC": {"marker": "o", "linestyle": "-"},
      "PWL MC": {"marker": "o", "linestyle": "-"},
      "WENO-Z": {"marker": "o", "linestyle": "-"},
      "WENO-Z char": {"marker": "s", "linestyle": ":"},
   }
   for method, _ in scan["methods"]:
      group = sorted(
         [row for row in rows if row["method"] == method],
         key=lambda row: row["x"],
         reverse=True,
      )
      x = np.array([row["x"] for row in group], dtype=float)
      error = np.array([row["error_l2"] for row in group], dtype=float)
      method_data[method] = (x, error)
      order = group[0]["observed_order"]
      ax.loglog(x, error, linewidth=2.4, markersize=7.5, label=f"{method} p={order:.2f}", **line_styles[method])

   for method, order, label in (("PWC", 1.0, "dx"), ("PWL MC", 2.0, "dx^2"), ("WENO-Z", 5.0, "dx^5")):
      x, error = method_data[method]
      reference = reference_power_law(x, error, order)
      ax.loglog(x, reference, color="black", linestyle="--", linewidth=2.0, alpha=0.72)
      text_index = min(1, len(x) - 1)
      ax.annotate(
         label,
         xy=(x[text_index], reference[text_index]),
         xytext=(5, 0),
         textcoords="offset points",
         fontsize=10,
         va="center",
         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
      )

   local_lines = []
   for method in ("PWL MC", "WENO-Z", "WENO-Z char"):
      group = sorted(
         [row for row in rows if row["method"] == method],
         key=lambda row: row["x"],
         reverse=True,
      )
      local = [row["local_order_to_next"] for row in group[:-1] if np.isfinite(row["local_order_to_next"])]
      local_lines.append(f"{method} local p: " + ", ".join(f"{value:.2f}" for value in local))
   ax.text(
      0.03,
      0.04,
      "\n".join(local_lines),
      transform=ax.transAxes,
      fontsize=9.5,
      bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86, "pad": 4.0},
   )

   ax.set_xlabel("dx")
   ax.set_ylabel("characteristic L2 error")
   ax.set_title(
      f"Hydro smooth spatial reconstruction scan: {scan['name']}\n"
      f"{riemann_name(scan['riemann']).upper()}/{timestep_name(scan['timestep']).upper()}, "
      f"CFL={scan['cfl']:g}, t={scan['final_time']:g}, amp={scan['amplitude']:.0e}, "
      f"BC=periodic, threads={DEFAULT_THREADS}, float64"
   )
   ax.invert_xaxis()
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.legend(fontsize=10)
   SCANDIR.mkdir(parents=True, exist_ok=True)
   save_plot(fig, SCANDIR / "hydro_spatial_reconstruction_convergence.png", dpi=170)
   plt.close(fig)
   write_plot_index()


def plot_mhd_method_scans(rows):
   time_scan = MHD_METHOD_SCAN["time"]
   discontinuity_scan = MHD_METHOD_SCAN["discontinuity"]
   time_ct = ct_method_name(time_scan["ct_method"])
   discontinuity_ct = ct_method_name(discontinuity_scan["ct_method"])
   fig, axs = plt.subplots(1, 3, figsize=(18.5, 6.0), constrained_layout=True)
   draw_scan(
      axs[0],
      rows,
      "time_integrator",
      f"Time integrator\nsmooth fast+ wave, {time_scan['nx']}x{time_scan['ny']}\n"
      f"{short_recon_label(time_scan['recon'])}/{riemann_name(time_scan['riemann']).upper()}/{time_ct}, "
      f"{timestep_name(time_scan['reference_timestep']).upper()} ref",
      (("Euler", 1, "dt"), ("midpoint", 2, "dt^2"), ("SSPRK3", 3, "dt^3"), ("RK4", 4, "dt^4")),
   )
   draw_scan(
      axs[1],
      rows,
      "riemann_solver",
      f"MHD Riemann solver\n{discontinuity_scan['name']}\n"
      f"{short_recon_label(discontinuity_scan['riemann_recon'])}/"
      f"{timestep_name(discontinuity_scan['timestep']).upper()}/{discontinuity_ct}, CFL={discontinuity_scan['cfl']:g}",
      (("HLLD", 1, "dx"),),
   )
   draw_scan(
      axs[2],
      rows,
      "reconstruction",
      f"MHD reconstruction\n{discontinuity_scan['name']}\n"
      f"{riemann_name(discontinuity_scan['reconstruction_riemann']).upper()}/"
      f"{timestep_name(discontinuity_scan['timestep']).upper()}/{discontinuity_ct}, CFL={discontinuity_scan['cfl']:g}",
      (("WENO-Z char", 1, "dx"),),
   )
   fig.suptitle(
      f"Numba MHD method scans: smooth fast+ wave for time, {discontinuity_scan['name']} for Riemann/reconstruction, "
      f"threads={DEFAULT_THREADS}, {discontinuity_ct}, float64",
      fontsize=12,
   )
   SCANDIR.mkdir(parents=True, exist_ok=True)
   save_plot(fig, SCANDIR / "mhd_method_convergence.png", dpi=170)
   plt.close(fig)
   write_plot_index()


def hydro_characteristic_convergence_rows():
   rows = []
   resolutions = (32, 64, 128)
   modes = (
      HYDRO_MODE_ENTROPY,
      HYDRO_MODE_SHEAR,
      HYDRO_MODE_SHEAR_Z,
      HYDRO_MODE_SOUND_MINUS,
      HYDRO_MODE_SOUND_PLUS,
   )
   for mode in modes:
      amplitude = 1.0e-8 if mode in (HYDRO_MODE_SOUND_MINUS, HYDRO_MODE_SOUND_PLUS) else 1.0e-6
      for n in resolutions:
         row = run_hydro_characteristic_wave_case(
            HydroCharacteristicWaveConfig(
               nx=n,
               ny=n,
               ng=3,
               final_time=0.05,
               mode=mode,
               amplitude=amplitude,
               recon=RECON_WENOZ,
               recon_vars=RECON_VARS_PRIMITIVE,
               limiter=LIMITER_MC,
               riemann=RIEMANN_HLLC,
               timestep=TIME_SSPRK3,
            )
         )
         require_finite_row(f"hydro_characteristic_{hydro_mode_name(mode)}_{n}", row)
         rows.append(
            {
               "mode": hydro_mode_name(mode),
               "nx": row["nx"],
               "ny": row["ny"],
               "dx": 1.0 / row["nx"],
               "steps": row["steps"],
               "amplitude": amplitude,
               "mode_l2": row["mode_l2"],
               "rho_l2": row["rho_l2"],
               "pressure_l2": row["pressure_l2"],
               "velocity_l2": row["velocity_l2"],
               "mass_rel_drift": row["mass_rel_drift"],
               "energy_rel_drift": row["energy_rel_drift"],
               "rate_mcells_s": row["million_cell_updates_per_second"],
               "observed_order": np.nan,
            }
         )
   for mode_name in sorted({row["mode"] for row in rows}):
      group = [row for row in rows if row["mode"] == mode_name]
      order = observed_order([row["dx"] for row in group], [row["mode_l2"] for row in group])
      for row in group:
         row["observed_order"] = order
   return rows


def mhd_characteristic_convergence_rows():
   rows = []
   resolutions = (32, 64, 128)
   modes = (
      MHD_MODE_FAST_MINUS,
      MHD_MODE_ALFVEN_MINUS,
      MHD_MODE_SLOW_MINUS,
      MHD_MODE_ENTROPY,
      MHD_MODE_SLOW_PLUS,
      MHD_MODE_ALFVEN_PLUS,
      MHD_MODE_FAST_PLUS,
   )
   for mode in modes:
      for n in resolutions:
         row = run_mhd_characteristic_wave_case(
            MHDCharacteristicWaveConfig(
               nx=n,
               ny=n,
               ng=3,
               final_time=0.05,
               mode=mode,
               amplitude=1.0e-7,
               recon=RECON_WENOZ_CHAR,
               recon_vars=RECON_VARS_PRIMITIVE,
               limiter=LIMITER_MC,
               riemann=RIEMANN_HLLD,
               timestep=TIME_SSPRK3,
               ct_method=CT_FLUX_TOTH,
            )
         )
         require_finite_row(f"mhd_characteristic_{mhd_mode_name(mode)}_{n}", row)
         rows.append(
            {
               "mode": mhd_mode_name(mode),
               "nx": row["nx"],
               "ny": row["ny"],
               "dx": 1.0 / row["nx"],
               "steps": row["steps"],
               "amplitude": 1.0e-7,
               "mode_l2": row["mode_l2"],
               "rho_l2": row["rho_l2"],
               "pressure_l2": row["pressure_l2"],
               "velocity_l2": row["velocity_l2"],
               "magnetic_l2": row["magnetic_l2"],
               "mass_rel_drift": row["mass_rel_drift"],
               "energy_rel_drift": row["energy_rel_drift"],
               "divb_final": row["divb_final"],
               "rate_mcells_s": row["million_cell_updates_per_second"],
               "observed_order": np.nan,
            }
         )
   for mode_name in sorted({row["mode"] for row in rows}):
      group = [row for row in rows if row["mode"] == mode_name]
      order = observed_order([row["dx"] for row in group], [row["mode_l2"] for row in group])
      for row in group:
         row["observed_order"] = order
   return rows


def check_hydro_eigensystem():
   rho = 1.3
   pressure = 0.9
   gamma = 5.0 / 3.0
   vx = 0.2
   vy = -0.13
   normal_x = 2.0 / np.sqrt(5.0)
   normal_y = 1.0 / np.sqrt(5.0)
   right = hydro_primitive_right_eigenvectors(rho, pressure, gamma, normal_x, normal_y)
   left = hydro_primitive_left_eigenvectors(rho, pressure, gamma, normal_x, normal_y)
   eigenvalues = hydro_characteristic_eigenvalues(rho, pressure, vx, vy, gamma, normal_x, normal_y)
   normal_velocity = vx * normal_x + vy * normal_y
   jacobian = np.array(
      (
         (normal_velocity, rho * normal_x, rho * normal_y, 0.0, 0.0),
         (0.0, normal_velocity, 0.0, 0.0, normal_x / rho),
         (0.0, 0.0, normal_velocity, 0.0, normal_y / rho),
         (0.0, 0.0, 0.0, normal_velocity, 0.0),
         (0.0, gamma * pressure * normal_x, gamma * pressure * normal_y, 0.0, normal_velocity),
      ),
      dtype=float,
   )
   identity_error = float(np.max(np.abs(left @ right - np.eye(5))))
   eigen_error = float(np.max(np.abs(jacobian @ right - right @ np.diag(eigenvalues))))
   if identity_error > 1.0e-12 or eigen_error > 1.0e-12:
      raise AssertionError(f"hydro eigensystem check failed: identity={identity_error:.3e}, eigen={eigen_error:.3e}")
   return identity_error, eigen_error


def check_mhd_eigensystem():
   rho = 1.3
   pressure = 0.9
   gamma = 5.0 / 3.0
   v_normal = 0.2
   b_normal = 0.5
   b_tangent = 0.3
   b_z = -0.2
   right = mhd_primitive_reduced_right_eigenvectors(rho, pressure, b_normal, b_tangent, b_z, gamma)
   left = mhd_primitive_reduced_left_eigenvectors(rho, pressure, b_normal, b_tangent, b_z, gamma)
   eigenvalues = mhd_primitive_reduced_eigenvalues(rho, pressure, v_normal, b_normal, b_tangent, b_z, gamma)
   jacobian = mhd_primitive_reduced_jacobian(rho, v_normal, pressure, b_normal, b_tangent, b_z, gamma)
   identity_error = float(np.max(np.abs(left @ right - np.eye(7))))
   eigen_error = float(np.max(np.abs(jacobian @ right - right @ np.diag(eigenvalues))))
   if identity_error > 1.0e-12 or eigen_error > 1.0e-12:
      raise AssertionError(f"MHD eigensystem check failed: identity={identity_error:.3e}, eigen={eigen_error:.3e}")
   return identity_error, eigen_error


def check_mhd_characteristic_projection():
   rho_ref = 1.3
   pressure_ref = 0.9
   gamma = 5.0 / 3.0
   b_normal = 0.5
   b_tangent = 0.3
   b_z_ref = -0.2
   reduced = np.array((1.27, 0.11, -0.09, 0.04, 0.83, 0.21, -0.14), dtype=np.float64)
   left = mhd_primitive_reduced_left_eigenvectors(rho_ref, pressure_ref, b_normal, b_tangent, b_z_ref, gamma)
   expected = left @ reduced

   sound2 = gamma * pressure_ref / rho_ref
   b2 = (b_normal * b_normal + b_tangent * b_tangent + b_z_ref * b_z_ref) / rho_ref
   ca2 = b_normal * b_normal / rho_ref
   discr = max((sound2 + b2) * (sound2 + b2) - 4.0 * sound2 * ca2, 0.0)
   fast2 = 0.5 * (sound2 + b2 + np.sqrt(discr))
   slow2 = 0.5 * (sound2 + b2 - np.sqrt(discr))
   sqrt_rho = np.sqrt(rho_ref)
   sound_speed = np.sqrt(sound2)
   fast_speed = np.sqrt(max(fast2, 0.0))
   slow_speed = np.sqrt(max(slow2, 0.0))
   b_perp = np.sqrt(b_tangent * b_tangent + b_z_ref * b_z_ref)
   beta_t = b_tangent / b_perp
   beta_z = b_z_ref / b_perp
   denom = fast2 - slow2
   alpha_f = np.sqrt(max((sound2 - slow2) / denom, 0.0))
   alpha_s = np.sqrt(max((fast2 - sound2) / denom, 0.0))
   normal_sign = 1.0 if b_normal >= 0.0 else -1.0

   i = 3
   j = 3
   Vcache = np.zeros((NVAR_MHD, 7, 7), dtype=np.float64)
   Vcache[RHO, i, j] = reduced[0]
   Vcache[MOMX, i, j] = reduced[1]
   Vcache[MOMY, i, j] = reduced[2]
   Vcache[MOMZ, i, j] = reduced[3]
   Vcache[ENERGY, i, j] = reduced[4]
   Vcache[BX, i, j] = b_normal
   Vcache[BY, i, j] = reduced[5]
   Vcache[BZ, i, j] = reduced[6]

   actual = np.array(
      mhd_characteristics_at_offset(
         Vcache, i, j, AXIS_X, 0, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
         alpha_f, alpha_s, beta_t, beta_z, normal_sign
      ),
      dtype=np.float64,
   )
   roundtrip = np.array(
      mhd_characteristic_to_primitive(
         actual[0], actual[1], actual[2], actual[3], actual[4], actual[5], actual[6],
         rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
         alpha_f, alpha_s, beta_t, beta_z, normal_sign
      ),
      dtype=np.float64,
   )
   projection_error = float(np.max(np.abs(actual - expected)))
   roundtrip_error = float(np.max(np.abs(roundtrip - reduced)))
   if projection_error > 1.0e-12 or roundtrip_error > 1.0e-12:
      raise AssertionError(
         f"MHD characteristic projection check failed: projection={projection_error:.3e}, roundtrip={roundtrip_error:.3e}"
      )
   return projection_error, roundtrip_error


def plot_hydro_characteristic_convergence(rows, outpath):
   fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
   for mode_name in ("entropy", "shear", "shear-z", "sound-", "sound+"):
      group = sorted([row for row in rows if row["mode"] == mode_name], key=lambda row: row["dx"], reverse=True)
      dx = np.array([row["dx"] for row in group], dtype=float)
      error = np.array([row["mode_l2"] for row in group], dtype=float)
      order = group[0]["observed_order"]
      ax.loglog(dx, error, marker="o", linewidth=2.4, markersize=7.0, label=f"{mode_name} p={order:.2f}")
   reference_group = sorted([row for row in rows if row["mode"] == "sound+"], key=lambda row: row["dx"], reverse=True)
   dx_ref = np.array([row["dx"] for row in reference_group], dtype=float)
   err_ref = np.array([row["mode_l2"] for row in reference_group], dtype=float)
   ref = reference_power_law(dx_ref, err_ref, 3.0)
   ax.loglog(dx_ref, ref, color="black", linestyle="--", linewidth=2.0, alpha=0.75)
   ax.annotate("dx^3", xy=(dx_ref[1], ref[1]), xytext=(5, 0), textcoords="offset points", fontsize=11)
   ax.set_xlabel("dx")
   ax.set_ylabel("characteristic mode L2")
   ax.set_title(
      "Hydro characteristic-mode convergence\n"
      "WENO-Z prim/HLLC/SSPRK3, CFL=0.9, BC=periodic, amp=1e-6 non-sound / 1e-8 sound"
   )
   ax.invert_xaxis()
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.legend(fontsize=10)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def plot_mhd_characteristic_convergence(rows, outpath):
   fig, ax = plt.subplots(figsize=(9.5, 6.4), constrained_layout=True)
   mode_order = ("fast-", "alfven-", "slow-", "entropy", "slow+", "alfven+", "fast+")
   for mode_name in mode_order:
      group = sorted([row for row in rows if row["mode"] == mode_name], key=lambda row: row["dx"], reverse=True)
      dx = np.array([row["dx"] for row in group], dtype=float)
      error = np.array([row["mode_l2"] for row in group], dtype=float)
      order = group[0]["observed_order"]
      ax.loglog(dx, error, marker="o", linewidth=2.4, markersize=7.0, label=f"{mode_name} p={order:.2f}")
   reference_group = sorted([row for row in rows if row["mode"] == "fast+"], key=lambda row: row["dx"], reverse=True)
   dx_ref = np.array([row["dx"] for row in reference_group], dtype=float)
   err_ref = np.array([row["mode_l2"] for row in reference_group], dtype=float)
   ref = reference_power_law(dx_ref, err_ref, 2.0)
   ax.loglog(dx_ref, ref, color="black", linestyle="--", linewidth=2.0, alpha=0.75)
   ax.annotate("dx^2", xy=(dx_ref[1], ref[1]), xytext=(5, 0), textcoords="offset points", fontsize=11)
   ax.set_xlabel("dx")
   ax.set_ylabel("characteristic mode L2")
   ax.set_title("MHD characteristic-mode convergence\nWENO-Z char prim/HLLD/SSPRK3/CT, CFL=0.9, BC=periodic, amp=1e-7")
   ax.invert_xaxis()
   ax.grid(True, which="major", alpha=0.25)
   ax.grid(False, which="minor")
   ax.legend(fontsize=9, ncol=2)
   outpath.parent.mkdir(parents=True, exist_ok=True)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def case_hydro_characteristic_convergence():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   identity_error, eigen_error = check_hydro_eigensystem()
   mhd_identity_error, mhd_eigen_error = check_mhd_eigensystem()
   mhd_projection_error, mhd_projection_roundtrip = check_mhd_characteristic_projection()
   rows = hydro_characteristic_convergence_rows()
   write_csv(rows, OUTDIR / "hydro_characteristic_convergence.csv")
   plot_hydro_characteristic_convergence(rows, OUTDIR / "hydro_characteristic_convergence.png")
   write_plot_index()
   min_order = min(row["observed_order"] for row in rows)
   max_error = max(row["mode_l2"] for row in rows if row["nx"] == 128)
   if min_order < 2.4:
      raise AssertionError(f"hydro characteristic convergence order too low: {min_order:.3f}")
   return {
      "modes": sorted({row["mode"] for row in rows}),
      "eigensystem_identity_error": identity_error,
      "eigensystem_residual": eigen_error,
      "mhd_eigensystem_identity_error": mhd_identity_error,
      "mhd_eigensystem_residual": mhd_eigen_error,
      "mhd_projection_error": mhd_projection_error,
      "mhd_projection_roundtrip_error": mhd_projection_roundtrip,
      "min_observed_order": min_order,
      "max_128_mode_l2": max_error,
   }


def case_mhd_characteristic_convergence():
   SCANDIR.mkdir(parents=True, exist_ok=True)
   identity_error, eigen_error = check_mhd_eigensystem()
   projection_error, projection_roundtrip = check_mhd_characteristic_projection()
   rows = mhd_characteristic_convergence_rows()
   write_csv(rows, SCANDIR / "mhd_characteristic_convergence.csv")
   plot_mhd_characteristic_convergence(rows, SCANDIR / "mhd_characteristic_convergence.png")
   write_plot_index()
   min_order = min(row["observed_order"] for row in rows)
   max_error = max(row["mode_l2"] for row in rows if row["nx"] == 128)
   max_divb = max(abs(row["divb_final"]) for row in rows)
   if min_order < 1.8:
      raise AssertionError(f"MHD characteristic convergence order too low: {min_order:.3f}")
   if max_divb > 1.0e-10:
      raise AssertionError(f"MHD characteristic divB too high: {max_divb:.3e}")
   return {
      "modes": sorted({row["mode"] for row in rows}),
      "eigensystem_identity_error": identity_error,
      "eigensystem_residual": eigen_error,
      "projection_error": projection_error,
      "projection_roundtrip_error": projection_roundtrip,
      "min_observed_order": min_order,
      "max_128_mode_l2": max_error,
      "max_divb": max_divb,
   }


def sod_initial_state(nx, ny, ng, gamma, left, right):
   Uvec = np.zeros((NVAR_HYDRO, nx + 2 * ng, ny + 2 * ng), dtype=np.float64)
   for ii in range(nx):
      x = (ii + 0.5) / nx
      rho, velocity, pressure = left if x < 0.5 else right
      energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity
      for jj in range(ny):
         i = ng + ii
         j = ng + jj
         Uvec[RHO, i, j] = rho
         Uvec[MOMX, i, j] = rho * velocity
         Uvec[MOMY, i, j] = 0.0
         Uvec[MOMZ, i, j] = 0.0
         Uvec[ENERGY, i, j] = energy
   apply_boundaries(Uvec, ng, BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC)
   return Uvec


def case_sod_exact_plot():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   gamma = 1.4
   nx = 512
   ny = 8
   ng = 2
   final_time = 0.2
   cfl = 0.45
   left = (1.0, 0.0, 1.0)
   right = (0.125, 0.0, 0.1)
   dx = 1.0 / nx
   dy = 1.0 / ny
   bc = (BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC)
   runtime_threads = configure_numba_threads(DEFAULT_THREADS)

   Uvec = sod_initial_state(nx, ny, ng, gamma, left, right)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, nx, ny)
   evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc[0], bc[1], bc[2], bc[3], RECON_PLM, RECON_VARS_PRIMITIVE, LIMITER_MC, RIEMANN_HLLC,
      TIME_SSPRK3, 2.0 * dx, 8, CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )

   Uvec = sod_initial_state(nx, ny, ng, gamma, left, right)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, nx, ny)
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc[0], bc[1], bc[2], bc[3], RECON_PLM, RECON_VARS_PRIMITIVE, LIMITER_MC, RIEMANN_HLLC,
      TIME_SSPRK3, final_time, 20000, CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   if abs(time_value - final_time) > 1.0e-12:
      raise AssertionError(f"Sod run ended at t={time_value}, expected {final_time}")

   interior = (slice(ng, ng + nx), slice(ng, ng + ny))
   rho_num = np.mean(Uvec[RHO][interior], axis=1)
   vx_cell = Uvec[MOMX][interior] / Uvec[RHO][interior]
   vy_cell = Uvec[MOMY][interior] / Uvec[RHO][interior]
   vz_cell = Uvec[MOMZ][interior] / Uvec[RHO][interior]
   pressure_cell = (gamma - 1.0) * (
      Uvec[ENERGY][interior]
      - 0.5 * Uvec[RHO][interior] * (vx_cell * vx_cell + vy_cell * vy_cell + vz_cell * vz_cell)
   )
   pressure_num = np.mean(pressure_cell, axis=1)
   velocity_num = np.mean(vx_cell, axis=1)
   x = (np.arange(nx) + 0.5) / nx
   rho_exact, pressure_exact, velocity_exact = sod_exact_solution(x, time_value, gamma, left, right)

   rho_l1 = float(np.mean(np.abs(rho_num - rho_exact)))
   pressure_l1 = float(np.mean(np.abs(pressure_num - pressure_exact)))
   velocity_l1 = float(np.mean(np.abs(velocity_num - velocity_exact)))
   if not np.all(np.isfinite(rho_num)) or not np.all(np.isfinite(pressure_num)) or np.min(pressure_num) <= 0.0:
      raise AssertionError("Sod run produced non-finite or non-positive primitive fields")
   if rho_l1 > 3.0e-2 or pressure_l1 > 3.0e-2 or velocity_l1 > 6.0e-2:
      raise AssertionError(f"Sod exact errors too large: rho={rho_l1:.3e}, P={pressure_l1:.3e}, vx={velocity_l1:.3e}")

   rows = [
      {
         "x": x[i],
         "rho_num": rho_num[i],
         "rho_exact": rho_exact[i],
         "pressure_num": pressure_num[i],
         "pressure_exact": pressure_exact[i],
         "velocity_num": velocity_num[i],
         "velocity_exact": velocity_exact[i],
      }
      for i in range(nx)
   ]
   write_csv(rows, OUTDIR / "sod_exact.csv")

   fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
   fields = (
      ("rho", rho_num, rho_exact),
      ("P", pressure_num, pressure_exact),
      ("vx", velocity_num, velocity_exact),
   )
   for ax, (label, numerical, exact) in zip(axs, fields):
      ax.plot(
         x,
         numerical,
         color="#2878b5",
         linewidth=2.4,
         marker="o",
         markersize=3.8,
         markevery=6,
         label="sim",
         zorder=2,
      )
      ax.plot(x, exact, color="black", linestyle=":", linewidth=2.2, label="exact", zorder=3)
      ax.set_ylabel(label)
      ax.grid(True, which="major", alpha=0.25)
      ax.tick_params(labelsize=10)
   axs[0].set_title(f"Sod shock tube: PWL prim/MC/HLLC/SSPRK3, BC={bc_tuple_name(bc)}, t={time_value:.3f}, {nx}x{ny}")
   axs[-1].set_xlabel("x")
   axs[0].legend(loc="best", fontsize=11)
   save_plot(fig, OUTDIR / "sod_exact.png", dpi=170)
   plt.close(fig)
   write_plot_index()
   return {
      "rho_l1": rho_l1,
      "pressure_l1": pressure_l1,
      "velocity_l1": velocity_l1,
      "steps": steps,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "threads": runtime_threads,
   }


def sod_profiles(Uvec, nx, ny, ng, gamma):
   interior = (slice(ng, ng + nx), slice(ng, ng + ny))
   rho_num = np.mean(Uvec[RHO][interior], axis=1)
   vx_cell = Uvec[MOMX][interior] / Uvec[RHO][interior]
   vy_cell = Uvec[MOMY][interior] / Uvec[RHO][interior]
   vz_cell = Uvec[MOMZ][interior] / Uvec[RHO][interior]
   pressure_cell = (gamma - 1.0) * (
      Uvec[ENERGY][interior]
      - 0.5 * Uvec[RHO][interior] * (vx_cell * vx_cell + vy_cell * vy_cell + vz_cell * vz_cell)
   )
   return rho_num, np.mean(pressure_cell, axis=1), np.mean(vx_cell, axis=1)


def run_sod_method(
   nx,
   ny,
   ng,
   gamma,
   final_time,
   cfl,
   left,
   right,
   recon,
   riemann=RIEMANN_HLLC,
   timestep=TIME_RK4,
   recon_vars=RECON_VARS_PRIMITIVE,
   limiter=LIMITER_MC,
):
   dx = 1.0 / nx
   dy = 1.0 / ny
   bc = (BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC)
   Uvec = sod_initial_state(nx, ny, ng, gamma, left, right)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, nx, ny)
   evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc[0], bc[1], bc[2], bc[3], recon, recon_vars, limiter, riemann,
      timestep, 2.0 * dx, 8, CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )

   Uvec = sod_initial_state(nx, ny, ng, gamma, left, right)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, nx, ny)
   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc[0], bc[1], bc[2], bc[3], recon, recon_vars, limiter, riemann,
      timestep, final_time, 20000, CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   if abs(time_value - final_time) > 1.0e-12:
      raise AssertionError(f"Sod run ended at t={time_value}, expected {final_time}")
   rho_num, pressure_num, velocity_num = sod_profiles(Uvec, nx, ny, ng, gamma)
   if not np.all(np.isfinite(rho_num)) or not np.all(np.isfinite(pressure_num)) or np.min(rho_num) <= 0.0 or np.min(pressure_num) <= 0.0:
      raise AssertionError(f"Sod {recon_name(recon)} run produced non-finite or non-positive primitive fields")
   return {
      "recon": recon,
      "recon_name": recon_name(recon),
      "time": time_value,
      "steps": steps,
      "nx": nx,
      "ny": ny,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(nx, ny, steps, timestep, elapsed),
      "rho": rho_num,
      "pressure": pressure_num,
      "velocity": velocity_num,
   }


def sod_error_metrics(method, rho_exact, pressure_exact, velocity_exact):
   rho = method["rho"]
   pressure = method["pressure"]
   velocity = method["velocity"]
   return {
      "method": method["recon_name"],
      "rho_l1": float(np.mean(np.abs(rho - rho_exact))),
      "pressure_l1": float(np.mean(np.abs(pressure - pressure_exact))),
      "velocity_l1": float(np.mean(np.abs(velocity - velocity_exact))),
      "rho_l2": float(np.sqrt(np.mean((rho - rho_exact) ** 2))),
      "pressure_l2": float(np.sqrt(np.mean((pressure - pressure_exact) ** 2))),
      "velocity_l2": float(np.sqrt(np.mean((velocity - velocity_exact) ** 2))),
      "rho_min": float(np.min(rho)),
      "pressure_min": float(np.min(pressure)),
      "rho_overshoot": float(max(0.0, np.max(rho) - max(np.max(rho_exact), 1.0))),
      "pressure_overshoot": float(max(0.0, np.max(pressure) - max(np.max(pressure_exact), 1.0))),
      "steps": method["steps"],
      "dt_min": method["dt_min"],
      "dt_max": method["dt_max"],
      "million_cell_updates_per_second": method["million_cell_updates_per_second"],
   }


def plot_sod_wenoz_char_scalars(metrics, outpath, title):
   methods = [row["method"] for row in metrics]
   x = np.arange(len(methods))
   width = 0.24
   fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)

   l1_fields = (("rho", "rho_l1"), ("P", "pressure_l1"), ("vx", "velocity_l1"))
   for offset, (label, key) in zip((-width, 0.0, width), l1_fields):
      axs[0, 0].bar(x + offset, [row[key] for row in metrics], width=width, label=label)
   axs[0, 0].set_title("L1 error vs exact")
   axs[0, 0].set_yscale("log")
   axs[0, 0].legend(fontsize=9)

   l2_fields = (("rho", "rho_l2"), ("P", "pressure_l2"), ("vx", "velocity_l2"))
   for offset, (label, key) in zip((-width, 0.0, width), l2_fields):
      axs[0, 1].bar(x + offset, [row[key] for row in metrics], width=width, label=label)
   axs[0, 1].set_title("L2 error vs exact")
   axs[0, 1].set_yscale("log")
   axs[0, 1].legend(fontsize=9)

   overshoot_fields = (("rho", "rho_overshoot"), ("P", "pressure_overshoot"))
   for offset, (label, key) in zip((-0.5 * width, 0.5 * width), overshoot_fields):
      axs[1, 0].bar(x + offset, [max(row[key], 1.0e-16) for row in metrics], width=width, label=label)
   axs[1, 0].set_title("positive overshoot")
   axs[1, 0].set_yscale("log")
   axs[1, 0].legend(fontsize=9)

   axs[1, 1].bar(x, [row["million_cell_updates_per_second"] for row in metrics], width=0.45, color="#4c78a8")
   axs[1, 1].set_title("throughput")
   axs[1, 1].set_ylabel("M cells/s")

   for ax in axs.flat:
      ax.set_xticks(x)
      ax.set_xticklabels(methods)
      ax.grid(True, which="major", axis="y", alpha=0.25)
      ax.grid(False, which="minor")
   fig.suptitle(title, fontsize=12)
   save_plot(fig, outpath, dpi=180)
   plt.close(fig)


def sod_exact_conserved_totals(time_value, gamma, left, right):
   return sod_conserved_integrals(time_value, gamma, left, right, quadrature_order=96)


def collect_sod_conserved_history(nx, ny, ng, gamma, final_time, cfl, left, right, recon, output_times):
   dx = 1.0 / nx
   dy = 1.0 / ny
   bc = (BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC)
   Uvec = sod_initial_state(nx, ny, ng, gamma, left, right)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, nx, ny)
   time_value = 0.0
   rows = []
   method = "WENO-Z char" if recon == RECON_WENOZ_CHAR else "WENO-Z"

   for target_time in output_times:
      delta = float(target_time - time_value)
      if delta > 1.0e-14:
         reached, _, _, _ = evolve_to_time(
            Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
            bc[0], bc[1], bc[2], bc[3], recon, RECON_VARS_PRIMITIVE, LIMITER_MC, RIEMANN_HLLC,
            TIME_RK4, delta, 20000, CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
         )
         time_value += reached
      scalars = conserved_totals(Uvec, nx, ny, ng, dx, dy)
      rows.append(
         {
            "method": method,
            "time": time_value,
            "mass": scalars["mass"],
            "energy": scalars["energy"],
            "momx": scalars["momx"],
         }
      )
   if abs(time_value - final_time) > 1.0e-12:
      raise AssertionError(f"Sod scalar history ended at t={time_value}, expected {final_time}")
   return rows


def collect_sod_exact_conserved_history(nx, gamma, left, right, output_times):
   rows = []
   for time_value in output_times:
      totals = sod_exact_conserved_totals(float(time_value), gamma, left, right)
      rows.append(
         {
            "method": "exact",
            "time": float(time_value),
            "mass": totals["mass"],
            "energy": totals["energy"],
            "momx": totals["momx"],
         }
      )
   return rows


def plot_sod_conserved_history(rows, outpath, title):
   methods = ("WENO-Z", "WENO-Z char", "exact")
   colors = {"WENO-Z": "#2878b5", "WENO-Z char": "#d95f02", "exact": "black"}
   styles = {"WENO-Z": "-", "WENO-Z char": "-", "exact": ":"}
   markers = {"WENO-Z": "o", "WENO-Z char": "s", "exact": None}
   fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
   fields = (("mass", "mass"), ("energy", "energy"), ("x-momentum", "momx"))
   for ax, (title_text, key) in zip(axs, fields):
      for method in methods:
         group = [row for row in rows if row["method"] == method]
         if not group:
            continue
         time_values = np.array([row["time"] for row in group], dtype=float)
         values = np.array([row[key] for row in group], dtype=float)
         kwargs = {"linewidth": 2.2, "color": colors[method], "linestyle": styles[method], "label": method}
         if markers[method] is not None and len(time_values) <= 40:
            kwargs.update({"marker": markers[method], "markersize": 4.0})
         ax.plot(time_values, values, **kwargs)
      ax.set_title(title_text)
      ax.set_xlabel("time")
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
      ax.ticklabel_format(axis="y", style="plain", useOffset=False)
   axs[0].legend(loc="best", fontsize=9)
   fig.suptitle(title, fontsize=12)
   save_plot(fig, outpath, dpi=180)
   plt.close(fig)


def run_sod_wenoz_char_compare(stem, left, right, label, cfl=0.9, final_time=0.2):
   OUTDIR.mkdir(parents=True, exist_ok=True)
   outstem = OUTDIR / stem
   outstem.parent.mkdir(parents=True, exist_ok=True)
   gamma = 1.4
   nx = 512
   ny = 8
   ng = 3
   bc = (BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC)
   runtime_threads = configure_numba_threads(DEFAULT_THREADS)

   component = run_sod_method(nx, ny, ng, gamma, final_time, cfl, left, right, RECON_WENOZ)
   characteristic = run_sod_method(nx, ny, ng, gamma, final_time, cfl, left, right, RECON_WENOZ_CHAR)
   x = (np.arange(nx) + 0.5) / nx
   rho_exact, pressure_exact, velocity_exact = sod_exact_solution(x, final_time, gamma, left, right)
   component_metrics = sod_error_metrics(component, rho_exact, pressure_exact, velocity_exact)
   characteristic_metrics = sod_error_metrics(characteristic, rho_exact, pressure_exact, velocity_exact)
   metrics = [component_metrics, characteristic_metrics]

   write_csv(metrics, outstem.with_name(outstem.name + "_metrics.csv"))
   rows = [
      {
         "x": x[i],
         "rho_exact": rho_exact[i],
         "rho_wenoz": component["rho"][i],
         "rho_wenoz_char": characteristic["rho"][i],
         "pressure_exact": pressure_exact[i],
         "pressure_wenoz": component["pressure"][i],
         "pressure_wenoz_char": characteristic["pressure"][i],
         "velocity_exact": velocity_exact[i],
         "velocity_wenoz": component["velocity"][i],
         "velocity_wenoz_char": characteristic["velocity"][i],
      }
      for i in range(nx)
   ]
   write_csv(rows, outstem.with_suffix(".csv"))

   fig, axs = plt.subplots(3, 1, figsize=(10.0, 8.8), sharex=True, constrained_layout=True)
   fields = (
      ("rho", "rho"),
      ("P", "pressure"),
      ("vx", "velocity"),
   )
   for ax, (field_label, key) in zip(axs, fields):
      exact = {"rho": rho_exact, "pressure": pressure_exact, "velocity": velocity_exact}[key]
      ax.plot(
         x,
         component[key],
         color="#2878b5",
         linewidth=2.2,
         marker="o",
         markersize=3.3,
         markevery=7,
         label="WENO-Z",
         zorder=2,
      )
      ax.plot(
         x,
         characteristic[key],
         color="#d95f02",
         linewidth=2.0,
         marker="s",
         markersize=3.0,
         markevery=7,
         label="WENO-Z char",
         zorder=3,
      )
      ax.plot(x, exact, color="black", linestyle=":", linewidth=2.2, label="exact", zorder=4)
      ax.set_ylabel(field_label)
      ax.grid(True, which="major", alpha=0.25)
      ax.tick_params(labelsize=10)
   axs[0].set_title(
      f"{label}: WENO-Z vs WENO-Z char, HLLC/RK4, CFL={cfl:g}\n"
      f"BC={bc_tuple_name(bc)}, t={final_time:.3f}, {nx}x{ny}, threads={runtime_threads}"
   )
   axs[-1].set_xlabel("x")
   axs[0].legend(loc="best", fontsize=10)
   save_plot(fig, outstem.with_suffix(".png"), dpi=180)
   plt.close(fig)
   plot_sod_wenoz_char_scalars(
      metrics,
      outstem.with_name(outstem.name.replace("_compare", "_scalars") + ".png"),
      f"{label} scalars: WENO-Z vs WENO-Z char, HLLC/RK4, CFL={cfl:g}, t={final_time:.3f}, {nx}x{ny}",
   )
   output_times = np.linspace(0.0, final_time, 21)
   conserved_rows = []
   conserved_rows.extend(collect_sod_conserved_history(nx, ny, ng, gamma, final_time, cfl, left, right, RECON_WENOZ, output_times))
   conserved_rows.extend(collect_sod_conserved_history(nx, ny, ng, gamma, final_time, cfl, left, right, RECON_WENOZ_CHAR, output_times))
   conserved_rows.extend(collect_sod_exact_conserved_history(nx, gamma, left, right, output_times))
   write_csv(conserved_rows, outstem.with_name(outstem.name.replace("_compare", "_conserved") + ".csv"))
   plot_sod_conserved_history(
      conserved_rows,
      outstem.with_name(outstem.name.replace("_compare", "_conserved") + ".png"),
      f"{label} conserved totals: WENO-Z vs WENO-Z char vs exact, HLLC/RK4, CFL={cfl:g}",
   )
   write_plot_index()
   return {
      "component_rho_l1": component_metrics["rho_l1"],
      "char_rho_l1": characteristic_metrics["rho_l1"],
      "component_pressure_l1": component_metrics["pressure_l1"],
      "char_pressure_l1": characteristic_metrics["pressure_l1"],
      "component_velocity_l1": component_metrics["velocity_l1"],
      "char_velocity_l1": characteristic_metrics["velocity_l1"],
      "component_mcells_s": component_metrics["million_cell_updates_per_second"],
      "char_mcells_s": characteristic_metrics["million_cell_updates_per_second"],
   }


def case_sod_wenoz_char_compare():
   return run_sod_wenoz_char_compare(
      "IC/sod/sod_wenoz_char_compare",
      (1.0, 0.0, 1.0),
      (0.125, 0.0, 0.1),
      "Sod shock tube",
   )


def case_sod_100to1_wenoz_char_compare():
   return run_sod_wenoz_char_compare(
      "IC/sod/sod_100to1_wenoz_char_compare",
      (100.0, 0.0, 100.0),
      (1.0, 0.0, 1.0),
      "Sod shock tube 100:1 rho/P jump",
      cfl=0.45,
      final_time=0.15,
   )


def circular_alfven_profile(Uvec, config):
   ng = config.ng
   sx = slice(ng, ng + config.nx)
   j = ng + config.ny // 2
   rho = Uvec[RHO, sx, j]
   bx = Uvec[BX, sx, j]
   by = Uvec[BY, sx, j]
   bz = Uvec[BZ, sx, j]
   vx = Uvec[MOMX, sx, j] / rho
   vy = Uvec[MOMY, sx, j] / rho
   vz = Uvec[MOMZ, sx, j] / rho
   pressure = (config.gamma - 1.0) * (
      Uvec[ENERGY, sx, j]
      - 0.5 * (Uvec[MOMX, sx, j] ** 2 + Uvec[MOMY, sx, j] ** 2 + Uvec[MOMZ, sx, j] ** 2) / rho
      - 0.5 * (bx * bx + by * by + bz * bz)
   )
   return {
      "x": (np.arange(config.nx) + 0.5) / config.nx,
      "rho": rho,
      "pressure": pressure,
      "vx": vx,
      "vy": vy,
      "vz": vz,
      "bx": bx,
      "by": by,
      "bz": bz,
   }


def circular_alfven_rows():
   rows = []
   finest_profile = None
   finest_config = None
   for n in (32, 64, 128):
      config = CircularAlfvenWaveConfig(
         nx=n,
         ny=8,
         ng=3,
         final_time=1.0,
         cfl=0.9,
         recon=RECON_WENOZ_CHAR,
         recon_vars=RECON_VARS_PRIMITIVE,
         limiter=LIMITER_MC,
         riemann=RIEMANN_HLLD,
         timestep=TIME_RK4,
         ct_method=CT_FLUX_TOTH,
         amplitude=0.1,
         wave_number=1,
      )
      row, Uvec = evolve_circular_alfven_wave_case(config)
      require_finite_row(f"circular_alfven_{n}", row)
      rows.append(
         {
            "nx": row["nx"],
            "ny": row["ny"],
            "dx": 1.0 / row["nx"],
            "steps": row["steps"],
            "cfl": config.cfl,
            "final_time": row["time"],
            "recon": recon_name(config.recon),
            "riemann": riemann_name(config.riemann),
            "timestep": timestep_name(config.timestep),
            "ct_method": config.ct_method,
            "ct_method_name": ct_method_name(config.ct_method),
            "primitive_l2": row["primitive_l2"],
            "rho_l2": row["rho_l2"],
            "pressure_l2": row["pressure_l2"],
            "velocity_l2": row["velocity_l2"],
            "magnetic_l2": row["magnetic_l2"],
            "mass_rel_drift": row["mass_rel_drift"],
            "energy_rel_drift": row["energy_rel_drift"],
            "divb_initial": row["divb_initial"],
            "divb_final": row["divb_final"],
            "rate_mcells_s": row["million_cell_updates_per_second"],
            "observed_order": np.nan,
         }
      )
      if n == 128:
         finest_profile = circular_alfven_profile(Uvec, config)
         finest_config = config
   order = observed_order([row["dx"] for row in rows], [row["primitive_l2"] for row in rows])
   for row in rows:
      row["observed_order"] = order
   return rows, finest_profile, finest_config


def plot_circular_alfven_validation(rows, profile, config, outpath):
   fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
   group = sorted(rows, key=lambda row: row["dx"], reverse=True)
   dx = np.array([row["dx"] for row in group], dtype=float)
   primitive = np.array([row["primitive_l2"] for row in group], dtype=float)
   magnetic = np.array([row["magnetic_l2"] for row in group], dtype=float)
   velocity = np.array([row["velocity_l2"] for row in group], dtype=float)
   order = group[0]["observed_order"]
   axs[0].loglog(dx, primitive, marker="o", linewidth=2.4, markersize=7.0, label=f"primitive p={order:.2f}")
   axs[0].loglog(dx, magnetic, marker="o", linewidth=2.0, markersize=6.0, label="B")
   axs[0].loglog(dx, velocity, marker="o", linewidth=2.0, markersize=6.0, label="v")
   reference = reference_power_law(dx, primitive, 2.0)
   axs[0].loglog(dx, reference, color="black", linestyle="--", linewidth=2.0, alpha=0.75)
   axs[0].annotate("dx^2", xy=(dx[1], reference[1]), xytext=(5, 0), textcoords="offset points", fontsize=10)
   axs[0].set_xlabel("dx")
   axs[0].set_ylabel("normalized L2 error")
   axs[0].set_title("Convergence")
   axs[0].invert_xaxis()
   axs[0].grid(True, which="major", alpha=0.25)
   axs[0].grid(False, which="minor")
   axs[0].legend(fontsize=9)

   x = profile["x"]
   _, _, _, vy_exact, vz_exact, _, by_exact, bz_exact = circular_alfven_wave_exact(
      x,
      config.final_time,
      config.rho0,
      config.pressure0,
      config.bx0,
      config.amplitude,
      config.wave_number,
      config.direction,
   )
   axs[1].plot(x, profile["by"], marker="o", markersize=3.5, linewidth=1.8, label="By num")
   axs[1].plot(x, by_exact, color=axs[1].lines[-1].get_color(), linestyle=":", linewidth=2.0, label="By exact")
   axs[1].plot(x, profile["bz"], marker="o", markersize=3.5, linewidth=1.8, label="Bz num")
   axs[1].plot(x, bz_exact, color=axs[1].lines[-1].get_color(), linestyle=":", linewidth=2.0, label="Bz exact")
   axs[1].plot(x, profile["vy"], linewidth=1.4, alpha=0.75, label="vy num")
   axs[1].plot(x, vy_exact, color=axs[1].lines[-1].get_color(), linestyle=":", linewidth=1.8, label="vy exact")
   axs[1].plot(x, profile["vz"], linewidth=1.4, alpha=0.75, label="vz num")
   axs[1].plot(x, vz_exact, color=axs[1].lines[-1].get_color(), linestyle=":", linewidth=1.8, label="vz exact")
   axs[1].set_xlabel("x")
   axs[1].set_title(f"Final profile, {config.nx}x{config.ny}, t={config.final_time:g}")
   axs[1].grid(True, which="major", alpha=0.25)
   axs[1].legend(fontsize=8, ncol=2)

   fig.suptitle(
      f"Circular Alfven wave: WENO-Z char/HLLD/RK4/{ct_method_name(config.ct_method)}, CFL=0.9, BC=periodic",
      fontsize=12,
   )
   outpath.parent.mkdir(parents=True, exist_ok=True)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def circular_alfven_matrix_config(nx=64, **kwargs):
   options = {
      "nx": nx,
      "ny": 8,
      "ng": 3,
      "final_time": 1.0,
      "cfl": 0.9,
      "recon": RECON_WENOZ_CHAR,
      "recon_vars": RECON_VARS_PRIMITIVE,
      "limiter": LIMITER_MC,
      "riemann": RIEMANN_HLLD,
      "timestep": TIME_RK4,
      "ct_method": CT_FLUX_TOTH,
      "amplitude": 0.1,
      "wave_number": 1,
   }
   options.update(kwargs)
   return CircularAlfvenWaveConfig(**options)


def circular_alfven_matrix_row(scan, method, config):
   row, Uvec = evolve_circular_alfven_wave_case(config)
   require_finite_row(f"circular_alfven_matrix_{scan}_{method}_{config.nx}", row)
   return {
      "scan": scan,
      "method": method,
      "nx": row["nx"],
      "ny": row["ny"],
      "dx": 1.0 / row["nx"],
      "steps": row["steps"],
      "cfl": config.cfl,
      "final_time": row["time"],
      "recon": recon_name(config.recon),
      "riemann": riemann_name(config.riemann),
      "timestep": timestep_name(config.timestep),
      "ct_method": config.ct_method,
      "ct_method_name": ct_method_name(config.ct_method),
      "primitive_l2": row["primitive_l2"],
      "rho_l2": row["rho_l2"],
      "pressure_l2": row["pressure_l2"],
      "velocity_l2": row["velocity_l2"],
      "magnetic_l2": row["magnetic_l2"],
      "mass_rel_drift": row["mass_rel_drift"],
      "energy_rel_drift": row["energy_rel_drift"],
      "divb_initial": row["divb_initial"],
      "divb_final": row["divb_final"],
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "observed_order": np.nan,
   }, Uvec


def circular_alfven_matrix_rows():
   rows = []
   finest_profile = None
   finest_config = None

   for n in (32, 64, 128):
      config = circular_alfven_matrix_config(nx=n)
      row, Uvec = circular_alfven_matrix_row("convergence", f"{n}x8", config)
      rows.append(row)
      if n == 128:
         finest_profile = circular_alfven_profile(Uvec, config)
         finest_config = config

   convergence_rows = [row for row in rows if row["scan"] == "convergence"]
   order = observed_order([row["dx"] for row in convergence_rows], [row["primitive_l2"] for row in convergence_rows])
   for row in convergence_rows:
      row["observed_order"] = order

   return rows, finest_profile, finest_config


def plot_circular_alfven_matrix(rows, profile, config, scalar_rows, outpath):
   fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)

   convergence = sorted([row for row in rows if row["scan"] == "convergence"], key=lambda row: row["dx"], reverse=True)
   dx = np.array([row["dx"] for row in convergence], dtype=float)
   primitive = np.array([row["primitive_l2"] for row in convergence], dtype=float)
   magnetic = np.array([row["magnetic_l2"] for row in convergence], dtype=float)
   velocity = np.array([row["velocity_l2"] for row in convergence], dtype=float)
   order = convergence[0]["observed_order"]
   axs[0, 0].loglog(dx, primitive, marker="o", linewidth=2.4, markersize=7.0, label=f"primitive p={order:.2f}")
   axs[0, 0].loglog(dx, magnetic, marker="s", linewidth=2.0, markersize=6.0, label="B")
   axs[0, 0].loglog(dx, velocity, marker="^", linewidth=2.0, markersize=6.0, label="v")
   reference = reference_power_law(dx, primitive, 2.0)
   axs[0, 0].loglog(dx, reference, color="black", linestyle="--", linewidth=2.0, alpha=0.75)
   axs[0, 0].annotate("dx^2", xy=(dx[1], reference[1]), xytext=(5, 0), textcoords="offset points", fontsize=10)
   axs[0, 0].set_xlabel("dx")
   axs[0, 0].set_ylabel("normalized L2")
   axs[0, 0].set_title("Baseline convergence")
   axs[0, 0].invert_xaxis()
   axs[0, 0].grid(True, which="major", alpha=0.25)
   axs[0, 0].grid(False, which="minor")
   axs[0, 0].legend(fontsize=9)

   x = profile["x"]
   _, _, _, vy_exact, vz_exact, _, by_exact, bz_exact = circular_alfven_wave_exact(
      x,
      config.final_time,
      config.rho0,
      config.pressure0,
      config.bx0,
      config.amplitude,
      config.wave_number,
      config.direction,
   )
   axs[0, 1].plot(x, profile["by"], marker="o", markersize=3.0, linewidth=1.8, label="By num")
   axs[0, 1].plot(x, by_exact, color=axs[0, 1].lines[-1].get_color(), linestyle=":", linewidth=2.0, label="By exact")
   axs[0, 1].plot(x, profile["bz"], marker="s", markersize=3.0, linewidth=1.8, label="Bz num")
   axs[0, 1].plot(x, bz_exact, color=axs[0, 1].lines[-1].get_color(), linestyle=":", linewidth=2.0, label="Bz exact")
   axs[0, 1].plot(x, profile["vy"], linewidth=1.3, alpha=0.8, label="vy num")
   axs[0, 1].plot(x, vy_exact, color=axs[0, 1].lines[-1].get_color(), linestyle=":", linewidth=1.7, label="vy exact")
   axs[0, 1].plot(x, profile["vz"], linewidth=1.3, alpha=0.8, label="vz num")
   axs[0, 1].plot(x, vz_exact, color=axs[0, 1].lines[-1].get_color(), linestyle=":", linewidth=1.7, label="vz exact")
   axs[0, 1].set_xlabel("x")
   axs[0, 1].set_title(f"Final overlay, {config.nx}x{config.ny}")
   axs[0, 1].grid(True, which="major", alpha=0.25)
   axs[0, 1].legend(fontsize=7.5, ncol=2)

   time_values = np.array([row["time"] for row in scalar_rows], dtype=float)
   markers = marker_kwargs(time_values)
   axs[1, 0].semilogy(time_values, log_floor([row["mass_rel_drift"] for row in scalar_rows]), label="mass", linewidth=2.2, **markers)
   axs[1, 0].semilogy(time_values, log_floor([row["energy_rel_drift"] for row in scalar_rows]), label="energy", linewidth=2.2, **markers)
   axs[1, 0].semilogy(time_values, log_floor([row["momentum_drift"] for row in scalar_rows]), label="momentum", linewidth=2.2, **markers)
   axs[1, 0].set_xlabel("time")
   axs[1, 0].set_ylabel("drift")
   axs[1, 0].set_title("Conserved scalars")
   axs[1, 0].grid(True, which="major", alpha=0.25)
   axs[1, 0].grid(False, which="minor")
   axs[1, 0].legend(fontsize=9)

   axs[1, 1].semilogy(time_values, log_floor([row["divb"] for row in scalar_rows]), linewidth=2.2, color="#d62728", label="divB", **markers)
   axs[1, 1].set_xlabel("time")
   axs[1, 1].set_ylabel("normalized divB")
   axs[1, 1].set_title("Magnetic constraint")
   axs[1, 1].grid(True, which="major", alpha=0.25)
   axs[1, 1].grid(False, which="minor")
   axs[1, 1].legend(fontsize=9)

   avg_rate = np.mean([row["million_cell_updates_per_second"] for row in scalar_rows if row["step"] > 0])
   fig.suptitle(
      "Circular Alfven nonlinear MHD validation: amp=0.1, CFL=0.9, BC=periodic, float64\n"
      f"WENO-Z char/HLLD/RK4/{ct_method_name(config.ct_method)}, {config.nx}x{config.ny}, avg {avg_rate:.2f} M cells/s",
      fontsize=12,
   )
   outpath.parent.mkdir(parents=True, exist_ok=True)
   save_plot(fig, outpath, dpi=170)
   plt.close(fig)


def case_circular_alfven_wave():
   outdir = OUTDIR / "IC" / "circular_alfven"
   rows, profile, config = circular_alfven_rows()
   write_csv(rows, outdir / "circular_alfven_validation.csv")
   plot_circular_alfven_validation(rows, profile, config, outdir / "circular_alfven_validation.png")
   write_plot_index()
   min_order = min(row["observed_order"] for row in rows)
   finest = min(rows, key=lambda row: row["dx"])
   if min_order < 1.8:
      raise AssertionError(f"circular Alfven convergence order too low: {min_order:.3f}")
   if abs(finest["divb_final"]) > 1.0e-12:
      raise AssertionError(f"circular Alfven divB too high: {finest['divb_final']:.3e}")
   return {
      "observed_order": min_order,
      "finest_primitive_l2": finest["primitive_l2"],
      "finest_magnetic_l2": finest["magnetic_l2"],
      "finest_velocity_l2": finest["velocity_l2"],
      "finest_divb": finest["divb_final"],
      "finest_rate_mcells_s": finest["rate_mcells_s"],
   }


def case_circular_alfven_matrix():
   outdir = OUTDIR / "IC" / "circular_alfven"
   rows, profile, config = circular_alfven_matrix_rows()
   scalar_rows, _ = collect_circular_alfven_diagnostics(config)
   write_csv(rows, outdir / "circular_alfven_matrix.csv")
   write_csv(scalar_rows, outdir / "circular_alfven_scalars.csv")
   plot_circular_alfven_matrix(rows, profile, config, scalar_rows, outdir / "circular_alfven_matrix.png")
   write_plot_index()
   convergence = [row for row in rows if row["scan"] == "convergence"]
   baseline_order = min(row["observed_order"] for row in convergence)
   if baseline_order < 1.8:
      raise AssertionError(f"circular Alfven matrix convergence order too low: {baseline_order:.3f}")
   final_scalars = scalar_rows[-1]
   if abs(final_scalars["divb"]) > 1.0e-12:
      raise AssertionError(f"circular Alfven flux-Toth divB too high: {final_scalars['divb']:.3e}")
   worst_error = max(row["primitive_l2"] for row in rows)
   if worst_error > 2.0:
      raise AssertionError(f"circular Alfven matrix has unexpectedly large primitive L2={worst_error:.3e}")
   return {
      "baseline_order": baseline_order,
      "best_primitive_l2": min(row["primitive_l2"] for row in rows),
      "worst_primitive_l2": worst_error,
      "final_mass_rel_drift": final_scalars["mass_rel_drift"],
      "final_energy_rel_drift": final_scalars["energy_rel_drift"],
      "final_divb": final_scalars["divb"],
      "average_rate_mcells_s": np.mean([row["million_cell_updates_per_second"] for row in scalar_rows if row["step"] > 0]),
   }


def case_field_loop_advection():
   outdir = OUTDIR / "IC" / "field_loop"
   outdir.mkdir(parents=True, exist_ok=True)
   final_time = 0.5
   base = {
      "nx": 128,
      "ny": 128,
      "steps": 0,
      "ng": 3,
      "cfl": 0.9,
      "bc": (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC),
      "recon": RECON_WENOZ_CHAR,
      "recon_vars": RECON_VARS_PRIMITIVE,
      "limiter": LIMITER_MC,
      "riemann": RIEMANN_HLLD,
      "timestep": TIME_RK4,
      "ic": 15,
      "diffusion_method": DIFFUSION_NONE,
   }
   configs = {
      "flux-Toth": MHDRunConfig(**base, ct_method=CT_FLUX_TOTH),
      "no CT": MHDRunConfig(**base, ct_method=CT_OFF),
   }
   histories = {}
   final_states = {}
   csv_rows = []
   for label, config in configs.items():
      rows, state = collect_mhd_to_time_diagnostics(config, final_time, nsamples=32)
      for row in rows:
         row["method"] = label
      histories[label] = rows
      final_states[label] = state
      csv_rows.extend(rows)
   initial_state = mhd_ic_initial_state(configs["flux-Toth"])
   write_csv(csv_rows, outdir / "field_loop_advection_scalars.csv")
   plot_field_loop_advection(initial_state, final_states, histories, configs["flux-Toth"], final_time, outdir / "field_loop_advection.png")
   write_plot_index()

   flux_final = histories["flux-Toth"][-1]
   no_ct_final = histories["no CT"][-1]
   if flux_final["divb"] > 1.0e-12:
      raise AssertionError(f"field-loop flux-Toth divB too high: {flux_final['divb']:.3e}")
   if no_ct_final["divb"] <= 10.0 * flux_final["divb"]:
      raise AssertionError(f"field-loop no-CT divB did not separate: CT={flux_final['divb']:.3e}, no CT={no_ct_final['divb']:.3e}")
   return {
      "flux_toth_divb": flux_final["divb"],
      "no_ct_divb": no_ct_final["divb"],
      "flux_toth_magnetic_energy_ratio": flux_final["magnetic_energy"] / histories["flux-Toth"][0]["magnetic_energy"],
      "no_ct_magnetic_energy_ratio": no_ct_final["magnetic_energy"] / histories["no CT"][0]["magnetic_energy"],
      "flux_toth_avg_rate_mcells_s": np.mean([row["million_cell_updates_per_second"] for row in histories["flux-Toth"] if row["step"] > 0]),
   }


def mhd_profile_1d(Uvec, config):
   fields = primitive_field_arrays(Uvec, config.nx, config.ny, config.ng, config.gamma)
   return {
      "rho": np.mean(fields["rho"], axis=1),
      "P": np.mean(fields["P"], axis=1),
      "vx": np.mean(fields["vx"], axis=1),
      "vy": np.mean(fields["vy"], axis=1),
      "Bx": np.mean(fields["Bx"], axis=1),
      "By": np.mean(fields["By"], axis=1),
      "Bz": np.mean(fields["Bz"], axis=1),
   }


def run_mhd_discontinuity_method(config, final_time, output_times):
   validate_common_numeric_options(config, mhd=True)
   bc = validate_bc(config.bc)
   configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny

   warm = mhd_ic_initial_state(config)
   warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt = make_workspace(
      warm, config.nx, config.ny
   )
   evolve_to_time(
      warm, warm_new, warm_tmp, warm_rhs, warm_fx, warm_fy, warm_ez, warm_xl, warm_xr, warm_yb, warm_yt,
      dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
      config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
      2.0 * dx, 16, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )

   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   scalars0 = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
   rows = []
   method = riemann_name(config.riemann).upper()
   append_scalar_diagnostics_row(rows, 0, 0.0, 0.0, 0.0, 0.0, scalars0, scalars0)
   rows[-1]["method"] = method

   time_value = 0.0
   total_steps = 0
   total_elapsed = 0.0
   dt_min = np.inf
   dt_max = 0.0
   for target_time in output_times[1:]:
      delta = float(target_time - time_value)
      if delta <= 1.0e-14:
         continue
      start = time.time()
      reached, steps, segment_dt_min, segment_dt_max = evolve_to_time(
         Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT,
         dx, dy, config.cfl, config.gamma, config.ng, bc[0], bc[1], bc[2], bc[3],
         config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep,
         delta, 20000, config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
      )
      elapsed = time.time() - start
      time_value += reached
      total_steps += steps
      total_elapsed += elapsed
      dt_min = min(dt_min, segment_dt_min)
      dt_max = max(dt_max, segment_dt_max)
      scalars = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
      append_scalar_diagnostics_row(
         rows,
         total_steps,
         time_value,
         reached / max(steps, 1),
         elapsed,
         float(config.nx * config.ny * steps),
         scalars,
         scalars0,
      )
      rows[-1]["method"] = method

   if abs(time_value - final_time) > 1.0e-12:
      raise AssertionError(f"{method} MHD discontinuity ended at t={time_value}, expected {final_time}")
   profile = mhd_profile_1d(Uvec, config)
   scalars = state_scalar_diagnostics(Uvec, config.nx, config.ny, config.ng, dx, dy, config.gamma)
   speed = throughput_metrics(config.nx, config.ny, total_steps, config.timestep, total_elapsed)
   summary = {
      "method": method,
      "ic": hydro_initial_condition_name(config.ic),
      "ic_id": config.ic,
      "nx": config.nx,
      "ny": config.ny,
      "threads": config.threads,
      "numba_threads": configure_numba_threads(config.threads),
      "cfl": config.cfl,
      "final_time": time_value,
      "steps": total_steps,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": total_elapsed,
      **speed,
      "mass_rel_drift": relative_drift(scalars["mass"], scalars0["mass"]),
      "energy_rel_drift": relative_drift(scalars["energy"], scalars0["energy"]),
      "momentum_drift": momentum_drift(scalars, scalars0),
      "divb_initial": scalars0["divb"],
      "divb_final": scalars["divb"],
      "min_rho": scalars["min_rho"],
      "min_pressure": scalars["min_pressure"],
   }
   return profile, rows, summary


def plot_mhd_riemann_discontinuity_profiles(x, profiles, summaries, outpath, title):
   fields = (("rho", "rho"), ("P", "P"), ("vx", "vx"), ("vy", "vy"), ("By", "By"), ("Bz", "Bz"))
   colors = {"LLF": "#2878b5", "HLL": "#d66b2a", "HLLE": "#269a3e", "HLLD": "#d62728"}
   markers = {"LLF": "o", "HLL": "s", "HLLE": "^", "HLLD": "D"}
   fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True, constrained_layout=True)
   for ax, (label, key) in zip(axs.ravel(), fields):
      for summary in summaries:
         method = summary["method"]
         ax.plot(
            x,
            profiles[method][key],
            linewidth=2.0,
            marker=markers[method],
            markersize=3.2,
            markevery=16,
            color=colors[method],
            label=method,
         )
      ax.set_title(label)
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
   for ax in axs[-1, :]:
      ax.set_xlabel("x")
   axs[0, 0].legend(loc="best", fontsize=9)
   fig.suptitle(title, fontsize=12)
   save_plot(fig, outpath, dpi=180)
   plt.close(fig)


def plot_mhd_riemann_discontinuity_differences(x, profiles, summaries, outpath, title):
   fields = (("rho", "rho"), ("P", "P"), ("vx", "vx"), ("vy", "vy"), ("By", "By"), ("Bz", "Bz"))
   colors = {"LLF": "#2878b5", "HLL": "#d66b2a", "HLLE": "#269a3e"}
   markers = {"LLF": "o", "HLL": "s", "HLLE": "^"}
   baseline = profiles["HLLD"]
   fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True, constrained_layout=True)
   for ax, (label, key) in zip(axs.ravel(), fields):
      for summary in summaries:
         method = summary["method"]
         if method == "HLLD":
            continue
         ax.plot(
            x,
            profiles[method][key] - baseline[key],
            linewidth=2.0,
            marker=markers[method],
            markersize=3.0,
            markevery=16,
            color=colors[method],
            label=f"{method} - HLLD",
         )
      ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
      ax.set_title(label)
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
   for ax in axs[-1, :]:
      ax.set_xlabel("x")
   axs[0, 0].legend(loc="best", fontsize=9)
   fig.suptitle(title, fontsize=12)
   save_plot(fig, outpath, dpi=180)
   plt.close(fig)


def plot_mhd_riemann_discontinuity_scalars(rows, summaries, outpath, title):
   methods = [row["method"] for row in summaries]
   colors = {"LLF": "#2878b5", "HLL": "#d66b2a", "HLLE": "#269a3e", "HLLD": "#d62728"}
   markers = {"LLF": "o", "HLL": "s", "HLLE": "^", "HLLD": "D"}
   fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True)
   axs = axs.ravel()
   for method in methods:
      group = [row for row in rows if row["method"] == method]
      time_values = np.array([row["time"] for row in group], dtype=float)
      kwargs = {
         "linewidth": 2.0,
         "marker": markers[method],
         "markersize": 3.8,
         "color": colors[method],
         "label": method,
      }
      axs[0].semilogy(time_values, log_floor([row["mass_rel_drift"] for row in group]), **kwargs)
      axs[1].semilogy(time_values, log_floor([row["energy_rel_drift"] for row in group]), **kwargs)
      axs[2].semilogy(time_values, log_floor([row["momentum_drift"] for row in group]), **kwargs)
      axs[3].semilogy(time_values, log_floor([row["divb"] for row in group]), **kwargs)
      axs[4].plot(time_values, [row["min_pressure"] for row in group], **kwargs)
      active = [row for row in group if row["step"] > 0]
      axs[5].plot(
         [row["time"] for row in active],
         [row["million_cell_updates_per_second"] for row in active],
         **kwargs,
      )

   axs[0].set_title("Mass Drift")
   axs[1].set_title("Energy Drift")
   axs[2].set_title("Momentum Change")
   axs[3].set_title("Magnetic Constraint")
   axs[3].set_ylabel("normalized divB")
   axs[4].set_title("Minimum Pressure")
   axs[5].set_title("Throughput")
   axs[5].set_ylabel("M cells/s")
   for ax in axs:
      ax.set_xlabel("time")
      ax.grid(True, which="major", alpha=0.25)
      ax.grid(False, which="minor")
   axs[0].legend(loc="best", fontsize=9)
   fig.suptitle(title, fontsize=12)
   save_plot(fig, outpath, dpi=180)
   plt.close(fig)


def case_mhd_riemann_discontinuity_compare():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   outdir = OUTDIR / "IC" / "mhd_discontinuity"
   outdir.mkdir(parents=True, exist_ok=True)
   scan_config = MHD_METHOD_SCAN["discontinuity"]
   nx = max(scan_config["nx_values"])
   final_time = scan_config["final_time"]
   output_times = np.linspace(0.0, final_time, 21)
   profiles = {}
   scalar_rows = []
   summaries = []
   for _, solver in scan_config["riemann_methods"]:
      config = make_mhd_discontinuity_config(scan_config, nx, riemann=solver, recon=scan_config["riemann_recon"])
      profile, rows, summary = run_mhd_discontinuity_method(config, final_time, output_times)
      require_finite_row(summary["method"], summary)
      if summary["min_rho"] <= 0.0 or summary["min_pressure"] <= 0.0:
         raise AssertionError(f"{summary['method']} produced non-positive rho/P: {summary}")
      profiles[summary["method"]] = profile
      scalar_rows.extend(rows)
      summaries.append(summary)

   x = (np.arange(nx) + 0.5) / nx
   profile_rows = []
   for i, xi in enumerate(x):
      row = {"x": xi}
      for summary in summaries:
         method = summary["method"].lower()
         for field in ("rho", "P", "vx", "vy", "Bx", "By", "Bz"):
            row[f"{field}_{method}"] = profiles[summary["method"]][field][i]
      profile_rows.append(row)
   write_csv(profile_rows, outdir / "balsara1a_riemann_profiles.csv")
   write_csv(scalar_rows, outdir / "balsara1a_riemann_scalars.csv")
   write_csv(summaries, outdir / "balsara1a_riemann_summary.csv")
   title = (
      f"MHD discontinuity: IC={hydro_initial_condition_name(scan_config['ic'])} ({scan_config['ic']}), "
      f"WENO-Z char {recon_vars_name(scan_config['recon_vars'])}/"
      f"{limiter_name(scan_config['limiter']).upper()}/{timestep_name(scan_config['timestep']).upper()}/"
      f"{ct_method_name(scan_config['ct_method'])}\n"
      f"BC={bc_tuple_name(scan_config['bc'])}, CFL={scan_config['cfl']:g}, t={final_time:.3f}, "
      f"{nx}x{scan_config['ny']}, threads={DEFAULT_THREADS}"
   )
   plot_mhd_riemann_discontinuity_profiles(
      x,
      profiles,
      summaries,
      outdir / "balsara1a_riemann_profiles.png",
      title,
   )
   plot_mhd_riemann_discontinuity_differences(
      x,
      profiles,
      summaries,
      outdir / "balsara1a_riemann_differences.png",
      f"{title}\nlocal differences from HLLD",
   )
   plot_mhd_riemann_discontinuity_scalars(
      scalar_rows,
      summaries,
      outdir / "balsara1a_riemann_scalars.png",
      title,
   )
   write_plot_index()
   return {
      "ic": hydro_initial_condition_name(scan_config["ic"]),
      "cfl": scan_config["cfl"],
      "final_time": final_time,
      "methods": [row["method"] for row in summaries],
      "rates_mcells_s": {row["method"]: row["million_cell_updates_per_second"] for row in summaries},
      "divb_final": {row["method"]: row["divb_final"] for row in summaries},
      "min_pressure": {row["method"]: row["min_pressure"] for row in summaries},
   }


def case_resistive_mhd_exact_plot():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   config = ResistiveWaveConfig(
      nx=128,
      ny=8,
      ng=2,
      final_time=0.2,
      steps=256,
      amplitude=0.1,
      resistivity=1.0e-2,
      wave_number=2,
      diffusion_tolerance=1.0e-10,
      diffusion_max_steps=300,
   )
   row, final_state, history = collect_resistive_wave_history(config)
   require_finite_row("resistive_mhd_exact", row)
   x, by_num, bz_num, pressure_num, by_exact, bz_exact, pressure_exact = resistive_wave_profiles(final_state, config)
   by_l2 = float(np.sqrt(np.mean((by_num - by_exact) ** 2)) / config.amplitude)
   bz_l2 = float(np.sqrt(np.mean((bz_num - bz_exact) ** 2)) / config.amplitude)
   pressure_l2 = float(np.sqrt(np.mean((pressure_num - pressure_exact) ** 2)) / config.pressure0)
   if max(by_l2, bz_l2) > 5.0e-3:
      raise AssertionError(f"resistive MHD exact magnetic errors too large: By={by_l2:.3e}, Bz={bz_l2:.3e}")
   if pressure_l2 > 1.0e-4:
      raise AssertionError(f"resistive MHD exact pressure error too large: P={pressure_l2:.3e}")
   max_amplitude_error = max(item["relative_error"] for item in history)
   if max_amplitude_error > 5.0e-4:
      raise AssertionError(f"resistive MHD amplitude decay error too large: {max_amplitude_error:.3e}")

   rows = [
      {
         "x": x[i],
         "by_num": by_num[i],
         "by_exact": by_exact[i],
         "bz_num": bz_num[i],
         "bz_exact": bz_exact[i],
         "pressure_num": pressure_num[i],
         "pressure_exact": pressure_exact[i],
      }
      for i in range(config.nx)
   ]
   write_csv(rows, OUTDIR / "resistive_mhd_exact.csv")
   write_csv(history, OUTDIR / "resistive_mhd_amplitude.csv")
   plot_resistive_mhd_exact(config, row, final_state, OUTDIR / "resistive_mhd_exact.png")
   plot_resistive_mhd_amplitude(config, row, history, OUTDIR / "resistive_mhd_amplitude.png")
   plot_resistive_mhd_amplitude(config, row, history, OUTDIR / "resistive_mhd_amplitude_eta_m.png")
   write_plot_index()
   return {
      "by_l2": by_l2,
      "bz_l2": bz_l2,
      "pressure_l2": pressure_l2,
      "amplitude_rel_error": history[-1]["relative_error"],
      "max_amplitude_rel_error": max_amplitude_error,
      "rate": row["million_cell_updates_per_second"],
      "diffusion_error": row["diffusion_error"],
      "diffusion_steps_max": row["diffusion_steps_max"],
      "diffusion_steps_total": row["diffusion_steps_total"],
   }


def case_resistive_diffusion_timestep_benchmark():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   config = ResistiveWaveConfig(
      nx=64,
      ny=64,
      ng=2,
      final_time=0.1,
      steps=16,
      amplitude=0.1,
      resistivity=5.0e-2,
      wave_number=2,
      diffusion_tolerance=1.0e-10,
      diffusion_max_steps=500,
   )
   explicit_steps, explicit_dt, explicit_dt_limit = explicit_diffusion_step_budget(config, diffusion_cfl=0.45)
   explicit_config = ResistiveWaveConfig(**{
      **config.__dict__,
      "steps": explicit_steps,
      "diffusion_method": DIFFUSION_EXPLICIT,
   })
   implicit_config = ResistiveWaveConfig(**{
      **config.__dict__,
      "diffusion_method": DIFFUSION_IMPLICIT,
   })
   explicit_row, explicit_state = evolve_resistive_wave_case(explicit_config)
   explicit_row["explicit_dt_limit"] = explicit_dt_limit
   explicit_row["dt"] = explicit_dt
   implicit_row, implicit_state = evolve_resistive_wave_case(implicit_config)
   rows = [
      resistive_diffusion_benchmark_row("explicit FTCS", explicit_row, explicit_state, explicit_config, explicit_steps),
      resistive_diffusion_benchmark_row("implicit BE SOR", implicit_row, implicit_state, implicit_config, explicit_steps),
   ]
   for row in rows:
      require_finite_row(row["method"], row)
      if row["amplitude_relative_error"] > 5.0e-2:
         raise AssertionError(
            f"{row['method']} resistive amplitude error too large: "
            f"{row['amplitude_relative_error']:.3e}"
         )
   write_csv(rows, OUTDIR / "resistive_mhd_diffusion_timestep.csv")
   plot_resistive_diffusion_timestep_benchmark(rows, OUTDIR / "resistive_mhd_diffusion_timestep.png")
   write_plot_index()
   return {
      "explicit_steps": rows[0]["steps"],
      "implicit_steps": rows[1]["steps"],
      "timestep_gain": rows[1]["timestep_gain_vs_explicit"],
      "explicit_seconds": rows[0]["elapsed"],
      "implicit_seconds": rows[1]["elapsed"],
      "runtime_ratio_explicit_over_implicit": rows[0]["elapsed"] / rows[1]["elapsed"],
      "implicit_sor_sweeps": rows[1]["diffusion_steps_total"],
      "explicit_amplitude_error": rows[0]["amplitude_relative_error"],
      "implicit_amplitude_error": rows[1]["amplitude_relative_error"],
   }


def case_default_threads():
   row = run_hydro_case(HydroRunConfig(nx=64, ny=64, steps=8))
   if row["threads"] != DEFAULT_THREADS:
      raise AssertionError(f"default threads={row['threads']}, expected {DEFAULT_THREADS}")
   if row["numba_threads"] != DEFAULT_THREADS:
      raise AssertionError(f"runtime threads={row['numba_threads']}, expected {DEFAULT_THREADS}")
   if row["dtype"] != "float64":
      raise AssertionError(f"default dtype={row['dtype']}, expected float64")
   if row["recon"] != RECON_PLM or row["recon_vars"] != RECON_VARS_PRIMITIVE or row["limiter"] != LIMITER_MC:
      raise AssertionError(f"unexpected production reconstruction defaults: {row}")
   if row["riemann"] != RIEMANN_HLLC or row["timestep"] != TIME_SSPRK3:
      raise AssertionError(f"unexpected production solver defaults: {row}")
   require_close_to_roundoff("mass_rel_drift", row["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("energy_rel_drift", row["energy_rel_drift"], 1e-12)
   return {"threads": row["threads"], "dtype": row["dtype"], "rate": row["million_cell_updates_per_second"]}


def case_entropy_wave_accuracy():
   row = run_entropy_wave_case(EntropyWaveConfig(nx=128, ny=128, final_time=0.1))
   require_close_to_roundoff("mass_rel_drift", row["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("energy_rel_drift", row["energy_rel_drift"], 1e-12)
   if row["rho_l2"] > 2.0e-3:
      raise AssertionError(f"entropy-wave rho_l2={row['rho_l2']:.3e} exceeds 2e-3")
   if row["pressure_l2"] > 1e-12 or row["vx_l2"] > 1e-12 or row["vy_l2"] > 1e-12:
      raise AssertionError(
         f"entropy-wave constant fields drifted: pressure={row['pressure_l2']:.3e}, "
         f"vx={row['vx_l2']:.3e}, vy={row['vy_l2']:.3e}"
      )
   return {key: row[key] for key in ("rho_l2", "pressure_l2", "vx_l2", "vy_l2", "million_cell_updates_per_second")}


def case_entropy_wave_convergence():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = []
   for n in (32, 64, 128):
      row = run_entropy_wave_case(EntropyWaveConfig(nx=n, ny=n, final_time=0.1))
      rows.append(row)
   rates = [rows[i]["rho_l2"] / rows[i + 1]["rho_l2"] for i in range(len(rows) - 1)]
   if min(rates) < 1.8:
      raise AssertionError(f"entropy-wave convergence ratios too small: {rates}")
   write_csv(rows, OUTDIR / "entropy_wave_convergence.csv")
   return {"rho_l2": [row["rho_l2"] for row in rows], "ratios": rates}


def case_acoustic_wave_accuracy():
   row = run_acoustic_wave_case(AcousticWaveConfig(nx=128, ny=128, final_time=0.05))
   require_close_to_roundoff("mass_rel_drift", row["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("energy_rel_drift", row["energy_rel_drift"], 1e-12)
   if row["rho_l2"] > 5.0e-3 or row["pressure_l2"] > 5.0e-3:
      raise AssertionError(
         f"acoustic-wave density/pressure errors too large: rho={row['rho_l2']:.3e}, "
         f"pressure={row['pressure_l2']:.3e}"
      )
   if np.hypot(row["vx_l2"], row["vy_l2"]) > 2.0e-3:
      raise AssertionError(f"acoustic-wave velocity error too large: vx={row['vx_l2']:.3e}, vy={row['vy_l2']:.3e}")
   return {key: row[key] for key in ("rho_l2", "pressure_l2", "vx_l2", "vy_l2", "million_cell_updates_per_second")}


def case_acoustic_wave_convergence():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = []
   for n in (32, 64, 128):
      row = run_acoustic_wave_case(AcousticWaveConfig(nx=n, ny=n, final_time=0.05))
      rows.append(row)
   rates = [rows[i]["rho_l2"] / rows[i + 1]["rho_l2"] for i in range(len(rows) - 1)]
   if min(rates) < 2.5:
      raise AssertionError(f"acoustic-wave convergence ratios too small: {rates}")
   write_csv(rows, OUTDIR / "acoustic_wave_convergence.csv")
   return {"rho_l2": [row["rho_l2"] for row in rows], "ratios": rates}


def case_riemann_solvers():
   runs = {
      "llf": run_entropy_wave_case(EntropyWaveConfig(nx=128, ny=128, final_time=0.1, riemann=RIEMANN_LLF)),
      "hll": run_entropy_wave_case(EntropyWaveConfig(nx=128, ny=128, final_time=0.1, riemann=RIEMANN_HLL)),
      "hlle": run_entropy_wave_case(EntropyWaveConfig(nx=128, ny=128, final_time=0.1, riemann=RIEMANN_HLLE)),
      "hllc": run_entropy_wave_case(EntropyWaveConfig(nx=128, ny=128, final_time=0.1, riemann=RIEMANN_HLLC)),
   }
   for name, row in runs.items():
      require_finite_row(name, row)
      require_close_to_roundoff(f"{name}_mass_rel_drift", row["mass_rel_drift"], 1e-12)
      require_close_to_roundoff(f"{name}_energy_rel_drift", row["energy_rel_drift"], 1e-12)
   return {f"{name}_rho_l2": row["rho_l2"] for name, row in runs.items()}


def case_reconstruction_variables():
   primitive = run_acoustic_wave_case(AcousticWaveConfig(nx=64, ny=64, final_time=0.05, recon_vars=RECON_VARS_PRIMITIVE))
   conserved = run_acoustic_wave_case(AcousticWaveConfig(nx=64, ny=64, final_time=0.05, recon_vars=RECON_VARS_CONSERVED))
   for name, row in (("primitive", primitive), ("conserved", conserved)):
      require_close_to_roundoff(f"{name}_mass_rel_drift", row["mass_rel_drift"], 1e-12)
      require_close_to_roundoff(f"{name}_energy_rel_drift", row["energy_rel_drift"], 1e-12)
   if primitive["recon_vars"] != RECON_VARS_PRIMITIVE or conserved["recon_vars"] != RECON_VARS_CONSERVED:
      raise AssertionError("reconstruction variable switch did not propagate to output rows")
   return {"primitive_rho_l2": primitive["rho_l2"], "conserved_rho_l2": conserved["rho_l2"]}


def case_wenoz_reconstruction():
   row = run_entropy_wave_case(EntropyWaveConfig(nx=64, ny=64, ng=3, final_time=0.05, recon=RECON_WENOZ))
   require_close_to_roundoff("wenoz_mass_rel_drift", row["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("wenoz_energy_rel_drift", row["energy_rel_drift"], 1e-12)
   if row["recon"] != RECON_WENOZ:
      raise AssertionError("WENO-Z reconstruction switch did not propagate to output row")
   if row["rho_l2"] > 5.0e-4:
      raise AssertionError(f"WENO-Z entropy-wave rho_l2={row['rho_l2']:.3e} exceeds 5e-4")
   return {"rho_l2": row["rho_l2"], "rate": row["million_cell_updates_per_second"]}


def case_method_scans():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = method_scan_rows()
   for row in rows:
      require_finite_row(f"{row['scan']}_{row['method']}", row)
      require_optional_close_to_roundoff(f"{row['scan']}_{row['method']}_mass_rel_drift", row["mass_rel_drift"], 1e-12)
      require_optional_close_to_roundoff(f"{row['scan']}_{row['method']}_energy_rel_drift", row["energy_rel_drift"], 1e-12)
   write_csv(rows, SCANDIR / "hydro_method_scans.csv")
   plot_method_scans(rows)
   summary = {}
   expected_time_orders = HYDRO_METHOD_SCAN["time"]["expected_order_ranges"]
   for scan in sorted({row["scan"] for row in rows}):
      for method in sorted({row["method"] for row in rows if row["scan"] == scan}):
         group = sorted([row for row in rows if row["scan"] == scan and row["method"] == method], key=lambda row: row["x"])
         finest = group[0]
         if scan == "time_integrator" and method in expected_time_orders:
            low, high = expected_time_orders[method]
            order = finest["observed_order"]
            if order < low or order > high:
               raise AssertionError(f"{scan}:{method} observed order {order:.3f} outside [{low:.1f}, {high:.1f}]")
         summary[f"{scan}:{method}"] = {
            "test": finest["test"],
            "x_label": finest["x_label"],
            "smallest_x": finest["x"],
            "slope_label": finest.get("slope_label", "p"),
            "slope": finest["observed_order"],
            "error_l2": finest["error_l2"],
            "rate_mcells_s": finest["rate_mcells_s"],
         }
   return summary


def case_hydro_direct_reconstruction():
   rows = hydro_direct_reconstruction_rows()
   write_csv(rows, SCANDIR / "hydro_direct_reconstruction_convergence.csv")
   plot_hydro_direct_reconstruction(rows)
   summary = {"average_reconstruction_eval_s": sum(row["elapsed"] for row in rows) / len(rows)}
   for method in sorted({row["method"] for row in rows}):
      group = [row for row in rows if row["method"] == method]
      summary[f"{method}_fit_order"] = group[0]["observed_order"]
      local = [row["local_order_to_next"] for row in sorted(group, key=lambda row: row["x"], reverse=True)]
      summary[f"{method}_local_orders"] = [value for value in local if np.isfinite(value)]
   return summary


def case_hydro_spatial_reconstruction():
   rows = hydro_spatial_reconstruction_rows()
   write_csv(rows, SCANDIR / "hydro_spatial_reconstruction_convergence.csv")
   plot_hydro_spatial_reconstruction(rows)
   summary = {}
   for method in sorted({row["method"] for row in rows}):
      group = [row for row in rows if row["method"] == method]
      summary[f"{method}_fit_order"] = group[0]["observed_order"]
      local = [row["local_order_to_next"] for row in sorted(group, key=lambda row: row["x"], reverse=True)]
      summary[f"{method}_local_orders"] = [value for value in local if np.isfinite(value)]
      finest = min(group, key=lambda row: row["dx"])
      summary[f"{method}_finest_rate_mcells_s"] = finest["rate_mcells_s"]
   summary["average_rate_mcells_s"] = sum(row["rate_mcells_s"] for row in rows) / len(rows)
   return summary


def case_mhd_method_scans():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = mhd_method_scan_rows()
   for row in rows:
      require_finite_row(f"{row['scan']}_{row['method']}", row)
      if row["scan"] == "time_integrator":
         require_optional_close_to_roundoff(f"{row['scan']}_{row['method']}_mass_rel_drift", row["mass_rel_drift"], 1e-10)
         require_optional_close_to_roundoff(f"{row['scan']}_{row['method']}_energy_rel_drift", row["energy_rel_drift"], 1e-10)
      if has_numeric_value(row["divb_final"]) and abs(row["divb_final"]) > 1.0e-8:
         raise AssertionError(f"{row['scan']}:{row['method']} divB={row['divb_final']:.3e} exceeds 1e-8")
   write_csv(rows, SCANDIR / "mhd_method_scans.csv")
   plot_mhd_method_scans(rows)
   summary = {}
   expected_time_orders = MHD_METHOD_SCAN["time"]["expected_order_ranges"]
   for scan in sorted({row["scan"] for row in rows}):
      for method in sorted({row["method"] for row in rows if row["scan"] == scan}):
         group = sorted([row for row in rows if row["scan"] == scan and row["method"] == method], key=lambda row: row["x"])
         finest = group[0]
         if scan == "time_integrator" and method in expected_time_orders:
            low, high = expected_time_orders[method]
            order = finest["observed_order"]
            if order < low or order > high:
               raise AssertionError(f"{scan}:{method} observed order {order:.3f} outside [{low:.1f}, {high:.1f}]")
         summary[f"{scan}:{method}"] = {
            "test": finest["test"],
            "x_label": finest["x_label"],
            "smallest_x": finest["x"],
            "slope_label": finest.get("slope_label", "p"),
            "slope": finest["observed_order"],
            "error_l2": finest["error_l2"],
            "divb_final": finest["divb_final"],
            "rate_mcells_s": finest["rate_mcells_s"],
         }
   return summary


def case_memory_shapes():
   hydro = run_hydro_case(HydroRunConfig(nx=16, ny=16, steps=2))
   mhd = run_mhd_case(MHDRunConfig(nx=16, ny=16, steps=2, recon=RECON_PLM, cfl=0.1))
   if hydro["nvar"] != 5:
      raise AssertionError(f"hydro backend allocated nvar={hydro['nvar']}, expected 5")
   if mhd["nvar"] != 8:
      raise AssertionError(f"MHD backend allocated nvar={mhd['nvar']}, expected 8")
   return {"hydro_nvar": hydro["nvar"], "mhd_nvar": mhd["nvar"]}


def case_mhd_wenoz_ct():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   ct_config = MHDRunConfig(nx=32, ny=32, steps=32, cfl=0.08, ct_method=CT_FLUX_TOTH, diffusion_method=DIFFUSION_NONE)
   no_ct_config = MHDRunConfig(nx=32, ny=32, steps=32, cfl=0.08, ct_method=CT_OFF, diffusion_method=DIFFUSION_NONE)
   ct_rows, _ = collect_mhd_diagnostics(ct_config)
   no_ct_rows, _ = collect_mhd_diagnostics(no_ct_config)
   for row in ct_rows:
      row["method"] = "flux-Toth"
   for row in no_ct_rows:
      row["method"] = "no CT"
   for name, rows in (("flux_toth", ct_rows), ("no_ct", no_ct_rows)):
      final = rows[-1]
      require_finite_row(name, final)
      require_close_to_roundoff(f"{name}_mass_rel_drift", final["mass_rel_drift"], 1e-12)
   write_csv(ct_rows + no_ct_rows, OUTDIR / "mhd_divb_smoke.csv")
   plot_ct_divb_history(
      (("flux-Toth", ct_rows), ("no CT", no_ct_rows)),
      OUTDIR / "mhd_divb_smoke.png",
      config_label(ct_config).replace("flux-Toth", "flux-Toth vs no CT"),
   )
   write_plot_index()
   return {
      "ct_divb": ct_rows[-1]["divb"],
      "no_ct_divb": no_ct_rows[-1]["divb"],
      "ct_avg_rate_mcells_s": np.mean([row["million_cell_updates_per_second"] for row in ct_rows if row["step"] > 0]),
   }


def case_mhd_wenoz_char_smoke():
   row = run_mhd_case(
      MHDRunConfig(
         nx=16,
         ny=16,
         steps=2,
         ng=3,
         recon=RECON_WENOZ_CHAR,
         recon_vars=RECON_VARS_PRIMITIVE,
         riemann=RIEMANN_LLF,
         timestep=TIME_SSPRK3,
         ct_method=CT_FLUX_TOTH,
         cfl=0.2,
         diffusion_method=DIFFUSION_NONE,
      )
   )
   require_finite_row("mhd_wenoz_char", row)
   require_close_to_roundoff("mhd_wenoz_char_mass_rel_drift", row["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("mhd_wenoz_char_energy_rel_drift", row["energy_rel_drift"], 1e-12)
   if row["recon"] != RECON_WENOZ_CHAR:
      raise AssertionError(f"MHD characteristic WENO-Z switch did not propagate: {row}")
   return {
      "divb": row["divb_final"],
      "rate_mcells_s": row["million_cell_updates_per_second"],
      "mass_rel_drift": row["mass_rel_drift"],
      "energy_rel_drift": row["energy_rel_drift"],
   }


def case_mhd_diffusion():
   rows = {}
   for name, method in (("explicit", DIFFUSION_EXPLICIT), ("implicit", DIFFUSION_IMPLICIT)):
      row = run_mhd_case(
         MHDRunConfig(
            nx=24,
            ny=24,
            steps=3,
            recon=RECON_PLM,
            cfl=0.08,
            diffusion_method=method,
            ct_method=CT_FLUX_TOTH,
            viscosity=0.0,
            resistivity=1.0e-4,
            diffusion_tolerance=1.0e-9,
            diffusion_max_steps=300,
         )
      )
      require_finite_row(f"mhd_diffusion_{name}", row)
      if row["diffusion_method"] != method:
         raise AssertionError(f"MHD diffusion method did not propagate for {name}: {row}")
      require_close_to_roundoff(f"mhd_diffusion_{name}_mass_rel_drift", row["mass_rel_drift"], 1e-12)
      require_close_to_roundoff(f"mhd_diffusion_{name}_energy_rel_drift", row["energy_rel_drift"], 1e-12)
      rows[name] = row
   return {
      "explicit_rate": rows["explicit"]["million_cell_updates_per_second"],
      "implicit_rate": rows["implicit"]["million_cell_updates_per_second"],
      "explicit_divb": rows["explicit"]["divb_final"],
      "implicit_divb": rows["implicit"]["divb_final"],
      "explicit_dt_min": rows["explicit"]["dt_min"],
      "implicit_dt_min": rows["implicit"]["dt_min"],
      "implicit_diffusion_error": rows["implicit"]["diffusion_error"],
      "implicit_diffusion_steps_max": rows["implicit"]["diffusion_steps_max"],
      "implicit_diffusion_steps_total": rows["implicit"]["diffusion_steps_total"],
   }


def case_mhd_diffusion_diagnostics():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   config = MHDRunConfig(
      nx=48,
      ny=48,
      steps=24,
      recon=RECON_PLM,
      cfl=0.08,
      diffusion_method=DIFFUSION_IMPLICIT,
      ct_method=CT_FLUX_TOTH,
      viscosity=0.0,
      resistivity=1.0e-4,
      diffusion_tolerance=1.0e-9,
      diffusion_max_steps=300,
   )
   rows, final_state = collect_mhd_diagnostics(config)
   for row in rows:
      require_finite_row(f"mhd_diffusion_diagnostics_step_{row['step']}", row)
   active_rows = [row for row in rows if row["step"] > 0]
   max_diffusion_error = max(row["diffusion_error"] for row in active_rows)
   max_diffusion_steps = max(row["diffusion_steps"] for row in active_rows)
   max_mass_drift = max(row["mass_rel_drift"] for row in rows)
   max_energy_drift = max(row["energy_rel_drift"] for row in rows)
   max_momentum_drift = max(row["momentum_drift"] for row in rows)
   if max_diffusion_error > 1.0e-8:
      raise AssertionError(f"implicit diffusion relative update too large: {max_diffusion_error:.3e}")
   require_close_to_roundoff("mhd_diffusion_diagnostics_mass", max_mass_drift, 1e-12)
   require_close_to_roundoff("mhd_diffusion_diagnostics_energy", max_energy_drift, 1e-12)
   write_csv(rows, OUTDIR / "mhd_diffusion_diagnostics.csv")
   speed_row = run_mhd_case(config)
   plot_scalar_diagnostics(
      rows,
      OUTDIR / "mhd_diffusion_scalars.png",
      config_label(config),
      mhd=True,
   )
   plot_diffusion_diagnostics(
      rows,
      OUTDIR / "mhd_diffusion_error.png",
      config_label(config),
   )
   plot_final_heatmap(
      final_state,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      OUTDIR / "mhd_diffusion_heatmap.png",
      f"{config_label(config)}, t={rows[-1]['time']:.4f}",
   )
   write_plot_index()
   return {
      "steps": len(active_rows),
      "max_diffusion_error": max_diffusion_error,
      "max_diffusion_steps": max_diffusion_steps,
      "max_mass_drift": max_mass_drift,
      "max_energy_drift": max_energy_drift,
      "max_momentum_drift": max_momentum_drift,
      "rate": speed_row["million_cell_updates_per_second"],
   }


def case_implicit_diffusion_slowdown():
   base = MHDRunConfig(
      nx=128,
      ny=128,
      steps=32,
      threads=DEFAULT_THREADS,
      ng=3,
      recon=RECON_PWC,
      recon_vars=RECON_VARS_CONSERVED,
      riemann=RIEMANN_LLF,
      timestep=TIME_EULER,
      cfl=0.08,
      ct_method=CT_FLUX_TOTH,
      diffusion_method=DIFFUSION_NONE,
      viscosity=0.0,
      resistivity=1.0e-4,
   )
   no_diffusion = run_mhd_case(base)
   diffusion = run_mhd_case(
      MHDRunConfig(
         **{
            **base.__dict__,
            "diffusion_method": DIFFUSION_IMPLICIT,
            "diffusion_tolerance": 1.0e-10,
            "diffusion_max_steps": 300,
         }
      )
   )
   slowdown = no_diffusion["million_cell_updates_per_second"] / diffusion["million_cell_updates_per_second"]
   return {
      "no_diffusion_mcells_s": no_diffusion["million_cell_updates_per_second"],
      "diffusion_mcells_s": diffusion["million_cell_updates_per_second"],
      "slowdown": slowdown,
      "diffusion_steps_max": diffusion["diffusion_steps_max"],
      "diffusion_steps_total": diffusion["diffusion_steps_total"],
      "diffusion_error": diffusion["diffusion_error"],
   }


def case_mhd_rejects_hllc():
   try:
      run_mhd_case(MHDRunConfig(nx=16, ny=16, steps=1, riemann=RIEMANN_HLLC))
   except ValueError as err:
      if "hydro-only" not in str(err):
         raise
      return {"rejected_riemann": RIEMANN_HLLC}
   raise AssertionError("MHD HLLC unexpectedly ran")


def case_mhd_riemann_solvers():
   runs = {
      "llf": run_mhd_case(MHDRunConfig(nx=24, ny=24, steps=2, recon=RECON_PLM, cfl=0.05, riemann=RIEMANN_LLF)),
      "hll": run_mhd_case(MHDRunConfig(nx=24, ny=24, steps=2, recon=RECON_PLM, cfl=0.05, riemann=RIEMANN_HLL)),
      "hlle": run_mhd_case(MHDRunConfig(nx=24, ny=24, steps=2, recon=RECON_PLM, cfl=0.05, riemann=RIEMANN_HLLE)),
      "hlld": run_mhd_case(MHDRunConfig(nx=24, ny=24, steps=2, recon=RECON_PLM, cfl=0.05, riemann=RIEMANN_HLLD)),
   }
   for name, row in runs.items():
      require_finite_row(name, row)
      require_close_to_roundoff(f"{name}_mass_rel_drift", row["mass_rel_drift"], 1e-12)
   return {f"{name}_divb": row["divb_final"] for name, row in runs.items()}


def case_boundary_conditions():
   rows = [
      run_hydro_case(HydroRunConfig(nx=32, ny=32, steps=4, bc=(BC_REFLECTIVE, BC_REFLECTIVE, BC_REFLECTIVE, BC_REFLECTIVE))),
      run_hydro_case(HydroRunConfig(nx=32, ny=32, steps=4, bc=(BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW))),
   ]
   for name, row in zip(("reflective", "outflow"), rows):
      require_finite_row(name, row)
   require_close_to_roundoff("reflective_mass_rel_drift", rows[0]["mass_rel_drift"], 1e-12)
   require_close_to_roundoff("reflective_energy_rel_drift", rows[0]["energy_rel_drift"], 1e-12)
   return {row["bc"]: row["million_cell_updates_per_second"] for row in rows}


def case_hydro_initial_conditions():
   rows = []
   for ic in SUPPORTED_HYDRO_INITIAL_CONDITIONS:
      delrho_factor = 0.01 if ic == 6 else 6.0
      row = run_hydro_case(HydroRunConfig(nx=32, ny=32, steps=2, cfl=0.15, ic=ic, delrho_factor=delrho_factor))
      require_finite_row(f"ic_{ic}", row)
      require_close_to_roundoff(f"ic_{ic}_mass_rel_drift", row["mass_rel_drift"], 1e-11)
      require_close_to_roundoff(f"ic_{ic}_energy_rel_drift", row["energy_rel_drift"], 1e-11)
      rows.append(row)
   return {"ics": [row["ic"] for row in rows], "min_rate": min(row["million_cell_updates_per_second"] for row in rows)}


def case_unsupported_magnetized_ic():
   try:
      run_hydro_case(HydroRunConfig(nx=16, ny=16, steps=1, ic=8))
   except ValueError as err:
      if "magnetized" not in str(err):
         raise
      return {"rejected_ic": 8}
   raise AssertionError("magnetized IC 8 unexpectedly ran through the Numba hydro backend")


def timed_case_row(label, func, config):
   start = time.time()
   row = func(config)
   wall = time.time() - start
   hot = row["elapsed"]
   return {
      "case": label,
      "nx": row["nx"],
      "ny": row["ny"],
      "steps": row["steps"],
      "threads": row["numba_threads"],
      "backend": row["backend"],
      "total_wall_s": wall,
      "hot_solver_s": hot,
      "overhead_s": wall - hot,
      "hot_mcells_s": row["million_cell_updates_per_second"],
      "rhs_stage_hot_mcells_s": row["rhs_stage_million_cell_updates_per_second"],
   }


def case_hot_run_timing():
   rows = (
      timed_case_row(
         "hydro",
         run_hydro_case,
         HydroRunConfig(
            nx=256,
            ny=256,
            steps=32,
            ng=3,
            recon=RECON_WENOZ_CHAR,
            recon_vars=RECON_VARS_PRIMITIVE,
            limiter=LIMITER_MC,
            riemann=RIEMANN_HLLC,
            timestep=TIME_RK4,
            cfl=0.9,
         ),
      ),
      timed_case_row(
         "mhd",
         run_mhd_case,
         MHDRunConfig(
            nx=256,
            ny=256,
            steps=32,
            ng=3,
            ic=10,
            recon=RECON_WENOZ_CHAR,
            recon_vars=RECON_VARS_PRIMITIVE,
            limiter=LIMITER_MC,
            riemann=RIEMANN_HLLD,
            timestep=TIME_RK4,
            ct_method=CT_FLUX_TOTH,
            cfl=0.9,
            bc=(BC_OUTFLOW, BC_OUTFLOW, BC_PERIODIC, BC_PERIODIC),
         ),
      ),
   )
   write_csv(list(rows), OUTDIR / "hot_run_timing.csv")
   return {
      row["case"]: {
         "total_wall_s": row["total_wall_s"],
         "hot_solver_s": row["hot_solver_s"],
         "overhead_s": row["overhead_s"],
         "hot_mcells_s": row["hot_mcells_s"],
      }
      for row in rows
   }


CASES = (
   ("default_threads", case_default_threads),
   ("entropy_wave_accuracy", case_entropy_wave_accuracy),
   ("entropy_wave_convergence", case_entropy_wave_convergence),
   ("acoustic_wave_accuracy", case_acoustic_wave_accuracy),
   ("acoustic_wave_convergence", case_acoustic_wave_convergence),
   ("riemann_solvers", case_riemann_solvers),
   ("reconstruction_variables", case_reconstruction_variables),
   ("wenoz_reconstruction", case_wenoz_reconstruction),
   ("method_scans", case_method_scans),
   ("hydro_direct_reconstruction", case_hydro_direct_reconstruction),
   ("hydro_spatial_reconstruction", case_hydro_spatial_reconstruction),
   ("mhd_method_scans", case_mhd_method_scans),
   ("hydro_characteristic_convergence", case_hydro_characteristic_convergence),
   ("mhd_characteristic_convergence", case_mhd_characteristic_convergence),
   ("circular_alfven_wave", case_circular_alfven_wave),
   ("circular_alfven_matrix", case_circular_alfven_matrix),
   ("field_loop_advection", case_field_loop_advection),
   ("sod_exact_plot", case_sod_exact_plot),
   ("sod_wenoz_char_compare", case_sod_wenoz_char_compare),
   ("sod_100to1_wenoz_char_compare", case_sod_100to1_wenoz_char_compare),
   ("mhd_riemann_discontinuity_compare", case_mhd_riemann_discontinuity_compare),
   ("resistive_mhd_exact_plot", case_resistive_mhd_exact_plot),
   ("resistive_diffusion_timestep_benchmark", case_resistive_diffusion_timestep_benchmark),
   ("memory_shapes", case_memory_shapes),
   ("mhd_wenoz_ct", case_mhd_wenoz_ct),
   ("mhd_wenoz_char_smoke", case_mhd_wenoz_char_smoke),
   ("mhd_diffusion", case_mhd_diffusion),
   ("mhd_diffusion_diagnostics", case_mhd_diffusion_diagnostics),
   ("implicit_diffusion_slowdown", case_implicit_diffusion_slowdown),
   ("mhd_rejects_hllc", case_mhd_rejects_hllc),
   ("mhd_riemann_solvers", case_mhd_riemann_solvers),
   ("boundary_conditions", case_boundary_conditions),
   ("hydro_initial_conditions", case_hydro_initial_conditions),
   ("unsupported_magnetized_ic", case_unsupported_magnetized_ic),
   ("hot_run_timing", case_hot_run_timing),
)


DEFAULT_CASES = (
   "default_threads",
   "sod_exact_plot",
   "resistive_mhd_exact_plot",
   "resistive_diffusion_timestep_benchmark",
   "mhd_wenoz_ct",
   "mhd_diffusion",
   "mhd_diffusion_diagnostics",
)


def parse_args():
   parser = argparse.ArgumentParser(description="Run Numba hydro validation cases.")
   parser.add_argument("--case", action="append", choices=[name for name, _ in CASES],
                       help="Run only the selected case. Can be passed more than once.")
   parser.add_argument("--all", action="store_true", help="Run every validation case, including convergence scans.")
   return parser.parse_args()


def main():
   args = parse_args()
   if args.case:
      selected = set(args.case)
   elif args.all:
      selected = {name for name, _ in CASES}
   else:
      selected = set(DEFAULT_CASES)
   failed = []
   for name, func in CASES:
      if name not in selected:
         continue
      start = time.time()
      try:
         result = func()
      except Exception as err:
         failed.append(name)
         print(f"FAIL {name}: {err}")
         traceback.print_exc()
      else:
         elapsed = time.time() - start
         suffix = f" {result}" if result is not None else ""
         print(f"PASS {name} ({elapsed:.2f}s){suffix}")
   if failed:
      print(f"FAILED {len(failed)} case(s): {', '.join(failed)}")
      return 1
   print("All selected Numba validation cases passed.")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
