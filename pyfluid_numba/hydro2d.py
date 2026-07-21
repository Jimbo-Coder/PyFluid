import time

import numpy as np

from pyfluid.runtime import configure_numba_threads
from pyfluid_numba.diagnostics import (
   acoustic_wave_errors,
   acoustic_wave_step_budget,
   conserved_totals,
   circular_alfven_wave_errors,
   circular_alfven_wave_step_budget,
   entropy_wave_errors,
   entropy_wave_step_budget,
   hydro_characteristic_wave_errors,
   hydro_characteristic_wave_step_budget,
   mhd_characteristic_wave_errors,
   mhd_characteristic_wave_step_budget,
   normalized_divb,
)
from pyfluid_numba.initial_data import (
   acoustic_wave_initial_state,
   circular_alfven_wave_initial_state,
   entropy_wave_initial_state,
   hydro_characteristic_wave_initial_state,
   hydro_ic_initial_state,
   initial_state,
   mhd_characteristic_wave_initial_state,
   mhd_ic_initial_state,
   resistive_wave_initial_state,
)
from pyfluid_numba.options import *
from pyfluid_numba.diffusion import explicit_diffusion_update, implicit_diffusion_update
from pyfluid_numba.timestepping import evolve_steps, evolve_steps_with_diagnostics, evolve_to_time

def make_workspace(Uvec, nx, ny):
   nvar = Uvec.shape[0]
   Fx = np.zeros((nvar, nx + 1, ny), dtype=Uvec.dtype)
   Fy = np.zeros((nvar, nx, ny + 1), dtype=Uvec.dtype)
   if nvar == NVAR_HYDRO:
      return (
         Uvec.copy(),
         Uvec.copy(),
         np.zeros_like(Uvec),
         Fx,
         Fy,
         np.zeros((1, 1), dtype=Uvec.dtype),
         np.zeros((1, 1, 1), dtype=Uvec.dtype),
         np.zeros((1, 1, 1), dtype=Uvec.dtype),
         np.zeros((1, 1, 1), dtype=Uvec.dtype),
         np.zeros((1, 1, 1), dtype=Uvec.dtype),
      )
   return (
      Uvec.copy(),
      Uvec.copy(),
      np.zeros_like(Uvec),
      Fx,
      Fy,
      np.zeros((nx + 1, ny + 1), dtype=Uvec.dtype),
      np.zeros_like(Fx),
      np.zeros_like(Fx),
      np.zeros_like(Fy),
      np.zeros_like(Fy),
   )


def throughput_metrics(nx, ny, steps, timestep, elapsed):
   cell_updates = float(nx * ny * steps)
   rhs_stages = timestep_rhs_stages(timestep)
   rhs_stage_cell_updates = cell_updates * rhs_stages
   return {
      "rhs_stages_per_step": rhs_stages,
      "cell_updates": cell_updates,
      "cell_updates_per_second": cell_updates / elapsed,
      "million_cell_updates_per_second": cell_updates / elapsed / 1.0e6,
      "rhs_stage_cell_updates": rhs_stage_cell_updates,
      "rhs_stage_cell_updates_per_second": rhs_stage_cell_updates / elapsed,
      "rhs_stage_million_cell_updates_per_second": rhs_stage_cell_updates / elapsed / 1.0e6,
   }


def run_hydro_case(config):
   if not isinstance(config, HydroRunConfig):
      config = HydroRunConfig(**config)
   ic = validate_hydro_initial_condition(config.ic)
   validate_common_numeric_options(config, mhd=False)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   iname = hydro_initial_condition_name(ic)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   if ic == SIN_WAVE_IC:
      Uvec = initial_state(config.nx, config.ny, config.ng, config.gamma, bc)
   else:
      Uvec = hydro_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   if ic == SIN_WAVE_IC:
      Uvec = initial_state(config.nx, config.ny, config.ng, config.gamma, bc)
   else:
      Uvec = hydro_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)

   start = time.time()
   time_value, dt_min, dt_max = evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.steps,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start

   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   return {
      "backend": f"numba_hydro2d_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "nvar": Uvec.shape[0],
      "dtype": str(Uvec.dtype),
      "steps": config.steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "ic": ic,
      "ic_name": iname,
      "delrho_factor": config.delrho_factor,
      "bc": str(bc),
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, config.steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
   }


def run_mhd_case(config):
   if not isinstance(config, MHDRunConfig):
      config = MHDRunConfig(**config)
   ic = validate_mhd_initial_condition(config.ic)
   validate_common_numeric_options(config, mhd=True)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   dname = diffusion_method_name(config.diffusion_method)
   cname = ct_method_name(config.ct_method)
   iname = hydro_initial_condition_name(ic)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   viscosity, resistivity = resolve_diffusion_coefficients(config, dx)
   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

   evolve_steps_with_diagnostics(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      config.ct_method, config.diffusion_method, viscosity, resistivity, config.diffusion_cfl,
      config.diffusion_tolerance, config.diffusion_max_steps, config.diffusion_sor_omega
   )
   Uvec = mhd_ic_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb0 = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)

   start = time.time()
   time_value, dt_min, dt_max, diffusion_error, diffusion_steps_max, diffusion_steps_total = evolve_steps_with_diagnostics(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.steps,
      config.ct_method, config.diffusion_method, viscosity, resistivity, config.diffusion_cfl,
      config.diffusion_tolerance, config.diffusion_max_steps, config.diffusion_sor_omega
   )
   elapsed = time.time() - start

   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)
   return {
      "backend": f"numba_mhd2d_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "nvar": Uvec.shape[0],
      "dtype": str(Uvec.dtype),
      "steps": config.steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "ic": ic,
      "ic_name": iname,
      "delrho_factor": config.delrho_factor,
      "bc": str(bc),
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "ct_method": config.ct_method,
      "ct_method_name": cname,
      "diffusion": bool(config.diffusion_method != DIFFUSION_NONE),
      "diffusion_method": config.diffusion_method,
      "diffusion_method_name": dname,
      "viscosity": viscosity,
      "resistivity": resistivity,
      "diffusion_error": diffusion_error,
      "diffusion_steps_max": diffusion_steps_max,
      "diffusion_steps_total": diffusion_steps_total,
      "diffusion_sor_omega": config.diffusion_sor_omega,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, config.steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      "divb_initial": divb0,
      "divb_final": divb,
      "divb_delta": divb - divb0,
   }


def run_entropy_wave_case(config):
   if not isinstance(config, EntropyWaveConfig):
      config = EntropyWaveConfig(**config)
   validate_common_numeric_options(config, mhd=False)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = entropy_wave_step_budget(config)
   Uvec = entropy_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   Uvec = entropy_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.final_time, max_steps,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, config.final_time, steps, max_steps)

   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   errors = entropy_wave_errors(Uvec, config, time_value)
   return {
      "backend": f"numba_hydro2d_entropy_wave_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "dtype": str(Uvec.dtype),
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      **errors,
   }


def run_acoustic_wave_case(config):
   if not isinstance(config, AcousticWaveConfig):
      config = AcousticWaveConfig(**config)
   validate_common_numeric_options(config, mhd=False)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = acoustic_wave_step_budget(config)
   Uvec = acoustic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   Uvec = acoustic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.final_time, max_steps,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, config.final_time, steps, max_steps)

   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   errors = acoustic_wave_errors(Uvec, config, time_value)
   return {
      "backend": f"numba_hydro2d_acoustic_wave_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "dtype": str(Uvec.dtype),
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      **errors,
   }


def run_hydro_characteristic_wave_case(config):
   if not isinstance(config, HydroCharacteristicWaveConfig):
      config = HydroCharacteristicWaveConfig(**config)
   validate_common_numeric_options(config, mhd=False)
   validate_hydro_characteristic_mode(config.mode)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   mname = hydro_mode_name(config.mode)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = hydro_characteristic_wave_step_budget(config)
   Uvec = hydro_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   Uvec = hydro_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

   start = time.time()
   time_value, steps, dt_min, dt_max = evolve_to_time(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, config.final_time, max_steps,
      CT_OFF, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   elapsed = time.time() - start
   require_reached_final_time(time_value, config.final_time, steps, max_steps)

   totals = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   errors = hydro_characteristic_wave_errors(Uvec, config, time_value)
   return {
      "backend": f"numba_hydro2d_characteristic_{mname}_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "dtype": str(Uvec.dtype),
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "mode": config.mode,
      "mode_name": mname,
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      **errors,
   }


def run_mhd_characteristic_wave_case(config):
   if not isinstance(config, MHDCharacteristicWaveConfig):
      config = MHDCharacteristicWaveConfig(**config)
   validate_common_numeric_options(config, mhd=True)
   validate_mhd_characteristic_mode(config.mode)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   cname = ct_method_name(config.ct_method)
   mname = mhd_mode_name(config.mode)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = mhd_characteristic_wave_step_budget(config)
   Uvec = mhd_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb0 = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   Uvec = mhd_characteristic_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

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
   errors = mhd_characteristic_wave_errors(Uvec, config, time_value)
   return {
      "backend": f"numba_mhd2d_characteristic_{mname}_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "dtype": str(Uvec.dtype),
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "mode": config.mode,
      "mode_name": mname,
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "ct_method": config.ct_method,
      "ct_method_name": cname,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      "divb_initial": divb0,
      "divb_final": divb,
      "divb_delta": divb - divb0,
      **errors,
   }


def evolve_circular_alfven_wave_case(config):
   if not isinstance(config, CircularAlfvenWaveConfig):
      config = CircularAlfvenWaveConfig(**config)
   validate_circular_alfven_options(config)
   rname = riemann_name(config.riemann)
   xname = recon_name(config.recon)
   vname = recon_vars_name(config.recon_vars)
   lname = limiter_name(config.limiter)
   tname = timestep_name(config.timestep)
   cname = ct_method_name(config.ct_method)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   max_steps = circular_alfven_wave_step_budget(config)
   Uvec = circular_alfven_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)
   totals0 = conserved_totals(Uvec, config.nx, config.ny, config.ng, dx, dy)
   divb0 = normalized_divb(Uvec, config.nx, config.ny, config.ng, dx, dy)

   evolve_steps(
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, config.cfl, config.gamma, config.ng,
      bc[0], bc[1], bc[2], bc[3], config.recon, config.recon_vars, config.limiter, config.riemann, config.timestep, 2,
      config.ct_method, DIFFUSION_NONE, 0.0, 0.0, 0.0
   )
   Uvec = circular_alfven_wave_initial_state(config)
   Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT = make_workspace(Uvec, config.nx, config.ny)

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
   errors = circular_alfven_wave_errors(Uvec, config, time_value)
   row = {
      "backend": f"numba_mhd2d_circular_alfven_{xname}_{vname}_{lname}_{rname}_{tname}",
      "nx": config.nx,
      "ny": config.ny,
      "dtype": str(Uvec.dtype),
      "steps": steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "recon": config.recon,
      "recon_vars": config.recon_vars,
      "limiter": config.limiter,
      "riemann": config.riemann,
      "timestep": config.timestep,
      "ct_method": config.ct_method,
      "ct_method_name": cname,
      "time": time_value,
      "dt_min": dt_min,
      "dt_max": dt_max,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, steps, config.timestep, elapsed),
      "mass_rel_drift": abs(totals["mass"] - totals0["mass"]) / max(abs(totals0["mass"]), 1.0),
      "energy_rel_drift": abs(totals["energy"] - totals0["energy"]) / max(abs(totals0["energy"]), 1.0),
      "divb_initial": divb0,
      "divb_final": divb,
      "divb_delta": divb - divb0,
      **errors,
   }
   return row, Uvec


def run_circular_alfven_wave_case(config):
   row, _ = evolve_circular_alfven_wave_case(config)
   return row


def evolve_resistive_wave_case(config):
   if not isinstance(config, ResistiveWaveConfig):
      config = ResistiveWaveConfig(**config)
   dname = diffusion_method_name(config.diffusion_method)
   bc = validate_bc(config.bc)
   runtime_threads = configure_numba_threads(config.threads)
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   dt = config.final_time / config.steps
   Uvec = resistive_wave_initial_state(config)
   Utmp = Uvec.copy()
   rhs = np.zeros_like(Uvec)

   if config.diffusion_method == DIFFUSION_EXPLICIT:
      explicit_diffusion_update(
         Uvec, Utmp, rhs, dx, dy, dt, config.ng,
         bc[0], bc[1], bc[2], bc[3],
         0.0, config.resistivity
      )
   elif config.diffusion_method == DIFFUSION_IMPLICIT:
      implicit_diffusion_update(
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
   elif config.diffusion_method != DIFFUSION_NONE:
      raise ValueError(f"unsupported resistive wave diffusion method: {config.diffusion_method}")
   Uvec = resistive_wave_initial_state(config)
   Utmp = Uvec.copy()
   rhs = np.zeros_like(Uvec)

   diffusion_error = 0.0
   diffusion_steps_max = 0
   diffusion_steps_total = 0
   start = time.time()
   for _ in range(config.steps):
      if config.diffusion_method == DIFFUSION_EXPLICIT:
         explicit_diffusion_update(
            Uvec, Utmp, rhs, dx, dy, dt, config.ng,
            bc[0], bc[1], bc[2], bc[3],
            0.0, config.resistivity
         )
         diffusion_error = 0.0
         diffusion_steps = 0
      elif config.diffusion_method == DIFFUSION_IMPLICIT:
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
      else:
         diffusion_error = 0.0
         diffusion_steps = 0
      diffusion_steps_total += diffusion_steps
      if diffusion_steps > diffusion_steps_max:
         diffusion_steps_max = diffusion_steps
   elapsed = time.time() - start

   row = {
      "backend": "numba_mhd2d_resistive_wave",
      "nx": config.nx,
      "ny": config.ny,
      "nvar": Uvec.shape[0],
      "dtype": str(Uvec.dtype),
      "steps": config.steps,
      "threads": config.threads,
      "numba_threads": runtime_threads,
      "bc": str(bc),
      "time": config.final_time,
      "dt_min": dt,
      "dt_max": dt,
      "elapsed": elapsed,
      **throughput_metrics(config.nx, config.ny, config.steps, TIME_EULER, elapsed),
      "diffusion_method": config.diffusion_method,
      "diffusion_method_name": dname,
      "resistivity": config.resistivity,
      "diffusion_error": diffusion_error,
      "diffusion_steps_max": diffusion_steps_max,
      "diffusion_steps_total": diffusion_steps_total,
      "diffusion_sor_omega": config.diffusion_sor_omega,
   }
   return row, Uvec


def run_resistive_wave_case(config):
   row, _ = evolve_resistive_wave_case(config)
   return row
