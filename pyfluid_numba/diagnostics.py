from numba import njit, prange
import numpy as np

from pyfluid_numba.options import *
from pyfluid_numba.eigensystem import (
   hydro_mode_norm,
   hydro_mode_primitive_perturbation,
   hydro_mode_speed,
   mhd_mode_norm,
   mhd_mode_primitive_perturbation,
   mhd_mode_speed,
)
from pyfluid_numba.physics import pressure_state, pressure_state_mhd, max_signal_speed_array

@njit(cache=True, parallel=True)
def conserved_totals_array(Uvec, nx, ny, ng, dx, dy):
   mass = 0.0
   energy = 0.0
   momx = 0.0
   momy = 0.0
   momz = 0.0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      mass += Uvec[RHO, i, j]
      energy += Uvec[ENERGY, i, j]
      momx += Uvec[MOMX, i, j]
      momy += Uvec[MOMY, i, j]
      momz += Uvec[MOMZ, i, j]
   cell_volume = dx * dy
   return mass * cell_volume, energy * cell_volume, momx * cell_volume, momy * cell_volume, momz * cell_volume


def conserved_totals(Uvec, nx, ny, ng, dx, dy):
   mass, energy, momx, momy, momz = conserved_totals_array(Uvec, nx, ny, ng, dx, dy)
   return {"mass": float(mass), "energy": float(energy), "momx": float(momx), "momy": float(momy), "momz": float(momz)}


@njit(cache=True, parallel=True)
def scalar_diagnostics_array(Uvec, nx, ny, ng, dx, dy, gamma):
   mass = 0.0
   energy_total = 0.0
   momx_total = 0.0
   momy_total = 0.0
   momz_total = 0.0
   internal_energy = 0.0
   kinetic_energy = 0.0
   magnetic_energy = 0.0
   min_rho = 1.0e300
   min_pressure = 1.0e300
   divb2 = 0.0
   b2 = 0.0
   mhd = Uvec.shape[0] > NVAR_HYDRO
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy

   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      kinetic = 0.5 * (mx * mx + my * my + mz * mz) / rho
      magnetic = 0.0
      if mhd:
         bx = Uvec[BX, i, j]
         by = Uvec[BY, i, j]
         bz = Uvec[BZ, i, j]
         magnetic = 0.5 * (bx * bx + by * by + bz * bz)
         pressure = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
         ic = ng - 1 + ii
         jc = ng - 1 + jj
         divb = (
            (Uvec[BX, ic + 1, jc + 1] + Uvec[BX, ic + 1, jc] - Uvec[BX, ic, jc + 1] - Uvec[BX, ic, jc]) * inv_2dx
            + (Uvec[BY, ic + 1, jc + 1] + Uvec[BY, ic, jc + 1] - Uvec[BY, ic + 1, jc] - Uvec[BY, ic, jc]) * inv_2dy
         )
         divb2 += divb * divb
         b2 += bx * bx + by * by + bz * bz
      else:
         pressure = pressure_state(rho, mx, my, mz, energy, gamma)
      mass += rho
      energy_total += energy
      momx_total += mx
      momy_total += my
      momz_total += mz
      internal_energy += pressure / (gamma - 1.0)
      kinetic_energy += kinetic
      magnetic_energy += magnetic
      min_rho = min(min_rho, rho)
      min_pressure = min(min_pressure, pressure)

   cell_volume = dx * dy
   divb = 0.0
   if mhd:
      cells = nx * ny
      divb_l2 = np.sqrt(divb2 / cells)
      b_rms = np.sqrt(b2 / cells)
      length = dx if dx < dy else dy
      normalizer = b_rms / length
      if normalizer > 0.0:
         divb = divb_l2 / normalizer
   return (
      mass * cell_volume,
      energy_total * cell_volume,
      momx_total * cell_volume,
      momy_total * cell_volume,
      momz_total * cell_volume,
      np.sqrt(momx_total * momx_total + momy_total * momy_total + momz_total * momz_total) * cell_volume,
      internal_energy * cell_volume,
      kinetic_energy * cell_volume,
      magnetic_energy * cell_volume,
      min_rho,
      min_pressure,
      divb,
   )


def scalar_diagnostics(Uvec, nx, ny, ng, dx, dy, gamma):
   (
      mass,
      energy,
      momx,
      momy,
      momz,
      momentum_norm,
      internal_energy,
      kinetic_energy,
      magnetic_energy,
      min_rho,
      min_pressure,
      divb,
   ) = scalar_diagnostics_array(Uvec, nx, ny, ng, dx, dy, gamma)
   return {
      "mass": float(mass),
      "energy": float(energy),
      "momx": float(momx),
      "momy": float(momy),
      "momz": float(momz),
      "momentum_norm": float(momentum_norm),
      "internal_energy": float(internal_energy),
      "kinetic_energy": float(kinetic_energy),
      "magnetic_energy": float(magnetic_energy),
      "min_rho": float(min_rho),
      "min_pressure": float(min_pressure),
      "divb": float(divb),
   }


@njit(cache=True, parallel=True)
def normalized_divb_array(Uvec, nx, ny, ng, dx, dy):
   if Uvec.shape[0] <= NVAR_HYDRO:
      return 0.0
   divb2 = 0.0
   b2 = 0.0
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      ic = ng - 1 + ii
      jc = ng - 1 + jj
      divb = (
         (Uvec[BX, ic + 1, jc + 1] + Uvec[BX, ic + 1, jc] - Uvec[BX, ic, jc + 1] - Uvec[BX, ic, jc]) * inv_2dx
         + (Uvec[BY, ic + 1, jc + 1] + Uvec[BY, ic, jc + 1] - Uvec[BY, ic + 1, jc] - Uvec[BY, ic, jc]) * inv_2dy
      )
      divb2 += divb * divb
      i = ng + ii
      j = ng + jj
      b2 += Uvec[BX, i, j] * Uvec[BX, i, j] + Uvec[BY, i, j] * Uvec[BY, i, j] + Uvec[BZ, i, j] * Uvec[BZ, i, j]
   cells = nx * ny
   divb_l2 = np.sqrt(divb2 / cells)
   b_rms = np.sqrt(b2 / cells)
   length = dx if dx < dy else dy
   normalizer = b_rms / length
   normalized_l2 = 0.0
   if normalizer > 0.0:
      normalized_l2 = divb_l2 / normalizer
   return normalized_l2


def normalized_divb(Uvec, nx, ny, ng, dx, dy):
   return float(normalized_divb_array(Uvec, nx, ny, ng, dx, dy))


def max_signal_speed(Uvec, nx, ny, ng, gamma):
   speed = float(max_signal_speed_array(Uvec, nx, ny, ng, gamma))
   if not np.isfinite(speed) or speed <= 0.0:
      raise ValueError(f"non-positive or non-finite hydro signal speed: {speed}")
   return speed


@njit(cache=True, parallel=True)
def entropy_wave_error_array(Uvec, nx, ny, ng, gamma, rho0, pressure0, amplitude, vx, vy, kx, ky, time_value):
   rho_l2 = 0.0
   pressure_l2 = 0.0
   vx_l2 = 0.0
   vy_l2 = 0.0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * (x - vx * time_value) + ky * (y - vy * time_value))
      rho_exact = rho0 + amplitude * np.sin(phase)
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      pressure = pressure_state(rho, mx, my, mz, energy, gamma)
      vx_num = mx / rho
      vy_num = my / rho
      rho_l2 += (rho - rho_exact) ** 2
      pressure_l2 += (pressure - pressure0) ** 2
      vx_l2 += (vx_num - vx) ** 2
      vy_l2 += (vy_num - vy) ** 2
   norm = 1.0 / (nx * ny)
   return (
      np.sqrt(rho_l2 * norm) / amplitude,
      np.sqrt(pressure_l2 * norm) / pressure0,
      np.sqrt(vx_l2 * norm),
      np.sqrt(vy_l2 * norm),
   )


def entropy_wave_errors(Uvec, config, time_value):
   rho_l2, pressure_l2, vx_l2, vy_l2 = entropy_wave_error_array(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.rho0,
      config.pressure0,
      config.amplitude,
      config.vx,
      config.vy,
      config.kx,
      config.ky,
      time_value,
   )
   return {
      "rho_l2": float(rho_l2),
      "pressure_l2": float(pressure_l2),
      "vx_l2": float(vx_l2),
      "vy_l2": float(vy_l2),
   }


@njit(cache=True, parallel=True)
def acoustic_wave_error_array(Uvec, nx, ny, ng, gamma, rho0, pressure0, amplitude, kx, ky, time_value):
   rho_l2 = 0.0
   pressure_l2 = 0.0
   vx_l2 = 0.0
   vy_l2 = 0.0
   sound_speed = np.sqrt(gamma * pressure0 / rho0)
   k_norm = np.sqrt(kx * kx + ky * ky)
   nx_hat = kx / k_norm
   ny_hat = ky / k_norm
   velocity_scale = sound_speed * amplitude / rho0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * x + ky * y - sound_speed * k_norm * time_value)
      drho = amplitude * np.sin(phase)
      rho_exact = rho0 + drho
      pressure_exact = pressure0 + sound_speed * sound_speed * drho
      normal_velocity = sound_speed * drho / rho0
      vx_exact = nx_hat * normal_velocity
      vy_exact = ny_hat * normal_velocity
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      pressure = pressure_state(rho, mx, my, mz, energy, gamma)
      vx_num = mx / rho
      vy_num = my / rho
      rho_l2 += (rho - rho_exact) ** 2
      pressure_l2 += (pressure - pressure_exact) ** 2
      vx_l2 += (vx_num - vx_exact) ** 2
      vy_l2 += (vy_num - vy_exact) ** 2
   norm = 1.0 / (nx * ny)
   return (
      np.sqrt(rho_l2 * norm) / amplitude,
      np.sqrt(pressure_l2 * norm) / (sound_speed * sound_speed * amplitude),
      np.sqrt(vx_l2 * norm) / velocity_scale,
      np.sqrt(vy_l2 * norm) / velocity_scale,
   )


def acoustic_wave_errors(Uvec, config, time_value):
   rho_l2, pressure_l2, vx_l2, vy_l2 = acoustic_wave_error_array(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.rho0,
      config.pressure0,
      config.amplitude,
      config.kx,
      config.ky,
      time_value,
   )
   return {
      "rho_l2": float(rho_l2),
      "pressure_l2": float(pressure_l2),
      "vx_l2": float(vx_l2),
      "vy_l2": float(vy_l2),
   }


def entropy_wave_step_budget(config):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   sound_speed = np.sqrt(config.gamma * config.pressure0 / (config.rho0 - abs(config.amplitude)))
   max_frequency = (abs(config.vx) + sound_speed) / dx + (abs(config.vy) + sound_speed) / dy
   estimated_dt = config.cfl / max_frequency
   estimated_steps = max(1, int(np.ceil(config.final_time / estimated_dt)))
   return max(1000, 8 * estimated_steps)


def acoustic_wave_step_budget(config):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   sound_speed = np.sqrt(config.gamma * config.pressure0 / (config.rho0 - abs(config.amplitude)))
   max_speed = sound_speed * (1.0 + abs(config.amplitude) / config.rho0)
   estimated_dt = config.cfl / (max_speed / dx + max_speed / dy)
   estimated_steps = max(1, int(np.ceil(config.final_time / estimated_dt)))
   return max(1000, 8 * estimated_steps)


@njit(cache=True, parallel=True)
def hydro_characteristic_wave_error_array(
   Uvec, nx, ny, ng, gamma, mode, rho0, pressure0, vx0, vy0, vz0, amplitude, kx, ky, time_value
):
   rho_l2 = 0.0
   pressure_l2 = 0.0
   velocity_l2 = 0.0
   mode_l2 = 0.0
   speed = hydro_mode_speed(mode, rho0, pressure0, vx0, vy0, gamma, kx, ky)
   k_norm = np.sqrt(kx * kx + ky * ky)
   mode_norm = hydro_mode_norm(rho0, pressure0, gamma, kx, ky, mode)
   base_drho, base_dvx, base_dvy, base_dvz, base_dpressure = hydro_mode_primitive_perturbation(
      mode, rho0, pressure0, gamma, kx, ky, 1.0, 1.0
   )
   rho_norm = amplitude * max(abs(base_drho), 1.0)
   pressure_norm = amplitude * max(abs(base_dpressure), 1.0)
   velocity_norm = amplitude * max(np.sqrt(base_dvx * base_dvx + base_dvy * base_dvy + base_dvz * base_dvz), 1.0)
   if mode_norm <= 0.0:
      mode_norm = 1.0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * x + ky * y - speed * k_norm * time_value)
      wave = np.sin(phase)
      drho, dvx, dvy, dvz, dpressure = hydro_mode_primitive_perturbation(
         mode, rho0, pressure0, gamma, kx, ky, amplitude, wave
      )
      rho_exact = rho0 + drho
      vx_exact = vx0 + dvx
      vy_exact = vy0 + dvy
      vz_exact = vz0 + dvz
      pressure_exact = pressure0 + dpressure
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      vx = mx / rho
      vy = my / rho
      vz = mz / rho
      pressure = pressure_state(rho, mx, my, mz, energy, gamma)
      erho = rho - rho_exact
      evx = vx - vx_exact
      evy = vy - vy_exact
      evz = vz - vz_exact
      ep = pressure - pressure_exact
      rho_l2 += erho * erho
      pressure_l2 += ep * ep
      velocity_l2 += evx * evx + evy * evy + evz * evz
      mode_l2 += erho * erho + evx * evx + evy * evy + evz * evz + ep * ep
   norm = 1.0 / (nx * ny)
   return (
      np.sqrt(mode_l2 * norm) / (amplitude * mode_norm),
      np.sqrt(rho_l2 * norm) / rho_norm,
      np.sqrt(pressure_l2 * norm) / pressure_norm,
      np.sqrt(velocity_l2 * norm) / velocity_norm,
   )


def hydro_characteristic_wave_errors(Uvec, config, time_value):
   mode_l2, rho_l2, pressure_l2, velocity_l2 = hydro_characteristic_wave_error_array(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.mode,
      config.rho0,
      config.pressure0,
      config.vx0,
      config.vy0,
      config.vz0,
      config.amplitude,
      config.kx,
      config.ky,
      time_value,
   )
   return {
      "mode_l2": float(mode_l2),
      "rho_l2": float(rho_l2),
      "pressure_l2": float(pressure_l2),
      "velocity_l2": float(velocity_l2),
   }


def hydro_characteristic_wave_step_budget(config):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   sound_speed = np.sqrt(config.gamma * config.pressure0 / (config.rho0 - abs(config.amplitude)))
   max_frequency = (abs(config.vx0) + sound_speed) / dx + (abs(config.vy0) + sound_speed) / dy
   estimated_dt = config.cfl / max_frequency
   estimated_steps = max(1, int(np.ceil(config.final_time / estimated_dt)))
   return max(1000, 8 * estimated_steps)


@njit(cache=True, parallel=True)
def mhd_characteristic_wave_error_array(
   Uvec,
   nx,
   ny,
   ng,
   gamma,
   mode,
   rho0,
   pressure0,
   vx0,
   vy0,
   vz0,
   bx0,
   by0,
   bz0,
   amplitude,
   kx,
   ky,
   time_value,
):
   rho_l2 = 0.0
   pressure_l2 = 0.0
   velocity_l2 = 0.0
   magnetic_l2 = 0.0
   mode_l2 = 0.0
   speed = mhd_mode_speed(mode, rho0, pressure0, vx0, vy0, bx0, by0, bz0, gamma, kx, ky)
   k_norm = np.sqrt(kx * kx + ky * ky)
   mode_norm = mhd_mode_norm(rho0, pressure0, bx0, by0, bz0, gamma, kx, ky, mode)
   base_drho, base_dvx, base_dvy, base_dvz, base_dpressure, base_dbx, base_dby, base_dbz = mhd_mode_primitive_perturbation(
      mode, rho0, pressure0, bx0, by0, bz0, gamma, kx, ky, 1.0, 1.0
   )
   rho_norm = amplitude * max(abs(base_drho), 1.0)
   pressure_norm = amplitude * max(abs(base_dpressure), 1.0)
   velocity_norm = amplitude * max(np.sqrt(base_dvx * base_dvx + base_dvy * base_dvy + base_dvz * base_dvz), 1.0)
   magnetic_norm = amplitude * max(np.sqrt(base_dbx * base_dbx + base_dby * base_dby + base_dbz * base_dbz), 1.0)
   if mode_norm <= 0.0:
      mode_norm = 1.0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * x + ky * y - speed * k_norm * time_value)
      wave = np.sin(phase)
      drho, dvx, dvy, dvz, dpressure, dbx, dby, dbz = mhd_mode_primitive_perturbation(
         mode, rho0, pressure0, bx0, by0, bz0, gamma, kx, ky, amplitude, wave
      )
      rho_exact = rho0 + drho
      vx_exact = vx0 + dvx
      vy_exact = vy0 + dvy
      vz_exact = vz0 + dvz
      pressure_exact = pressure0 + dpressure
      bx_exact = bx0 + dbx
      by_exact = by0 + dby
      bz_exact = bz0 + dbz
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      bx = Uvec[BX, i, j]
      by = Uvec[BY, i, j]
      bz = Uvec[BZ, i, j]
      vx = mx / rho
      vy = my / rho
      vz = mz / rho
      pressure = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
      erho = rho - rho_exact
      evx = vx - vx_exact
      evy = vy - vy_exact
      evz = vz - vz_exact
      ep = pressure - pressure_exact
      ebx = bx - bx_exact
      eby = by - by_exact
      ebz = bz - bz_exact
      rho_l2 += erho * erho
      pressure_l2 += ep * ep
      velocity_l2 += evx * evx + evy * evy + evz * evz
      magnetic_l2 += ebx * ebx + eby * eby + ebz * ebz
      mode_l2 += erho * erho + evx * evx + evy * evy + evz * evz + ep * ep + ebx * ebx + eby * eby + ebz * ebz
   norm = 1.0 / (nx * ny)
   return (
      np.sqrt(mode_l2 * norm) / (amplitude * mode_norm),
      np.sqrt(rho_l2 * norm) / rho_norm,
      np.sqrt(pressure_l2 * norm) / pressure_norm,
      np.sqrt(velocity_l2 * norm) / velocity_norm,
      np.sqrt(magnetic_l2 * norm) / magnetic_norm,
   )


def mhd_characteristic_wave_errors(Uvec, config, time_value):
   mode_l2, rho_l2, pressure_l2, velocity_l2, magnetic_l2 = mhd_characteristic_wave_error_array(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.mode,
      config.rho0,
      config.pressure0,
      config.vx0,
      config.vy0,
      config.vz0,
      config.bx0,
      config.by0,
      config.bz0,
      config.amplitude,
      config.kx,
      config.ky,
      time_value,
   )
   return {
      "mode_l2": float(mode_l2),
      "rho_l2": float(rho_l2),
      "pressure_l2": float(pressure_l2),
      "velocity_l2": float(velocity_l2),
      "magnetic_l2": float(magnetic_l2),
   }


def mhd_characteristic_wave_step_budget(config):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   cf_x = mhd_mode_speed(MHD_MODE_FAST_PLUS, config.rho0, config.pressure0, config.vx0, config.vy0, config.bx0, config.by0, config.bz0, config.gamma, 1, 0) - config.vx0
   cf_y = mhd_mode_speed(MHD_MODE_FAST_PLUS, config.rho0, config.pressure0, config.vx0, config.vy0, config.bx0, config.by0, config.bz0, config.gamma, 0, 1) - config.vy0
   max_frequency = (abs(config.vx0) + abs(cf_x)) / dx + (abs(config.vy0) + abs(cf_y)) / dy
   estimated_dt = config.cfl / max_frequency
   estimated_steps = max(1, int(np.ceil(config.final_time / estimated_dt)))
   return max(1000, 8 * estimated_steps)


@njit(cache=True, parallel=True)
def circular_alfven_wave_error_array(
   Uvec, nx, ny, ng, gamma, rho0, pressure0, bx0, amplitude, wave_number, direction, time_value
):
   rho_l2 = 0.0
   pressure_l2 = 0.0
   velocity_l2 = 0.0
   magnetic_l2 = 0.0
   primitive_l2 = 0.0
   ca = abs(bx0) / np.sqrt(rho0)
   speed = direction * ca
   velocity_factor = -bx0 / (rho0 * speed)
   k = 2.0 * np.pi * wave_number
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      phase = k * (x - speed * time_value)
      by_exact = amplitude * np.cos(phase)
      bz_exact = amplitude * np.sin(phase)
      vy_exact = velocity_factor * by_exact
      vz_exact = velocity_factor * bz_exact
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      bx = Uvec[BX, i, j]
      by = Uvec[BY, i, j]
      bz = Uvec[BZ, i, j]
      vx = mx / rho
      vy = my / rho
      vz = mz / rho
      pressure = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
      erho = rho - rho0
      ep = pressure - pressure0
      evx = vx
      evy = vy - vy_exact
      evz = vz - vz_exact
      ebx = bx - bx0
      eby = by - by_exact
      ebz = bz - bz_exact
      rho_l2 += erho * erho
      pressure_l2 += ep * ep
      velocity_l2 += evx * evx + evy * evy + evz * evz
      magnetic_l2 += ebx * ebx + eby * eby + ebz * ebz
      primitive_l2 += erho * erho + ep * ep + evx * evx + evy * evy + evz * evz + ebx * ebx + eby * eby + ebz * ebz
   norm = 1.0 / (nx * ny)
   return (
      np.sqrt(primitive_l2 * norm) / amplitude,
      np.sqrt(rho_l2 * norm) / rho0,
      np.sqrt(pressure_l2 * norm) / pressure0,
      np.sqrt(velocity_l2 * norm) / (amplitude / np.sqrt(rho0)),
      np.sqrt(magnetic_l2 * norm) / amplitude,
   )


def circular_alfven_wave_errors(Uvec, config, time_value):
   primitive_l2, rho_l2, pressure_l2, velocity_l2, magnetic_l2 = circular_alfven_wave_error_array(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.rho0,
      config.pressure0,
      config.bx0,
      config.amplitude,
      config.wave_number,
      config.direction,
      time_value,
   )
   return {
      "primitive_l2": float(primitive_l2),
      "rho_l2": float(rho_l2),
      "pressure_l2": float(pressure_l2),
      "velocity_l2": float(velocity_l2),
      "magnetic_l2": float(magnetic_l2),
   }


def circular_alfven_wave_step_budget(config):
   dx = 1.0 / config.nx
   dy = 1.0 / config.ny
   ca = abs(config.bx0) / np.sqrt(config.rho0)
   sound_speed = np.sqrt(config.gamma * config.pressure0 / config.rho0)
   fast_speed = np.sqrt(sound_speed * sound_speed + (config.bx0 * config.bx0 + config.amplitude * config.amplitude) / config.rho0)
   estimated_dt = config.cfl / (fast_speed / dx + fast_speed / dy)
   estimated_steps = max(1, int(np.ceil(config.final_time / estimated_dt)))
   crossing_steps = max(1, int(np.ceil(config.final_time * ca / dx)))
   return max(1000, 8 * estimated_steps, 2 * crossing_steps)
