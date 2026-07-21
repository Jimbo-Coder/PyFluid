from numba import njit, prange
import numpy as np

from pyfluid_numba.boundaries import apply_boundaries
from pyfluid_numba.eigensystem import (
   hydro_mode_primitive_perturbation,
   hydro_mode_speed,
   mhd_mode_primitive_perturbation,
   mhd_mode_speed,
)
from pyfluid_numba.options import *
from pyfluid_numba.physics import conserved_from_primitive, conserved_from_primitive_mhd

@njit(cache=True, parallel=True)
def initialize_state_inplace(Uvec, nx, ny, ng, gamma):
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
      vx = 0.15 + 0.02 * np.cos(2.0 * np.pi * y)
      vy = -0.10 + 0.02 * np.sin(2.0 * np.pi * x)
      vz = 0.0
      pressure = 1.0 + 0.03 * np.cos(2.0 * np.pi * x)
      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j] = rho
      Uvec[MOMX, i, j] = rho * vx
      Uvec[MOMY, i, j] = rho * vy
      Uvec[MOMZ, i, j] = rho * vz
      Uvec[ENERGY, i, j] = pressure / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)


def initial_state(nx, ny, ng=DEFAULT_NG, gamma=DEFAULT_GAMMA, bc=(BC_PERIODIC, BC_PERIODIC, BC_PERIODIC, BC_PERIODIC), dtype=np.float64):
   Uvec = np.zeros((NVAR_HYDRO, nx + 2 * ng, ny + 2 * ng), dtype=dtype)
   initialize_state_inplace(Uvec, nx, ny, ng, gamma)
   apply_boundaries(Uvec, ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_hydro_ic_inplace(Uvec, nx, ny, ng, gamma, ic, delrho_factor):
   delrho = delrho_factor
   del_pressure = gamma * delrho
   width = 0.25
   cs0 = np.sqrt(gamma)
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = -1.0 + (ii + 0.5) * 2.0 / nx
      y = -1.0 + (jj + 0.5) * 2.0 / ny
      rho = 1.0
      vx = 0.0
      vy = 0.0
      vz = 0.0
      pressure = 1.0

      if ic == 0 or ic == 1 or ic == 2 or ic == 3:
         selected = (
            (ic == 0 and x < 0.0)
            or (ic == 1 and x >= 0.0)
            or (ic == 2 and y < 0.0)
            or (ic == 3 and y >= 0.0)
         )
         if selected:
            rho += delrho
            pressure += del_pressure
      elif ic == 4:
         if abs(x) < width:
            rho += delrho
            pressure += del_pressure
            vx += cs0 * delrho
      elif ic == 5:
         gfac = np.exp(-0.5 * (x / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
         rho += gfac * delrho
         pressure += gfac * del_pressure
         vx += gfac * cs0 * delrho
      elif ic == 6:
         kx = 2.0 * np.pi / (2.0 + 2.0 / nx)
         wave = np.sin(kx * x)
         rho += delrho * wave
         pressure += del_pressure * wave
      elif ic == 7:
         y1 = -0.5
         y2 = 0.5
         a = 0.05
         sig = 0.2
         uflow = 1.0
         aperture = 0.01
         tanfac = np.tanh((y - y1) / a) - np.tanh((y - y2) / a)
         expfac = np.exp(-((y - y1) / sig) ** 2) + np.exp(-((y - y2) / sig) ** 2)
         rho = 1.0 + 0.5 * delrho * tanfac
         vx = uflow * (tanfac - 1.0)
         vy = aperture * np.sin(np.pi * x) * expfac
         pressure = 10.0

      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j], Uvec[MOMX, i, j], Uvec[MOMY, i, j], Uvec[MOMZ, i, j], Uvec[ENERGY, i, j] = conserved_from_primitive(
         rho, vx, vy, vz, pressure, gamma
      )


def hydro_ic_initial_state(config, dtype=np.float64):
   ic = validate_hydro_initial_condition(config.ic)
   bc = validate_bc(config.bc)
   Uvec = np.zeros((NVAR_HYDRO, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_hydro_ic_inplace(Uvec, config.nx, config.ny, config.ng, config.gamma, ic, config.delrho_factor)
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_mhd_ic_inplace(Uvec, nx, ny, ng, gamma, ic, delrho_factor):
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = -1.0 + (ii + 0.5) * 2.0 / nx
      y = -1.0 + (jj + 0.5) * 2.0 / ny
      rho = 1.0
      vx = 0.0
      vy = 0.0
      vz = 0.0
      pressure = 1.0
      bx = 0.0
      by = 0.0
      bz = 0.0

      if ic == 8:
         y1 = -0.5
         y2 = 0.5
         a = 0.05
         sig = 0.2
         uflow = 1.0
         aperture = 0.01
         tanfac = np.tanh((y - y1) / a) - np.tanh((y - y2) / a)
         expfac = np.exp(-((y - y1) / sig) ** 2) + np.exp(-((y - y2) / sig) ** 2)
         rho = 1.0 + 0.5 * delrho_factor * tanfac
         vx = uflow * (tanfac - 1.0)
         vy = aperture * np.sin(np.pi * x) * expfac
         pressure = 10.0
         bx = 1.5
         by = -1.5
         bz = 1.5
      elif ic == 10:
         if x < 0.0:
            rho = 1.0
            pressure = 1.0
            bx = 0.5
            by = 1.0
         else:
            rho = 0.125
            pressure = 0.1
            bx = 0.5
            by = -1.0
      elif ic == 11:
         rho = 1.0
         bx = 5.0
         if x < 0.0:
            pressure = 30.0
            by = 6.0
            bz = 6.0
         else:
            pressure = 1.0
            by = 0.7
            bz = 0.7
      elif ic == 12:
         rho = 1.0
         bx = 10.0
         if x < 0.0:
            pressure = 1000.0
            by = 7.0
            bz = 7.0
         else:
            pressure = 0.1
            by = 0.7
            bz = 0.7
      elif ic == 13:
         rho = 1.0
         pressure = 0.1
         bx = 10.0
         if x < 0.0:
            vx = 0.999
            by = 7.0
            bz = 7.0
         else:
            vx = -0.999
            by = -7.0
            bz = -7.0
      elif ic == 14:
         bx = 2.0
         if x < 0.0:
            rho = 1.08
            pressure = 0.95
            vx = 0.4
            vy = 0.3
            vz = 0.2
            by = 0.3
            bz = 0.3
         else:
            rho = 1.0
            pressure = 1.0
            vx = -0.45
            vy = -0.2
            vz = -0.2
            by = -0.7
            bz = 0.5
      elif ic == 15:
         rho = 1.0
         vx = 1.0
         vy = 1.0
         pressure = 1.0

      i = ng + ii
      j = ng + jj
      (
         Uvec[RHO, i, j],
         Uvec[MOMX, i, j],
         Uvec[MOMY, i, j],
         Uvec[MOMZ, i, j],
         Uvec[ENERGY, i, j],
         Uvec[BX, i, j],
         Uvec[BY, i, j],
         Uvec[BZ, i, j],
      ) = conserved_from_primitive_mhd(rho, vx, vy, vz, pressure, bx, by, bz, gamma)


@njit(cache=True, parallel=True)
def initialize_field_loop_vector_potential(A, nx, ny, ng):
   radius = 0.15
   amplitude = 1.0e-3
   for idx in prange((nx + 2 * ng) * (ny + 2 * ng)):
      i = idx // (ny + 2 * ng)
      j = idx - i * (ny + 2 * ng)
      ii = (i - ng) % nx
      jj = (j - ng) % ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      dx = x - 0.5
      dy = y - 0.5
      r = np.sqrt(dx * dx + dy * dy)
      A[i, j] = amplitude * max(radius - r, 0.0)


@njit(cache=True, parallel=True)
def initialize_field_loop_magnetic_field(Uvec, A, nx, ny, ng, gamma):
   dx = 1.0 / nx
   dy = 1.0 / ny
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   rho = 1.0
   vx = 1.0
   vy = 1.0
   vz = 0.0
   pressure = 1.0
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      bx = (A[i, j] + A[i - 1, j] - A[i, j - 1] - A[i - 1, j - 1]) * inv_2dy
      by = -(A[i, j] + A[i, j - 1] - A[i - 1, j] - A[i - 1, j - 1]) * inv_2dx
      bz = 0.0
      (
         Uvec[RHO, i, j],
         Uvec[MOMX, i, j],
         Uvec[MOMY, i, j],
         Uvec[MOMZ, i, j],
         Uvec[ENERGY, i, j],
         Uvec[BX, i, j],
         Uvec[BY, i, j],
         Uvec[BZ, i, j],
      ) = conserved_from_primitive_mhd(rho, vx, vy, vz, pressure, bx, by, bz, gamma)


def initialize_field_loop_advection_inplace(Uvec, nx, ny, ng, gamma):
   A = np.zeros((nx + 2 * ng, ny + 2 * ng), dtype=Uvec.dtype)
   initialize_field_loop_vector_potential(A, nx, ny, ng)
   initialize_field_loop_magnetic_field(Uvec, A, nx, ny, ng, gamma)


def mhd_ic_initial_state(config, dtype=np.float64):
   ic = validate_mhd_initial_condition(config.ic)
   bc = validate_bc(config.bc)
   Uvec = np.zeros((NVAR_MHD, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   if ic == 15:
      initialize_field_loop_advection_inplace(Uvec, config.nx, config.ny, config.ng, config.gamma)
   else:
      initialize_mhd_ic_inplace(Uvec, config.nx, config.ny, config.ng, config.gamma, ic, config.delrho_factor)
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_entropy_wave_inplace(Uvec, nx, ny, ng, gamma, rho0, pressure0, amplitude, vx, vy, kx, ky, time_value):
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * (x - vx * time_value) + ky * (y - vy * time_value))
      rho = rho0 + amplitude * np.sin(phase)
      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j] = rho
      Uvec[MOMX, i, j] = rho * vx
      Uvec[MOMY, i, j] = rho * vy
      Uvec[MOMZ, i, j] = 0.0
      Uvec[ENERGY, i, j] = pressure0 / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)


def entropy_wave_initial_state(config, dtype=np.float64):
   Uvec = np.zeros((NVAR_HYDRO, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_entropy_wave_inplace(
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
      0.0,
   )
   apply_boundaries(Uvec, config.ng, config.bc[0], config.bc[1], config.bc[2], config.bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_acoustic_wave_inplace(Uvec, nx, ny, ng, gamma, rho0, pressure0, amplitude, kx, ky, time_value):
   sound_speed = np.sqrt(gamma * pressure0 / rho0)
   k_norm = np.sqrt(kx * kx + ky * ky)
   nx_hat = kx / k_norm
   ny_hat = ky / k_norm
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      y = (jj + 0.5) / ny
      phase = 2.0 * np.pi * (kx * x + ky * y - sound_speed * k_norm * time_value)
      drho = amplitude * np.sin(phase)
      rho = rho0 + drho
      pressure = pressure0 + sound_speed * sound_speed * drho
      normal_velocity = sound_speed * drho / rho0
      vx = nx_hat * normal_velocity
      vy = ny_hat * normal_velocity
      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j] = rho
      Uvec[MOMX, i, j] = rho * vx
      Uvec[MOMY, i, j] = rho * vy
      Uvec[MOMZ, i, j] = 0.0
      Uvec[ENERGY, i, j] = pressure / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)


def acoustic_wave_initial_state(config, dtype=np.float64):
   Uvec = np.zeros((NVAR_HYDRO, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_acoustic_wave_inplace(
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
      0.0,
   )
   apply_boundaries(Uvec, config.ng, config.bc[0], config.bc[1], config.bc[2], config.bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_resistive_wave_inplace(Uvec, nx, ny, ng, gamma, rho0, pressure0, amplitude, wave_number):
   k = 2.0 * np.pi * wave_number
   energy = pressure0 / (gamma - 1.0) + 0.5 * amplitude * amplitude
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j] = rho0
      Uvec[MOMX, i, j] = 0.0
      Uvec[MOMY, i, j] = 0.0
      Uvec[MOMZ, i, j] = 0.0
      Uvec[ENERGY, i, j] = energy
      Uvec[BX, i, j] = 0.0
      Uvec[BY, i, j] = amplitude * np.sin(k * x)
      Uvec[BZ, i, j] = amplitude * np.cos(k * x)


def resistive_wave_initial_state(config, dtype=np.float64):
   bc = validate_bc(config.bc)
   Uvec = np.zeros((NVAR_MHD, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_resistive_wave_inplace(
      Uvec,
      config.nx,
      config.ny,
      config.ng,
      config.gamma,
      config.rho0,
      config.pressure0,
      config.amplitude,
      config.wave_number,
   )
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_hydro_characteristic_wave_inplace(
   Uvec, nx, ny, ng, gamma, mode, rho0, pressure0, vx0, vy0, vz0, amplitude, kx, ky, time_value
):
   speed = hydro_mode_speed(mode, rho0, pressure0, vx0, vy0, gamma, kx, ky)
   k_norm = np.sqrt(kx * kx + ky * ky)
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
      rho = rho0 + drho
      vx = vx0 + dvx
      vy = vy0 + dvy
      vz = vz0 + dvz
      pressure = pressure0 + dpressure
      i = ng + ii
      j = ng + jj
      Uvec[RHO, i, j], Uvec[MOMX, i, j], Uvec[MOMY, i, j], Uvec[MOMZ, i, j], Uvec[ENERGY, i, j] = conserved_from_primitive(
         rho, vx, vy, vz, pressure, gamma
      )


def hydro_characteristic_wave_initial_state(config, time_value=0.0, dtype=np.float64):
   bc = validate_bc(config.bc)
   validate_hydro_characteristic_mode(config.mode)
   Uvec = np.zeros((NVAR_HYDRO, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_hydro_characteristic_wave_inplace(
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
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_mhd_characteristic_wave_inplace(
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
   speed = mhd_mode_speed(mode, rho0, pressure0, vx0, vy0, bx0, by0, bz0, gamma, kx, ky)
   k_norm = np.sqrt(kx * kx + ky * ky)
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
      rho = rho0 + drho
      vx = vx0 + dvx
      vy = vy0 + dvy
      vz = vz0 + dvz
      pressure = pressure0 + dpressure
      bx = bx0 + dbx
      by = by0 + dby
      bz = bz0 + dbz
      i = ng + ii
      j = ng + jj
      (
         Uvec[RHO, i, j],
         Uvec[MOMX, i, j],
         Uvec[MOMY, i, j],
         Uvec[MOMZ, i, j],
         Uvec[ENERGY, i, j],
         Uvec[BX, i, j],
         Uvec[BY, i, j],
         Uvec[BZ, i, j],
      ) = conserved_from_primitive_mhd(rho, vx, vy, vz, pressure, bx, by, bz, gamma)


def mhd_characteristic_wave_initial_state(config, time_value=0.0, dtype=np.float64):
   bc = validate_bc(config.bc)
   validate_mhd_characteristic_mode(config.mode)
   Uvec = np.zeros((NVAR_MHD, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_mhd_characteristic_wave_inplace(
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
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec


@njit(cache=True, parallel=True)
def initialize_circular_alfven_wave_inplace(
   Uvec,
   nx,
   ny,
   ng,
   gamma,
   rho0,
   pressure0,
   bx0,
   amplitude,
   wave_number,
   direction,
   time_value,
):
   ca = abs(bx0) / np.sqrt(rho0)
   speed = direction * ca
   velocity_factor = -bx0 / (rho0 * speed)
   k = 2.0 * np.pi * wave_number
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      x = (ii + 0.5) / nx
      phase = k * (x - speed * time_value)
      by = amplitude * np.cos(phase)
      bz = amplitude * np.sin(phase)
      vy = velocity_factor * by
      vz = velocity_factor * bz
      i = ng + ii
      j = ng + jj
      (
         Uvec[RHO, i, j],
         Uvec[MOMX, i, j],
         Uvec[MOMY, i, j],
         Uvec[MOMZ, i, j],
         Uvec[ENERGY, i, j],
         Uvec[BX, i, j],
         Uvec[BY, i, j],
         Uvec[BZ, i, j],
      ) = conserved_from_primitive_mhd(rho0, 0.0, vy, vz, pressure0, bx0, by, bz, gamma)


def circular_alfven_wave_initial_state(config, time_value=0.0, dtype=np.float64):
   bc = validate_bc(config.bc)
   validate_circular_alfven_options(config)
   Uvec = np.zeros((NVAR_MHD, config.nx + 2 * config.ng, config.ny + 2 * config.ng), dtype=dtype)
   initialize_circular_alfven_wave_inplace(
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
   apply_boundaries(Uvec, config.ng, bc[0], bc[1], bc[2], bc[3])
   return Uvec
