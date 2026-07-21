from numba import njit, prange
import numpy as np

from pyfluid_numba.options import *

@njit(cache=True, inline="always")
def pressure_state(rho, mx, my, mz, energy, gamma):
   inv_rho = 1.0 / rho
   kinetic = 0.5 * (mx * mx + my * my + mz * mz) * inv_rho
   return (gamma - 1.0) * (energy - kinetic)


@njit(cache=True, inline="always")
def pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma):
   inv_rho = 1.0 / rho
   kinetic = 0.5 * (mx * mx + my * my + mz * mz) * inv_rho
   magnetic = 0.5 * (bx * bx + by * by + bz * bz)
   return (gamma - 1.0) * (energy - kinetic - magnetic)


@njit(cache=True)
def primitive_pressure(Uvec, i, j, gamma):
   rho = Uvec[RHO, i, j]
   if Uvec.shape[0] > NVAR_HYDRO:
      return pressure_state_mhd(
         rho,
         Uvec[MOMX, i, j],
         Uvec[MOMY, i, j],
         Uvec[MOMZ, i, j],
         Uvec[ENERGY, i, j],
         Uvec[BX, i, j],
         Uvec[BY, i, j],
         Uvec[BZ, i, j],
         gamma,
      )
   return pressure_state(rho, Uvec[MOMX, i, j], Uvec[MOMY, i, j], Uvec[MOMZ, i, j], Uvec[ENERGY, i, j], gamma)


@njit(cache=True, inline="always")
def fast_speed(rho, pressure, bx, by, bz, normal_b, gamma):
   sound2 = gamma * pressure / rho
   alfven2 = (bx * bx + by * by + bz * bz) / rho
   normal_alfven2 = normal_b * normal_b / rho
   discr = (sound2 + alfven2) * (sound2 + alfven2) - 4.0 * sound2 * normal_alfven2
   if discr < 0.0 and discr > -MAGNETIC_EPS:
      discr = 0.0
   return np.sqrt(0.5 * (sound2 + alfven2 + np.sqrt(discr)))


@njit(cache=True, inline="always")
def flux_x_cell(u0, u1, u2, u3, u4, gamma):
   rho = u0
   vx = u1 / rho
   p = pressure_state(rho, u1, u2, u3, u4, gamma)
   return (
      u1,
      u1 * vx + p,
      u2 * vx,
      u3 * vx,
      (u4 + p) * vx,
      abs(vx) + np.sqrt(gamma * p / rho),
   )


@njit(cache=True, inline="always")
def flux_y_cell(u0, u1, u2, u3, u4, gamma):
   rho = u0
   vy = u2 / rho
   p = pressure_state(rho, u1, u2, u3, u4, gamma)
   return (
      u2,
      u1 * vy,
      u2 * vy + p,
      u3 * vy,
      (u4 + p) * vy,
      abs(vy) + np.sqrt(gamma * p / rho),
   )


@njit(cache=True, inline="always")
def flux_x_mhd_cell(u0, u1, u2, u3, u4, u5, u6, u7, gamma):
   rho = u0
   vx = u1 / rho
   vy = u2 / rho
   vz = u3 / rho
   bx = u5
   by = u6
   bz = u7
   pressure = pressure_state_mhd(rho, u1, u2, u3, u4, bx, by, bz, gamma)
   magnetic = 0.5 * (bx * bx + by * by + bz * bz)
   vdotb = vx * bx + vy * by + vz * bz
   return (
      u1,
      u1 * vx + pressure + magnetic - bx * bx,
      u2 * vx - bx * by,
      u3 * vx - bx * bz,
      (u4 + pressure + magnetic) * vx - bx * vdotb,
      0.0,
      by * vx - bx * vy,
      bz * vx - bx * vz,
      abs(vx) + fast_speed(rho, pressure, bx, by, bz, bx, gamma),
   )


@njit(cache=True, inline="always")
def flux_y_mhd_cell(u0, u1, u2, u3, u4, u5, u6, u7, gamma):
   rho = u0
   vx = u1 / rho
   vy = u2 / rho
   vz = u3 / rho
   bx = u5
   by = u6
   bz = u7
   pressure = pressure_state_mhd(rho, u1, u2, u3, u4, bx, by, bz, gamma)
   magnetic = 0.5 * (bx * bx + by * by + bz * bz)
   vdotb = vx * bx + vy * by + vz * bz
   return (
      u2,
      u1 * vy - by * bx,
      u2 * vy + pressure + magnetic - by * by,
      u3 * vy - by * bz,
      (u4 + pressure + magnetic) * vy - by * vdotb,
      bx * vy - by * vx,
      0.0,
      bz * vy - by * vz,
      abs(vy) + fast_speed(rho, pressure, bx, by, bz, by, gamma),
   )


@njit(cache=True, inline="always")
def conserved_from_primitive(rho, vx, vy, vz, pressure, gamma):
   return (
      rho,
      rho * vx,
      rho * vy,
      rho * vz,
      pressure / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz),
   )


@njit(cache=True, inline="always")
def conserved_from_primitive_mhd(rho, vx, vy, vz, pressure, bx, by, bz, gamma):
   return (
      rho,
      rho * vx,
      rho * vy,
      rho * vz,
      pressure / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz) + 0.5 * (bx * bx + by * by + bz * bz),
      bx,
      by,
      bz,
   )


@njit(cache=True)
def velocity_at(Uvec, component, i, j):
   return Uvec[component, i, j] / Uvec[RHO, i, j]


@njit(cache=True, parallel=True)
def max_signal_speed_array(Uvec, nx, ny, ng, gamma):
   max_speed = 0.0
   mhd = Uvec.shape[0] > NVAR_HYDRO
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
      if mhd:
         bx = Uvec[BX, i, j]
         by = Uvec[BY, i, j]
         bz = Uvec[BZ, i, j]
         pressure = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
         vx_speed = abs(mx / rho) + fast_speed(rho, pressure, bx, by, bz, bx, gamma)
         vy_speed = abs(my / rho) + fast_speed(rho, pressure, bx, by, bz, by, gamma)
      else:
         pressure = pressure_state(rho, mx, my, mz, energy, gamma)
         sound_speed = np.sqrt(gamma * pressure / rho)
         vx_speed = abs(mx / rho) + sound_speed
         vy_speed = abs(my / rho) + sound_speed
      speed = vx_speed if vx_speed > vy_speed else vy_speed
      max_speed = max(max_speed, speed)
   return max_speed


@njit(cache=True, parallel=True)
def max_signal_frequency_array(Uvec, nx, ny, ng, gamma, dx, dy):
   max_frequency = 0.0
   mhd = Uvec.shape[0] > NVAR_HYDRO
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
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
      if mhd:
         bx = Uvec[BX, i, j]
         by = Uvec[BY, i, j]
         bz = Uvec[BZ, i, j]
         pressure = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
         x_speed = abs(mx / rho) + fast_speed(rho, pressure, bx, by, bz, bx, gamma)
         y_speed = abs(my / rho) + fast_speed(rho, pressure, bx, by, bz, by, gamma)
      else:
         pressure = pressure_state(rho, mx, my, mz, energy, gamma)
         sound_speed = np.sqrt(gamma * pressure / rho)
         x_speed = abs(mx / rho) + sound_speed
         y_speed = abs(my / rho) + sound_speed
      frequency = x_speed * inv_dx + y_speed * inv_dy
      max_frequency = max(max_frequency, frequency)
   return max_frequency
