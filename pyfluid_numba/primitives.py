from numba import njit, prange

from pyfluid_numba.options import *
from pyfluid_numba.physics import pressure_state, pressure_state_mhd


@njit(cache=True, parallel=True)
def fill_primitive_cache(Uvec, Vcache, gamma):
   nvar, nxtot, nytot = Uvec.shape
   mhd = nvar > NVAR_HYDRO
   for idx in prange(nxtot * nytot):
      i = idx // nytot
      j = idx - i * nytot
      rho = Uvec[RHO, i, j]
      mx = Uvec[MOMX, i, j]
      my = Uvec[MOMY, i, j]
      mz = Uvec[MOMZ, i, j]
      energy = Uvec[ENERGY, i, j]
      Vcache[RHO, i, j] = rho
      Vcache[MOMX, i, j] = mx / rho
      Vcache[MOMY, i, j] = my / rho
      Vcache[MOMZ, i, j] = mz / rho
      if mhd:
         bx = Uvec[BX, i, j]
         by = Uvec[BY, i, j]
         bz = Uvec[BZ, i, j]
         Vcache[ENERGY, i, j] = pressure_state_mhd(rho, mx, my, mz, energy, bx, by, bz, gamma)
         Vcache[BX, i, j] = bx
         Vcache[BY, i, j] = by
         Vcache[BZ, i, j] = bz
      else:
         Vcache[ENERGY, i, j] = pressure_state(rho, mx, my, mz, energy, gamma)
