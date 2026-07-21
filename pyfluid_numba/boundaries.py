from numba import njit, prange

from pyfluid_numba.options import *

@njit(cache=True)
def apply_periodic_boundaries(Uvec, ng):
   nvar, nxtot, nytot = Uvec.shape
   nx = nxtot - 2 * ng
   ny = nytot - 2 * ng

   for q in range(nvar):
      for g in range(ng):
         for jj in range(ny):
            j = ng + jj
            Uvec[q, g, j] = Uvec[q, nx + g, j]
            Uvec[q, ng + nx + g, j] = Uvec[q, ng + g, j]

   for q in range(nvar):
      for i in range(nxtot):
         for g in range(ng):
            Uvec[q, i, g] = Uvec[q, i, ny + g]
            Uvec[q, i, ng + ny + g] = Uvec[q, i, ng + g]


@njit(cache=True, parallel=True)
def apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top):
   nvar, nxtot, nytot = Uvec.shape
   nx = nxtot - 2 * ng
   ny = nytot - 2 * ng

   for idx in prange(nvar * ng * ny):
      q = idx // (ng * ny)
      rem = idx - q * ng * ny
      g = rem // ny
      jj = rem - g * ny
      j = ng + jj
      if bc_left == BC_PERIODIC:
         Uvec[q, g, j] = Uvec[q, nx + g, j]
      elif bc_left == BC_REFLECTIVE:
         Uvec[q, g, j] = Uvec[q, 2 * ng - 1 - g, j]
         if q == MOMX:
            Uvec[q, g, j] = -Uvec[q, g, j]
      else:
         Uvec[q, g, j] = Uvec[q, ng, j]
      if bc_right == BC_PERIODIC:
         Uvec[q, ng + nx + g, j] = Uvec[q, ng + g, j]
      elif bc_right == BC_REFLECTIVE:
         Uvec[q, ng + nx + g, j] = Uvec[q, ng + nx - 1 - g, j]
         if q == MOMX:
            Uvec[q, ng + nx + g, j] = -Uvec[q, ng + nx + g, j]
      else:
         Uvec[q, ng + nx + g, j] = Uvec[q, ng + nx - 1, j]

   for idx in prange(nvar * nxtot * ng):
      q = idx // (nxtot * ng)
      rem = idx - q * nxtot * ng
      i = rem // ng
      g = rem - i * ng
      if bc_bottom == BC_PERIODIC:
         Uvec[q, i, g] = Uvec[q, i, ny + g]
      elif bc_bottom == BC_REFLECTIVE:
         Uvec[q, i, g] = Uvec[q, i, 2 * ng - 1 - g]
         if q == MOMY:
            Uvec[q, i, g] = -Uvec[q, i, g]
      else:
         Uvec[q, i, g] = Uvec[q, i, ng]
      if bc_top == BC_PERIODIC:
         Uvec[q, i, ng + ny + g] = Uvec[q, i, ng + g]
      elif bc_top == BC_REFLECTIVE:
         Uvec[q, i, ng + ny + g] = Uvec[q, i, ng + ny - 1 - g]
         if q == MOMY:
            Uvec[q, i, ng + ny + g] = -Uvec[q, i, ng + ny + g]
      else:
         Uvec[q, i, ng + ny + g] = Uvec[q, i, ng + ny - 1]
