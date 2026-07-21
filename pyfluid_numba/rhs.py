from numba import literally, njit, prange

from pyfluid_numba.options import *
from pyfluid_numba.diffusion import add_resistive_emf_curl_rhs, fill_resistive_ez_corners
from pyfluid_numba.physics import velocity_at
from pyfluid_numba.primitives import fill_primitive_cache
from pyfluid_numba.reconstruction import (
   reconstruct_hydro_characteristic_cached,
   reconstruct_component,
   reconstruct_faces,
   reconstruct_faces_from_primitive_cache,
   reconstruct_primitive,
   reconstruct_primitive_cached,
)
from pyfluid_numba.riemann import (
   hll_flux_x_cell,
   hll_flux_y_cell,
   hll_flux_x_mhd_cell,
   hll_flux_y_mhd_cell,
   hlld_flux_x_mhd_cell,
   hlld_flux_y_mhd_cell,
   hllc_flux_x_cell,
   hllc_flux_y_cell,
   llf_flux_x_mhd_cell,
   llf_flux_y_mhd_cell,
   llf_flux_x_cell,
   llf_flux_y_cell,
)

@njit(cache=True, parallel=True)
def compute_fluxes_hydro(Uvec, Fx, Fy, gamma, ng, recon, recon_vars, limiter, riemann):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   nxp1, ny = Fx.shape[1], Fx.shape[2]
   nx, nyp1 = Fy.shape[1], Fy.shape[2]
   primitive_recon = recon == RECON_WENOZ or recon_vars == RECON_VARS_PRIMITIVE

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng - 1 + ii
      j = ng + jj

      if primitive_recon:
         pl0 = reconstruct_primitive(Uvec, 0, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
         pl1 = reconstruct_primitive(Uvec, 1, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
         pl2 = reconstruct_primitive(Uvec, 2, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
         pl3 = reconstruct_primitive(Uvec, 3, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
         pl4 = reconstruct_primitive(Uvec, 4, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
         pr0 = reconstruct_primitive(Uvec, 0, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
         pr1 = reconstruct_primitive(Uvec, 1, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
         pr2 = reconstruct_primitive(Uvec, 2, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
         pr3 = reconstruct_primitive(Uvec, 3, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
         pr4 = reconstruct_primitive(Uvec, 4, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
         l0 = pl0
         l1 = pl0 * pl1
         l2 = pl0 * pl2
         l3 = pl0 * pl3
         l4 = pl4 / (gamma - 1.0) + 0.5 * pl0 * (pl1 * pl1 + pl2 * pl2 + pl3 * pl3)
         r0 = pr0
         r1 = pr0 * pr1
         r2 = pr0 * pr2
         r3 = pr0 * pr3
         r4 = pr4 / (gamma - 1.0) + 0.5 * pr0 * (pr1 * pr1 + pr2 * pr2 + pr3 * pr3)
      else:
         l0 = reconstruct_component(Uvec, 0, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         l1 = reconstruct_component(Uvec, 1, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         l2 = reconstruct_component(Uvec, 2, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         l3 = reconstruct_component(Uvec, 3, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         l4 = reconstruct_component(Uvec, 4, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         r0 = reconstruct_component(Uvec, 0, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         r1 = reconstruct_component(Uvec, 1, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         r2 = reconstruct_component(Uvec, 2, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         r3 = reconstruct_component(Uvec, 3, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         r4 = reconstruct_component(Uvec, 4, i, j, recon, limiter, AXIS_X, SIDE_UPPER)

      if riemann == RIEMANN_HLLC:
         f0, f1, f2, f3, f4, _ = hllc_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
      elif riemann == RIEMANN_HLL:
         f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, False)
      elif riemann == RIEMANN_HLLE:
         f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, True)
      else:
         f0, f1, f2, f3, f4, _ = llf_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)

      Fx[0, ii, jj] = f0
      Fx[1, ii, jj] = f1
      Fx[2, ii, jj] = f2
      Fx[3, ii, jj] = f3
      Fx[4, ii, jj] = f4

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      i = ng + ii
      j = ng - 1 + jj

      if primitive_recon:
         pb0 = reconstruct_primitive(Uvec, 0, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb1 = reconstruct_primitive(Uvec, 1, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb2 = reconstruct_primitive(Uvec, 2, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb3 = reconstruct_primitive(Uvec, 3, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb4 = reconstruct_primitive(Uvec, 4, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
         pt0 = reconstruct_primitive(Uvec, 0, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt1 = reconstruct_primitive(Uvec, 1, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt2 = reconstruct_primitive(Uvec, 2, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt3 = reconstruct_primitive(Uvec, 3, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt4 = reconstruct_primitive(Uvec, 4, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
         b0 = pb0
         b1 = pb0 * pb1
         b2 = pb0 * pb2
         b3 = pb0 * pb3
         b4 = pb4 / (gamma - 1.0) + 0.5 * pb0 * (pb1 * pb1 + pb2 * pb2 + pb3 * pb3)
         t0 = pt0
         t1 = pt0 * pt1
         t2 = pt0 * pt2
         t3 = pt0 * pt3
         t4 = pt4 / (gamma - 1.0) + 0.5 * pt0 * (pt1 * pt1 + pt2 * pt2 + pt3 * pt3)
      else:
         b0 = reconstruct_component(Uvec, 0, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         b1 = reconstruct_component(Uvec, 1, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         b2 = reconstruct_component(Uvec, 2, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         b3 = reconstruct_component(Uvec, 3, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         b4 = reconstruct_component(Uvec, 4, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         t0 = reconstruct_component(Uvec, 0, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         t1 = reconstruct_component(Uvec, 1, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         t2 = reconstruct_component(Uvec, 2, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         t3 = reconstruct_component(Uvec, 3, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         t4 = reconstruct_component(Uvec, 4, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)

      if riemann == RIEMANN_HLLC:
         f0, f1, f2, f3, f4, _ = hllc_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
      elif riemann == RIEMANN_HLL:
         f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, False)
      elif riemann == RIEMANN_HLLE:
         f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, True)
      else:
         f0, f1, f2, f3, f4, _ = llf_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)

      Fy[0, ii, jj] = f0
      Fy[1, ii, jj] = f1
      Fy[2, ii, jj] = f2
      Fy[3, ii, jj] = f3
      Fy[4, ii, jj] = f4


@njit(cache=True)
def fill_fluxes(Uvec, scratch, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   if Uvec.shape[0] == NVAR_HYDRO:
      if recon == RECON_WENOZ or recon == RECON_WENOZ_CHAR or recon_vars == RECON_VARS_PRIMITIVE:
         fill_primitive_cache(Uvec, scratch, gamma)
         compute_fluxes_hydro_from_primitive_cache(scratch, Fx, Fy, gamma, ng, recon, limiter, riemann)
      else:
         compute_fluxes_hydro(Uvec, Fx, Fy, gamma, ng, recon, recon_vars, limiter, riemann)
   else:
      if recon == RECON_WENOZ or recon == RECON_WENOZ_CHAR or recon_vars == RECON_VARS_PRIMITIVE:
         fill_primitive_cache(Uvec, scratch, gamma)
         reconstruct_faces_from_primitive_cache(scratch, UxL, UxR, UyB, UyT, gamma, ng, recon, limiter)
      else:
         reconstruct_faces(Uvec, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter)
      compute_fluxes(UxL, UxR, UyB, UyT, Fx, Fy, gamma, riemann)
   if ct_method == CT_FLUX_TOTH and Uvec.shape[0] > NVAR_HYDRO:
      flux_ct_toth_correction(Fx, Fy, UxL, UyB)


@njit(cache=True, parallel=True)
def compute_fluxes_hydro_from_primitive_cache(Vcache, Fx, Fy, gamma, ng, recon, limiter, riemann):
   recon = literally(recon)
   limiter = literally(limiter)
   riemann = literally(riemann)
   nxp1, ny = Fx.shape[1], Fx.shape[2]
   nx, nyp1 = Fy.shape[1], Fy.shape[2]

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng - 1 + ii
      j = ng + jj

      if recon == RECON_WENOZ_CHAR:
         pl0, pl1, pl2, pl3, pl4, pr0, pr1, pr2, pr3, pr4 = reconstruct_hydro_characteristic_cached(
            Vcache, i, j, gamma, AXIS_X
         )
      else:
         pl0 = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         pl1 = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         pl2 = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         pl3 = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         pl4 = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
         pr0 = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         pr1 = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         pr2 = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         pr3 = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
         pr4 = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_X, SIDE_UPPER)
      l0 = pl0
      l1 = pl0 * pl1
      l2 = pl0 * pl2
      l3 = pl0 * pl3
      l4 = pl4 / (gamma - 1.0) + 0.5 * pl0 * (pl1 * pl1 + pl2 * pl2 + pl3 * pl3)
      r0 = pr0
      r1 = pr0 * pr1
      r2 = pr0 * pr2
      r3 = pr0 * pr3
      r4 = pr4 / (gamma - 1.0) + 0.5 * pr0 * (pr1 * pr1 + pr2 * pr2 + pr3 * pr3)

      if riemann == RIEMANN_HLLC:
         f0, f1, f2, f3, f4, _ = hllc_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
      elif riemann == RIEMANN_HLL:
         f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, False)
      elif riemann == RIEMANN_HLLE:
         f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, True)
      else:
         f0, f1, f2, f3, f4, _ = llf_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)

      Fx[0, ii, jj] = f0
      Fx[1, ii, jj] = f1
      Fx[2, ii, jj] = f2
      Fx[3, ii, jj] = f3
      Fx[4, ii, jj] = f4

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      i = ng + ii
      j = ng - 1 + jj

      if recon == RECON_WENOZ_CHAR:
         pb0, pb1, pb2, pb3, pb4, pt0, pt1, pt2, pt3, pt4 = reconstruct_hydro_characteristic_cached(
            Vcache, i, j, gamma, AXIS_Y
         )
      else:
         pb0 = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb1 = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb2 = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb3 = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         pb4 = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
         pt0 = reconstruct_primitive_cached(Vcache, 0, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt1 = reconstruct_primitive_cached(Vcache, 1, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt2 = reconstruct_primitive_cached(Vcache, 2, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt3 = reconstruct_primitive_cached(Vcache, 3, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
         pt4 = reconstruct_primitive_cached(Vcache, 4, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)
      b0 = pb0
      b1 = pb0 * pb1
      b2 = pb0 * pb2
      b3 = pb0 * pb3
      b4 = pb4 / (gamma - 1.0) + 0.5 * pb0 * (pb1 * pb1 + pb2 * pb2 + pb3 * pb3)
      t0 = pt0
      t1 = pt0 * pt1
      t2 = pt0 * pt2
      t3 = pt0 * pt3
      t4 = pt4 / (gamma - 1.0) + 0.5 * pt0 * (pt1 * pt1 + pt2 * pt2 + pt3 * pt3)

      if riemann == RIEMANN_HLLC:
         f0, f1, f2, f3, f4, _ = hllc_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
      elif riemann == RIEMANN_HLL:
         f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, False)
      elif riemann == RIEMANN_HLLE:
         f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, True)
      else:
         f0, f1, f2, f3, f4, _ = llf_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)

      Fy[0, ii, jj] = f0
      Fy[1, ii, jj] = f1
      Fy[2, ii, jj] = f2
      Fy[3, ii, jj] = f3
      Fy[4, ii, jj] = f4


@njit(cache=True, parallel=True)
def compute_fluxes(UxL, UxR, UyB, UyT, Fx, Fy, gamma, riemann):
   riemann = literally(riemann)
   nxp1, ny = Fx.shape[1], Fx.shape[2]
   nx, nyp1 = Fy.shape[1], Fy.shape[2]
   mhd = Fx.shape[0] > NVAR_HYDRO

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      if mhd:
         l0, l1, l2, l3 = UxL[0, ii, jj], UxL[1, ii, jj], UxL[2, ii, jj], UxL[3, ii, jj]
         l4, l5, l6, l7 = UxL[4, ii, jj], UxL[5, ii, jj], UxL[6, ii, jj], UxL[7, ii, jj]
         r0, r1, r2, r3 = UxR[0, ii, jj], UxR[1, ii, jj], UxR[2, ii, jj], UxR[3, ii, jj]
         r4, r5, r6, r7 = UxR[4, ii, jj], UxR[5, ii, jj], UxR[6, ii, jj], UxR[7, ii, jj]
         if riemann == RIEMANN_HLL:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hll_flux_x_mhd_cell(
               l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma, False
            )
         elif riemann == RIEMANN_HLLE:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hll_flux_x_mhd_cell(
               l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma, True
            )
         elif riemann == RIEMANN_HLLD:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hlld_flux_x_mhd_cell(
               l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma
            )
         else:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = llf_flux_x_mhd_cell(
               l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma
            )
         Fx[0, ii, jj] = f0
         Fx[1, ii, jj] = f1
         Fx[2, ii, jj] = f2
         Fx[3, ii, jj] = f3
         Fx[4, ii, jj] = f4
         Fx[5, ii, jj] = f5
         Fx[6, ii, jj] = f6
         Fx[7, ii, jj] = f7
      else:
         l0, l1, l2, l3, l4 = UxL[0, ii, jj], UxL[1, ii, jj], UxL[2, ii, jj], UxL[3, ii, jj], UxL[4, ii, jj]
         r0, r1, r2, r3, r4 = UxR[0, ii, jj], UxR[1, ii, jj], UxR[2, ii, jj], UxR[3, ii, jj], UxR[4, ii, jj]
         if riemann == RIEMANN_HLLC:
            f0, f1, f2, f3, f4, _ = hllc_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
         elif riemann == RIEMANN_HLL:
            f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, False)
         elif riemann == RIEMANN_HLLE:
            f0, f1, f2, f3, f4, _ = hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, True)
         else:
            f0, f1, f2, f3, f4, _ = llf_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
         Fx[0, ii, jj] = f0
         Fx[1, ii, jj] = f1
         Fx[2, ii, jj] = f2
         Fx[3, ii, jj] = f3
         Fx[4, ii, jj] = f4

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      if mhd:
         b0, b1, b2, b3 = UyB[0, ii, jj], UyB[1, ii, jj], UyB[2, ii, jj], UyB[3, ii, jj]
         b4, b5, b6, b7 = UyB[4, ii, jj], UyB[5, ii, jj], UyB[6, ii, jj], UyB[7, ii, jj]
         t0, t1, t2, t3 = UyT[0, ii, jj], UyT[1, ii, jj], UyT[2, ii, jj], UyT[3, ii, jj]
         t4, t5, t6, t7 = UyT[4, ii, jj], UyT[5, ii, jj], UyT[6, ii, jj], UyT[7, ii, jj]
         if riemann == RIEMANN_HLL:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hll_flux_y_mhd_cell(
               b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma, False
            )
         elif riemann == RIEMANN_HLLE:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hll_flux_y_mhd_cell(
               b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma, True
            )
         elif riemann == RIEMANN_HLLD:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = hlld_flux_y_mhd_cell(
               b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma
            )
         else:
            f0, f1, f2, f3, f4, f5, f6, f7, _ = llf_flux_y_mhd_cell(
               b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma
            )
         Fy[0, ii, jj] = f0
         Fy[1, ii, jj] = f1
         Fy[2, ii, jj] = f2
         Fy[3, ii, jj] = f3
         Fy[4, ii, jj] = f4
         Fy[5, ii, jj] = f5
         Fy[6, ii, jj] = f6
         Fy[7, ii, jj] = f7
      else:
         b0, b1, b2, b3, b4 = UyB[0, ii, jj], UyB[1, ii, jj], UyB[2, ii, jj], UyB[3, ii, jj], UyB[4, ii, jj]
         t0, t1, t2, t3, t4 = UyT[0, ii, jj], UyT[1, ii, jj], UyT[2, ii, jj], UyT[3, ii, jj], UyT[4, ii, jj]
         if riemann == RIEMANN_HLLC:
            f0, f1, f2, f3, f4, _ = hllc_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
         elif riemann == RIEMANN_HLL:
            f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, False)
         elif riemann == RIEMANN_HLLE:
            f0, f1, f2, f3, f4, _ = hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, True)
         else:
            f0, f1, f2, f3, f4, _ = llf_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
         Fy[0, ii, jj] = f0
         Fy[1, ii, jj] = f1
         Fy[2, ii, jj] = f2
         Fy[3, ii, jj] = f3
         Fy[4, ii, jj] = f4


@njit(cache=True, parallel=True)
def compute_rhs_from_fluxes(rhs, Fx, Fy, dx, dy, ng):
   nvar = rhs.shape[0]
   nx = rhs.shape[1] - 2 * ng
   ny = rhs.shape[2] - 2 * ng
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      for q in range(nvar):
         rhs[q, i, j] = -(
            (Fx[q, ii + 1, jj] - Fx[q, ii, jj]) * inv_dx
            + (Fy[q, ii, jj + 1] - Fy[q, ii, jj]) * inv_dy
         )


@njit(cache=True, parallel=True)
def flux_ct_toth_correction(Fx, Fy, UxL, UyB):
   # Toth-style cell-centered flux-CT. This is the arithmetic edge-EMF average
   # written directly as corrected face fluxes F_x(By) and F_y(Bx).
   nxp1 = Fx.shape[1]
   ny = Fx.shape[2]
   nx = Fy.shape[1]
   nyp1 = Fy.shape[2]
   fx_by = UxL[0]
   fy_bx = UyB[0]

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      jm = jj - 1
      if jm < 0:
         jm = ny - 1
      jp = jj + 1
      if jp >= ny:
         jp = 0
      il = ii - 1
      if il < 0:
         il = nx - 1
      ir = ii
      if ir >= nx:
         ir = 0
      fx_by[ii, jj] = 0.125 * (
         2.0 * Fx[BY, ii, jj]
         + Fx[BY, ii, jm]
         + Fx[BY, ii, jp]
         - (Fy[BX, il, jj] + Fy[BX, ir, jj] + Fy[BX, il, jj + 1] + Fy[BX, ir, jj + 1])
      )

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      im = ii - 1
      if im < 0:
         im = nx - 1
      ip = ii + 1
      if ip >= nx:
         ip = 0
      jb = jj - 1
      if jb < 0:
         jb = ny - 1
      jt = jj
      if jt >= ny:
         jt = 0
      fy_bx[ii, jj] = 0.125 * (
         2.0 * Fy[BX, ii, jj]
         + Fy[BX, im, jj]
         + Fy[BX, ip, jj]
         - (Fx[BY, ii, jb] + Fx[BY, ii + 1, jb] + Fx[BY, ii, jt] + Fx[BY, ii + 1, jt])
      )

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      Fx[BY, ii, jj] = fx_by[ii, jj]

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      Fy[BX, ii, jj] = fy_bx[ii, jj]


@njit(cache=True, parallel=True)
def add_diffusion_rhs(rhs, Uvec, Ez, dx, dy, ng, viscosity, resistivity):
   nx = Uvec.shape[1] - 2 * ng
   ny = Uvec.shape[2] - 2 * ng
   inv_dx2 = 1.0 / (dx * dx)
   inv_dy2 = 1.0 / (dy * dy)
   mhd = Uvec.shape[0] > NVAR_HYDRO
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      rho = Uvec[RHO, i, j]
      vx = velocity_at(Uvec, MOMX, i, j)
      vy = velocity_at(Uvec, MOMY, i, j)
      vz = velocity_at(Uvec, MOMZ, i, j)
      lap_vx = (
         (velocity_at(Uvec, MOMX, i + 1, j) - 2.0 * vx + velocity_at(Uvec, MOMX, i - 1, j)) * inv_dx2
         + (velocity_at(Uvec, MOMX, i, j + 1) - 2.0 * vx + velocity_at(Uvec, MOMX, i, j - 1)) * inv_dy2
      )
      lap_vy = (
         (velocity_at(Uvec, MOMY, i + 1, j) - 2.0 * vy + velocity_at(Uvec, MOMY, i - 1, j)) * inv_dx2
         + (velocity_at(Uvec, MOMY, i, j + 1) - 2.0 * vy + velocity_at(Uvec, MOMY, i, j - 1)) * inv_dy2
      )
      lap_vz = (
         (velocity_at(Uvec, MOMZ, i + 1, j) - 2.0 * vz + velocity_at(Uvec, MOMZ, i - 1, j)) * inv_dx2
         + (velocity_at(Uvec, MOMZ, i, j + 1) - 2.0 * vz + velocity_at(Uvec, MOMZ, i, j - 1)) * inv_dy2
      )
      mom_rhs_x = viscosity * rho * lap_vx
      mom_rhs_y = viscosity * rho * lap_vy
      mom_rhs_z = viscosity * rho * lap_vz
      rhs[MOMX, i, j] += mom_rhs_x
      rhs[MOMY, i, j] += mom_rhs_y
      rhs[MOMZ, i, j] += mom_rhs_z
      if mhd and resistivity > 0.0:
         lap_bz = (Uvec[BZ, i + 1, j] - 2.0 * Uvec[BZ, i, j] + Uvec[BZ, i - 1, j]) * inv_dx2 + (
            Uvec[BZ, i, j + 1] - 2.0 * Uvec[BZ, i, j] + Uvec[BZ, i, j - 1]
         ) * inv_dy2
         rhs[BZ, i, j] += resistivity * lap_bz
   if mhd and resistivity > 0.0:
      fill_resistive_ez_corners(Uvec, Ez, dx, dy, ng, resistivity)
      add_resistive_emf_curl_rhs(rhs, Ez, dx, dy, ng)


@njit(cache=True)
def fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion = literally(diffusion)
   fill_fluxes(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
   compute_rhs_from_fluxes(rhs, Fx, Fy, dx, dy, ng)
   if diffusion:
      add_diffusion_rhs(rhs, Uvec, Ez, dx, dy, ng, viscosity, resistivity)
