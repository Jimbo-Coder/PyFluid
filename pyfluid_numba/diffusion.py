from numba import njit, prange

from pyfluid_numba.boundaries import apply_boundaries
from pyfluid_numba.options import BX, BY, BZ, ENERGY, MOMX, MOMY, MOMZ, NVAR_HYDRO, RHO


@njit(cache=True, parallel=True)
def copy_interior_local(src, dst, ng):
   nvar = src.shape[0]
   nx = src.shape[1] - 2 * ng
   ny = src.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      dst[q, ng + ii, ng + jj] = src[q, ng + ii, ng + jj]


@njit(cache=True, parallel=True)
def fill_resistive_ez_corners(state, ez, dx, dy, ng, resistivity):
   nx = state.shape[1] - 2 * ng
   ny = state.shape[2] - 2 * ng
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   for idx in prange((nx + 1) * (ny + 1)):
      ii = idx // (ny + 1)
      jj = idx - ii * (ny + 1)
      ic = ng - 1 + ii
      jc = ng - 1 + jj
      dby_dx = (state[BY, ic + 1, jc + 1] + state[BY, ic + 1, jc] - state[BY, ic, jc + 1] - state[BY, ic, jc]) * inv_2dx
      dbx_dy = (state[BX, ic + 1, jc + 1] + state[BX, ic, jc + 1] - state[BX, ic + 1, jc] - state[BX, ic, jc]) * inv_2dy
      ez[ii, jj] = resistivity * (dby_dx - dbx_dy)


@njit(cache=True, parallel=True)
def apply_resistive_emf_curl_to_state(base, out, ez, dx, dy, dt, ng):
   nx = out.shape[1] - 2 * ng
   ny = out.shape[2] - 2 * ng
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      d_ez_dy = (ez[ii, jj + 1] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii + 1, jj]) * inv_2dy
      d_ez_dx = (ez[ii + 1, jj] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii, jj + 1]) * inv_2dx
      out[BX, i, j] = base[BX, i, j] - dt * d_ez_dy
      out[BY, i, j] = base[BY, i, j] + dt * d_ez_dx


@njit(cache=True, parallel=True)
def add_resistive_emf_curl_rhs(rhs, ez, dx, dy, ng):
   nx = rhs.shape[1] - 2 * ng
   ny = rhs.shape[2] - 2 * ng
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      d_ez_dy = (ez[ii, jj + 1] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii + 1, jj]) * inv_2dy
      d_ez_dx = (ez[ii + 1, jj] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii, jj + 1]) * inv_2dx
      rhs[BX, i, j] += -d_ez_dy
      rhs[BY, i, j] += d_ez_dx


@njit(cache=True, parallel=True)
def relax_resistive_transverse_magnetic(base, solution, ez, dx, dy, dt, ng, resistivity, omega):
   nx = solution.shape[1] - 2 * ng
   ny = solution.shape[2] - 2 * ng
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   denom_bx = 1.0 + dt * resistivity / (dy * dy)
   denom_by = 1.0 + dt * resistivity / (dx * dx)
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      d_ez_dy = (ez[ii, jj + 1] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii + 1, jj]) * inv_2dy
      d_ez_dx = (ez[ii + 1, jj] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii, jj + 1]) * inv_2dx
      residual_bx = solution[BX, i, j] - base[BX, i, j] + dt * d_ez_dy
      residual_by = solution[BY, i, j] - base[BY, i, j] - dt * d_ez_dx
      solution[BX, i, j] = solution[BX, i, j] - omega * residual_bx / denom_bx
      solution[BY, i, j] = solution[BY, i, j] - omega * residual_by / denom_by


@njit(cache=True, parallel=True)
def implicit_diffusion_error(base, solution, ez, dx, dy, dt, ng, viscosity, resistivity):
   nx = solution.shape[1] - 2 * ng
   ny = solution.shape[2] - 2 * ng
   vx = viscosity * dt / (dx * dx)
   vy = viscosity * dt / (dy * dy)
   bx = resistivity * dt / (dx * dx)
   by = resistivity * dt / (dy * dy)
   diffuse_momentum = viscosity > 0.0
   diffuse_magnetic = solution.shape[0] > NVAR_HYDRO and resistivity > 0.0
   if diffuse_magnetic:
      fill_resistive_ez_corners(solution, ez, dx, dy, ng, resistivity)
   numerator = 0.0
   denominator = 0.0
   inv_2dx = 0.5 / dx
   inv_2dy = 0.5 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      if diffuse_momentum:
         for q in (MOMX, MOMY, MOMZ):
            residual = (
               (1.0 + 2.0 * (vx + vy)) * solution[q, i, j]
               - vx * (solution[q, i - 1, j] + solution[q, i + 1, j])
               - vy * (solution[q, i, j - 1] + solution[q, i, j + 1])
               - base[q, i, j]
            )
            numerator += abs(residual)
            denominator += abs(solution[q, i, j])
      if diffuse_magnetic:
         d_ez_dy = (ez[ii, jj + 1] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii + 1, jj]) * inv_2dy
         d_ez_dx = (ez[ii + 1, jj] - ez[ii, jj] + ez[ii + 1, jj + 1] - ez[ii, jj + 1]) * inv_2dx
         residual_bx = solution[BX, i, j] - base[BX, i, j] + dt * d_ez_dy
         residual_by = solution[BY, i, j] - base[BY, i, j] - dt * d_ez_dx
         residual_bz = (
            (1.0 + 2.0 * (bx + by)) * solution[BZ, i, j]
            - bx * (solution[BZ, i - 1, j] + solution[BZ, i + 1, j])
            - by * (solution[BZ, i, j - 1] + solution[BZ, i, j + 1])
            - base[BZ, i, j]
         )
         numerator += abs(residual_bx) + abs(residual_by) + abs(residual_bz)
         denominator += abs(solution[BX, i, j]) + abs(solution[BY, i, j]) + abs(solution[BZ, i, j])
   if denominator <= 0.0:
      denominator = 1.0
   return numerator / denominator


@njit(cache=True, parallel=True)
def red_black_sor_implicit_diffusion_color(base, solution, dx, dy, dt, ng, viscosity, resistivity, omega, color):
   nx = solution.shape[1] - 2 * ng
   ny = solution.shape[2] - 2 * ng
   vx = viscosity * dt / (dx * dx)
   vy = viscosity * dt / (dy * dy)
   vden = 1.0 + 2.0 * (vx + vy)
   bx = resistivity * dt / (dx * dx)
   by = resistivity * dt / (dy * dy)
   bden = 1.0 + 2.0 * (bx + by)
   diffuse_momentum = viscosity > 0.0
   has_mhd_state = solution.shape[0] > NVAR_HYDRO
   diffuse_magnetic = has_mhd_state and resistivity > 0.0

   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      if ((ii + jj) & 1) != color:
         continue
      i = ng + ii
      j = ng + jj
      solution[RHO, i, j] = base[RHO, i, j]
      solution[ENERGY, i, j] = base[ENERGY, i, j]

      if diffuse_momentum:
         for q in (MOMX, MOMY, MOMZ):
            relaxed = (
               base[q, i, j]
               + vx * (solution[q, i - 1, j] + solution[q, i + 1, j])
               + vy * (solution[q, i, j - 1] + solution[q, i, j + 1])
            ) / vden
            solution[q, i, j] = solution[q, i, j] + omega * (relaxed - solution[q, i, j])

      if diffuse_magnetic:
         relaxed = (
            base[BZ, i, j]
            + bx * (solution[BZ, i - 1, j] + solution[BZ, i + 1, j])
            + by * (solution[BZ, i, j - 1] + solution[BZ, i, j + 1])
         ) / bden
         solution[BZ, i, j] = solution[BZ, i, j] + omega * (relaxed - solution[BZ, i, j])


@njit(cache=True, parallel=True)
def explicit_diffusion_sweep(base, next_state, dx, dy, dt, ng, viscosity, resistivity):
   nx = next_state.shape[1] - 2 * ng
   ny = next_state.shape[2] - 2 * ng
   vx = viscosity * dt / (dx * dx)
   vy = viscosity * dt / (dy * dy)
   bx = resistivity * dt / (dx * dx)
   by = resistivity * dt / (dy * dy)
   has_mhd_state = next_state.shape[0] > NVAR_HYDRO
   diffuse_magnetic = has_mhd_state and resistivity > 0.0

   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      next_state[RHO, i, j] = base[RHO, i, j]
      next_state[ENERGY, i, j] = base[ENERGY, i, j]
      next_state[MOMX, i, j] = (
         base[MOMX, i, j]
         + vx * (base[MOMX, i - 1, j] - 2.0 * base[MOMX, i, j] + base[MOMX, i + 1, j])
         + vy * (base[MOMX, i, j - 1] - 2.0 * base[MOMX, i, j] + base[MOMX, i, j + 1])
      )
      next_state[MOMY, i, j] = (
         base[MOMY, i, j]
         + vx * (base[MOMY, i - 1, j] - 2.0 * base[MOMY, i, j] + base[MOMY, i + 1, j])
         + vy * (base[MOMY, i, j - 1] - 2.0 * base[MOMY, i, j] + base[MOMY, i, j + 1])
      )
      next_state[MOMZ, i, j] = (
         base[MOMZ, i, j]
         + vx * (base[MOMZ, i - 1, j] - 2.0 * base[MOMZ, i, j] + base[MOMZ, i + 1, j])
         + vy * (base[MOMZ, i, j - 1] - 2.0 * base[MOMZ, i, j] + base[MOMZ, i, j + 1])
      )

      if diffuse_magnetic:
         next_state[BX, i, j] = base[BX, i, j]
         next_state[BY, i, j] = base[BY, i, j]
         next_state[BZ, i, j] = (
            base[BZ, i, j]
            + bx * (base[BZ, i - 1, j] - 2.0 * base[BZ, i, j] + base[BZ, i + 1, j])
            + by * (base[BZ, i, j - 1] - 2.0 * base[BZ, i, j] + base[BZ, i, j + 1])
         )
      elif has_mhd_state:
         next_state[BX, i, j] = base[BX, i, j]
         next_state[BY, i, j] = base[BY, i, j]
         next_state[BZ, i, j] = base[BZ, i, j]


@njit(cache=True)
def explicit_diffusion_update(
   Uvec, next_state, scratch, dx, dy, dt, ng,
   bc_left, bc_right, bc_bottom, bc_top,
   viscosity, resistivity
):
   if viscosity <= 0.0 and resistivity <= 0.0:
      return
   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   explicit_diffusion_sweep(Uvec, next_state, dx, dy, dt, ng, viscosity, resistivity)
   if Uvec.shape[0] > NVAR_HYDRO and resistivity > 0.0:
      fill_resistive_ez_corners(Uvec, scratch[RHO], dx, dy, ng, resistivity)
      apply_resistive_emf_curl_to_state(Uvec, next_state, scratch[RHO], dx, dy, dt, ng)
   apply_boundaries(next_state, ng, bc_left, bc_right, bc_bottom, bc_top)
   copy_interior_local(next_state, Uvec, ng)
   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)


@njit(cache=True)
def implicit_diffusion_update(
   Uvec, guess, scratch, dx, dy, dt, ng,
   bc_left, bc_right, bc_bottom, bc_top,
   viscosity, resistivity, tolerance, max_steps, sor_omega
):
   if viscosity <= 0.0 and resistivity <= 0.0:
      return 0.0, 0
   explicit_diffusion_sweep(Uvec, guess, dx, dy, dt, ng, viscosity, resistivity)
   if Uvec.shape[0] > NVAR_HYDRO and resistivity > 0.0:
      fill_resistive_ez_corners(Uvec, scratch[RHO], dx, dy, ng, resistivity)
      apply_resistive_emf_curl_to_state(Uvec, guess, scratch[RHO], dx, dy, dt, ng)
   apply_boundaries(guess, ng, bc_left, bc_right, bc_bottom, bc_top)

   error = 1.0e300
   step = 0
   while error > tolerance and step < max_steps:
      red_black_sor_implicit_diffusion_color(Uvec, guess, dx, dy, dt, ng, viscosity, resistivity, sor_omega, 0)
      apply_boundaries(guess, ng, bc_left, bc_right, bc_bottom, bc_top)
      red_black_sor_implicit_diffusion_color(Uvec, guess, dx, dy, dt, ng, viscosity, resistivity, sor_omega, 1)
      if Uvec.shape[0] > NVAR_HYDRO and resistivity > 0.0:
         fill_resistive_ez_corners(guess, scratch[RHO], dx, dy, ng, resistivity)
         relax_resistive_transverse_magnetic(Uvec, guess, scratch[RHO], dx, dy, dt, ng, resistivity, sor_omega)
      apply_boundaries(guess, ng, bc_left, bc_right, bc_bottom, bc_top)
      error = implicit_diffusion_error(Uvec, guess, scratch[RHO], dx, dy, dt, ng, viscosity, resistivity)
      step += 1
   copy_interior_local(guess, Uvec, ng)
   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   return error, step
