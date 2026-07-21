from numba import literally, njit, prange

from pyfluid_numba.boundaries import apply_boundaries, apply_periodic_boundaries
from pyfluid_numba.diffusion import implicit_diffusion_update
from pyfluid_numba.options import *
from pyfluid_numba.physics import max_signal_frequency_array
from pyfluid_numba.rhs import fill_fluxes, fill_rhs

@njit(cache=True, parallel=True)
def advance_from_rhs(base, out, rhs, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      out[q, i, j] = base[q, i, j] + dt * rhs[q, i, j]


@njit(cache=True, parallel=True)
def advance_from_fluxes(base, out, Fx, Fy, dx, dy, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      for q in range(nvar):
         out[q, i, j] = base[q, i, j] - dt * (
            (Fx[q, ii + 1, jj] - Fx[q, ii, jj]) * inv_dx
            + (Fy[q, ii, jj + 1] - Fy[q, ii, jj]) * inv_dy
         )


@njit(cache=True, parallel=True)
def ssprk3_stage2_from_fluxes(base, stage1, stage2, Fx, Fy, dx, dy, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      for q in range(nvar):
         div = (Fx[q, ii + 1, jj] - Fx[q, ii, jj]) * inv_dx + (Fy[q, ii, jj + 1] - Fy[q, ii, jj]) * inv_dy
         stage2[q, i, j] = 0.75 * base[q, i, j] + 0.25 * (stage1[q, i, j] - dt * div)


@njit(cache=True, parallel=True)
def corrector_final_from_fluxes(base, predictor, out, Fx, Fy, dx, dy, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      for q in range(nvar):
         div = (Fx[q, ii + 1, jj] - Fx[q, ii, jj]) * inv_dx + (Fy[q, ii, jj + 1] - Fy[q, ii, jj]) * inv_dy
         out[q, i, j] = 0.5 * (base[q, i, j] + predictor[q, i, j] - dt * div)


@njit(cache=True, parallel=True)
def ssprk3_stage3_from_fluxes(base, stage2, out, Fx, Fy, dx, dy, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   inv_dx = 1.0 / dx
   inv_dy = 1.0 / dy
   for idx in prange(nx * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng + ii
      j = ng + jj
      for q in range(nvar):
         div = (Fx[q, ii + 1, jj] - Fx[q, ii, jj]) * inv_dx + (Fy[q, ii, jj + 1] - Fy[q, ii, jj]) * inv_dy
         out[q, i, j] = (base[q, i, j] + 2.0 * (stage2[q, i, j] - dt * div)) / 3.0


@njit(cache=True)
def apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top):
   bc_left = literally(bc_left)
   bc_right = literally(bc_right)
   bc_bottom = literally(bc_bottom)
   bc_top = literally(bc_top)
   if bc_left == BC_PERIODIC and bc_right == BC_PERIODIC and bc_bottom == BC_PERIODIC and bc_top == BC_PERIODIC:
      apply_periodic_boundaries(Uvec, ng)
   else:
      apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)


@njit(cache=True, parallel=True)
def corrector_final_state(base, predictor, out, rhs, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      out[q, i, j] = 0.5 * (base[q, i, j] + predictor[q, i, j] + dt * rhs[q, i, j])


@njit(cache=True, parallel=True)
def add_scaled_rhs(accum, rhs, scale, ng):
   nvar = accum.shape[0]
   nx = accum.shape[1] - 2 * ng
   ny = accum.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      accum[q, i, j] += scale * rhs[q, i, j]


@njit(cache=True, parallel=True)
def set_scaled_rhs(accum, rhs, scale, ng):
   nvar = accum.shape[0]
   nx = accum.shape[1] - 2 * ng
   ny = accum.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      accum[q, i, j] = scale * rhs[q, i, j]


@njit(cache=True, parallel=True)
def rk4_final_state(base, out, accum, rhs, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      out[q, i, j] = base[q, i, j] + (dt / 6.0) * (accum[q, i, j] + rhs[q, i, j])


@njit(cache=True, parallel=True)
def ssprk3_stage2(base, stage1, stage2, rhs, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      stage2[q, i, j] = 0.75 * base[q, i, j] + 0.25 * (stage1[q, i, j] + dt * rhs[q, i, j])


@njit(cache=True, parallel=True)
def ssprk3_stage3(base, stage2, out, rhs, dt, ng):
   nvar = base.shape[0]
   nx = base.shape[1] - 2 * ng
   ny = base.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      i = ng + ii
      j = ng + jj
      out[q, i, j] = (base[q, i, j] + 2.0 * (stage2[q, i, j] + dt * rhs[q, i, j])) / 3.0


@njit(cache=True, parallel=True)
def copy_interior(src, dst, ng):
   nvar = src.shape[0]
   nx = src.shape[1] - 2 * ng
   ny = src.shape[2] - 2 * ng
   for idx in prange(nvar * nx * ny):
      q = idx // (nx * ny)
      rem = idx - q * nx * ny
      ii = rem // ny
      jj = rem - ii * ny
      dst[q, ng + ii, ng + jj] = src[q, ng + ii, ng + jj]


@njit(cache=True)
def euler_step(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
   ct_method, diffusion, viscosity, resistivity
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion = literally(diffusion)
   apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   if not diffusion:
      fill_fluxes(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(Uvec, Unew, Fx, Fy, dx, dy, dt, ng)
      return
   fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   advance_from_rhs(Uvec, Unew, rhs, dt, ng)


@njit(cache=True)
def corrector_step(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
   ct_method, diffusion, viscosity, resistivity
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion = literally(diffusion)
   if not diffusion:
      apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(Uvec, Unew, Fx, Fy, dx, dy, dt, ng)

      apply_step_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      corrector_final_from_fluxes(Uvec, Unew, Utmp, Fx, Fy, dx, dy, dt, ng)
      copy_interior(Utmp, Unew, ng)
      return

   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   advance_from_rhs(Uvec, Unew, rhs, dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   corrector_final_state(Uvec, Unew, Utmp, rhs, dt, ng)
   copy_interior(Utmp, Unew, ng)


@njit(cache=True)
def midpoint_step(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
   ct_method, diffusion, viscosity, resistivity
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion = literally(diffusion)
   if not diffusion:
      apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(Uvec, Unew, Fx, Fy, dx, dy, 0.5 * dt, ng)

      apply_step_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(Uvec, Unew, Fx, Fy, dx, dy, dt, ng)
      return

   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   advance_from_rhs(Uvec, Unew, rhs, 0.5 * dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   advance_from_rhs(Uvec, Unew, rhs, dt, ng)


@njit(cache=True)
def rk4_step(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
   ct_method, diffusion, viscosity, resistivity
):
   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   set_scaled_rhs(Utmp, rhs, 1.0, ng)
   advance_from_rhs(Uvec, Unew, rhs, 0.5 * dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   add_scaled_rhs(Utmp, rhs, 2.0, ng)
   advance_from_rhs(Uvec, Unew, rhs, 0.5 * dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   add_scaled_rhs(Utmp, rhs, 2.0, ng)
   advance_from_rhs(Uvec, Unew, rhs, dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   rk4_final_state(Uvec, Unew, Utmp, rhs, dt, ng)


@njit(cache=True)
def ssprk3_step(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
   ct_method, diffusion, viscosity, resistivity
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion = literally(diffusion)
   if not diffusion:
      apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(Uvec, Unew, Fx, Fy, dx, dy, dt, ng)

      apply_step_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      ssprk3_stage2_from_fluxes(Uvec, Unew, Utmp, Fx, Fy, dx, dy, dt, ng)

      apply_step_boundaries(Utmp, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      ssprk3_stage3_from_fluxes(Uvec, Utmp, Unew, Fx, Fy, dx, dy, dt, ng)
      return

   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Uvec, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   advance_from_rhs(Uvec, Unew, rhs, dt, ng)

   apply_boundaries(Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   ssprk3_stage2(Uvec, Unew, Utmp, rhs, dt, ng)

   apply_boundaries(Utmp, ng, bc_left, bc_right, bc_bottom, bc_top)
   fill_rhs(Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, gamma, ng, recon, recon_vars, limiter, riemann, ct_method, diffusion, viscosity, resistivity)
   ssprk3_stage3(Uvec, Utmp, Unew, rhs, dt, ng)


@njit(cache=True)
def finish_step(Uvec, Unew, ng, bc_left, bc_right, bc_bottom, bc_top):
   copy_interior(Unew, Uvec, ng)
   apply_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)


def select_time_stepper(timestep):
   if timestep == TIME_EULER:
      return euler_step
   if timestep == TIME_CORRECTOR:
      return corrector_step
   if timestep == TIME_MIDPOINT:
      return midpoint_step
   if timestep == TIME_RK4:
      return rk4_step
   if timestep == TIME_SSPRK3:
      return ssprk3_step
   raise ValueError(f"unsupported Numba timestepper: {timestep}")


@njit(cache=True)
def adaptive_hydro_dt(Uvec, nx, ny, ng, gamma, cfl, dx, dy, max_dt, diffusion_method, viscosity, resistivity, diffusion_cfl):
   frequency = max_signal_frequency_array(Uvec, nx, ny, ng, gamma, dx, dy)
   dt = cfl / frequency
   if diffusion_method == DIFFUSION_EXPLICIT:
      coefficient = viscosity
      if resistivity > coefficient:
         coefficient = resistivity
      if coefficient > 0.0:
         diffusion_dt = diffusion_cfl / (2.0 * coefficient * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
         if diffusion_dt < dt:
            dt = diffusion_dt
   if max_dt > 0.0 and dt > max_dt:
      dt = max_dt
   return dt


@njit
def _evolve_steps(
   time_stepper,
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, steps,
   ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl, diffusion_tolerance, diffusion_max_steps,
   diffusion_sor_omega
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion_method = literally(diffusion_method)
   nx = Uvec.shape[1] - 2 * ng
   ny = Uvec.shape[2] - 2 * ng
   time_value = 0.0
   dt_min = 1.0e300
   dt_max = 0.0
   diffusion_error = 0.0
   diffusion_steps_max = 0
   diffusion_steps_total = 0
   for _ in range(steps):
      dt = adaptive_hydro_dt(Uvec, nx, ny, ng, gamma, cfl, dx, dy, -1.0, diffusion_method, viscosity, resistivity, diffusion_cfl)
      if diffusion_method == DIFFUSION_EXPLICIT:
         time_stepper(
            Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
            bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
            ct_method, True, viscosity, resistivity
         )
      else:
         time_stepper(
            Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
            bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
            ct_method, False, viscosity, resistivity
         )
      if diffusion_method == DIFFUSION_IMPLICIT:
         step_error, diffusion_steps = implicit_diffusion_update(
            Unew, Utmp, rhs, dx, dy, dt, ng,
            bc_left, bc_right, bc_bottom, bc_top,
            viscosity, resistivity, diffusion_tolerance, diffusion_max_steps, diffusion_sor_omega
         )
         diffusion_error = step_error
         diffusion_steps_total += diffusion_steps
         if diffusion_steps > diffusion_steps_max:
            diffusion_steps_max = diffusion_steps
      finish_step(Uvec, Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
      time_value += dt
      if dt < dt_min:
         dt_min = dt
      if dt > dt_max:
         dt_max = dt
   if steps == 0:
      dt_min = 0.0
   return time_value, dt_min, dt_max, diffusion_error, diffusion_steps_max, diffusion_steps_total


@njit(cache=True)
def _evolve_steps_euler(
   Uvec, Unew, scratch, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, ct_method, steps
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   nx = Uvec.shape[1] - 2 * ng
   ny = Uvec.shape[2] - 2 * ng
   time_value = 0.0
   dt_min = 1.0e300
   dt_max = 0.0
   current = Uvec
   next_state = Unew
   current_is_uvec = True
   for _ in range(steps):
      dt = adaptive_hydro_dt(current, nx, ny, ng, gamma, cfl, dx, dy, -1.0, DIFFUSION_NONE, 0.0, 0.0, 0.0)
      apply_step_boundaries(current, ng, bc_left, bc_right, bc_bottom, bc_top)
      fill_fluxes(current, scratch, Fx, Fy, Ez, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter, riemann, ct_method)
      advance_from_fluxes(current, next_state, Fx, Fy, dx, dy, dt, ng)
      current, next_state = next_state, current
      current_is_uvec = not current_is_uvec
      time_value += dt
      if dt < dt_min:
         dt_min = dt
      if dt > dt_max:
         dt_max = dt
   if steps == 0:
      dt_min = 0.0
   if not current_is_uvec:
      copy_interior(current, Uvec, ng)
   apply_step_boundaries(Uvec, ng, bc_left, bc_right, bc_bottom, bc_top)
   return time_value, dt_min, dt_max


def evolve_steps(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, timestep, steps,
   ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl,
   diffusion_tolerance=DEFAULT_DIFFUSION_TOLERANCE,
   diffusion_max_steps=DEFAULT_DIFFUSION_MAX_STEPS,
   diffusion_sor_omega=DEFAULT_DIFFUSION_SOR_OMEGA,
):
   if (
      timestep == TIME_EULER
      and diffusion_method == DIFFUSION_NONE
   ):
      time_value, dt_min, dt_max = _evolve_steps_euler(
         Uvec, Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
         bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, ct_method, steps
      )
      return time_value, dt_min, dt_max
   time_stepper = select_time_stepper(timestep)
   time_value, dt_min, dt_max, _, _, _ = _evolve_steps(
      time_stepper,
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, steps,
      ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl, diffusion_tolerance, diffusion_max_steps,
      diffusion_sor_omega
   )
   return time_value, dt_min, dt_max


def evolve_steps_with_diagnostics(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, timestep, steps,
   ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl,
   diffusion_tolerance=DEFAULT_DIFFUSION_TOLERANCE,
   diffusion_max_steps=DEFAULT_DIFFUSION_MAX_STEPS,
   diffusion_sor_omega=DEFAULT_DIFFUSION_SOR_OMEGA,
):
   if (
      timestep == TIME_EULER
      and diffusion_method == DIFFUSION_NONE
   ):
      time_value, dt_min, dt_max = _evolve_steps_euler(
         Uvec, Unew, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
         bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, ct_method, steps
      )
      return time_value, dt_min, dt_max, 0.0, 0, 0
   time_stepper = select_time_stepper(timestep)
   return _evolve_steps(
      time_stepper,
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, steps,
      ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl, diffusion_tolerance, diffusion_max_steps,
      diffusion_sor_omega
   )


@njit
def _evolve_to_time(
   time_stepper,
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, final_time, max_steps,
   ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl, diffusion_tolerance, diffusion_max_steps,
   diffusion_sor_omega
):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   riemann = literally(riemann)
   ct_method = literally(ct_method)
   diffusion_method = literally(diffusion_method)
   nx = Uvec.shape[1] - 2 * ng
   ny = Uvec.shape[2] - 2 * ng
   time_value = 0.0
   steps = 0
   dt_min = 1.0e300
   dt_max = 0.0
   while time_value < final_time and steps < max_steps:
      dt = adaptive_hydro_dt(Uvec, nx, ny, ng, gamma, cfl, dx, dy, final_time - time_value, diffusion_method, viscosity, resistivity, diffusion_cfl)
      if diffusion_method == DIFFUSION_EXPLICIT:
         time_stepper(
            Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
            bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
            ct_method, True, viscosity, resistivity
         )
      else:
         time_stepper(
            Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, dt, gamma, ng,
            bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann,
            ct_method, False, viscosity, resistivity
         )
      if diffusion_method == DIFFUSION_IMPLICIT:
         implicit_diffusion_update(
            Unew, Utmp, rhs, dx, dy, dt, ng,
            bc_left, bc_right, bc_bottom, bc_top,
            viscosity, resistivity, diffusion_tolerance, diffusion_max_steps, diffusion_sor_omega
         )
      finish_step(Uvec, Unew, ng, bc_left, bc_right, bc_bottom, bc_top)
      time_value += dt
      steps += 1
      if dt < dt_min:
         dt_min = dt
      if dt > dt_max:
         dt_max = dt
   if steps == 0:
      dt_min = 0.0
   return time_value, steps, dt_min, dt_max


def evolve_to_time(
   Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
   bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, timestep, final_time, max_steps,
   ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl,
   diffusion_tolerance=DEFAULT_DIFFUSION_TOLERANCE,
   diffusion_max_steps=DEFAULT_DIFFUSION_MAX_STEPS,
   diffusion_sor_omega=DEFAULT_DIFFUSION_SOR_OMEGA,
):
   time_stepper = select_time_stepper(timestep)
   return _evolve_to_time(
      time_stepper,
      Uvec, Unew, Utmp, rhs, Fx, Fy, Ez, UxL, UxR, UyB, UyT, dx, dy, cfl, gamma, ng,
      bc_left, bc_right, bc_bottom, bc_top, recon, recon_vars, limiter, riemann, final_time, max_steps,
      ct_method, diffusion_method, viscosity, resistivity, diffusion_cfl, diffusion_tolerance, diffusion_max_steps,
      diffusion_sor_omega
   )
