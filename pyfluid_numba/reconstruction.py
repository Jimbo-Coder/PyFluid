from numba import literally, njit, prange
import numpy as np

from pyfluid_numba.options import *
from pyfluid_numba.physics import (
   conserved_from_primitive,
   conserved_from_primitive_mhd,
   primitive_pressure,
)

@njit(cache=True, inline="always")
def minmod_scalar(a, b):
   if a * b <= 0.0:
      return 0.0
   if abs(a) < abs(b):
      return a
   return b


@njit(cache=True, inline="always")
def minmod3_scalar(a, b, c):
   if a * b <= 0.0 or a * c <= 0.0:
      return 0.0
   mag = abs(a)
   if abs(b) < mag:
      mag = abs(b)
   if abs(c) < mag:
      mag = abs(c)
   return np.sign(a) * mag


@njit(cache=True)
def limited_slope_pair(left_diff, right_diff, limiter):
   limiter = literally(limiter)
   if limiter == LIMITER_MC:
      return minmod3_scalar(0.5 * (left_diff + right_diff), 2.0 * left_diff, 2.0 * right_diff)
   return minmod_scalar(left_diff, right_diff)


@njit(cache=True)
def component_value(Uvec, q, i, j, axis, offset):
   axis = literally(axis)
   if axis == AXIS_X:
      return Uvec[q, i + offset, j]
   return Uvec[q, i, j + offset]


@njit(cache=True, inline="always")
def limited_slope(Uvec, q, i, j, limiter, axis):
   center = component_value(Uvec, q, i, j, axis, 0)
   return limited_slope_pair(center - component_value(Uvec, q, i, j, axis, -1), component_value(Uvec, q, i, j, axis, 1) - center, limiter)


@njit(cache=True, inline="always")
def primitive_value(Uvec, primitive_index, i, j, gamma):
   rho = Uvec[RHO, i, j]
   if primitive_index == 0:
      return rho
   if primitive_index == 1:
      return Uvec[MOMX, i, j] / rho
   if primitive_index == 2:
      return Uvec[MOMY, i, j] / rho
   if primitive_index == 3:
      return Uvec[MOMZ, i, j] / rho
   if primitive_index == 4:
      return primitive_pressure(Uvec, i, j, gamma)
   if primitive_index == 5:
      return Uvec[BX, i, j]
   if primitive_index == 6:
      return Uvec[BY, i, j]
   return Uvec[BZ, i, j]


@njit(cache=True)
def primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, offset):
   axis = literally(axis)
   if axis == AXIS_X:
      return primitive_value(Uvec, primitive_index, i + offset, j, gamma)
   return primitive_value(Uvec, primitive_index, i, j + offset, gamma)


@njit(cache=True)
def primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, offset):
   axis = literally(axis)
   if axis == AXIS_X:
      return Vcache[primitive_index, i + offset, j]
   return Vcache[primitive_index, i, j + offset]


@njit(cache=True)
def limited_primitive_slope(Uvec, primitive_index, i, j, gamma, limiter, axis):
   center = primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 0)
   left = primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, -1)
   right = primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 1)
   return limited_slope_pair(center - left, right - center, limiter)


@njit(cache=True, inline="always")
def limited_primitive_cache_slope(Vcache, primitive_index, i, j, limiter, axis):
   center = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0)
   left = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -1)
   right = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1)
   return limited_slope_pair(center - left, right - center, limiter)


@njit(cache=True, inline="always")
def weno_z_left(vmm, vm, v0, vp, vpp):
   d0 = vmm - 2.0 * vm + v0
   e0 = vmm - 4.0 * vm + 3.0 * v0
   d1 = vm - 2.0 * v0 + vp
   e1 = vm - vp
   d2 = v0 - 2.0 * vp + vpp
   e2 = 3.0 * v0 - 4.0 * vp + vpp
   beta0 = (13.0 / 12.0) * d0 * d0 + 0.25 * e0 * e0
   beta1 = (13.0 / 12.0) * d1 * d1 + 0.25 * e1 * e1
   beta2 = (13.0 / 12.0) * d2 * d2 + 0.25 * e2 * e2
   tau5 = abs(beta0 - beta2)
   r0 = tau5 / (WENO_EPS + beta0)
   r1 = tau5 / (WENO_EPS + beta1)
   r2 = tau5 / (WENO_EPS + beta2)
   a0 = 0.1 * (1.0 + r0 * r0)
   a1 = 0.6 * (1.0 + r1 * r1)
   a2 = 0.3 * (1.0 + r2 * r2)
   asum = a0 + a1 + a2
   w0 = a0 / asum
   w1 = a1 / asum
   w2 = a2 / asum
   p0 = (2.0 * vmm - 7.0 * vm + 11.0 * v0) / 6.0
   p1 = (-vm + 5.0 * v0 + 2.0 * vp) / 6.0
   p2 = (2.0 * v0 + 5.0 * vp - vpp) / 6.0
   return w0 * p0 + w1 * p1 + w2 * p2


@njit(cache=True, inline="always")
def weno_z_right(vm, v0, vp, vpp, vppp):
   return weno_z_left(vppp, vpp, vp, v0, vm)


@njit(cache=True, inline="always")
def hydro_characteristic_value(Vcache, char_index, i, j, axis, offset, rho_ref, sound_speed):
   if axis == AXIS_X:
      normal_x = 1.0
      normal_y = 0.0
      tangent_x = 0.0
      tangent_y = 1.0
   else:
      normal_x = 0.0
      normal_y = 1.0
      tangent_x = -1.0
      tangent_y = 0.0
   rho = primitive_cache_value_offset(Vcache, 0, i, j, axis, offset)
   vx = primitive_cache_value_offset(Vcache, 1, i, j, axis, offset)
   vy = primitive_cache_value_offset(Vcache, 2, i, j, axis, offset)
   vz = primitive_cache_value_offset(Vcache, 3, i, j, axis, offset)
   pressure = primitive_cache_value_offset(Vcache, 4, i, j, axis, offset)
   normal_velocity = normal_x * vx + normal_y * vy
   tangent_velocity = tangent_x * vx + tangent_y * vy
   inv_c2 = 1.0 / (sound_speed * sound_speed)
   if char_index == 0:
      return rho - pressure * inv_c2
   if char_index == 1:
      return tangent_velocity
   if char_index == 2:
      return vz
   if char_index == 3:
      return 0.5 * (pressure * inv_c2 - rho_ref * normal_velocity / sound_speed)
   return 0.5 * (pressure * inv_c2 + rho_ref * normal_velocity / sound_speed)


@njit(cache=True, inline="always")
def weno_z_characteristic_cached(Vcache, char_index, i, j, axis, side, rho_ref, sound_speed):
   if side == SIDE_LOWER:
      return weno_z_left(
         hydro_characteristic_value(Vcache, char_index, i, j, axis, -2, rho_ref, sound_speed),
         hydro_characteristic_value(Vcache, char_index, i, j, axis, -1, rho_ref, sound_speed),
         hydro_characteristic_value(Vcache, char_index, i, j, axis, 0, rho_ref, sound_speed),
         hydro_characteristic_value(Vcache, char_index, i, j, axis, 1, rho_ref, sound_speed),
         hydro_characteristic_value(Vcache, char_index, i, j, axis, 2, rho_ref, sound_speed),
      )
   return weno_z_right(
      hydro_characteristic_value(Vcache, char_index, i, j, axis, -1, rho_ref, sound_speed),
      hydro_characteristic_value(Vcache, char_index, i, j, axis, 0, rho_ref, sound_speed),
      hydro_characteristic_value(Vcache, char_index, i, j, axis, 1, rho_ref, sound_speed),
      hydro_characteristic_value(Vcache, char_index, i, j, axis, 2, rho_ref, sound_speed),
      hydro_characteristic_value(Vcache, char_index, i, j, axis, 3, rho_ref, sound_speed),
   )


@njit(cache=True, inline="always")
def hydro_characteristic_to_primitive(w0, w1, w2, w3, w4, rho_ref, sound_speed, axis):
   if axis == AXIS_X:
      normal_x = 1.0
      normal_y = 0.0
      tangent_x = 0.0
      tangent_y = 1.0
   else:
      normal_x = 0.0
      normal_y = 1.0
      tangent_x = -1.0
      tangent_y = 0.0
   rho = w0 + w3 + w4
   normal_velocity = sound_speed * (w4 - w3) / rho_ref
   tangent_velocity = w1
   vx = normal_x * normal_velocity + tangent_x * tangent_velocity
   vy = normal_y * normal_velocity + tangent_y * tangent_velocity
   vz = w2
   pressure = sound_speed * sound_speed * (w3 + w4)
   return rho, vx, vy, vz, pressure


@njit(cache=True)
def reconstruct_hydro_characteristic_cached(Vcache, i, j, gamma, axis):
   rho_left = primitive_cache_value_offset(Vcache, 0, i, j, axis, 0)
   rho_right = primitive_cache_value_offset(Vcache, 0, i, j, axis, 1)
   pressure_left = primitive_cache_value_offset(Vcache, 4, i, j, axis, 0)
   pressure_right = primitive_cache_value_offset(Vcache, 4, i, j, axis, 1)
   rho_ref = 0.5 * (rho_left + rho_right)
   pressure_ref = 0.5 * (pressure_left + pressure_right)
   sound_speed = np.sqrt(gamma * pressure_ref / rho_ref)

   wl0 = weno_z_characteristic_cached(Vcache, 0, i, j, axis, SIDE_LOWER, rho_ref, sound_speed)
   wl1 = weno_z_characteristic_cached(Vcache, 1, i, j, axis, SIDE_LOWER, rho_ref, sound_speed)
   wl2 = weno_z_characteristic_cached(Vcache, 2, i, j, axis, SIDE_LOWER, rho_ref, sound_speed)
   wl3 = weno_z_characteristic_cached(Vcache, 3, i, j, axis, SIDE_LOWER, rho_ref, sound_speed)
   wl4 = weno_z_characteristic_cached(Vcache, 4, i, j, axis, SIDE_LOWER, rho_ref, sound_speed)
   wr0 = weno_z_characteristic_cached(Vcache, 0, i, j, axis, SIDE_UPPER, rho_ref, sound_speed)
   wr1 = weno_z_characteristic_cached(Vcache, 1, i, j, axis, SIDE_UPPER, rho_ref, sound_speed)
   wr2 = weno_z_characteristic_cached(Vcache, 2, i, j, axis, SIDE_UPPER, rho_ref, sound_speed)
   wr3 = weno_z_characteristic_cached(Vcache, 3, i, j, axis, SIDE_UPPER, rho_ref, sound_speed)
   wr4 = weno_z_characteristic_cached(Vcache, 4, i, j, axis, SIDE_UPPER, rho_ref, sound_speed)
   pl0, pl1, pl2, pl3, pl4 = hydro_characteristic_to_primitive(wl0, wl1, wl2, wl3, wl4, rho_ref, sound_speed, axis)
   pr0, pr1, pr2, pr3, pr4 = hydro_characteristic_to_primitive(wr0, wr1, wr2, wr3, wr4, rho_ref, sound_speed, axis)
   return pl0, pl1, pl2, pl3, pl4, pr0, pr1, pr2, pr3, pr4


@njit(cache=True, inline="always")
def mhd_reduced_primitive_value(Vcache, reduced_index, i, j, axis, offset):
   if axis == AXIS_X:
      if reduced_index == 0:
         return primitive_cache_value_offset(Vcache, 0, i, j, axis, offset)
      if reduced_index == 1:
         return primitive_cache_value_offset(Vcache, 1, i, j, axis, offset)
      if reduced_index == 2:
         return primitive_cache_value_offset(Vcache, 2, i, j, axis, offset)
      if reduced_index == 3:
         return primitive_cache_value_offset(Vcache, 3, i, j, axis, offset)
      if reduced_index == 4:
         return primitive_cache_value_offset(Vcache, 4, i, j, axis, offset)
      if reduced_index == 5:
         return primitive_cache_value_offset(Vcache, 6, i, j, axis, offset)
      return primitive_cache_value_offset(Vcache, 7, i, j, axis, offset)
   if reduced_index == 0:
      return primitive_cache_value_offset(Vcache, 0, i, j, axis, offset)
   if reduced_index == 1:
      return primitive_cache_value_offset(Vcache, 2, i, j, axis, offset)
   if reduced_index == 2:
      return primitive_cache_value_offset(Vcache, 1, i, j, axis, offset)
   if reduced_index == 3:
      return primitive_cache_value_offset(Vcache, 3, i, j, axis, offset)
   if reduced_index == 4:
      return primitive_cache_value_offset(Vcache, 4, i, j, axis, offset)
   if reduced_index == 5:
      return primitive_cache_value_offset(Vcache, 5, i, j, axis, offset)
   return primitive_cache_value_offset(Vcache, 7, i, j, axis, offset)

@njit(cache=True)
def mhd_characteristics_at_offset(
   Vcache,
   i,
   j,
   axis,
   offset,
   rho_ref,
   sqrt_rho,
   sound_speed,
   fast_speed,
   slow_speed,
   alpha_f,
   alpha_s,
   beta_t,
   beta_z,
   normal_sign,
):
   rho = mhd_reduced_primitive_value(Vcache, 0, i, j, axis, offset)
   v_normal = mhd_reduced_primitive_value(Vcache, 1, i, j, axis, offset)
   v_tangent = mhd_reduced_primitive_value(Vcache, 2, i, j, axis, offset)
   v_z = mhd_reduced_primitive_value(Vcache, 3, i, j, axis, offset)
   pressure = mhd_reduced_primitive_value(Vcache, 4, i, j, axis, offset)
   b_tangent = mhd_reduced_primitive_value(Vcache, 5, i, j, axis, offset)
   b_z = mhd_reduced_primitive_value(Vcache, 6, i, j, axis, offset)

   inv_sound2 = 1.0 / (sound_speed * sound_speed)
   compressive = pressure * inv_sound2 / rho_ref
   b_parallel = (beta_t * b_tangent + beta_z * b_z) / (sound_speed * sqrt_rho)
   fast_sum = alpha_f * compressive + alpha_s * b_parallel
   slow_sum = alpha_s * compressive - alpha_f * b_parallel

   denom = alpha_f * alpha_f * fast_speed * fast_speed + alpha_s * alpha_s * slow_speed * slow_speed
   if denom <= 1.0e-14:
      denom = sound_speed * sound_speed
   v_parallel = normal_sign * (beta_t * v_tangent + beta_z * v_z)
   fast_diff = (alpha_f * fast_speed * v_normal - alpha_s * slow_speed * v_parallel) / denom
   slow_diff = (alpha_s * slow_speed * v_normal + alpha_f * fast_speed * v_parallel) / denom

   alfven_sum = (beta_z * b_tangent - beta_t * b_z) / sqrt_rho
   alfven_diff = normal_sign * (beta_z * v_tangent - beta_t * v_z)

   fast_minus = 0.5 * (fast_sum - fast_diff)
   alfven_minus = 0.5 * (alfven_sum + alfven_diff)
   slow_minus = 0.5 * (slow_sum - slow_diff)
   entropy = rho - pressure * inv_sound2
   slow_plus = 0.5 * (slow_sum + slow_diff)
   alfven_plus = 0.5 * (alfven_sum - alfven_diff)
   fast_plus = 0.5 * (fast_sum + fast_diff)
   return fast_minus, alfven_minus, slow_minus, entropy, slow_plus, alfven_plus, fast_plus


@njit(cache=True)
def mhd_characteristic_to_primitive(
   fast_minus,
   alfven_minus,
   slow_minus,
   entropy,
   slow_plus,
   alfven_plus,
   fast_plus,
   rho_ref,
   sqrt_rho,
   sound_speed,
   fast_speed,
   slow_speed,
   alpha_f,
   alpha_s,
   beta_t,
   beta_z,
   normal_sign,
):
   fast_sum = fast_minus + fast_plus
   fast_diff = fast_plus - fast_minus
   slow_sum = slow_minus + slow_plus
   slow_diff = slow_plus - slow_minus
   alfven_sum = alfven_minus + alfven_plus
   alfven_diff = alfven_minus - alfven_plus
   sound2 = sound_speed * sound_speed

   rho = alpha_f * rho_ref * fast_sum + alpha_s * rho_ref * slow_sum + entropy
   v_normal = alpha_f * fast_speed * fast_diff + alpha_s * slow_speed * slow_diff
   v_tangent = normal_sign * (
      -alpha_s * slow_speed * beta_t * fast_diff
      + alpha_f * fast_speed * beta_t * slow_diff
      + beta_z * alfven_diff
   )
   v_z = normal_sign * (
      -alpha_s * slow_speed * beta_z * fast_diff
      + alpha_f * fast_speed * beta_z * slow_diff
      - beta_t * alfven_diff
   )
   pressure = alpha_f * rho_ref * sound2 * fast_sum + alpha_s * rho_ref * sound2 * slow_sum
   b_tangent = (
      alpha_s * sound_speed * sqrt_rho * beta_t * fast_sum
      - alpha_f * sound_speed * sqrt_rho * beta_t * slow_sum
      + beta_z * sqrt_rho * alfven_sum
   )
   b_z = (
      alpha_s * sound_speed * sqrt_rho * beta_z * fast_sum
      - alpha_f * sound_speed * sqrt_rho * beta_z * slow_sum
      - beta_t * sqrt_rho * alfven_sum
   )
   return rho, v_normal, v_tangent, v_z, pressure, b_tangent, b_z


@njit(cache=True)
def reconstruct_mhd_characteristic_cached(Vcache, i, j, gamma, axis):
   rho_ref = 0.5 * (
      primitive_cache_value_offset(Vcache, 0, i, j, axis, 0)
      + primitive_cache_value_offset(Vcache, 0, i, j, axis, 1)
   )
   pressure_ref = 0.5 * (
      primitive_cache_value_offset(Vcache, 4, i, j, axis, 0)
      + primitive_cache_value_offset(Vcache, 4, i, j, axis, 1)
   )
   if axis == AXIS_X:
      b_normal_left = weno_z_left(
         primitive_cache_value_offset(Vcache, 5, i, j, axis, -2),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, -1),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 0),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 1),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 2),
      )
      b_normal_right = weno_z_right(
         primitive_cache_value_offset(Vcache, 5, i, j, axis, -1),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 0),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 1),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 2),
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 3),
      )
      b_normal = 0.5 * (b_normal_left + b_normal_right)
      b_tangent = 0.5 * (
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 0)
         + primitive_cache_value_offset(Vcache, 6, i, j, axis, 1)
      )
   else:
      b_normal_left = weno_z_left(
         primitive_cache_value_offset(Vcache, 6, i, j, axis, -2),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, -1),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 0),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 1),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 2),
      )
      b_normal_right = weno_z_right(
         primitive_cache_value_offset(Vcache, 6, i, j, axis, -1),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 0),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 1),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 2),
         primitive_cache_value_offset(Vcache, 6, i, j, axis, 3),
      )
      b_normal = 0.5 * (b_normal_left + b_normal_right)
      b_tangent = 0.5 * (
         primitive_cache_value_offset(Vcache, 5, i, j, axis, 0)
         + primitive_cache_value_offset(Vcache, 5, i, j, axis, 1)
      )
   b_z = 0.5 * (
      primitive_cache_value_offset(Vcache, 7, i, j, axis, 0)
      + primitive_cache_value_offset(Vcache, 7, i, j, axis, 1)
   )
   sound2 = gamma * pressure_ref / rho_ref
   b2 = (b_normal * b_normal + b_tangent * b_tangent + b_z * b_z) / rho_ref
   ca2 = b_normal * b_normal / rho_ref
   discr = (sound2 + b2) * (sound2 + b2) - 4.0 * sound2 * ca2
   if discr < 0.0 and discr > -1.0e-14:
      discr = 0.0
   if discr < 0.0:
      discr = 0.0
   root = np.sqrt(discr)
   fast2 = 0.5 * (sound2 + b2 + root)
   slow2 = 0.5 * (sound2 + b2 - root)
   if fast2 < 0.0:
      fast2 = 0.0
   if slow2 < 0.0:
      slow2 = 0.0
   sound_speed = np.sqrt(sound2)
   fast_speed = np.sqrt(fast2)
   slow_speed = np.sqrt(slow2)
   sqrt_rho = np.sqrt(rho_ref)
   b_perp = np.sqrt(b_tangent * b_tangent + b_z * b_z)
   if b_perp > 1.0e-14:
      beta_t = b_tangent / b_perp
      beta_z = b_z / b_perp
   else:
      beta_t = 1.0 / np.sqrt(2.0)
      beta_z = 1.0 / np.sqrt(2.0)
   denom = fast2 - slow2
   if denom <= 1.0e-14:
      alpha_f = 1.0
      alpha_s = 0.0
   else:
      alpha_f = np.sqrt(max((sound2 - slow2) / denom, 0.0))
      alpha_s = np.sqrt(max((fast2 - sound2) / denom, 0.0))
   normal_sign = 1.0
   if b_normal < 0.0:
      normal_sign = -1.0

   cm2_0, cm2_1, cm2_2, cm2_3, cm2_4, cm2_5, cm2_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, -2, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   cm1_0, cm1_1, cm1_2, cm1_3, cm1_4, cm1_5, cm1_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, -1, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   c0_0, c0_1, c0_2, c0_3, c0_4, c0_5, c0_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, 0, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   cp1_0, cp1_1, cp1_2, cp1_3, cp1_4, cp1_5, cp1_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, 1, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   cp2_0, cp2_1, cp2_2, cp2_3, cp2_4, cp2_5, cp2_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, 2, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   cp3_0, cp3_1, cp3_2, cp3_3, cp3_4, cp3_5, cp3_6 = mhd_characteristics_at_offset(
      Vcache, i, j, axis, 3, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )

   wl0 = weno_z_left(cm2_0, cm1_0, c0_0, cp1_0, cp2_0)
   wl1 = weno_z_left(cm2_1, cm1_1, c0_1, cp1_1, cp2_1)
   wl2 = weno_z_left(cm2_2, cm1_2, c0_2, cp1_2, cp2_2)
   wl3 = weno_z_left(cm2_3, cm1_3, c0_3, cp1_3, cp2_3)
   wl4 = weno_z_left(cm2_4, cm1_4, c0_4, cp1_4, cp2_4)
   wl5 = weno_z_left(cm2_5, cm1_5, c0_5, cp1_5, cp2_5)
   wl6 = weno_z_left(cm2_6, cm1_6, c0_6, cp1_6, cp2_6)
   wr0 = weno_z_right(cm1_0, c0_0, cp1_0, cp2_0, cp3_0)
   wr1 = weno_z_right(cm1_1, c0_1, cp1_1, cp2_1, cp3_1)
   wr2 = weno_z_right(cm1_2, c0_2, cp1_2, cp2_2, cp3_2)
   wr3 = weno_z_right(cm1_3, c0_3, cp1_3, cp2_3, cp3_3)
   wr4 = weno_z_right(cm1_4, c0_4, cp1_4, cp2_4, cp3_4)
   wr5 = weno_z_right(cm1_5, c0_5, cp1_5, cp2_5, cp3_5)
   wr6 = weno_z_right(cm1_6, c0_6, cp1_6, cp2_6, cp3_6)

   pl0, pl1, pl2, pl3, pl4, pl5, pl6 = mhd_characteristic_to_primitive(
      wl0, wl1, wl2, wl3, wl4, wl5, wl6, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   pr0, pr1, pr2, pr3, pr4, pr5, pr6 = mhd_characteristic_to_primitive(
      wr0, wr1, wr2, wr3, wr4, wr5, wr6, rho_ref, sqrt_rho, sound_speed, fast_speed, slow_speed,
      alpha_f, alpha_s, beta_t, beta_z, normal_sign
   )
   if axis == AXIS_X:
      return (
         pl0, pl1, pl2, pl3, pl4, b_normal, pl5, pl6,
         pr0, pr1, pr2, pr3, pr4, b_normal, pr5, pr6,
      )
   return (
      pl0, pl2, pl1, pl3, pl4, pl5, b_normal, pl6,
      pr0, pr2, pr1, pr3, pr4, pr5, b_normal, pr6,
   )


@njit(cache=True)
def reconstruct_primitive(Uvec, primitive_index, i, j, gamma, recon, limiter, axis, side):
   recon = literally(recon)
   limiter = literally(limiter)
   axis = literally(axis)
   side = literally(side)
   if recon == RECON_WENOZ:
      if side == SIDE_LOWER:
         return weno_z_left(
            primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, -2),
            primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, -1),
            primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 0),
            primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 1),
            primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 2),
         )
      return weno_z_right(
         primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, -1),
         primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 0),
         primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 1),
         primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 2),
         primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 3),
      )
   if recon == RECON_PLM:
      if side == SIDE_LOWER:
         return primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 0) + 0.5 * limited_primitive_slope(
            Uvec, primitive_index, i, j, gamma, limiter, axis
         )
      next_i = i + 1 if axis == AXIS_X else i
      next_j = j if axis == AXIS_X else j + 1
      return primitive_value(Uvec, primitive_index, next_i, next_j, gamma) - 0.5 * limited_primitive_slope(
         Uvec, primitive_index, next_i, next_j, gamma, limiter, axis
      )
   if side == SIDE_LOWER:
      return primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 0)
   return primitive_value_offset(Uvec, primitive_index, i, j, gamma, axis, 1)


@njit(cache=True)
def reconstruct_primitive_cached(Vcache, primitive_index, i, j, recon, limiter, axis, side):
   recon = literally(recon)
   limiter = literally(limiter)
   axis = literally(axis)
   side = literally(side)
   if recon == RECON_WENOZ:
      if side == SIDE_LOWER:
         return weno_z_left(
            primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -2),
            primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -1),
            primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0),
            primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1),
            primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 2),
         )
      return weno_z_right(
         primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -1),
         primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0),
         primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1),
         primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 2),
         primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 3),
      )
   if recon == RECON_PLM:
      if side == SIDE_LOWER:
         return primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0) + 0.5 * limited_primitive_cache_slope(
            Vcache, primitive_index, i, j, limiter, axis
         )
      next_i = i + 1 if axis == AXIS_X else i
      next_j = j if axis == AXIS_X else j + 1
      return Vcache[primitive_index, next_i, next_j] - 0.5 * limited_primitive_cache_slope(
         Vcache, primitive_index, next_i, next_j, limiter, axis
      )
   if side == SIDE_LOWER:
      return primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0)
   return primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1)


@njit(cache=True, inline="always")
def reconstruct_primitive_cached_pair(Vcache, primitive_index, i, j, recon, limiter, axis):
   recon = literally(recon)
   limiter = literally(limiter)
   axis = literally(axis)
   if recon == RECON_WENOZ:
      vm2 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -2)
      vm1 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, -1)
      v0 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0)
      vp1 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1)
      vp2 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 2)
      vp3 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 3)
      return weno_z_left(vm2, vm1, v0, vp1, vp2), weno_z_right(vm1, v0, vp1, vp2, vp3)
   if recon == RECON_PLM:
      v0 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0)
      vp1 = primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1)
      next_i = i + 1 if axis == AXIS_X else i
      next_j = j if axis == AXIS_X else j + 1
      return (
         v0 + 0.5 * limited_primitive_cache_slope(Vcache, primitive_index, i, j, limiter, axis),
         vp1 - 0.5 * limited_primitive_cache_slope(Vcache, primitive_index, next_i, next_j, limiter, axis),
      )
   return (
      primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 0),
      primitive_cache_value_offset(Vcache, primitive_index, i, j, axis, 1),
   )


@njit(cache=True)
def reconstruct_component(Uvec, q, i, j, recon, limiter, axis, side):
   recon = literally(recon)
   limiter = literally(limiter)
   axis = literally(axis)
   side = literally(side)
   if recon == RECON_PLM:
      if side == SIDE_LOWER:
         return component_value(Uvec, q, i, j, axis, 0) + 0.5 * limited_slope(Uvec, q, i, j, limiter, axis)
      next_i = i + 1 if axis == AXIS_X else i
      next_j = j if axis == AXIS_X else j + 1
      return Uvec[q, next_i, next_j] - 0.5 * limited_slope(Uvec, q, next_i, next_j, limiter, axis)
   if side == SIDE_LOWER:
      return component_value(Uvec, q, i, j, axis, 0)
   return component_value(Uvec, q, i, j, axis, 1)


@njit(cache=True, parallel=True)
def reconstruct_faces(Uvec, UxL, UxR, UyB, UyT, gamma, ng, recon, recon_vars, limiter):
   recon = literally(recon)
   recon_vars = literally(recon_vars)
   limiter = literally(limiter)
   nxp1 = UxL.shape[1]
   ny = UxL.shape[2]
   nx = UyB.shape[1]
   nyp1 = UyB.shape[2]
   mhd = Uvec.shape[0] > NVAR_HYDRO
   nvar = Uvec.shape[0]
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
         if mhd:
            pl5 = reconstruct_primitive(Uvec, 5, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
            pl6 = reconstruct_primitive(Uvec, 6, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
            pl7 = reconstruct_primitive(Uvec, 7, i, j, gamma, recon, limiter, AXIS_X, SIDE_LOWER)
            pr5 = reconstruct_primitive(Uvec, 5, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
            pr6 = reconstruct_primitive(Uvec, 6, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
            pr7 = reconstruct_primitive(Uvec, 7, i, j, gamma, recon, limiter, AXIS_X, SIDE_UPPER)
            (
               UxL[0, ii, jj],
               UxL[1, ii, jj],
               UxL[2, ii, jj],
               UxL[3, ii, jj],
               UxL[4, ii, jj],
               UxL[5, ii, jj],
               UxL[6, ii, jj],
               UxL[7, ii, jj],
            ) = conserved_from_primitive_mhd(pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, gamma)
            (
               UxR[0, ii, jj],
               UxR[1, ii, jj],
               UxR[2, ii, jj],
               UxR[3, ii, jj],
               UxR[4, ii, jj],
               UxR[5, ii, jj],
               UxR[6, ii, jj],
               UxR[7, ii, jj],
            ) = conserved_from_primitive_mhd(pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7, gamma)
         else:
            UxL[0, ii, jj], UxL[1, ii, jj], UxL[2, ii, jj], UxL[3, ii, jj], UxL[4, ii, jj] = conserved_from_primitive(
               pl0, pl1, pl2, pl3, pl4, gamma
            )
            UxR[0, ii, jj], UxR[1, ii, jj], UxR[2, ii, jj], UxR[3, ii, jj], UxR[4, ii, jj] = conserved_from_primitive(
               pr0, pr1, pr2, pr3, pr4, gamma
            )
      else:
         for q in range(nvar):
            UxL[q, ii, jj] = reconstruct_component(Uvec, q, i, j, recon, limiter, AXIS_X, SIDE_LOWER)
            UxR[q, ii, jj] = reconstruct_component(Uvec, q, i, j, recon, limiter, AXIS_X, SIDE_UPPER)

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
         if mhd:
            pb5 = reconstruct_primitive(Uvec, 5, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
            pb6 = reconstruct_primitive(Uvec, 6, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
            pb7 = reconstruct_primitive(Uvec, 7, i, j, gamma, recon, limiter, AXIS_Y, SIDE_LOWER)
            pt5 = reconstruct_primitive(Uvec, 5, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
            pt6 = reconstruct_primitive(Uvec, 6, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
            pt7 = reconstruct_primitive(Uvec, 7, i, j, gamma, recon, limiter, AXIS_Y, SIDE_UPPER)
            (
               UyB[0, ii, jj],
               UyB[1, ii, jj],
               UyB[2, ii, jj],
               UyB[3, ii, jj],
               UyB[4, ii, jj],
               UyB[5, ii, jj],
               UyB[6, ii, jj],
               UyB[7, ii, jj],
            ) = conserved_from_primitive_mhd(pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, gamma)
            (
               UyT[0, ii, jj],
               UyT[1, ii, jj],
               UyT[2, ii, jj],
               UyT[3, ii, jj],
               UyT[4, ii, jj],
               UyT[5, ii, jj],
               UyT[6, ii, jj],
               UyT[7, ii, jj],
            ) = conserved_from_primitive_mhd(pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, gamma)
         else:
            UyB[0, ii, jj], UyB[1, ii, jj], UyB[2, ii, jj], UyB[3, ii, jj], UyB[4, ii, jj] = conserved_from_primitive(
               pb0, pb1, pb2, pb3, pb4, gamma
            )
            UyT[0, ii, jj], UyT[1, ii, jj], UyT[2, ii, jj], UyT[3, ii, jj], UyT[4, ii, jj] = conserved_from_primitive(
               pt0, pt1, pt2, pt3, pt4, gamma
            )
      else:
         for q in range(nvar):
            UyB[q, ii, jj] = reconstruct_component(Uvec, q, i, j, recon, limiter, AXIS_Y, SIDE_LOWER)
            UyT[q, ii, jj] = reconstruct_component(Uvec, q, i, j, recon, limiter, AXIS_Y, SIDE_UPPER)


@njit(cache=True, parallel=True)
def reconstruct_faces_from_primitive_cache(Vcache, UxL, UxR, UyB, UyT, gamma, ng, recon, limiter):
   recon = literally(recon)
   limiter = literally(limiter)
   nxp1 = UxL.shape[1]
   ny = UxL.shape[2]
   nx = UyB.shape[1]
   nyp1 = UyB.shape[2]
   mhd = Vcache.shape[0] > NVAR_HYDRO

   for idx in prange(nxp1 * ny):
      ii = idx // ny
      jj = idx - ii * ny
      i = ng - 1 + ii
      j = ng + jj
      if mhd:
         if recon == RECON_WENOZ_CHAR:
            pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7 = (
               reconstruct_mhd_characteristic_cached(Vcache, i, j, gamma, AXIS_X)
            )
         else:
            pl0, pr0 = reconstruct_primitive_cached_pair(Vcache, 0, i, j, recon, limiter, AXIS_X)
            pl1, pr1 = reconstruct_primitive_cached_pair(Vcache, 1, i, j, recon, limiter, AXIS_X)
            pl2, pr2 = reconstruct_primitive_cached_pair(Vcache, 2, i, j, recon, limiter, AXIS_X)
            pl3, pr3 = reconstruct_primitive_cached_pair(Vcache, 3, i, j, recon, limiter, AXIS_X)
            pl4, pr4 = reconstruct_primitive_cached_pair(Vcache, 4, i, j, recon, limiter, AXIS_X)
            pl5, pr5 = reconstruct_primitive_cached_pair(Vcache, 5, i, j, recon, limiter, AXIS_X)
            pl6, pr6 = reconstruct_primitive_cached_pair(Vcache, 6, i, j, recon, limiter, AXIS_X)
            pl7, pr7 = reconstruct_primitive_cached_pair(Vcache, 7, i, j, recon, limiter, AXIS_X)
         (
            UxL[0, ii, jj],
            UxL[1, ii, jj],
            UxL[2, ii, jj],
            UxL[3, ii, jj],
            UxL[4, ii, jj],
            UxL[5, ii, jj],
            UxL[6, ii, jj],
            UxL[7, ii, jj],
         ) = conserved_from_primitive_mhd(pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7, gamma)
         (
            UxR[0, ii, jj],
            UxR[1, ii, jj],
            UxR[2, ii, jj],
            UxR[3, ii, jj],
            UxR[4, ii, jj],
            UxR[5, ii, jj],
            UxR[6, ii, jj],
            UxR[7, ii, jj],
            ) = conserved_from_primitive_mhd(pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7, gamma)
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
         UxL[0, ii, jj], UxL[1, ii, jj], UxL[2, ii, jj], UxL[3, ii, jj], UxL[4, ii, jj] = conserved_from_primitive(
            pl0, pl1, pl2, pl3, pl4, gamma
         )
         UxR[0, ii, jj], UxR[1, ii, jj], UxR[2, ii, jj], UxR[3, ii, jj], UxR[4, ii, jj] = conserved_from_primitive(
            pr0, pr1, pr2, pr3, pr4, gamma
         )

   for idx in prange(nx * nyp1):
      ii = idx // nyp1
      jj = idx - ii * nyp1
      i = ng + ii
      j = ng - 1 + jj
      if mhd:
         if recon == RECON_WENOZ_CHAR:
            pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7 = (
               reconstruct_mhd_characteristic_cached(Vcache, i, j, gamma, AXIS_Y)
            )
         else:
            pb0, pt0 = reconstruct_primitive_cached_pair(Vcache, 0, i, j, recon, limiter, AXIS_Y)
            pb1, pt1 = reconstruct_primitive_cached_pair(Vcache, 1, i, j, recon, limiter, AXIS_Y)
            pb2, pt2 = reconstruct_primitive_cached_pair(Vcache, 2, i, j, recon, limiter, AXIS_Y)
            pb3, pt3 = reconstruct_primitive_cached_pair(Vcache, 3, i, j, recon, limiter, AXIS_Y)
            pb4, pt4 = reconstruct_primitive_cached_pair(Vcache, 4, i, j, recon, limiter, AXIS_Y)
            pb5, pt5 = reconstruct_primitive_cached_pair(Vcache, 5, i, j, recon, limiter, AXIS_Y)
            pb6, pt6 = reconstruct_primitive_cached_pair(Vcache, 6, i, j, recon, limiter, AXIS_Y)
            pb7, pt7 = reconstruct_primitive_cached_pair(Vcache, 7, i, j, recon, limiter, AXIS_Y)
         (
            UyB[0, ii, jj],
            UyB[1, ii, jj],
            UyB[2, ii, jj],
            UyB[3, ii, jj],
            UyB[4, ii, jj],
            UyB[5, ii, jj],
            UyB[6, ii, jj],
            UyB[7, ii, jj],
         ) = conserved_from_primitive_mhd(pb0, pb1, pb2, pb3, pb4, pb5, pb6, pb7, gamma)
         (
            UyT[0, ii, jj],
            UyT[1, ii, jj],
            UyT[2, ii, jj],
            UyT[3, ii, jj],
            UyT[4, ii, jj],
            UyT[5, ii, jj],
            UyT[6, ii, jj],
            UyT[7, ii, jj],
         ) = conserved_from_primitive_mhd(pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, gamma)
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
         UyB[0, ii, jj], UyB[1, ii, jj], UyB[2, ii, jj], UyB[3, ii, jj], UyB[4, ii, jj] = conserved_from_primitive(
            pb0, pb1, pb2, pb3, pb4, gamma
         )
         UyT[0, ii, jj], UyT[1, ii, jj], UyT[2, ii, jj], UyT[3, ii, jj], UyT[4, ii, jj] = conserved_from_primitive(
            pt0, pt1, pt2, pt3, pt4, gamma
         )
