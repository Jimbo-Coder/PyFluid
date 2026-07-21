from numba import literally, njit
import numpy as np

from pyfluid_numba.options import *
from pyfluid_numba.physics import (
   fast_speed,
   flux_x_cell,
   flux_y_cell,
   flux_x_mhd_cell,
   flux_y_mhd_cell,
   pressure_state,
   pressure_state_mhd,
)

@njit(cache=True, inline="always")
def hydro_hll_speeds_x(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma):
   rho_l = l0
   rho_r = r0
   vx_l = l1 / rho_l
   vx_r = r1 / rho_r
   p_l = pressure_state(rho_l, l1, l2, l3, l4, gamma)
   p_r = pressure_state(rho_r, r1, r2, r3, r4, gamma)
   a_l = np.sqrt(gamma * p_l / rho_l)
   a_r = np.sqrt(gamma * p_r / rho_r)
   s_l = vx_l - a_l if vx_l - a_l < vx_r - a_r else vx_r - a_r
   s_r = vx_l + a_l if vx_l + a_l > vx_r + a_r else vx_r + a_r
   return s_l, s_r


@njit(cache=True, inline="always")
def hydro_hll_speeds_y(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma):
   rho_b = b0
   rho_t = t0
   vy_b = b2 / rho_b
   vy_t = t2 / rho_t
   p_b = pressure_state(rho_b, b1, b2, b3, b4, gamma)
   p_t = pressure_state(rho_t, t1, t2, t3, t4, gamma)
   a_b = np.sqrt(gamma * p_b / rho_b)
   a_t = np.sqrt(gamma * p_t / rho_t)
   s_b = vy_b - a_b if vy_b - a_b < vy_t - a_t else vy_t - a_t
   s_t = vy_b + a_b if vy_b + a_b > vy_t + a_t else vy_t + a_t
   return s_b, s_t


@njit(cache=True, inline="always")
def hydro_hlle_speeds_x(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma):
   rho_l = l0
   rho_r = r0
   vx_l = l1 / rho_l
   vy_l = l2 / rho_l
   vz_l = l3 / rho_l
   vx_r = r1 / rho_r
   vy_r = r2 / rho_r
   vz_r = r3 / rho_r
   p_l = pressure_state(rho_l, l1, l2, l3, l4, gamma)
   p_r = pressure_state(rho_r, r1, r2, r3, r4, gamma)
   a_l = np.sqrt(gamma * p_l / rho_l)
   a_r = np.sqrt(gamma * p_r / rho_r)
   sqrt_l = np.sqrt(rho_l)
   sqrt_r = np.sqrt(rho_r)
   inv_sum = 1.0 / (sqrt_l + sqrt_r)
   vx_roe = (sqrt_l * vx_l + sqrt_r * vx_r) * inv_sum
   vy_roe = (sqrt_l * vy_l + sqrt_r * vy_r) * inv_sum
   vz_roe = (sqrt_l * vz_l + sqrt_r * vz_r) * inv_sum
   h_l = (l4 + p_l) / rho_l
   h_r = (r4 + p_r) / rho_r
   h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv_sum
   a2_roe = (gamma - 1.0) * (h_roe - 0.5 * (vx_roe * vx_roe + vy_roe * vy_roe + vz_roe * vz_roe))
   if a2_roe < 0.0:
      a2_roe = 0.0
   a_roe = np.sqrt(a2_roe)
   q_l = np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * max(0.0, p_r / p_l - 1.0))
   q_r = np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * max(0.0, p_l / p_r - 1.0))
   s_l = vx_roe - a_roe * q_l
   s_r = vx_roe + a_roe * q_r
   side_l = vx_l - a_l if vx_l - a_l < vx_r - a_r else vx_r - a_r
   side_r = vx_l + a_l if vx_l + a_l > vx_r + a_r else vx_r + a_r
   if side_l < s_l:
      s_l = side_l
   if side_r > s_r:
      s_r = side_r
   return s_l, s_r


@njit(cache=True, inline="always")
def hydro_hlle_speeds_y(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma):
   rho_b = b0
   rho_t = t0
   vx_b = b1 / rho_b
   vy_b = b2 / rho_b
   vz_b = b3 / rho_b
   vx_t = t1 / rho_t
   vy_t = t2 / rho_t
   vz_t = t3 / rho_t
   p_b = pressure_state(rho_b, b1, b2, b3, b4, gamma)
   p_t = pressure_state(rho_t, t1, t2, t3, t4, gamma)
   a_b = np.sqrt(gamma * p_b / rho_b)
   a_t = np.sqrt(gamma * p_t / rho_t)
   sqrt_b = np.sqrt(rho_b)
   sqrt_t = np.sqrt(rho_t)
   inv_sum = 1.0 / (sqrt_b + sqrt_t)
   vx_roe = (sqrt_b * vx_b + sqrt_t * vx_t) * inv_sum
   vy_roe = (sqrt_b * vy_b + sqrt_t * vy_t) * inv_sum
   vz_roe = (sqrt_b * vz_b + sqrt_t * vz_t) * inv_sum
   h_b = (b4 + p_b) / rho_b
   h_t = (t4 + p_t) / rho_t
   h_roe = (sqrt_b * h_b + sqrt_t * h_t) * inv_sum
   a2_roe = (gamma - 1.0) * (h_roe - 0.5 * (vx_roe * vx_roe + vy_roe * vy_roe + vz_roe * vz_roe))
   if a2_roe < 0.0:
      a2_roe = 0.0
   a_roe = np.sqrt(a2_roe)
   q_b = np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * max(0.0, p_t / p_b - 1.0))
   q_t = np.sqrt(1.0 + ((gamma + 1.0) / (2.0 * gamma)) * max(0.0, p_b / p_t - 1.0))
   s_b = vy_roe - a_roe * q_b
   s_t = vy_roe + a_roe * q_t
   side_b = vy_b - a_b if vy_b - a_b < vy_t - a_t else vy_t - a_t
   side_t = vy_b + a_b if vy_b + a_b > vy_t + a_t else vy_t + a_t
   if side_b < s_b:
      s_b = side_b
   if side_t > s_t:
      s_t = side_t
   return s_b, s_t


@njit(cache=True, inline="always")
def mhd_hll_speeds_x(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma):
   rho_l = l0
   rho_r = r0
   vx_l = l1 / rho_l
   vx_r = r1 / rho_r
   p_l = pressure_state_mhd(rho_l, l1, l2, l3, l4, l5, l6, l7, gamma)
   p_r = pressure_state_mhd(rho_r, r1, r2, r3, r4, r5, r6, r7, gamma)
   cf_l = fast_speed(rho_l, p_l, l5, l6, l7, l5, gamma)
   cf_r = fast_speed(rho_r, p_r, r5, r6, r7, r5, gamma)
   s_l = vx_l - cf_l if vx_l - cf_l < vx_r - cf_r else vx_r - cf_r
   s_r = vx_l + cf_l if vx_l + cf_l > vx_r + cf_r else vx_r + cf_r
   return s_l, s_r


@njit(cache=True, inline="always")
def mhd_hll_speeds_y(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma):
   rho_b = b0
   rho_t = t0
   vy_b = b2 / rho_b
   vy_t = t2 / rho_t
   p_b = pressure_state_mhd(rho_b, b1, b2, b3, b4, b5, b6, b7, gamma)
   p_t = pressure_state_mhd(rho_t, t1, t2, t3, t4, t5, t6, t7, gamma)
   cf_b = fast_speed(rho_b, p_b, b5, b6, b7, b6, gamma)
   cf_t = fast_speed(rho_t, p_t, t5, t6, t7, t6, gamma)
   s_b = vy_b - cf_b if vy_b - cf_b < vy_t - cf_t else vy_t - cf_t
   s_t = vy_b + cf_b if vy_b + cf_b > vy_t + cf_t else vy_t + cf_t
   return s_b, s_t


@njit(cache=True, inline="always")
def mhd_hlle_speeds_x(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma):
   s_l, s_r = mhd_hll_speeds_x(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   sqrt_l = np.sqrt(l0)
   sqrt_r = np.sqrt(r0)
   inv_sum = 1.0 / (sqrt_l + sqrt_r)
   vx_roe = (sqrt_l * (l1 / l0) + sqrt_r * (r1 / r0)) * inv_sum
   bx_roe = (sqrt_l * l5 + sqrt_r * r5) * inv_sum
   by_roe = (sqrt_l * l6 + sqrt_r * r6) * inv_sum
   bz_roe = (sqrt_l * l7 + sqrt_r * r7) * inv_sum
   p_l = pressure_state_mhd(l0, l1, l2, l3, l4, l5, l6, l7, gamma)
   p_r = pressure_state_mhd(r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   rho_roe = sqrt_l * sqrt_r
   p_roe = (sqrt_l * p_l + sqrt_r * p_r) * inv_sum
   cf_roe = fast_speed(rho_roe, p_roe, bx_roe, by_roe, bz_roe, bx_roe, gamma)
   roe_l = vx_roe - cf_roe
   roe_r = vx_roe + cf_roe
   if roe_l < s_l:
      s_l = roe_l
   if roe_r > s_r:
      s_r = roe_r
   return s_l, s_r


@njit(cache=True, inline="always")
def mhd_hlle_speeds_y(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma):
   s_b, s_t = mhd_hll_speeds_y(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   sqrt_b = np.sqrt(b0)
   sqrt_t = np.sqrt(t0)
   inv_sum = 1.0 / (sqrt_b + sqrt_t)
   vy_roe = (sqrt_b * (b2 / b0) + sqrt_t * (t2 / t0)) * inv_sum
   bx_roe = (sqrt_b * b5 + sqrt_t * t5) * inv_sum
   by_roe = (sqrt_b * b6 + sqrt_t * t6) * inv_sum
   bz_roe = (sqrt_b * b7 + sqrt_t * t7) * inv_sum
   p_b = pressure_state_mhd(b0, b1, b2, b3, b4, b5, b6, b7, gamma)
   p_t = pressure_state_mhd(t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   rho_roe = sqrt_b * sqrt_t
   p_roe = (sqrt_b * p_b + sqrt_t * p_t) * inv_sum
   cf_roe = fast_speed(rho_roe, p_roe, bx_roe, by_roe, bz_roe, by_roe, gamma)
   roe_b = vy_roe - cf_roe
   roe_t = vy_roe + cf_roe
   if roe_b < s_b:
      s_b = roe_b
   if roe_t > s_t:
      s_t = roe_t
   return s_b, s_t


@njit(cache=True)
def hll_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma, einfeldt):
   einfeldt = literally(einfeldt)
   fl0, fl1, fl2, fl3, fl4, _ = flux_x_cell(l0, l1, l2, l3, l4, gamma)
   fr0, fr1, fr2, fr3, fr4, _ = flux_x_cell(r0, r1, r2, r3, r4, gamma)
   if einfeldt:
      s_l, s_r = hydro_hlle_speeds_x(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
   else:
      s_l, s_r = hydro_hll_speeds_x(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma)
   alpha = abs(s_l) if abs(s_l) > abs(s_r) else abs(s_r)
   if s_l >= 0.0:
      return fl0, fl1, fl2, fl3, fl4, alpha
   if s_r <= 0.0:
      return fr0, fr1, fr2, fr3, fr4, alpha
   inv = 1.0 / (s_r - s_l)
   return (
      (s_r * fl0 - s_l * fr0 + s_l * s_r * (r0 - l0)) * inv,
      (s_r * fl1 - s_l * fr1 + s_l * s_r * (r1 - l1)) * inv,
      (s_r * fl2 - s_l * fr2 + s_l * s_r * (r2 - l2)) * inv,
      (s_r * fl3 - s_l * fr3 + s_l * s_r * (r3 - l3)) * inv,
      (s_r * fl4 - s_l * fr4 + s_l * s_r * (r4 - l4)) * inv,
      alpha,
   )


@njit(cache=True)
def hll_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma, einfeldt):
   einfeldt = literally(einfeldt)
   fb0, fb1, fb2, fb3, fb4, _ = flux_y_cell(b0, b1, b2, b3, b4, gamma)
   ft0, ft1, ft2, ft3, ft4, _ = flux_y_cell(t0, t1, t2, t3, t4, gamma)
   if einfeldt:
      s_b, s_t = hydro_hlle_speeds_y(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
   else:
      s_b, s_t = hydro_hll_speeds_y(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma)
   alpha = abs(s_b) if abs(s_b) > abs(s_t) else abs(s_t)
   if s_b >= 0.0:
      return fb0, fb1, fb2, fb3, fb4, alpha
   if s_t <= 0.0:
      return ft0, ft1, ft2, ft3, ft4, alpha
   inv = 1.0 / (s_t - s_b)
   return (
      (s_t * fb0 - s_b * ft0 + s_b * s_t * (t0 - b0)) * inv,
      (s_t * fb1 - s_b * ft1 + s_b * s_t * (t1 - b1)) * inv,
      (s_t * fb2 - s_b * ft2 + s_b * s_t * (t2 - b2)) * inv,
      (s_t * fb3 - s_b * ft3 + s_b * s_t * (t3 - b3)) * inv,
      (s_t * fb4 - s_b * ft4 + s_b * s_t * (t4 - b4)) * inv,
      alpha,
   )


@njit(cache=True)
def hll_flux_x_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma, einfeldt):
   einfeldt = literally(einfeldt)
   fl0, fl1, fl2, fl3, fl4, fl5, fl6, fl7, _ = flux_x_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, gamma)
   fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, _ = flux_x_mhd_cell(r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   if einfeldt:
      s_l, s_r = mhd_hlle_speeds_x(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   else:
      s_l, s_r = mhd_hll_speeds_x(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   alpha = abs(s_l) if abs(s_l) > abs(s_r) else abs(s_r)
   if s_l >= 0.0:
      return fl0, fl1, fl2, fl3, fl4, fl5, fl6, fl7, alpha
   if s_r <= 0.0:
      return fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, alpha
   inv = 1.0 / (s_r - s_l)
   return (
      (s_r * fl0 - s_l * fr0 + s_l * s_r * (r0 - l0)) * inv,
      (s_r * fl1 - s_l * fr1 + s_l * s_r * (r1 - l1)) * inv,
      (s_r * fl2 - s_l * fr2 + s_l * s_r * (r2 - l2)) * inv,
      (s_r * fl3 - s_l * fr3 + s_l * s_r * (r3 - l3)) * inv,
      (s_r * fl4 - s_l * fr4 + s_l * s_r * (r4 - l4)) * inv,
      0.0,
      (s_r * fl6 - s_l * fr6 + s_l * s_r * (r6 - l6)) * inv,
      (s_r * fl7 - s_l * fr7 + s_l * s_r * (r7 - l7)) * inv,
      alpha,
   )


@njit(cache=True)
def hll_flux_y_mhd_cell(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma, einfeldt):
   einfeldt = literally(einfeldt)
   fb0, fb1, fb2, fb3, fb4, fb5, fb6, fb7, _ = flux_y_mhd_cell(b0, b1, b2, b3, b4, b5, b6, b7, gamma)
   ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, _ = flux_y_mhd_cell(t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   if einfeldt:
      s_b, s_t = mhd_hlle_speeds_y(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   else:
      s_b, s_t = mhd_hll_speeds_y(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   alpha = abs(s_b) if abs(s_b) > abs(s_t) else abs(s_t)
   if s_b >= 0.0:
      return fb0, fb1, fb2, fb3, fb4, fb5, fb6, fb7, alpha
   if s_t <= 0.0:
      return ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, alpha
   inv = 1.0 / (s_t - s_b)
   return (
      (s_t * fb0 - s_b * ft0 + s_b * s_t * (t0 - b0)) * inv,
      (s_t * fb1 - s_b * ft1 + s_b * s_t * (t1 - b1)) * inv,
      (s_t * fb2 - s_b * ft2 + s_b * s_t * (t2 - b2)) * inv,
      (s_t * fb3 - s_b * ft3 + s_b * s_t * (t3 - b3)) * inv,
      (s_t * fb4 - s_b * ft4 + s_b * s_t * (t4 - b4)) * inv,
      (s_t * fb5 - s_b * ft5 + s_b * s_t * (t5 - b5)) * inv,
      0.0,
      (s_t * fb7 - s_b * ft7 + s_b * s_t * (t7 - b7)) * inv,
      alpha,
   )


@njit(cache=True, inline="always")
def flux_normal_mhd_primitive(rho, vn, vt1, vt2, pressure, energy, bn, bt1, bt2, gamma):
   magnetic = 0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2)
   total_pressure = pressure + magnetic
   vdotb = vn * bn + vt1 * bt1 + vt2 * bt2
   return (
      rho * vn,
      rho * vn * vn + total_pressure - bn * bn,
      rho * vt1 * vn - bn * bt1,
      rho * vt2 * vn - bn * bt2,
      (energy + total_pressure) * vn - bn * vdotb,
      0.0,
      bt1 * vn - bn * vt1,
      bt2 * vn - bn * vt2,
      abs(vn) + fast_speed(rho, pressure, bn, bt1, bt2, bn, gamma),
   )


@njit(cache=True, inline="always")
def hlld_star_state(rho, vn, vt1, vt2, pressure, energy, bn, bt1, bt2, wave_speed, contact_speed, total_pressure_star, gamma):
   rho_wave = rho * (wave_speed - vn)
   rho_star = rho_wave / (wave_speed - contact_speed)
   denom = rho * (wave_speed - vn) * (wave_speed - contact_speed) - bn * bn
   if abs(denom) <= MAGNETIC_EPS:
      vt1_star = vt1
      vt2_star = vt2
      bt1_star = bt1
      bt2_star = bt2
   else:
      factor = bn * (contact_speed - vn) / denom
      vt1_star = vt1 - bt1 * factor
      vt2_star = vt2 - bt2 * factor
      bfactor = (rho * (wave_speed - vn) * (wave_speed - vn) - bn * bn) / denom
      bt1_star = bt1 * bfactor
      bt2_star = bt2 * bfactor
   vdotb = vn * bn + vt1 * bt1 + vt2 * bt2
   vdotb_star = contact_speed * bn + vt1_star * bt1_star + vt2_star * bt2_star
   energy_star = (
      (wave_speed - vn) * energy
      - (pressure + 0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2)) * vn
      + total_pressure_star * contact_speed
      + bn * (vdotb - vdotb_star)
   ) / (wave_speed - contact_speed)
   return rho_star, contact_speed, vt1_star, vt2_star, energy_star, bn, bt1_star, bt2_star


@njit(cache=True, inline="always")
def hlld_flux_normal_mhd_cell(l0, ln, lt1, lt2, l4, lbn, lbt1, lbt2, r0, rn, rt1, rt2, r4, rbn, rbt1, rbt2, gamma):
   rho_l = l0
   rho_r = r0
   vn_l = ln / rho_l
   vt1_l = lt1 / rho_l
   vt2_l = lt2 / rho_l
   vn_r = rn / rho_r
   vt1_r = rt1 / rho_r
   vt2_r = rt2 / rho_r
   pressure_l = pressure_state_mhd(rho_l, ln, lt1, lt2, l4, lbn, lbt1, lbt2, gamma)
   pressure_r = pressure_state_mhd(rho_r, rn, rt1, rt2, r4, rbn, rbt1, rbt2, gamma)

   bn = 0.5 * (lbn + rbn)
   kinetic_l = 0.5 * rho_l * (vn_l * vn_l + vt1_l * vt1_l + vt2_l * vt2_l)
   kinetic_r = 0.5 * rho_r * (vn_r * vn_r + vt1_r * vt1_r + vt2_r * vt2_r)
   energy_l = pressure_l / (gamma - 1.0) + kinetic_l + 0.5 * (bn * bn + lbt1 * lbt1 + lbt2 * lbt2)
   energy_r = pressure_r / (gamma - 1.0) + kinetic_r + 0.5 * (bn * bn + rbt1 * rbt1 + rbt2 * rbt2)

   fl0, fln, flt1, flt2, fl4, flbn, flbt1, flbt2, al = flux_normal_mhd_primitive(
      rho_l, vn_l, vt1_l, vt2_l, pressure_l, energy_l, bn, lbt1, lbt2, gamma
   )
   fr0, frn, frt1, frt2, fr4, frbn, frbt1, frbt2, ar = flux_normal_mhd_primitive(
      rho_r, vn_r, vt1_r, vt2_r, pressure_r, energy_r, bn, rbt1, rbt2, gamma
   )

   cf_l = al - abs(vn_l)
   cf_r = ar - abs(vn_r)
   s_l = vn_l - cf_l if vn_l - cf_l < vn_r - cf_r else vn_r - cf_r
   s_r = vn_l + cf_l if vn_l + cf_l > vn_r + cf_r else vn_r + cf_r
   alpha = abs(s_l) if abs(s_l) > abs(s_r) else abs(s_r)
   if s_l >= 0.0:
      return fl0, fln, flt1, flt2, fl4, flbn, flbt1, flbt2, alpha
   if s_r <= 0.0:
      return fr0, frn, frt1, frt2, fr4, frbn, frbt1, frbt2, alpha

   total_pressure_l = pressure_l + 0.5 * (bn * bn + lbt1 * lbt1 + lbt2 * lbt2)
   total_pressure_r = pressure_r + 0.5 * (bn * bn + rbt1 * rbt1 + rbt2 * rbt2)
   lfac = rho_l * (s_l - vn_l)
   rfac = rho_r * (s_r - vn_r)
   contact_speed = (rfac * vn_r - lfac * vn_l + total_pressure_l - total_pressure_r) / (rfac - lfac)
   total_pressure_star = total_pressure_l + lfac * (contact_speed - vn_l)

   (
      rho_l_star,
      vn_l_star,
      vt1_l_star,
      vt2_l_star,
      energy_l_star,
      bn_l_star,
      bt1_l_star,
      bt2_l_star,
   ) = hlld_star_state(
      rho_l, vn_l, vt1_l, vt2_l, pressure_l, energy_l, bn, lbt1, lbt2, s_l, contact_speed, total_pressure_star, gamma
   )
   (
      rho_r_star,
      vn_r_star,
      vt1_r_star,
      vt2_r_star,
      energy_r_star,
      bn_r_star,
      bt1_r_star,
      bt2_r_star,
   ) = hlld_star_state(
      rho_r, vn_r, vt1_r, vt2_r, pressure_r, energy_r, bn, rbt1, rbt2, s_r, contact_speed, total_pressure_star, gamma
   )

   ul0, uln, ult1, ult2 = rho_l, rho_l * vn_l, rho_l * vt1_l, rho_l * vt2_l
   ur0, urn, urt1, urt2 = rho_r, rho_r * vn_r, rho_r * vt1_r, rho_r * vt2_r
   uls0, ulsn = rho_l_star, rho_l_star * vn_l_star
   ulst1, ulst2 = rho_l_star * vt1_l_star, rho_l_star * vt2_l_star
   urs0, ursn = rho_r_star, rho_r_star * vn_r_star
   urst1, urst2 = rho_r_star * vt1_r_star, rho_r_star * vt2_r_star

   fls0 = fl0 + s_l * (uls0 - ul0)
   flsn = fln + s_l * (ulsn - uln)
   flst1 = flt1 + s_l * (ulst1 - ult1)
   flst2 = flt2 + s_l * (ulst2 - ult2)
   fls4 = fl4 + s_l * (energy_l_star - energy_l)
   flsbn = 0.0
   flsbt1 = flbt1 + s_l * (bt1_l_star - lbt1)
   flsbt2 = flbt2 + s_l * (bt2_l_star - lbt2)

   frs0 = fr0 + s_r * (urs0 - ur0)
   frsn = frn + s_r * (ursn - urn)
   frst1 = frt1 + s_r * (urst1 - urt1)
   frst2 = frt2 + s_r * (urst2 - urt2)
   frs4 = fr4 + s_r * (energy_r_star - energy_r)
   frsbn = 0.0
   frsbt1 = frbt1 + s_r * (bt1_r_star - rbt1)
   frsbt2 = frbt2 + s_r * (bt2_r_star - rbt2)

   abs_bn = abs(bn)
   if abs_bn <= MAGNETIC_EPS:
      if contact_speed >= 0.0:
         return fls0, flsn, flst1, flst2, fls4, flsbn, flsbt1, flsbt2, alpha
      return frs0, frsn, frst1, frst2, frs4, frsbn, frsbt1, frsbt2, alpha

   sqrt_l = np.sqrt(rho_l_star)
   sqrt_r = np.sqrt(rho_r_star)
   inv_sqrt_sum = 1.0 / (sqrt_l + sqrt_r)
   sign_bn = 1.0
   if bn < 0.0:
      sign_bn = -1.0
   s_l_star = contact_speed - abs_bn / sqrt_l
   s_r_star = contact_speed + abs_bn / sqrt_r

   vt1_double = (sqrt_l * vt1_l_star + sqrt_r * vt1_r_star + (bt1_r_star - bt1_l_star) * sign_bn) * inv_sqrt_sum
   vt2_double = (sqrt_l * vt2_l_star + sqrt_r * vt2_r_star + (bt2_r_star - bt2_l_star) * sign_bn) * inv_sqrt_sum
   bt1_double = (
      sqrt_l * bt1_r_star
      + sqrt_r * bt1_l_star
      + sqrt_l * sqrt_r * (vt1_r_star - vt1_l_star) * sign_bn
   ) * inv_sqrt_sum
   bt2_double = (
      sqrt_l * bt2_r_star
      + sqrt_r * bt2_l_star
      + sqrt_l * sqrt_r * (vt2_r_star - vt2_l_star) * sign_bn
   ) * inv_sqrt_sum

   vdotb_l_star = contact_speed * bn + vt1_l_star * bt1_l_star + vt2_l_star * bt2_l_star
   vdotb_r_star = contact_speed * bn + vt1_r_star * bt1_r_star + vt2_r_star * bt2_r_star
   vdotb_double = contact_speed * bn + vt1_double * bt1_double + vt2_double * bt2_double
   energy_l_double = energy_l_star - sqrt_l * (vdotb_l_star - vdotb_double) * sign_bn
   energy_r_double = energy_r_star + sqrt_r * (vdotb_r_star - vdotb_double) * sign_bn

   uld0, uldn = rho_l_star, rho_l_star * contact_speed
   uldt1, uldt2 = rho_l_star * vt1_double, rho_l_star * vt2_double
   urd0, urdn = rho_r_star, rho_r_star * contact_speed
   urdt1, urdt2 = rho_r_star * vt1_double, rho_r_star * vt2_double

   fld0 = fls0 + s_l_star * (uld0 - uls0)
   fldn = flsn + s_l_star * (uldn - ulsn)
   fldt1 = flst1 + s_l_star * (uldt1 - ulst1)
   fldt2 = flst2 + s_l_star * (uldt2 - ulst2)
   fld4 = fls4 + s_l_star * (energy_l_double - energy_l_star)
   fldbn = 0.0
   fldbt1 = flsbt1 + s_l_star * (bt1_double - bt1_l_star)
   fldbt2 = flsbt2 + s_l_star * (bt2_double - bt2_l_star)

   frd0 = frs0 + s_r_star * (urd0 - urs0)
   frdn = frsn + s_r_star * (urdn - ursn)
   frdt1 = frst1 + s_r_star * (urdt1 - urst1)
   frdt2 = frst2 + s_r_star * (urdt2 - urst2)
   frd4 = frs4 + s_r_star * (energy_r_double - energy_r_star)
   frdbn = 0.0
   frdbt1 = frsbt1 + s_r_star * (bt1_double - bt1_r_star)
   frdbt2 = frsbt2 + s_r_star * (bt2_double - bt2_r_star)

   if s_l_star >= 0.0:
      return fls0, flsn, flst1, flst2, fls4, flsbn, flsbt1, flsbt2, alpha
   if contact_speed >= 0.0:
      return fld0, fldn, fldt1, fldt2, fld4, fldbn, fldbt1, fldbt2, alpha
   if s_r_star > 0.0:
      return frd0, frdn, frdt1, frdt2, frd4, frdbn, frdbt1, frdbt2, alpha
   return frs0, frsn, frst1, frst2, frs4, frsbn, frsbt1, frsbt2, alpha


@njit(cache=True, inline="always")
def hlld_flux_x_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma):
   return hlld_flux_normal_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma)


@njit(cache=True, inline="always")
def hlld_flux_y_mhd_cell(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma):
   f0, fn, ft1, ft2, f4, fbn, fbt1, fbt2, alpha = hlld_flux_normal_mhd_cell(
      b0, b2, b1, b3, b4, b6, b5, b7, t0, t2, t1, t3, t4, t6, t5, t7, gamma
   )
   return f0, ft1, fn, ft2, f4, fbt1, 0.0, fbt2, alpha


@njit(cache=True, inline="always")
def hllc_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma):
   rho_l = l0
   vx_l = l1 / rho_l
   vy_l = l2 / rho_l
   vz_l = l3 / rho_l
   p_l = pressure_state(rho_l, l1, l2, l3, l4, gamma)
   rho_r = r0
   vx_r = r1 / rho_r
   vy_r = r2 / rho_r
   vz_r = r3 / rho_r
   p_r = pressure_state(rho_r, r1, r2, r3, r4, gamma)
   fl0, fl1, fl2, fl3, fl4, _ = flux_x_cell(l0, l1, l2, l3, l4, gamma)
   fr0, fr1, fr2, fr3, fr4, _ = flux_x_cell(r0, r1, r2, r3, r4, gamma)
   a_l = np.sqrt(gamma * p_l / rho_l)
   a_r = np.sqrt(gamma * p_r / rho_r)
   s_l = vx_l - a_l if vx_l - a_l < vx_r - a_r else vx_r - a_r
   s_r = vx_l + a_l if vx_l + a_l > vx_r + a_r else vx_r + a_r
   alpha = abs(s_l) if abs(s_l) > abs(s_r) else abs(s_r)
   if s_l >= 0.0:
      return fl0, fl1, fl2, fl3, fl4, alpha
   if s_r <= 0.0:
      return fr0, fr1, fr2, fr3, fr4, alpha

   lfac = rho_l * (s_l - vx_l)
   rfac = rho_r * (s_r - vx_r)
   s_m = (p_r - p_l + lfac * vx_l - rfac * vx_r) / (lfac - rfac)
   if s_m >= 0.0:
      star_rho = lfac / (s_l - s_m)
      star_e = star_rho * (l4 / rho_l + (s_m - vx_l) * (s_m + p_l / lfac))
      return (
         fl0 + s_l * (star_rho - l0),
         fl1 + s_l * (star_rho * s_m - l1),
         fl2 + s_l * (star_rho * vy_l - l2),
         fl3 + s_l * (star_rho * vz_l - l3),
         fl4 + s_l * (star_e - l4),
         alpha,
      )

   star_rho = rfac / (s_r - s_m)
   star_e = star_rho * (r4 / rho_r + (s_m - vx_r) * (s_m + p_r / rfac))
   return (
      fr0 + s_r * (star_rho - r0),
      fr1 + s_r * (star_rho * s_m - r1),
      fr2 + s_r * (star_rho * vy_r - r2),
      fr3 + s_r * (star_rho * vz_r - r3),
      fr4 + s_r * (star_e - r4),
      alpha,
   )


@njit(cache=True, inline="always")
def hllc_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma):
   rho_b = b0
   vx_b = b1 / rho_b
   vy_b = b2 / rho_b
   vz_b = b3 / rho_b
   p_b = pressure_state(rho_b, b1, b2, b3, b4, gamma)
   rho_t = t0
   vx_t = t1 / rho_t
   vy_t = t2 / rho_t
   vz_t = t3 / rho_t
   p_t = pressure_state(rho_t, t1, t2, t3, t4, gamma)
   fb0, fb1, fb2, fb3, fb4, _ = flux_y_cell(b0, b1, b2, b3, b4, gamma)
   ft0, ft1, ft2, ft3, ft4, _ = flux_y_cell(t0, t1, t2, t3, t4, gamma)
   a_b = np.sqrt(gamma * p_b / rho_b)
   a_t = np.sqrt(gamma * p_t / rho_t)
   s_b = vy_b - a_b if vy_b - a_b < vy_t - a_t else vy_t - a_t
   s_t = vy_b + a_b if vy_b + a_b > vy_t + a_t else vy_t + a_t
   alpha = abs(s_b) if abs(s_b) > abs(s_t) else abs(s_t)
   if s_b >= 0.0:
      return fb0, fb1, fb2, fb3, fb4, alpha
   if s_t <= 0.0:
      return ft0, ft1, ft2, ft3, ft4, alpha

   bfac = rho_b * (s_b - vy_b)
   tfac = rho_t * (s_t - vy_t)
   s_m = (p_t - p_b + bfac * vy_b - tfac * vy_t) / (bfac - tfac)
   if s_m >= 0.0:
      star_rho = bfac / (s_b - s_m)
      star_e = star_rho * (b4 / rho_b + (s_m - vy_b) * (s_m + p_b / bfac))
      return (
         fb0 + s_b * (star_rho - b0),
         fb1 + s_b * (star_rho * vx_b - b1),
         fb2 + s_b * (star_rho * s_m - b2),
         fb3 + s_b * (star_rho * vz_b - b3),
         fb4 + s_b * (star_e - b4),
         alpha,
      )

   star_rho = tfac / (s_t - s_m)
   star_e = star_rho * (t4 / rho_t + (s_m - vy_t) * (s_m + p_t / tfac))
   return (
      ft0 + s_t * (star_rho - t0),
      ft1 + s_t * (star_rho * vx_t - t1),
      ft2 + s_t * (star_rho * s_m - t2),
      ft3 + s_t * (star_rho * vz_t - t3),
      ft4 + s_t * (star_e - t4),
      alpha,
   )


@njit(cache=True, inline="always")
def llf_flux_x_cell(l0, l1, l2, l3, l4, r0, r1, r2, r3, r4, gamma):
   fl0, fl1, fl2, fl3, fl4, al = flux_x_cell(l0, l1, l2, l3, l4, gamma)
   fr0, fr1, fr2, fr3, fr4, ar = flux_x_cell(r0, r1, r2, r3, r4, gamma)
   a = al if al > ar else ar
   return (
      0.5 * (fl0 + fr0 - a * (r0 - l0)),
      0.5 * (fl1 + fr1 - a * (r1 - l1)),
      0.5 * (fl2 + fr2 - a * (r2 - l2)),
      0.5 * (fl3 + fr3 - a * (r3 - l3)),
      0.5 * (fl4 + fr4 - a * (r4 - l4)),
      a,
   )


@njit(cache=True, inline="always")
def llf_flux_y_cell(b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, gamma):
   fb0, fb1, fb2, fb3, fb4, ab = flux_y_cell(b0, b1, b2, b3, b4, gamma)
   ft0, ft1, ft2, ft3, ft4, at = flux_y_cell(t0, t1, t2, t3, t4, gamma)
   a = ab if ab > at else at
   return (
      0.5 * (fb0 + ft0 - a * (t0 - b0)),
      0.5 * (fb1 + ft1 - a * (t1 - b1)),
      0.5 * (fb2 + ft2 - a * (t2 - b2)),
      0.5 * (fb3 + ft3 - a * (t3 - b3)),
      0.5 * (fb4 + ft4 - a * (t4 - b4)),
      a,
   )


@njit(cache=True, inline="always")
def llf_flux_x_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, gamma):
   fl0, fl1, fl2, fl3, fl4, fl5, fl6, fl7, al = flux_x_mhd_cell(l0, l1, l2, l3, l4, l5, l6, l7, gamma)
   fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, ar = flux_x_mhd_cell(r0, r1, r2, r3, r4, r5, r6, r7, gamma)
   a = al if al > ar else ar
   return (
      0.5 * (fl0 + fr0 - a * (r0 - l0)),
      0.5 * (fl1 + fr1 - a * (r1 - l1)),
      0.5 * (fl2 + fr2 - a * (r2 - l2)),
      0.5 * (fl3 + fr3 - a * (r3 - l3)),
      0.5 * (fl4 + fr4 - a * (r4 - l4)),
      0.0,
      0.5 * (fl6 + fr6 - a * (r6 - l6)),
      0.5 * (fl7 + fr7 - a * (r7 - l7)),
      a,
   )


@njit(cache=True, inline="always")
def llf_flux_y_mhd_cell(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, gamma):
   fb0, fb1, fb2, fb3, fb4, fb5, fb6, fb7, ab = flux_y_mhd_cell(b0, b1, b2, b3, b4, b5, b6, b7, gamma)
   ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, at = flux_y_mhd_cell(t0, t1, t2, t3, t4, t5, t6, t7, gamma)
   a = ab if ab > at else at
   return (
      0.5 * (fb0 + ft0 - a * (t0 - b0)),
      0.5 * (fb1 + ft1 - a * (t1 - b1)),
      0.5 * (fb2 + ft2 - a * (t2 - b2)),
      0.5 * (fb3 + ft3 - a * (t3 - b3)),
      0.5 * (fb4 + ft4 - a * (t4 - b4)),
      0.5 * (fb5 + ft5 - a * (t5 - b5)),
      0.0,
      0.5 * (fb7 + ft7 - a * (t7 - b7)),
      a,
   )
