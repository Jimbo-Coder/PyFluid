import numpy as np
from numba import njit


HYDRO_MODE_ENTROPY = 0
HYDRO_MODE_SHEAR = 1
HYDRO_MODE_SHEAR_Z = 2
HYDRO_MODE_SOUND_MINUS = 3
HYDRO_MODE_SOUND_PLUS = 4

MHD_MODE_FAST_MINUS = 0
MHD_MODE_ALFVEN_MINUS = 1
MHD_MODE_SLOW_MINUS = 2
MHD_MODE_ENTROPY = 3
MHD_MODE_SLOW_PLUS = 4
MHD_MODE_ALFVEN_PLUS = 5
MHD_MODE_FAST_PLUS = 6


def hydro_mode_name(mode):
   if mode == HYDRO_MODE_ENTROPY:
      return "entropy"
   if mode == HYDRO_MODE_SHEAR:
      return "shear"
   if mode == HYDRO_MODE_SHEAR_Z:
      return "shear-z"
   if mode == HYDRO_MODE_SOUND_MINUS:
      return "sound-"
   if mode == HYDRO_MODE_SOUND_PLUS:
      return "sound+"
   raise ValueError(f"unsupported hydro characteristic mode: {mode}")


def hydro_characteristic_modes():
   return (
      HYDRO_MODE_ENTROPY,
      HYDRO_MODE_SHEAR,
      HYDRO_MODE_SHEAR_Z,
      HYDRO_MODE_SOUND_MINUS,
      HYDRO_MODE_SOUND_PLUS,
   )


def validate_hydro_characteristic_mode(mode):
   mode = int(mode)
   hydro_mode_name(mode)
   return mode


def hydro_primitive_right_eigenvectors(rho, pressure, gamma, normal_x, normal_y):
   sound_speed = np.sqrt(gamma * pressure / rho)
   tangent_x = -normal_y
   tangent_y = normal_x
   right = np.zeros((5, 5), dtype=float)
   right[:, HYDRO_MODE_ENTROPY] = (1.0, 0.0, 0.0, 0.0, 0.0)
   right[:, HYDRO_MODE_SHEAR] = (0.0, tangent_x, tangent_y, 0.0, 0.0)
   right[:, HYDRO_MODE_SHEAR_Z] = (0.0, 0.0, 0.0, 1.0, 0.0)
   right[:, HYDRO_MODE_SOUND_MINUS] = (
      1.0,
      -sound_speed * normal_x / rho,
      -sound_speed * normal_y / rho,
      0.0,
      sound_speed * sound_speed,
   )
   right[:, HYDRO_MODE_SOUND_PLUS] = (
      1.0,
      sound_speed * normal_x / rho,
      sound_speed * normal_y / rho,
      0.0,
      sound_speed * sound_speed,
   )
   return right


def hydro_primitive_left_eigenvectors(rho, pressure, gamma, normal_x, normal_y):
   return np.linalg.inv(hydro_primitive_right_eigenvectors(rho, pressure, gamma, normal_x, normal_y))


def hydro_characteristic_eigenvalues(rho, pressure, vx, vy, gamma, normal_x, normal_y):
   sound_speed = np.sqrt(gamma * pressure / rho)
   normal_velocity = vx * normal_x + vy * normal_y
   return np.array(
      (
         normal_velocity,
         normal_velocity,
         normal_velocity,
         normal_velocity - sound_speed,
         normal_velocity + sound_speed,
      ),
      dtype=np.float64,
   )


def mhd_mode_name(mode):
   if mode == MHD_MODE_FAST_MINUS:
      return "fast-"
   if mode == MHD_MODE_ALFVEN_MINUS:
      return "alfven-"
   if mode == MHD_MODE_SLOW_MINUS:
      return "slow-"
   if mode == MHD_MODE_ENTROPY:
      return "entropy"
   if mode == MHD_MODE_SLOW_PLUS:
      return "slow+"
   if mode == MHD_MODE_ALFVEN_PLUS:
      return "alfven+"
   if mode == MHD_MODE_FAST_PLUS:
      return "fast+"
   raise ValueError(f"unsupported MHD characteristic mode: {mode}")


def mhd_characteristic_modes():
   return (
      MHD_MODE_FAST_MINUS,
      MHD_MODE_ALFVEN_MINUS,
      MHD_MODE_SLOW_MINUS,
      MHD_MODE_ENTROPY,
      MHD_MODE_SLOW_PLUS,
      MHD_MODE_ALFVEN_PLUS,
      MHD_MODE_FAST_PLUS,
   )


def validate_mhd_characteristic_mode(mode):
   mode = int(mode)
   mhd_mode_name(mode)
   return mode


@njit(cache=True, inline="always")
def mhd_magnetosonic_speeds(rho, pressure, b_normal, b_tangent, b_z, gamma):
   sound2 = gamma * pressure / rho
   b2 = (b_normal * b_normal + b_tangent * b_tangent + b_z * b_z) / rho
   ca2 = b_normal * b_normal / rho
   discr = (sound2 + b2) * (sound2 + b2) - 4.0 * sound2 * ca2
   if discr < 0.0 and discr > -1.0e-14:
      discr = 0.0
   discr = max(discr, 0.0)
   fast2 = 0.5 * (sound2 + b2 + np.sqrt(discr))
   slow2 = 0.5 * (sound2 + b2 - np.sqrt(discr))
   return np.sqrt(sound2), np.sqrt(ca2), np.sqrt(max(fast2, 0.0)), np.sqrt(max(slow2, 0.0))


@njit(cache=True)
def mhd_primitive_reduced_right_eigenvectors(rho, pressure, b_normal, b_tangent, b_z, gamma):
   sound_speed, alfven_speed, fast_speed, slow_speed = mhd_magnetosonic_speeds(
      rho, pressure, b_normal, b_tangent, b_z, gamma
   )
   sqrt_rho = np.sqrt(rho)
   b_perp = np.sqrt(b_tangent * b_tangent + b_z * b_z)
   if b_perp > 1.0e-14:
      beta_t = b_tangent / b_perp
      beta_z = b_z / b_perp
   else:
      beta_t = 1.0 / np.sqrt(2.0)
      beta_z = 1.0 / np.sqrt(2.0)
   denom = fast_speed * fast_speed - slow_speed * slow_speed
   if denom <= 1.0e-14:
      alpha_f = 1.0
      alpha_s = 0.0
   else:
      alpha_f = np.sqrt(max((sound_speed * sound_speed - slow_speed * slow_speed) / denom, 0.0))
      alpha_s = np.sqrt(max((fast_speed * fast_speed - sound_speed * sound_speed) / denom, 0.0))
   normal_sign = 1.0 if b_normal >= 0.0 else -1.0
   right = np.zeros((7, 7), dtype=float)
   right[:, MHD_MODE_FAST_MINUS] = (
      alpha_f * rho,
      -alpha_f * fast_speed,
      alpha_s * slow_speed * beta_t * normal_sign,
      alpha_s * slow_speed * beta_z * normal_sign,
      alpha_f * rho * sound_speed * sound_speed,
      alpha_s * sound_speed * sqrt_rho * beta_t,
      alpha_s * sound_speed * sqrt_rho * beta_z,
   )
   right[:, MHD_MODE_ALFVEN_MINUS] = (
      0.0,
      0.0,
      beta_z * normal_sign,
      -beta_t * normal_sign,
      0.0,
      beta_z * sqrt_rho,
      -beta_t * sqrt_rho,
   )
   right[:, MHD_MODE_SLOW_MINUS] = (
      alpha_s * rho,
      -alpha_s * slow_speed,
      -alpha_f * fast_speed * beta_t * normal_sign,
      -alpha_f * fast_speed * beta_z * normal_sign,
      alpha_s * rho * sound_speed * sound_speed,
      -alpha_f * sound_speed * sqrt_rho * beta_t,
      -alpha_f * sound_speed * sqrt_rho * beta_z,
   )
   right[:, MHD_MODE_ENTROPY] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
   right[:, MHD_MODE_SLOW_PLUS] = (
      alpha_s * rho,
      alpha_s * slow_speed,
      alpha_f * fast_speed * beta_t * normal_sign,
      alpha_f * fast_speed * beta_z * normal_sign,
      alpha_s * rho * sound_speed * sound_speed,
      -alpha_f * sound_speed * sqrt_rho * beta_t,
      -alpha_f * sound_speed * sqrt_rho * beta_z,
   )
   right[:, MHD_MODE_ALFVEN_PLUS] = (
      0.0,
      0.0,
      -beta_z * normal_sign,
      beta_t * normal_sign,
      0.0,
      beta_z * sqrt_rho,
      -beta_t * sqrt_rho,
   )
   right[:, MHD_MODE_FAST_PLUS] = (
      alpha_f * rho,
      alpha_f * fast_speed,
      -alpha_s * slow_speed * beta_t * normal_sign,
      -alpha_s * slow_speed * beta_z * normal_sign,
      alpha_f * rho * sound_speed * sound_speed,
      alpha_s * sound_speed * sqrt_rho * beta_t,
      alpha_s * sound_speed * sqrt_rho * beta_z,
   )
   return right


def mhd_primitive_reduced_left_eigenvectors(rho, pressure, b_normal, b_tangent, b_z, gamma):
   return np.linalg.inv(mhd_primitive_reduced_right_eigenvectors(rho, pressure, b_normal, b_tangent, b_z, gamma))


@njit(cache=True)
def mhd_primitive_reduced_eigenvalues(rho, pressure, v_normal, b_normal, b_tangent, b_z, gamma):
   _, alfven_speed, fast_speed, slow_speed = mhd_magnetosonic_speeds(rho, pressure, b_normal, b_tangent, b_z, gamma)
   return np.array(
      (
         v_normal - fast_speed,
         v_normal - alfven_speed,
         v_normal - slow_speed,
         v_normal,
         v_normal + slow_speed,
         v_normal + alfven_speed,
         v_normal + fast_speed,
      ),
      dtype=np.float64,
   )


def mhd_primitive_reduced_jacobian(rho, v_normal, pressure, b_normal, b_tangent, b_z, gamma):
   return np.array(
      (
         (v_normal, rho, 0.0, 0.0, 0.0, 0.0, 0.0),
         (0.0, v_normal, 0.0, 0.0, 1.0 / rho, b_tangent / rho, b_z / rho),
         (0.0, 0.0, v_normal, 0.0, 0.0, -b_normal / rho, 0.0),
         (0.0, 0.0, 0.0, v_normal, 0.0, 0.0, -b_normal / rho),
         (0.0, gamma * pressure, 0.0, 0.0, v_normal, 0.0, 0.0),
         (0.0, b_tangent, -b_normal, 0.0, 0.0, v_normal, 0.0),
         (0.0, b_z, 0.0, -b_normal, 0.0, 0.0, v_normal),
      ),
      dtype=float,
   )


@njit(cache=True, inline="always")
def hydro_mode_speed(mode, rho, pressure, vx, vy, gamma, kx, ky):
   k_norm = np.sqrt(kx * kx + ky * ky)
   normal_x = kx / k_norm
   normal_y = ky / k_norm
   sound_speed = np.sqrt(gamma * pressure / rho)
   speed = vx * normal_x + vy * normal_y
   if mode == HYDRO_MODE_SOUND_MINUS:
      speed -= sound_speed
   elif mode == HYDRO_MODE_SOUND_PLUS:
      speed += sound_speed
   return speed


@njit(cache=True, inline="always")
def hydro_mode_primitive_perturbation(mode, rho, pressure, gamma, kx, ky, amplitude, wave):
   k_norm = np.sqrt(kx * kx + ky * ky)
   normal_x = kx / k_norm
   normal_y = ky / k_norm
   tangent_x = -normal_y
   tangent_y = normal_x
   sound_speed = np.sqrt(gamma * pressure / rho)
   value = amplitude * wave
   drho = 0.0
   dvx = 0.0
   dvy = 0.0
   dvz = 0.0
   dpressure = 0.0
   if mode == HYDRO_MODE_ENTROPY:
      drho = value
   elif mode == HYDRO_MODE_SHEAR:
      dvx = tangent_x * value
      dvy = tangent_y * value
   elif mode == HYDRO_MODE_SHEAR_Z:
      dvz = value
   elif mode == HYDRO_MODE_SOUND_MINUS:
      drho = value
      dvx = -sound_speed * normal_x * value / rho
      dvy = -sound_speed * normal_y * value / rho
      dpressure = sound_speed * sound_speed * value
   elif mode == HYDRO_MODE_SOUND_PLUS:
      drho = value
      dvx = sound_speed * normal_x * value / rho
      dvy = sound_speed * normal_y * value / rho
      dpressure = sound_speed * sound_speed * value
   return drho, dvx, dvy, dvz, dpressure


@njit(cache=True, inline="always")
def hydro_mode_norm(rho, pressure, gamma, kx, ky, mode):
   drho, dvx, dvy, dvz, dpressure = hydro_mode_primitive_perturbation(mode, rho, pressure, gamma, kx, ky, 1.0, 1.0)
   return np.sqrt(drho * drho + dvx * dvx + dvy * dvy + dvz * dvz + dpressure * dpressure)


@njit(cache=True, inline="always")
def mhd_mode_speed(mode, rho, pressure, vx, vy, bx, by, bz, gamma, kx, ky):
   k_norm = np.sqrt(kx * kx + ky * ky)
   normal_x = kx / k_norm
   normal_y = ky / k_norm
   v_normal = vx * normal_x + vy * normal_y
   b_normal = bx * normal_x + by * normal_y
   b_tangent = -normal_y * bx + normal_x * by
   eigvals = mhd_primitive_reduced_eigenvalues(rho, pressure, v_normal, b_normal, b_tangent, bz, gamma)
   return eigvals[mode]


@njit(cache=True, inline="always")
def mhd_mode_primitive_perturbation(mode, rho, pressure, bx, by, bz, gamma, kx, ky, amplitude, wave):
   k_norm = np.sqrt(kx * kx + ky * ky)
   normal_x = kx / k_norm
   normal_y = ky / k_norm
   tangent_x = -normal_y
   tangent_y = normal_x
   b_normal = bx * normal_x + by * normal_y
   b_tangent = bx * tangent_x + by * tangent_y
   right = mhd_primitive_reduced_right_eigenvectors(rho, pressure, b_normal, b_tangent, bz, gamma)
   value = amplitude * wave
   drho = value * right[0, mode]
   dv_normal = value * right[1, mode]
   dv_tangent = value * right[2, mode]
   dvz = value * right[3, mode]
   dpressure = value * right[4, mode]
   db_tangent = value * right[5, mode]
   dbz = value * right[6, mode]
   dvx = normal_x * dv_normal + tangent_x * dv_tangent
   dvy = normal_y * dv_normal + tangent_y * dv_tangent
   dbx = tangent_x * db_tangent
   dby = tangent_y * db_tangent
   return drho, dvx, dvy, dvz, dpressure, dbx, dby, dbz


@njit(cache=True, inline="always")
def mhd_mode_norm(rho, pressure, bx, by, bz, gamma, kx, ky, mode):
   drho, dvx, dvy, dvz, dpressure, dbx, dby, dbz = mhd_mode_primitive_perturbation(
      mode, rho, pressure, bx, by, bz, gamma, kx, ky, 1.0, 1.0
   )
   return np.sqrt(
      drho * drho
      + dvx * dvx
      + dvy * dvy
      + dvz * dvz
      + dpressure * dpressure
      + dbx * dbx
      + dby * dby
      + dbz * dbz
   )
