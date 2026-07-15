from __future__ import annotations

import sympy as sp

from .kernel_equations import StructKernel


def hydro_characteristic_kernels() -> tuple[StructKernel, StructKernel]:
   rho, vn, vt1, vt2, pressure = sp.symbols(
      "primitive.rho primitive.vn primitive.vt1 primitive.vt2 primitive.pressure"
   )
   rho_ref, inv_rho, sound_speed, sound2, inv_sound_speed, inv_sound2 = sp.symbols(
      "basis.rho basis.inv_rho basis.sound_speed basis.sound2 "
      "basis.inv_sound_speed basis.inv_sound2"
   )
   projection = StructKernel(
      name="hydro_characteristics",
      return_type="HydroCharacteristics",
      arguments=(
         ("const HydroNormalPrimitive&", "primitive"),
         ("const HydroCharacteristicBasis&", "basis"),
      ),
      outputs=("entropy", "shear1", "shear2", "sound_minus", "sound_plus"),
      expressions=(
         rho - pressure * inv_sound2,
         vt1,
         vt2,
         sp.Rational(1, 2) * (pressure * inv_sound2 - rho_ref * vn * inv_sound_speed),
         sp.Rational(1, 2) * (pressure * inv_sound2 + rho_ref * vn * inv_sound_speed),
      ),
   )

   entropy, shear1, shear2, sound_minus, sound_plus = sp.symbols(
      "waves.entropy waves.shear1 waves.shear2 waves.sound_minus waves.sound_plus"
   )
   recovery = StructKernel(
      name="hydro_normal_primitive_from_characteristics",
      return_type="HydroNormalPrimitive",
      arguments=(
         ("const HydroCharacteristics&", "waves"),
         ("const HydroCharacteristicBasis&", "basis"),
      ),
      outputs=("rho", "vn", "vt1", "vt2", "pressure"),
      expressions=(
         entropy + sound_minus + sound_plus,
         sound_speed * (sound_plus - sound_minus) * inv_rho,
         shear1,
         shear2,
         sound2 * (sound_minus + sound_plus),
      ),
   )
   return projection, recovery


def mhd_characteristic_kernels() -> tuple[StructKernel, StructKernel]:
   rho, vn, vt1, vt2, pressure, bt1, bt2 = sp.symbols(
      "primitive.rho primitive.vn primitive.vt1 primitive.vt2 "
      "primitive.pressure primitive.bt1 primitive.bt2"
   )
   basis_rho, inv_rho, sqrt_rho, inv_sqrt_rho = sp.symbols(
      "basis.rho basis.inv_rho basis.sqrt_rho basis.inv_sqrt_rho"
   )
   sound_speed, inv_sound_speed, sound2, inv_sound2, fast_speed, slow_speed = sp.symbols(
      "basis.sound_speed basis.inv_sound_speed basis.sound2 basis.inv_sound2 "
      "basis.fast_speed basis.slow_speed"
   )
   alpha_f, alpha_s, beta_t1, beta_t2, normal_sign = sp.symbols(
      "basis.alpha_f basis.alpha_s basis.beta_t1 basis.beta_t2 "
      "basis.normal_sign"
   )
   inv_velocity_denom = sp.Symbol("basis.inv_velocity_denom")
   compressive = pressure * inv_sound2 * inv_rho
   b_parallel = (beta_t1 * bt1 + beta_t2 * bt2) * inv_sound_speed * inv_sqrt_rho
   fast_sum = alpha_f * compressive + alpha_s * b_parallel
   slow_sum = alpha_s * compressive - alpha_f * b_parallel
   v_parallel = normal_sign * (beta_t1 * vt1 + beta_t2 * vt2)
   fast_diff = (
      alpha_f * fast_speed * vn - alpha_s * slow_speed * v_parallel
   ) * inv_velocity_denom
   slow_diff = (
      alpha_s * slow_speed * vn + alpha_f * fast_speed * v_parallel
   ) * inv_velocity_denom
   alfven_sum = (beta_t2 * bt1 - beta_t1 * bt2) * inv_sqrt_rho
   alfven_diff = normal_sign * (beta_t2 * vt1 - beta_t1 * vt2)
   projection = StructKernel(
      name="mhd_characteristics",
      return_type="MHDCharacteristics",
      arguments=(
         ("const MHDReducedPrimitive&", "primitive"),
         ("const MHDCharacteristicBasis&", "basis"),
      ),
      outputs=(
         "fast_minus",
         "alfven_minus",
         "slow_minus",
         "entropy",
         "slow_plus",
         "alfven_plus",
         "fast_plus",
      ),
      expressions=(
         sp.Rational(1, 2) * (fast_sum - fast_diff),
         sp.Rational(1, 2) * (alfven_sum + alfven_diff),
         sp.Rational(1, 2) * (slow_sum - slow_diff),
         rho - pressure * inv_sound2,
         sp.Rational(1, 2) * (slow_sum + slow_diff),
         sp.Rational(1, 2) * (alfven_sum - alfven_diff),
         sp.Rational(1, 2) * (fast_sum + fast_diff),
      ),
   )

   fast_minus, alfven_minus, slow_minus, entropy = sp.symbols(
      "waves.fast_minus waves.alfven_minus waves.slow_minus waves.entropy"
   )
   slow_plus, alfven_plus, fast_plus = sp.symbols(
      "waves.slow_plus waves.alfven_plus waves.fast_plus"
   )
   fast_sum = fast_minus + fast_plus
   fast_diff = fast_plus - fast_minus
   slow_sum = slow_minus + slow_plus
   slow_diff = slow_plus - slow_minus
   alfven_sum = alfven_minus + alfven_plus
   alfven_diff = alfven_minus - alfven_plus
   recovery = StructKernel(
      name="mhd_reduced_from_characteristics",
      return_type="MHDReducedPrimitive",
      arguments=(
         ("const MHDCharacteristics&", "waves"),
         ("const MHDCharacteristicBasis&", "basis"),
      ),
      outputs=("rho", "vn", "vt1", "vt2", "pressure", "bt1", "bt2"),
      expressions=(
         alpha_f * basis_rho * fast_sum + alpha_s * basis_rho * slow_sum + entropy,
         alpha_f * fast_speed * fast_diff + alpha_s * slow_speed * slow_diff,
         normal_sign
         * (
            -alpha_s * slow_speed * beta_t1 * fast_diff
            + alpha_f * fast_speed * beta_t1 * slow_diff
            + beta_t2 * alfven_diff
         ),
         normal_sign
         * (
            -alpha_s * slow_speed * beta_t2 * fast_diff
            + alpha_f * fast_speed * beta_t2 * slow_diff
            - beta_t1 * alfven_diff
         ),
         alpha_f * basis_rho * sound2 * fast_sum
         + alpha_s * basis_rho * sound2 * slow_sum,
         alpha_s * sound_speed * sqrt_rho * beta_t1 * fast_sum
         - alpha_f * sound_speed * sqrt_rho * beta_t1 * slow_sum
         + beta_t2 * sqrt_rho * alfven_sum,
         alpha_s * sound_speed * sqrt_rho * beta_t2 * fast_sum
         - alpha_f * sound_speed * sqrt_rho * beta_t2 * slow_sum
         - beta_t1 * sqrt_rho * alfven_sum,
      ),
   )
   return projection, recovery


def characteristic_kernels() -> tuple[StructKernel, ...]:
   return hydro_characteristic_kernels() + mhd_characteristic_kernels()
