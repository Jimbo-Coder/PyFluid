#pragma once

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

#include "pyfluid_amrex/characteristics.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"

namespace pyfluid_amrex {

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDCharacteristicBasis mhd_characteristic_basis(
   amrex::Real rho_ref,
   amrex::Real pressure_ref,
   amrex::Real b_normal,
   amrex::Real b_tangent1,
   amrex::Real b_tangent2,
   EOSParams const& eos
) noexcept
{
   const amrex::Real sound_speed = sound_speed_from_primitive<EOSMethod>(rho_ref, pressure_ref, eos);
   const amrex::Real sound2 = sound_speed * sound_speed;
   const amrex::Real b2 = (b_normal * b_normal + b_tangent1 * b_tangent1 + b_tangent2 * b_tangent2) / rho_ref;
   const amrex::Real ca2 = b_normal * b_normal / rho_ref;
   amrex::Real discriminant = (sound2 + b2) * (sound2 + b2) - 4.0 * sound2 * ca2;
   if (discriminant < 0.0 && discriminant > -1.0e-14) {
      discriminant = 0.0;
   }
   discriminant = amrex::max(discriminant, 0.0);
   const amrex::Real root = std::sqrt(discriminant);
   const amrex::Real fast2 = amrex::max(0.5 * (sound2 + b2 + root), 0.0);
   const amrex::Real slow2 = amrex::max(0.5 * (sound2 + b2 - root), 0.0);
   const amrex::Real b_perp = std::sqrt(b_tangent1 * b_tangent1 + b_tangent2 * b_tangent2);

   amrex::Real beta_t1 = 1.0 / std::sqrt(2.0);
   amrex::Real beta_t2 = beta_t1;
   if (b_perp > 1.0e-14) {
      beta_t1 = b_tangent1 / b_perp;
      beta_t2 = b_tangent2 / b_perp;
   }

   amrex::Real alpha_f = 1.0;
   amrex::Real alpha_s = 0.0;
   const amrex::Real denominator = fast2 - slow2;
   if (denominator > 1.0e-14) {
      alpha_f = std::sqrt(amrex::max((sound2 - slow2) / denominator, 0.0));
      alpha_s = std::sqrt(amrex::max((fast2 - sound2) / denominator, 0.0));
   }
   amrex::Real velocity_denominator = alpha_f * alpha_f * fast2 + alpha_s * alpha_s * slow2;
   if (velocity_denominator <= 1.0e-14) {
      velocity_denominator = sound2;
   }
   const amrex::Real sqrt_rho = std::sqrt(rho_ref);
   const amrex::Real inv_sound_speed = 1.0 / sound_speed;

   return {
      rho_ref,
      1.0 / rho_ref,
      sqrt_rho,
      1.0 / sqrt_rho,
      sound_speed,
      inv_sound_speed,
      sound2,
      inv_sound_speed * inv_sound_speed,
      std::sqrt(fast2),
      std::sqrt(slow2),
      alpha_f,
      alpha_s,
      beta_t1,
      beta_t2,
      b_normal < 0.0 ? -1.0 : 1.0,
      velocity_denominator,
      1.0 / velocity_denominator,
   };
}

} // namespace pyfluid_amrex
