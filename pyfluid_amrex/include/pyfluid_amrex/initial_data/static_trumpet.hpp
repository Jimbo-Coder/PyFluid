#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

// Regular conformal variables for the analytic static Schwarzschild trumpet.
// The physical ADM metric diverges at the puncture, while the evolved Z4c
// variables have finite radial limits. A_tilde is direction dependent there;
// its spherical average is zero and supplies the single nodal puncture value.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState static_trumpet_z4c_initial_data_state(
   const FixedSpacetimeParameters& parameters,
   amrex::Real position1,
   amrex::Real position2,
   amrex::Real position3
) noexcept
{
   const amrex::Real displacement[3]{
      position1 - parameters.origin1,
      position2 - parameters.origin2,
      position3 - parameters.origin3,
   };
   const amrex::Real radius_squared =
      displacement[0] * displacement[0]
      + displacement[1] * displacement[1]
      + displacement[2] * displacement[2];
   const amrex::Real radius = std::sqrt(radius_squared);
   const amrex::Real radius_plus_mass = radius + parameters.mass;
   const amrex::Real inverse_radius_plus_mass = 1.0 / radius_plus_mass;
   const amrex::Real conformal_curvature_scale = parameters.mass
      * inverse_radius_plus_mass * inverse_radius_plus_mass;

   Z4cState state;
   state.chi = radius_squared * inverse_radius_plus_mass
      * inverse_radius_plus_mass;
   state.gamma_tilde11 = 1.0;
   state.gamma_tilde22 = 1.0;
   state.gamma_tilde33 = 1.0;
   state.k_hat = conformal_curvature_scale;
   state.lapse = radius * inverse_radius_plus_mass;
   state.shift1 = parameters.mass * displacement[0]
      * inverse_radius_plus_mass * inverse_radius_plus_mass;
   state.shift2 = parameters.mass * displacement[1]
      * inverse_radius_plus_mass * inverse_radius_plus_mass;
   state.shift3 = parameters.mass * displacement[2]
      * inverse_radius_plus_mass * inverse_radius_plus_mass;

   if (radius_squared > 0.0) {
      const amrex::Real inverse_radius_squared = 1.0 / radius_squared;
      const amrex::Real unit1_unit1 = displacement[0] * displacement[0]
         * inverse_radius_squared;
      const amrex::Real unit1_unit2 = displacement[0] * displacement[1]
         * inverse_radius_squared;
      const amrex::Real unit1_unit3 = displacement[0] * displacement[2]
         * inverse_radius_squared;
      const amrex::Real unit2_unit2 = displacement[1] * displacement[1]
         * inverse_radius_squared;
      const amrex::Real unit2_unit3 = displacement[1] * displacement[2]
         * inverse_radius_squared;
      const amrex::Real unit3_unit3 = displacement[2] * displacement[2]
         * inverse_radius_squared;
      state.a_tilde11 = conformal_curvature_scale
         * (2.0 / 3.0 - 2.0 * unit1_unit1);
      state.a_tilde12 = -2.0 * conformal_curvature_scale * unit1_unit2;
      state.a_tilde13 = -2.0 * conformal_curvature_scale * unit1_unit3;
      state.a_tilde22 = conformal_curvature_scale
         * (2.0 / 3.0 - 2.0 * unit2_unit2);
      state.a_tilde23 = -2.0 * conformal_curvature_scale * unit2_unit3;
      state.a_tilde33 = conformal_curvature_scale
         * (2.0 / 3.0 - 2.0 * unit3_unit3);
   }
   return state;
}

} // namespace pyfluid_amrex
