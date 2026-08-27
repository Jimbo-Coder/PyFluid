#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"
#include "pyfluid_amrex/initial_data/static_trumpet.hpp"
#include "pyfluid_amrex/initial_data/tov.hpp"
#include "pyfluid_amrex/patch_maps.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

struct InitialDataFieldSelection {
   bool magnetic_from_id = true;
   bool spacetime_from_id = true;
   bool flat_spacetime = false;
   bool lapse_from_id = true;
   bool unit_lapse = false;
   bool shift_from_id = true;
   bool zero_shift = false;
};

InitialDataFieldSelection initial_data_field_selection(RunConfig const& cfg);

struct RelativisticInitialDataSample {
   bool has_fluid = false;
   bool has_spacetime = false;
   RHDPrimitive fluid{};
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
   Z4cState spacetime{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Z4cState z4c_state_from_spacetime_geometry(
   FixedSpacetimeGeometry const& geometry
) noexcept
{
   SpacetimeMetric const& metric = geometry.metric;
   SpacetimeMetricDerivatives const& derivatives = geometry.derivatives;
   const amrex::Real determinant = metric.sqrt_det_gamma
      * metric.sqrt_det_gamma;
   const amrex::Real chi = 1.0 / std::cbrt(determinant);
   const amrex::Real trace_extrinsic =
      metric.inv_gamma11 * derivatives.extrinsic11
      + 2.0 * metric.inv_gamma12 * derivatives.extrinsic12
      + 2.0 * metric.inv_gamma13 * derivatives.extrinsic13
      + metric.inv_gamma22 * derivatives.extrinsic22
      + 2.0 * metric.inv_gamma23 * derivatives.extrinsic23
      + metric.inv_gamma33 * derivatives.extrinsic33;
   Z4cState state;
   state.chi = chi;
   state.gamma_tilde11 = chi * metric.gamma11;
   state.gamma_tilde12 = chi * metric.gamma12;
   state.gamma_tilde13 = chi * metric.gamma13;
   state.gamma_tilde22 = chi * metric.gamma22;
   state.gamma_tilde23 = chi * metric.gamma23;
   state.gamma_tilde33 = chi * metric.gamma33;
   state.k_hat = trace_extrinsic;
   state.a_tilde11 = chi * (
      derivatives.extrinsic11 - metric.gamma11 * trace_extrinsic / 3.0
   );
   state.a_tilde12 = chi * (
      derivatives.extrinsic12 - metric.gamma12 * trace_extrinsic / 3.0
   );
   state.a_tilde13 = chi * (
      derivatives.extrinsic13 - metric.gamma13 * trace_extrinsic / 3.0
   );
   state.a_tilde22 = chi * (
      derivatives.extrinsic22 - metric.gamma22 * trace_extrinsic / 3.0
   );
   state.a_tilde23 = chi * (
      derivatives.extrinsic23 - metric.gamma23 * trace_extrinsic / 3.0
   );
   state.a_tilde33 = chi * (
      derivatives.extrinsic33 - metric.gamma33 * trace_extrinsic / 3.0
   );
   state.lapse = metric.lapse;
   state.shift1 = metric.shift1;
   state.shift2 = metric.shift2;
   state.shift3 = metric.shift3;
   return state;
}

struct RelativisticInitialDataView {
   TovInitialDataView tov{};
   FixedSpacetimeParameters fixed_spacetime{};
   bool fixed_spacetime_active = false;
   InitialDataFieldSelection fields{};

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool active() const noexcept
   {
      return has_fluid() || has_spacetime();
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool has_fluid() const noexcept
   {
      return tov.active();
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool has_spacetime() const noexcept
   {
      return tov.active() || fixed_spacetime_active;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   RelativisticInitialDataSample sample_tov_radius(
      amrex::Real radius
   ) const noexcept
   {
      RelativisticInitialDataSample output;
      if (tov.active()) {
         const TovSample point = tov.sample(radius);
         output.has_fluid = true;
         output.has_spacetime = true;
         output.fluid = {
            point.rho,
            point.specific_internal_energy,
            point.pressure,
            0.0,
            0.0,
            0.0,
         };
         output.spacetime.chi = 1.0 / (
            point.conformal_factor * point.conformal_factor
            * point.conformal_factor * point.conformal_factor
         );
         output.spacetime.lapse = point.lapse;
      }
      return output;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   RelativisticInitialDataSample sample(PatchCoordinate const& physical) const noexcept
   {
      if (tov.active()) {
         return sample_tov_radius(cartesian_radius(physical));
      }
      RelativisticInitialDataSample output;
      if (fixed_spacetime_active) {
         if (fixed_spacetime.metric == FixedSpacetimeMetric::StaticTrumpet) {
            output.has_spacetime = true;
            output.spacetime = static_trumpet_z4c_initial_data_state(
               fixed_spacetime,
               physical.q1,
               physical.q2,
               physical.q3
            );
            return output;
         }
         const FixedSpacetimeGeometry geometry = initial_data_spacetime_geometry(
            fixed_spacetime,
            physical.q1,
            physical.q2,
            physical.q3,
            0.0
         );
         output.has_spacetime = true;
         output.spacetime = z4c_state_from_spacetime_geometry(geometry);
      }
      return output;
   }
};

class RelativisticInitialDataProvider {
public:
   explicit RelativisticInitialDataProvider(RunConfig const& cfg);

   RelativisticInitialDataView view() const noexcept;
   bool active() const noexcept;

private:
   TovInitialData tov_;
   FixedSpacetimeParameters fixed_spacetime_{};
   bool fixed_spacetime_active_ = false;
   InitialDataFieldSelection fields_{};
};

} // namespace pyfluid_amrex
