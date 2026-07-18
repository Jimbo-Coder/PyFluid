#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
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
   RelativisticInitialDataSample sample(PatchCoordinate const& physical) const noexcept
   {
      RelativisticInitialDataSample output;
      if (tov.active()) {
         const amrex::Real radius = std::sqrt(
            physical.q1 * physical.q1
            + physical.q2 * physical.q2
            + physical.q3 * physical.q3
         );
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
         return output;
      }
      if (fixed_spacetime_active) {
         const FixedSpacetimeGeometry geometry = fixed_spacetime_geometry(
            fixed_spacetime,
            physical.q1,
            physical.q2,
            physical.q3,
            0.0
         );
         const SpacetimeMetric& metric = geometry.metric;
         const amrex::Real determinant = metric.sqrt_det_gamma
            * metric.sqrt_det_gamma;
         const amrex::Real chi = 1.0 / std::cbrt(determinant);
         output.has_spacetime = true;
         output.spacetime.chi = chi;
         output.spacetime.gamma_tilde11 = chi * metric.gamma11;
         output.spacetime.gamma_tilde12 = chi * metric.gamma12;
         output.spacetime.gamma_tilde13 = chi * metric.gamma13;
         output.spacetime.gamma_tilde22 = chi * metric.gamma22;
         output.spacetime.gamma_tilde23 = chi * metric.gamma23;
         output.spacetime.gamma_tilde33 = chi * metric.gamma33;
         output.spacetime.lapse = metric.lapse;
         output.spacetime.shift1 = metric.shift1;
         output.spacetime.shift2 = metric.shift2;
         output.spacetime.shift3 = metric.shift3;
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
