#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <vector>

#include "pyfluid_amrex/config.hpp"

namespace pyfluid_amrex {

struct TovSample {
   amrex::Real isotropic_radius = 0.0;
   amrex::Real rho = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real specific_internal_energy = 0.0;
   amrex::Real lapse = 1.0;
   amrex::Real conformal_factor = 1.0;
};

struct TovInitialDataView {
   TovSample const* samples = nullptr;
   int sample_count = 0;
   amrex::Real gravitational_mass = 0.0;
   amrex::Real surface_radius = 0.0;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   bool active() const noexcept
   {
      return samples != nullptr && sample_count > 1;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   TovSample sample(amrex::Real radius) const noexcept
   {
      if (!active()) {
         return {};
      }
      if (radius >= surface_radius) {
         const amrex::Real mass_over_radius = gravitational_mass / (2.0 * radius);
         const amrex::Real factor = 1.0 + mass_over_radius;
         return {
            radius,
            0.0,
            0.0,
            0.0,
            (1.0 - mass_over_radius) / factor,
            factor,
         };
      }
      int lower = 0;
      int upper = sample_count - 1;
      while (upper - lower > 1) {
         const int middle = lower + (upper - lower) / 2;
         if (samples[middle].isotropic_radius <= radius) {
            lower = middle;
         } else {
            upper = middle;
         }
      }
      const TovSample& left = samples[lower];
      const TovSample& right = samples[upper];
      const amrex::Real width = right.isotropic_radius - left.isotropic_radius;
      const amrex::Real weight = width > 0.0
         ? (radius - left.isotropic_radius) / width
         : 0.0;
      return {
         radius,
         left.rho + weight * (right.rho - left.rho),
         left.pressure + weight * (right.pressure - left.pressure),
         left.specific_internal_energy + weight * (
            right.specific_internal_energy - left.specific_internal_energy
         ),
         left.lapse + weight * (right.lapse - left.lapse),
         left.conformal_factor + weight * (
            right.conformal_factor - left.conformal_factor
         ),
      };
   }
};

class TovInitialData {
public:
   TovInitialData() = default;
   explicit TovInitialData(RunConfig const& cfg);

   TovInitialDataView view() const noexcept;
   bool active() const noexcept { return !host_samples_.empty(); }

private:
   std::vector<TovSample> host_samples_;
   amrex::Gpu::DeviceVector<TovSample> device_samples_;
   amrex::Real gravitational_mass_ = 0.0;
   amrex::Real surface_radius_ = 0.0;
};

} // namespace pyfluid_amrex
