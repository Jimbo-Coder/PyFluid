#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <string>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

struct TovSample {
   amrex::Real isotropic_radius = 0.0;
   amrex::Real rho = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real specific_internal_energy = 0.0;
   amrex::Real lapse = 1.0;
   amrex::Real conformal_factor = 1.0;
};

struct TovProfilePoint {
   amrex::Real log_specific_enthalpy = 0.0;
   amrex::Real areal_radius = 0.0;
   amrex::Real isotropic_radius = 0.0;
   amrex::Real rho = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real specific_internal_energy = 0.0;
   amrex::Real energy_density = 0.0;
   amrex::Real enclosed_gravitational_mass = 0.0;
   amrex::Real enclosed_baryonic_mass = 0.0;
   amrex::Real enclosed_proper_mass = 0.0;
   amrex::Real lapse = 1.0;
   amrex::Real conformal_factor = 1.0;
   amrex::Real tidal_y = 2.0;
};

struct TovObservables {
   amrex::Real central_density = 0.0;
   amrex::Real central_pressure = 0.0;
   amrex::Real central_energy_density = 0.0;
   amrex::Real gravitational_mass = 0.0;
   amrex::Real baryonic_mass = 0.0;
   amrex::Real proper_mass = 0.0;
   amrex::Real komar_mass = 0.0;
   amrex::Real areal_radius = 0.0;
   amrex::Real isotropic_radius = 0.0;
   amrex::Real compactness = 0.0;
   amrex::Real binding_energy = 0.0;
   amrex::Real surface_redshift = 0.0;
   amrex::Real injection_energy = 1.0;
   amrex::Real love_number_k2 = 0.0;
   amrex::Real tidal_deformability = 0.0;
   bool tidal_observables_available = false;
};

struct TovSolveOptions {
   amrex::Real central_density = 1.28e-3;
   amrex::Real absolute_tolerance = 1.0e-12;
   amrex::Real relative_tolerance = 1.0e-10;
   int max_steps = 100000;
};

class TovSolution {
public:
   TovSolution() = default;
   TovSolution(EOSParams const& eos, TovSolveOptions const& options);

   TovObservables const& observables() const noexcept { return observables_; }
   std::vector<TovProfilePoint> const& profile() const noexcept { return profile_; }
   bool empty() const noexcept { return profile_.empty(); }
   TovSample sample_isotropic(amrex::Real radius) const noexcept;

private:
   TovObservables observables_;
   std::vector<TovProfilePoint> profile_;
};

std::vector<TovSolution> solve_tov_sequence(
   EOSParams const& eos,
   std::vector<amrex::Real> const& central_densities,
   TovSolveOptions options = {}
);

void write_tov_sequence(
   std::string const& path,
   std::string const& eos_label,
   std::vector<TovSolution> const& sequence
);

struct TovInitialDataView {
   TovSample const* samples = nullptr;
   int sample_count = 0;
   amrex::Real gravitational_mass = 0.0;
   amrex::Real surface_radius = 0.0;
   amrex::Real injection_energy = 1.0;

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
   TovSolution const& solution() const noexcept { return solution_; }

private:
   TovSolution solution_;
   std::vector<TovSample> host_samples_;
   amrex::Gpu::DeviceVector<TovSample> device_samples_;
   amrex::Real gravitational_mass_ = 0.0;
   amrex::Real surface_radius_ = 0.0;
   amrex::Real injection_energy_ = 1.0;
};

} // namespace pyfluid_amrex
