#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <string>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/interpolation_weights.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct TovSample {
   amrex::Real isotropic_radius = 0.0_rt;
   amrex::Real rho = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real specific_internal_energy = 0.0_rt;
   amrex::Real lapse = 1.0_rt;
   amrex::Real conformal_factor = 1.0_rt;
};

// Metric derivatives come from the TOV equations, not differences of the
// adaptive output table. Quintic segments therefore share two derivatives.
struct TovInterpolationPoint {
   TovSample state;
   amrex::Real lapse_first = 0.0_rt;
   amrex::Real lapse_second = 0.0_rt;
   amrex::Real conformal_first = 0.0_rt;
   amrex::Real conformal_second = 0.0_rt;
};

struct TovProfilePoint {
   amrex::Real log_specific_enthalpy = 0.0_rt;
   amrex::Real areal_radius = 0.0_rt;
   amrex::Real isotropic_radius = 0.0_rt;
   amrex::Real rho = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real specific_internal_energy = 0.0_rt;
   amrex::Real energy_density = 0.0_rt;
   amrex::Real enclosed_gravitational_mass = 0.0_rt;
   amrex::Real enclosed_baryonic_mass = 0.0_rt;
   amrex::Real enclosed_proper_mass = 0.0_rt;
   amrex::Real lapse = 1.0_rt;
   amrex::Real conformal_factor = 1.0_rt;
   amrex::Real tidal_y = 2.0_rt;
};

struct TovObservables {
   amrex::Real central_density = 0.0_rt;
   amrex::Real central_pressure = 0.0_rt;
   amrex::Real central_energy_density = 0.0_rt;
   amrex::Real gravitational_mass = 0.0_rt;
   amrex::Real baryonic_mass = 0.0_rt;
   amrex::Real proper_mass = 0.0_rt;
   amrex::Real komar_mass = 0.0_rt;
   amrex::Real areal_radius = 0.0_rt;
   amrex::Real isotropic_radius = 0.0_rt;
   amrex::Real compactness = 0.0_rt;
   amrex::Real binding_energy = 0.0_rt;
   amrex::Real surface_redshift = 0.0_rt;
   amrex::Real injection_energy = 1.0_rt;
   amrex::Real love_number_k2 = 0.0_rt;
   amrex::Real tidal_deformability = 0.0_rt;
   bool tidal_observables_available = false;
};

struct TovSolveOptions {
   amrex::Real central_density = 1.28e-3_rt;
   amrex::Real absolute_tolerance = 1.0e-12_rt;
   amrex::Real relative_tolerance = 1.0e-10_rt;
   int max_steps = 100000;
};

class TovSolution {
public:
   TovSolution() = default;
   TovSolution(EOSParams const& eos, TovSolveOptions const& options);

   TovObservables const& observables() const noexcept { return observables_; }
   std::vector<TovProfilePoint> const& profile() const noexcept { return profile_; }
   bool empty() const noexcept { return profile_.empty(); }
   std::vector<TovInterpolationPoint> const& interpolation_points() const noexcept { return interpolation_points_; }
   TovSample sample_isotropic(amrex::Real radius) const noexcept;

private:
   TovObservables observables_;
   std::vector<TovProfilePoint> profile_;
   std::vector<TovInterpolationPoint> interpolation_points_;
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
   TovInterpolationPoint const* samples = nullptr;
   int sample_count = 0;
   amrex::Real gravitational_mass = 0.0_rt;
   amrex::Real surface_radius = 0.0_rt;
   amrex::Real injection_energy = 1.0_rt;

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
         const amrex::Real mass_over_radius = gravitational_mass / (2.0_rt * radius);
         const amrex::Real factor = 1.0_rt + mass_over_radius;
         return {
            radius,
            0.0_rt,
            0.0_rt,
            0.0_rt,
            (1.0_rt - mass_over_radius) / factor,
            factor,
         };
      }
      int lower = 0;
      int upper = sample_count - 1;
      while (upper - lower > 1) {
         const int middle = lower + (upper - lower) / 2;
         if (samples[middle].state.isotropic_radius <= radius) {
            lower = middle;
         } else {
            upper = middle;
         }
      }
      const auto& left_jet = samples[lower];
      const auto& right_jet = samples[upper];
      const TovSample& left = left_jet.state;
      const TovSample& right = right_jet.state;
      const amrex::Real width = right.isotropic_radius - left.isotropic_radius;
      const amrex::Real weight = width > 0.0_rt
         ? (radius - left.isotropic_radius) / width
         : 0.0_rt;
      return {
         radius,
         left.rho + weight * (right.rho - left.rho),
         left.pressure + weight * (right.pressure - left.pressure),
         left.specific_internal_energy + weight * (
            right.specific_internal_energy - left.specific_internal_energy
         ),
         quintic_hermite(weight, width, left.lapse, right.lapse,
            left_jet.lapse_first, right_jet.lapse_first,
            left_jet.lapse_second, right_jet.lapse_second),
         quintic_hermite(weight, width, left.conformal_factor, right.conformal_factor,
            left_jet.conformal_first, right_jet.conformal_first,
            left_jet.conformal_second, right_jet.conformal_second),
      };
   }
};

class TovInitialData {
public:
   TovInitialData() = default;
   explicit TovInitialData(RunConfig const& cfg);

   TovInitialDataView view() const noexcept;
   bool active() const noexcept { return !solution_.empty(); }
   TovSolution const& solution() const noexcept { return solution_; }

private:
   TovSolution solution_;
   amrex::Gpu::DeviceVector<TovInterpolationPoint> device_samples_;
};

} // namespace pyfluid_amrex
