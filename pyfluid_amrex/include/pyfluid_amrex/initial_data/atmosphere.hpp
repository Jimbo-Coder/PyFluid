#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"

namespace pyfluid_amrex {

struct RelativisticInitialDataView;
struct RunConfig;

enum class InitialAtmosphereProfile : int {
   Flat,
   PowerLaw,
};

struct InitialAtmosphereParameters {
   InitialAtmosphereProfile profile = InitialAtmosphereProfile::Flat;
   amrex::Real density = 1.0e-12;
   amrex::Real pressure = 1.0e-12;
   amrex::Real transition_radius = 1.0;
   amrex::Real power = 4.0;
   EOSParams eos{};
};

InitialAtmosphereParameters initial_atmosphere_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
);

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool initial_matter_exceeds_atmosphere(
   amrex::Real density,
   amrex::Real atmosphere_density
) noexcept
{
   constexpr amrex::Real tolerance =
      1024.0 * std::numeric_limits<amrex::Real>::epsilon();
   return density > atmosphere_density * (1.0 + tolerance);
}

inline EOSThermodynamicState recovery_atmosphere_state(
   int eos_method,
   EOSParams const& eos
)
{
   static_cast<void>(eos_method);
   return eos_project_to_floors<EOSRuntime>(
      eos.rho_floor, eos.pressure_floor, eos
   );
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EOSThermodynamicState initial_atmosphere_state(
   InitialAtmosphereParameters const& atmosphere,
   InitialDataSamplePoint const& sample
) noexcept
{
   amrex::Real density = atmosphere.density;
   if (atmosphere.profile == InitialAtmosphereProfile::PowerLaw) {
      const amrex::Real radius = cartesian_radius(sample.physical);
      if (radius > atmosphere.transition_radius) {
         density *= std::pow(
            atmosphere.transition_radius / radius, atmosphere.power
         );
      }
   }

   if (density <= atmosphere.eos.rho_floor) {
      return eos_project_to_floors<EOSMethod>(
         atmosphere.eos.rho_floor,
         atmosphere.eos.pressure_floor,
         atmosphere.eos
      );
   }

   amrex::Real pressure = atmosphere.pressure;
   if (atmosphere.profile == InitialAtmosphereProfile::PowerLaw) {
      amrex::Real atmosphere_gamma = atmosphere.eos.gamma;
      if constexpr (EOSMethod == EOSRuntime) {
         if (atmosphere.eos.method == EOSCompose) {
            atmosphere_gamma = atmosphere.eos.table.low_density_gamma;
         } else if (atmosphere.eos.method != EOSIdealGas) {
            atmosphere_gamma = atmosphere.eos.segment_gamma[0];
         }
      } else if constexpr (EOSMethod == EOSCompose) {
         atmosphere_gamma = atmosphere.eos.table.low_density_gamma;
      } else if constexpr (EOSMethod != EOSIdealGas) {
         atmosphere_gamma = atmosphere.eos.segment_gamma[0];
      }
      pressure *= std::pow(
         density / atmosphere.density, atmosphere_gamma
      );
   }
   return eos_project_to_floors<EOSMethod>(density, pressure, atmosphere.eos);
}

} // namespace pyfluid_amrex
