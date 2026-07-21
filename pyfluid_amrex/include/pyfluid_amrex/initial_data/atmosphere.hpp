#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>

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

inline EOSThermodynamicState recovery_atmosphere_state(
   int eos_method,
   EOSParams const& eos
)
{
   if (eos_method == EOSPolytrope) {
      return eos_project_to_floors<EOSPolytrope>(
         eos.rho_floor, eos.pressure_floor, eos
      );
   }
   if (eos_method == EOSHybridPolytrope) {
      return eos_project_to_floors<EOSHybridPolytrope>(
         eos.rho_floor, eos.pressure_floor, eos
      );
   }
   return eos_project_to_floors<EOSIdealGas>(
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
      const amrex::Real atmosphere_gamma = EOSMethod == EOSIdealGas
         ? atmosphere.eos.gamma
         : atmosphere.eos.segment_gamma[0];
      pressure *= std::pow(
         density / atmosphere.density, atmosphere_gamma
      );
   }
   return eos_project_to_floors<EOSMethod>(density, pressure, atmosphere.eos);
}

} // namespace pyfluid_amrex
