#pragma once

#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <limits>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/relativistic_exterior.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RelativisticInitialDataView;
struct RunConfig;

enum class InitialAtmosphereProfile : int {
   Flat,
   PowerLaw,
};

struct InitialAtmosphereParameters {
   InitialAtmosphereProfile profile = InitialAtmosphereProfile::Flat;
   amrex::Real density = 1.0e-12_rt;
   amrex::Real pressure = 1.0e-12_rt;
   amrex::Real transition_radius = 1.0_rt;
   amrex::Real power = 4.0_rt;
   EOSParams eos{};
   RelativisticExteriorTreatment treatment =
      RelativisticExteriorTreatment::Atmosphere;
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
      1024.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   return density > atmosphere_density * (1.0_rt + tolerance);
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EOSThermodynamicState initial_atmosphere_state(
   InitialAtmosphereParameters const& atmosphere,
   InitialDataSamplePoint const& sample
) noexcept
{
   if (atmosphere.treatment == RelativisticExteriorTreatment::Vacuum) {
      return {0.0_rt, 0.0_rt, 0.0_rt};
   }
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
