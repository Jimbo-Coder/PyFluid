#pragma once

#include <AMReX_GpuQualifiers.H>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

struct RunConfig;

enum class RelativisticExteriorTreatment : int {
   Atmosphere,
   Vacuum,
};

// Runtime policy and its complete device payload. Flux, recovery, and AMR
// kernels consume only the thermodynamic base, so selecting vacuum does not
// create another template product or duplicate the relativistic operators.
struct RelativisticExteriorState : EOSThermodynamicState {
   RelativisticExteriorTreatment treatment =
      RelativisticExteriorTreatment::Atmosphere;

   AMREX_GPU_HOST_DEVICE constexpr RelativisticExteriorState() noexcept
      : EOSThermodynamicState{0.0, 0.0, 0.0}
   {}

   AMREX_GPU_HOST_DEVICE constexpr RelativisticExteriorState(
      EOSThermodynamicState thermodynamic,
      RelativisticExteriorTreatment selected_treatment
   ) noexcept
      : EOSThermodynamicState{thermodynamic},
        treatment(selected_treatment)
   {}

   AMREX_GPU_HOST_DEVICE constexpr bool is_vacuum() const noexcept
   {
      return treatment == RelativisticExteriorTreatment::Vacuum;
   }
};

RelativisticExteriorState relativistic_exterior_state(RunConfig const& cfg);

} // namespace pyfluid_amrex
