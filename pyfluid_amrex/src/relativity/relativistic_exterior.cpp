#include "pyfluid_amrex/relativistic_exterior.hpp"

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"

namespace pyfluid_amrex {

RelativisticExteriorState relativistic_exterior_state(RunConfig const& cfg)
{
   if (cfg.relativistic_exterior == "vacuum") {
      return {
         EOSThermodynamicState{0.0, 0.0, 0.0},
         RelativisticExteriorTreatment::Vacuum,
      };
   }
   const EOSParams eos = eos_params(cfg);
   return {
      eos_project_to_floors<EOSRuntime>(
         eos.rho_floor, eos.pressure_floor, eos
      ),
      RelativisticExteriorTreatment::Atmosphere,
   };
}

} // namespace pyfluid_amrex
