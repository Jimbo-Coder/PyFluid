#include "pyfluid_amrex/eos/config.hpp"

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/presets.hpp"

namespace pyfluid_amrex {

EOSParams eos_params(RunConfig const& cfg)
{
   const eos::PiecewisePolytropePreset preset =
      eos::piecewise_polytrope_preset(cfg.eos_preset);
   if (preset != eos::PiecewisePolytropePreset::custom) {
      EOSParams parameters = eos::make_read_piecewise_polytrope(
         preset,
         cfg.thermal_gamma,
         cfg.rho_floor,
         cfg.pressure_floor
      );
      parameters.gamma = cfg.gamma;
      return parameters;
   }
   EOSParams parameters = eos::make_piecewise_polytrope(
      cfg.polytropic_k,
      cfg.polytropic_gamma,
      cfg.polytropic_density_breaks,
      cfg.polytropic_segment_gammas,
      cfg.thermal_gamma,
      cfg.rho_floor,
      cfg.pressure_floor
   );
   parameters.gamma = cfg.gamma;
   return parameters;
}

} // namespace pyfluid_amrex
