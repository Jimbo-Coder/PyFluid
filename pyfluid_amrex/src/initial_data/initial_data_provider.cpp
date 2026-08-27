#include "pyfluid_amrex/initial_data/provider.hpp"

#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"

namespace pyfluid_amrex {

InitialAtmosphereParameters initial_atmosphere_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
)
{
   InitialAtmosphereParameters atmosphere;
   atmosphere.eos = eos_params(cfg);
   atmosphere.profile = cfg.initial_data_atmosphere == "power_law"
      ? InitialAtmosphereProfile::PowerLaw
      : InitialAtmosphereProfile::Flat;
   atmosphere.density = cfg.initial_data_atmosphere_density > 0.0
      ? cfg.initial_data_atmosphere_density
      : atmosphere.eos.rho_floor;
   atmosphere.pressure = cfg.initial_data_atmosphere_pressure > 0.0
      ? cfg.initial_data_atmosphere_pressure
      : atmosphere.eos.pressure_floor;
   atmosphere.transition_radius =
      cfg.initial_data_atmosphere_transition_radius > 0.0
      ? cfg.initial_data_atmosphere_transition_radius
      : 4.0 * provider.tov.surface_radius;
   atmosphere.power = cfg.initial_data_atmosphere_power;
   if (atmosphere.profile == InitialAtmosphereProfile::PowerLaw
       && !(atmosphere.transition_radius > 0.0)) {
      amrex::Abort(
         "power-law initial atmosphere requires an explicit transition radius "
         "unless a TOV provider supplies its surface radius"
      );
   }
   return atmosphere;
}

InitialDataFieldSelection initial_data_field_selection(RunConfig const& cfg)
{
   return {
      cfg.initial_data_magnetic == "id",
      cfg.initial_data_spacetime == "id",
      cfg.initial_data_spacetime == "flat",
      cfg.initial_data_lapse == "id",
      cfg.initial_data_lapse == "one",
      cfg.initial_data_shift == "id",
      cfg.initial_data_shift == "zero",
   };
}

RelativisticInitialDataProvider::RelativisticInitialDataProvider(RunConfig const& cfg)
   : tov_(cfg),
     fixed_spacetime_(fixed_spacetime_parameters(cfg)),
     fixed_spacetime_active_(
        !tov_.active()
        && (cfg.spacetime_evolution == "fixed"
            || cfg.spacetime_initial_data == "kerr_schild"
            || cfg.spacetime_initial_data == "static_trumpet")
     ),
     fields_(initial_data_field_selection(cfg))
{}

RelativisticInitialDataView RelativisticInitialDataProvider::view() const noexcept
{
   return {tov_.view(), fixed_spacetime_, fixed_spacetime_active_, fields_};
}

bool RelativisticInitialDataProvider::active() const noexcept
{
   return tov_.active() || fixed_spacetime_active_;
}

} // namespace pyfluid_amrex
