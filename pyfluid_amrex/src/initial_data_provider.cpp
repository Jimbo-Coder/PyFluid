#include "pyfluid_amrex/initial_data/provider.hpp"

namespace pyfluid_amrex {

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
        cfg.spacetime_evolution == "fixed" && !tov_.active()
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
