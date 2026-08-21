#include "pyfluid_amrex/eos/config.hpp"

#include <AMReX.H>

#include <exception>
#include <string>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/presets.hpp"
#include "pyfluid_amrex/eos/tabulated.hpp"
#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex {

EOSParams eos_params(RunConfig const& cfg)
{
   if (cfg.eos == "compose") {
      if (!cfg.tabulated_eos) {
         amrex::Abort("CompOSE EoS data must be prepared before use");
      }
      EOSParams parameters{};
      parameters.method = EOSCompose;
      parameters.gamma = cfg.gamma;
      parameters.thermal_gamma = cfg.thermal_gamma;
      parameters.rho_floor = cfg.rho_floor;
      parameters.pressure_floor = cfg.pressure_floor;
      parameters.table = cfg.tabulated_eos->view();
      return parameters;
   }
   const eos::PiecewisePolytropePreset preset =
      eos::piecewise_polytrope_preset(cfg.eos_preset);
   if (preset != eos::PiecewisePolytropePreset::custom) {
      EOSParams parameters = eos::make_read_piecewise_polytrope(
         preset,
         cfg.thermal_gamma,
         cfg.rho_floor,
         cfg.pressure_floor
      );
      parameters.method = eos_method_code(cfg);
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
   parameters.method = eos_method_code(cfg);
   parameters.gamma = cfg.gamma;
   return parameters;
}

EOSIdentity eos_identity(RunConfig const& cfg)
{
   if (cfg.eos != "compose") {
      return {cfg.eos, {}, {}, 0.0, 0, 0, false};
   }
   if (!cfg.tabulated_eos) {
      amrex::Abort("CompOSE EoS data must be prepared before use");
   }
   return {
      cfg.tabulated_eos->backend(),
      cfg.tabulated_eos->fingerprint(),
      cfg.tabulated_eos->source_directory(),
      cfg.tabulated_eos->baryon_mass_mev(),
      cfg.tabulated_eos->source_points(),
      cfg.tabulated_eos->retained_points(),
      cfg.tabulated_eos->was_truncated_for_causality(),
   };
}

void prepare_eos(RunConfig& cfg)
{
   if (cfg.eos != "compose") {
      cfg.tabulated_eos.reset();
      return;
   }
   if (cfg.eos_table.empty()) {
      return;
   }
   try {
      cfg.tabulated_eos = eos::load_compose_cold_table(cfg.eos_table);
      if (cfg.tabulated_eos->was_truncated_for_causality()) {
         amrex::Print()
            << "WARNING: CompOSE cold table retained "
            << cfg.tabulated_eos->retained_points() << "/"
            << cfg.tabulated_eos->source_points()
            << " causal points and uses the documented constant-sound-speed "
               "high-density extension\n";
      }
   } catch (std::exception const& error) {
      amrex::Abort(
         std::string("failed to load pyfluid.eos_table: ") + error.what()
      );
   }
}

} // namespace pyfluid_amrex
