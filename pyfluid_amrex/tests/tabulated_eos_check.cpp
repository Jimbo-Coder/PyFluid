#include <AMReX.H>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/eos/tabulated.hpp"
#include "pyfluid_amrex/eos/tabulated_barotrope.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"

namespace {

bool close_relative(amrex::Real left, amrex::Real right, amrex::Real tolerance)
{
   return std::abs(left - right) <= tolerance * std::max({
      amrex::Real{1.0e-300}, std::abs(left), std::abs(right)
   });
}

amrex::Real relative_error(amrex::Real left, amrex::Real right)
{
   return std::abs(left - right) / std::max({
      amrex::Real{1.0e-300}, std::abs(left), std::abs(right)
   });
}

bool check_table(std::string const& directory)
{
   const auto storage = pyfluid_amrex::eos::load_compose_cold_table(directory);
   const pyfluid_amrex::TabulatedEOSView table = storage->view();
   if (table.size < 2 || storage->source_points() < table.size
       || !(storage->baryon_mass_mev() > 0.0)
       || storage->backend() != "compose_cold_barotrope_v1"
       || storage->fingerprint().size() != 16) {
      std::cerr << "invalid table metadata for " << directory << '\n';
      return false;
   }

   pyfluid_amrex::EOSParams eos{};
   eos.method = pyfluid_amrex::EOSCompose;
   eos.thermal_gamma = 1.8;
   eos.rho_floor = 1.0e-18;
   eos.pressure_floor = 1.0e-24;
   eos.table = table;

   const amrex::Real minimum_density = std::exp(table.log_density[0]);
   const amrex::Real maximum_density = std::exp(table.log_density[table.size - 1]);
   for (int sample = -1; sample <= 18; ++sample) {
      const amrex::Real fraction = static_cast<amrex::Real>(
         std::max(sample, 0)
      ) / 16.0;
      const amrex::Real log_density = table.log_density[0]
         + fraction * (table.log_density[table.size - 1] - table.log_density[0]);
      const amrex::Real density = sample < 0
         ? 0.5 * minimum_density
         : std::exp(log_density);
      const pyfluid_amrex::TabulatedColdState cold =
         pyfluid_amrex::tabulated_cold_from_density(density, table);
      if (!(cold.density > 0.0) || !(cold.pressure > 0.0)
          || !(cold.energy_density >= cold.density)
          || !(cold.specific_enthalpy > 1.0)
          || !(cold.sound_speed_squared >= 0.0)
          || !(cold.sound_speed_squared < 1.0)) {
         std::cerr << "invalid cold state for " << directory
                   << " sample=" << sample << '\n';
         return false;
      }
      const pyfluid_amrex::TabulatedColdState inverse =
         pyfluid_amrex::tabulated_cold_from_log_enthalpy(
            pyfluid_amrex::tabulated_log_enthalpy_from_density(density, table),
            table
         );
      if (!close_relative(
             inverse.density,
             density,
             sample <= 0 ? 2.0e-8 : 2.0e-12
          )
          || !close_relative(
             inverse.pressure,
             cold.pressure,
             sample <= 0 ? 2.0e-8 : 5.0e-12
          )) {
         std::cerr << "enthalpy inversion mismatch for " << directory
                   << " sample=" << sample
                   << " rho=" << density
                   << " inverse_rho=" << inverse.density
                   << " pressure=" << cold.pressure
                   << " inverse_pressure=" << inverse.pressure << '\n';
         return false;
      }

      const amrex::Real cold_internal = cold.energy_density - density;
      const amrex::Real thermal_internal = 0.03 * density;
      const amrex::Real pressure =
         pyfluid_amrex::eos_pressure_from_density_energy<
            pyfluid_amrex::EOSRuntime
         >(density, cold_internal + thermal_internal, eos);
      const amrex::Real recovered_internal =
         pyfluid_amrex::eos_internal_energy_density_from_pressure<
            pyfluid_amrex::EOSRuntime
         >(density, pressure, eos);
      if (!close_relative(
             recovered_internal, cold_internal + thermal_internal, 2.0e-12
          )) {
         std::cerr << "cold-plus-thermal inversion mismatch for "
                   << directory << " sample=" << sample << '\n';
         return false;
      }
   }

   const amrex::Real density = std::sqrt(minimum_density * maximum_density);
   const pyfluid_amrex::TabulatedColdState cold =
      pyfluid_amrex::tabulated_cold_from_density(density, table);
   const amrex::Real eps = cold.energy_density / density - 1.0 + 0.02;
   const amrex::Real pressure = pyfluid_amrex::relativistic_pressure<
      pyfluid_amrex::EOSRuntime
   >(density, eps, eos);
   const pyfluid_amrex::RelativisticCharacteristicState characteristic =
      pyfluid_amrex::relativistic_characteristic_state<
         pyfluid_amrex::EOSRuntime
      >(density, eps, eos);
   if (!close_relative(
          characteristic.specific_enthalpy,
          pyfluid_amrex::relativistic_specific_enthalpy<
             pyfluid_amrex::EOSRuntime
          >(density, eps, eos),
          5.0e-14
       )
       || !close_relative(
          characteristic.sound_speed_squared,
          pyfluid_amrex::relativistic_sound_speed_squared<
             pyfluid_amrex::EOSRuntime
          >(density, eps, eos),
          5.0e-14
       )) {
      std::cerr << "CompOSE characteristic closure mismatch for "
                << directory << '\n';
      return false;
   }
   const pyfluid_amrex::RHDPrimitive primitive{
      density, eps, pressure, 0.23, -0.07, 0.03
   };
   const pyfluid_amrex::SpacetimeMetric metric =
      pyfluid_amrex::minkowski_spacetime_metric();
   const pyfluid_amrex::RHDConserved conserved =
      pyfluid_amrex::generated::rhd_primitive_to_conserved(primitive, metric);
   const pyfluid_amrex::RHDRecoveryResult recovered =
      pyfluid_amrex::recover_rhd_runtime(
         pyfluid_amrex::EOSCompose, conserved, metric, eos
      );
   if (!recovered.attempt.succeeded()
       || !close_relative(recovered.primitive.rho, primitive.rho, 5.0e-9)
       || !close_relative(recovered.primitive.pressure, primitive.pressure, 5.0e-9)) {
      std::cerr << "RHD CompOSE C2P round trip failed for " << directory
                << " status=" << static_cast<int>(recovered.attempt.status)
                << " residual=" << recovered.attempt.residual << '\n';
      return false;
   }

   const pyfluid_amrex::RMHDPrimitive magnetic_primitive{
      density,
      eps,
      pressure,
      0.17,
      -0.05,
      0.02,
      0.3,
      -0.2,
      0.1,
   };
   const pyfluid_amrex::RMHDConserved magnetic_conserved =
      pyfluid_amrex::generated::rmhd_primitive_to_conserved(
         magnetic_primitive, metric
      );
   const pyfluid_amrex::RMHDRecoveryResult magnetic_recovered =
      pyfluid_amrex::recover_srmhd<true>(
         magnetic_conserved, metric, eos
      );
   if (!magnetic_recovered.succeeded()) {
      std::cerr << "RMHD CompOSE C2P recovery failed for " << directory
                << " status="
                << static_cast<int>(magnetic_recovered.attempt.status)
                << " residual=" << magnetic_recovered.attempt.residual << '\n';
      return false;
   }
   const pyfluid_amrex::RMHDConserved magnetic_round_trip =
      pyfluid_amrex::generated::rmhd_primitive_to_conserved(
         magnetic_recovered.primitive, metric
      );
   const amrex::Real magnetic_primitive_error = std::max({
      relative_error(magnetic_recovered.primitive.rho, magnetic_primitive.rho),
      relative_error(magnetic_recovered.primitive.eps, magnetic_primitive.eps),
      relative_error(
         magnetic_recovered.primitive.pressure, magnetic_primitive.pressure
      ),
      relative_error(magnetic_recovered.primitive.v1, magnetic_primitive.v1),
      relative_error(magnetic_recovered.primitive.v2, magnetic_primitive.v2),
      relative_error(magnetic_recovered.primitive.v3, magnetic_primitive.v3),
      relative_error(magnetic_recovered.primitive.b1, magnetic_primitive.b1),
      relative_error(magnetic_recovered.primitive.b2, magnetic_primitive.b2),
      relative_error(magnetic_recovered.primitive.b3, magnetic_primitive.b3),
   });
   const amrex::Real magnetic_conserved_error = std::max({
      relative_error(magnetic_round_trip.dens, magnetic_conserved.dens),
      relative_error(magnetic_round_trip.mom1, magnetic_conserved.mom1),
      relative_error(magnetic_round_trip.mom2, magnetic_conserved.mom2),
      relative_error(magnetic_round_trip.mom3, magnetic_conserved.mom3),
      relative_error(magnetic_round_trip.tau, magnetic_conserved.tau),
      relative_error(magnetic_round_trip.b1, magnetic_conserved.b1),
      relative_error(magnetic_round_trip.b2, magnetic_conserved.b2),
      relative_error(magnetic_round_trip.b3, magnetic_conserved.b3),
   });
   if (magnetic_primitive_error > 1.0e-5
       || magnetic_conserved_error > 5.0e-9) {
      std::cerr << "RMHD CompOSE C2P round trip failed for " << directory
                << " status="
                << static_cast<int>(magnetic_recovered.attempt.status)
                << " residual=" << magnetic_recovered.attempt.residual
                << " primitive_error=" << magnetic_primitive_error
                << " conserved_error=" << magnetic_conserved_error << '\n';
      return false;
   }

   std::cout << "PASS CompOSE cold table " << directory
             << " points=" << table.size << '/' << storage->source_points()
             << " baryon_mass_mev=" << storage->baryon_mass_mev()
             << " causal_truncation="
             << (storage->was_truncated_for_causality() ? 1 : 0)
             << " rmhd_primitive_error=" << magnetic_primitive_error
             << " rmhd_conserved_error=" << magnetic_conserved_error
             << '\n';
   return true;
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   bool success = argc >= 2;
   for (int argument = 1; argument < argc; ++argument) {
      success = check_table(argv[argument]) && success;
   }
   if (argc < 2) {
      std::cerr << "usage: pyfluid_amrex_tabulated_eos_check <table-dir>...\n";
   }
   amrex::Finalize();
   return success ? 0 : 1;
}
