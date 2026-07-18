#include "pyfluid_amrex/eos/presets.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pyfluid_amrex::eos {
namespace {

// Read et al. (2009), Tables II and III. Published densities and pressures
// are converted once into code units with G = c = M_sun = 1.
constexpr amrex::Real gravitational_constant_cgs = 6.67430e-8;
constexpr amrex::Real speed_of_light_cgs = 2.99792458e10;
constexpr amrex::Real solar_mass_parameter_cgs = 1.3271244e26;
constexpr amrex::Real solar_mass_length_cgs =
   solar_mass_parameter_cgs / (speed_of_light_cgs * speed_of_light_cgs);
constexpr amrex::Real density_cgs_to_code =
   gravitational_constant_cgs
   * solar_mass_length_cgs * solar_mass_length_cgs
   / (speed_of_light_cgs * speed_of_light_cgs);
constexpr amrex::Real pressure_cgs_to_code =
   gravitational_constant_cgs
   * solar_mass_length_cgs * solar_mass_length_cgs
   / (
      speed_of_light_cgs * speed_of_light_cgs
      * speed_of_light_cgs * speed_of_light_cgs
   );

constexpr std::array<ReadPiecewisePolytropeFit, 4> read_fits{{
   {"sly", 34.384, {{3.005, 2.988, 2.851}}},
   {"apr4", 34.269, {{2.830, 3.445, 3.348}}},
   {"h4", 34.669, {{2.909, 2.246, 2.144}}},
   {"ms1", 34.858, {{3.224, 3.033, 1.325}}},
}};

constexpr amrex::Real crust_initial_k_over_c2_cgs = 6.80110e-9;
constexpr amrex::Real crust_initial_k_cgs =
   crust_initial_k_over_c2_cgs
   * speed_of_light_cgs * speed_of_light_cgs;
constexpr std::array<amrex::Real, 4> crust_gamma{{
   1.58425,
   1.28733,
   0.62223,
   1.35692,
}};
constexpr std::array<amrex::Real, 3> crust_density_break_cgs{{
   2.44034e7,
   3.78358e11,
   2.62780e12,
}};
constexpr amrex::Real core_density1_cgs = 5.0118723362727148e14;
constexpr amrex::Real core_density2_cgs = 1.0e15;

amrex::Real density_to_code(amrex::Real density_cgs)
{
   return density_cgs * density_cgs_to_code;
}

amrex::Real pressure_to_code(amrex::Real pressure_cgs)
{
   return pressure_cgs * pressure_cgs_to_code;
}

} // namespace

PiecewisePolytropePreset piecewise_polytrope_preset(std::string_view name)
{
   for (std::size_t index = 0; index < PiecewisePolytropePresetNames.size(); ++index) {
      if (name == PiecewisePolytropePresetNames[index]) {
         return static_cast<PiecewisePolytropePreset>(index);
      }
   }
   throw std::invalid_argument(
      "unsupported piecewise-polytrope preset: " + std::string(name)
   );
}

ReadPiecewisePolytropeFit read_piecewise_polytrope_fit(
   PiecewisePolytropePreset preset
)
{
   if (preset == PiecewisePolytropePreset::custom) {
      throw std::invalid_argument("custom piecewise polytropes do not have a Read fit");
   }
   return read_fits[static_cast<std::size_t>(preset) - 1];
}

EOSParams make_piecewise_polytrope(
   amrex::Real initial_k,
   amrex::Real initial_gamma,
   std::vector<amrex::Real> const& density_breaks,
   std::vector<amrex::Real> const& following_gammas,
   amrex::Real thermal_gamma,
   amrex::Real rho_floor,
   amrex::Real pressure_floor
)
{
   if (!(initial_k > 0.0) || !(initial_gamma > 0.0)
       || initial_gamma == 1.0) {
      throw std::invalid_argument(
         "piecewise polytropes require positive K and positive, non-unit gamma"
      );
   }
   if (density_breaks.size() != following_gammas.size()
       || density_breaks.size() >= static_cast<std::size_t>(MaxPolytropeSegments)) {
      throw std::invalid_argument(
         "piecewise-polytrope breaks and following gammas must define at most eight segments"
      );
   }

   EOSParams parameters{};
   parameters.thermal_gamma = thermal_gamma;
   parameters.rho_floor = rho_floor;
   parameters.pressure_floor = pressure_floor;
   parameters.polytrope_segments = 1 + static_cast<int>(density_breaks.size());
   parameters.segment_k[0] = initial_k;
   parameters.segment_gamma[0] = initial_gamma;
   for (int segment = 1; segment < parameters.polytrope_segments; ++segment) {
      const amrex::Real density_break = density_breaks[segment - 1];
      const amrex::Real previous_gamma = parameters.segment_gamma[segment - 1];
      const amrex::Real gamma = following_gammas[segment - 1];
      const amrex::Real previous_k = parameters.segment_k[segment - 1];
      if (!(density_break > 0.0)
          || (segment > 1
              && !(density_break > density_breaks[segment - 2]))
          || !(gamma > 0.0)
          || gamma == 1.0) {
         throw std::invalid_argument(
            "piecewise-polytrope breaks must increase and gammas must be positive and non-unit"
         );
      }
      const amrex::Real k = previous_k * std::pow(
         density_break, previous_gamma - gamma
      );
      parameters.segment_density_break[segment - 1] = density_break;
      parameters.segment_gamma[segment] = gamma;
      parameters.segment_k[segment] = k;
      parameters.segment_energy_offset[segment] =
         parameters.segment_energy_offset[segment - 1]
         + previous_k * std::pow(density_break, previous_gamma - 1.0)
            / (previous_gamma - 1.0)
         - k * std::pow(density_break, gamma - 1.0) / (gamma - 1.0);
   }
   return parameters;
}

EOSParams make_read_piecewise_polytrope(
   PiecewisePolytropePreset preset,
   amrex::Real thermal_gamma,
   amrex::Real rho_floor,
   amrex::Real pressure_floor
)
{
   const ReadPiecewisePolytropeFit fit = read_piecewise_polytrope_fit(preset);
   const amrex::Real pressure1_cgs = std::pow(
      amrex::Real(10.0), fit.log10_pressure1_cgs
   );
   const amrex::Real core_k1_cgs = pressure1_cgs
      / std::pow(core_density1_cgs, fit.core_gamma[0]);

   amrex::Real effective_crust_k_cgs = crust_initial_k_cgs;
   for (int segment = 1; segment < 4; ++segment) {
      effective_crust_k_cgs *= std::pow(
         crust_density_break_cgs[segment - 1],
         crust_gamma[segment - 1] - crust_gamma[segment]
      );
   }
   const amrex::Real core_join_density_cgs = std::pow(
      effective_crust_k_cgs / core_k1_cgs,
      1.0 / (fit.core_gamma[0] - crust_gamma[3])
   );

   std::vector<amrex::Real> density_breaks;
   density_breaks.reserve(6);
   for (amrex::Real density_cgs : crust_density_break_cgs) {
      density_breaks.push_back(density_to_code(density_cgs));
   }
   density_breaks.push_back(density_to_code(core_join_density_cgs));
   density_breaks.push_back(density_to_code(core_density1_cgs));
   density_breaks.push_back(density_to_code(core_density2_cgs));

   std::vector<amrex::Real> following_gammas{
      crust_gamma[1],
      crust_gamma[2],
      crust_gamma[3],
      fit.core_gamma[0],
      fit.core_gamma[1],
      fit.core_gamma[2],
   };
   const amrex::Real initial_k_code =
      pressure_to_code(crust_initial_k_cgs)
      / std::pow(density_cgs_to_code, crust_gamma[0]);
   return make_piecewise_polytrope(
      initial_k_code,
      crust_gamma[0],
      density_breaks,
      following_gammas,
      thermal_gamma,
      rho_floor,
      pressure_floor
   );
}

} // namespace pyfluid_amrex::eos
