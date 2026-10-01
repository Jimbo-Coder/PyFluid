#include "pyfluid_amrex/eos/presets.hpp"

#include "eos/units.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pyfluid_amrex::eos {

using namespace amrex::literals;
namespace {

// Read et al. (2009), Tables II and III. Published densities and pressures
// are converted once into code units with G = c = M_sun = 1.
constexpr std::array<ReadPiecewisePolytropeFit, 4> read_fits{{
   {"sly", 34.384_rt, {{3.005_rt, 2.988_rt, 2.851_rt}}},
   {"apr4", 34.269_rt, {{2.830_rt, 3.445_rt, 3.348_rt}}},
   {"h4", 34.669_rt, {{2.909_rt, 2.246_rt, 2.144_rt}}},
   {"ms1", 34.858_rt, {{3.224_rt, 3.033_rt, 1.325_rt}}},
}};

constexpr amrex::Real crust_initial_k_over_c2_cgs = 6.80110e-9_rt;
constexpr std::array<amrex::Real, 4> crust_gamma{{
   1.58425_rt,
   1.28733_rt,
   0.62223_rt,
   1.35692_rt,
}};
constexpr std::array<amrex::Real, 3> crust_density_break_cgs{{
   2.44034e7_rt,
   3.78358e11_rt,
   2.62780e12_rt,
}};
constexpr amrex::Real core_density1_cgs = 5.0118723362727148e14_rt;
constexpr amrex::Real core_density2_cgs = 1.0e15_rt;

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
   if (!(initial_k > 0.0_rt) || !(initial_gamma > 0.0_rt)
       || initial_gamma == 1.0_rt) {
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
      if (!(density_break > 0.0_rt)
          || (segment > 1
              && !(density_break > density_breaks[segment - 2]))
          || !(gamma > 0.0_rt)
          || gamma == 1.0_rt) {
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
         + previous_k * std::pow(density_break, previous_gamma - 1.0_rt)
            / (previous_gamma - 1.0_rt)
         - k * std::pow(density_break, gamma - 1.0_rt) / (gamma - 1.0_rt);
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
      amrex::Real(10.0_rt), fit.log10_pressure1_cgs
   );
   // Build the fit in code units. Powers of CGS density can exceed the
   // FP32 range even when the final polytropic coefficient is representable.
   const amrex::Real core_density1 = units::density_from_cgs(core_density1_cgs);
   const amrex::Real core_k1 = units::pressure_from_cgs(pressure1_cgs)
      / std::pow(core_density1, fit.core_gamma[0]);
   const amrex::Real initial_k_code = crust_initial_k_over_c2_cgs
      * std::pow(units::density_cgs_to_code, 1.0_rt - crust_gamma[0]);

   std::vector<amrex::Real> density_breaks;
   density_breaks.reserve(6);
   for (amrex::Real density_cgs : crust_density_break_cgs) {
      density_breaks.push_back(units::density_from_cgs(density_cgs));
   }
   amrex::Real effective_crust_k = initial_k_code;
   for (int segment = 1; segment < 4; ++segment) {
      effective_crust_k *= std::pow(
         density_breaks[segment - 1],
         crust_gamma[segment - 1] - crust_gamma[segment]
      );
   }
   const amrex::Real core_join_density = std::pow(
      effective_crust_k / core_k1,
      1.0_rt / (fit.core_gamma[0] - crust_gamma[3])
   );
   density_breaks.push_back(core_join_density);
   density_breaks.push_back(core_density1);
   density_breaks.push_back(units::density_from_cgs(core_density2_cgs));

   std::vector<amrex::Real> following_gammas{
      crust_gamma[1],
      crust_gamma[2],
      crust_gamma[3],
      fit.core_gamma[0],
      fit.core_gamma[1],
      fit.core_gamma[2],
   };
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
