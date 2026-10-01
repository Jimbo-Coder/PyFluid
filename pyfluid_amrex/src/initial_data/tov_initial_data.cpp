#include "pyfluid_amrex/initial_data/tov.hpp"

#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/eos/config.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

struct TovIntegrationState {
   amrex::Real radius_squared = 0.0_rt;
   amrex::Real mass_over_radius = 0.0_rt;
   amrex::Real log_conformal_factor = 0.0_rt;
   amrex::Real baryonic_mass = 0.0_rt;
   amrex::Real proper_mass = 0.0_rt;
   amrex::Real komar_mass_without_injection_energy = 0.0_rt;
   amrex::Real tidal_y = 2.0_rt;
};

struct TovRawPoint {
   amrex::Real log_specific_enthalpy = 0.0_rt;
   TovIntegrationState state;
};

TovIntegrationState add_scaled(
   TovIntegrationState state,
   TovIntegrationState const& increment,
   amrex::Real scale
) noexcept
{
   state.radius_squared += scale * increment.radius_squared;
   state.mass_over_radius += scale * increment.mass_over_radius;
   state.log_conformal_factor += scale * increment.log_conformal_factor;
   state.baryonic_mass += scale * increment.baryonic_mass;
   state.proper_mass += scale * increment.proper_mass;
   state.komar_mass_without_injection_energy +=
      scale * increment.komar_mass_without_injection_energy;
   state.tidal_y += scale * increment.tidal_y;
   return state;
}

TovIntegrationState linear_combination(
   TovIntegrationState state,
   amrex::Real step,
   std::initializer_list<std::pair<amrex::Real, TovIntegrationState const*>> terms
) noexcept
{
   for (auto const& [coefficient, derivative] : terms) {
      state = add_scaled(state, *derivative, step * coefficient);
   }
   return state;
}

amrex::Real surface_enthalpy_sound_speed_ratio(EOSParams const& eos) noexcept
{
   const amrex::Real gamma = eos.segment_gamma[0];
   if (gamma < 2.0_rt) {
      return 0.0_rt;
   }
   if (std::abs(gamma - 2.0_rt) <= 64.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
      const amrex::Real rest_energy = 1.0_rt + eos.segment_energy_offset[0];
      return rest_energy * rest_energy / (gamma * eos.segment_k[0]);
   }
   return std::numeric_limits<amrex::Real>::infinity();
}

TovIntegrationState tov_derivative(
   amrex::Real log_specific_enthalpy,
   TovIntegrationState const& state,
   EOSParams const& eos,
   bool evolve_tidal
)
{
   const ColdBarotropeState matter = cold_barotrope_from_log_enthalpy(
      std::max(log_specific_enthalpy, amrex::Real(0.0_rt)), eos
   );
   const amrex::Real u = state.radius_squared;
   const amrex::Real v = state.mass_over_radius;
   const amrex::Real denominator = v + constants::four_pi * u * matter.pressure;
   const amrex::Real radial_factor = 1.0_rt - 2.0_rt * v;
   if (!(u > 0.0_rt) || !(denominator > 0.0_rt) || !(radial_factor > 0.0_rt)) {
      throw std::runtime_error("invalid state in enthalpy-coordinate TOV integration");
   }

   TovIntegrationState derivative;
   derivative.radius_squared = -2.0_rt * u * radial_factor / denominator;
   derivative.mass_over_radius = -radial_factor * (
      constants::four_pi * u * matter.energy_density - v
   ) / denominator;
   const amrex::Real sqrt_radial_factor = std::sqrt(radial_factor);
   derivative.log_conformal_factor =
      sqrt_radial_factor / (1.0_rt + sqrt_radial_factor) * v / denominator;

   const amrex::Real radius = std::sqrt(u);
   const amrex::Real radius_derivative = derivative.radius_squared / (2.0_rt * radius);
   const amrex::Real proper_shell = constants::four_pi * u
      * radius_derivative / sqrt_radial_factor;
   derivative.baryonic_mass = proper_shell * matter.rho;
   derivative.proper_mass = proper_shell * matter.energy_density;
   derivative.komar_mass_without_injection_energy = proper_shell * (
      matter.energy_density + 3.0_rt * matter.pressure
   ) / matter.specific_enthalpy;

   if (evolve_tidal) {
      amrex::Real enthalpy_sound_speed_ratio = 0.0_rt;
      if (matter.rho > 0.0_rt && matter.sound_speed_squared > 0.0_rt) {
         enthalpy_sound_speed_ratio =
            (matter.energy_density + matter.pressure) / matter.sound_speed_squared;
      } else {
         enthalpy_sound_speed_ratio = surface_enthalpy_sound_speed_ratio(eos);
      }
      const amrex::Real f = (
         1.0_rt - constants::four_pi * u
            * (matter.energy_density - matter.pressure)
      ) / radial_factor;
      const amrex::Real gravity = v + constants::four_pi * u * matter.pressure;
      const amrex::Real radius_squared_q = constants::four_pi * u * (
         5.0_rt * matter.energy_density + 9.0_rt * matter.pressure
         + enthalpy_sound_speed_ratio
      ) / radial_factor - 6.0_rt / radial_factor
         - 4.0_rt * gravity * gravity / (radial_factor * radial_factor);
      const amrex::Real y = state.tidal_y;
      const amrex::Real tidal_radius_derivative =
         -(y * y + y * f + radius_squared_q) / radius;
      derivative.tidal_y = tidal_radius_derivative * radius_derivative;
   } else {
      derivative.tidal_y = 0.0_rt;
   }
   return derivative;
}

struct DormandPrinceStep {
   TovIntegrationState fifth_order;
   TovIntegrationState fourth_order;
};

DormandPrinceStep dormand_prince_step(
   amrex::Real coordinate,
   TovIntegrationState const& state,
   amrex::Real step,
   EOSParams const& eos,
   bool evolve_tidal
)
{
   const TovIntegrationState k1 = tov_derivative(coordinate, state, eos, evolve_tidal);
   const TovIntegrationState k2 = tov_derivative(
      coordinate + step / 5.0_rt,
      linear_combination(state, step, {{1.0_rt / 5.0_rt, &k1}}),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k3 = tov_derivative(
      coordinate + 3.0_rt * step / 10.0_rt,
      linear_combination(state, step, {{3.0_rt / 40.0_rt, &k1}, {9.0_rt / 40.0_rt, &k2}}),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k4 = tov_derivative(
      coordinate + 4.0_rt * step / 5.0_rt,
      linear_combination(
         state, step, {{44.0_rt / 45.0_rt, &k1}, {-56.0_rt / 15.0_rt, &k2}, {32.0_rt / 9.0_rt, &k3}}
      ),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k5 = tov_derivative(
      coordinate + 8.0_rt * step / 9.0_rt,
      linear_combination(
         state,
         step,
         {
            {19372.0_rt / 6561.0_rt, &k1},
            {-25360.0_rt / 2187.0_rt, &k2},
            {64448.0_rt / 6561.0_rt, &k3},
            {-212.0_rt / 729.0_rt, &k4},
         }
      ),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k6 = tov_derivative(
      coordinate + step,
      linear_combination(
         state,
         step,
         {
            {9017.0_rt / 3168.0_rt, &k1},
            {-355.0_rt / 33.0_rt, &k2},
            {46732.0_rt / 5247.0_rt, &k3},
            {49.0_rt / 176.0_rt, &k4},
            {-5103.0_rt / 18656.0_rt, &k5},
         }
      ),
      eos,
      evolve_tidal
   );
   const TovIntegrationState fifth = linear_combination(
      state,
      step,
      {
         {35.0_rt / 384.0_rt, &k1},
         {500.0_rt / 1113.0_rt, &k3},
         {125.0_rt / 192.0_rt, &k4},
         {-2187.0_rt / 6784.0_rt, &k5},
         {11.0_rt / 84.0_rt, &k6},
      }
   );
   const TovIntegrationState k7 = tov_derivative(
      coordinate + step, fifth, eos, evolve_tidal
   );
   const TovIntegrationState fourth = linear_combination(
      state,
      step,
      {
         {5179.0_rt / 57600.0_rt, &k1},
         {7571.0_rt / 16695.0_rt, &k3},
         {393.0_rt / 640.0_rt, &k4},
         {-92097.0_rt / 339200.0_rt, &k5},
         {187.0_rt / 2100.0_rt, &k6},
         {1.0_rt / 40.0_rt, &k7},
      }
   );
   return {fifth, fourth};
}

std::array<amrex::Real, 7> state_components(TovIntegrationState const& state) noexcept
{
   return {
      state.radius_squared,
      state.mass_over_radius,
      state.log_conformal_factor,
      state.baryonic_mass,
      state.proper_mass,
      state.komar_mass_without_injection_energy,
      state.tidal_y,
   };
}

amrex::Real normalized_step_error(
   TovIntegrationState const& initial,
   DormandPrinceStep const& candidate,
   TovSolveOptions const& options,
   bool evolve_tidal
) noexcept
{
   const auto before = state_components(initial);
   const auto fifth = state_components(candidate.fifth_order);
   const auto fourth = state_components(candidate.fourth_order);
   const int component_count = evolve_tidal ? 7 : 6;
   amrex::Real error = 0.0_rt;
   for (int component = 0; component < component_count; ++component) {
      const amrex::Real scale = options.absolute_tolerance
         + options.relative_tolerance * std::max(
            std::abs(before[component]), std::abs(fifth[component])
         );
      error = std::max(error, std::abs(fifth[component] - fourth[component]) / scale);
   }
   return error;
}

amrex::Real love_number_k2(amrex::Real compactness, amrex::Real y) noexcept
{
   const amrex::Real one_minus_two_c = 1.0_rt - 2.0_rt * compactness;
   const amrex::Real response = 2.0_rt + 2.0_rt * compactness * (y - 1.0_rt) - y;
   const amrex::Real numerator = 8.0_rt * std::pow(compactness, 5.0_rt)
      * one_minus_two_c * one_minus_two_c * response / 5.0_rt;
   const amrex::Real denominator =
      2.0_rt * compactness * (
         6.0_rt - 3.0_rt * y + 3.0_rt * compactness * (5.0_rt * y - 8.0_rt)
      )
      + 4.0_rt * std::pow(compactness, 3.0_rt) * (
         13.0_rt - 11.0_rt * y
         + compactness * (3.0_rt * y - 2.0_rt)
         + 2.0_rt * compactness * compactness * (1.0_rt + y)
      )
      + 3.0_rt * one_minus_two_c * one_minus_two_c * response
         * std::log(one_minus_two_c);
   return numerator / denominator;
}

TovInterpolationPoint interpolation_point(TovProfilePoint const& point)
{
   TovInterpolationPoint result;
   result.state = {point.isotropic_radius, point.rho, point.pressure,
                   point.specific_internal_energy, point.lapse, point.conformal_factor};
   const amrex::Real radius = point.isotropic_radius;
   const amrex::Real conformal = point.conformal_factor;
   const amrex::Real energy = point.energy_density;
   const amrex::Real pressure = point.pressure;
   if (radius == 0.0_rt) {
      const amrex::Real conformal_fourth = std::pow(conformal, 4.0_rt);
      result.lapse_second = constants::four_pi * point.lapse * conformal_fourth * (energy / 3.0_rt + pressure);
      result.conformal_second = -constants::two_pi / 3.0_rt * energy * conformal_fourth * conformal;
      return result;
   }

   // dR/dr = R sqrt(1-2m/R)/r and d(log alpha)/dr follows
   // hydrostatic equilibrium. Differentiate these TOV identities once.
   const amrex::Real areal = point.areal_radius;
   const amrex::Real mass = point.enclosed_gravitational_mass;
   const amrex::Real radial_factor = std::sqrt(1.0_rt - 2.0_rt * mass / areal);
   const amrex::Real areal_first = areal * radial_factor / radius;
   const amrex::Real mass_first = constants::four_pi * areal * areal * energy * areal_first;
   const amrex::Real factor_first = (mass * areal_first / (areal * areal) - mass_first / areal) / radial_factor;
   const amrex::Real conformal_log_first = -mass / (radius * areal * (1.0_rt + radial_factor));
   const amrex::Real conformal_log_second = factor_first / (2.0_rt * radius) - conformal_log_first / radius;
   const amrex::Real numerator = mass + constants::four_pi * areal * areal * areal * pressure;
   const amrex::Real denominator = radius * areal * radial_factor;
   const amrex::Real lapse_log_first = numerator / denominator;
   const amrex::Real pressure_first = -(energy + pressure) * lapse_log_first;
   const amrex::Real numerator_first = mass_first + constants::four_pi * areal * areal
      * (3.0_rt * areal_first * pressure + areal * pressure_first);
   const amrex::Real lapse_log_second = numerator_first / denominator
      - lapse_log_first * (1.0_rt / radius + areal_first / areal + factor_first / radial_factor);
   result.conformal_first = conformal * conformal_log_first;
   result.conformal_second = conformal * (conformal_log_first * conformal_log_first + conformal_log_second);
   result.lapse_first = point.lapse * lapse_log_first;
   result.lapse_second = point.lapse * (lapse_log_first * lapse_log_first + lapse_log_second);
   return result;
}

} // namespace

TovSolution::TovSolution(EOSParams const& eos, TovSolveOptions const& options)
{
   if (!(options.central_density > 0.0_rt)
       || !(options.absolute_tolerance > 0.0_rt)
       || !(options.relative_tolerance > 0.0_rt)
       || options.max_steps < 16) {
      throw std::invalid_argument("invalid TOV solve options");
   }
   const ColdBarotropeState center_matter = cold_barotrope_from_density(
      options.central_density, eos
   );
   const amrex::Real central_log_enthalpy = std::log(center_matter.specific_enthalpy);
   if (!(central_log_enthalpy > 0.0_rt)) {
      throw std::runtime_error("TOV central specific enthalpy is not above its surface value");
   }
   const bool evolve_tidal = eos.segment_gamma[0] <= 2.0_rt;
   // The regular center expansion must start at a distinct representable
   // enthalpy. In FP32 the original 1e-8 relative offset rounded back to the
   // center while its nonzero radius remained, making the first ODE step stiff.
   const amrex::Real requested_center_offset = std::max(
      1.0e-12_rt, central_log_enthalpy * std::max(
         1.0e-8_rt, 64.0_rt * std::numeric_limits<amrex::Real>::epsilon()
      )
   );
   const amrex::Real first_coordinate = central_log_enthalpy - requested_center_offset;
   const amrex::Real center_offset = central_log_enthalpy - first_coordinate;
   const amrex::Real first_radius_squared = 3.0_rt * center_offset / (
      2.0_rt * constants::pi
      * (center_matter.energy_density + 3.0_rt * center_matter.pressure)
   );
   const amrex::Real first_radius = std::sqrt(first_radius_squared);
   TovIntegrationState state;
   state.radius_squared = first_radius_squared;
   state.mass_over_radius = constants::four_pi
      * center_matter.energy_density * first_radius_squared / 3.0_rt;
   state.baryonic_mass = constants::four_pi
      * center_matter.rho * first_radius_squared * first_radius / 3.0_rt;
   state.proper_mass = constants::four_pi
      * center_matter.energy_density * first_radius_squared * first_radius / 3.0_rt;
   state.komar_mass_without_injection_energy = constants::four_pi
      * (center_matter.energy_density + 3.0_rt * center_matter.pressure)
      * first_radius_squared * first_radius
      / (3.0_rt * center_matter.specific_enthalpy);

   std::vector<TovRawPoint> raw_points;
   raw_points.reserve(1024);
   raw_points.push_back({central_log_enthalpy, {}});
   raw_points.push_back({first_coordinate, state});

   amrex::Real coordinate = first_coordinate;
   const amrex::Real maximum_step = central_log_enthalpy / 256.0_rt;
   amrex::Real step = -std::min(maximum_step, coordinate);
   int attempts = 0;
   while (coordinate > 0.0_rt && attempts < options.max_steps) {
      ++attempts;
      step = -std::min(std::abs(step), coordinate);
      const DormandPrinceStep candidate = dormand_prince_step(
         coordinate, state, step, eos, evolve_tidal
      );
      const amrex::Real error = normalized_step_error(
         state, candidate, options, evolve_tidal
      );
      if (std::isfinite(error) && error <= 1.0_rt) {
         coordinate = std::max(amrex::Real(0.0_rt), coordinate + step);
         state = candidate.fifth_order;
         raw_points.push_back({coordinate, state});
      }
      const amrex::Real factor = error > 0.0_rt && std::isfinite(error)
         ? std::clamp(0.9_rt * std::pow(error, -0.2_rt), 0.2_rt, 5.0_rt)
         : 2.0_rt;
      step = -std::min(maximum_step, std::abs(step) * factor);
      if (std::abs(step) < 16.0_rt * std::numeric_limits<amrex::Real>::epsilon()
            * central_log_enthalpy) {
         std::ostringstream message;
         message << std::scientific << "TOV adaptive integration step underflow: h="
                 << coordinate << " step=" << step << " normalized_error=" << error
                 << " attempts=" << attempts << " atol=" << options.absolute_tolerance
                 << " rtol=" << options.relative_tolerance;
         throw std::runtime_error(message.str());
      }
   }
   if (coordinate > 0.0_rt) {
      throw std::runtime_error("TOV adaptive integration exceeded its max_steps safety bound");
   }

   const amrex::Real areal_radius = std::sqrt(state.radius_squared);
   const amrex::Real gravitational_mass = state.mass_over_radius * areal_radius;
   if (!(areal_radius > 2.0_rt * gravitational_mass)) {
      throw std::runtime_error("TOV surface lies inside its Schwarzschild radius");
   }
   const amrex::Real compactness = gravitational_mass / areal_radius;
   const amrex::Real injection_energy = std::sqrt(1.0_rt - 2.0_rt * compactness);
   const amrex::Real isotropic_radius = 0.5_rt * (
      areal_radius - gravitational_mass
      + std::sqrt(areal_radius * (areal_radius - 2.0_rt * gravitational_mass))
   );
   const amrex::Real surface_log_conformal_factor = 0.5_rt * std::log(
      areal_radius / isotropic_radius
   );
   const amrex::Real conformal_shift = surface_log_conformal_factor
      - state.log_conformal_factor;

   observables_.central_density = options.central_density;
   observables_.central_pressure = center_matter.pressure;
   observables_.central_energy_density = center_matter.energy_density;
   observables_.gravitational_mass = gravitational_mass;
   observables_.baryonic_mass = state.baryonic_mass;
   observables_.proper_mass = state.proper_mass;
   observables_.komar_mass = injection_energy
      * state.komar_mass_without_injection_energy;
   observables_.areal_radius = areal_radius;
   observables_.isotropic_radius = isotropic_radius;
   observables_.compactness = compactness;
   observables_.binding_energy = state.baryonic_mass - gravitational_mass;
   observables_.surface_redshift = 1.0_rt / injection_energy - 1.0_rt;
   observables_.injection_energy = injection_energy;
   observables_.tidal_observables_available = evolve_tidal;
   if (evolve_tidal) {
      observables_.love_number_k2 = love_number_k2(compactness, state.tidal_y);
      observables_.tidal_deformability = 2.0_rt * observables_.love_number_k2
         / (3.0_rt * std::pow(compactness, 5.0_rt));
   }

   profile_.reserve(raw_points.size());
   for (TovRawPoint const& raw : raw_points) {
      const ColdBarotropeState matter = raw.log_specific_enthalpy > 0.0_rt
         ? cold_barotrope_from_log_enthalpy(raw.log_specific_enthalpy, eos)
         : ColdBarotropeState{};
      const amrex::Real radius = std::sqrt(std::max(
         raw.state.radius_squared, amrex::Real(0.0_rt)
      ));
      const amrex::Real conformal_factor = std::exp(
         raw.state.log_conformal_factor + conformal_shift
      );
      profile_.push_back({
         raw.log_specific_enthalpy,
         radius,
         radius / (conformal_factor * conformal_factor),
         matter.rho,
         matter.pressure,
         matter.specific_internal_energy,
         matter.energy_density,
         raw.state.mass_over_radius * radius,
         raw.state.baryonic_mass,
         raw.state.proper_mass,
         injection_energy / std::exp(raw.log_specific_enthalpy),
         conformal_factor,
         raw.state.tidal_y,
      });
   }
   interpolation_points_.reserve(profile_.size());
   for (TovProfilePoint const& point : profile_) {
      interpolation_points_.push_back(interpolation_point(point));
   }

}

TovSample TovSolution::sample_isotropic(amrex::Real radius) const noexcept
{
   const TovInitialDataView view{interpolation_points_.data(),
      static_cast<int>(interpolation_points_.size()), observables_.gravitational_mass,
      observables_.isotropic_radius, observables_.injection_energy};
   return view.sample(radius);
}

std::vector<TovSolution> solve_tov_sequence(
   EOSParams const& eos,
   std::vector<amrex::Real> const& central_densities,
   TovSolveOptions options
)
{
   std::vector<TovSolution> sequence;
   sequence.reserve(central_densities.size());
   for (amrex::Real central_density : central_densities) {
      options.central_density = central_density;
      sequence.emplace_back(eos, options);
   }
   return sequence;
}

void write_tov_sequence(
   std::string const& path,
   std::string const& eos_label,
   std::vector<TovSolution> const& sequence
)
{
   std::ofstream output(path);
   if (!output) {
      throw std::runtime_error("failed to open TOV sequence output: " + path);
   }
   if (!eos_label.empty()) {
      output << "# eos_label=" << eos_label << '\n';
   }
   output << "central_density\tcentral_pressure\tcentral_energy_density\t"
          << "gravitational_mass\tbaryonic_mass\tproper_mass\tkomar_mass\t"
          << "areal_radius\tisotropic_radius\tcompactness\tbinding_energy\t"
          << "surface_redshift\tlove_number_k2\ttidal_deformability\t"
          << "tidal_available\tstable_branch\n";
   output << std::setprecision(17);
   for (std::size_t index = 0; index < sequence.size(); ++index) {
      TovObservables const& value = sequence[index].observables();
      const std::size_t lower = index == 0 ? 0 : index - 1;
      const std::size_t upper = std::min(index + 1, sequence.size() - 1);
      const bool stable = sequence[upper].observables().gravitational_mass
         >= sequence[lower].observables().gravitational_mass;
      output << value.central_density << '\t'
             << value.central_pressure << '\t'
             << value.central_energy_density << '\t'
             << value.gravitational_mass << '\t'
             << value.baryonic_mass << '\t'
             << value.proper_mass << '\t'
             << value.komar_mass << '\t'
             << value.areal_radius << '\t'
             << value.isotropic_radius << '\t'
             << value.compactness << '\t'
             << value.binding_energy << '\t'
             << value.surface_redshift << '\t'
             << value.love_number_k2 << '\t'
             << value.tidal_deformability << '\t'
             << static_cast<int>(value.tidal_observables_available) << '\t'
             << static_cast<int>(stable) << '\n';
   }
}

TovInitialData::TovInitialData(RunConfig const& cfg)
{
   if (cfg.ic != "tov" && cfg.ic != "rotating_star") {
      return;
   }
   const EOSParams eos = eos_params(cfg);
   solution_ = TovSolution(
      eos,
      {
         cfg.tov_central_density,
         cfg.tov_absolute_tolerance,
         cfg.tov_relative_tolerance,
         cfg.tov_max_steps,
      }
   );
   TovObservables const& observables = solution_.observables();
   const auto& host_samples = solution_.interpolation_points();
   device_samples_.resize(host_samples.size());
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      host_samples.begin(),
      host_samples.end(),
      device_samples_.begin()
   );
   amrex::Print() << std::setprecision(12)
                  << "TOV initial data: M=" << observables.gravitational_mass
                  << " Mb=" << observables.baryonic_mass
                  << " Mp=" << observables.proper_mass
                  << " Mkomar=" << observables.komar_mass
                  << " R_areal=" << observables.areal_radius
                  << " R_isotropic=" << observables.isotropic_radius
                  << " C=" << observables.compactness
                  << " Lambda=" << observables.tidal_deformability
                  << " samples=" << host_samples.size() << "\n";
}

TovInitialDataView TovInitialData::view() const noexcept
{
   const auto& observables = solution_.observables();
   return {
      device_samples_.empty() ? nullptr : device_samples_.data(),
      static_cast<int>(device_samples_.size()),
      observables.gravitational_mass,
      observables.isotropic_radius,
      observables.injection_energy,
   };
}

} // namespace pyfluid_amrex
