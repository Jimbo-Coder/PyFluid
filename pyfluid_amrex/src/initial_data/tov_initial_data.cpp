#include "pyfluid_amrex/initial_data/tov.hpp"

#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/eos/config.hpp"

namespace pyfluid_amrex {
namespace {

struct TovIntegrationState {
   amrex::Real radius_squared = 0.0;
   amrex::Real mass_over_radius = 0.0;
   amrex::Real log_conformal_factor = 0.0;
   amrex::Real baryonic_mass = 0.0;
   amrex::Real proper_mass = 0.0;
   amrex::Real komar_mass_without_injection_energy = 0.0;
   amrex::Real tidal_y = 2.0;
};

struct TovRawPoint {
   amrex::Real log_specific_enthalpy = 0.0;
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
   if (gamma < 2.0) {
      return 0.0;
   }
   if (std::abs(gamma - 2.0) <= 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
      const amrex::Real rest_energy = 1.0 + eos.segment_energy_offset[0];
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
      std::max(log_specific_enthalpy, amrex::Real(0.0)), eos
   );
   const amrex::Real u = state.radius_squared;
   const amrex::Real v = state.mass_over_radius;
   const amrex::Real denominator = v + constants::four_pi * u * matter.pressure;
   const amrex::Real radial_factor = 1.0 - 2.0 * v;
   if (!(u > 0.0) || !(denominator > 0.0) || !(radial_factor > 0.0)) {
      throw std::runtime_error("invalid state in enthalpy-coordinate TOV integration");
   }

   TovIntegrationState derivative;
   derivative.radius_squared = -2.0 * u * radial_factor / denominator;
   derivative.mass_over_radius = -radial_factor * (
      constants::four_pi * u * matter.energy_density - v
   ) / denominator;
   const amrex::Real sqrt_radial_factor = std::sqrt(radial_factor);
   derivative.log_conformal_factor =
      sqrt_radial_factor / (1.0 + sqrt_radial_factor) * v / denominator;

   const amrex::Real radius = std::sqrt(u);
   const amrex::Real radius_derivative = derivative.radius_squared / (2.0 * radius);
   const amrex::Real proper_shell = constants::four_pi * u
      * radius_derivative / sqrt_radial_factor;
   derivative.baryonic_mass = proper_shell * matter.rho;
   derivative.proper_mass = proper_shell * matter.energy_density;
   derivative.komar_mass_without_injection_energy = proper_shell * (
      matter.energy_density + 3.0 * matter.pressure
   ) / matter.specific_enthalpy;

   if (evolve_tidal) {
      amrex::Real enthalpy_sound_speed_ratio = 0.0;
      if (matter.rho > 0.0 && matter.sound_speed_squared > 0.0) {
         enthalpy_sound_speed_ratio =
            (matter.energy_density + matter.pressure) / matter.sound_speed_squared;
      } else {
         enthalpy_sound_speed_ratio = surface_enthalpy_sound_speed_ratio(eos);
      }
      const amrex::Real f = (
         1.0 - constants::four_pi * u
            * (matter.energy_density - matter.pressure)
      ) / radial_factor;
      const amrex::Real gravity = v + constants::four_pi * u * matter.pressure;
      const amrex::Real radius_squared_q = constants::four_pi * u * (
         5.0 * matter.energy_density + 9.0 * matter.pressure
         + enthalpy_sound_speed_ratio
      ) / radial_factor - 6.0 / radial_factor
         - 4.0 * gravity * gravity / (radial_factor * radial_factor);
      const amrex::Real y = state.tidal_y;
      const amrex::Real tidal_radius_derivative =
         -(y * y + y * f + radius_squared_q) / radius;
      derivative.tidal_y = tidal_radius_derivative * radius_derivative;
   } else {
      derivative.tidal_y = 0.0;
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
      coordinate + step / 5.0,
      linear_combination(state, step, {{1.0 / 5.0, &k1}}),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k3 = tov_derivative(
      coordinate + 3.0 * step / 10.0,
      linear_combination(state, step, {{3.0 / 40.0, &k1}, {9.0 / 40.0, &k2}}),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k4 = tov_derivative(
      coordinate + 4.0 * step / 5.0,
      linear_combination(
         state, step, {{44.0 / 45.0, &k1}, {-56.0 / 15.0, &k2}, {32.0 / 9.0, &k3}}
      ),
      eos,
      evolve_tidal
   );
   const TovIntegrationState k5 = tov_derivative(
      coordinate + 8.0 * step / 9.0,
      linear_combination(
         state,
         step,
         {
            {19372.0 / 6561.0, &k1},
            {-25360.0 / 2187.0, &k2},
            {64448.0 / 6561.0, &k3},
            {-212.0 / 729.0, &k4},
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
            {9017.0 / 3168.0, &k1},
            {-355.0 / 33.0, &k2},
            {46732.0 / 5247.0, &k3},
            {49.0 / 176.0, &k4},
            {-5103.0 / 18656.0, &k5},
         }
      ),
      eos,
      evolve_tidal
   );
   const TovIntegrationState fifth = linear_combination(
      state,
      step,
      {
         {35.0 / 384.0, &k1},
         {500.0 / 1113.0, &k3},
         {125.0 / 192.0, &k4},
         {-2187.0 / 6784.0, &k5},
         {11.0 / 84.0, &k6},
      }
   );
   const TovIntegrationState k7 = tov_derivative(
      coordinate + step, fifth, eos, evolve_tidal
   );
   const TovIntegrationState fourth = linear_combination(
      state,
      step,
      {
         {5179.0 / 57600.0, &k1},
         {7571.0 / 16695.0, &k3},
         {393.0 / 640.0, &k4},
         {-92097.0 / 339200.0, &k5},
         {187.0 / 2100.0, &k6},
         {1.0 / 40.0, &k7},
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
   amrex::Real error = 0.0;
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
   const amrex::Real one_minus_two_c = 1.0 - 2.0 * compactness;
   const amrex::Real response = 2.0 + 2.0 * compactness * (y - 1.0) - y;
   const amrex::Real numerator = 8.0 * std::pow(compactness, 5)
      * one_minus_two_c * one_minus_two_c * response / 5.0;
   const amrex::Real denominator =
      2.0 * compactness * (
         6.0 - 3.0 * y + 3.0 * compactness * (5.0 * y - 8.0)
      )
      + 4.0 * std::pow(compactness, 3) * (
         13.0 - 11.0 * y
         + compactness * (3.0 * y - 2.0)
         + 2.0 * compactness * compactness * (1.0 + y)
      )
      + 3.0 * one_minus_two_c * one_minus_two_c * response
         * std::log(one_minus_two_c);
   return numerator / denominator;
}

TovSample interpolate_sample(
   TovProfilePoint const& left,
   TovProfilePoint const& right,
   amrex::Real isotropic_radius
) noexcept
{
   const amrex::Real width = right.isotropic_radius - left.isotropic_radius;
   const amrex::Real weight = width > 0.0
      ? (isotropic_radius - left.isotropic_radius) / width
      : 0.0;
   return {
      isotropic_radius,
      left.rho + weight * (right.rho - left.rho),
      left.pressure + weight * (right.pressure - left.pressure),
      left.specific_internal_energy + weight * (
         right.specific_internal_energy - left.specific_internal_energy
      ),
      left.lapse + weight * (right.lapse - left.lapse),
      left.conformal_factor + weight * (
         right.conformal_factor - left.conformal_factor
      ),
   };
}

} // namespace

TovSolution::TovSolution(EOSParams const& eos, TovSolveOptions const& options)
{
   if (!(options.central_density > 0.0)
       || !(options.absolute_tolerance > 0.0)
       || !(options.relative_tolerance > 0.0)
       || options.max_steps < 16) {
      throw std::invalid_argument("invalid TOV solve options");
   }
   const ColdBarotropeState center_matter = cold_barotrope_from_density(
      options.central_density, eos
   );
   const amrex::Real central_log_enthalpy = std::log(center_matter.specific_enthalpy);
   if (!(central_log_enthalpy > 0.0)) {
      throw std::runtime_error("TOV central specific enthalpy is not above its surface value");
   }
   const bool evolve_tidal = eos.segment_gamma[0] <= 2.0;
   const amrex::Real center_offset = std::max(
      1.0e-12, central_log_enthalpy * 1.0e-8
   );
   const amrex::Real first_coordinate = central_log_enthalpy - center_offset;
   const amrex::Real first_radius_squared = 3.0 * center_offset / (
      2.0 * constants::pi
      * (center_matter.energy_density + 3.0 * center_matter.pressure)
   );
   const amrex::Real first_radius = std::sqrt(first_radius_squared);
   TovIntegrationState state;
   state.radius_squared = first_radius_squared;
   state.mass_over_radius = constants::four_pi
      * center_matter.energy_density * first_radius_squared / 3.0;
   state.baryonic_mass = constants::four_pi
      * center_matter.rho * first_radius_squared * first_radius / 3.0;
   state.proper_mass = constants::four_pi
      * center_matter.energy_density * first_radius_squared * first_radius / 3.0;
   state.komar_mass_without_injection_energy = constants::four_pi
      * (center_matter.energy_density + 3.0 * center_matter.pressure)
      * first_radius_squared * first_radius
      / (3.0 * center_matter.specific_enthalpy);

   std::vector<TovRawPoint> raw_points;
   raw_points.reserve(1024);
   raw_points.push_back({central_log_enthalpy, {}});
   raw_points.push_back({first_coordinate, state});

   amrex::Real coordinate = first_coordinate;
   const amrex::Real maximum_step = central_log_enthalpy / 256.0;
   amrex::Real step = -std::min(maximum_step, coordinate);
   int attempts = 0;
   while (coordinate > 0.0 && attempts < options.max_steps) {
      ++attempts;
      step = -std::min(std::abs(step), coordinate);
      const DormandPrinceStep candidate = dormand_prince_step(
         coordinate, state, step, eos, evolve_tidal
      );
      const amrex::Real error = normalized_step_error(
         state, candidate, options, evolve_tidal
      );
      if (std::isfinite(error) && error <= 1.0) {
         coordinate = std::max(amrex::Real(0.0), coordinate + step);
         state = candidate.fifth_order;
         raw_points.push_back({coordinate, state});
      }
      const amrex::Real factor = error > 0.0 && std::isfinite(error)
         ? std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0)
         : 2.0;
      step = -std::min(maximum_step, std::abs(step) * factor);
      if (std::abs(step) < 16.0 * std::numeric_limits<amrex::Real>::epsilon()
            * central_log_enthalpy) {
         throw std::runtime_error("TOV adaptive integration step underflow");
      }
   }
   if (coordinate > 0.0) {
      throw std::runtime_error("TOV adaptive integration exceeded its max_steps safety bound");
   }

   const amrex::Real areal_radius = std::sqrt(state.radius_squared);
   const amrex::Real gravitational_mass = state.mass_over_radius * areal_radius;
   if (!(areal_radius > 2.0 * gravitational_mass)) {
      throw std::runtime_error("TOV surface lies inside its Schwarzschild radius");
   }
   const amrex::Real compactness = gravitational_mass / areal_radius;
   const amrex::Real injection_energy = std::sqrt(1.0 - 2.0 * compactness);
   const amrex::Real isotropic_radius = 0.5 * (
      areal_radius - gravitational_mass
      + std::sqrt(areal_radius * (areal_radius - 2.0 * gravitational_mass))
   );
   const amrex::Real surface_log_conformal_factor = 0.5 * std::log(
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
   observables_.surface_redshift = 1.0 / injection_energy - 1.0;
   observables_.injection_energy = injection_energy;
   observables_.tidal_observables_available = evolve_tidal;
   if (evolve_tidal) {
      observables_.love_number_k2 = love_number_k2(compactness, state.tidal_y);
      observables_.tidal_deformability = 2.0 * observables_.love_number_k2
         / (3.0 * std::pow(compactness, 5));
   }

   profile_.reserve(raw_points.size());
   for (TovRawPoint const& raw : raw_points) {
      const ColdBarotropeState matter = raw.log_specific_enthalpy > 0.0
         ? cold_barotrope_from_log_enthalpy(raw.log_specific_enthalpy, eos)
         : ColdBarotropeState{};
      const amrex::Real radius = std::sqrt(std::max(
         raw.state.radius_squared, amrex::Real(0.0)
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
}

TovSample TovSolution::sample_isotropic(amrex::Real radius) const noexcept
{
   if (profile_.empty()) {
      return {};
   }
   if (radius >= observables_.isotropic_radius) {
      const amrex::Real mass_over_radius = observables_.gravitational_mass
         / (2.0 * radius);
      const amrex::Real factor = 1.0 + mass_over_radius;
      return {
         radius,
         0.0,
         0.0,
         0.0,
         (1.0 - mass_over_radius) / factor,
         factor,
      };
   }
   auto const upper = std::upper_bound(
      profile_.begin(),
      profile_.end(),
      radius,
      [](amrex::Real value, TovProfilePoint const& point) {
         return value < point.isotropic_radius;
      }
   );
   if (upper == profile_.begin()) {
      return interpolate_sample(profile_.front(), profile_[1], radius);
   }
   return interpolate_sample(*(upper - 1), *upper, radius);
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
   if (cfg.ic != "tov") {
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
   gravitational_mass_ = observables.gravitational_mass;
   surface_radius_ = observables.isotropic_radius;
   host_samples_.reserve(solution_.profile().size());
   for (TovProfilePoint const& point : solution_.profile()) {
      host_samples_.push_back({
         point.isotropic_radius,
         point.rho,
         point.pressure,
         point.specific_internal_energy,
         point.lapse,
         point.conformal_factor,
      });
   }
   device_samples_.resize(host_samples_.size());
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      host_samples_.begin(),
      host_samples_.end(),
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
                  << " samples=" << host_samples_.size() << "\n";
}

TovInitialDataView TovInitialData::view() const noexcept
{
   return {
      device_samples_.empty() ? nullptr : device_samples_.data(),
      static_cast<int>(device_samples_.size()),
      gravitational_mass_,
      surface_radius_,
   };
}

} // namespace pyfluid_amrex
