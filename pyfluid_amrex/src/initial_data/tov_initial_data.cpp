#include "pyfluid_amrex/initial_data/tov.hpp"

#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos.hpp"

namespace pyfluid_amrex {
namespace {

struct TovIntegrationState {
   amrex::Real mass;
   amrex::Real pressure;
   amrex::Real log_lapse;
   amrex::Real log_isotropic_ratio;
};

amrex::Real cold_density_from_pressure(amrex::Real pressure, EOSParams const& eos)
{
   int segment = 0;
   while (segment < eos.polytrope_segments - 1) {
      const amrex::Real break_pressure = eos.segment_k[segment]
         * std::pow(eos.segment_density_break[segment], eos.segment_gamma[segment]);
      if (pressure < break_pressure) {
         break;
      }
      ++segment;
   }
   return std::pow(
      pressure / eos.segment_k[segment], 1.0 / eos.segment_gamma[segment]
   );
}

TovIntegrationState tov_derivative(
   amrex::Real radius,
   TovIntegrationState const& state,
   EOSParams const& eos,
   amrex::Real surface_pressure
)
{
   if (!(state.pressure > surface_pressure)) {
      return {};
   }
   const amrex::Real rho = cold_density_from_pressure(state.pressure, eos);
   const amrex::Real energy_density = rho
      + eos_cold_energy_density<EOSPolytrope>(rho, eos);
   const amrex::Real compactness_factor = 1.0 - 2.0 * state.mass / radius;
   if (!(compactness_factor > 0.0)) {
      throw std::runtime_error("TOV integration crossed an apparent horizon");
   }
   const amrex::Real gravity = (
      state.mass + constants::four_pi * radius * radius * radius * state.pressure
   ) / (radius * radius * compactness_factor);
   return {
      constants::four_pi * radius * radius * energy_density,
      -(energy_density + state.pressure) * gravity,
      gravity,
      (1.0 / std::sqrt(compactness_factor) - 1.0) / radius,
   };
}

TovIntegrationState add_scaled(
   TovIntegrationState const& state,
   TovIntegrationState const& derivative,
   amrex::Real scale
)
{
   return {
      state.mass + scale * derivative.mass,
      state.pressure + scale * derivative.pressure,
      state.log_lapse + scale * derivative.log_lapse,
      state.log_isotropic_ratio + scale * derivative.log_isotropic_ratio,
   };
}

} // namespace

TovInitialData::TovInitialData(RunConfig const& cfg)
{
   if (cfg.ic != "tov") {
      return;
   }
   const EOSParams eos = eos_params(cfg);
   const amrex::Real central_density = cfg.tov_central_density;
   const amrex::Real central_pressure = eos_cold_pressure<EOSPolytrope>(
      central_density, eos
   );
   const amrex::Real central_energy_density = central_density
      + eos_cold_energy_density<EOSPolytrope>(central_density, eos);
   const amrex::Real surface_density = std::max(
      cfg.rho_floor, central_density * 1.0e-10
   );
   const amrex::Real surface_pressure = eos_cold_pressure<EOSPolytrope>(
      surface_density, eos
   );
   const amrex::Real dr = cfg.tov_radial_step;

   std::vector<amrex::Real> areal_radius;
   std::vector<TovIntegrationState> states;
   areal_radius.reserve(cfg.tov_radial_points);
   states.reserve(cfg.tov_radial_points);
   areal_radius.push_back(0.0);
   states.push_back({0.0, central_pressure, 0.0, 0.0});

   const amrex::Real first_radius = dr;
   const amrex::Real first_mass = constants::four_pi * central_energy_density
      * first_radius * first_radius * first_radius / 3.0;
   const amrex::Real central_gravity_coefficient = constants::two_pi * (
      central_energy_density / 3.0 + central_pressure
   );
   TovIntegrationState state{
      first_mass,
      central_pressure - central_gravity_coefficient
         * (central_energy_density + central_pressure) * first_radius * first_radius,
      central_gravity_coefficient * first_radius * first_radius,
      constants::two_pi * central_energy_density * first_radius * first_radius / 3.0,
   };
   amrex::Real radius = first_radius;

   for (int point = 1; point < cfg.tov_radial_points; ++point) {
      if (!(state.pressure > surface_pressure)) {
         break;
      }
      areal_radius.push_back(radius);
      states.push_back(state);
      const TovIntegrationState k1 = tov_derivative(radius, state, eos, surface_pressure);
      const TovIntegrationState k2 = tov_derivative(
         radius + 0.5 * dr, add_scaled(state, k1, 0.5 * dr), eos, surface_pressure
      );
      const TovIntegrationState k3 = tov_derivative(
         radius + 0.5 * dr, add_scaled(state, k2, 0.5 * dr), eos, surface_pressure
      );
      const TovIntegrationState k4 = tov_derivative(
         radius + dr, add_scaled(state, k3, dr), eos, surface_pressure
      );
      state.mass += dr * (k1.mass + 2.0 * k2.mass + 2.0 * k3.mass + k4.mass) / 6.0;
      state.pressure += dr * (
         k1.pressure + 2.0 * k2.pressure + 2.0 * k3.pressure + k4.pressure
      ) / 6.0;
      state.log_lapse += dr * (
         k1.log_lapse + 2.0 * k2.log_lapse + 2.0 * k3.log_lapse + k4.log_lapse
      ) / 6.0;
      state.log_isotropic_ratio += dr * (
         k1.log_isotropic_ratio + 2.0 * k2.log_isotropic_ratio
         + 2.0 * k3.log_isotropic_ratio + k4.log_isotropic_ratio
      ) / 6.0;
      radius += dr;
   }
   if (states.size() < 3 || state.pressure > surface_pressure) {
      throw std::runtime_error(
         "TOV integration did not reach the stellar surface; increase pyfluid.tov_radial_points"
      );
   }

   gravitational_mass_ = states.back().mass;
   const amrex::Real areal_surface = areal_radius.back();
   if (!(areal_surface > 2.0 * gravitational_mass_)) {
      throw std::runtime_error("TOV surface lies inside its Schwarzschild radius");
   }
   surface_radius_ = 0.5 * (
      areal_surface - gravitational_mass_
      + std::sqrt(areal_surface * (areal_surface - 2.0 * gravitational_mass_))
   );
   const amrex::Real unscaled_surface = areal_surface
      * std::exp(states.back().log_isotropic_ratio);
   const amrex::Real isotropic_scale = surface_radius_ / unscaled_surface;
   const amrex::Real surface_lapse = std::sqrt(
      1.0 - 2.0 * gravitational_mass_ / areal_surface
   );
   const amrex::Real lapse_shift = std::log(surface_lapse) - states.back().log_lapse;

   host_samples_.resize(states.size());
   for (std::size_t index = 1; index < states.size(); ++index) {
      const amrex::Real rho = cold_density_from_pressure(states[index].pressure, eos);
      const amrex::Real isotropic_radius = isotropic_scale * areal_radius[index]
         * std::exp(states[index].log_isotropic_ratio);
      host_samples_[index] = {
         isotropic_radius,
         rho,
         states[index].pressure,
         eos_cold_energy_density<EOSPolytrope>(rho, eos) / rho,
         std::exp(states[index].log_lapse + lapse_shift),
         std::sqrt(areal_radius[index] / isotropic_radius),
      };
   }
   host_samples_[0] = host_samples_[1];
   host_samples_[0].isotropic_radius = 0.0;

   device_samples_.resize(host_samples_.size());
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      host_samples_.begin(),
      host_samples_.end(),
      device_samples_.begin()
   );
   amrex::Print() << "TOV initial data: M=" << gravitational_mass_
                  << " R_areal=" << areal_surface
                  << " R_isotropic=" << surface_radius_
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
