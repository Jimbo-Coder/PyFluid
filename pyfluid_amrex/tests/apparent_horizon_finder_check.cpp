#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "pyfluid_amrex/apparent_horizon_finder.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"

namespace {

constexpr amrex::Real pi = std::numbers::pi_v<amrex::Real>;

pyfluid_amrex::ApparentHorizonGeometryGrid make_kerr_geometry_grid(
   pyfluid_amrex::FixedSpacetimeParameters const& parameters,
   int num_radial,
   int num_theta
)
{
   const int num_phi = 2 * num_theta;
   constexpr amrex::Real minimum_radius = 0.55;
   constexpr amrex::Real maximum_radius = 2.75;
   std::vector<amrex::Real> radial(num_radial);
   for (int radial_index = 0; radial_index < num_radial; ++radial_index) {
      radial[radial_index] = minimum_radius
         + (maximum_radius - minimum_radius) * radial_index
            / (num_radial - 1);
   }
   std::vector<pyfluid_amrex::ApparentHorizonGeometrySample> samples(
      static_cast<std::size_t>(num_radial * num_theta * num_phi)
   );
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   for (int radial_index = 0; radial_index < num_radial; ++radial_index) {
      for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
         const amrex::Real theta = (theta_index + 0.5) * dtheta;
         for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
            const amrex::Real phi = (phi_index + 0.5) * dphi;
            const amrex::Real x = parameters.origin1
               + radial[radial_index] * std::sin(theta) * std::cos(phi);
            const amrex::Real y = parameters.origin2
               + radial[radial_index] * std::sin(theta) * std::sin(phi);
            const amrex::Real z = parameters.origin3
               + radial[radial_index] * std::cos(theta);
            const auto geometry =
               pyfluid_amrex::exact_kerr_schild_initial_data_geometry(
                  parameters, x, y, z
               );
            const std::size_t index = (
               static_cast<std::size_t>(radial_index) * num_theta
               + theta_index
            ) * num_phi + phi_index;
            samples[index] = pyfluid_amrex::pack_apparent_horizon_geometry(
               geometry.metric, geometry.derivatives
            );
         }
      }
   }
   return {
      {parameters.origin1, parameters.origin2, parameters.origin3},
      std::move(radial),
      num_theta,
      num_phi,
      std::move(samples),
   };
}

amrex::Real exact_kerr_euclidean_horizon_radius(
   pyfluid_amrex::FixedSpacetimeParameters const& parameters,
   amrex::Real theta
)
{
   const amrex::Real kerr_radius =
      pyfluid_amrex::kerr_schild_outer_horizon_radius(parameters);
   const amrex::Real spin_length = parameters.mass
      * parameters.dimensionless_spin3;
   const amrex::Real equatorial_radius_squared =
      kerr_radius * kerr_radius + spin_length * spin_length;
   return 1.0 / std::sqrt(
      std::sin(theta) * std::sin(theta) / equatorial_radius_squared
      + std::cos(theta) * std::cos(theta)
         / (kerr_radius * kerr_radius)
   );
}

bool check_spinning_kerr_horizon()
{
   pyfluid_amrex::FixedSpacetimeParameters parameters;
   parameters.metric = pyfluid_amrex::FixedSpacetimeMetric::KerrSchild;
   parameters.mass = 1.0;
   parameters.origin1 = 0.13;
   parameters.origin2 = -0.07;
   parameters.origin3 = 0.09;
   parameters.dimensionless_spin3 = 0.6;
   const auto geometry = make_kerr_geometry_grid(parameters, 64, 32);

   pyfluid_amrex::ApparentHorizonFinderParameters finder_parameters;
   finder_parameters.coarsest_num_theta = 8;
   finder_parameters.target_num_theta = 16;
   finder_parameters.max_iterations_per_level = 6000;
   finder_parameters.theta_linf_times_mass_tolerance = 2.0e-5;
   const auto result = pyfluid_amrex::find_apparent_horizon(
      geometry, 2.25, finder_parameters
   );
   if (!result.converged) {
      std::cerr << "Kerr horizon finder failed after " << result.iterations
                << " iterations: " << result.failure_reason
                << " Linf=" << result.theta_linf_times_mass
                << " L2=" << result.theta_l2_times_mass << '\n';
      return false;
   }
   amrex::Real maximum_radius_error = 0.0;
   const amrex::Real dtheta = pi / result.surface.num_theta();
   for (int theta_index = 0;
        theta_index < result.surface.num_theta();
        ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5) * dtheta;
      const amrex::Real exact_radius = exact_kerr_euclidean_horizon_radius(
         parameters, theta
      );
      for (int phi_index = 0;
           phi_index < result.surface.num_phi();
           ++phi_index) {
         const std::size_t index = static_cast<std::size_t>(theta_index)
            * result.surface.num_phi() + phi_index;
         maximum_radius_error = std::max(
            maximum_radius_error,
            std::abs(result.surface.radius()[index] - exact_radius)
         );
      }
   }
   const amrex::Real horizon_radius =
      pyfluid_amrex::kerr_schild_outer_horizon_radius(parameters);
   const amrex::Real exact_area = 8.0 * pi * parameters.mass * horizon_radius;
   const amrex::Real area_relative_error = std::abs(
      result.area / exact_area - 1.0
   );
   const amrex::Real centroid_error = std::sqrt(
      (result.centroid[0] - parameters.origin1)
         * (result.centroid[0] - parameters.origin1)
      + (result.centroid[1] - parameters.origin2)
         * (result.centroid[1] - parameters.origin2)
      + (result.centroid[2] - parameters.origin3)
         * (result.centroid[2] - parameters.origin3)
   );
   if (!(result.theta_linf_times_mass <= 2.0e-5)
       || maximum_radius_error > 9.0e-3
       || area_relative_error > 8.0e-3
       || centroid_error > 2.0e-3
       || !(result.surface.signed_radial_distance(
               parameters.origin1, parameters.origin2, parameters.origin3
            ) < 0.0)
       || !(result.surface.signed_radial_distance(
               parameters.origin1 + 3.0,
               parameters.origin2,
               parameters.origin3
            ) > 0.0)) {
      std::cerr << "Kerr horizon mismatch: iterations=" << result.iterations
                << " Linf=" << result.theta_linf_times_mass
                << " L2=" << result.theta_l2_times_mass
                << " radius_error=" << maximum_radius_error
                << " area_error=" << area_relative_error
                << " centroid_error=" << centroid_error << '\n';
      return false;
   }
   std::cout << "Kerr AH: iterations=" << result.iterations
             << " Linf=" << result.theta_linf_times_mass
             << " radius_error=" << maximum_radius_error
             << " area_error=" << area_relative_error << '\n';
   return true;
}

} // namespace

int main()
{
   if (!check_spinning_kerr_horizon()) {
      return 1;
   }
   std::cout << "PASS apparent-horizon finder check\n";
   return 0;
}
