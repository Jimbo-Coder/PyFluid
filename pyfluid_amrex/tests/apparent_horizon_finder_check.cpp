#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyfluid_amrex/apparent_horizon_finder.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"

namespace {

using namespace amrex::literals;

constexpr amrex::Real pi = std::numbers::pi_v<amrex::Real>;

bool check_geometry_reflection_parity()
{
   constexpr std::array<std::array<int, 2>, 6> symmetric_indices{{
      {{0, 0}}, {{0, 1}}, {{0, 2}},
      {{1, 1}}, {{1, 2}}, {{2, 2}},
   }};
   pyfluid_amrex::ApparentHorizonGeometrySample original{};
   for (int component = 0;
        component < pyfluid_amrex::ApparentHorizonGeometryComponents;
        ++component) {
      original[component] = static_cast<amrex::Real>(component + 1);
   }
   for (int reflection_mask = 0; reflection_mask < 8; ++reflection_mask) {
      std::array<int, 3> direction_parity{{1, 1, 1}};
      for (int direction = 0; direction < 3; ++direction) {
         if ((reflection_mask & (1 << direction)) != 0) {
            direction_parity[direction] = -1;
         }
      }
      auto reflected = original;
      pyfluid_amrex::reflect_apparent_horizon_geometry(
         reflected, reflection_mask
      );
      const auto expected_rank_two = [&](int first_component) {
         for (int component = 0; component < 6; ++component) {
            const auto indices = symmetric_indices[component];
            const amrex::Real expected = original[first_component + component]
               * direction_parity[indices[0]]
               * direction_parity[indices[1]];
            if (reflected[first_component + component] != expected) {
               return false;
            }
         }
         return true;
      };
      if (!expected_rank_two(pyfluid_amrex::AhGamma11)
          || !expected_rank_two(pyfluid_amrex::AhInverseGamma11)
          || !expected_rank_two(pyfluid_amrex::AhExtrinsic11)) {
         std::cerr << "apparent-horizon rank-two reflection parity mismatch\n";
         return false;
      }
      for (int derivative_direction = 0;
           derivative_direction < 3;
           ++derivative_direction) {
         const int first_component = pyfluid_amrex::AhD1Gamma11
            + 6 * derivative_direction;
         for (int component = 0; component < 6; ++component) {
            const auto indices = symmetric_indices[component];
            const amrex::Real expected = original[first_component + component]
               * direction_parity[derivative_direction]
               * direction_parity[indices[0]]
               * direction_parity[indices[1]];
            if (reflected[first_component + component] != expected) {
               std::cerr
                  << "apparent-horizon derivative reflection parity mismatch\n";
               return false;
            }
         }
      }
      pyfluid_amrex::reflect_apparent_horizon_geometry(
         reflected, reflection_mask
      );
      if (reflected != original) {
         std::cerr << "apparent-horizon reflection is not an involution\n";
         return false;
      }
   }
   return true;
}

pyfluid_amrex::ApparentHorizonGeometryGrid make_kerr_geometry_grid(
   pyfluid_amrex::FixedSpacetimeParameters const& parameters,
   int num_radial,
   int num_theta
)
{
   const int num_phi = 2 * num_theta;
   constexpr amrex::Real minimum_radius = 0.55_rt;
   constexpr amrex::Real maximum_radius = 2.75_rt;
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
   const amrex::Real dphi = 2.0_rt * pi / num_phi;
   for (int radial_index = 0; radial_index < num_radial; ++radial_index) {
      for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
         const amrex::Real theta = (theta_index + 0.5_rt) * dtheta;
         for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
            const amrex::Real phi = (phi_index + 0.5_rt) * dphi;
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
   return 1.0_rt / std::sqrt(
      std::sin(theta) * std::sin(theta) / equatorial_radius_squared
      + std::cos(theta) * std::cos(theta)
         / (kerr_radius * kerr_radius)
   );
}

bool check_spinning_kerr_horizon(int target_num_theta)
{
   pyfluid_amrex::FixedSpacetimeParameters parameters;
   parameters.metric = pyfluid_amrex::FixedSpacetimeMetric::KerrSchild;
   parameters.mass = 1.0_rt;
   parameters.origin1 = 0.13_rt;
   parameters.origin2 = -0.07_rt;
   parameters.origin3 = 0.09_rt;
   parameters.dimensionless_spin3 = 0.6_rt;
   const auto geometry = make_kerr_geometry_grid(
      parameters, std::max(64, 2 * target_num_theta),
      std::max(32, target_num_theta)
   );

   pyfluid_amrex::ApparentHorizonFinderParameters finder_parameters;
   finder_parameters.coarsest_num_theta = 8;
   finder_parameters.target_num_theta = target_num_theta;
   finder_parameters.max_iterations_per_level = 20000;
   finder_parameters.theta_linf_times_mass_tolerance = 2.0e-5_rt;
   const auto solve_start = std::chrono::steady_clock::now();
   const auto result = pyfluid_amrex::find_apparent_horizon(
      geometry, 2.25_rt, finder_parameters
   );
   const auto solve_end = std::chrono::steady_clock::now();
   const double solve_seconds = std::chrono::duration<double>(
      solve_end - solve_start
   ).count();
   if (!result.converged) {
      std::cerr << "Kerr horizon finder failed after " << result.iterations
                << " iterations: " << result.failure_reason
                << " Linf=" << result.theta_linf_times_mass
                << " L2=" << result.theta_l2_times_mass << '\n';
      return false;
   }
   amrex::Real maximum_radius_error = 0.0_rt;
   const amrex::Real dtheta = pi / result.surface.num_theta();
   for (int theta_index = 0;
        theta_index < result.surface.num_theta();
        ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5_rt) * dtheta;
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
   const amrex::Real exact_area = 8.0_rt * pi * parameters.mass * horizon_radius;
   const amrex::Real area_relative_error = std::abs(
      result.area / exact_area - 1.0_rt
   );
   const amrex::Real kerr_mass_relative_error = std::abs(
      result.kerr_horizon_mass / parameters.mass - 1.0_rt
   );
   const amrex::Real kerr_spin_error = std::abs(
      result.kerr_dimensionless_spin
      - std::abs(parameters.dimensionless_spin3)
   );
   const amrex::Real quasi_local_spin_error = std::abs(
      result.christodoulou_dimensionless_spin[2]
      - parameters.dimensionless_spin3
   );
   const amrex::Real transverse_quasi_local_spin = std::hypot(
      result.christodoulou_dimensionless_spin[0],
      result.christodoulou_dimensionless_spin[1]
   );
   const amrex::Real christodoulou_mass_relative_error = std::abs(
      result.christodoulou_mass / parameters.mass - 1.0_rt
   );
   const amrex::Real centroid_error = std::sqrt(
      (result.centroid[0] - parameters.origin1)
         * (result.centroid[0] - parameters.origin1)
      + (result.centroid[1] - parameters.origin2)
         * (result.centroid[1] - parameters.origin2)
      + (result.centroid[2] - parameters.origin3)
         * (result.centroid[2] - parameters.origin3)
   );
   if (!(result.theta_linf_times_mass <= 2.0e-5_rt)
       || maximum_radius_error > 9.0e-3_rt
       || area_relative_error > 8.0e-3_rt
       || kerr_mass_relative_error > 2.0e-2_rt
       || kerr_spin_error > 7.0e-2_rt
       || quasi_local_spin_error > 3.0e-2_rt
       || transverse_quasi_local_spin > 2.0e-3_rt
       || christodoulou_mass_relative_error > 2.0e-2_rt
       || !(result.polar_circumference
            < result.equatorial_circumference)
       || centroid_error > 2.0e-3_rt
       || !(result.surface.signed_radial_distance(
               parameters.origin1, parameters.origin2, parameters.origin3
            ) < 0.0_rt)
       || !(result.surface.signed_radial_distance(
               parameters.origin1 + 3.0_rt,
               parameters.origin2,
               parameters.origin3
            ) > 0.0_rt)) {
      std::cerr << "Kerr horizon mismatch: iterations=" << result.iterations
                << " Linf=" << result.theta_linf_times_mass
                << " L2=" << result.theta_l2_times_mass
                << " radius_error=" << maximum_radius_error
                << " area_error=" << area_relative_error
                << " kerr_mass_error=" << kerr_mass_relative_error
                << " kerr_spin_error=" << kerr_spin_error
                << " quasi_local_spin_error=" << quasi_local_spin_error
                << " transverse_quasi_local_spin="
                << transverse_quasi_local_spin
                << " christodoulou_mass_error="
                << christodoulou_mass_relative_error
                << " centroid_error=" << centroid_error << '\n';
      return false;
   }
   std::cout << "Kerr AH: iterations=" << result.iterations
             << " polar_points=" << target_num_theta
             << " solve_seconds=" << solve_seconds
             << " Linf=" << result.theta_linf_times_mass
             << " radius_error=" << maximum_radius_error
             << " area_error=" << area_relative_error
             << " kerr_mass_error=" << kerr_mass_relative_error
             << " kerr_spin_error=" << kerr_spin_error
             << " quasi_local_spin_error=" << quasi_local_spin_error
             << " transverse_quasi_local_spin="
             << transverse_quasi_local_spin
             << " christodoulou_mass_error="
             << christodoulou_mass_relative_error << '\n';
   return true;
}

} // namespace

int main(int argc, char** argv)
{
   int target_num_theta = 16;
   if (argc == 3 && std::string(argv[1]) == "--polar-points") {
      target_num_theta = std::stoi(argv[2]);
   } else if (argc != 1) {
      throw std::invalid_argument(
         "usage: pyfluid_amrex_apparent_horizon_check "
         "[--polar-points EVEN_INTEGER]"
      );
   }
   if (!check_geometry_reflection_parity()
       || !check_spinning_kerr_horizon(target_num_theta)) {
      return 1;
   }
   std::cout << "PASS apparent-horizon finder check\n";
   return 0;
}
