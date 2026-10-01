#include "pyfluid_amrex/surface_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/sampled_sphere.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>>
gauss_legendre_rule(int points)
{
   std::vector<amrex::Real> nodes(points);
   std::vector<amrex::Real> weights(points);
   const int roots = (points + 1) / 2;
   for (int root = 0; root < roots; ++root) {
      amrex::Real x = std::cos(
         constants::pi * (static_cast<amrex::Real>(root) + 0.75_rt)
         / (static_cast<amrex::Real>(points) + 0.5_rt)
      );
      amrex::Real derivative = 0.0_rt;
      for (int iteration = 0; iteration < 32; ++iteration) {
         amrex::Real p0 = 1.0_rt;
         amrex::Real p1 = x;
         for (int order = 2; order <= points; ++order) {
            const amrex::Real current = (
               (2.0_rt * order - 1.0_rt) * x * p1 - (order - 1.0_rt) * p0
            ) / order;
            p0 = p1;
            p1 = current;
         }
         derivative = points * (x * p1 - p0) / (x * x - 1.0_rt);
         const amrex::Real update = p1 / derivative;
         x -= update;
         if (std::abs(update) < 8.0_rt * std::numeric_limits<amrex::Real>::epsilon()) {
            break;
         }
      }
      const amrex::Real weight = 2.0_rt / ((1.0_rt - x * x) * derivative * derivative);
      nodes[root] = -x;
      nodes[points - 1 - root] = x;
      weights[root] = weight;
      weights[points - 1 - root] = weight;
   }
   return {std::move(nodes), std::move(weights)};
}

} // namespace

ParametricSurface ParametricSurface::sphere(
   amrex::GpuArray<amrex::Real, 3> center,
   amrex::Real radius,
   int polar_points
)
{
   if (!(radius > 0.0_rt) || polar_points < 2) {
      throw std::invalid_argument("a spherical surface requires radius > 0 and at least two polar points");
   }
   const int azimuthal_points = 2 * polar_points;
   const auto [mu, mu_weights] = gauss_legendre_rule(polar_points);
   const amrex::Real dphi = constants::two_pi / azimuthal_points;
   std::vector<SurfaceQuadraturePoint> points;
   points.reserve(polar_points * azimuthal_points);
   for (int polar = 0; polar < polar_points; ++polar) {
      const amrex::Real sin_theta = std::sqrt(amrex::max(0.0_rt, 1.0_rt - mu[polar] * mu[polar]));
      const amrex::Real theta = std::acos(mu[polar]);
      const amrex::Real area_weight = radius * radius * mu_weights[polar] * dphi;
      const amrex::Real solid_angle_weight = mu_weights[polar] * dphi;
      for (int azimuthal = 0; azimuthal < azimuthal_points; ++azimuthal) {
         const amrex::Real phi = (azimuthal + 0.5_rt) * dphi;
         const amrex::GpuArray<amrex::Real, 3> normal{{
            sin_theta * std::cos(phi),
            sin_theta * std::sin(phi),
            mu[polar],
         }};
         points.push_back({
            {
               center[0] + radius * normal[0],
               center[1] + radius * normal[1],
               center[2] + radius * normal[2],
            },
            normal,
            {
               area_weight * normal[0],
               area_weight * normal[1],
               area_weight * normal[2],
            },
            theta,
            phi,
            solid_angle_weight,
         });
      }
   }
   return ParametricSurface(center, radius, std::move(points));
}

ParametricSurface ParametricSurface::radial_graph(
   amrex::GpuArray<amrex::Real, 3> center,
   int num_theta,
   int num_phi,
   std::vector<amrex::Real> radius
)
{
   const bool finite_positive_radius = std::all_of(
      radius.begin(), radius.end(), [](amrex::Real value) {
         return std::isfinite(value) && value > 0.0_rt;
      }
   );
   if (num_theta < 4 || num_phi < 4 || num_phi % 2 != 0
       || radius.size() != static_cast<std::size_t>(num_theta * num_phi)
       || !finite_positive_radius) {
      throw std::invalid_argument(
         "a radial-graph surface requires a finite positive even-phi angular grid"
      );
   }

   constexpr int offsets[7]{-3, -2, -1, 0, 1, 2, 3};
   constexpr amrex::Real first[7]{
      -1.0_rt, 9.0_rt, -45.0_rt, 0.0_rt, 45.0_rt, -9.0_rt, 1.0_rt
   };
   const amrex::Real dtheta = constants::pi / num_theta;
   const amrex::Real dphi = constants::two_pi / num_phi;
   std::vector<SurfaceQuadraturePoint> points;
   points.reserve(radius.size());
   amrex::Real mean_radius = 0.0_rt;
   amrex::Real total_solid_angle = 0.0_rt;
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5_rt) * dtheta;
      const amrex::Real sin_theta = std::sin(theta);
      const amrex::Real cos_theta = std::cos(theta);
      const amrex::Real solid_angle_weight = sin_theta * dtheta * dphi;
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5_rt) * dphi;
         const std::size_t index = static_cast<std::size_t>(theta_index)
            * num_phi + phi_index;
         amrex::Real radius_theta = 0.0_rt;
         amrex::Real radius_phi = 0.0_rt;
         for (int stencil = 0; stencil < 7; ++stencil) {
            radius_theta += first[stencil]
               * sampled_sphere_detail::surface_value(
                  radius,
                  theta_index + offsets[stencil],
                  phi_index,
                  num_theta,
                  num_phi
               );
            radius_phi += first[stencil]
               * sampled_sphere_detail::surface_value(
                  radius,
                  theta_index,
                  phi_index + offsets[stencil],
                  num_theta,
                  num_phi
               );
         }
         radius_theta /= 60.0_rt * dtheta;
         radius_phi /= 60.0_rt * dphi;

         const amrex::Real sin_phi = std::sin(phi);
         const amrex::Real cos_phi = std::cos(phi);
         const amrex::GpuArray<amrex::Real, 3> radial_unit{{
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
         }};
         const amrex::GpuArray<amrex::Real, 3> polar_unit{{
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta,
         }};
         const amrex::GpuArray<amrex::Real, 3> azimuthal_unit{{
            -sin_phi,
            cos_phi,
            0.0_rt,
         }};
         const amrex::Real point_radius = radius[index];
         const amrex::Real radial_area = point_radius * point_radius * sin_theta;
         const amrex::Real polar_area = -point_radius * radius_theta * sin_theta;
         const amrex::Real azimuthal_area = -point_radius * radius_phi;
         amrex::GpuArray<amrex::Real, 3> area_vector{};
         for (int direction = 0; direction < 3; ++direction) {
            area_vector[direction] = dtheta * dphi * (
               radial_area * radial_unit[direction]
               + polar_area * polar_unit[direction]
               + azimuthal_area * azimuthal_unit[direction]
            );
         }
         points.push_back({
            {
               center[0] + point_radius * radial_unit[0],
               center[1] + point_radius * radial_unit[1],
               center[2] + point_radius * radial_unit[2],
            },
            radial_unit,
            area_vector,
            theta,
            phi,
            solid_angle_weight,
         });
         mean_radius += solid_angle_weight * point_radius;
         total_solid_angle += solid_angle_weight;
      }
   }
   mean_radius /= total_solid_angle;
   return ParametricSurface(center, mean_radius, std::move(points));
}

std::vector<amrex::Real> ParametricSurface::integrate_normal_fluxes(
   std::vector<amrex::Real> const& sampled_fluxes,
   int num_integrands
) const
{
   if (num_integrands <= 0
       || sampled_fluxes.size() != points_.size() * 3 * num_integrands) {
      throw std::invalid_argument(
         "surface flux samples must contain three components per integrand and point"
      );
   }
   std::vector<amrex::Real> integrals(num_integrands, 0.0_rt);
   for (std::size_t point = 0; point < points_.size(); ++point) {
      for (int integrand = 0; integrand < num_integrands; ++integrand) {
         const std::size_t component = 3 * (
            point * static_cast<std::size_t>(num_integrands) + integrand
         );
         for (int direction = 0; direction < 3; ++direction) {
            integrals[integrand] += points_[point].area_vector[direction]
               * sampled_fluxes[component + direction];
         }
      }
   }
   return integrals;
}

} // namespace pyfluid_amrex
