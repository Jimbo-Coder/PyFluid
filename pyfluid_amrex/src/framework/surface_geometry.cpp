#include "pyfluid_amrex/surface_geometry.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "pyfluid_amrex/constants.hpp"

namespace pyfluid_amrex {
namespace {

std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>>
gauss_legendre_rule(int points)
{
   std::vector<amrex::Real> nodes(points);
   std::vector<amrex::Real> weights(points);
   const int roots = (points + 1) / 2;
   for (int root = 0; root < roots; ++root) {
      amrex::Real x = std::cos(
         constants::pi * (static_cast<amrex::Real>(root) + 0.75)
         / (static_cast<amrex::Real>(points) + 0.5)
      );
      amrex::Real derivative = 0.0;
      for (int iteration = 0; iteration < 32; ++iteration) {
         amrex::Real p0 = 1.0;
         amrex::Real p1 = x;
         for (int order = 2; order <= points; ++order) {
            const amrex::Real current = (
               (2.0 * order - 1.0) * x * p1 - (order - 1.0) * p0
            ) / order;
            p0 = p1;
            p1 = current;
         }
         derivative = points * (x * p1 - p0) / (x * x - 1.0);
         const amrex::Real update = p1 / derivative;
         x -= update;
         if (std::abs(update) < 8.0 * std::numeric_limits<amrex::Real>::epsilon()) {
            break;
         }
      }
      const amrex::Real weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
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
   if (!(radius > 0.0) || polar_points < 2) {
      throw std::invalid_argument("a spherical surface requires radius > 0 and at least two polar points");
   }
   const int azimuthal_points = 2 * polar_points;
   const auto [mu, mu_weights] = gauss_legendre_rule(polar_points);
   const amrex::Real dphi = constants::two_pi / azimuthal_points;
   std::vector<SurfaceQuadraturePoint> points;
   points.reserve(polar_points * azimuthal_points);
   for (int polar = 0; polar < polar_points; ++polar) {
      const amrex::Real sin_theta = std::sqrt(amrex::max(0.0, 1.0 - mu[polar] * mu[polar]));
      const amrex::Real theta = std::acos(mu[polar]);
      const amrex::Real area_weight = radius * radius * mu_weights[polar] * dphi;
      const amrex::Real solid_angle_weight = mu_weights[polar] * dphi;
      for (int azimuthal = 0; azimuthal < azimuthal_points; ++azimuthal) {
         const amrex::Real phi = (azimuthal + 0.5) * dphi;
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
   std::vector<amrex::Real> integrals(num_integrands, 0.0);
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
