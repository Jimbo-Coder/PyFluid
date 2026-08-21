#include "pyfluid_amrex/apparent_horizon_finder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "pyfluid_amrex/apparent_horizon.hpp"

namespace pyfluid_amrex {
namespace {

constexpr amrex::Real pi = std::numbers::pi_v<amrex::Real>;

int wrap_index(int index, int count) noexcept
{
   index %= count;
   return index < 0 ? index + count : index;
}

struct AngularIndex {
   int theta = 0;
   int phi_shift = 0;
};

AngularIndex map_theta_index(int index, int num_theta, int num_phi) noexcept
{
   int shift = 0;
   while (index < 0 || index >= num_theta) {
      if (index < 0) {
         index = -index - 1;
         shift += num_phi / 2;
      }
      if (index >= num_theta) {
         index = 2 * num_theta - index - 1;
         shift += num_phi / 2;
      }
   }
   return {index, shift};
}

void canonicalize_angles(amrex::Real& theta, amrex::Real& phi) noexcept
{
   while (theta < 0.0 || theta > pi) {
      if (theta < 0.0) {
         theta = -theta;
         phi += pi;
      }
      if (theta > pi) {
         theta = 2.0 * pi - theta;
         phi += pi;
      }
   }
   phi = std::fmod(phi, 2.0 * pi);
   if (phi < 0.0) {
      phi += 2.0 * pi;
   }
}

std::size_t surface_index(int theta, int phi, int num_phi) noexcept
{
   return static_cast<std::size_t>(theta) * num_phi + phi;
}

amrex::Real surface_value(
   std::vector<amrex::Real> const& values,
   int theta,
   int phi,
   int num_theta,
   int num_phi
) noexcept
{
   const AngularIndex mapped = map_theta_index(theta, num_theta, num_phi);
   return values[surface_index(
      mapped.theta, wrap_index(phi + mapped.phi_shift, num_phi), num_phi
   )];
}

amrex::Real interpolate_surface(
   std::vector<amrex::Real> const& values,
   int num_theta,
   int num_phi,
   amrex::Real theta,
   amrex::Real phi
)
{
   canonicalize_angles(theta, phi);
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   const amrex::Real theta_index = theta / dtheta - 0.5;
   const amrex::Real phi_index = phi / dphi - 0.5;
   const int theta_lower = static_cast<int>(std::floor(theta_index));
   const int phi_lower = static_cast<int>(std::floor(phi_index));
   const amrex::Real theta_fraction = theta_index - theta_lower;
   const amrex::Real phi_fraction = phi_index - phi_lower;
   amrex::Real result = 0.0;
   for (int theta_offset = 0; theta_offset < 2; ++theta_offset) {
      const amrex::Real theta_weight = theta_offset == 0
         ? 1.0 - theta_fraction : theta_fraction;
      const AngularIndex mapped = map_theta_index(
         theta_lower + theta_offset, num_theta, num_phi
      );
      for (int phi_offset = 0; phi_offset < 2; ++phi_offset) {
         const amrex::Real phi_weight = phi_offset == 0
            ? 1.0 - phi_fraction : phi_fraction;
         const int mapped_phi = wrap_index(
            phi_lower + phi_offset + mapped.phi_shift, num_phi
         );
         result += theta_weight * phi_weight
            * values[surface_index(mapped.theta, mapped_phi, num_phi)];
      }
   }
   return result;
}

struct SurfaceDerivatives {
   amrex::Real theta = 0.0;
   amrex::Real phi = 0.0;
   amrex::Real theta_theta = 0.0;
   amrex::Real theta_phi = 0.0;
   amrex::Real phi_phi = 0.0;
};

SurfaceDerivatives surface_derivatives(
   std::vector<amrex::Real> const& radius,
   int theta,
   int phi,
   int num_theta,
   int num_phi
) noexcept
{
   constexpr int offsets[7]{-3, -2, -1, 0, 1, 2, 3};
   constexpr amrex::Real first[7]{-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0};
   constexpr amrex::Real second[7]{2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0};
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   SurfaceDerivatives derivatives;
   for (int stencil = 0; stencil < 7; ++stencil) {
      derivatives.theta += first[stencil] * surface_value(
         radius, theta + offsets[stencil], phi, num_theta, num_phi
      );
      derivatives.phi += first[stencil] * surface_value(
         radius, theta, phi + offsets[stencil], num_theta, num_phi
      );
      derivatives.theta_theta += second[stencil] * surface_value(
         radius, theta + offsets[stencil], phi, num_theta, num_phi
      );
      derivatives.phi_phi += second[stencil] * surface_value(
         radius, theta, phi + offsets[stencil], num_theta, num_phi
      );
   }
   derivatives.theta /= 60.0 * dtheta;
   derivatives.phi /= 60.0 * dphi;
   derivatives.theta_theta /= 180.0 * dtheta * dtheta;
   derivatives.phi_phi /= 180.0 * dphi * dphi;
   for (int theta_stencil = 0; theta_stencil < 7; ++theta_stencil) {
      for (int phi_stencil = 0; phi_stencil < 7; ++phi_stencil) {
         derivatives.theta_phi += first[theta_stencil] * first[phi_stencil]
            * surface_value(
               radius,
               theta + offsets[theta_stencil],
               phi + offsets[phi_stencil],
               num_theta,
               num_phi
            );
      }
   }
   derivatives.theta_phi /= 3600.0 * dtheta * dphi;
   return derivatives;
}

void add_outer_product(
   amrex::Real output[3][3],
   amrex::Real coefficient,
   amrex::Real const first[3],
   amrex::Real const second[3]
) noexcept
{
   for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
         output[row][column] += coefficient * first[row] * second[column];
      }
   }
}

void radial_graph_level_set_geometry(
   amrex::Real radius,
   amrex::Real theta,
   amrex::Real phi,
   SurfaceDerivatives const& derivatives,
   amrex::Real gradient[3],
   amrex::Real hessian[3][3]
) noexcept
{
   const amrex::Real sin_theta = std::sin(theta);
   const amrex::Real cos_theta = std::cos(theta);
   const amrex::Real sin_phi = std::sin(phi);
   const amrex::Real cos_phi = std::cos(phi);
   const amrex::Real radial[3]{
      sin_theta * cos_phi,
      sin_theta * sin_phi,
      cos_theta,
   };
   const amrex::Real polar[3]{
      cos_theta * cos_phi,
      cos_theta * sin_phi,
      -sin_theta,
   };
   const amrex::Real azimuthal[3]{-sin_phi, cos_phi, 0.0};
   const amrex::Real inverse_radius = 1.0 / radius;
   const amrex::Real inverse_sin_theta = 1.0 / sin_theta;
   amrex::Real gradient_theta[3]{};
   amrex::Real gradient_phi[3]{};
   for (int direction = 0; direction < 3; ++direction) {
      gradient_theta[direction] = polar[direction] * inverse_radius;
      gradient_phi[direction] = azimuthal[direction]
         * inverse_radius * inverse_sin_theta;
      gradient[direction] = radial[direction]
         - derivatives.theta * gradient_theta[direction]
         - derivatives.phi * gradient_phi[direction];
   }

   const amrex::Real inverse_radius_squared = inverse_radius * inverse_radius;
   const amrex::Real cot_theta = cos_theta * inverse_sin_theta;
   add_outer_product(hessian, inverse_radius, polar, polar);
   add_outer_product(hessian, inverse_radius, azimuthal, azimuthal);

   amrex::Real hessian_theta[3][3]{};
   add_outer_product(hessian_theta, -inverse_radius_squared, radial, polar);
   add_outer_product(hessian_theta, -inverse_radius_squared, polar, radial);
   add_outer_product(
      hessian_theta,
      cot_theta * inverse_radius_squared,
      azimuthal,
      azimuthal
   );
   amrex::Real hessian_phi[3][3]{};
   const amrex::Real azimuthal_factor = -inverse_radius_squared
      * inverse_sin_theta;
   add_outer_product(hessian_phi, azimuthal_factor, radial, azimuthal);
   add_outer_product(hessian_phi, azimuthal_factor, azimuthal, radial);
   add_outer_product(
      hessian_phi,
      cot_theta * azimuthal_factor,
      polar,
      azimuthal
   );
   add_outer_product(
      hessian_phi,
      cot_theta * azimuthal_factor,
      azimuthal,
      polar
   );
   for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
         hessian[row][column] -= derivatives.theta * hessian_theta[row][column]
            + derivatives.phi * hessian_phi[row][column];
      }
   }
   add_outer_product(
      hessian, -derivatives.theta_theta, gradient_theta, gradient_theta
   );
   add_outer_product(
      hessian, -derivatives.theta_phi, gradient_theta, gradient_phi
   );
   add_outer_product(
      hessian, -derivatives.theta_phi, gradient_phi, gradient_theta
   );
   add_outer_product(
      hessian, -derivatives.phi_phi, gradient_phi, gradient_phi
   );
}

amrex::Real spatial_dot(
   SpacetimeMetric const& metric,
   amrex::Real const first[3],
   amrex::Real const second[3]
) noexcept
{
   const amrex::Real gamma[3][3]{
      {metric.gamma11, metric.gamma12, metric.gamma13},
      {metric.gamma12, metric.gamma22, metric.gamma23},
      {metric.gamma13, metric.gamma23, metric.gamma33},
   };
   amrex::Real result = 0.0;
   for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
         result += gamma[row][column] * first[row] * second[column];
      }
   }
   return result;
}

struct SurfaceEvaluation {
   std::vector<ApparentHorizonGeometrySample> geometry;
   std::vector<amrex::Real> expansion;
   std::vector<amrex::Real> area_weight;
   amrex::Real linf = std::numeric_limits<amrex::Real>::infinity();
   amrex::Real l2 = std::numeric_limits<amrex::Real>::infinity();
   amrex::Real timestep = 0.0;
};

bool radius_in_grid(
   ApparentHorizonGeometryGrid const& geometry,
   std::vector<amrex::Real> const& radius
) noexcept
{
   const auto& radial = geometry.radial_coordinates();
   return std::all_of(radius.begin(), radius.end(), [&](amrex::Real value) {
      return std::isfinite(value) && value >= radial.front()
         && value <= radial.back();
   });
}

std::vector<ApparentHorizonGeometrySample> cache_surface_geometry(
   ApparentHorizonGeometryGrid const& geometry,
   std::vector<amrex::Real> const& radius,
   int num_theta,
   int num_phi
)
{
   std::vector<ApparentHorizonGeometrySample> samples(radius.size());
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5) * dtheta;
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5) * dphi;
         const std::size_t index = surface_index(
            theta_index, phi_index, num_phi
         );
         samples[index] = geometry.interpolate(radius[index], theta, phi);
      }
   }
   return samples;
}

void evaluate_expansion(
   std::vector<amrex::Real> const& radius,
   int num_theta,
   int num_phi,
   std::vector<ApparentHorizonGeometrySample> const& geometry,
   std::vector<amrex::Real>& expansion,
   std::vector<amrex::Real>* area_weight,
   amrex::Real* minimum_proper_spacing
)
{
   expansion.resize(radius.size());
   if (area_weight != nullptr) {
      area_weight->assign(radius.size(), 0.0);
   }
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5) * dtheta;
      const amrex::Real sin_theta = std::sin(theta);
      const amrex::Real cos_theta = std::cos(theta);
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5) * dphi;
         const amrex::Real sin_phi = std::sin(phi);
         const amrex::Real cos_phi = std::cos(phi);
         const std::size_t index = surface_index(
            theta_index, phi_index, num_phi
         );
         const SurfaceDerivatives derivatives = surface_derivatives(
            radius, theta_index, phi_index, num_theta, num_phi
         );
         amrex::Real gradient[3]{};
         amrex::Real hessian[3][3]{};
         radial_graph_level_set_geometry(
            radius[index], theta, phi, derivatives, gradient, hessian
         );
         SpacetimeMetric metric;
         SpacetimeMetricDerivatives metric_derivatives;
         unpack_apparent_horizon_geometry(
            geometry[index], metric, metric_derivatives
         );
         expansion[index] = apparent_horizon_expansion(
            metric, metric_derivatives, gradient, hessian
         );

         const amrex::Real radial[3]{
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
         };
         const amrex::Real polar[3]{
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta,
         };
         const amrex::Real azimuthal[3]{-sin_phi, cos_phi, 0.0};
         amrex::Real tangent_theta[3]{};
         amrex::Real tangent_phi[3]{};
         for (int direction = 0; direction < 3; ++direction) {
            tangent_theta[direction] = derivatives.theta * radial[direction]
               + radius[index] * polar[direction];
            tangent_phi[direction] = derivatives.phi * radial[direction]
               + radius[index] * sin_theta * azimuthal[direction];
         }
         const amrex::Real induced_theta_theta = spatial_dot(
            metric, tangent_theta, tangent_theta
         );
         const amrex::Real induced_theta_phi = spatial_dot(
            metric, tangent_theta, tangent_phi
         );
         const amrex::Real induced_phi_phi = spatial_dot(
            metric, tangent_phi, tangent_phi
         );
         const amrex::Real determinant = induced_theta_theta
            * induced_phi_phi - induced_theta_phi * induced_theta_phi;
         const amrex::Real local_area = determinant > 0.0
            ? std::sqrt(determinant) * dtheta * dphi
            : std::numeric_limits<amrex::Real>::quiet_NaN();
         if (area_weight != nullptr) {
            (*area_weight)[index] = local_area;
         }
         if (minimum_proper_spacing != nullptr) {
            if (!(induced_theta_theta > 0.0) || !(induced_phi_phi > 0.0)) {
               *minimum_proper_spacing =
                  std::numeric_limits<amrex::Real>::quiet_NaN();
            } else {
               const amrex::Real theta_spacing =
                  std::sqrt(induced_theta_theta) * dtheta;
               const amrex::Real phi_spacing =
                  std::sqrt(induced_phi_phi) * dphi;
               const amrex::Real combined_spacing = 1.0 / std::sqrt(
                  1.0 / (theta_spacing * theta_spacing)
                  + 1.0 / (phi_spacing * phi_spacing)
               );
               *minimum_proper_spacing = std::min(
                  *minimum_proper_spacing, combined_spacing
               );
            }
         }
      }
   }
}

SurfaceEvaluation evaluate_surface(
   ApparentHorizonGeometryGrid const& geometry_grid,
   std::vector<amrex::Real> const& radius,
   int num_theta,
   int num_phi,
   amrex::Real mass_scale,
   bool calculate_timestep
)
{
   SurfaceEvaluation evaluation;
   evaluation.geometry = cache_surface_geometry(
      geometry_grid, radius, num_theta, num_phi
   );
   amrex::Real minimum_spacing = std::numeric_limits<amrex::Real>::infinity();
   evaluate_expansion(
      radius,
      num_theta,
      num_phi,
      evaluation.geometry,
      evaluation.expansion,
      &evaluation.area_weight,
      calculate_timestep ? &minimum_spacing : nullptr
   );
   amrex::Real area = 0.0;
   amrex::Real weighted_square = 0.0;
   evaluation.linf = 0.0;
   for (std::size_t index = 0; index < radius.size(); ++index) {
      const amrex::Real scaled_expansion = mass_scale
         * evaluation.expansion[index];
      if (!std::isfinite(scaled_expansion)
          || !std::isfinite(evaluation.area_weight[index])) {
         evaluation.linf = std::numeric_limits<amrex::Real>::infinity();
         evaluation.l2 = std::numeric_limits<amrex::Real>::infinity();
         evaluation.timestep = 0.0;
         return evaluation;
      }
      evaluation.linf = std::max(
         evaluation.linf, std::abs(scaled_expansion)
      );
      area += evaluation.area_weight[index];
      weighted_square += scaled_expansion * scaled_expansion
         * evaluation.area_weight[index];
   }
   evaluation.l2 = area > 0.0
      ? std::sqrt(weighted_square / area)
      : std::numeric_limits<amrex::Real>::infinity();
   evaluation.timestep = calculate_timestep ? minimum_spacing : 0.0;
   return evaluation;
}

bool finite_surface(std::vector<amrex::Real> const& radius) noexcept
{
   return std::all_of(radius.begin(), radius.end(), [](amrex::Real value) {
      return std::isfinite(value) && value > 0.0;
   });
}

void evaluate_rhs(
   std::vector<amrex::Real> const& radius,
   std::vector<amrex::Real> const& velocity,
   int num_theta,
   int num_phi,
   std::vector<ApparentHorizonGeometrySample> const& cached_geometry,
   amrex::Real eta,
   std::vector<amrex::Real>& radius_rhs,
   std::vector<amrex::Real>& velocity_rhs
)
{
   std::vector<amrex::Real> expansion;
   evaluate_expansion(
      radius,
      num_theta,
      num_phi,
      cached_geometry,
      expansion,
      nullptr,
      nullptr
   );
   radius_rhs.resize(radius.size());
   velocity_rhs.resize(radius.size());
   for (std::size_t index = 0; index < radius.size(); ++index) {
      radius_rhs[index] = velocity[index] - eta * radius[index];
      velocity_rhs[index] = -expansion[index];
   }
}

void axpy(
   std::vector<amrex::Real> const& input,
   std::vector<amrex::Real> const& rhs,
   amrex::Real scale,
   std::vector<amrex::Real>& output
)
{
   output.resize(input.size());
   for (std::size_t index = 0; index < input.size(); ++index) {
      output[index] = input[index] + scale * rhs[index];
   }
}

void blend_ssprk3(
   std::vector<amrex::Real> const& initial,
   std::vector<amrex::Real> const& stage,
   std::vector<amrex::Real> const& rhs,
   amrex::Real initial_weight,
   amrex::Real stage_weight,
   amrex::Real timestep,
   std::vector<amrex::Real>& output
)
{
   output.resize(initial.size());
   for (std::size_t index = 0; index < initial.size(); ++index) {
      output[index] = initial_weight * initial[index]
         + stage_weight * (stage[index] + timestep * rhs[index]);
   }
}

bool ssprk3_step(
   std::vector<amrex::Real>& radius,
   std::vector<amrex::Real>& velocity,
   int num_theta,
   int num_phi,
   std::vector<ApparentHorizonGeometrySample> const& initial_geometry,
   amrex::Real eta,
   amrex::Real timestep,
   ApparentHorizonGeometryGrid const& geometry_grid
)
{
   std::vector<amrex::Real> radius_rhs;
   std::vector<amrex::Real> velocity_rhs;
   std::vector<amrex::Real> radius_stage1;
   std::vector<amrex::Real> velocity_stage1;
   std::vector<amrex::Real> radius_stage2;
   std::vector<amrex::Real> velocity_stage2;
   std::vector<amrex::Real> radius_next;
   std::vector<amrex::Real> velocity_next;

   evaluate_rhs(
      radius,
      velocity,
      num_theta,
      num_phi,
      initial_geometry,
      eta,
      radius_rhs,
      velocity_rhs
   );
   axpy(radius, radius_rhs, timestep, radius_stage1);
   axpy(velocity, velocity_rhs, timestep, velocity_stage1);
   if (!finite_surface(radius_stage1)
       || !radius_in_grid(geometry_grid, radius_stage1)) {
      return false;
   }
   const auto stage1_geometry = cache_surface_geometry(
      geometry_grid, radius_stage1, num_theta, num_phi
   );

   evaluate_rhs(
      radius_stage1,
      velocity_stage1,
      num_theta,
      num_phi,
      stage1_geometry,
      eta,
      radius_rhs,
      velocity_rhs
   );
   blend_ssprk3(
      radius, radius_stage1, radius_rhs, 0.75, 0.25, timestep, radius_stage2
   );
   blend_ssprk3(
      velocity,
      velocity_stage1,
      velocity_rhs,
      0.75,
      0.25,
      timestep,
      velocity_stage2
   );
   if (!finite_surface(radius_stage2)
       || !radius_in_grid(geometry_grid, radius_stage2)) {
      return false;
   }
   const auto stage2_geometry = cache_surface_geometry(
      geometry_grid, radius_stage2, num_theta, num_phi
   );

   evaluate_rhs(
      radius_stage2,
      velocity_stage2,
      num_theta,
      num_phi,
      stage2_geometry,
      eta,
      radius_rhs,
      velocity_rhs
   );
   blend_ssprk3(
      radius,
      radius_stage2,
      radius_rhs,
      1.0 / 3.0,
      2.0 / 3.0,
      timestep,
      radius_next
   );
   blend_ssprk3(
      velocity,
      velocity_stage2,
      velocity_rhs,
      1.0 / 3.0,
      2.0 / 3.0,
      timestep,
      velocity_next
   );
   if (!finite_surface(radius_next)
       || !radius_in_grid(geometry_grid, radius_next)) {
      return false;
   }
   radius = std::move(radius_next);
   velocity = std::move(velocity_next);
   return true;
}

std::vector<int> angular_levels(
   ApparentHorizonFinderParameters const& parameters
)
{
   if (parameters.coarsest_num_theta < 8
       || parameters.target_num_theta < parameters.coarsest_num_theta
       || parameters.coarsest_num_theta % 2 != 0
       || parameters.target_num_theta % 2 != 0) {
      throw std::invalid_argument(
         "apparent-horizon angular resolutions must be even with 8 <= coarse <= target"
      );
   }
   std::vector<int> levels;
   int current = parameters.coarsest_num_theta;
   while (current < parameters.target_num_theta) {
      levels.push_back(current);
      current = std::min(2 * current, parameters.target_num_theta);
   }
   levels.push_back(parameters.target_num_theta);
   return levels;
}

void calculate_diagnostics(
   ApparentHorizonGeometryGrid const& geometry_grid,
   SurfaceEvaluation const& evaluation,
   ApparentHorizonResult& result
)
{
   const auto& surface = result.surface;
   const int num_theta = surface.num_theta();
   const int num_phi = surface.num_phi();
   const auto& radius = surface.radius();
   const auto center = surface.center();
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   result.area = 0.0;
   result.centroid = {};
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5) * dtheta;
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5) * dphi;
         const std::size_t index = surface_index(
            theta_index, phi_index, num_phi
         );
         const amrex::Real weight = evaluation.area_weight[index];
         const amrex::Real x[3]{
            center[0] + radius[index] * std::sin(theta) * std::cos(phi),
            center[1] + radius[index] * std::sin(theta) * std::sin(phi),
            center[2] + radius[index] * std::cos(theta),
         };
         result.area += weight;
         for (int direction = 0; direction < 3; ++direction) {
            result.centroid[direction] += weight * x[direction];
         }
      }
   }
   if (!(result.area > 0.0) || !std::isfinite(result.area)) {
      result.converged = false;
      result.failure_reason = "found surface has non-positive proper area";
      return;
   }
   for (int direction = 0; direction < 3; ++direction) {
      result.centroid[direction] /= result.area;
   }
   result.irreducible_mass = std::sqrt(result.area / (16.0 * pi));
   result.minimum_radius = std::numeric_limits<amrex::Real>::infinity();
   result.maximum_radius = 0.0;
   result.mean_radius = 0.0;
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5) * dtheta;
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5) * dphi;
         const std::size_t index = surface_index(
            theta_index, phi_index, num_phi
         );
         const amrex::Real x[3]{
            center[0] + radius[index] * std::sin(theta) * std::cos(phi),
            center[1] + radius[index] * std::sin(theta) * std::sin(phi),
            center[2] + radius[index] * std::cos(theta),
         };
         const amrex::Real relative_radius = std::sqrt(
            (x[0] - result.centroid[0]) * (x[0] - result.centroid[0])
            + (x[1] - result.centroid[1]) * (x[1] - result.centroid[1])
            + (x[2] - result.centroid[2]) * (x[2] - result.centroid[2])
         );
         result.minimum_radius = std::min(
            result.minimum_radius, relative_radius
         );
         result.maximum_radius = std::max(
            result.maximum_radius, relative_radius
         );
         result.mean_radius += evaluation.area_weight[index]
            * relative_radius / result.area;
      }
   }
   (void)geometry_grid;
}

} // namespace

ApparentHorizonGeometrySample pack_apparent_horizon_geometry(
   SpacetimeMetric const& metric,
   SpacetimeMetricDerivatives const& derivatives
)
{
   return {
      metric.gamma11,
      metric.gamma12,
      metric.gamma13,
      metric.gamma22,
      metric.gamma23,
      metric.gamma33,
      metric.inv_gamma11,
      metric.inv_gamma12,
      metric.inv_gamma13,
      metric.inv_gamma22,
      metric.inv_gamma23,
      metric.inv_gamma33,
      derivatives.d1_gamma11,
      derivatives.d1_gamma12,
      derivatives.d1_gamma13,
      derivatives.d1_gamma22,
      derivatives.d1_gamma23,
      derivatives.d1_gamma33,
      derivatives.d2_gamma11,
      derivatives.d2_gamma12,
      derivatives.d2_gamma13,
      derivatives.d2_gamma22,
      derivatives.d2_gamma23,
      derivatives.d2_gamma33,
      derivatives.d3_gamma11,
      derivatives.d3_gamma12,
      derivatives.d3_gamma13,
      derivatives.d3_gamma22,
      derivatives.d3_gamma23,
      derivatives.d3_gamma33,
      derivatives.extrinsic11,
      derivatives.extrinsic12,
      derivatives.extrinsic13,
      derivatives.extrinsic22,
      derivatives.extrinsic23,
      derivatives.extrinsic33,
   };
}

void unpack_apparent_horizon_geometry(
   ApparentHorizonGeometrySample const& sample,
   SpacetimeMetric& metric,
   SpacetimeMetricDerivatives& derivatives
)
{
   int component = 0;
   metric.gamma11 = sample[component++];
   metric.gamma12 = sample[component++];
   metric.gamma13 = sample[component++];
   metric.gamma22 = sample[component++];
   metric.gamma23 = sample[component++];
   metric.gamma33 = sample[component++];
   metric.inv_gamma11 = sample[component++];
   metric.inv_gamma12 = sample[component++];
   metric.inv_gamma13 = sample[component++];
   metric.inv_gamma22 = sample[component++];
   metric.inv_gamma23 = sample[component++];
   metric.inv_gamma33 = sample[component++];
   derivatives.d1_gamma11 = sample[component++];
   derivatives.d1_gamma12 = sample[component++];
   derivatives.d1_gamma13 = sample[component++];
   derivatives.d1_gamma22 = sample[component++];
   derivatives.d1_gamma23 = sample[component++];
   derivatives.d1_gamma33 = sample[component++];
   derivatives.d2_gamma11 = sample[component++];
   derivatives.d2_gamma12 = sample[component++];
   derivatives.d2_gamma13 = sample[component++];
   derivatives.d2_gamma22 = sample[component++];
   derivatives.d2_gamma23 = sample[component++];
   derivatives.d2_gamma33 = sample[component++];
   derivatives.d3_gamma11 = sample[component++];
   derivatives.d3_gamma12 = sample[component++];
   derivatives.d3_gamma13 = sample[component++];
   derivatives.d3_gamma22 = sample[component++];
   derivatives.d3_gamma23 = sample[component++];
   derivatives.d3_gamma33 = sample[component++];
   derivatives.extrinsic11 = sample[component++];
   derivatives.extrinsic12 = sample[component++];
   derivatives.extrinsic13 = sample[component++];
   derivatives.extrinsic22 = sample[component++];
   derivatives.extrinsic23 = sample[component++];
   derivatives.extrinsic33 = sample[component++];
}

ApparentHorizonGeometryGrid::ApparentHorizonGeometryGrid(
   std::array<amrex::Real, 3> center,
   std::vector<amrex::Real> radial_coordinates,
   int num_theta,
   int num_phi,
   std::vector<ApparentHorizonGeometrySample> samples
)
   : center_(center),
     radial_coordinates_(std::move(radial_coordinates)),
     num_theta_(num_theta),
     num_phi_(num_phi),
     samples_(std::move(samples))
{
   if (radial_coordinates_.size() < 4 || num_theta_ < 8
       || num_phi_ < 2 * num_theta_ || num_phi_ % 2 != 0) {
      throw std::invalid_argument(
         "apparent-horizon geometry grid requires Nr>=4, Ntheta>=8, and even Nphi>=2*Ntheta"
      );
   }
   if (!std::is_sorted(radial_coordinates_.begin(), radial_coordinates_.end())
       || std::adjacent_find(
          radial_coordinates_.begin(), radial_coordinates_.end()
       ) != radial_coordinates_.end()) {
      throw std::invalid_argument(
         "apparent-horizon radial coordinates must be strictly increasing"
      );
   }
   const std::size_t expected = radial_coordinates_.size()
      * static_cast<std::size_t>(num_theta_)
      * static_cast<std::size_t>(num_phi_);
   if (samples_.size() != expected) {
      throw std::invalid_argument(
         "apparent-horizon geometry sample count does not match the spherical grid"
      );
   }
}

std::size_t ApparentHorizonGeometryGrid::index(
   int radial,
   int theta,
   int phi
) const noexcept
{
   return (
      static_cast<std::size_t>(radial) * num_theta_ + theta
   ) * num_phi_ + phi;
}

ApparentHorizonGeometrySample
ApparentHorizonGeometryGrid::interpolate_angular(
   int radial,
   amrex::Real theta,
   amrex::Real phi
) const
{
   canonicalize_angles(theta, phi);
   const amrex::Real dtheta = pi / num_theta_;
   const amrex::Real dphi = 2.0 * pi / num_phi_;
   const amrex::Real theta_grid = theta / dtheta - 0.5;
   const amrex::Real phi_grid = phi / dphi - 0.5;
   const int theta_lower = static_cast<int>(std::floor(theta_grid));
   const int phi_lower = static_cast<int>(std::floor(phi_grid));
   const amrex::Real theta_fraction = theta_grid - theta_lower;
   const amrex::Real phi_fraction = phi_grid - phi_lower;
   ApparentHorizonGeometrySample result{};
   for (int theta_offset = 0; theta_offset < 2; ++theta_offset) {
      const AngularIndex mapped = map_theta_index(
         theta_lower + theta_offset, num_theta_, num_phi_
      );
      const amrex::Real theta_weight = theta_offset == 0
         ? 1.0 - theta_fraction : theta_fraction;
      for (int phi_offset = 0; phi_offset < 2; ++phi_offset) {
         const int mapped_phi = wrap_index(
            phi_lower + phi_offset + mapped.phi_shift, num_phi_
         );
         const amrex::Real phi_weight = phi_offset == 0
            ? 1.0 - phi_fraction : phi_fraction;
         const auto& sample = samples_[index(radial, mapped.theta, mapped_phi)];
         for (int component = 0;
              component < ApparentHorizonGeometryComponents;
              ++component) {
            result[component] += theta_weight * phi_weight * sample[component];
         }
      }
   }
   return result;
}

ApparentHorizonGeometrySample ApparentHorizonGeometryGrid::interpolate(
   amrex::Real radius,
   amrex::Real theta,
   amrex::Real phi
) const
{
   if (radius < radial_coordinates_.front()
       || radius > radial_coordinates_.back()) {
      throw std::out_of_range(
         "apparent-horizon trial surface left the radial interpolation volume"
      );
   }
   auto upper = std::lower_bound(
      radial_coordinates_.begin(), radial_coordinates_.end(), radius
   );
   int lower_index = 0;
   amrex::Real fraction = 0.0;
   if (upper == radial_coordinates_.begin()) {
      lower_index = 0;
   } else if (upper == radial_coordinates_.end()) {
      lower_index = static_cast<int>(radial_coordinates_.size()) - 2;
      fraction = 1.0;
   } else {
      const int upper_index = static_cast<int>(
         upper - radial_coordinates_.begin()
      );
      lower_index = upper_index - 1;
      fraction = (radius - radial_coordinates_[lower_index])
         / (radial_coordinates_[upper_index]
            - radial_coordinates_[lower_index]);
   }
   const auto lower = interpolate_angular(lower_index, theta, phi);
   const auto upper_sample = interpolate_angular(
      lower_index + 1, theta, phi
   );
   ApparentHorizonGeometrySample result{};
   for (int component = 0;
        component < ApparentHorizonGeometryComponents;
        ++component) {
      result[component] = (1.0 - fraction) * lower[component]
         + fraction * upper_sample[component];
   }
   return result;
}

ApparentHorizonSurface::ApparentHorizonSurface(
   std::array<amrex::Real, 3> center,
   int num_theta,
   int num_phi,
   std::vector<amrex::Real> radius
)
   : center_(center),
     num_theta_(num_theta),
     num_phi_(num_phi),
     radius_(std::move(radius))
{
   if (num_theta_ < 2 || num_phi_ < 2 || num_phi_ % 2 != 0
       || radius_.size() != static_cast<std::size_t>(num_theta_ * num_phi_)
       || !finite_surface(radius_)) {
      throw std::invalid_argument("invalid apparent-horizon radial surface");
   }
}

amrex::Real ApparentHorizonSurface::radius_at(
   amrex::Real theta,
   amrex::Real phi
) const
{
   if (radius_.empty()) {
      throw std::logic_error("apparent-horizon surface is empty");
   }
   return interpolate_surface(radius_, num_theta_, num_phi_, theta, phi);
}

amrex::Real ApparentHorizonSurface::signed_radial_distance(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) const
{
   const amrex::Real relative_x = x - center_[0];
   const amrex::Real relative_y = y - center_[1];
   const amrex::Real relative_z = z - center_[2];
   const amrex::Real coordinate_radius = std::sqrt(
      relative_x * relative_x + relative_y * relative_y
      + relative_z * relative_z
   );
   if (coordinate_radius == 0.0) {
      return -radius_at(0.0, 0.0);
   }
   const amrex::Real theta = std::acos(std::clamp(
      relative_z / coordinate_radius, amrex::Real(-1.0), amrex::Real(1.0)
   ));
   const amrex::Real phi = std::atan2(relative_y, relative_x);
   return coordinate_radius - radius_at(theta, phi);
}

ApparentHorizonResult find_apparent_horizon(
   ApparentHorizonGeometryGrid const& geometry,
   amrex::Real initial_radius,
   ApparentHorizonFinderParameters const& parameters,
   std::optional<ApparentHorizonSurface> const& initial_surface
)
{
   if (!(parameters.mass_scale > 0.0) || !(parameters.cfl > 0.0)
       || !(parameters.cfl <= 1.0) || !(parameters.eta_times_mass > 0.0)
       || parameters.max_iterations_per_level <= 0
       || parameters.theta_linf_times_mass_tolerance < 0.0
       || parameters.theta_l2_times_mass_tolerance < 0.0) {
      throw std::invalid_argument("invalid apparent-horizon finder parameters");
   }
   const auto levels = angular_levels(parameters);
   if (geometry.num_theta() < parameters.target_num_theta
       || geometry.num_phi() < 2 * parameters.target_num_theta) {
      throw std::invalid_argument(
         "apparent-horizon geometry grid is coarser than the target surface"
      );
   }
   if (!(initial_radius > 0.0)) {
      throw std::invalid_argument(
         "apparent-horizon initial radius must be positive"
      );
   }

   ApparentHorizonResult result;
   std::optional<ApparentHorizonSurface> previous_surface = initial_surface;
   const amrex::Real eta = parameters.eta_times_mass / parameters.mass_scale;
   for (int num_theta : levels) {
      const int num_phi = 2 * num_theta;
      std::vector<amrex::Real> radius(
         static_cast<std::size_t>(num_theta * num_phi), initial_radius
      );
      if (previous_surface.has_value()) {
         const amrex::Real dtheta = pi / num_theta;
         const amrex::Real dphi = 2.0 * pi / num_phi;
         for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
            const amrex::Real theta = (theta_index + 0.5) * dtheta;
            for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
               radius[surface_index(theta_index, phi_index, num_phi)] =
                  previous_surface->radius_at(
                     theta, (phi_index + 0.5) * dphi
                  );
            }
         }
      }
      if (!radius_in_grid(geometry, radius)) {
         result.failure_reason =
            "initial apparent-horizon surface lies outside the interpolation volume";
         return result;
      }
      std::vector<amrex::Real> velocity(radius.size());
      std::transform(
         radius.begin(), radius.end(), velocity.begin(),
         [eta](amrex::Real value) { return eta * value; }
      );
      std::optional<std::vector<amrex::Real>> over_relaxation_history;
      bool level_converged = false;
      for (int iteration = 0;
           iteration < parameters.max_iterations_per_level;
           ++iteration) {
         SurfaceEvaluation evaluation = evaluate_surface(
            geometry,
            radius,
            num_theta,
            num_phi,
            parameters.mass_scale,
            true
         );
         result.theta_linf_times_mass = evaluation.linf;
         result.theta_l2_times_mass = evaluation.l2;
         if (!std::isfinite(evaluation.linf)
             || !std::isfinite(evaluation.l2)) {
            result.failure_reason =
               "apparent-horizon expansion became nonfinite";
            return result;
         }
         if (evaluation.linf
                <= parameters.theta_linf_times_mass_tolerance
             && evaluation.l2
                <= parameters.theta_l2_times_mass_tolerance) {
            level_converged = true;
            break;
         }
         // The user-facing CFL multiplies the calculated stable step rather
         // than the raw proper angular spacing.  For the sixth-order centered
         // second derivative, max(-D2)*dx^2 = 1088/180 and SSPRK3 reaches
         // sqrt(3) on the imaginary axis.  The extra factor 1/2 is the
         // nonlinear/cached-geometry safety used by this radial-graph owner;
         // a user CFL near one therefore remains the intended setting.
         const amrex::Real sixth_order_ssprk3_wave_factor =
            0.5 * std::sqrt(3.0 / (1088.0 / 180.0));
         const amrex::Real timestep = parameters.cfl
            * sixth_order_ssprk3_wave_factor * evaluation.timestep;
         if (!(timestep > 0.0) || !std::isfinite(timestep)) {
            result.failure_reason =
               "apparent-horizon pseudo-timestep is non-positive or nonfinite";
            return result;
         }
         if (!ssprk3_step(
                radius,
                velocity,
                num_theta,
                num_phi,
                evaluation.geometry,
                eta,
                timestep,
                geometry
             )) {
            result.failure_reason =
               "apparent-horizon trial surface left the interpolation volume";
            return result;
         }
         ++result.iterations;
         result.pseudo_time += timestep;

         if (parameters.over_relaxation
             && parameters.over_relaxation_interval > 0
             && (iteration + 1) % parameters.over_relaxation_interval == 0) {
            if (!over_relaxation_history.has_value()) {
               over_relaxation_history = radius;
            } else {
               SurfaceEvaluation current = evaluate_surface(
                  geometry,
                  radius,
                  num_theta,
                  num_phi,
                  parameters.mass_scale,
                  false
               );
               amrex::Real best_residual = current.linf;
               std::vector<amrex::Real> best_radius = radius;
               amrex::Real factor = 1.0;
               for (int trial_index = 0; trial_index < 12; ++trial_index) {
                  std::vector<amrex::Real> trial(radius.size());
                  for (std::size_t index = 0; index < radius.size(); ++index) {
                     trial[index] = radius[index] + factor
                        * (radius[index] - (*over_relaxation_history)[index]);
                  }
                  if (!finite_surface(trial)
                      || !radius_in_grid(geometry, trial)) {
                     break;
                  }
                  const SurfaceEvaluation trial_evaluation = evaluate_surface(
                     geometry,
                     trial,
                     num_theta,
                     num_phi,
                     parameters.mass_scale,
                     false
                  );
                  if (!(trial_evaluation.linf < best_residual)) {
                     break;
                  }
                  best_residual = trial_evaluation.linf;
                  best_radius = std::move(trial);
                  factor *= parameters.over_relaxation_growth;
               }
               const amrex::Real relative_improvement = current.linf > 0.0
                  ? (current.linf - best_residual) / current.linf : 0.0;
               if (relative_improvement
                   >= parameters.over_relaxation_minimum_improvement) {
                  radius = std::move(best_radius);
                  std::transform(
                     radius.begin(), radius.end(), velocity.begin(),
                     [eta](amrex::Real value) { return eta * value; }
                  );
                  over_relaxation_history.reset();
               } else {
                  over_relaxation_history = radius;
               }
            }
         }
      }
      if (!level_converged) {
         result.failure_reason =
            "apparent-horizon relaxation reached its iteration limit";
         return result;
      }
      previous_surface.emplace(
         geometry.center(), num_theta, num_phi, std::move(radius)
      );
   }

   result.converged = true;
   result.surface = std::move(*previous_surface);
   const SurfaceEvaluation final_evaluation = evaluate_surface(
      geometry,
      result.surface.radius(),
      result.surface.num_theta(),
      result.surface.num_phi(),
      parameters.mass_scale,
      false
   );
   result.theta_linf_times_mass = final_evaluation.linf;
   result.theta_l2_times_mass = final_evaluation.l2;
   calculate_diagnostics(geometry, final_evaluation, result);
   return result;
}

} // namespace pyfluid_amrex
