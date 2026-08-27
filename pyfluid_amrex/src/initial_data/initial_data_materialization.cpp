#include "pyfluid_amrex/initial_data/materialization.hpp"

#include <AMReX.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFabUtil.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/initial_data/centering.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/spacetime_coupling.hpp"
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

namespace {

char const* conformal_constraint_formulation_name(
   ConformalConstraintFormulation formulation
) noexcept
{
   switch (formulation) {
   case ConformalConstraintFormulation::ctt:
      return "ctt";
   case ConformalConstraintFormulation::cttk:
      return "cttk";
   case ConformalConstraintFormulation::xcts:
      return "xcts";
   }
   return "unknown";
}

amrex::Vector<GridFieldSymmetryKind> scalar_symmetries(int num_components)
{
   return amrex::Vector<GridFieldSymmetryKind>(
      num_components, GridFieldSymmetryKind::Scalar
   );
}

amrex::Vector<GridFieldSymmetryKind> vector_symmetries()
{
   return {
      GridFieldSymmetryKind::Vector1,
      GridFieldSymmetryKind::Vector2,
      GridFieldSymmetryKind::Vector3,
   };
}

void accumulate_local_linf_norms(
   amrex::MultiFab const& difference,
   amrex::MultiFab const& solution,
   int num_components,
   amrex::Real& update_norm,
   amrex::Real& solution_norm
)
{
   for (int component = 0; component < num_components; ++component) {
      update_norm = std::max(
         update_norm, difference.norm0(component, 0, true)
      );
      solution_norm = std::max(
         solution_norm, solution.norm0(component, 0, true)
      );
   }
}

amrex::Real global_relative_fixed_point_defect(
   amrex::Real relaxed_matter_update_norm,
   amrex::Real other_update_norm,
   amrex::Real solution_norm,
   amrex::Real matter_relaxation
)
{
   if (!(matter_relaxation > 0.0) || !(matter_relaxation <= 1.0)
       || !std::isfinite(matter_relaxation)) {
      amrex::Abort("matter fixed-point relaxation must lie in (0, 1]");
   }
   std::array<amrex::Real, 3> norms{
      relaxed_matter_update_norm,
      other_update_norm,
      solution_norm,
   };
   amrex::ParallelAllReduce::Max(
      norms.data(),
      static_cast<int>(norms.size()),
      amrex::ParallelContext::CommunicatorSub()
   );
   // A relaxed matter update is lambda * (F(u) - u).  Dividing only that
   // sector by lambda makes the stopping criterion measure the fixed-point
   // defect rather than the user-selected damping.  Geometry and magnetic
   // updates retain their own undamped scale.
   return std::max(norms[0] / matter_relaxation, norms[1])
      / std::max(amrex::Real(1.0), norms[2]);
}

struct CircularContinuationSchedule {
   amrex::Vector<amrex::Real> stellar_angular_velocities;
   amrex::Vector<amrex::Real> disk_angular_velocities;
};

CircularContinuationSchedule circular_continuation_schedule(
   RotatingStarParameters const& rotating_star,
   EquilibriumTorusParameters const& equilibrium_torus
)
{
   CircularContinuationSchedule schedule;
   int continuation_steps = 1;
   if (!rotating_star.continuation_targets.empty()) {
      continuation_steps = static_cast<int>(
         rotating_star.continuation_targets.size()
      ) + 1;
   } else if (
      rotating_star.active
      && rotating_star.angular_velocity
         != rotating_star.continuation_source_angular_velocity
   ) {
      continuation_steps = std::max(
         continuation_steps, rotating_star.continuation_steps
      );
   }
   if (
      equilibrium_torus.active
      && equilibrium_torus.angular_velocity
         != equilibrium_torus.continuation_source_angular_velocity
   ) {
      continuation_steps = std::max(
         continuation_steps, equilibrium_torus.continuation_steps
      );
   }

   schedule.stellar_angular_velocities.reserve(continuation_steps);
   schedule.disk_angular_velocities.reserve(continuation_steps);
   for (int step = 1; step <= continuation_steps; ++step) {
      amrex::Real stellar_angular_velocity = rotating_star.angular_velocity;
      if (!rotating_star.continuation_targets.empty()) {
         stellar_angular_velocity = step < continuation_steps
            ? rotating_star.continuation_targets[step - 1]
            : rotating_star.angular_velocity;
      } else if (rotating_star.active) {
         const amrex::Real fraction = static_cast<amrex::Real>(step)
            / static_cast<amrex::Real>(continuation_steps);
         stellar_angular_velocity =
            rotating_star.continuation_source_angular_velocity
            + fraction * (
               rotating_star.angular_velocity
               - rotating_star.continuation_source_angular_velocity
            );
      }
      const amrex::Real fraction = static_cast<amrex::Real>(step)
         / static_cast<amrex::Real>(continuation_steps);
      const amrex::Real disk_angular_velocity =
         equilibrium_torus.continuation_source_angular_velocity
         + fraction * (
            equilibrium_torus.angular_velocity
            - equilibrium_torus.continuation_source_angular_velocity
         );
      schedule.stellar_angular_velocities.push_back(
         stellar_angular_velocity
      );
      schedule.disk_angular_velocities.push_back(disk_angular_velocity);
   }
   return schedule;
}

#if (AMREX_SPACEDIM >= 2)
ConformalBlackHoleSummary axisymmetric_black_hole_throat_summary(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& spacetime_state,
   amrex::Real throat_radius
)
{
   if (hierarchy.geometry.size() != 1 || !(throat_radius > 0.0)) {
      amrex::Abort("invalid axisymmetric black-hole throat diagnostic");
   }
   const amrex::Geometry& geometry = hierarchy.geometry[0];
   const amrex::Box nodal_domain = amrex::convert(
      geometry.Domain(), amrex::IntVect::TheNodeVector()
   );
   const int radial_index = nodal_domain.smallEnd(0);
   const int theta_lower = nodal_domain.smallEnd(1);
   const int theta_upper = nodal_domain.bigEnd(1);
   const int equator_index = std::clamp(
      static_cast<int>(std::llround(
         (0.5 * constants::pi - geometry.ProbLo(1))
         / geometry.CellSize(1)
      )) + theta_lower,
      theta_lower,
      theta_upper
   );
#if (AMREX_SPACEDIM == 3)
   const int phi_index = nodal_domain.smallEnd(2);
#endif
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const std::unique_ptr<amrex::iMultiFab> owner =
      spacetime_state.OwnerMask(geometry.periodicity());
   amrex::MultiFab integrand(
      spacetime_state.boxArray(), spacetime_state.DistributionMap(), 4, 0
   );
   integrand.setVal(0.0);
   for (amrex::MFIter mfi(
           spacetime_state, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox() & nodal_domain;
      const auto state = spacetime_state.const_array(mfi);
      const auto owned = owner->const_array(mfi);
      const auto output = integrand.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         if (i != radial_index
#if (AMREX_SPACEDIM == 3)
             || k != phi_index
#endif
             || owned(i, j, k) == 0) {
            return;
         }
         const Z4cState z4c_state = load_z4c_state(state, i, j, k);
         const SpacetimeMetric metric =
            generated::z4c_spacetime_metric(z4c_state);
         const amrex::Real tangent_determinant =
            metric.gamma22 * metric.gamma33
            - metric.gamma23 * metric.gamma23;
         const amrex::Real endpoint_weight =
            j == theta_lower || j == theta_upper ? 0.5 : 1.0;
         const amrex::Real theta = lower[1]
            + static_cast<amrex::Real>(j - theta_lower) * spacing[1];
         output(i, j, k, 0) = endpoint_weight * std::sin(theta)
            * std::sqrt(tangent_determinant);
         output(i, j, k, 1) = endpoint_weight
            * std::sqrt(metric.gamma22);
         if (j == equator_index) {
            output(i, j, k, 2) = std::sin(theta)
               * std::sqrt(metric.gamma33);
         }
         const amrex::Real trace_third =
            (z4c_state.k_hat + 2.0 * z4c_state.theta) / 3.0;
         const amrex::Real inverse_chi = 1.0 / z4c_state.chi;
         const amrex::Real extrinsic13 = inverse_chi
            * (z4c_state.a_tilde13
               + z4c_state.gamma_tilde13 * trace_third);
         const amrex::Real extrinsic23 = inverse_chi
            * (z4c_state.a_tilde23
               + z4c_state.gamma_tilde23 * trace_third);
         const amrex::Real extrinsic33 = inverse_chi
            * (z4c_state.a_tilde33
               + z4c_state.gamma_tilde33 * trace_third);
         // The native spherical owner is exactly axisymmetric, so
         // phi^i = (partial_phi)^i is a true Killing vector. In the stored
         // orthonormal spherical basis its only component is r sin(theta).
         const amrex::Real axial_curvature = throat_radius * std::sin(theta)
            * (
               extrinsic13 * metric.inv_gamma11
               + extrinsic23 * metric.inv_gamma12
               + extrinsic33 * metric.inv_gamma13
            ) / std::sqrt(metric.inv_gamma11);
         output(i, j, k, 3) = output(i, j, k, 0) * axial_curvature;
      });
   }
   if (integrand.contains_nan(0, 4, 0)) {
      amrex::Abort("black-hole throat geometry integrand is nonfinite");
   }
   amrex::Real theta_integral = integrand.sum(0, true) * spacing[1];
   amrex::Real polar_integral = integrand.sum(1, true) * spacing[1];
   amrex::Real equatorial_metric_radius = integrand.sum(2, true);
   amrex::Real angular_momentum_integral =
      integrand.sum(3, true) * spacing[1];
   amrex::Real reductions[4]{
      theta_integral,
      polar_integral,
      equatorial_metric_radius,
      angular_momentum_integral,
   };
   amrex::ParallelDescriptor::ReduceRealSum(reductions, 4);
   theta_integral = reductions[0];
   polar_integral = reductions[1];
   equatorial_metric_radius = reductions[2];
   angular_momentum_integral = reductions[3];
   const amrex::Real theta_symmetry_factor =
      hierarchy.reflection_hi[1] ? 2.0 : 1.0;
   const amrex::Real area = 2.0 * constants::pi * theta_symmetry_factor
      * throat_radius * throat_radius * theta_integral;
   const amrex::Real equatorial_circumference = 2.0 * constants::pi
      * throat_radius * equatorial_metric_radius;
   const amrex::Real polar_circumference = 2.0 * theta_symmetry_factor
      * throat_radius * polar_integral;
   const amrex::Real kerr_horizon_mass = equatorial_circumference
      / (4.0 * constants::pi);
   const amrex::Real angular_momentum = 0.25 * theta_symmetry_factor
      * throat_radius * throat_radius * angular_momentum_integral;
   const amrex::Real irreducible_mass = std::sqrt(
      area / (16.0 * constants::pi)
   );
   const amrex::Real christodoulou_mass = irreducible_mass * std::sqrt(
      1.0 + angular_momentum * angular_momentum
         / (
            4.0 * irreducible_mass * irreducible_mass
               * irreducible_mass * irreducible_mass
         )
   );
   const amrex::Real christodoulou_dimensionless_spin = angular_momentum
      / (christodoulou_mass * christodoulou_mass);
   if (!(area > 0.0) || !(equatorial_circumference > 0.0)
       || !(polar_circumference > 0.0) || !std::isfinite(area)
       || !std::isfinite(kerr_horizon_mass)
       || !std::isfinite(angular_momentum)
       || !(christodoulou_mass > 0.0)
       || !std::isfinite(christodoulou_dimensionless_spin)
       || std::abs(christodoulou_dimensionless_spin) > 1.0 + 1.0e-8) {
      amrex::Abort("black-hole throat geometry diagnostic is invalid");
   }
   const amrex::Real kerr_area_factor = area / (
      8.0 * constants::pi * kerr_horizon_mass * kerr_horizon_mass
   );
   amrex::Real kerr_dimensionless_spin = -1.0;
   if (std::isfinite(kerr_area_factor)
       && kerr_area_factor >= 1.0 - 5.0e-2
       && kerr_area_factor <= 2.0 + 5.0e-2) {
      const amrex::Real bounded_area_factor = std::clamp(
         kerr_area_factor, amrex::Real(1.0), amrex::Real(2.0)
      );
      kerr_dimensionless_spin = std::sqrt(std::max(
         amrex::Real(0.0),
         amrex::Real(1.0)
            - (bounded_area_factor - 1.0) * (bounded_area_factor - 1.0)
      ));
   }
   return {
      area,
      irreducible_mass,
      equatorial_circumference,
      polar_circumference,
      kerr_horizon_mass,
      kerr_dimensionless_spin,
      angular_momentum,
      christodoulou_mass,
      std::clamp(
         christodoulou_dimensionless_spin,
         amrex::Real(-1.0),
         amrex::Real(1.0)
      ),
   };
}
#endif

amrex::Real reflected_domain_volume_factor(
   InitialDataEllipticHierarchy const& hierarchy
) noexcept
{
   amrex::Real factor = 1.0;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      if (hierarchy.reflection_lo[direction] != 0
          || hierarchy.reflection_hi[direction] != 0) {
         factor *= 2.0;
      }
   }
   return factor;
}

amrex::Vector<GridFieldSymmetryKind> z4c_symmetries()
{
   amrex::Vector<GridFieldSymmetryKind> result;
   result.reserve(Z4C_NUM_COMPONENTS);
   for (GridFieldInfo const& field : z4c_state_grid_fields().fields) {
      for (GridFieldComponentInfo const& component : field.components) {
         result.push_back(component.symmetry);
      }
   }
   return result;
}

void materialize_conformal_geometry(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& shift,
   amrex::Vector<amrex::MultiFab const*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   for (int level = 0; level < num_levels; ++level) {
      spacetime_state[level]->setVal(0.0);
      for (amrex::MFIter mfi(
              *spacetime_state[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[level]->const_array(mfi);
         const auto lapse_psi =
            lapse_times_conformal_factor[level]->const_array(mfi);
         const auto shift_vector = shift[level]->const_array(mfi);
         const auto curvature = mean_curvature[level]->const_array(mfi);
         const auto extrinsic = conformal_extrinsic[level]->const_array(mfi);
         const auto geometry = spacetime_state[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            store_z4c_state(
               geometry,
               i,
               j,
               k,
               generated::ctt_canonical_z4c_state(
                  psi(i, j, k, 0),
                  extrinsic(i, j, k, CONFORMAL_A11),
                  extrinsic(i, j, k, CONFORMAL_A12),
                  extrinsic(i, j, k, CONFORMAL_A13),
                  extrinsic(i, j, k, CONFORMAL_A22),
                  extrinsic(i, j, k, CONFORMAL_A23),
                  extrinsic(i, j, k, CONFORMAL_A33),
                  curvature(i, j, k, 0),
                  generated::xcts_lapse_from_conformal_unknowns(
                     psi(i, j, k, 0), lapse_psi(i, j, k, 0)
                  ),
                  shift_vector(i, j, k, 0),
                  shift_vector(i, j, k, 1),
                  shift_vector(i, j, k, 2)
               )
            );
         });
      }
   }
   const auto boundary = initial_data_boundary_records(
      hierarchy, z4c_symmetries()
   );
   fill_initial_data_ghosts(hierarchy, spacetime_state, boundary);
   for (int level = 0; level < num_levels; ++level) {
      transfer_grid_field_centering(
         *cell_spacetime[level],
         *spacetime_state[level],
         hierarchy.geometry[level],
         2
      );
   }
   fill_initial_data_ghosts(hierarchy, cell_spacetime, boundary);
}

#if (AMREX_SPACEDIM >= 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real axisymmetric_nodal_derivative(
   amrex::Array4<amrex::Real const> const& field,
   int i,
   int j,
   int k,
   int direction,
   amrex::Dim3 domain_lo,
   amrex::Dim3 domain_hi,
   amrex::Real const* radial_faces,
   amrex::Real inverse_angular_spacing
) noexcept
{
   if (direction != 0) {
      return 0.5 * inverse_angular_spacing * (
         field(i, j + 1, k, 0) - field(i, j - 1, k, 0)
      );
   }
   int lower_i = i - 1;
   int upper_i = i + 1;
   const bool radial_origin = radial_faces[0] == 0.0;
   if (i == domain_lo.x && !radial_origin) {
      lower_i = i + 1;
      upper_i = i + 2;
   } else if (i == domain_hi.x) {
      lower_i = i - 2;
      upper_i = i - 1;
   }
   const amrex::Real lower_coordinate =
      radial_faces[lower_i - domain_lo.x];
   const amrex::Real center = radial_faces[i - domain_lo.x];
   const amrex::Real upper_coordinate =
      radial_faces[upper_i - domain_lo.x];
   return field(lower_i, j, k, 0) * (center - upper_coordinate) / (
            (lower_coordinate - center)
            * (lower_coordinate - upper_coordinate)
         )
      + field(i, j, k, 0) * (
            2.0 * center - lower_coordinate - upper_coordinate
         ) / (
            (center - lower_coordinate) * (center - upper_coordinate)
         )
      + field(upper_i, j, k, 0) * (center - lower_coordinate) / (
            (upper_coordinate - lower_coordinate)
            * (upper_coordinate - center)
         );
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real extrapolate_axisymmetric_cell_component_to_inner_vertex(
   amrex::Array4<amrex::Real const> const& field,
   int vertex_j,
   int vertex_k,
   int component,
   amrex::Dim3 cell_lo,
   amrex::Dim3 cell_hi,
   amrex::Real const* radial_centers,
   amrex::Real inner_radius
) noexcept
{
   const amrex::Real radius0 = radial_centers[0];
   const amrex::Real radius1 = radial_centers[1];
   const amrex::Real radius2 = radial_centers[2];
   const amrex::Real weight0 =
      (inner_radius - radius1) * (inner_radius - radius2)
      / ((radius0 - radius1) * (radius0 - radius2));
   const amrex::Real weight1 =
      (inner_radius - radius0) * (inner_radius - radius2)
      / ((radius1 - radius0) * (radius1 - radius2));
   const amrex::Real weight2 =
      (inner_radius - radius0) * (inner_radius - radius1)
      / ((radius2 - radius0) * (radius2 - radius1));
   const int first_j = amrex::max(
      cell_lo.y, amrex::min(vertex_j - 1, cell_hi.y)
   );
   const int second_j = amrex::max(
      cell_lo.y, amrex::min(vertex_j, cell_hi.y)
   );
#if (AMREX_SPACEDIM == 3)
   const int first_k = amrex::max(
      cell_lo.z, amrex::min(vertex_k - 1, cell_hi.z)
   );
   const int second_k = amrex::max(
      cell_lo.z, amrex::min(vertex_k, cell_hi.z)
   );
#else
   const int first_k = vertex_k;
   const int second_k = vertex_k;
#endif
   amrex::Real value = 0.0;
   int count = 0;
   for (int cell_j = first_j; cell_j <= second_j; ++cell_j) {
      for (int cell_k = first_k; cell_k <= second_k; ++cell_k) {
         value += weight0 * field(cell_lo.x, cell_j, cell_k, component)
            + weight1 * field(cell_lo.x + 1, cell_j, cell_k, component)
            + weight2 * field(cell_lo.x + 2, cell_j, cell_k, component);
         ++count;
      }
   }
   return value / static_cast<amrex::Real>(count);
}

void materialize_axisymmetric_cell_geometry(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& conformal_factor,
   amrex::MultiFab const& lapse_times_conformal_factor,
   amrex::MultiFab const& azimuthal_shift,
   amrex::MultiFab const& conformal_extrinsic,
   amrex::MultiFab& cell_spacetime
)
{
   for (amrex::MFIter mfi(cell_spacetime, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto psi = conformal_factor.const_array(mfi);
      const auto lapse_psi = lapse_times_conformal_factor.const_array(mfi);
      const auto beta = azimuthal_shift.const_array(mfi);
      const auto extrinsic = conformal_extrinsic.const_array(mfi);
      const auto geometry = cell_spacetime.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         store_z4c_state(
            geometry,
            i,
            j,
            k,
            generated::ctt_canonical_z4c_state(
               psi(i, j, k, 0),
               extrinsic(i, j, k, CONFORMAL_A11),
               extrinsic(i, j, k, CONFORMAL_A12),
               extrinsic(i, j, k, CONFORMAL_A13),
               extrinsic(i, j, k, CONFORMAL_A22),
               extrinsic(i, j, k, CONFORMAL_A23),
               extrinsic(i, j, k, CONFORMAL_A33),
               0.0,
               generated::xcts_lapse_from_conformal_unknowns(
                  psi(i, j, k, 0), lapse_psi(i, j, k, 0)
               ),
               0.0,
               0.0,
               beta(i, j, k, 0)
            )
         );
      });
   }
   amrex::ignore_unused(hierarchy);
}

void enforce_axisymmetric_radial_vertex_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& cell_field,
   InitialDataEllipticBoundaryConditions const& boundary,
   amrex::Real asymptotic_value,
   amrex::MultiFab& vertex_field
)
{
   const InitialDataCoordinateLine radial = hierarchy.coordinate_line(0, 0);
   const amrex::Box cell_domain = hierarchy.geometry[0].Domain();
   const amrex::Box vertex_domain = amrex::surroundingNodes(cell_domain);
   const amrex::Dim3 cell_lo = amrex::lbound(cell_domain);
   const amrex::Dim3 cell_hi = amrex::ubound(cell_domain);
   const amrex::Dim3 vertex_lo = amrex::lbound(vertex_domain);
   const amrex::Dim3 vertex_hi = amrex::ubound(vertex_domain);
   const bool has_inner_shell = radial.faces.front() > 0.0;
   const amrex::Real inner_width = radial.widths.front();
   const amrex::Real outer_width = radial.widths.back();
   for (amrex::MFIter mfi(vertex_field); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto cell = cell_field.const_array(mfi);
      const auto coefficient_a = boundary.coefficient_a(0).const_array(mfi);
      const auto coefficient_b = boundary.coefficient_b(0).const_array(mfi);
      const auto coefficient_c = boundary.coefficient_c(0).const_array(mfi);
      const auto vertex = vertex_field.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const bool lower = has_inner_shell && i == vertex_lo.x;
         const bool upper = i == vertex_hi.x;
         if (!lower && !upper) {
            return;
         }
         const int cell_i = lower ? cell_lo.x : cell_hi.x;
         const int ghost_i = lower ? cell_lo.x - 1 : cell_hi.x + 1;
         const amrex::Real width = lower ? inner_width : outer_width;
         const int first_j = amrex::max(cell_lo.y, amrex::min(j - 1, cell_hi.y));
         const int second_j = amrex::max(cell_lo.y, amrex::min(j, cell_hi.y));
         amrex::Real face_sum = 0.0;
         int face_count = 0;
#if (AMREX_SPACEDIM == 3)
         const int first_k = k - 1;
         const int second_k = k;
#else
         const int first_k = k;
         const int second_k = k;
#endif
         for (int cell_j = first_j; cell_j <= second_j; ++cell_j) {
            for (int cell_k = first_k; cell_k <= second_k; ++cell_k) {
               const amrex::Real a = coefficient_a(
                  ghost_i, cell_j, cell_k, 0
               );
               const amrex::Real b = coefficient_b(
                  ghost_i, cell_j, cell_k, 0
               );
               const amrex::Real c = coefficient_c(
                  ghost_i, cell_j, cell_k, 0
               ) + a * asymptotic_value;
               const amrex::Real denominator = lower
                  ? 0.5 * a - b / width
                  : 0.5 * a + b / width;
               const amrex::Real delta = lower
                  ? -(0.5 * a + b / width) / denominator
                  : (b / width - 0.5 * a) / denominator;
               const amrex::Real gamma = c / denominator;
               face_sum += 0.5 * (
                  (1.0 + delta) * cell(cell_i, cell_j, cell_k, 0)
                  + gamma
               );
               ++face_count;
            }
         }
         vertex(i, j, k, 0) = face_sum / face_count;
      });
   }
}

void materialize_axisymmetric_vertex_geometry(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab& conformal_factor,
   amrex::MultiFab& lapse_times_conformal_factor,
   amrex::MultiFab& azimuthal_shift,
   amrex::MultiFab const& conformal_extrinsic,
   InitialDataEllipticBoundaryConditions const& scalar_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_boundary,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   amrex::MultiFab& spacetime_state
)
{
   const amrex::BoxArray vertex_grids = amrex::convert(
      hierarchy.grids[0], amrex::IntVect::TheNodeVector()
   );
   amrex::MultiFab vertex_psi(
      vertex_grids, hierarchy.distribution[0], 1, 1
   );
   amrex::MultiFab vertex_lapse_psi(
      vertex_grids, hierarchy.distribution[0], 1, 1
   );
   amrex::MultiFab vertex_beta(
      vertex_grids, hierarchy.distribution[0], 1, 1
   );
   const auto scalar_boundary_records = initial_data_boundary_records(
      hierarchy, scalar_boundary
   );
   const auto lapse_boundary_records = initial_data_boundary_records(
      hierarchy, lapse_boundary
   );
   const auto shift_boundary_records = initial_data_boundary_records(
      hierarchy, shift_boundary
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_psi},
      amrex::Vector<amrex::MultiFab const*>{&conformal_factor},
      scalar_boundary_records,
      2
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_lapse_psi},
      amrex::Vector<amrex::MultiFab const*>{&lapse_times_conformal_factor},
      lapse_boundary_records,
      2
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_beta},
      amrex::Vector<amrex::MultiFab const*>{&azimuthal_shift},
      shift_boundary_records,
      2
   );
   detail::fill_initial_data_vertex_ghosts(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_psi},
      scalar_boundary_records
   );
   detail::fill_initial_data_vertex_ghosts(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_lapse_psi},
      lapse_boundary_records
   );
   detail::fill_initial_data_vertex_ghosts(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_beta},
      shift_boundary_records
   );
   enforce_axisymmetric_radial_vertex_boundary(
      hierarchy,
      conformal_factor,
      scalar_boundary,
      1.0,
      vertex_psi
   );
   enforce_axisymmetric_radial_vertex_boundary(
      hierarchy,
      lapse_times_conformal_factor,
      lapse_boundary,
      1.0,
      vertex_lapse_psi
   );
   enforce_axisymmetric_radial_vertex_boundary(
      hierarchy,
      azimuthal_shift,
      shift_boundary,
      0.0,
      vertex_beta
   );
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   if (radial_coordinates.centers.size() < 3) {
      amrex::Abort(
         "axisymmetric vertex geometry requires at least three radial cells"
      );
   }
   amrex::Gpu::AsyncArray<amrex::Real> radial_face_storage(
      radial_coordinates.faces.data(), radial_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   const amrex::Real* radial_faces = radial_face_storage.data();
   const amrex::Real* radial_centers = radial_center_storage.data();
   const amrex::Dim3 cell_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Dim3 cell_hi = amrex::ubound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Dim3 domain_lo = amrex::lbound(
      amrex::surroundingNodes(hierarchy.geometry[0].Domain())
   );
   const amrex::Dim3 domain_hi = amrex::ubound(
      amrex::surroundingNodes(hierarchy.geometry[0].Domain())
   );
   const amrex::Real inverse_two_dtheta = 0.5 / spacing[1];
   const amrex::Real upper_theta = hierarchy.geometry[0].ProbHi(1);
   const bool upper_is_axis = std::abs(upper_theta - constants::pi)
      <= 64.0 * std::numeric_limits<amrex::Real>::epsilon();
   for (amrex::MFIter mfi(spacetime_state, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto psi = vertex_psi.const_array(mfi);
      const auto lapse_psi = vertex_lapse_psi.const_array(mfi);
      const auto beta = vertex_beta.array(mfi);
      const auto cell_extrinsic = conformal_extrinsic.const_array(mfi);
      const auto geometry = spacetime_state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real radius = radial_faces[i - domain_lo.x];
         const amrex::Real theta = problo[1]
            + (j - domain_lo.y) * spacing[1];
         const bool radial_origin = radial_faces[0] == 0.0
            && i == domain_lo.x;
         const bool angular_axis = j == domain_lo.y
            || (upper_is_axis && j == domain_hi.y);
         if (radial_origin || angular_axis) {
            beta(i, j, k, 0) = 0.0;
         }
         amrex::Real a_r_phi = 0.0;
         amrex::Real a_theta_phi = 0.0;
         if (!radial_origin && !angular_axis) {
            const bool lapse_zero_inner_boundary = i == domain_lo.x
               && radial_faces[0] > 0.0
               && lapse_psi(i, j, k, 0) == 0.0;
            if (lapse_zero_inner_boundary) {
               // On an inversion-symmetric maximal throat, alpha*psi and
               // the longitudinal shift both vanish while their quotient
               // has a finite limit. The XCTS solve owns that limit at cell
               // centers, where alpha*psi is positive. Materialize the
               // solved conformal extrinsic curvature to the exact throat
               // instead of independently evaluating its algebraic 0/0
               // representation at the boundary node.
               a_r_phi =
                  extrapolate_axisymmetric_cell_component_to_inner_vertex(
                     cell_extrinsic,
                     j,
                     k,
                     CONFORMAL_A13,
                     cell_lo,
                     cell_hi,
                     radial_centers,
                     radial_faces[0]
                  );
               a_theta_phi =
                  extrapolate_axisymmetric_cell_component_to_inner_vertex(
                     cell_extrinsic,
                     j,
                     k,
                     CONFORMAL_A23,
                     cell_lo,
                     cell_hi,
                     radial_centers,
                     radial_faces[0]
                  );
            } else {
               const amrex::Real d_beta_r = axisymmetric_nodal_derivative(
                  beta, i, j, k, 0, domain_lo, domain_hi,
                  radial_faces, 1.0 / spacing[1]
               );
               const amrex::Real d_beta_theta = j == domain_hi.y
                  ? 0.0
                  : (beta(i, j + 1, k, 0) - beta(i, j - 1, k, 0))
                       * inverse_two_dtheta;
               const amrex::Real scale =
                  generated::xcts_conformal_extrinsic_scale(
                     psi(i, j, k, 0), lapse_psi(i, j, k, 0)
                  );
               a_r_phi = scale * (
                  d_beta_r - beta(i, j, k, 0) / radius
               );
               a_theta_phi = scale * (
                  d_beta_theta - beta(i, j, k, 0) / std::tan(theta)
               ) / radius;
            }
         }
         store_z4c_state(
            geometry,
            i,
            j,
            k,
            generated::ctt_canonical_z4c_state(
               psi(i, j, k, 0),
               0.0,
               0.0,
               a_r_phi,
               0.0,
               a_theta_phi,
               0.0,
               0.0,
               generated::xcts_lapse_from_conformal_unknowns(
                  psi(i, j, k, 0), lapse_psi(i, j, k, 0)
               ),
               0.0,
               0.0,
               beta(i, j, k, 0)
            )
         );
      });
   }
   // Nodal valid regions overlap at AMReX box boundaries.  The derivative
   // inputs above are ghost-filled from one synchronized owner, and the
   // completed ADM/Z4c field must retain that same single-valued contract for
   // portable output and later materialization.
   spacetime_state.OverrideSync(hierarchy.geometry[0].periodicity());
}

template <int EOSMethod>
amrex::Real axisymmetric_matter_outer_radius(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& primitive_state,
   InitialAtmosphereParameters const& atmosphere
)
{
   amrex::MultiFab support(
      hierarchy.grids[0], hierarchy.distribution[0], 1, 0
   );
   support.setVal(0.0);
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   amrex::Gpu::AsyncArray<amrex::Real> radial_face_storage(
      radial_coordinates.faces.data(), radial_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_center_storage(
      angular_coordinates.centers.data(), angular_coordinates.centers.size()
   );
   const amrex::Real* radial_faces = radial_face_storage.data();
   const amrex::Real* radial_center = radial_center_storage.data();
   const amrex::Real* angular_center = angular_center_storage.data();
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
#if (AMREX_SPACEDIM == 3)
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta phi");
   const amrex::Real phi_lo = hierarchy.geometry[0].ProbLo(2);
   const amrex::Real dphi = hierarchy.geometry[0].CellSize(2);
#else
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
   const amrex::Real phi_lo = 0.0;
   const amrex::Real dphi = 0.0;
#endif
   for (amrex::MFIter mfi(support, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state.const_array(mfi);
      const auto output = support.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const int radial_index = i - domain_lo.x;
         const int angular_index = j - domain_lo.y;
         const PatchCoordinate mesh{
            radial_center[radial_index],
            angular_center[angular_index],
            phi_lo + (k - domain_lo.z + 0.5) * dphi,
         };
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(MeshSpherical, axes, mesh),
         };
         const EOSThermodynamicState ambient =
            initial_atmosphere_state<EOSMethod>(atmosphere, sample);
         if (initial_matter_exceeds_atmosphere(
                primitive(i, j, k, RHO), ambient.rho
             )) {
            output(i, j, k, 0) = radial_faces[radial_index + 1];
         }
      });
   }
   return support.max(0, 0, false);
}

template <int EOSMethod>
void update_axisymmetric_global_diagnostics(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& primitive_state,
   amrex::MultiFab const& cell_spacetime,
   amrex::MultiFab const& conformal_factor,
   amrex::MultiFab const& conformal_extrinsic,
   amrex::MultiFab const& energy_density,
   amrex::MultiFab const& azimuthal_momentum_density,
   EOSParams const& eos,
   RotatingStarUpdateSummary& summary
)
{
   amrex::MultiFab integrands(
      hierarchy.grids[0], hierarchy.distribution[0], 3, 0
   );
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   amrex::Gpu::AsyncArray<amrex::Real> radial_face_storage(
      radial_coordinates.faces.data(), radial_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_face_storage(
      angular_coordinates.faces.data(), angular_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_center_storage(
      angular_coordinates.centers.data(), angular_coordinates.centers.size()
   );
   const amrex::Real* radial_faces = radial_face_storage.data();
   const amrex::Real* angular_faces = angular_face_storage.data();
   const amrex::Real* radial_center = radial_center_storage.data();
   const amrex::Real* angular_center = angular_center_storage.data();
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Real theta_lo = hierarchy.geometry[0].ProbLo(1);
   const amrex::Real theta_hi = hierarchy.geometry[0].ProbHi(1);
   const amrex::Real half_pi = 0.5 * constants::pi;
   const amrex::Real theta_symmetry_factor =
      (std::abs(theta_hi - theta_lo - half_pi)
      <= 64.0 * std::numeric_limits<amrex::Real>::epsilon())
      ? 2.0 : 1.0;
#if (AMREX_SPACEDIM == 3)
   const amrex::Real azimuthal_measure = hierarchy.geometry[0].CellSize(2);
#else
   const amrex::Real azimuthal_measure = constants::two_pi;
#endif
   for (amrex::MFIter mfi(integrands, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state.const_array(mfi);
      const auto geometry = cell_spacetime.const_array(mfi);
      const auto psi = conformal_factor.const_array(mfi);
      const auto extrinsic = conformal_extrinsic.const_array(mfi);
      const auto energy = energy_density.const_array(mfi);
      const auto momentum = azimuthal_momentum_density.const_array(mfi);
      const auto output = integrands.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const int radial_index = i - domain_lo.x;
         const int angular_index = j - domain_lo.y;
         const amrex::Real radius = radial_center[radial_index];
         const amrex::Real theta = angular_center[angular_index];
         const amrex::Real flat_volume = azimuthal_measure
            * theta_symmetry_factor
            * (radial_faces[radial_index + 1]
                  * radial_faces[radial_index + 1]
                  * radial_faces[radial_index + 1]
               - radial_faces[radial_index] * radial_faces[radial_index]
                  * radial_faces[radial_index]) / 3.0
            * (std::cos(angular_faces[angular_index])
               - std::cos(angular_faces[angular_index + 1]));
         const amrex::Real psi_value = psi(i, j, k, 0);
         const amrex::Real psi_squared = psi_value * psi_value;
         const amrex::Real psi_sixth = psi_squared * psi_squared
            * psi_squared;
         const amrex::Real psi_fifth = psi_sixth / psi_value;
         const amrex::Real psi_minus_seventh = 1.0 / (
            psi_sixth * psi_value
         );
         const RHDPrimitive fluid = rhd_primitive_at<EOSMethod>(
            primitive, i, j, k, eos
         );
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const amrex::Real velocity_squared = spatial_vector_norm_squared(
            fluid.v1, fluid.v2, fluid.v3, metric
         );
         const amrex::Real lorentz_factor = 1.0 / std::sqrt(
            1.0 - velocity_squared
         );
         output(i, j, k, 0) = flat_volume * (
            psi_fifth * energy(i, j, k, 0)
            + psi_minus_seventh
               * extrinsic(i, j, k, CONFORMAL_A_SQUARED)
               / (16.0 * constants::pi)
         );
         output(i, j, k, 1) = flat_volume * psi_sixth
            * momentum(i, j, k, 0) * radius * std::sin(theta);
         output(i, j, k, 2) = flat_volume * psi_sixth
            * fluid.rho * lorentz_factor;
      });
   }
   std::array<amrex::Real, 3> global_integrals{
      integrands.sum(0, true),
      integrands.sum(1, true),
      integrands.sum(2, true),
   };
   amrex::ParallelDescriptor::ReduceRealSum(
      global_integrals.data(), static_cast<int>(global_integrals.size())
   );
   summary.adm_mass = global_integrals[0];
   summary.angular_momentum = global_integrals[1];
   summary.baryonic_mass = global_integrals[2];
   summary.dimensionless_spin = summary.angular_momentum / (
      summary.adm_mass * summary.adm_mass
   );
   if (!(summary.adm_mass > 0.0) || !(summary.baryonic_mass > 0.0)
       || !std::isfinite(summary.angular_momentum)
       || !std::isfinite(summary.dimensionless_spin)) {
      amrex::Abort("axisymmetric rotating-star global diagnostic is invalid");
   }
}

template <int EOSMethod>
void update_axisymmetric_equilibrium_torus_global_diagnostics(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& primitive_state,
   amrex::MultiFab const& cell_spacetime,
   amrex::MultiFab const& conformal_factor,
   amrex::MultiFab const& azimuthal_momentum_density,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters,
   EquilibriumTorusUpdateSummary& summary
)
{
   if (!parameters.active) {
      return;
   }
   amrex::MultiFab integrands(
      hierarchy.grids[0], hierarchy.distribution[0], 2, 0
   );
   integrands.setVal(0.0);
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   amrex::Gpu::AsyncArray<amrex::Real> radial_face_storage(
      radial_coordinates.faces.data(), radial_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_face_storage(
      angular_coordinates.faces.data(), angular_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_center_storage(
      angular_coordinates.centers.data(), angular_coordinates.centers.size()
   );
   const amrex::Real* radial_faces = radial_face_storage.data();
   const amrex::Real* angular_faces = angular_face_storage.data();
   const amrex::Real* radial_center = radial_center_storage.data();
   const amrex::Real* angular_center = angular_center_storage.data();
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Real theta_extent = hierarchy.geometry[0].ProbHi(1)
      - hierarchy.geometry[0].ProbLo(1);
   const amrex::Real theta_symmetry_factor =
      std::abs(theta_extent - 0.5 * constants::pi)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon()
      ? 2.0 : 1.0;
#if (AMREX_SPACEDIM == 3)
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta phi");
   const amrex::Real phi_lo = hierarchy.geometry[0].ProbLo(2);
   const amrex::Real dphi = hierarchy.geometry[0].CellSize(2);
   const amrex::Real azimuthal_measure = dphi;
#else
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
   const amrex::Real phi_lo = 0.0;
   const amrex::Real dphi = 0.0;
   const amrex::Real azimuthal_measure = constants::two_pi;
#endif
   for (amrex::MFIter mfi(integrands, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state.const_array(mfi);
      const auto geometry = cell_spacetime.const_array(mfi);
      const auto psi = conformal_factor.const_array(mfi);
      const auto momentum = azimuthal_momentum_density.const_array(mfi);
      const auto output = integrands.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const int radial_index = i - domain_lo.x;
         const int angular_index = j - domain_lo.y;
         const amrex::Real radius = radial_center[radial_index];
         if (radius < parameters.inner_radius) {
            return;
         }
         const amrex::Real theta = angular_center[angular_index];
         const PatchCoordinate mesh{
            radius,
            theta,
            phi_lo + (k - domain_lo.z + 0.5) * dphi,
         };
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(MeshSpherical, axes, mesh),
         };
         const EOSThermodynamicState atmosphere =
            initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
         const RHDPrimitive fluid = rhd_primitive_at<EOSMethod>(
            primitive, i, j, k, eos
         );
         if (!initial_matter_exceeds_atmosphere(
                fluid.rho, atmosphere.rho
             )) {
            return;
         }
         const amrex::Real flat_volume = azimuthal_measure
            * theta_symmetry_factor
            * (radial_faces[radial_index + 1]
                  * radial_faces[radial_index + 1]
                  * radial_faces[radial_index + 1]
               - radial_faces[radial_index]
                  * radial_faces[radial_index]
                  * radial_faces[radial_index]) / 3.0
            * (std::cos(angular_faces[angular_index])
               - std::cos(angular_faces[angular_index + 1]));
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const amrex::Real velocity_squared = spatial_vector_norm_squared(
            fluid.v1, fluid.v2, fluid.v3, metric
         );
         const amrex::Real lorentz_factor = 1.0 / std::sqrt(
            1.0 - velocity_squared
         );
         const amrex::Real psi_value = psi(i, j, k, 0);
         const amrex::Real psi_squared = psi_value * psi_value;
         const amrex::Real psi_sixth = psi_squared * psi_squared
            * psi_squared;
         output(i, j, k, 0) = flat_volume * psi_sixth
            * fluid.rho * lorentz_factor;
         output(i, j, k, 1) = flat_volume * psi_sixth
            * momentum(i, j, k, 0) * radius * std::sin(theta);
      });
   }
   std::array<amrex::Real, 2> global_integrals{
      integrands.sum(0, true),
      integrands.sum(1, true),
   };
   amrex::ParallelDescriptor::ReduceRealSum(
      global_integrals.data(), static_cast<int>(global_integrals.size())
   );
   summary.baryonic_mass = global_integrals[0];
   summary.angular_momentum = global_integrals[1];
   if (!(summary.baryonic_mass > 0.0)
       || !std::isfinite(summary.angular_momentum)) {
      amrex::Abort("axisymmetric equilibrium-torus diagnostics are invalid");
   }
}
#endif

#if (AMREX_SPACEDIM == 3)
template <PhysicalSystem System, int EOSMethod>
void update_equilibrium_torus_global_diagnostics(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters,
   EquilibriumTorusUpdateSummary& summary
)
{
   if (!parameters.active) {
      return;
   }
   summary.baryonic_mass = 0.0;
   summary.angular_momentum = 0.0;
   const int finest_level = static_cast<int>(hierarchy.geometry.size()) - 1;
   for (int level = 0; level <= finest_level; ++level) {
      amrex::iMultiFab valid(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 0
      );
      if (level < finest_level) {
         valid = amrex::makeFineMask(
            *primitive_state[level],
            hierarchy.grids[level + 1],
            hierarchy.refinement_ratio[level],
            hierarchy.geometry[level].periodicity(),
            1,
            0
         );
      } else {
         valid.setVal(1);
      }
      amrex::MultiFab integrands(
         hierarchy.grids[level], hierarchy.distribution[level], 2, 0
      );
      integrands.setVal(0.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Real cell_volume = dx[0] * dx[1] * dx[2];
      for (amrex::MFIter mfi(integrands, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_state[level]->const_array(mfi);
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto valid_cell = valid.const_array(mfi);
         const auto output = integrands.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            if (valid_cell(i, j, k, 0) == 0) {
               return;
            }
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - domain_lo.x) + 0.5)
                  * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - domain_lo.y) + 0.5)
                  * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - domain_lo.z) + 0.5)
                  * dx[2],
            };
            const amrex::Real r1 = physical.q1 - parameters.center1;
            const amrex::Real r2 = physical.q2 - parameters.center2;
            const amrex::Real r3 = physical.q3 - parameters.center3;
            if (std::sqrt(r1 * r1 + r2 * r2 + r3 * r3)
                < parameters.inner_radius) {
               return;
            }
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
            const RHDPrimitive fluid = rhd_primitive_at<EOSMethod>(
               primitive, i, j, k, eos
            );
            if (!initial_matter_exceeds_atmosphere(
                   fluid.rho, atmosphere.rho
                )) {
               return;
            }
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            EulerianMatter matter;
            if constexpr (System == PhysicalSystem::RHD) {
               matter = generated::rhd_eulerian_matter(fluid, metric);
            } else {
                  matter = generated::rmhd_eulerian_matter(
                     rmhd_primitive_at<EOSMethod>(primitive, i, j, k, eos),
                     metric
               );
            }
            const amrex::Real velocity_squared = spatial_vector_norm_squared(
               fluid.v1, fluid.v2, fluid.v3, metric
            );
            const amrex::Real lorentz_factor = 1.0 / std::sqrt(
               1.0 - velocity_squared
            );
            const amrex::Real angular_momentum1 =
               r2 * matter.momentum3 - r3 * matter.momentum2;
            const amrex::Real angular_momentum2 =
               r3 * matter.momentum1 - r1 * matter.momentum3;
            const amrex::Real angular_momentum3 =
               r1 * matter.momentum2 - r2 * matter.momentum1;
            output(i, j, k, 0) = cell_volume * metric.sqrt_det_gamma
               * fluid.rho * lorentz_factor;
            output(i, j, k, 1) = cell_volume * metric.sqrt_det_gamma * (
               parameters.axis1 * angular_momentum1
               + parameters.axis2 * angular_momentum2
               + parameters.axis3 * angular_momentum3
            );
         });
      }
      summary.baryonic_mass += integrands.sum(0, true);
      summary.angular_momentum += integrands.sum(1, true);
   }
   std::array<amrex::Real, 2> global_integrals{
      summary.baryonic_mass,
      summary.angular_momentum,
   };
   amrex::ParallelDescriptor::ReduceRealSum(
      global_integrals.data(), static_cast<int>(global_integrals.size())
   );
   summary.baryonic_mass = global_integrals[0];
   summary.angular_momentum = global_integrals[1];
   // Configuration admits a rotating disk on a reflected Cartesian domain
   // only when every reflection is its equatorial plane.  The projected
   // angular momentum therefore has the same even parity as the mass, so both
   // reductions reconstruct the omitted physical sectors with one factor.
   const amrex::Real volume_factor = reflected_domain_volume_factor(hierarchy);
   summary.baryonic_mass *= volume_factor;
   summary.angular_momentum *= volume_factor;
   if (!(summary.baryonic_mass > 0.0)
       || !std::isfinite(summary.angular_momentum)) {
      amrex::Abort("3D equilibrium-torus diagnostics are invalid");
   }
}
#endif

template <PhysicalSystem System, int EOSMethod>
ConformalMaterializationResult materialize_conformal_initial_data_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   EOSParams const& eos,
   InitialDataFieldSelection const& fields,
   ConformalConstraintFormulation formulation,
   RotatingStarParameters const& rotating_star,
   EquilibriumTorusParameters const& equilibrium_torus,
   bool seed_from_spacetime,
   ConformalContinuationObserver const& continuation_observer,
   ConformalMatterObserver const& matter_observer,
   ConformalStellarMatterUpdater const& stellar_matter_updater
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   amrex::ignore_unused(fields);
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(primitive_state.size()) != num_levels
       || static_cast<int>(spacetime_state.size()) != num_levels
       || static_cast<int>(cell_spacetime.size()) != num_levels) {
      amrex::Abort("invalid conformal initial-data hierarchy materialization");
   }
   amrex::Real maximum_cell_half_diagonal = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      maximum_cell_half_diagonal = std::max(
         maximum_cell_half_diagonal,
         amrex::Real(0.5) * std::sqrt(
            AMREX_D_TERM(
               dx[0] * dx[0], + dx[1] * dx[1], + dx[2] * dx[2]
            )
         )
      );
   }

   amrex::Vector<std::unique_ptr<amrex::MultiFab>> conformal_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vector_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> lapse_conformal_storage(
      num_levels
   );
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> shift_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> curvature_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> extrinsic_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> energy_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> momentum_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> stress_trace_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_energy_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_momentum_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> vertex_stress_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_conformal(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_vector(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_lapse(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_curvature(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_primitive(num_levels);
   amrex::Vector<amrex::MultiFab*> conformal_factor(num_levels);
   amrex::Vector<amrex::MultiFab*> vector_potential(num_levels);
   amrex::Vector<amrex::MultiFab*> xcts_shift(num_levels);
   amrex::Vector<amrex::MultiFab*> lapse_times_conformal_factor(num_levels);
   amrex::Vector<amrex::MultiFab const*> lapse_conformal_view(num_levels);
   amrex::Vector<amrex::MultiFab const*> shift_view(num_levels);
   amrex::Vector<amrex::MultiFab*> mean_curvature_mutable(num_levels);
   amrex::Vector<amrex::MultiFab const*> mean_curvature(num_levels);
   amrex::Vector<amrex::MultiFab*> conformal_extrinsic(num_levels);
   amrex::Vector<amrex::MultiFab const*> conformal_factor_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> conformal_extrinsic_const(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_energy(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_momentum(num_levels);
   amrex::Vector<amrex::MultiFab*> cell_stress(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_energy_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_momentum_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> cell_stress_trace(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_energy(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_momentum(num_levels);
   amrex::Vector<amrex::MultiFab*> vertex_stress(num_levels);
   amrex::Vector<amrex::MultiFab const*> energy_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> momentum_density(num_levels);
   amrex::Vector<amrex::MultiFab const*> stress_trace(num_levels);

   for (int level = 0; level < num_levels; ++level) {
      if (primitive_state[level] == nullptr
          || spacetime_state[level] == nullptr || cell_spacetime[level] == nullptr) {
         amrex::Abort(
            "conformal initial data received an unallocated hierarchy level"
         );
      }
      const amrex::BoxArray vertex_grids = amrex::convert(
         hierarchy.grids[level], amrex::IntVect::TheNodeVector()
      );
      if (primitive_state[level]->boxArray() != hierarchy.grids[level]
          || cell_spacetime[level]->boxArray() != hierarchy.grids[level]
          || spacetime_state[level]->boxArray() != vertex_grids) {
         amrex::Abort(
            "conformal initial-data fields do not match the staggered hierarchy"
         );
      }
      conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      vector_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 1
      );
      lapse_conformal_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      shift_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 1
      );
      curvature_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 1
      );
      extrinsic_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level],
         CONFORMAL_A_NUM_COMPONENTS, 1
      );
      energy_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      momentum_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 3, 1
      );
      stress_trace_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      vertex_energy_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      vertex_momentum_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 0
      );
      vertex_stress_storage[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_conformal[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_vector[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 3, 0
      );
      previous_lapse[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_curvature[level] = std::make_unique<amrex::MultiFab>(
         vertex_grids, hierarchy.distribution[level], 1, 0
      );
      previous_primitive[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level],
         hierarchy.distribution[level],
         primitive_state[level]->nComp(),
         0
      );
      conformal_storage[level]->setVal(1.0);
      vector_storage[level]->setVal(0.0);
      lapse_conformal_storage[level]->setVal(1.0);
      shift_storage[level]->setVal(0.0);
      curvature_storage[level]->setVal(0.0);
      extrinsic_storage[level]->setVal(0.0);
      conformal_factor[level] = conformal_storage[level].get();
      vector_potential[level] = vector_storage[level].get();
      xcts_shift[level] = vector_storage[level].get();
      lapse_times_conformal_factor[level] = lapse_conformal_storage[level].get();
      lapse_conformal_view[level] = formulation != ConformalConstraintFormulation::xcts
         ? conformal_storage[level].get()
         : lapse_conformal_storage[level].get();
      shift_view[level] = formulation != ConformalConstraintFormulation::xcts
         ? shift_storage[level].get()
         : vector_storage[level].get();
      mean_curvature_mutable[level] = curvature_storage[level].get();
      mean_curvature[level] = curvature_storage[level].get();
      conformal_extrinsic[level] = extrinsic_storage[level].get();
      conformal_factor_const[level] = conformal_storage[level].get();
      conformal_extrinsic_const[level] = extrinsic_storage[level].get();
      cell_energy[level] = energy_storage[level].get();
      cell_momentum[level] = momentum_storage[level].get();
      cell_stress[level] = stress_trace_storage[level].get();
      cell_energy_density[level] = energy_storage[level].get();
      cell_momentum_density[level] = momentum_storage[level].get();
      cell_stress_trace[level] = stress_trace_storage[level].get();
      vertex_energy[level] = vertex_energy_storage[level].get();
      vertex_momentum[level] = vertex_momentum_storage[level].get();
      vertex_stress[level] = vertex_stress_storage[level].get();
      energy_density[level] = vertex_energy_storage[level].get();
      momentum_density[level] = vertex_momentum_storage[level].get();
      stress_trace[level] = vertex_stress_storage[level].get();

      if (seed_from_spacetime) {
         if (formulation != ConformalConstraintFormulation::xcts) {
            amrex::Abort(
               "saved conformal initial-data seeds currently require XCTS"
            );
         }
         for (amrex::MFIter mfi(
                 *spacetime_state[level], amrex::TilingIfNotGPU()
              ); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto saved = spacetime_state[level]->const_array(mfi);
            const auto psi = conformal_storage[level]->array(mfi);
            const auto lapse_psi = lapse_conformal_storage[level]->array(mfi);
            const auto shift = vector_storage[level]->array(mfi);
            const auto curvature = curvature_storage[level]->array(mfi);
            const auto extrinsic = extrinsic_storage[level]->array(mfi);
            amrex::ParallelFor(
               box,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const Z4cState state = load_z4c_state(saved, i, j, k);
                  const amrex::Real conformal_factor =
                     1.0 / std::sqrt(std::sqrt(state.chi));
                  const amrex::Real conformal_factor_squared =
                     conformal_factor * conformal_factor;
                  const amrex::Real conformal_factor_sixth =
                     conformal_factor_squared * conformal_factor_squared
                     * conformal_factor_squared;
                  const amrex::Real conformal_a11 =
                     state.a_tilde11 * conformal_factor_sixth;
                  const amrex::Real conformal_a12 =
                     state.a_tilde12 * conformal_factor_sixth;
                  const amrex::Real conformal_a13 =
                     state.a_tilde13 * conformal_factor_sixth;
                  const amrex::Real conformal_a22 =
                     state.a_tilde22 * conformal_factor_sixth;
                  const amrex::Real conformal_a23 =
                     state.a_tilde23 * conformal_factor_sixth;
                  const amrex::Real conformal_a33 =
                     state.a_tilde33 * conformal_factor_sixth;
                  psi(i, j, k, 0) = conformal_factor;
                  lapse_psi(i, j, k, 0) = state.lapse * conformal_factor;
                  shift(i, j, k, 0) = state.shift1;
                  shift(i, j, k, 1) = state.shift2;
                  shift(i, j, k, 2) = state.shift3;
                  curvature(i, j, k, 0) = state.k_hat + 2.0 * state.theta;
                  extrinsic(i, j, k, CONFORMAL_A11) = conformal_a11;
                  extrinsic(i, j, k, CONFORMAL_A12) = conformal_a12;
                  extrinsic(i, j, k, CONFORMAL_A13) = conformal_a13;
                  extrinsic(i, j, k, CONFORMAL_A22) = conformal_a22;
                  extrinsic(i, j, k, CONFORMAL_A23) = conformal_a23;
                  extrinsic(i, j, k, CONFORMAL_A33) = conformal_a33;
                  extrinsic(i, j, k, CONFORMAL_A_SQUARED) =
                     conformal_a11 * conformal_a11
                     + conformal_a22 * conformal_a22
                     + conformal_a33 * conformal_a33
                     + 2.0 * (
                        conformal_a12 * conformal_a12
                        + conformal_a13 * conformal_a13
                        + conformal_a23 * conformal_a23
                     );
               }
            );
         }
      }
   }

   auto conformal_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );
   set_initial_data_reflection_boundaries(
      hierarchy, conformal_boundary, scalar_symmetries(1)
   );
   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> elliptic_center{{
      AMREX_D_DECL(
         rotating_star.rotation_center.q1,
         rotating_star.rotation_center.q2,
         rotating_star.rotation_center.q3
      )
   }};
   set_centered_power_law_robin_boundary(
      hierarchy,
      conformal_boundary,
      amrex::Vector<amrex::Real>{0.0},
      1.0,
      &elliptic_center
   );
   auto lapse_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );
   set_initial_data_reflection_boundaries(
      hierarchy, lapse_boundary, scalar_symmetries(1)
   );
   set_centered_power_law_robin_boundary(
      hierarchy,
      lapse_boundary,
      amrex::Vector<amrex::Real>{0.0},
      1.0,
      &elliptic_center
   );
   auto vector_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      3,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::vertex
   );
   set_initial_data_reflection_boundaries(
      hierarchy, vector_boundary, vector_symmetries()
   );

   constexpr int max_constraint_iterations = 100;
   CttOptions options;
   options.max_coupled_iterations = max_constraint_iterations;
   options.relative_update_tolerance = 1.0e-10;
   options.relaxation = 0.5;
   options.linear.boundary_interpolation_order = 4;
   options.linear.relative_tolerance = 1.0e-12;
   XctsOptions xcts_options;
   xcts_options.max_coupled_iterations = options.max_coupled_iterations;
   xcts_options.relative_update_tolerance = options.relative_update_tolerance;
   xcts_options.relaxation = options.relaxation;
   xcts_options.linear = options.linear;
   CttkOptions cttk_options;
   cttk_options.max_coupled_iterations = options.max_coupled_iterations;
   cttk_options.relative_update_tolerance = options.relative_update_tolerance;
   cttk_options.relaxation = options.relaxation;
   cttk_options.linear = options.linear;

   ConformalMaterializationResult result;
   const bool coupled_equilibrium_torus = equilibrium_torus.active
      && equilibrium_torus.gravity == EquilibriumTorusGravity::Coupled;
   const bool update_stellar_matter = stellar_matter_updater
      || (rotating_star.active
          && (rotating_star.angular_velocity != 0.0
              || coupled_equilibrium_torus));
   const bool iterate_matter = update_stellar_matter
      || coupled_equilibrium_torus;
   const int max_matter_iterations = iterate_matter
      ? rotating_star.max_iterations * (coupled_equilibrium_torus ? 2 : 1)
      : 12;
   const amrex::Real matter_tolerance = iterate_matter
      ? rotating_star.relative_tolerance
      : 1.0e-9;
   const CircularContinuationSchedule continuation_schedule =
      circular_continuation_schedule(rotating_star, equilibrium_torus);
   const bool coupled_xcts_matter_iteration =
      iterate_matter
      && formulation == ConformalConstraintFormulation::xcts;
   const auto materialize_current_geometry = [&]() {
      materialize_conformal_geometry(
         hierarchy,
         conformal_factor_const,
         lapse_conformal_view,
         shift_view,
         mean_curvature,
         conformal_extrinsic_const,
         spacetime_state,
         cell_spacetime
      );
   };
   const auto update_surface_diagnostic = [&](RotatingStarParameters parameters) {
      parameters.injection_energy = result.rotating_star.injection_energy;
      amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
      for (int level = 0; level < num_levels; ++level) {
         cell_spacetime_const[level] = cell_spacetime[level];
      }
      const RotatingStarSurfaceDiagnostic diagnostic =
         rotating_star_surface_diagnostic(
            hierarchy, cell_spacetime_const, parameters
         );
      result.rotating_star.equatorial_log_enthalpy_gradient =
         diagnostic.equatorial_log_enthalpy_gradient;
      result.rotating_star.polar_log_enthalpy_gradient =
         diagnostic.polar_log_enthalpy_gradient;
      result.rotating_star.mass_shedding_indicator =
         diagnostic.mass_shedding_indicator;
   };
   for (int continuation_step = 1;
        continuation_step <= static_cast<int>(
           continuation_schedule.stellar_angular_velocities.size()
        );
        ++continuation_step) {
      RotatingStarParameters continued_rotating_star = rotating_star;
      continued_rotating_star.angular_velocity =
         continuation_schedule.stellar_angular_velocities[
            continuation_step - 1
         ];
      EquilibriumTorusParameters continued_torus = equilibrium_torus;
      continued_torus.angular_velocity =
         continuation_schedule.disk_angular_velocities[
            continuation_step - 1
         ];
      if (coupled_equilibrium_torus) {
         continued_rotating_star.update_outer_radius =
            equilibrium_torus.inner_radius;
      }
      result.relative_matter_update = std::numeric_limits<amrex::Real>::max();
      int step_matter_iterations = 0;
      bool strict_xcts_polish = false;
      bool torus_activated = !coupled_equilibrium_torus;
      bool coupled_torus_updated =
         seed_from_spacetime && coupled_equilibrium_torus;
      bool stellar_matter_updated = false;
      for (int matter_iteration = 1;
           matter_iteration <= max_matter_iterations;
           ++matter_iteration) {
         materialize_conformal_geometry(
            hierarchy,
            conformal_factor_const,
            lapse_conformal_view,
            shift_view,
            mean_curvature,
            conformal_extrinsic_const,
            spacetime_state,
            cell_spacetime
         );
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Copy(
               *previous_primitive[level],
               *primitive_state[level],
               0,
               0,
               primitive_state[level]->nComp(),
               0
            );
         }
         // The conformal unknowns start from flat space.  Preserve the spherical
         // seed through the first constraint solve so the Euler update sees the
         // star's lapse and shift rather than erasing it in the flat seed metric.
         const bool preserve_spherical_seed =
            continuation_step == 1 && matter_iteration == 1;
         amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
         for (int level = 0; level < num_levels; ++level) {
            cell_spacetime_const[level] = cell_spacetime[level];
         }
         amrex::Real stellar_matter_external_update = 0.0;
         if (stellar_matter_updater) {
            const ConformalStellarMatterUpdateResult updated =
               stellar_matter_updater(
                  continued_rotating_star, !preserve_spherical_seed
               );
            stellar_matter_external_update = updated.relative_update;
            if (!preserve_spherical_seed) {
               result.rotating_star = updated.rotating_star;
               stellar_matter_updated = true;
            }
         } else if (update_stellar_matter && !preserve_spherical_seed) {
            result.rotating_star = update_rotating_star_primitives(
               hierarchy,
               primitive_state,
               cell_spacetime_const,
               EOSMethod,
               eos,
               continued_rotating_star
            );
            stellar_matter_updated = true;
         }
         if (update_stellar_matter && !preserve_spherical_seed) {
            if (amrex::ParallelDescriptor::IOProcessor()) {
               amrex::Print()
                  << "rotating continuation step=" << continuation_step
                  << "/"
                  << continuation_schedule.stellar_angular_velocities.size()
                  << " target_stellar_omega="
                  << continued_rotating_star.angular_velocity
                  << " target_disk_omega=" << continued_torus.angular_velocity
                  << " matter_iteration=" << matter_iteration
                  << " maximum_density=" << result.rotating_star.maximum_density
                  << " maximum_velocity=" << result.rotating_star.maximum_velocity
                  << " injection_energy=" << result.rotating_star.injection_energy
                  << " boundary_density_excess="
                  << result.rotating_star.boundary_density_excess << '\n';
            }
         }
         if (equilibrium_torus.active
             && result.rotating_star.matter_outer_radius
                   - maximum_cell_half_diagonal
                >= equilibrium_torus.inner_radius) {
            amrex::Abort(
               "equilibrium-torus inner radius must lie outside the central "
               "star matter support"
            );
         }
         if (coupled_equilibrium_torus && torus_activated) {
            continued_torus.continue_existing_matter =
               coupled_torus_updated;
            result.equilibrium_torus = update_equilibrium_torus_primitives(
               hierarchy,
               primitive_state,
               cell_spacetime_const,
               EOSMethod,
               eos,
               continued_torus
            );
            coupled_torus_updated =
               result.equilibrium_torus.maximum_density > 0.0;
         }
         if (matter_observer) {
            matter_observer();
         }
         for (int level = 0; level < num_levels; ++level) {
            if (primitive_state[level]->contains_nan(
                   0, primitive_state[level]->nComp(), 0
                )) {
               amrex::Abort("conformal initial-data primitive state is nonfinite");
            }
            if (cell_spacetime[level]->contains_nan(
                   0, cell_spacetime[level]->nComp(), 0
                )) {
               amrex::Abort("conformal initial-data cell geometry is nonfinite");
            }
            amrex::MultiFab::Copy(
               *previous_conformal[level], *conformal_storage[level], 0, 0, 1, 0
            );
            amrex::MultiFab::Copy(
               *previous_vector[level], *vector_storage[level], 0, 0, 3, 0
            );
            amrex::MultiFab::Copy(
               *previous_lapse[level],
               *lapse_conformal_storage[level],
               0,
               0,
               1,
               0
            );
            if (formulation == ConformalConstraintFormulation::cttk) {
               amrex::MultiFab::Copy(
                  *previous_curvature[level], *curvature_storage[level], 0, 0, 1, 0
               );
            }
            for (amrex::MFIter mfi(
                    *energy_storage[level], amrex::TilingIfNotGPU()
                 ); mfi.isValid(); ++mfi) {
               const amrex::Box box = mfi.tilebox();
               const auto primitive = primitive_state[level]->const_array(mfi);
               const auto cell_geometry = cell_spacetime[level]->const_array(mfi);
               const auto energy = energy_storage[level]->array(mfi);
               const auto momentum = momentum_storage[level]->array(mfi);
               const auto stress_trace_field = stress_trace_storage[level]->array(mfi);
               amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  const Z4cState geometry = load_z4c_state(
                     cell_geometry, i, j, k
                  );
                  const SpacetimeMetric metric = generated::z4c_spacetime_metric(geometry);
                  EulerianMatter matter;
                  if constexpr (System == PhysicalSystem::RHD) {
                     matter = generated::rhd_eulerian_matter(
                        rhd_primitive_at<EOSMethod>(primitive, i, j, k, eos), metric
                     );
                  } else {
                     matter = generated::rmhd_eulerian_matter(
                        rmhd_primitive_at<EOSMethod>(
                           primitive, i, j, k, eos
                        ),
                        metric
                     );
                  }
                  energy(i, j, k, 0) = matter.rho;
                  momentum(i, j, k, 0) = matter.momentum1;
                  momentum(i, j, k, 1) = matter.momentum2;
                  momentum(i, j, k, 2) = matter.momentum3;
                  stress_trace_field(i, j, k, 0) =
                     matter.stress11 * metric.inv_gamma11
                     + 2.0 * matter.stress12 * metric.inv_gamma12
                     + 2.0 * matter.stress13 * metric.inv_gamma13
                     + matter.stress22 * metric.inv_gamma22
                     + 2.0 * matter.stress23 * metric.inv_gamma23
                     + matter.stress33 * metric.inv_gamma33;
               });
            }
            if (energy_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("conformal initial-data energy source is nonfinite");
            }
            if (momentum_storage[level]->contains_nan(0, 3, 0)) {
               amrex::Abort("conformal initial-data momentum source is nonfinite");
            }
            if (stress_trace_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("conformal initial-data stress trace is nonfinite");
            }
         }
         fill_initial_data_ghosts(
            hierarchy,
            cell_energy,
            initial_data_boundary_records(hierarchy, scalar_symmetries(1))
         );
         fill_initial_data_ghosts(
            hierarchy,
            cell_momentum,
            initial_data_boundary_records(hierarchy, vector_symmetries())
         );
         fill_initial_data_ghosts(
            hierarchy,
            cell_stress,
            initial_data_boundary_records(hierarchy, scalar_symmetries(1))
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_energy,
            cell_energy_density,
            initial_data_boundary_records(hierarchy, scalar_symmetries(1)),
            2
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_momentum,
            cell_momentum_density,
            initial_data_boundary_records(hierarchy, vector_symmetries()),
            2
         );
         materialize_initial_data_vertices(
            hierarchy,
            vertex_stress,
            cell_stress_trace,
            initial_data_boundary_records(hierarchy, scalar_symmetries(1)),
            2
         );
         for (int level = 0; level < num_levels; ++level) {
            if (vertex_energy_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("vertex initial-data energy source is nonfinite");
            }
            if (vertex_momentum_storage[level]->contains_nan(0, 3, 0)) {
               amrex::Abort("vertex initial-data momentum source is nonfinite");
            }
            if (vertex_stress_storage[level]->contains_nan(0, 1, 0)) {
               amrex::Abort("vertex initial-data stress trace is nonfinite");
            }
         }

         strict_xcts_polish = strict_xcts_polish
            || (coupled_xcts_matter_iteration
                && result.relative_matter_update <= 10.0 * matter_tolerance);
         const bool polish_coupled_xcts = strict_xcts_polish;
         const bool settle_large_xcts_update =
            coupled_xcts_matter_iteration
            && result.relative_matter_update > std::sqrt(matter_tolerance);
         const bool fully_solve_coupled_xcts =
            polish_coupled_xcts || settle_large_xcts_update;
         if (formulation == ConformalConstraintFormulation::ctt) {
            const CttResult constraints = solve_ctt_constraints(
               hierarchy,
               conformal_factor,
               vector_potential,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               conformal_boundary,
               vector_boundary,
               options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               0.0,
               constraints.momentum_residual,
               constraints.converged,
            };
         } else if (formulation == ConformalConstraintFormulation::cttk) {
            const CttkResult constraints = solve_cttk_hybrid_constraints(
               hierarchy,
               conformal_factor,
               mean_curvature_mutable,
               vector_potential,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               conformal_boundary,
               vector_boundary,
               cttk_options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               0.0,
               constraints.momentum_residual,
               constraints.converged,
            };
         } else {
            XctsOptions iteration_options = xcts_options;
            if (coupled_xcts_matter_iteration
                && !fully_solve_coupled_xcts) {
               // Follow the self-consistent-field iteration used by established
               // rotating-star solvers: advance the metric with the current
               // matter, update the matter, and converge the combined fixed
               // point. Accurate XCTS solves settle large changes and certify
               // the final fixed point; two relaxed sweeps avoid fully
               // converging each intermediate frozen-matter problem.
               iteration_options.max_coupled_iterations = 2;
            }
            const XctsResult constraints = solve_xcts_constraints(
               hierarchy,
               conformal_factor,
               lapse_times_conformal_factor,
               xcts_shift,
               conformal_extrinsic,
               energy_density,
               momentum_density,
               stress_trace,
               conformal_boundary,
               lapse_boundary,
               vector_boundary,
               iteration_options
            );
            result.constraints = {
               constraints.coupled_iterations,
               constraints.relative_update,
               constraints.hamiltonian_residual,
               constraints.lapse_residual,
               constraints.momentum_residual,
               constraints.converged,
            };
         }
         if (!result.constraints.converged
             && (!coupled_xcts_matter_iteration
                 || fully_solve_coupled_xcts)) {
            amrex::Print()
               << conformal_constraint_formulation_name(formulation)
               << " initial-data solve failed: iterations="
               << result.constraints.coupled_iterations
               << " relative_update=" << result.constraints.relative_update
               << " hamiltonian_residual="
               << result.constraints.hamiltonian_residual
               << " lapse_residual=" << result.constraints.lapse_residual
               << " momentum_residual=" << result.constraints.momentum_residual
               << '\n';
            amrex::Abort("conformal constraint initial-data solve did not converge");
         }

         amrex::Real relaxed_matter_update_norm = 0.0;
         amrex::Real other_update_norm = 0.0;
         amrex::Real solution_norm = 0.0;
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Subtract(
               *previous_primitive[level],
               *primitive_state[level],
               0,
               0,
               primitive_state[level]->nComp(),
               0
            );
            const int num_hydro_components = std::min(
               primitive_state[level]->nComp(),
               static_cast<int>(NVAR_HYDRO)
            );
            accumulate_local_linf_norms(
               *previous_primitive[level],
               *primitive_state[level],
               num_hydro_components,
               relaxed_matter_update_norm,
               solution_norm
            );
            if (primitive_state[level]->nComp() > num_hydro_components) {
               for (int component = num_hydro_components;
                    component < primitive_state[level]->nComp();
                    ++component) {
                  other_update_norm = std::max(
                     other_update_norm,
                     previous_primitive[level]->norm0(component, 0, true)
                  );
                  solution_norm = std::max(
                     solution_norm,
                     primitive_state[level]->norm0(component, 0, true)
                  );
               }
            }
            amrex::MultiFab::Subtract(
               *previous_conformal[level], *conformal_storage[level], 0, 0, 1, 0
            );
            accumulate_local_linf_norms(
               *previous_conformal[level],
               *conformal_storage[level],
               1,
               other_update_norm,
               solution_norm
            );
            amrex::MultiFab::Subtract(
               *previous_vector[level], *vector_storage[level], 0, 0, 3, 0
            );
            accumulate_local_linf_norms(
               *previous_vector[level],
               *vector_storage[level],
               3,
               other_update_norm,
               solution_norm
            );
            if (formulation == ConformalConstraintFormulation::xcts) {
               amrex::MultiFab::Subtract(
                  *previous_lapse[level],
                  *lapse_conformal_storage[level],
                  0,
                  0,
                  1,
                  0
               );
               accumulate_local_linf_norms(
                  *previous_lapse[level],
                  *lapse_conformal_storage[level],
                  1,
                  other_update_norm,
                  solution_norm
               );
            } else if (formulation == ConformalConstraintFormulation::cttk) {
               amrex::MultiFab::Subtract(
                  *previous_curvature[level], *curvature_storage[level], 0, 0, 1, 0
               );
               accumulate_local_linf_norms(
                  *previous_curvature[level],
                  *curvature_storage[level],
                  1,
                  other_update_norm,
                  solution_norm
               );
            }
         }
         step_matter_iterations = matter_iteration;
         amrex::Real active_matter_relaxation = 1.0;
         if (update_stellar_matter && !preserve_spherical_seed) {
            active_matter_relaxation =
               continued_rotating_star.matter_relaxation;
         }
         if (coupled_equilibrium_torus && torus_activated) {
            active_matter_relaxation = std::min(
               active_matter_relaxation,
               continued_torus.matter_relaxation
            );
         }
         result.relative_matter_update = global_relative_fixed_point_defect(
            relaxed_matter_update_norm,
            other_update_norm,
            solution_norm,
            active_matter_relaxation
         );
         result.relative_matter_update = std::max(
            result.relative_matter_update, stellar_matter_external_update
         );
         if (iterate_matter && amrex::ParallelDescriptor::IOProcessor()) {
            const char* matter_phase = coupled_equilibrium_torus
                  && torus_activated
               ? "star_torus" : "central_star";
            amrex::Print()
               << "rotating continuation step=" << continuation_step
               << "/"
               << continuation_schedule.stellar_angular_velocities.size()
               << " target_stellar_omega="
               << continued_rotating_star.angular_velocity
               << " target_disk_omega=" << continued_torus.angular_velocity
               << " phase=" << matter_phase
               << " coupled_update_iteration=" << matter_iteration
               << " relative_update=" << result.relative_matter_update << '\n';
         }
         if (result.relative_matter_update <= matter_tolerance
             && (!coupled_xcts_matter_iteration
                 || result.constraints.converged)) {
            if (coupled_equilibrium_torus && !torus_activated) {
               if (!update_stellar_matter || stellar_matter_updated) {
                  torus_activated = true;
                  strict_xcts_polish = false;
                  result.relative_matter_update =
                     std::numeric_limits<amrex::Real>::max();
               }
               continue;
            }
            if (coupled_equilibrium_torus && !coupled_torus_updated) {
               continue;
            }
            break;
         }
      }
      result.matter_iterations += step_matter_iterations;
      if (result.relative_matter_update > matter_tolerance
          || (coupled_xcts_matter_iteration
              && !result.constraints.converged)
          || (coupled_equilibrium_torus
              && (!torus_activated || !coupled_torus_updated))) {
         amrex::Abort(
            "matter-coupled conformal fixed-point iteration did not converge "
            "at rotation continuation step "
            + std::to_string(continuation_step)
         );
      }
      materialize_current_geometry();
      if (equilibrium_torus.active
          && equilibrium_torus.gravity
             == EquilibriumTorusGravity::TestFluid) {
         EquilibriumTorusParameters test_fluid_torus = continued_torus;
         test_fluid_torus.matter_relaxation = 1.0;
         amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
         for (int level = 0; level < num_levels; ++level) {
            cell_spacetime_const[level] = cell_spacetime[level];
         }
         result.equilibrium_torus = update_equilibrium_torus_primitives(
            hierarchy,
            primitive_state,
            cell_spacetime_const,
            EOSMethod,
            eos,
            test_fluid_torus
         );
      }
      if (equilibrium_torus.active) {
         amrex::Vector<amrex::MultiFab const*> primitive_const(num_levels);
         amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
         for (int level = 0; level < num_levels; ++level) {
            primitive_const[level] = primitive_state[level];
            cell_spacetime_const[level] = cell_spacetime[level];
         }
         update_equilibrium_torus_global_diagnostics<System, EOSMethod>(
            hierarchy,
            primitive_const,
            cell_spacetime_const,
            eos,
            continued_torus,
            result.equilibrium_torus
         );
      }
      if (continuation_observer) {
         if (update_stellar_matter) {
            update_surface_diagnostic(continued_rotating_star);
         }
         continuation_observer(
            continuation_step,
            static_cast<int>(
               continuation_schedule.stellar_angular_velocities.size()
            ),
            continued_rotating_star.angular_velocity,
            continued_torus.angular_velocity,
            result.black_hole,
            result.rotating_star,
            result.equilibrium_torus
         );
      }
   }

   materialize_current_geometry();
   if (update_stellar_matter && !continuation_observer) {
      update_surface_diagnostic(rotating_star);
   }
   return result;
}

#if (AMREX_SPACEDIM >= 2)
template <int EOSMethod>
ConformalMaterializationResult
materialize_axisymmetric_spherical_xcts_initial_data_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   EOSParams const& eos,
   ConformalBlackHoleParameters const& black_hole,
   RotatingStarParameters const& rotating_star,
   EquilibriumTorusParameters const& equilibrium_torus,
   bool seed_from_spacetime,
   ConformalContinuationObserver const& continuation_observer
)
{
   if (hierarchy.geometry.size() != 1
       || primitive_state.size() != 1
       || spacetime_state.size() != 1 || cell_spacetime.size() != 1
       || primitive_state[0] == nullptr
       || spacetime_state[0] == nullptr || cell_spacetime[0] == nullptr) {
      amrex::Abort("invalid axisymmetric spherical XCTS materialization");
   }
   const amrex::BoxArray& grids = hierarchy.grids[0];
   const amrex::DistributionMapping& distribution = hierarchy.distribution[0];
   amrex::MultiFab conformal_factor(grids, distribution, 1, 1);
   amrex::MultiFab lapse_times_conformal_factor(grids, distribution, 1, 1);
   amrex::MultiFab azimuthal_shift(grids, distribution, 1, 1);
   amrex::MultiFab conformal_extrinsic(
      grids, distribution, CONFORMAL_A_NUM_COMPONENTS, 0
   );
   amrex::MultiFab energy_density(grids, distribution, 1, 0);
   amrex::MultiFab azimuthal_momentum_density(grids, distribution, 1, 0);
   amrex::MultiFab stress_trace(grids, distribution, 1, 0);
   amrex::MultiFab previous_conformal(grids, distribution, 1, 0);
   amrex::MultiFab previous_lapse(grids, distribution, 1, 0);
   amrex::MultiFab previous_shift(grids, distribution, 1, 0);
   amrex::MultiFab previous_extrinsic(
      grids, distribution, CONFORMAL_A_NUM_COMPONENTS, 0
   );
   amrex::MultiFab previous_primitive(
      grids, distribution, primitive_state[0]->nComp(), 0
   );

   // Stellar and loaded portable ID sources enter through the same ADM/Z4c
   // handoff. Extract their reduced XCTS unknowns from that state. A vacuum
   // Schwarzschild shell is instead uniquely determined by its mass and
   // apparent-horizon boundary, so both a fresh solve and a changed-grid
   // re-solve use the analytic shell iterate. Interpolated Z4c is the portable
   // handoff, not an authoritative conformal-unknown seed on a different map.
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const amrex::Real horizon_radius = 0.5 * black_hole.mass;
   const amrex::Real target_black_hole_spin =
      black_hole.active ? black_hole.dimensionless_spin : 0.0;
   amrex::Real estimated_horizon_rotation_parameter = 0.0;
   if (target_black_hole_spin != 0.0) {
      // Cook-Pfeiffer's tangential-shift coefficient is a free horizon
      // rotation parameter, not the physical spin.  The inertial single-hole
      // sign is opposite the Kerr horizon angular velocity; this is only a
      // first iterate and is calibrated against the quasi-local throat spin.
      estimated_horizon_rotation_parameter = -target_black_hole_spin / (
         2.0 * black_hole.mass
         * (1.0 + std::sqrt(
            1.0 - target_black_hole_spin * target_black_hole_spin
         ))
      );
   }
   // Start from the exact nonspinning shell and reach the first rotational
   // estimate by continuation. Directly seeding a moderate-spin conformally
   // flat XCTS fixed point can select the divergent branch.
   amrex::Real horizon_rotation_parameter = 0.0;
   const bool coupled_equilibrium_torus = equilibrium_torus.active
      && equilibrium_torus.gravity == EquilibriumTorusGravity::Coupled;
   if (black_hole.active) {
      const amrex::Real scale = std::max(black_hole.mass, amrex::Real(1.0));
      if (!(black_hole.mass > 0.0)
          || std::abs(hierarchy.geometry[0].ProbLo(0) - horizon_radius)
             > 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale
          || std::abs(radial_coordinates.faces.front() - horizon_radius)
             > 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale
          || !std::isfinite(target_black_hole_spin)
          || !(std::abs(target_black_hole_spin) < 1.0)
          || rotating_star.active
          || (equilibrium_torus.active
              && !(equilibrium_torus.inner_radius > horizon_radius))) {
         amrex::Abort(
            "the first spherical-shell black-hole XCTS owner requires a "
            "quasi-equilibrium horizon at r=M/2 and any optional torus inner "
            "radius must lie outside the horizon"
         );
      }
   }
   if (black_hole.active) {
      amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
         radial_coordinates.centers.data(), radial_coordinates.centers.size()
      );
      const amrex::Real* radial_center = radial_center_storage.data();
      const amrex::Dim3 domain_lo = amrex::lbound(
         hierarchy.geometry[0].Domain()
      );
      const auto black_hole_problo = hierarchy.geometry[0].ProbLoArray();
      const auto black_hole_spacing = hierarchy.geometry[0].CellSizeArray();
      for (amrex::MFIter mfi(conformal_factor, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor.array(mfi);
         const auto lapse_psi = lapse_times_conformal_factor.array(mfi);
         const auto beta = azimuthal_shift.array(mfi);
         const auto extrinsic = conformal_extrinsic.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real radius = radial_center[i - domain_lo.x];
            const amrex::Real compactness = horizon_radius / radius;
            const amrex::Real radial_ratio = radius / horizon_radius;
            const amrex::Real theta = black_hole_problo[1]
               + (j - domain_lo.y + 0.5)
                  * black_hole_spacing[1];
            psi(i, j, k, 0) = 1.0 + compactness;
            lapse_psi(i, j, k, 0) = 1.0 - compactness;
            // A regular seed matches beta_hat_phi=Omega_H r_H sin(theta),
            // has (d_r beta-beta/r)|_H=0 where alpha vanishes, and decays as
            // r^-2 at infinity. The elliptic solve owns the final profile.
            beta(i, j, k, 0) = horizon_rotation_parameter
               * horizon_radius * std::sin(theta)
               * (4.0 / (radial_ratio * radial_ratio)
                  - 3.0 / (
                     radial_ratio * radial_ratio * radial_ratio
                  ));
            for (int component = 0;
                 component < CONFORMAL_A_NUM_COMPONENTS;
                 ++component) {
               extrinsic(i, j, k, component) = 0.0;
            }
         });
      }
   } else {
      transfer_grid_field_centering(
         *cell_spacetime[0],
         *spacetime_state[0],
         hierarchy.geometry[0],
         2
      );
      for (amrex::MFIter mfi(conformal_factor, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto seed = cell_spacetime[0]->const_array(mfi);
         const auto psi = conformal_factor.array(mfi);
         const auto lapse_psi = lapse_times_conformal_factor.array(mfi);
         const auto beta = azimuthal_shift.array(mfi);
         const auto extrinsic = conformal_extrinsic.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const Z4cState state = load_z4c_state(seed, i, j, k);
            const amrex::Real value = 1.0 / std::sqrt(std::sqrt(state.chi));
            const amrex::Real value_squared = value * value;
            const amrex::Real value_sixth = value_squared * value_squared
               * value_squared;
            psi(i, j, k, 0) = value;
            lapse_psi(i, j, k, 0) = state.lapse * value;
            beta(i, j, k, 0) = state.shift3;
            extrinsic(i, j, k, CONFORMAL_A11) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A12) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A13) =
               state.a_tilde13 * value_sixth;
            extrinsic(i, j, k, CONFORMAL_A22) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A23) =
               state.a_tilde23 * value_sixth;
            extrinsic(i, j, k, CONFORMAL_A33) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A_SQUARED) = 2.0 * (
               extrinsic(i, j, k, CONFORMAL_A13)
                  * extrinsic(i, j, k, CONFORMAL_A13)
               + extrinsic(i, j, k, CONFORMAL_A23)
                  * extrinsic(i, j, k, CONFORMAL_A23)
            );
         });
      }
   }
   auto scalar_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      black_hole.active ? 1.0 / (2.0 * horizon_radius) : 1.0,
      black_hole.active ? 1.0 : 0.0,
      black_hole.active ? -1.0 / (2.0 * horizon_radius) : 0.0,
      InitialDataEllipticCentering::cell
   );
   set_initial_data_reflection_boundaries(
      hierarchy, scalar_boundary, scalar_symmetries(1)
   );
   set_axisymmetric_spherical_radial_robin_boundary(
      hierarchy, scalar_boundary, {0.0}, 1.0
   );
   auto lapse_boundary = black_hole.active
      ? InitialDataEllipticBoundaryConditions::constant(
           hierarchy,
           1,
           1.0,
           0.0,
           -1.0,
           InitialDataEllipticCentering::cell
        )
      : scalar_boundary.component(0);
   if (black_hole.active) {
      set_initial_data_reflection_boundaries(
         hierarchy, lapse_boundary, scalar_symmetries(1)
      );
      set_axisymmetric_spherical_radial_robin_boundary(
         hierarchy, lapse_boundary, {0.0}, 1.0
      );
   }
   auto shift_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::cell
   );
   set_axisymmetric_spherical_toroidal_regularity(
      hierarchy, shift_boundary
   );
   if (black_hole.active) {
      set_axisymmetric_spherical_rotating_horizon_boundary(
         hierarchy,
         shift_boundary,
         horizon_radius,
         horizon_rotation_parameter
      );
   }
   set_axisymmetric_spherical_radial_robin_boundary(
      hierarchy, shift_boundary, {0.0}, 2.0
   );

   const bool vacuum_black_hole = black_hole.active
      && !coupled_equilibrium_torus;
   XctsOptions xcts_options;
   const bool spinning_black_hole = black_hole.active
      && target_black_hole_spin != 0.0;
   xcts_options.max_coupled_iterations =
      vacuum_black_hole && !spinning_black_hole ? 4 : 80;
   xcts_options.relative_update_tolerance = 1.0e-10;
   xcts_options.relaxation =
      vacuum_black_hole && !spinning_black_hole ? 1.0 : 0.7;
   // The vacuum apparent-horizon shell is a decoupled linear control.  Its
   // algebraic residual only needs to remain well below the second-order
   // shell truncation error; asking MLMG for 1e-12 on the larger mapped
   // hierarchy stalls after the physically relevant digits have converged.
   xcts_options.linear.relative_tolerance =
      vacuum_black_hole && !spinning_black_hole
      ? 2.0e-8 : 1.0e-12;
   xcts_options.linear.absolute_tolerance =
      vacuum_black_hole && !spinning_black_hole
      ? 2.0e-8 : 1.0e-13;
   xcts_options.linear.max_iterations = 500;
   xcts_options.linear.verbose = 0;
   amrex::Vector<amrex::MultiFab*> conformal_fields{&conformal_factor};
   amrex::Vector<amrex::MultiFab*> lapse_fields{
      &lapse_times_conformal_factor
   };
   amrex::Vector<amrex::MultiFab*> shift_fields{&azimuthal_shift};
   amrex::Vector<amrex::MultiFab*> extrinsic_fields{&conformal_extrinsic};
   amrex::Vector<amrex::MultiFab const*> energy_fields{&energy_density};
   amrex::Vector<amrex::MultiFab const*> momentum_fields{
      &azimuthal_momentum_density
   };
   amrex::Vector<amrex::MultiFab const*> stress_fields{&stress_trace};
   amrex::Vector<amrex::MultiFab const*> cell_geometry_view{
      cell_spacetime[0]
   };

   const auto materialize_current_geometry = [&]() {
      materialize_axisymmetric_cell_geometry(
         hierarchy,
         conformal_factor,
         lapse_times_conformal_factor,
         azimuthal_shift,
         conformal_extrinsic,
         *cell_spacetime[0]
      );
      materialize_axisymmetric_vertex_geometry(
         hierarchy,
         conformal_factor,
         lapse_times_conformal_factor,
         azimuthal_shift,
         conformal_extrinsic,
         scalar_boundary,
         lapse_boundary,
         shift_boundary,
         *spacetime_state[0]
      );
   };
   const auto fill_matter_sources = [&]() {
      materialize_axisymmetric_cell_geometry(
         hierarchy,
         conformal_factor,
         lapse_times_conformal_factor,
         azimuthal_shift,
         conformal_extrinsic,
         *cell_spacetime[0]
      );
      for (amrex::MFIter mfi(energy_density, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_state[0]->const_array(mfi);
         const auto geometry = cell_spacetime[0]->const_array(mfi);
         const auto energy = energy_density.array(mfi);
         const auto momentum = azimuthal_momentum_density.array(mfi);
         const auto stress = stress_trace.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const EulerianMatter matter = generated::rhd_eulerian_matter(
               rhd_primitive_at<EOSMethod>(primitive, i, j, k, eos), metric
            );
            energy(i, j, k, 0) = matter.rho;
            momentum(i, j, k, 0) = matter.momentum3;
            stress(i, j, k, 0) =
               matter.stress11 * metric.inv_gamma11
               + 2.0 * matter.stress12 * metric.inv_gamma12
               + 2.0 * matter.stress13 * metric.inv_gamma13
               + matter.stress22 * metric.inv_gamma22
               + 2.0 * matter.stress23 * metric.inv_gamma23
               + matter.stress33 * metric.inv_gamma33;
         });
      }
   };
   const auto fill_sources = [&]() {
      if (vacuum_black_hole) {
         energy_density.setVal(0.0);
         azimuthal_momentum_density.setVal(0.0);
         stress_trace.setVal(0.0);
      } else {
         fill_matter_sources();
      }
   };
   const auto continue_horizon_rotation = [&](amrex::Real target_rotation,
                                               char const* failure_message) {
      XctsResult constraints;
      const amrex::Real maximum_step = 8.0e-3 / black_hole.mass;
      const amrex::Real minimum_step = 1.0e-6 / black_hole.mass;
      amrex::Real trial_step = std::min(
         maximum_step,
         std::abs(target_rotation - horizon_rotation_parameter)
      );
      while (std::abs(target_rotation - horizon_rotation_parameter)
             > 64.0 * std::numeric_limits<amrex::Real>::epsilon()
                  * std::max(amrex::Real(1.0), std::abs(target_rotation))) {
         const amrex::Real remaining =
            target_rotation - horizon_rotation_parameter;
         trial_step = std::min(trial_step, std::abs(remaining));
         const amrex::Real candidate = horizon_rotation_parameter
            + std::copysign(trial_step, remaining);
         amrex::MultiFab::Copy(
            previous_conformal, conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            previous_lapse, lapse_times_conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            previous_shift, azimuthal_shift, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            previous_extrinsic,
            conformal_extrinsic,
            0,
            0,
            CONFORMAL_A_NUM_COMPONENTS,
            0
         );
         set_axisymmetric_spherical_rotating_horizon_boundary(
            hierarchy,
            shift_boundary,
            horizon_radius,
            candidate
         );
         constraints = solve_axisymmetric_spherical_xcts_constraints(
            hierarchy,
            conformal_fields,
            lapse_fields,
            shift_fields,
            extrinsic_fields,
            energy_fields,
            momentum_fields,
            stress_fields,
            scalar_boundary,
            lapse_boundary,
            shift_boundary,
            xcts_options
         );
         if (constraints.converged) {
            horizon_rotation_parameter = candidate;
            materialize_current_geometry();
            const ConformalBlackHoleSummary throat =
               axisymmetric_black_hole_throat_summary(
                  hierarchy, *spacetime_state[0], horizon_radius
               );
            const amrex::Real measured_spin =
               throat.christodoulou_dimensionless_spin;
            // The Cook--Pfeiffer rotation coefficient is not the physical
            // spin and its Kerr estimate can lie beyond a nonlinear branch
            // fold. Stop continuation as soon as the measured spin brackets
            // the target; the outer calibration then solves for the precise
            // coefficient without needlessly overshooting the physical goal.
            if (target_black_hole_spin * measured_spin > 0.0
                && std::abs(measured_spin)
                   >= std::abs(target_black_hole_spin)) {
               return constraints;
            }
            trial_step = std::min(maximum_step, 1.25 * trial_step);
            continue;
         }

         amrex::MultiFab::Copy(
            conformal_factor, previous_conformal, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            lapse_times_conformal_factor, previous_lapse, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            azimuthal_shift, previous_shift, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            conformal_extrinsic,
            previous_extrinsic,
            0,
            0,
            CONFORMAL_A_NUM_COMPONENTS,
            0
         );
         set_axisymmetric_spherical_rotating_horizon_boundary(
            hierarchy,
            shift_boundary,
            horizon_radius,
            horizon_rotation_parameter
         );
         trial_step *= 0.5;
         if (trial_step < minimum_step) {
            amrex::Abort(failure_message);
         }
      }
      return constraints;
   };

   if (vacuum_black_hole) {
      fill_sources();
      XctsResult constraints;
      ConformalBlackHoleSummary black_hole_summary;
      amrex::Real previous_rotation_parameter =
         std::numeric_limits<amrex::Real>::quiet_NaN();
      amrex::Real previous_measured_spin =
         std::numeric_limits<amrex::Real>::quiet_NaN();
      constexpr int maximum_spin_calibrations = 10;
      const amrex::Real spin_tolerance = std::max(
         amrex::Real(2.0e-5),
         2.0e-4 * std::abs(target_black_hole_spin)
      );
      bool spin_calibrated = !spinning_black_hole;
      if (spinning_black_hole) {
         constraints = continue_horizon_rotation(
            estimated_horizon_rotation_parameter,
            "spherical-shell black-hole spin continuation did not converge"
         );
      }
      for (int calibration = 0;
           calibration < (spinning_black_hole
              ? maximum_spin_calibrations : 1);
           ++calibration) {
         constraints = solve_axisymmetric_spherical_xcts_constraints(
            hierarchy,
            conformal_fields,
            lapse_fields,
            shift_fields,
            extrinsic_fields,
            energy_fields,
            momentum_fields,
            stress_fields,
            scalar_boundary,
            lapse_boundary,
            shift_boundary,
            xcts_options
         );
         if (!constraints.converged) {
            amrex::Abort("spherical-shell black-hole XCTS did not converge");
         }
         materialize_current_geometry();
         black_hole_summary = axisymmetric_black_hole_throat_summary(
            hierarchy, *spacetime_state[0], horizon_radius
         );
         const amrex::Real measured_spin =
            black_hole_summary.christodoulou_dimensionless_spin;
         const amrex::Real spin_error =
            measured_spin - target_black_hole_spin;
         if (std::abs(spin_error) <= spin_tolerance) {
            spin_calibrated = true;
            break;
         }

         amrex::Real next_rotation_parameter =
            std::numeric_limits<amrex::Real>::quiet_NaN();
         if (std::isfinite(previous_measured_spin)
             && measured_spin != previous_measured_spin) {
            next_rotation_parameter = horizon_rotation_parameter
               - spin_error * (
                  horizon_rotation_parameter - previous_rotation_parameter
               ) / (measured_spin - previous_measured_spin);
         }
         if (!std::isfinite(next_rotation_parameter)
             && std::abs(measured_spin) > 1.0e-10) {
            next_rotation_parameter = horizon_rotation_parameter
               * target_black_hole_spin / measured_spin;
         }
         if (!std::isfinite(next_rotation_parameter)) {
            next_rotation_parameter = horizon_rotation_parameter
               - std::copysign(
                  0.025 / black_hole.mass,
                  spin_error
               );
         }
         const amrex::Real maximum_rotation_parameter =
            0.75 / black_hole.mass;
         next_rotation_parameter = std::clamp(
            next_rotation_parameter,
            -maximum_rotation_parameter,
            maximum_rotation_parameter
         );
         previous_rotation_parameter = horizon_rotation_parameter;
         previous_measured_spin = measured_spin;
         horizon_rotation_parameter = next_rotation_parameter;
         set_axisymmetric_spherical_rotating_horizon_boundary(
            hierarchy,
            shift_boundary,
            horizon_radius,
            horizon_rotation_parameter
         );
         if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
               << "black-hole spin calibration iteration="
               << calibration + 1
               << " target_chi=" << target_black_hole_spin
               << " measured_chi=" << measured_spin
               << " horizon_rotation_parameter="
               << horizon_rotation_parameter << '\n';
         }
      }
      if (!spin_calibrated) {
         amrex::Abort(
            "spherical-shell black-hole XCTS did not reach its physical "
            "horizon-spin target"
         );
      }
      ConformalMaterializationResult result;
      result.constraints = {
         constraints.coupled_iterations,
         constraints.relative_update,
         constraints.hamiltonian_residual,
         constraints.lapse_residual,
         constraints.momentum_residual,
         constraints.converged,
      };
      result.black_hole = black_hole_summary;
      if (equilibrium_torus.active) {
         EquilibriumTorusParameters test_fluid_torus = equilibrium_torus;
         test_fluid_torus.matter_relaxation = 1.0;
         result.equilibrium_torus =
            update_axisymmetric_spherical_equilibrium_torus_primitives(
               hierarchy,
               primitive_state,
               cell_geometry_view,
               EOSMethod,
               eos,
               test_fluid_torus
            );
         fill_matter_sources();
         update_axisymmetric_equilibrium_torus_global_diagnostics<EOSMethod>(
            hierarchy,
            *primitive_state[0],
            *cell_spacetime[0],
            conformal_factor,
            azimuthal_momentum_density,
            eos,
            equilibrium_torus,
            result.equilibrium_torus
         );
      }
      return result;
   }

   if (spinning_black_hole) {
      // Establish the rotating vacuum hole before activating a coupled disk.
      // This keeps the matter and horizon-spin fixed points individually
      // small and preserves the same continuation path as the vacuum owner.
      energy_density.setVal(0.0);
      azimuthal_momentum_density.setVal(0.0);
      stress_trace.setVal(0.0);
      continue_horizon_rotation(
         estimated_horizon_rotation_parameter,
         "coupled black-hole/disk spin continuation did not converge"
      );
      materialize_current_geometry();
   }

   const bool update_stellar_matter = rotating_star.active
      && (rotating_star.angular_velocity != 0.0
          || coupled_equilibrium_torus);
   const CircularContinuationSchedule continuation_schedule =
      circular_continuation_schedule(rotating_star, equilibrium_torus);

   ConformalMaterializationResult result;
   const bool iterate_matter = update_stellar_matter
      || coupled_equilibrium_torus;
   const bool calibrate_coupled_black_hole_spin =
      black_hole.active && coupled_equilibrium_torus;
   const int maximum_matter_iterations = iterate_matter
      ? (rotating_star.active
            ? rotating_star.max_iterations
            : equilibrium_torus.max_iterations)
         * (coupled_equilibrium_torus ? 2 : 1)
      : 12;
   const amrex::Real matter_tolerance = iterate_matter
      ? (rotating_star.active
            ? rotating_star.relative_tolerance
            : equilibrium_torus.relative_tolerance)
      : 1.0e-9;
   for (int continuation_step = 1;
        continuation_step <= static_cast<int>(
           continuation_schedule.stellar_angular_velocities.size()
        );
        ++continuation_step) {
      RotatingStarParameters continued = rotating_star;
      continued.angular_velocity =
         continuation_schedule
            .stellar_angular_velocities[continuation_step - 1];
      EquilibriumTorusParameters continued_torus = equilibrium_torus;
      continued_torus.angular_velocity =
         continuation_schedule.disk_angular_velocities[continuation_step - 1];
      if (coupled_equilibrium_torus) {
         continued.update_outer_radius = continued_torus.inner_radius;
      }
      int step_iterations = 0;
      bool torus_activated = !coupled_equilibrium_torus;
      bool coupled_torus_updated =
         seed_from_spacetime && coupled_equilibrium_torus;
      bool black_hole_spin_converged =
         !calibrate_coupled_black_hole_spin;
      result.relative_matter_update =
         std::numeric_limits<amrex::Real>::max();
      for (int matter_iteration = 1;
           matter_iteration <= maximum_matter_iterations;
           ++matter_iteration) {
         amrex::MultiFab::Copy(
            previous_primitive,
            *primitive_state[0],
            0,
            0,
            primitive_state[0]->nComp(),
            0
         );
         amrex::MultiFab::Copy(
            previous_conformal, conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            previous_lapse, lapse_times_conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Copy(
            previous_shift, azimuthal_shift, 0, 0, 1, 0
         );
         materialize_axisymmetric_cell_geometry(
            hierarchy,
            conformal_factor,
            lapse_times_conformal_factor,
            azimuthal_shift,
            conformal_extrinsic,
            *cell_spacetime[0]
         );
         const bool preserve_initial_stellar_seed =
            !continued_torus.active
            && continuation_step == 1 && matter_iteration == 1;
         if (update_stellar_matter && !preserve_initial_stellar_seed) {
            result.rotating_star =
               update_axisymmetric_spherical_rotating_star_primitives(
                  hierarchy,
                  primitive_state,
                  cell_geometry_view,
                  EOSMethod,
                  eos,
                  continued
               );
         }
         if (continued_torus.active
             && result.rotating_star.matter_outer_radius
                >= continued_torus.inner_radius) {
            amrex::Abort(
               "equilibrium-torus inner radius must lie outside the central "
               "star matter support"
            );
         }
         if (coupled_equilibrium_torus && torus_activated) {
            result.equilibrium_torus =
               update_axisymmetric_spherical_equilibrium_torus_primitives(
                  hierarchy,
                  primitive_state,
                  cell_geometry_view,
                  EOSMethod,
                  eos,
                  continued_torus
               );
            coupled_torus_updated = true;
         }
         fill_sources();
         const XctsResult constraints =
            solve_axisymmetric_spherical_xcts_constraints(
               hierarchy,
               conformal_fields,
               lapse_fields,
               shift_fields,
               extrinsic_fields,
               energy_fields,
               momentum_fields,
               stress_fields,
               scalar_boundary,
               lapse_boundary,
               shift_boundary,
               xcts_options
            );
         result.constraints = {
            constraints.coupled_iterations,
            constraints.relative_update,
            constraints.hamiltonian_residual,
            constraints.lapse_residual,
            constraints.momentum_residual,
            constraints.converged,
         };
         if (!constraints.converged) {
            amrex::Abort("axisymmetric spherical XCTS did not converge");
         }
         amrex::Real black_hole_spin_error = 0.0;
         if (black_hole.active) {
            materialize_current_geometry();
            result.black_hole = axisymmetric_black_hole_throat_summary(
               hierarchy, *spacetime_state[0], horizon_radius
            );
            const amrex::Real measured_spin =
               result.black_hole.christodoulou_dimensionless_spin;
            if (calibrate_coupled_black_hole_spin) {
               black_hole_spin_error =
                  measured_spin - target_black_hole_spin;
               const amrex::Real spin_tolerance = std::max(
                  amrex::Real(2.0e-5),
                  2.0e-4 * std::abs(target_black_hole_spin)
               );
               black_hole_spin_converged =
                  std::abs(black_hole_spin_error) <= spin_tolerance;
            }
         }
         amrex::MultiFab::Subtract(
            previous_primitive,
            *primitive_state[0],
            0,
            0,
            primitive_state[0]->nComp(),
            0
         );
         amrex::MultiFab::Subtract(
            previous_conformal, conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Subtract(
            previous_lapse, lapse_times_conformal_factor, 0, 0, 1, 0
         );
         amrex::MultiFab::Subtract(
            previous_shift, azimuthal_shift, 0, 0, 1, 0
         );
         amrex::Real relaxed_matter_update_norm = 0.0;
         amrex::Real other_update_norm = 0.0;
         amrex::Real solution_norm = 1.0;
         accumulate_local_linf_norms(
            previous_primitive,
            *primitive_state[0],
            primitive_state[0]->nComp(),
            relaxed_matter_update_norm,
            solution_norm
         );
         accumulate_local_linf_norms(
            previous_conformal,
            conformal_factor,
            1,
            other_update_norm,
            solution_norm
         );
         accumulate_local_linf_norms(
            previous_lapse,
            lapse_times_conformal_factor,
            1,
            other_update_norm,
            solution_norm
         );
         accumulate_local_linf_norms(
            previous_shift,
            azimuthal_shift,
            1,
            other_update_norm,
            solution_norm
         );
         amrex::Real active_matter_relaxation = 1.0;
         if (update_stellar_matter && !preserve_initial_stellar_seed) {
            active_matter_relaxation = continued.matter_relaxation;
         }
         if (coupled_equilibrium_torus && torus_activated) {
            active_matter_relaxation = std::min(
               active_matter_relaxation,
               continued_torus.matter_relaxation
            );
         }
         result.relative_matter_update = global_relative_fixed_point_defect(
            relaxed_matter_update_norm,
            other_update_norm,
            solution_norm,
            active_matter_relaxation
         );
         step_iterations = matter_iteration;
         if (amrex::ParallelDescriptor::IOProcessor()
             && iterate_matter) {
            const char* matter_phase = black_hole.active
               ? (torus_activated ? "black_hole_torus" : "black_hole")
               : ((coupled_equilibrium_torus && torus_activated)
                     ? "star_torus" : "central_star");
            amrex::Print()
               << "axisymmetric matter continuation step="
               << continuation_step << "/"
               << continuation_schedule.stellar_angular_velocities.size()
               << " stellar_omega=" << continued.angular_velocity
               << " disk_omega=" << continued_torus.angular_velocity
               << " phase=" << matter_phase
               << " matter_iteration=" << matter_iteration
               << " relative_update=" << result.relative_matter_update
               << '\n';
         }
         // Calibrate the physical horizon spin only after the current matter
         // fixed point has settled. Moving the horizon boundary while the
         // disk profile is still changing makes two otherwise contractive
         // iterations chase one another and destabilizes even the spin-zero
         // control. Calibrating target chi=0 here also removes the small
         // physical angular momentum induced by the surrounding disk.
         if (calibrate_coupled_black_hole_spin
             && result.relative_matter_update <= matter_tolerance
             && !black_hole_spin_converged) {
            const amrex::Real measured_spin =
               result.black_hole.christodoulou_dimensionless_spin;
            amrex::Real next_rotation_parameter = horizon_rotation_parameter;
            if (target_black_hole_spin == 0.0) {
               next_rotation_parameter += 0.25
                  * black_hole_spin_error / black_hole.mass;
            } else if (std::abs(measured_spin) > 1.0e-8) {
               const amrex::Real scaled_rotation = horizon_rotation_parameter
                  * target_black_hole_spin / measured_spin;
               next_rotation_parameter = 0.5
                  * (horizon_rotation_parameter + scaled_rotation);
            } else {
               next_rotation_parameter += 0.25
                  * black_hole_spin_error / black_hole.mass;
            }
            const amrex::Real maximum_rotation_parameter =
               0.75 / black_hole.mass;
            horizon_rotation_parameter = std::clamp(
               next_rotation_parameter,
               -maximum_rotation_parameter,
               maximum_rotation_parameter
            );
            set_axisymmetric_spherical_rotating_horizon_boundary(
               hierarchy,
               shift_boundary,
               horizon_radius,
               horizon_rotation_parameter
            );
            if (amrex::ParallelDescriptor::IOProcessor()) {
               amrex::Print()
                  << "coupled black-hole spin calibration"
                  << " target_chi=" << target_black_hole_spin
                  << " measured_chi=" << measured_spin
                  << " horizon_rotation_parameter="
                  << horizon_rotation_parameter << '\n';
            }
            continue;
         }
         if (result.relative_matter_update <= matter_tolerance
             && black_hole_spin_converged) {
            if (coupled_equilibrium_torus && !torus_activated) {
               torus_activated = true;
               continue;
            }
            break;
         }
      }
      result.matter_iterations += step_iterations;
      if (result.relative_matter_update > matter_tolerance
          || !black_hole_spin_converged
          || (coupled_equilibrium_torus
              && (!torus_activated || !coupled_torus_updated))) {
         amrex::Abort(
            "axisymmetric black-hole/stellar torus matter iteration did not "
            "converge"
         );
      }
      materialize_current_geometry();
      if (continued_torus.active
          && continued_torus.gravity
             == EquilibriumTorusGravity::TestFluid) {
         const amrex::Real central_matter_outer_radius =
            axisymmetric_matter_outer_radius<EOSMethod>(
               hierarchy,
               *primitive_state[0],
               continued_torus.atmosphere
            );
         if (central_matter_outer_radius >= continued_torus.inner_radius) {
            amrex::Abort(
               "equilibrium-torus inner radius must lie outside the central "
               "star matter support"
            );
         }
         result.rotating_star.matter_outer_radius = std::max(
            result.rotating_star.matter_outer_radius,
            central_matter_outer_radius
         );
         EquilibriumTorusParameters test_fluid_torus = continued_torus;
         test_fluid_torus.matter_relaxation = 1.0;
         result.equilibrium_torus =
            update_axisymmetric_spherical_equilibrium_torus_primitives(
               hierarchy,
               primitive_state,
               cell_geometry_view,
               EOSMethod,
               eos,
               test_fluid_torus
            );
      }
      fill_sources();
      if (!black_hole.active) {
         update_axisymmetric_global_diagnostics<EOSMethod>(
            hierarchy,
            *primitive_state[0],
            *cell_spacetime[0],
            conformal_factor,
            conformal_extrinsic,
            energy_density,
            azimuthal_momentum_density,
            eos,
            result.rotating_star
         );
      }
      update_axisymmetric_equilibrium_torus_global_diagnostics<EOSMethod>(
         hierarchy,
         *primitive_state[0],
         *cell_spacetime[0],
         conformal_factor,
         azimuthal_momentum_density,
         eos,
         continued_torus,
         result.equilibrium_torus
      );
      if (update_stellar_matter) {
         RotatingStarParameters diagnostic_parameters = continued;
         diagnostic_parameters.injection_energy =
            result.rotating_star.injection_energy;
         const RotatingStarSurfaceDiagnostic diagnostic =
            axisymmetric_spherical_rotating_star_surface_diagnostic(
               hierarchy, cell_geometry_view, diagnostic_parameters
            );
         result.rotating_star.equatorial_log_enthalpy_gradient =
            diagnostic.equatorial_log_enthalpy_gradient;
         result.rotating_star.polar_log_enthalpy_gradient =
            diagnostic.polar_log_enthalpy_gradient;
         result.rotating_star.mass_shedding_indicator =
            diagnostic.mass_shedding_indicator;
      }
      if (continuation_observer) {
         if (black_hole.active) {
            result.black_hole = axisymmetric_black_hole_throat_summary(
               hierarchy, *spacetime_state[0], horizon_radius
            );
         }
         continuation_observer(
            continuation_step,
            static_cast<int>(
               continuation_schedule.stellar_angular_velocities.size()
            ),
            continued.angular_velocity,
            continued_torus.angular_velocity,
            result.black_hole,
            result.rotating_star,
            result.equilibrium_torus
         );
      }
   }
   materialize_current_geometry();
   if (black_hole.active) {
      result.black_hole = axisymmetric_black_hole_throat_summary(
         hierarchy, *spacetime_state[0], horizon_radius
      );
   }
   return result;
}
#endif

} // namespace

ConformalMaterializationResult materialize_conformal_initial_data(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   PhysicalSystem system,
   int eos_method,
   EOSParams const& eos,
   InitialDataFieldSelection const& fields,
   int coordinate_kind,
   ConformalConstraintFormulation formulation,
   ConformalBlackHoleParameters const& black_hole,
   RotatingStarParameters const& rotating_star,
   EquilibriumTorusParameters const& equilibrium_torus,
   bool seed_from_spacetime,
   ConformalContinuationObserver const& continuation_observer,
   ConformalMatterObserver const& matter_observer,
   ConformalStellarMatterUpdater const& stellar_matter_updater
)
{
#if (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(
      fields, coordinate_kind, matter_observer, stellar_matter_updater
   );
   if (system != PhysicalSystem::RHD
       || formulation != ConformalConstraintFormulation::xcts) {
      amrex::Abort(
         "reduced axisymmetric spherical conformal data currently require "
         "RHD and XCTS"
      );
   }
   return materialize_axisymmetric_spherical_xcts_initial_data_method<
      EOSRuntime
   >(
      hierarchy, primitive_state, spacetime_state,
      cell_spacetime, eos, black_hole, rotating_star, equilibrium_torus,
      seed_from_spacetime,
      continuation_observer
   );
#elif (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, primitive_state, spacetime_state,
      cell_spacetime, system, eos_method, eos, fields, coordinate_kind,
      formulation,
      black_hole, rotating_star, equilibrium_torus, seed_from_spacetime,
      continuation_observer, matter_observer, stellar_matter_updater
   );
   amrex::Abort("conformal constraint initial data requires 2D or 3D");
   return {};
#else
   if (coordinate_kind == MeshSpherical) {
      if (system != PhysicalSystem::RHD
          || formulation != ConformalConstraintFormulation::xcts) {
         amrex::Abort(
            "axisymmetric spherical XCTS requires RHD storage and XCTS"
         );
      }
      amrex::ignore_unused(fields, eos_method);
      return materialize_axisymmetric_spherical_xcts_initial_data_method<
         EOSRuntime
      >(
         hierarchy,
         primitive_state,
         spacetime_state,
         cell_spacetime,
         eos,
         black_hole,
         rotating_star,
         equilibrium_torus,
         seed_from_spacetime,
         continuation_observer
      );
   }
   if (system == PhysicalSystem::RMHD) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RMHD,
         EOSRuntime
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star,
         equilibrium_torus,
         seed_from_spacetime,
         continuation_observer,
         matter_observer,
         stellar_matter_updater
      );
   }
   if (system != PhysicalSystem::RHD) {
      amrex::Abort("conformal constraint initial data requires RHD or RMHD");
   }
   return materialize_conformal_initial_data_method<
      PhysicalSystem::RHD,
      EOSRuntime
   >(
      hierarchy, primitive_state, spacetime_state,
      cell_spacetime, eos, fields, formulation, rotating_star,
      equilibrium_torus,
      seed_from_spacetime,
      continuation_observer,
      matter_observer,
      stellar_matter_updater
   );
#endif
}

} // namespace pyfluid_amrex
