#include "pyfluid_amrex/initial_data/rotating_star.hpp"

#include <AMReX.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {
namespace {

RotationLaw rotation_law_from_name(std::string const& name)
{
   for (RotationLawInfo const& law : RotationLawCatalog) {
      if (law.name == name) {
         return law.id;
      }
   }
   amrex::Abort("unknown generated rotation law: " + name);
   return RotationLaw::Uniform;
}

amrex::Real cubic_surface_derivative(
   amrex::Real value_minus,
   amrex::Real value_zero,
   amrex::Real value_plus,
   amrex::Real value_plus_two,
   amrex::Real spacing
)
{
   auto polynomial = [=](amrex::Real coordinate) {
      const amrex::Real basis_minus =
         -coordinate * (coordinate - 1.0) * (coordinate - 2.0) / 6.0;
      const amrex::Real basis_zero =
         (coordinate + 1.0) * (coordinate - 1.0)
         * (coordinate - 2.0) / 2.0;
      const amrex::Real basis_plus =
         -(coordinate + 1.0) * coordinate * (coordinate - 2.0) / 2.0;
      const amrex::Real basis_plus_two =
         (coordinate + 1.0) * coordinate * (coordinate - 1.0) / 6.0;
      return value_minus * basis_minus + value_zero * basis_zero
         + value_plus * basis_plus + value_plus_two * basis_plus_two;
   };
   amrex::Real lower = 0.0;
   amrex::Real upper = 1.0;
   for (int iteration = 0; iteration < 48; ++iteration) {
      const amrex::Real midpoint = 0.5 * (lower + upper);
      if (polynomial(midpoint) > 0.0) {
         lower = midpoint;
      } else {
         upper = midpoint;
      }
   }
   const amrex::Real coordinate = 0.5 * (lower + upper);
   const amrex::Real derivative_minus =
      -(3.0 * coordinate * coordinate - 6.0 * coordinate + 2.0) / 6.0;
   const amrex::Real derivative_zero =
      (3.0 * coordinate * coordinate - 4.0 * coordinate - 1.0) / 2.0;
   const amrex::Real derivative_plus =
      -(3.0 * coordinate * coordinate - 2.0 * coordinate - 2.0) / 2.0;
   const amrex::Real derivative_plus_two =
      (3.0 * coordinate * coordinate - 1.0) / 6.0;
   return (
      value_minus * derivative_minus + value_zero * derivative_zero
      + value_plus * derivative_plus + value_plus_two * derivative_plus_two
   ) / spacing;
}

amrex::Real ray_surface_derivative(
   std::vector<amrex::Real> const& values,
   std::size_t offset,
   int count,
   amrex::Real spacing
)
{
   for (int index = 1; index + 2 < count; ++index) {
      const amrex::Real inside = values[offset + index];
      const amrex::Real outside = values[offset + index + 1];
      if (inside > 0.0 && outside <= 0.0) {
         return cubic_surface_derivative(
            values[offset + index - 1],
            inside,
            outside,
            values[offset + index + 2],
            spacing
         );
      }
   }
   amrex::Abort("rotating-star surface ray does not contain a resolved enthalpy zero");
   return 0.0;
}

template <int EOSMethod>
RotatingStarUpdateSummary update_rotating_star_primitives_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   RotatingStarUpdateSummary summary;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> effective_potential_storage(
      num_levels
   );
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> matter_support_storage(
      num_levels
   );
   for (int level = 0; level < num_levels; ++level) {
      if (primitive_state[level] == nullptr || cell_spacetime[level] == nullptr) {
         amrex::Abort("rotating-star update received an unallocated level");
      }
      if (primitive_state[level]->nComp() < NVAR_HYDRO
          || cell_spacetime[level]->nComp() < Z4C_NUM_COMPONENTS) {
         amrex::Abort("rotating-star update received an incomplete state layout");
      }
      effective_potential_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 2, 1
      );
      matter_support_storage[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      matter_support_storage[level]->setVal(0.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const int ilo = domain.smallEnd(0);
      const int jlo = domain.smallEnd(1);
      const int klo = domain.smallEnd(2);
      for (amrex::MFIter mfi(
              *effective_potential_storage[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_state[level]->const_array(mfi);
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto effective = effective_potential_storage[level]->array(mfi);
         const auto matter_support = matter_support_storage[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - ilo) + 0.5) * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - jlo) + 0.5) * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - klo) + 0.5) * dx[2],
            };
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const RotatingStarPoint point = rotating_star_point(
               parameters, physical, metric
            );
            effective(i, j, k, 0) = point.timelike
               ? metric.lapse * std::sqrt(1.0 - point.velocity_squared)
               : std::numeric_limits<amrex::Real>::max();
            effective(i, j, k, 1) = point.timelike ? 0.0 : 1.0;
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
            matter_support(i, j, k) = initial_matter_exceeds_atmosphere(
               primitive(i, j, k, RHO), atmosphere.rho
            ) ? 1.0 : 0.0;
         });
      }
   }
   amrex::Vector<amrex::MultiFab*> effective_potential(num_levels);
   amrex::Vector<amrex::MultiFab*> matter_support(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      effective_potential[level] = effective_potential_storage[level].get();
      matter_support[level] = matter_support_storage[level].get();
   }
   fill_initial_data_ghosts(
      hierarchy,
      effective_potential,
      initial_data_boundary_records(
         hierarchy,
         {GridFieldSymmetryKind::Scalar, GridFieldSymmetryKind::Scalar}
      )
   );
   fill_initial_data_ghosts(
      hierarchy,
      matter_support,
      initial_data_boundary_records(
         hierarchy, {GridFieldSymmetryKind::Scalar}
      )
   );
   auto center_dx = hierarchy.geometry.back().CellSizeArray();
   std::vector<amrex::GpuArray<amrex::Real, 3>> center_point{{{
      parameters.rotation_center.q1
         + (hierarchy.reflection_lo[0] != 0 ? 0.5 * center_dx[0] : 0.0)
         - (hierarchy.reflection_hi[0] != 0 ? 0.5 * center_dx[0] : 0.0),
      parameters.rotation_center.q2
         + (hierarchy.reflection_lo[1] != 0 ? 0.5 * center_dx[1] : 0.0)
         - (hierarchy.reflection_hi[1] != 0 ? 0.5 * center_dx[1] : 0.0),
      parameters.rotation_center.q3
         + (hierarchy.reflection_lo[2] != 0 ? 0.5 * center_dx[2] : 0.0)
         - (hierarchy.reflection_hi[2] != 0 ? 0.5 * center_dx[2] : 0.0),
   }}};
   const HierarchyInterpolationPlan center_plan(
      std::move(center_point),
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian),
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      amrex::IndexType::TheCellType()
   );
   const std::vector<amrex::Real> center_values = center_plan.interpolate(
      effective_potential_storage, 0, 2
   );
   const amrex::Real center_effective_potential = center_values[0];
   if (!(center_effective_potential > 0.0)
       || !std::isfinite(center_effective_potential)
       || center_values[1] > 0.0) {
      if (amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print()
            << "rotating-star invalid center: effective_potential="
            << center_effective_potential
            << " timelike_failure_weight=" << center_values[1] << '\n';
      }
      amrex::Abort("rotating-star effective potential is invalid at the rotation center");
   }
   RotatingStarParameters iteration_parameters = parameters;
   iteration_parameters.injection_energy = std::exp(
      parameters.central_log_specific_enthalpy
   ) * center_effective_potential;
   summary.injection_energy = iteration_parameters.injection_energy;
   for (int level = 0; level < num_levels; ++level) {
      amrex::MultiFab diagnostics(
         hierarchy.grids[level], hierarchy.distribution[level], 7, 0
      );
      diagnostics.setVal(0.0);
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const int ilo = domain.smallEnd(0);
      const int ihi = domain.bigEnd(0);
      const int jlo = domain.smallEnd(1);
      const int jhi = domain.bigEnd(1);
      const int klo = domain.smallEnd(2);
      const int khi = domain.bigEnd(2);
      const auto reflection_lo = hierarchy.reflection_lo;
      const auto reflection_hi = hierarchy.reflection_hi;
      for (amrex::MFIter mfi(
              *primitive_state[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_state[level]->array(mfi);
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto matter_support = matter_support_storage[level]->const_array(mfi);
         const auto diagnostic = diagnostics.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - ilo) + 0.5) * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - jlo) + 0.5) * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - klo) + 0.5) * dx[2],
            };
            const InitialDataSamplePoint sample{physical, physical};
            const amrex::Real displacement1 =
               physical.q1 - iteration_parameters.rotation_center.q1;
            const amrex::Real displacement2 =
               physical.q2 - iteration_parameters.rotation_center.q2;
            const amrex::Real displacement3 =
               physical.q3 - iteration_parameters.rotation_center.q3;
            if (std::sqrt(
                   displacement1 * displacement1
                   + displacement2 * displacement2
                   + displacement3 * displacement3
                ) >= iteration_parameters.update_outer_radius) {
               return;
            }
            bool within_continuation_support = false;
            for (int dk = -1; dk <= 1; ++dk) {
               for (int dj = -1; dj <= 1; ++dj) {
                  for (int di = -1; di <= 1; ++di) {
                     within_continuation_support = within_continuation_support
                        || matter_support(i + di, j + dj, k + dk) > 0.0;
                  }
               }
            }
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const RotatingStarPoint target = rotating_star_point(
               iteration_parameters, sample.physical, metric
            );
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);

            ColdBarotropeState matter;
            amrex::Real v1 = 0.0;
            amrex::Real v2 = 0.0;
            amrex::Real v3 = 0.0;
            if (within_continuation_support && target.timelike
                && target.log_specific_enthalpy > 0.0) {
               const amrex::Real old_log_enthalpy = cold_log_specific_enthalpy(
                  primitive(i, j, k, RHO), eos
               );
               const amrex::Real relaxed_log_enthalpy =
                  (1.0 - iteration_parameters.matter_relaxation)
                     * old_log_enthalpy
                  + iteration_parameters.matter_relaxation
                     * target.log_specific_enthalpy;
               matter = cold_barotrope_from_log_enthalpy(
                  relaxed_log_enthalpy, eos
               );
               v1 = (1.0 - iteration_parameters.matter_relaxation)
                     * primitive(i, j, k, MOM1)
                  + iteration_parameters.matter_relaxation * target.v1;
               v2 = (1.0 - iteration_parameters.matter_relaxation)
                     * primitive(i, j, k, MOM2)
                  + iteration_parameters.matter_relaxation * target.v2;
               v3 = (1.0 - iteration_parameters.matter_relaxation)
                     * primitive(i, j, k, MOM3)
                  + iteration_parameters.matter_relaxation * target.v3;
            }
            if (!initial_matter_exceeds_atmosphere(
                   matter.rho, atmosphere.rho
                )) {
               matter.rho = atmosphere.rho;
               matter.pressure = atmosphere.pressure;
               matter.specific_internal_energy = atmosphere.specific_internal_energy;
               v1 = 0.0;
               v2 = 0.0;
               v3 = 0.0;
            }
            primitive(i, j, k, RHO) = matter.rho;
            primitive(i, j, k, MOM1) = v1;
            primitive(i, j, k, MOM2) = v2;
            primitive(i, j, k, MOM3) = v3;
            primitive(i, j, k, ENERGY) = matter.pressure;
            const bool contains_matter = initial_matter_exceeds_atmosphere(
               matter.rho, atmosphere.rho
            );
            diagnostic(i, j, k, 0) = contains_matter
               ? target.velocity_squared
               : 0.0;
            diagnostic(i, j, k, 1) = matter.rho;
            const bool boundary =
               (i == ilo && reflection_lo[0] == 0)
               || (i == ihi && reflection_hi[0] == 0)
               || (j == jlo && reflection_lo[1] == 0)
               || (j == jhi && reflection_hi[1] == 0)
               || (k == klo && reflection_lo[2] == 0)
               || (k == khi && reflection_hi[2] == 0);
            diagnostic(i, j, k, 2) = boundary && contains_matter
               ? matter.rho - atmosphere.rho
               : 0.0;
            diagnostic(i, j, k, 3) =
               within_continuation_support && !target.timelike ? 1.0 : 0.0;
            if (contains_matter) {
               const amrex::Real displacement1 =
                  physical.q1 - parameters.rotation_center.q1;
               const amrex::Real displacement2 =
                  physical.q2 - parameters.rotation_center.q2;
               const amrex::Real displacement3 =
                  physical.q3 - parameters.rotation_center.q3;
               const amrex::Real parallel =
                  displacement1 * parameters.rotation_axis.q1
                  + displacement2 * parameters.rotation_axis.q2
                  + displacement3 * parameters.rotation_axis.q3;
               const amrex::Real radius_squared =
                  displacement1 * displacement1
                  + displacement2 * displacement2
                  + displacement3 * displacement3;
               diagnostic(i, j, k, 4) = std::sqrt(amrex::max(
                  radius_squared - parallel * parallel,
                  amrex::Real(0.0)
               ));
               diagnostic(i, j, k, 5) = std::abs(parallel);
               diagnostic(i, j, k, 6) = std::sqrt(radius_squared)
                  + 0.5 * std::sqrt(
                     dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
                  );
            }
         });
      }
      summary.maximum_velocity = std::max(
         summary.maximum_velocity,
         std::sqrt(std::max(
            diagnostics.max(0, 0, false), amrex::Real(0.0)
         ))
      );
      summary.maximum_density = std::max(
         summary.maximum_density, diagnostics.max(1, 0, false)
      );
      summary.boundary_density_excess = std::max(
         summary.boundary_density_excess, diagnostics.max(2, 0, false)
      );
      summary.equatorial_radius = std::max(
         summary.equatorial_radius, diagnostics.max(4, 0, false)
      );
      summary.polar_radius = std::max(
         summary.polar_radius, diagnostics.max(5, 0, false)
      );
      summary.matter_outer_radius = std::max(
         summary.matter_outer_radius, diagnostics.max(6, 0, false)
      );
      if (diagnostics.max(3, 0, false) > 0.0) {
         amrex::Abort(
            "rotating-star helical flow is not timelike inside the continued "
            "matter support; reduce angular velocity"
         );
      }
      if (primitive_state[level]->contains_nan(0, NVAR_HYDRO, 0)) {
         amrex::Abort("rotating-star primitive update produced a nonfinite state");
      }
   }
   if (summary.boundary_density_excess > 0.0) {
      amrex::Abort(
         "rotating-star matter reaches the physical boundary; enlarge the domain"
      );
   }
   return summary;
}

#if (AMREX_SPACEDIM >= 2)
void validate_axisymmetric_spherical_rotation(
   InitialDataEllipticHierarchy const& hierarchy,
   RotatingStarParameters const& parameters
)
{
   const amrex::Real tolerance = 64.0
      * std::numeric_limits<amrex::Real>::epsilon();
   if (hierarchy.geometry.size() != 1
       || hierarchy.geometry[0].ProbLo(0) != 0.0
       || std::abs(parameters.rotation_center.q1) > tolerance
       || std::abs(parameters.rotation_center.q2) > tolerance
       || std::abs(parameters.rotation_center.q3) > tolerance
       || std::abs(parameters.rotation_axis.q1) > tolerance
       || std::abs(parameters.rotation_axis.q2) > tolerance
       || std::abs(parameters.rotation_axis.q3 - 1.0) > tolerance) {
      amrex::Abort(
         "axisymmetric spherical rotating-star data require one r-theta "
         "or r-theta-phi level centered at the origin and rotation about +z"
      );
   }
#if (AMREX_SPACEDIM == 3)
   const amrex::Real phi_extent = hierarchy.geometry[0].ProbHi(2)
      - hierarchy.geometry[0].ProbLo(2);
   if (!hierarchy.geometry[0].isPeriodic(2)
       || std::abs(phi_extent - constants::two_pi)
          > tolerance * std::max(amrex::Real(1.0), phi_extent)) {
      amrex::Abort(
         "3D axisymmetric spherical rotating-star data require periodic "
         "full-2pi phi storage"
      );
   }
#endif
}

template <int EOSMethod>
RotatingStarUpdateSummary
update_axisymmetric_spherical_rotating_star_primitives_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
)
{
   validate_axisymmetric_spherical_rotation(hierarchy, parameters);
   if (primitive_state.size() != 1 || cell_spacetime.size() != 1
       || primitive_state[0] == nullptr || cell_spacetime[0] == nullptr
       || primitive_state[0]->nComp() < NVAR_HYDRO
       || cell_spacetime[0]->nComp() < Z4C_NUM_COMPONENTS) {
      amrex::Abort("invalid axisymmetric rotating-star state layout");
   }
   amrex::MultiFab effective(
      hierarchy.grids[0], hierarchy.distribution[0], 2, 1
   );
   amrex::MultiFab matter_support(
      hierarchy.grids[0], hierarchy.distribution[0], 1, 1
   );
   matter_support.setVal(0.0);
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> radial_face_storage(
      radial_coordinates.faces.data(), radial_coordinates.faces.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_center_storage(
      angular_coordinates.centers.data(), angular_coordinates.centers.size()
   );
   const amrex::Real* radial_center = radial_center_storage.data();
   const amrex::Real* radial_face = radial_face_storage.data();
   const amrex::Real* angular_center = angular_center_storage.data();
   const amrex::Box domain = hierarchy.geometry[0].Domain();
   const amrex::Dim3 domain_lo = amrex::lbound(domain);
   const amrex::Dim3 domain_hi = amrex::ubound(domain);
#if (AMREX_SPACEDIM == 3)
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta phi");
   const amrex::Real phi_lo = hierarchy.geometry[0].ProbLo(2);
   const amrex::Real phi_hi = hierarchy.geometry[0].ProbHi(2);
   const amrex::Real dphi = hierarchy.geometry[0].CellSize(2);
#else
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
   const amrex::Real phi_lo = 0.0;
   const amrex::Real phi_hi = 0.0;
   const amrex::Real dphi = 0.0;
#endif
   for (amrex::MFIter mfi(effective, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state[0]->const_array(mfi);
      const auto geometry = cell_spacetime[0]->const_array(mfi);
      const auto effective_array = effective.array(mfi);
      const auto support = matter_support.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         PatchCoordinate mesh{
            radial_center[i - domain_lo.x],
            angular_center[j - domain_lo.y],
            phi_lo + (k - domain_lo.z + 0.5) * dphi,
         };
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(MeshSpherical, axes, mesh),
         };
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const RotatingStarPoint point =
            rotating_star_axisymmetric_spherical_point(
               parameters, sample.mesh.q1, sample.mesh.q2, metric
            );
         effective_array(i, j, k, 0) = point.timelike
            ? metric.lapse * std::sqrt(1.0 - point.velocity_squared)
            : std::numeric_limits<amrex::Real>::max();
         effective_array(i, j, k, 1) = point.timelike ? 0.0 : 1.0;
         const EOSThermodynamicState atmosphere =
            initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
         support(i, j, k) = initial_matter_exceeds_atmosphere(
            primitive(i, j, k, RHO), atmosphere.rho
         ) ? 1.0 : 0.0;
      });
   }
   amrex::Vector<amrex::MultiFab*> effective_fields{&effective};
   amrex::Vector<amrex::MultiFab*> support_fields{&matter_support};
   fill_initial_data_ghosts(
      hierarchy,
      effective_fields,
      initial_data_boundary_records(
         hierarchy,
         {GridFieldSymmetryKind::Scalar, GridFieldSymmetryKind::Scalar}
      )
   );
   fill_initial_data_ghosts(
      hierarchy,
      support_fields,
      initial_data_boundary_records(
         hierarchy, {GridFieldSymmetryKind::Scalar}
      )
   );

   const amrex::Real center_theta = 0.5 * (
      hierarchy.geometry[0].ProbLo(1) + hierarchy.geometry[0].ProbHi(1)
   );
   const amrex::Real center_phi = 0.5 * (
      phi_lo + phi_hi
   );
   std::vector<amrex::GpuArray<amrex::Real, 3>> center_points;
   for (int radial_index = 0; radial_index < 2; ++radial_index) {
      // HierarchyInterpolationPlan addresses the uniform logical AMReX
      // lattice.  The first mapped cells are uniformly spaced physically, so
      // select their logical centers here and retain the even-r quadratic
      // extrapolation below.
      const amrex::Real radius = (radial_index + 0.5) * spacing[0];
      const PatchCoordinate physical = mesh_coordinate_to_physical_point(
         MeshSpherical, axes, {radius, center_theta, center_phi}
      );
      center_points.push_back({{physical.q1, physical.q2, physical.q3}});
   }
   std::vector<std::unique_ptr<amrex::MultiFab>> effective_owner;
   effective_owner.push_back(std::make_unique<amrex::MultiFab>(
      effective, amrex::make_alias, 0, 2
   ));
   const HierarchyInterpolationPlan center_plan(
      std::move(center_points),
      MeshSpherical,
      axes,
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      amrex::IndexType::TheCellType()
   );
   const std::vector<amrex::Real> center_values =
      center_plan.interpolate(effective_owner, 0, 2);
   const amrex::Real center_effective_potential =
      1.125 * center_values[0] - 0.125 * center_values[2];
   const amrex::Real center_failure = std::max(
      center_values[1], center_values[3]
   );
   if (!(center_effective_potential > 0.0)
       || !std::isfinite(center_effective_potential)
       || center_failure > 0.0) {
      amrex::Abort("axisymmetric rotating-star center is not timelike");
   }
   RotatingStarParameters iteration_parameters = parameters;
   iteration_parameters.injection_energy = std::exp(
      parameters.central_log_specific_enthalpy
   ) * center_effective_potential;

   amrex::MultiFab diagnostics(
      hierarchy.grids[0], hierarchy.distribution[0], 8, 0
   );
   diagnostics.setVal(0.0);
   for (amrex::MFIter mfi(*primitive_state[0], amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state[0]->array(mfi);
      const auto geometry = cell_spacetime[0]->const_array(mfi);
      const auto support = matter_support.const_array(mfi);
      const auto diagnostic = diagnostics.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         PatchCoordinate mesh{
            radial_center[i - domain_lo.x],
            angular_center[j - domain_lo.y],
            phi_lo + (k - domain_lo.z + 0.5) * dphi,
         };
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(MeshSpherical, axes, mesh),
         };
         if (sample.mesh.q1 >= iteration_parameters.update_outer_radius) {
            return;
         }
         bool within_support = false;
         for (int dk = AMREX_SPACEDIM == 3 ? -1 : 0;
              dk <= (AMREX_SPACEDIM == 3 ? 1 : 0);
              ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
               for (int di = -1; di <= 1; ++di) {
                  within_support = within_support
                     || support(i + di, j + dj, k + dk) > 0.0;
               }
            }
         }
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const RotatingStarPoint target =
            rotating_star_axisymmetric_spherical_point(
               iteration_parameters, sample.mesh.q1, sample.mesh.q2, metric
            );
         const EOSThermodynamicState atmosphere =
            initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
         ColdBarotropeState matter;
         amrex::Real vr = 0.0;
         amrex::Real vtheta = 0.0;
         amrex::Real vphi = 0.0;
         if (within_support && target.timelike
             && target.log_specific_enthalpy > 0.0) {
            const amrex::Real old_log_enthalpy = cold_log_specific_enthalpy(
               primitive(i, j, k, RHO), eos
            );
            const amrex::Real relaxed_log_enthalpy =
               (1.0 - iteration_parameters.matter_relaxation)
                  * old_log_enthalpy
               + iteration_parameters.matter_relaxation
                  * target.log_specific_enthalpy;
            matter = cold_barotrope_from_log_enthalpy(
               relaxed_log_enthalpy, eos
            );
            vr = (1.0 - iteration_parameters.matter_relaxation)
                  * primitive(i, j, k, MOM1)
               + iteration_parameters.matter_relaxation * target.v1;
            vtheta = (1.0 - iteration_parameters.matter_relaxation)
                  * primitive(i, j, k, MOM2)
               + iteration_parameters.matter_relaxation * target.v2;
            vphi = (1.0 - iteration_parameters.matter_relaxation)
                  * primitive(i, j, k, MOM3)
               + iteration_parameters.matter_relaxation * target.v3;
         }
         if (!initial_matter_exceeds_atmosphere(
                matter.rho, atmosphere.rho
             )) {
            matter.rho = atmosphere.rho;
            matter.pressure = atmosphere.pressure;
            matter.specific_internal_energy = atmosphere.specific_internal_energy;
            vr = 0.0;
            vtheta = 0.0;
            vphi = 0.0;
         }
         primitive(i, j, k, RHO) = matter.rho;
         primitive(i, j, k, MOM1) = vr;
         primitive(i, j, k, MOM2) = vtheta;
         primitive(i, j, k, MOM3) = vphi;
         primitive(i, j, k, ENERGY) = matter.pressure;
         const bool contains_matter = initial_matter_exceeds_atmosphere(
            matter.rho, atmosphere.rho
         );
         diagnostic(i, j, k, 0) = contains_matter
            ? target.velocity_squared : 0.0;
         diagnostic(i, j, k, 1) = matter.rho;
         diagnostic(i, j, k, 2) = i == domain_hi.x && contains_matter
            ? matter.rho - atmosphere.rho : 0.0;
         diagnostic(i, j, k, 3) = within_support && !target.timelike
            ? 1.0 : 0.0;
         const bool failed_support = within_support && !target.timelike;
         diagnostic(i, j, k, 7) = failed_support ? sample.mesh.q1 : 0.0;
         if (contains_matter) {
            diagnostic(i, j, k, 4) = sample.mesh.q1
               * std::sin(sample.mesh.q2);
            diagnostic(i, j, k, 5) = sample.mesh.q1
               * std::abs(std::cos(sample.mesh.q2));
            diagnostic(i, j, k, 6) = radial_face[
               i - domain_lo.x + 1
            ];
         }
      });
   }
   RotatingStarUpdateSummary summary;
   summary.maximum_velocity = std::sqrt(std::max(
      diagnostics.max(0, 0, false), amrex::Real(0.0)
   ));
   summary.maximum_density = diagnostics.max(1, 0, false);
   summary.boundary_density_excess = diagnostics.max(2, 0, false);
   summary.injection_energy = iteration_parameters.injection_energy;
   summary.equatorial_radius = diagnostics.max(4, 0, false);
   summary.polar_radius = diagnostics.max(5, 0, false);
   summary.matter_outer_radius = diagnostics.max(6, 0, false);
   if (diagnostics.max(3, 0, false) > 0.0) {
      amrex::Print()
         << "axisymmetric rotating-star failed support maximum_radius="
         << diagnostics.max(7, 0, false) << '\n';
      amrex::Abort(
         "axisymmetric rotating-star helical flow is not timelike inside matter"
      );
   }
   if (summary.boundary_density_excess > 0.0) {
      amrex::Abort(
         "axisymmetric rotating-star matter reaches the radial boundary"
      );
   }
   if (primitive_state[0]->contains_nan(0, NVAR_HYDRO, 0)) {
      amrex::Abort("axisymmetric rotating-star update is nonfinite");
   }
   return summary;
}
#endif

} // namespace

RotatingStarSurfaceDiagnostic rotating_star_surface_diagnostic(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   RotatingStarParameters const& parameters
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(hierarchy, cell_spacetime, parameters);
   amrex::Abort("rotating-star surface diagnostics require the 3D executable");
   return {};
#else
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> log_enthalpy(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      log_enthalpy[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 1, 1
      );
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const int ilo = domain.smallEnd(0);
      const int jlo = domain.smallEnd(1);
      const int klo = domain.smallEnd(2);
      for (amrex::MFIter mfi(*log_enthalpy[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto enthalpy = log_enthalpy[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - ilo) + 0.5) * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - jlo) + 0.5) * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - klo) + 0.5) * dx[2],
            };
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const RotatingStarPoint point = rotating_star_point(
               parameters, physical, metric
            );
            enthalpy(i, j, k) = point.timelike
               ? point.log_specific_enthalpy
               : -std::numeric_limits<amrex::Real>::max();
         });
      }
   }
   amrex::Vector<amrex::MultiFab*> log_enthalpy_fields(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      log_enthalpy_fields[level] = log_enthalpy[level].get();
   }
   fill_initial_data_ghosts(
      hierarchy,
      log_enthalpy_fields,
      initial_data_boundary_records(
         hierarchy, {GridFieldSymmetryKind::Scalar}
      )
   );

   const std::array<amrex::Real, 3> axis{
      parameters.rotation_axis.q1,
      parameters.rotation_axis.q2,
      parameters.rotation_axis.q3,
   };
   int reference_direction = 0;
   for (int direction = 1; direction < 3; ++direction) {
      if (std::abs(axis[direction]) < std::abs(axis[reference_direction])) {
         reference_direction = direction;
      }
   }
   std::array<amrex::Real, 3> reference{0.0, 0.0, 0.0};
   reference[reference_direction] = 1.0;
   std::array<amrex::Real, 3> equator1{
      axis[1] * reference[2] - axis[2] * reference[1],
      axis[2] * reference[0] - axis[0] * reference[2],
      axis[0] * reference[1] - axis[1] * reference[0],
   };
   const amrex::Real equator1_norm = std::sqrt(
      equator1[0] * equator1[0] + equator1[1] * equator1[1]
      + equator1[2] * equator1[2]
   );
   for (amrex::Real& component : equator1) {
      component /= equator1_norm;
   }
   const std::array<amrex::Real, 3> equator2{
      axis[1] * equator1[2] - axis[2] * equator1[1],
      axis[2] * equator1[0] - axis[0] * equator1[2],
      axis[0] * equator1[1] - axis[1] * equator1[0],
   };
   const std::array<std::array<amrex::Real, 3>, 6> ray_directions{{
      axis,
      {-axis[0], -axis[1], -axis[2]},
      equator1,
      {-equator1[0], -equator1[1], -equator1[2]},
      equator2,
      {-equator2[0], -equator2[1], -equator2[2]},
   }};

   const auto lower = hierarchy.geometry[0].ProbLoArray();
   const auto upper = hierarchy.geometry[0].ProbHiArray();
   const auto coarse_dx = hierarchy.geometry[0].CellSizeArray();
   amrex::Real spacing = coarse_dx[0];
   for (int level = 0; level < num_levels; ++level) {
      const auto level_dx = hierarchy.geometry[level].CellSizeArray();
      spacing = std::min({spacing, level_dx[0], level_dx[1], level_dx[2]});
   }
   spacing *= 0.5;
   const std::array<amrex::Real, 3> center{
      parameters.rotation_center.q1,
      parameters.rotation_center.q2,
      parameters.rotation_center.q3,
   };
   std::vector<amrex::GpuArray<amrex::Real, 3>> points;
   std::array<std::size_t, 6> offsets{};
   std::array<int, 6> counts{};
   for (int ray = 0; ray < 6; ++ray) {
      offsets[ray] = points.size();
      amrex::Real extent = std::numeric_limits<amrex::Real>::max();
      for (int direction = 0; direction < 3; ++direction) {
         const amrex::Real ray_component = ray_directions[ray][direction];
         if (ray_component > 0.0) {
            const amrex::Real ray_extent =
               hierarchy.reflection_hi[direction] != 0
               ? (
                    2.0 * upper[direction] - center[direction]
                    - lower[direction] - 0.6 * coarse_dx[direction]
                 ) / ray_component
               : (
                    upper[direction] - 1.6 * coarse_dx[direction]
                    - center[direction]
                 ) / ray_component;
            extent = std::min(extent, ray_extent);
         } else if (ray_component < 0.0) {
            const amrex::Real ray_extent =
               hierarchy.reflection_lo[direction] != 0
               ? (
                    upper[direction] - 1.6 * coarse_dx[direction]
                    - 2.0 * lower[direction] + center[direction]
                 ) / (-ray_component)
               : (
                    lower[direction] + 0.6 * coarse_dx[direction]
                    - center[direction]
                 ) / ray_component;
            extent = std::min(extent, ray_extent);
         }
      }
      counts[ray] = static_cast<int>(std::floor(0.98 * extent / spacing)) + 1;
      if (counts[ray] < 6) {
         amrex::Abort("rotating-star surface diagnostic domain is too small");
      }
      for (int sample = 0; sample < counts[ray]; ++sample) {
         const amrex::Real radius = sample * spacing;
         amrex::GpuArray<amrex::Real, 3> point{{
            center[0] + radius * ray_directions[ray][0],
            center[1] + radius * ray_directions[ray][1],
            center[2] + radius * ray_directions[ray][2],
         }};
         for (int direction = 0; direction < 3; ++direction) {
            if (hierarchy.reflection_lo[direction] != 0
                && point[direction] < lower[direction]) {
               point[direction] =
                  2.0 * lower[direction] - point[direction];
            }
            if (hierarchy.reflection_hi[direction] != 0
                && point[direction] > upper[direction]) {
               point[direction] =
                  2.0 * upper[direction] - point[direction];
            }
            if (point[direction] == lower[direction]
                && hierarchy.reflection_lo[direction] != 0) {
               point[direction] += 0.5 * coarse_dx[direction];
            } else if (point[direction] == upper[direction]
                       && hierarchy.reflection_hi[direction] != 0) {
               point[direction] -= 0.5 * coarse_dx[direction];
            }
         }
         points.push_back(point);
      }
   }
   const HierarchyInterpolationPlan plan(
      std::move(points),
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian),
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      amrex::IndexType::TheCellType()
   );
   const std::vector<amrex::Real> values = plan.interpolate(log_enthalpy, 0, 1);
   amrex::Real polar_gradient = 0.0;
   amrex::Real equatorial_gradient = 0.0;
   for (int ray = 0; ray < 6; ++ray) {
      const amrex::Real derivative = ray_surface_derivative(
         values, offsets[ray], counts[ray], spacing
      );
      if (ray < 2) {
         polar_gradient += 0.5 * derivative;
      } else {
         equatorial_gradient += 0.25 * derivative;
      }
   }
   const amrex::Real indicator = equatorial_gradient / polar_gradient;
   if (!(polar_gradient < 0.0) || !std::isfinite(equatorial_gradient)
       || !std::isfinite(indicator)) {
      amrex::Abort(
         "rotating-star mass-shedding diagnostic requires a negative finite "
         "polar enthalpy gradient"
      );
   }
   return {
      equatorial_gradient,
      polar_gradient,
      indicator,
   };
#endif
}

RotatingStarSurfaceDiagnostic
axisymmetric_spherical_rotating_star_surface_diagnostic(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   RotatingStarParameters const& parameters
)
{
#if (AMREX_SPACEDIM < 2)
   amrex::ignore_unused(hierarchy, cell_spacetime, parameters);
   amrex::Abort(
      "axisymmetric spherical surface diagnostics require 2D or 3D"
   );
   return {};
#else
   validate_axisymmetric_spherical_rotation(hierarchy, parameters);
   if (cell_spacetime.size() != 1 || cell_spacetime[0] == nullptr) {
      amrex::Abort("invalid axisymmetric surface-diagnostic geometry");
   }
   auto log_enthalpy = std::make_unique<amrex::MultiFab>(
      hierarchy.grids[0], hierarchy.distribution[0], 1, 1
   );
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   amrex::Gpu::AsyncArray<amrex::Real> angular_center_storage(
      angular_coordinates.centers.data(), angular_coordinates.centers.size()
   );
   const amrex::Real* radial_center = radial_center_storage.data();
   const amrex::Real* angular_center = angular_center_storage.data();
#if (AMREX_SPACEDIM == 3)
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta phi");
   const amrex::Real phi_lo = hierarchy.geometry[0].ProbLo(2);
   const amrex::Real phi_hi = hierarchy.geometry[0].ProbHi(2);
   const amrex::Real dphi = hierarchy.geometry[0].CellSize(2);
#else
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
   const amrex::Real phi_lo = 0.0;
   const amrex::Real phi_hi = 0.0;
   const amrex::Real dphi = 0.0;
#endif
   for (amrex::MFIter mfi(*log_enthalpy, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto geometry = cell_spacetime[0]->const_array(mfi);
      const auto enthalpy = log_enthalpy->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         PatchCoordinate mesh{
            radial_center[i - domain_lo.x],
            angular_center[j - domain_lo.y],
            phi_lo + (k - domain_lo.z + 0.5) * dphi,
         };
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(MeshSpherical, axes, mesh),
         };
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const RotatingStarPoint point =
            rotating_star_axisymmetric_spherical_point(
               parameters, sample.mesh.q1, sample.mesh.q2, metric
            );
         enthalpy(i, j, k) = point.timelike
            ? point.log_specific_enthalpy
            : -std::numeric_limits<amrex::Real>::max();
      });
   }
   amrex::Vector<amrex::MultiFab*> enthalpy_fields{log_enthalpy.get()};
   fill_initial_data_ghosts(
      hierarchy,
      enthalpy_fields,
      initial_data_boundary_records(
         hierarchy, {GridFieldSymmetryKind::Scalar}
      )
   );
   const int radial_cells = hierarchy.geometry[0].Domain().length(0);
   if (radial_cells < 6) {
      amrex::Abort("axisymmetric surface diagnostic needs at least six radial cells");
   }
   int uniform_source_cells = radial_cells;
   const amrex::Real inner_width = radial_coordinates.widths.front();
   const amrex::Real width_tolerance = 64.0
      * std::numeric_limits<amrex::Real>::epsilon()
      * std::max(amrex::Real(1.0), inner_width);
   while (uniform_source_cells > 1
          && std::abs(
                radial_coordinates.widths[uniform_source_cells - 1]
                - inner_width
             ) > width_tolerance) {
      --uniform_source_cells;
   }
   const int radial_samples = uniform_source_cells - 1;
   const amrex::Real polar_theta = problo[1] + 0.5001 * spacing[1];
   const amrex::Real equatorial_theta =
      hierarchy.geometry[0].ProbHi(1) - 0.5001 * spacing[1];
   const amrex::Real sample_phi = 0.5 * (
      phi_lo + phi_hi
   );
   std::vector<amrex::GpuArray<amrex::Real, 3>> points;
   points.reserve(2 * radial_samples);
   for (amrex::Real theta : {polar_theta, equatorial_theta}) {
      for (int index = 0; index < radial_samples; ++index) {
         // Sample strictly between the first and last cell centers.  This
         // avoids grid-edge ambiguity after Cartesian-to-spherical mapping;
         // the stellar surface is well inside this radial interval.
         const amrex::Real radius = (index + 1.0) * spacing[0];
         const PatchCoordinate physical = mesh_coordinate_to_physical_point(
            MeshSpherical, axes, {radius, theta, sample_phi}
         );
         points.push_back({{physical.q1, physical.q2, physical.q3}});
      }
   }
   std::vector<std::unique_ptr<amrex::MultiFab>> enthalpy_owner;
   enthalpy_owner.push_back(std::move(log_enthalpy));
   const HierarchyInterpolationPlan plan(
      std::move(points),
      MeshSpherical,
      axes,
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      amrex::IndexType::TheCellType()
   );
   const std::vector<amrex::Real> values = plan.interpolate(
      enthalpy_owner, 0, 1
   );
   const amrex::Real polar_gradient = ray_surface_derivative(
      values, 0, radial_samples, inner_width
   );
   const amrex::Real equatorial_gradient = ray_surface_derivative(
      values, radial_samples, radial_samples, inner_width
   );
   const amrex::Real indicator = equatorial_gradient / polar_gradient;
   if (!(polar_gradient < 0.0) || !std::isfinite(equatorial_gradient)
       || !std::isfinite(indicator)) {
      amrex::Abort("invalid axisymmetric rotating-star surface diagnostic");
   }
   return {equatorial_gradient, polar_gradient, indicator};
#endif
}

RotatingStarParameters rotating_star_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
)
{
   RotatingStarParameters parameters;
   parameters.active = cfg.ic == "rotating_star"
      || (cfg.ic == "tov"
          && (cfg.initial_data_disk == "equilibrium_torus"
              || cfg.initial_data_disk == "tilted_torus"));
   if (!parameters.active) {
      return parameters;
   }
   parameters.rotation_law = rotation_law_from_name(
      cfg.initial_data_rotation_law
   );
   parameters.angular_velocity = cfg.initial_data_angular_velocity;
   parameters.differential_rotation_length =
      cfg.initial_data_differential_rotation_length;
   parameters.differential_rotation_length_b =
      cfg.initial_data_differential_rotation_length_b;
   parameters.rotation_power_q = cfg.initial_data_rotation_power_q;
   const amrex::Real axis_norm = std::sqrt(
      cfg.initial_data_rotation_axis1 * cfg.initial_data_rotation_axis1
      + cfg.initial_data_rotation_axis2 * cfg.initial_data_rotation_axis2
      + cfg.initial_data_rotation_axis3 * cfg.initial_data_rotation_axis3
   );
   parameters.rotation_axis = {
      cfg.initial_data_rotation_axis1 / axis_norm,
      cfg.initial_data_rotation_axis2 / axis_norm,
      cfg.initial_data_rotation_axis3 / axis_norm,
   };
   parameters.rotation_center = {
      cfg.initial_data_rotation_center1,
      cfg.initial_data_rotation_center2,
      cfg.initial_data_rotation_center3,
   };
   parameters.injection_energy = provider.tov.injection_energy;
   parameters.central_log_specific_enthalpy = cold_log_specific_enthalpy(
      cfg.tov_central_density, eos_params(cfg)
   );
   parameters.matter_relaxation = cfg.initial_data_matter_relaxation;
   parameters.relative_tolerance = cfg.initial_data_matter_relative_tolerance;
   parameters.max_iterations = cfg.initial_data_matter_max_iterations;
   parameters.continuation_steps = cfg.initial_data_rotation_continuation_steps;
   parameters.continuation_targets =
      cfg.initial_data_rotation_continuation_targets;
   parameters.atmosphere = initial_atmosphere_parameters(cfg, provider);
   if (!provider.tov.active() || !(parameters.injection_energy > 0.0)) {
      amrex::Abort("rotating-star continuation requires an active TOV seed");
   }
   return parameters;
}

RotatingStarUpdateSummary update_rotating_star_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, primitive_state, cell_spacetime, eos_method, eos, parameters
   );
   amrex::Abort("rotating-star equilibrium requires the 3D executable");
   return {};
#else
   if (!parameters.active) {
      return {};
   }
   amrex::ignore_unused(eos_method);
   if (cold_barotrope_supported(eos)) {
      return update_rotating_star_primitives_method<EOSRuntime>(
         hierarchy, primitive_state, cell_spacetime, eos, parameters
      );
   }
   amrex::Abort("rotating-star equilibrium requires a cold barotropic EOS");
   return {};
#endif
}

RotatingStarUpdateSummary update_axisymmetric_spherical_rotating_star_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   RotatingStarParameters const& parameters
)
{
#if (AMREX_SPACEDIM < 2)
   amrex::ignore_unused(
      hierarchy, primitive_state, cell_spacetime, eos_method, eos, parameters
   );
   amrex::Abort("axisymmetric rotating-star equilibrium requires 2D or 3D");
   return {};
#else
   if (!parameters.active) {
      return {};
   }
   amrex::ignore_unused(eos_method);
   if (cold_barotrope_supported(eos)) {
      return update_axisymmetric_spherical_rotating_star_primitives_method<
         EOSRuntime
      >(hierarchy, primitive_state, cell_spacetime, eos, parameters);
   }
   amrex::Abort("axisymmetric rotating stars require a cold barotropic EOS");
   return {};
#endif
}

} // namespace pyfluid_amrex
