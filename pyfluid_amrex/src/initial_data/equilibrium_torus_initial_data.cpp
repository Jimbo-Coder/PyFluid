#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"

#include <AMReX.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/cold_barotrope.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {
namespace {

DiskRotationLaw disk_rotation_law_from_name(std::string const& name)
{
   for (DiskRotationLawInfo const& law : DiskRotationLawCatalog) {
      if (law.name == name) {
         return law.id;
      }
   }
   amrex::Abort("unknown generated disk rotation law: " + name);
   return DiskRotationLaw::ConstantL;
}

#if (AMREX_SPACEDIM == 3)
std::array<amrex::Real, 3> torus_equatorial_direction(
   EquilibriumTorusParameters const& parameters
)
{
   const std::array<amrex::Real, 3> axis{
      parameters.axis1, parameters.axis2, parameters.axis3,
   };
   int reference_direction = 0;
   for (int direction = 1; direction < 3; ++direction) {
      if (std::abs(axis[direction]) < std::abs(axis[reference_direction])) {
         reference_direction = direction;
      }
   }
   std::array<amrex::Real, 3> reference{0.0, 0.0, 0.0};
   reference[reference_direction] = 1.0;
   std::array<amrex::Real, 3> radial{
      axis[1] * reference[2] - axis[2] * reference[1],
      axis[2] * reference[0] - axis[0] * reference[2],
      axis[0] * reference[1] - axis[1] * reference[0],
   };
   const amrex::Real norm = std::sqrt(
      radial[0] * radial[0] + radial[1] * radial[1]
      + radial[2] * radial[2]
   );
   for (amrex::Real& component : radial) {
      component /= norm;
   }
   return radial;
}

amrex::Real inner_edge_normalization_3d(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EquilibriumTorusParameters const& parameters
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> normalization(num_levels);
   amrex::Vector<amrex::MultiFab*> normalization_view(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      normalization[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 2, 1
      );
      normalization[level]->setVal(0.0);
      normalization_view[level] = normalization[level].get();
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      for (amrex::MFIter mfi(
              *normalization[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto output = normalization[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - domain_lo.x) + 0.5)
                  * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - domain_lo.y) + 0.5)
                  * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - domain_lo.z) + 0.5)
                  * dx[2],
            };
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const EquilibriumTorusPoint point = equilibrium_torus_point(
               parameters, 1.0, physical, metric
            );
            output(i, j, k, 0) = parameters.rotation_law
                  == DiskRotationLaw::ConstantL
               ? point.minus_covariant_time_velocity
               : std::exp(-point.log_specific_enthalpy);
            output(i, j, k, 1) = point.timelike ? 0.0 : 1.0;
         });
      }
   }
   fill_initial_data_ghosts(
      hierarchy,
      normalization_view,
      initial_data_boundary_records(
         hierarchy,
         {GridFieldSymmetryKind::Scalar, GridFieldSymmetryKind::Scalar}
      )
   );
   const std::array<amrex::Real, 3> radial =
      torus_equatorial_direction(parameters);
   amrex::GpuArray<amrex::Real, 3> inner_point{{
      parameters.center1 + parameters.inner_radius * radial[0],
      parameters.center2 + parameters.inner_radius * radial[1],
      parameters.center3 + parameters.inner_radius * radial[2],
   }};
   const auto coarse_dx = hierarchy.geometry[0].CellSizeArray();
   const auto coarse_problo = hierarchy.geometry[0].ProbLoArray();
   const auto coarse_probhi = hierarchy.geometry[0].ProbHiArray();
   for (int direction = 0; direction < 3; ++direction) {
      const bool lower_reflection = hierarchy.reflection_lo[direction] != 0;
      const bool upper_reflection = hierarchy.reflection_hi[direction] != 0;
      if (!lower_reflection && !upper_reflection) {
         continue;
      }
      const amrex::Real plane = lower_reflection
         ? coarse_problo[direction] : coarse_probhi[direction];
      const amrex::Real tolerance = 64.0
         * std::numeric_limits<amrex::Real>::epsilon()
         * std::max(amrex::Real(1.0), std::abs(plane));
      if (std::abs(inner_point[direction] - plane) > tolerance) {
         continue;
      }
      // On the corresponding full cell-centered lattice, linear
      // interpolation at an even reflection plane averages the two adjacent
      // centers.  Their values are identical by parity, so sampling the first
      // retained center reproduces the full-domain discrete normalization.
      const amrex::Real inward_sign = lower_reflection ? 1.0 : -1.0;
      inner_point[direction] = plane
         + inward_sign * 0.5 * coarse_dx[direction];
   }
   std::vector<amrex::GpuArray<amrex::Real, 3>> inner_points{inner_point};
   const HierarchyInterpolationPlan plan(
      std::move(inner_points),
      MeshCartesian,
      mesh_default_axes_layout(MeshCartesian),
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      amrex::IndexType::TheCellType()
   );
   const std::vector<amrex::Real> values = plan.interpolate(
      normalization, 0, 2
   );
   const amrex::Real result = values[0];
   if (!(result > 0.0) || !std::isfinite(result) || values[1] > 1.0e-12) {
      amrex::Abort(
         "equilibrium-torus inner edge does not admit a resolved timelike "
         "circular orbit"
      );
   }
   return result;
}

template <int EOSMethod>
EquilibriumTorusUpdateSummary update_equilibrium_torus_primitives_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(primitive_state.size()) != num_levels
       || static_cast<int>(cell_spacetime.size()) != num_levels) {
      amrex::Abort("invalid 3D equilibrium-torus hierarchy");
   }
   const amrex::Real inner_normalization = inner_edge_normalization_3d(
      hierarchy, cell_spacetime, parameters
   );
   EquilibriumTorusUpdateSummary summary;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> diagnostics(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      if (primitive_state[level] == nullptr || cell_spacetime[level] == nullptr
          || primitive_state[level]->nComp() < NVAR_HYDRO
          || cell_spacetime[level]->nComp() < Z4C_NUM_COMPONENTS) {
         amrex::Abort("invalid 3D equilibrium-torus state layout");
      }
      diagnostics[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level], hierarchy.distribution[level], 9, 0
      );
      diagnostics[level]->setVal(0.0);
      diagnostics[level]->setVal(
         std::numeric_limits<amrex::Real>::max(), 3, 1, 0
      );
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Dim3 domain_hi = amrex::ubound(domain);
      const auto reflection_lo = hierarchy.reflection_lo;
      const auto reflection_hi = hierarchy.reflection_hi;
      const amrex::Real half_diagonal = 0.5 * std::sqrt(
         dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
      );
      for (amrex::MFIter mfi(
              *primitive_state[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto primitive = primitive_state[level]->array(mfi);
         const auto geometry = cell_spacetime[level]->const_array(mfi);
         const auto diagnostic = diagnostics[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const PatchCoordinate physical{
               problo[0] + (static_cast<amrex::Real>(i - domain_lo.x) + 0.5)
                  * dx[0],
               problo[1] + (static_cast<amrex::Real>(j - domain_lo.y) + 0.5)
                  * dx[1],
               problo[2] + (static_cast<amrex::Real>(k - domain_lo.z) + 0.5)
                  * dx[2],
            };
            const amrex::Real displacement1 = physical.q1 - parameters.center1;
            const amrex::Real displacement2 = physical.q2 - parameters.center2;
            const amrex::Real displacement3 = physical.q3 - parameters.center3;
            const amrex::Real spherical_radius = std::sqrt(
               displacement1 * displacement1 + displacement2 * displacement2
               + displacement3 * displacement3
            );
            if (spherical_radius < parameters.inner_radius) {
               return;
            }
            const amrex::Real parallel = displacement1 * parameters.axis1
               + displacement2 * parameters.axis2
               + displacement3 * parameters.axis3;
            const amrex::Real cylindrical_radius = std::sqrt(amrex::max(
               spherical_radius * spherical_radius - parallel * parallel,
               amrex::Real(0.0)
            ));
            const InitialDataSamplePoint sample{physical, physical};
            const EOSThermodynamicState atmosphere =
               initial_atmosphere_state<EOSMethod>(parameters.atmosphere, sample);
            const bool contained_old_torus =
               parameters.continue_existing_matter
               && initial_matter_exceeds_atmosphere(
                    primitive(i, j, k, RHO), atmosphere.rho
                  );
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const EquilibriumTorusPoint target = equilibrium_torus_point(
               parameters, inner_normalization, physical, metric
            );
            diagnostic(i, j, k, 2) = target.timelike
               ? amrex::max(target.log_specific_enthalpy, amrex::Real(0.0))
               : 0.0;
            ColdBarotropeState matter;
            amrex::Real v1 = 0.0;
            amrex::Real v2 = 0.0;
            amrex::Real v3 = 0.0;
            if (target.timelike && target.log_specific_enthalpy > 0.0) {
               const amrex::Real old_log_enthalpy = contained_old_torus
                  ? cold_log_specific_enthalpy(
                       primitive(i, j, k, RHO), eos
                    )
                  : 0.0;
               const amrex::Real relaxed_log_enthalpy =
                  (1.0 - parameters.matter_relaxation) * old_log_enthalpy
                  + parameters.matter_relaxation
                     * target.log_specific_enthalpy;
               matter = cold_barotrope_from_log_enthalpy(
                  relaxed_log_enthalpy, eos
               );
               v1 = (1.0 - parameters.matter_relaxation)
                     * primitive(i, j, k, MOM1)
                  + parameters.matter_relaxation * target.v1;
               v2 = (1.0 - parameters.matter_relaxation)
                     * primitive(i, j, k, MOM2)
                  + parameters.matter_relaxation * target.v2;
               v3 = (1.0 - parameters.matter_relaxation)
                     * primitive(i, j, k, MOM3)
                  + parameters.matter_relaxation * target.v3;
            }
            if (!initial_matter_exceeds_atmosphere(
                   matter.rho, atmosphere.rho
                )) {
               matter.rho = atmosphere.rho;
               matter.pressure = atmosphere.pressure;
               matter.specific_internal_energy =
                  atmosphere.specific_internal_energy;
               v1 = 0.0;
               v2 = 0.0;
               v3 = 0.0;
            }
            primitive(i, j, k, RHO) = matter.rho;
            primitive(i, j, k, MOM1) = v1;
            primitive(i, j, k, MOM2) = v2;
            primitive(i, j, k, MOM3) = v3;
            primitive(i, j, k, ENERGY) = matter.pressure;
            const bool contains_torus = initial_matter_exceeds_atmosphere(
               matter.rho, atmosphere.rho
            );
            if (contains_torus) {
               diagnostic(i, j, k, 0) = matter.rho;
               diagnostic(i, j, k, 1) = target.velocity_squared;
               diagnostic(i, j, k, 3) = cylindrical_radius;
               diagnostic(i, j, k, 4) = spherical_radius + half_diagonal;
               diagnostic(i, j, k, 5) = cylindrical_radius;
            }
            const bool boundary =
               (i == domain_lo.x && reflection_lo[0] == 0)
               || (i == domain_hi.x && reflection_hi[0] == 0)
               || (j == domain_lo.y && reflection_lo[1] == 0)
               || (j == domain_hi.y && reflection_hi[1] == 0)
               || (k == domain_lo.z && reflection_lo[2] == 0)
               || (k == domain_hi.z && reflection_hi[2] == 0);
            diagnostic(i, j, k, 6) = boundary && contains_torus
               ? matter.rho - atmosphere.rho : 0.0;
            diagnostic(i, j, k, 7) = contained_old_torus && !target.timelike
               ? 1.0 : 0.0;
         });
      }
      summary.maximum_density = std::max(
         summary.maximum_density, diagnostics[level]->max(0, 0, false)
      );
      summary.maximum_velocity = std::max(
         summary.maximum_velocity,
         std::sqrt(std::max(
            diagnostics[level]->max(1, 0, false), amrex::Real(0.0)
         ))
      );
      summary.maximum_log_specific_enthalpy = std::max(
         summary.maximum_log_specific_enthalpy,
         diagnostics[level]->max(2, 0, false)
      );
      summary.inner_radius = summary.inner_radius == 0.0
         ? diagnostics[level]->min(3, 0, false)
         : std::min(summary.inner_radius, diagnostics[level]->min(3, 0, false));
      summary.outer_radius = std::max(
         summary.outer_radius, diagnostics[level]->max(4, 0, false)
      );
      summary.boundary_density_excess = std::max(
         summary.boundary_density_excess,
         diagnostics[level]->max(6, 0, false)
      );
      if (diagnostics[level]->max(7, 0, false) > 0.0) {
         amrex::Abort(
            "equilibrium-torus continuation lost a timelike orbit inside its "
            "previous matter support"
         );
      }
      if (primitive_state[level]->contains_nan(0, NVAR_HYDRO, 0)) {
         amrex::Abort("3D equilibrium-torus update is nonfinite");
      }
   }
   const bool resolved_support = summary.maximum_density > 0.0
      && summary.outer_radius > summary.inner_radius
      && std::isfinite(summary.inner_radius);
   if (!resolved_support) {
      if (amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print()
            << "3D torus unresolved: inner_normalization="
            << inner_normalization
            << " maximum_log_specific_enthalpy="
            << summary.maximum_log_specific_enthalpy
            << " maximum_density=" << summary.maximum_density << '\n';
      }
      if (!parameters.continue_existing_matter) {
         return summary;
      }
      amrex::Abort(
         "equilibrium-torus parameters do not produce resolved finite support"
      );
   }
   if (summary.boundary_density_excess > 0.0) {
      amrex::Abort(
         "equilibrium-torus matter reaches the physical boundary; enlarge "
         "the ID domain"
      );
   }
   const amrex::Real density_tolerance = 128.0
      * std::numeric_limits<amrex::Real>::epsilon();
   for (int level = 0; level < num_levels; ++level) {
      const amrex::Real maximum_density = summary.maximum_density;
      for (amrex::MFIter mfi(
              *diagnostics[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto diagnostic = diagnostics[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            diagnostic(i, j, k, 8) = diagnostic(i, j, k, 0)
                  >= maximum_density * (1.0 - density_tolerance)
               ? diagnostic(i, j, k, 5) : 0.0;
         });
      }
      summary.density_maximum_radius = std::max(
         summary.density_maximum_radius,
         diagnostics[level]->max(8, 0, false)
      );
   }
   return summary;
}
#endif

#if (AMREX_SPACEDIM >= 2)
struct ReferenceWeight {
   int radial_index = 0;
   int angular_index = 0;
   amrex::Real weight = 0.0;
   amrex::Real radius = 0.0;
   amrex::Real theta = 0.0;
};

std::vector<std::pair<int, amrex::Real>> linear_reference_weights(
   std::vector<amrex::Real> const& centers,
   amrex::Real coordinate,
   char const* label
)
{
   if (centers.size() < 2 || coordinate < centers.front()
       || coordinate > centers.back()) {
      amrex::Abort(
         std::string("equilibrium-torus ") + label
         + " lies outside cell-center interpolation support"
      );
   }
   const auto upper = std::lower_bound(
      centers.begin(), centers.end(), coordinate
   );
   if (upper == centers.end()) {
      return {{static_cast<int>(centers.size()) - 1, 1.0}};
   }
   const int upper_index = static_cast<int>(upper - centers.begin());
   const amrex::Real scale = std::max(amrex::Real(1.0), std::abs(coordinate));
   if (std::abs(*upper - coordinate)
       <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale) {
      return {{upper_index, 1.0}};
   }
   if (upper_index == 0) {
      return {{0, 1.0}};
   }
   const int lower_index = upper_index - 1;
   const amrex::Real fraction = (
      coordinate - centers[lower_index]
   ) / (centers[upper_index] - centers[lower_index]);
   return {{lower_index, 1.0 - fraction}, {upper_index, fraction}};
}

std::vector<std::pair<int, amrex::Real>> equatorial_reference_weights(
   InitialDataCoordinateLine const& angular_coordinates
)
{
   const std::vector<amrex::Real>& centers = angular_coordinates.centers;
   const amrex::Real equator = 0.5 * constants::pi;
   const amrex::Real tolerance = 64.0
      * std::numeric_limits<amrex::Real>::epsilon();
   if (std::abs(angular_coordinates.faces.back() - equator) <= tolerance) {
      if (centers.size() < 2) {
         amrex::Abort(
            "equilibrium-torus equatorial extrapolation needs two theta cells"
         );
      }
      const int last = static_cast<int>(centers.size()) - 1;
      // Axisymmetry and equatorial reflection make the scalar potential even
      // about pi/2.  These weights extrapolate a quadratic in distance^2 from
      // the last two cell centers to the boundary.
      return {{last, 9.0 / 8.0}, {last - 1, -1.0 / 8.0}};
   }
   if (std::abs(angular_coordinates.faces.front() - equator) <= tolerance) {
      if (centers.size() < 2) {
         amrex::Abort(
            "equilibrium-torus equatorial extrapolation needs two theta cells"
         );
      }
      return {{0, 9.0 / 8.0}, {1, -1.0 / 8.0}};
   }
   return linear_reference_weights(centers, equator, "equator");
}

amrex::Real inner_edge_normalization(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab const& cell_spacetime,
   EquilibriumTorusParameters const& parameters
)
{
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const InitialDataCoordinateLine angular_coordinates =
      hierarchy.coordinate_line(0, 1);
   const auto radial_weights = linear_reference_weights(
      radial_coordinates.centers, parameters.inner_radius, "inner radius"
   );
   const auto angular_weights = equatorial_reference_weights(
      angular_coordinates
   );
   std::vector<ReferenceWeight> host_weights;
   for (auto const& [radial_index, radial_weight] : radial_weights) {
      for (auto const& [angular_index, angular_weight] : angular_weights) {
         host_weights.push_back({
            radial_index,
            angular_index,
            radial_weight * angular_weight,
            radial_coordinates.centers[radial_index],
            angular_coordinates.centers[angular_index],
         });
      }
   }
   amrex::Gpu::AsyncArray<ReferenceWeight> device_weights(
      host_weights.data(), host_weights.size()
   );
   ReferenceWeight const* weights = device_weights.data();
   const int num_weights = static_cast<int>(host_weights.size());
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   amrex::MultiFab reduction(
      hierarchy.grids[0], hierarchy.distribution[0], 2, 0
   );
   reduction.setVal(0.0);
   for (amrex::MFIter mfi(reduction, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto geometry = cell_spacetime.const_array(mfi);
      const auto output = reduction.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
#if (AMREX_SPACEDIM == 3)
         // The normalization is a point value, not an azimuthal reduction.
         // Select one periodic phi plane from the axisymmetric storage.
         if (k != domain_lo.z) {
            return;
         }
#endif
         for (int weight_index = 0; weight_index < num_weights;
              ++weight_index) {
            const ReferenceWeight selected = weights[weight_index];
            if (i - domain_lo.x != selected.radial_index
                || j - domain_lo.y != selected.angular_index) {
               continue;
            }
            const SpacetimeMetric metric = generated::z4c_spacetime_metric(
               load_z4c_state(geometry, i, j, k)
            );
            const EquilibriumTorusPoint point =
               equilibrium_torus_axisymmetric_spherical_point(
                  parameters,
                  1.0,
                  selected.radius,
                  selected.theta,
                  metric
               );
            const amrex::Real normalization =
               parameters.rotation_law == DiskRotationLaw::ConstantL
               ? point.minus_covariant_time_velocity
               : std::exp(-point.log_specific_enthalpy);
            output(i, j, k, 0) = point.timelike
               ? selected.weight * normalization : 0.0;
            output(i, j, k, 1) = point.timelike ? 0.0 : 1.0;
         }
      });
   }
   if (reduction.max(1, 0, false) > 0.0) {
      amrex::Abort(
         "equilibrium-torus inner edge does not admit a timelike circular orbit"
      );
   }
   const amrex::Real result = reduction.sum(0, false);
   if (!(result > 0.0) || !std::isfinite(result)) {
      amrex::Abort(
         "equilibrium-torus inner-edge normalization is invalid"
      );
   }
   return result;
}

template <int EOSMethod>
EquilibriumTorusUpdateSummary
update_axisymmetric_spherical_equilibrium_torus_primitives_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
)
{
   if (hierarchy.geometry.size() != 1 || primitive_state.size() != 1
       || cell_spacetime.size() != 1 || primitive_state[0] == nullptr
       || cell_spacetime[0] == nullptr
       || primitive_state[0]->nComp() < NVAR_HYDRO
       || cell_spacetime[0]->nComp() < Z4C_NUM_COMPONENTS) {
      amrex::Abort("invalid axisymmetric equilibrium-torus state layout");
   }
   const amrex::Real inner_normalization =
      inner_edge_normalization(
         hierarchy, *cell_spacetime[0], parameters
      );
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
   const amrex::Real dphi = hierarchy.geometry[0].CellSize(2);
#else
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
   const amrex::Real phi_lo = 0.0;
   const amrex::Real dphi = 0.0;
#endif
   amrex::MultiFab diagnostics(
      hierarchy.grids[0], hierarchy.distribution[0], 9, 0
   );
   diagnostics.setVal(0.0);
   diagnostics.setVal(
      std::numeric_limits<amrex::Real>::max(), 3, 1, 0
   );
   for (amrex::MFIter mfi(*primitive_state[0], amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto primitive = primitive_state[0]->array(mfi);
      const auto geometry = cell_spacetime[0]->const_array(mfi);
      const auto diagnostic = diagnostics.array(mfi);
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
         const bool contained_old_torus = initial_matter_exceeds_atmosphere(
            primitive(i, j, k, RHO), atmosphere.rho
         );
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(geometry, i, j, k)
         );
         const EquilibriumTorusPoint target =
            equilibrium_torus_axisymmetric_spherical_point(
               parameters, inner_normalization, radius, theta, metric
            );
         ColdBarotropeState matter;
         amrex::Real vr = 0.0;
         amrex::Real vtheta = 0.0;
         amrex::Real vphi = 0.0;
         if (target.timelike && target.log_specific_enthalpy > 0.0) {
            const amrex::Real old_log_enthalpy = contained_old_torus
               ? cold_log_specific_enthalpy(
                    primitive(i, j, k, RHO), eos
                 )
               : 0.0;
            const amrex::Real relaxed_log_enthalpy =
               (1.0 - parameters.matter_relaxation) * old_log_enthalpy
               + parameters.matter_relaxation
                  * target.log_specific_enthalpy;
            matter = cold_barotrope_from_log_enthalpy(
               relaxed_log_enthalpy, eos
            );
            vr = (1.0 - parameters.matter_relaxation)
                  * primitive(i, j, k, MOM1)
               + parameters.matter_relaxation * target.v1;
            vtheta = (1.0 - parameters.matter_relaxation)
                  * primitive(i, j, k, MOM2)
               + parameters.matter_relaxation * target.v2;
            vphi = (1.0 - parameters.matter_relaxation)
                  * primitive(i, j, k, MOM3)
               + parameters.matter_relaxation * target.v3;
         }
         if (!initial_matter_exceeds_atmosphere(matter.rho, atmosphere.rho)) {
            matter.rho = atmosphere.rho;
            matter.pressure = atmosphere.pressure;
            matter.specific_internal_energy =
               atmosphere.specific_internal_energy;
            vr = 0.0;
            vtheta = 0.0;
            vphi = 0.0;
         }
         primitive(i, j, k, RHO) = matter.rho;
         primitive(i, j, k, MOM1) = vr;
         primitive(i, j, k, MOM2) = vtheta;
         primitive(i, j, k, MOM3) = vphi;
         primitive(i, j, k, ENERGY) = matter.pressure;
         const bool contains_torus = initial_matter_exceeds_atmosphere(
            matter.rho, atmosphere.rho
         );
         if (contains_torus) {
            diagnostic(i, j, k, 0) = matter.rho;
            diagnostic(i, j, k, 1) = target.velocity_squared;
            diagnostic(i, j, k, 2) = target.log_specific_enthalpy;
            diagnostic(i, j, k, 3) = radius;
            diagnostic(i, j, k, 4) = radial_face[radial_index + 1];
            diagnostic(i, j, k, 5) = radius * std::sin(theta);
         }
         diagnostic(i, j, k, 6) = i == domain_hi.x && contains_torus
            ? matter.rho - atmosphere.rho : 0.0;
         diagnostic(i, j, k, 7) = contained_old_torus && !target.timelike
            ? 1.0 : 0.0;
      });
   }
   EquilibriumTorusUpdateSummary summary;
   summary.maximum_density = diagnostics.max(0, 0, false);
   summary.maximum_velocity = std::sqrt(std::max(
      diagnostics.max(1, 0, false), amrex::Real(0.0)
   ));
   summary.maximum_log_specific_enthalpy = diagnostics.max(2, 0, false);
   summary.inner_radius = diagnostics.min(3, 0, false);
   summary.outer_radius = diagnostics.max(4, 0, false);
   summary.boundary_density_excess = diagnostics.max(6, 0, false);
   if (diagnostics.max(7, 0, false) > 0.0) {
      amrex::Abort(
         "equilibrium-torus continuation lost a timelike orbit inside its "
         "previous matter support"
      );
   }
   if (!(summary.maximum_density > 0.0)
       || !(summary.outer_radius > summary.inner_radius)
       || !std::isfinite(summary.inner_radius)) {
      amrex::Abort(
         "equilibrium-torus parameters do not produce resolved finite support"
      );
   }
   if (summary.boundary_density_excess > 0.0) {
      amrex::Abort(
         "equilibrium-torus matter reaches the radial boundary; enlarge the "
         "ID domain"
      );
   }
   const amrex::Real density_tolerance = 128.0
      * std::numeric_limits<amrex::Real>::epsilon();
   for (amrex::MFIter mfi(diagnostics, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto diagnostic = diagnostics.array(mfi);
      const amrex::Real maximum_density = summary.maximum_density;
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         diagnostic(i, j, k, 8) =
            diagnostic(i, j, k, 0)
               >= maximum_density * (1.0 - density_tolerance)
            ? diagnostic(i, j, k, 5) : 0.0;
      });
   }
   summary.density_maximum_radius = diagnostics.max(8, 0, false);
   if (primitive_state[0]->contains_nan(0, NVAR_HYDRO, 0)) {
      amrex::Abort("axisymmetric equilibrium-torus update is nonfinite");
   }
   return summary;
}
#endif

} // namespace

EquilibriumTorusParameters equilibrium_torus_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
)
{
   EquilibriumTorusParameters parameters;
   parameters.active = cfg.initial_data_disk == "equilibrium_torus"
      || cfg.initial_data_disk == "tilted_torus";
   if (!parameters.active) {
      return parameters;
   }
   parameters.gravity = cfg.initial_data_disk_gravity == "coupled"
      ? EquilibriumTorusGravity::Coupled
      : EquilibriumTorusGravity::TestFluid;
   parameters.rotation_law = disk_rotation_law_from_name(
      cfg.initial_data_disk_rotation_law
   );
   parameters.specific_angular_momentum =
      cfg.initial_data_disk_specific_angular_momentum;
   parameters.angular_velocity = cfg.initial_data_disk_angular_velocity;
   parameters.continuation_source_angular_velocity =
      parameters.angular_velocity;
   parameters.continuation_steps =
      cfg.initial_data_rotation_continuation_steps;
   parameters.differential_rotation_length =
      cfg.initial_data_disk_differential_rotation_length;
   parameters.rotation_power_q = cfg.initial_data_disk_rotation_power_q;
   parameters.inner_radius = cfg.initial_data_disk_inner_radius;
   const amrex::Real axis_norm = std::sqrt(
      cfg.initial_data_disk_axis1 * cfg.initial_data_disk_axis1
      + cfg.initial_data_disk_axis2 * cfg.initial_data_disk_axis2
      + cfg.initial_data_disk_axis3 * cfg.initial_data_disk_axis3
   );
   parameters.axis1 = cfg.initial_data_disk_axis1 / axis_norm;
   parameters.axis2 = cfg.initial_data_disk_axis2 / axis_norm;
   parameters.axis3 = cfg.initial_data_disk_axis3 / axis_norm;
   if (provider.fixed_spacetime.metric == FixedSpacetimeMetric::KerrSchild
       || provider.fixed_spacetime.metric
          == FixedSpacetimeMetric::StaticTrumpet) {
      parameters.center1 = provider.fixed_spacetime.origin1;
      parameters.center2 = provider.fixed_spacetime.origin2;
      parameters.center3 = provider.fixed_spacetime.origin3;
   } else {
      parameters.center1 = cfg.initial_data_rotation_center1;
      parameters.center2 = cfg.initial_data_rotation_center2;
      parameters.center3 = cfg.initial_data_rotation_center3;
   }
   parameters.matter_relaxation = cfg.initial_data_matter_relaxation;
   parameters.relative_tolerance =
      cfg.initial_data_matter_relative_tolerance;
   parameters.max_iterations = cfg.initial_data_matter_max_iterations;
   parameters.atmosphere = initial_atmosphere_parameters(cfg, provider);
   return parameters;
}

EquilibriumTorusUpdateSummary update_equilibrium_torus_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, primitive_state, cell_spacetime, eos_method, eos, parameters
   );
   amrex::Abort("general equilibrium-torus construction requires 3D");
   return {};
#else
   if (!parameters.active) {
      return {};
   }
   amrex::ignore_unused(eos_method);
   if (cold_barotrope_supported(eos)) {
      return update_equilibrium_torus_primitives_method<EOSRuntime>(
         hierarchy, primitive_state, cell_spacetime, eos, parameters
      );
   }
   amrex::Abort("equilibrium-torus construction requires a cold barotropic EOS");
   return {};
#endif
}

EquilibriumTorusUpdateSummary
update_axisymmetric_spherical_equilibrium_torus_primitives(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   int eos_method,
   EOSParams const& eos,
   EquilibriumTorusParameters const& parameters
)
{
#if (AMREX_SPACEDIM < 2)
   amrex::ignore_unused(
      hierarchy, primitive_state, cell_spacetime, eos_method, eos, parameters
   );
   amrex::Abort(
      "axisymmetric equilibrium-torus construction requires 2D or 3D"
   );
   return {};
#else
   if (!parameters.active) {
      return {};
   }
   amrex::ignore_unused(eos_method);
   if (cold_barotrope_supported(eos)) {
      return update_axisymmetric_spherical_equilibrium_torus_primitives_method<
         EOSRuntime
      >(hierarchy, primitive_state, cell_spacetime, eos, parameters);
   }
   amrex::Abort("equilibrium-torus construction requires a cold barotropic EOS");
   return {};
#endif
}

} // namespace pyfluid_amrex
