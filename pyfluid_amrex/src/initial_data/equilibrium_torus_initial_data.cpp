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

#if (AMREX_SPACEDIM == 2)
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

amrex::Real inner_edge_minus_covariant_time_velocity(
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
            output(i, j, k, 0) = point.timelike
               ? selected.weight * point.minus_covariant_time_velocity : 0.0;
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
         "equilibrium-torus inner-edge covariant time velocity is invalid"
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
   const amrex::Real inner_minus_u_t =
      inner_edge_minus_covariant_time_velocity(
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
   const MeshAxes axes = mesh_axes_layout(MeshSpherical, "r theta");
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
         const PatchCoordinate mesh{radius, theta, 0.0};
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
               parameters, inner_minus_u_t, radius, theta, metric
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
         "constant-l parameters do not produce a resolved finite torus"
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
   parameters.active = cfg.initial_data_disk == "equilibrium_torus";
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
   parameters.inner_radius = cfg.initial_data_disk_inner_radius;
   parameters.matter_relaxation = cfg.initial_data_matter_relaxation;
   parameters.atmosphere = initial_atmosphere_parameters(cfg, provider);
   return parameters;
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
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(
      hierarchy, primitive_state, cell_spacetime, eos_method, eos, parameters
   );
   amrex::Abort("axisymmetric equilibrium-torus construction requires 2D");
   return {};
#else
   if (!parameters.active) {
      return {};
   }
   if (eos_method == EOSPolytrope) {
      return update_axisymmetric_spherical_equilibrium_torus_primitives_method<
         EOSPolytrope
      >(hierarchy, primitive_state, cell_spacetime, eos, parameters);
   }
   if (eos_method == EOSHybridPolytrope) {
      return update_axisymmetric_spherical_equilibrium_torus_primitives_method<
         EOSHybridPolytrope
      >(hierarchy, primitive_state, cell_spacetime, eos, parameters);
   }
   amrex::Abort("equilibrium-torus construction requires a cold barotropic EOS");
   return {};
#endif
}

} // namespace pyfluid_amrex
