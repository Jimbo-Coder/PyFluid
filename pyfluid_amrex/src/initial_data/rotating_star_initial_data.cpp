#include "pyfluid_amrex/initial_data/rotating_star.hpp"

#include <AMReX.H>
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
            matter_support(i, j, k) =
               primitive(i, j, k, RHO) > atmosphere.rho ? 1.0 : 0.0;
         });
      }
      effective_potential_storage[level]->FillBoundary(
         hierarchy.geometry[level].periodicity()
      );
      matter_support_storage[level]->FillBoundary(
         hierarchy.geometry[level].periodicity()
      );
   }
   std::vector<amrex::GpuArray<amrex::Real, 3>> center_point{{{
      parameters.rotation_center.q1,
      parameters.rotation_center.q2,
      parameters.rotation_center.q3,
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
         hierarchy.grids[level], hierarchy.distribution[level], 6, 0
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
            if (!(matter.rho > atmosphere.rho)) {
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
            diagnostic(i, j, k, 0) = matter.rho > atmosphere.rho
               ? target.velocity_squared
               : 0.0;
            diagnostic(i, j, k, 1) = matter.rho;
            const bool boundary = i == ilo || i == ihi || j == jlo || j == jhi
               || k == klo || k == khi;
            diagnostic(i, j, k, 2) = boundary
               ? amrex::max(matter.rho - atmosphere.rho, amrex::Real(0.0))
               : 0.0;
            diagnostic(i, j, k, 3) =
               within_continuation_support && !target.timelike ? 1.0 : 0.0;
            if (matter.rho > atmosphere.rho) {
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

} // namespace

amrex::Real rotating_star_mass_shedding_indicator(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& cell_spacetime,
   RotatingStarParameters const& parameters
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(hierarchy, cell_spacetime, parameters);
   amrex::Abort("rotating-star surface diagnostics require the 3D executable");
   return 1.0;
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
      log_enthalpy[level]->FillBoundary(
         hierarchy.geometry[level].periodicity()
      );
   }

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
            extent = std::min(
               extent,
               (upper[direction] - 1.6 * coarse_dx[direction]
                - center[direction]) / ray_component
            );
         } else if (ray_component < 0.0) {
            extent = std::min(
               extent,
               (lower[direction] + 0.6 * coarse_dx[direction]
                - center[direction]) / ray_component
            );
         }
      }
      counts[ray] = static_cast<int>(std::floor(0.98 * extent / spacing)) + 1;
      if (counts[ray] < 6) {
         amrex::Abort("rotating-star surface diagnostic domain is too small");
      }
      for (int sample = 0; sample < counts[ray]; ++sample) {
         const amrex::Real radius = sample * spacing;
         points.push_back({
            center[0] + radius * ray_directions[ray][0],
            center[1] + radius * ray_directions[ray][1],
            center[2] + radius * ray_directions[ray][2],
         });
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
   return indicator;
#endif
}

RotatingStarParameters rotating_star_parameters(
   RunConfig const& cfg,
   RelativisticInitialDataView const& provider
)
{
   RotatingStarParameters parameters;
   parameters.active = cfg.ic == "rotating_star";
   if (!parameters.active) {
      return parameters;
   }
   parameters.rotation_law = rotation_law_from_name(
      cfg.initial_data_rotation_law
   );
   parameters.angular_velocity = cfg.initial_data_angular_velocity;
   parameters.differential_rotation_length =
      cfg.initial_data_differential_rotation_length;
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
   if (!parameters.active || parameters.angular_velocity == 0.0) {
      return {};
   }
   if (eos_method == EOSPolytrope) {
      return update_rotating_star_primitives_method<EOSPolytrope>(
         hierarchy, primitive_state, cell_spacetime, eos, parameters
      );
   }
   if (eos_method == EOSHybridPolytrope) {
      return update_rotating_star_primitives_method<EOSHybridPolytrope>(
         hierarchy, primitive_state, cell_spacetime, eos, parameters
      );
   }
   amrex::Abort("rotating-star equilibrium requires a cold barotropic EOS");
   return {};
#endif
}

} // namespace pyfluid_amrex
