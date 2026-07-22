#include "pyfluid_amrex/initial_data/rotating_star.hpp"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

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
         hierarchy.grids[level], hierarchy.distribution[level], 2, 0
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
            const RotatingStarPoint point = uniform_rotation_point(
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
            const RotatingStarPoint target = uniform_rotation_point(
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
   if (parameters.rotation_law != RotationLaw::Uniform) {
      amrex::Abort("selected rotation law has no Euler-integral implementation");
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
