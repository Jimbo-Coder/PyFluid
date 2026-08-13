#include "pyfluid_amrex/initial_data/materialization.hpp"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "pyfluid_amrex/centering_transfer.hpp"
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

#if (AMREX_SPACEDIM == 2)
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

void materialize_axisymmetric_vertex_geometry(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::MultiFab& conformal_factor,
   amrex::MultiFab& lapse_times_conformal_factor,
   amrex::MultiFab& azimuthal_shift,
   InitialDataEllipticBoundaryConditions const& scalar_boundary,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   amrex::MultiFab& spacetime_state
)
{
   const amrex::BoxArray vertex_grids = amrex::convert(
      hierarchy.grids[0], amrex::IntVect::TheNodeVector()
   );
   amrex::MultiFab vertex_psi(
      vertex_grids, hierarchy.distribution[0], 1, 0
   );
   amrex::MultiFab vertex_lapse_psi(
      vertex_grids, hierarchy.distribution[0], 1, 0
   );
   amrex::MultiFab vertex_beta(
      vertex_grids, hierarchy.distribution[0], 1, 0
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_psi},
      amrex::Vector<amrex::MultiFab const*>{&conformal_factor},
      initial_data_boundary_records(hierarchy, scalar_boundary),
      2
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_lapse_psi},
      amrex::Vector<amrex::MultiFab const*>{&lapse_times_conformal_factor},
      initial_data_boundary_records(hierarchy, scalar_boundary),
      2
   );
   materialize_initial_data_vertices(
      hierarchy,
      amrex::Vector<amrex::MultiFab*>{&vertex_beta},
      amrex::Vector<amrex::MultiFab const*>{&azimuthal_shift},
      initial_data_boundary_records(hierarchy, shift_boundary),
      2
   );
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const amrex::Dim3 domain_lo = amrex::lbound(
      amrex::surroundingNodes(hierarchy.geometry[0].Domain())
   );
   const amrex::Dim3 domain_hi = amrex::ubound(
      amrex::surroundingNodes(hierarchy.geometry[0].Domain())
   );
   const amrex::Real inverse_two_dr = 0.5 / spacing[0];
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
      const auto geometry = spacetime_state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real radius = problo[0]
            + (i - domain_lo.x) * spacing[0];
         const amrex::Real theta = problo[1]
            + (j - domain_lo.y) * spacing[1];
         const bool radial_origin = i == domain_lo.x;
         const bool angular_axis = j == domain_lo.y
            || (upper_is_axis && j == domain_hi.y);
         if (radial_origin || angular_axis) {
            beta(i, j, k, 0) = 0.0;
         }
         amrex::Real a_r_phi = 0.0;
         amrex::Real a_theta_phi = 0.0;
         if (!radial_origin && !angular_axis) {
            const amrex::Real d_beta_r = i == domain_hi.x
               ? (
                    3.0 * beta(i, j, k, 0)
                    - 4.0 * beta(i - 1, j, k, 0)
                    + beta(i - 2, j, k, 0)
                 ) / (2.0 * spacing[0])
               : (beta(i + 1, j, k, 0) - beta(i - 1, j, k, 0))
                    * inverse_two_dr;
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
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
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
         const amrex::Real radius = problo[0]
            + (i - domain_lo.x + 0.5) * spacing[0];
         const amrex::Real theta = problo[1]
            + (j - domain_lo.y + 0.5) * spacing[1];
         const amrex::Real flat_volume = constants::two_pi
            * theta_symmetry_factor * radius * radius * std::sin(theta)
            * spacing[0] * spacing[1];
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
   summary.adm_mass = integrands.sum(0, false);
   summary.angular_momentum = integrands.sum(1, false);
   summary.baryonic_mass = integrands.sum(2, false);
   summary.dimensionless_spin = summary.angular_momentum / (
      summary.adm_mass * summary.adm_mass
   );
   if (!(summary.adm_mass > 0.0) || !(summary.baryonic_mass > 0.0)
       || !std::isfinite(summary.angular_momentum)
       || !std::isfinite(summary.dimensionless_spin)) {
      amrex::Abort("axisymmetric rotating-star global diagnostic is invalid");
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
   bool seed_from_spacetime,
   ConformalContinuationObserver const& continuation_observer
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
   const bool update_rotating_matter = rotating_star.active
      && rotating_star.angular_velocity != 0.0;
   const int max_matter_iterations = update_rotating_matter
      ? rotating_star.max_iterations
      : 12;
   const amrex::Real matter_tolerance = update_rotating_matter
      ? rotating_star.relative_tolerance
      : 1.0e-9;
   amrex::Vector<amrex::Real> rotation_targets;
   if (update_rotating_matter) {
      rotation_targets = rotating_star.continuation_targets;
      if (rotation_targets.empty()) {
         rotation_targets.reserve(rotating_star.continuation_steps);
         for (int step = 1; step <= rotating_star.continuation_steps; ++step) {
            rotation_targets.push_back(
               rotating_star.angular_velocity
               * static_cast<amrex::Real>(step)
               / static_cast<amrex::Real>(rotating_star.continuation_steps)
            );
         }
      } else {
         rotation_targets.push_back(rotating_star.angular_velocity);
      }
   } else {
      rotation_targets.push_back(rotating_star.angular_velocity);
   }
   const bool coupled_xcts_matter_iteration =
      update_rotating_matter
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
        continuation_step <= static_cast<int>(rotation_targets.size());
        ++continuation_step) {
      RotatingStarParameters continued_rotating_star = rotating_star;
      continued_rotating_star.angular_velocity =
         rotation_targets[continuation_step - 1];
      result.relative_matter_update = std::numeric_limits<amrex::Real>::max();
      int step_matter_iterations = 0;
      bool strict_xcts_polish = false;
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
         if (update_rotating_matter && !preserve_spherical_seed) {
            amrex::Vector<amrex::MultiFab const*> cell_spacetime_const(num_levels);
            for (int level = 0; level < num_levels; ++level) {
               cell_spacetime_const[level] = cell_spacetime[level];
            }
            result.rotating_star = update_rotating_star_primitives(
               hierarchy,
               primitive_state,
               cell_spacetime_const,
               EOSMethod,
               eos,
               continued_rotating_star
            );
            if (amrex::ParallelDescriptor::IOProcessor()) {
               amrex::Print()
                  << "rotating continuation step=" << continuation_step
                  << "/" << rotation_targets.size()
                  << " target_omega="
                  << continued_rotating_star.angular_velocity
                  << " matter_iteration=" << matter_iteration
                  << " maximum_density=" << result.rotating_star.maximum_density
                  << " maximum_velocity=" << result.rotating_star.maximum_velocity
                  << " injection_energy=" << result.rotating_star.injection_energy
                  << " boundary_density_excess="
                  << result.rotating_star.boundary_density_excess << '\n';
            }
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
                        rmhd_primitive_at(primitive, i, j, k, eos), metric
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

         amrex::Real update_norm = 0.0;
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
            for (int component = 0;
                 component < primitive_state[level]->nComp();
                 ++component) {
               update_norm = std::max(
                  update_norm,
                  previous_primitive[level]->norm0(component, 0, false)
               );
               solution_norm = std::max(
                  solution_norm,
                  primitive_state[level]->norm0(component, 0, false)
               );
            }
            amrex::MultiFab::Subtract(
               *previous_conformal[level], *conformal_storage[level], 0, 0, 1, 0
            );
            update_norm = std::max(
               update_norm, previous_conformal[level]->norm0(0, 0, false)
            );
            solution_norm = std::max(
               solution_norm, conformal_storage[level]->norm0(0, 0, false)
            );
            amrex::MultiFab::Subtract(
               *previous_vector[level], *vector_storage[level], 0, 0, 3, 0
            );
            for (int component = 0; component < 3; ++component) {
               update_norm = std::max(
                  update_norm, previous_vector[level]->norm0(component, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, vector_storage[level]->norm0(component, 0, false)
               );
            }
            if (formulation == ConformalConstraintFormulation::xcts) {
               amrex::MultiFab::Subtract(
                  *previous_lapse[level],
                  *lapse_conformal_storage[level],
                  0,
                  0,
                  1,
                  0
               );
               update_norm = std::max(
                  update_norm, previous_lapse[level]->norm0(0, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, lapse_conformal_storage[level]->norm0(0, 0, false)
               );
            } else if (formulation == ConformalConstraintFormulation::cttk) {
               amrex::MultiFab::Subtract(
                  *previous_curvature[level], *curvature_storage[level], 0, 0, 1, 0
               );
               update_norm = std::max(
                  update_norm, previous_curvature[level]->norm0(0, 0, false)
               );
               solution_norm = std::max(
                  solution_norm, curvature_storage[level]->norm0(0, 0, false)
               );
            }
         }
         step_matter_iterations = matter_iteration;
         result.relative_matter_update = update_norm
            / std::max(amrex::Real(1.0), solution_norm);
         if (update_rotating_matter && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print()
               << "rotating continuation step=" << continuation_step
               << "/" << rotation_targets.size()
               << " target_omega="
               << continued_rotating_star.angular_velocity
               << " coupled_update_iteration=" << matter_iteration
               << " relative_update=" << result.relative_matter_update << '\n';
         }
         if (result.relative_matter_update <= matter_tolerance
             && (!coupled_xcts_matter_iteration
                 || result.constraints.converged)) {
            break;
         }
      }
      result.matter_iterations += step_matter_iterations;
      if (result.relative_matter_update > matter_tolerance
          || (coupled_xcts_matter_iteration
              && !result.constraints.converged)) {
         amrex::Abort(
            "matter-coupled conformal fixed-point iteration did not converge "
            "at rotation continuation step "
            + std::to_string(continuation_step)
         );
      }
      if (continuation_observer) {
         materialize_current_geometry();
         update_surface_diagnostic(continued_rotating_star);
         continuation_observer(
            continuation_step,
            static_cast<int>(rotation_targets.size()),
            continued_rotating_star.angular_velocity,
            result.rotating_star
         );
      }
   }

   materialize_current_geometry();
   if (update_rotating_matter && !continuation_observer) {
      update_surface_diagnostic(rotating_star);
   }
   return result;
}

#if (AMREX_SPACEDIM == 2)
template <int EOSMethod>
ConformalMaterializationResult
materialize_axisymmetric_spherical_xcts_initial_data_method(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   EOSParams const& eos,
   RotatingStarParameters const& rotating_star,
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
   amrex::MultiFab previous_primitive(
      grids, distribution, primitive_state[0]->nComp(), 0
   );

   // Every ID source, including an analytic TOV seed and a loaded portable
   // artifact, enters through the same ADM/Z4c handoff.  Extract the reduced
   // XCTS unknowns from that state instead of adding a second seed contract.
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
   auto scalar_boundary = InitialDataEllipticBoundaryConditions::constant(
      hierarchy,
      1,
      1.0,
      0.0,
      0.0,
      InitialDataEllipticCentering::cell
   );
   set_initial_data_reflection_boundaries(
      hierarchy, scalar_boundary, scalar_symmetries(1)
   );
   set_axisymmetric_spherical_radial_robin_boundary(
      hierarchy, scalar_boundary, {0.0}, 1.0
   );
   auto lapse_boundary = scalar_boundary.component(0);
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
   set_axisymmetric_spherical_radial_robin_boundary(
      hierarchy, shift_boundary, {0.0}, 2.0
   );

   XctsOptions xcts_options;
   xcts_options.max_coupled_iterations = 80;
   xcts_options.relative_update_tolerance = 1.0e-10;
   xcts_options.relaxation = 0.7;
   xcts_options.linear.relative_tolerance = 1.0e-12;
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
         scalar_boundary,
         shift_boundary,
         *spacetime_state[0]
      );
   };
   const auto fill_sources = [&]() {
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

   const bool update_rotating_matter = rotating_star.active
      && rotating_star.angular_velocity != 0.0;
   amrex::Vector<amrex::Real> rotation_targets;
   if (update_rotating_matter) {
      rotation_targets = rotating_star.continuation_targets;
      if (rotation_targets.empty()) {
         for (int step = 1; step <= rotating_star.continuation_steps; ++step) {
            rotation_targets.push_back(
               rotating_star.angular_velocity * step
               / static_cast<amrex::Real>(rotating_star.continuation_steps)
            );
         }
      } else {
         rotation_targets.push_back(rotating_star.angular_velocity);
      }
   } else {
      rotation_targets.push_back(rotating_star.angular_velocity);
   }

   ConformalMaterializationResult result;
   const int maximum_matter_iterations = update_rotating_matter
      ? rotating_star.max_iterations : 12;
   const amrex::Real matter_tolerance = update_rotating_matter
      ? rotating_star.relative_tolerance : 1.0e-9;
   for (int continuation_step = 1;
        continuation_step <= static_cast<int>(rotation_targets.size());
        ++continuation_step) {
      RotatingStarParameters continued = rotating_star;
      continued.angular_velocity = rotation_targets[continuation_step - 1];
      int step_iterations = 0;
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
         if (update_rotating_matter
             && !(continuation_step == 1 && matter_iteration == 1)) {
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
         amrex::Real update_norm = 0.0;
         amrex::Real solution_norm = 1.0;
         for (int component = 0;
              component < primitive_state[0]->nComp();
              ++component) {
            update_norm = std::max(
               update_norm, previous_primitive.norm0(component, 0, false)
            );
            solution_norm = std::max(
               solution_norm,
               primitive_state[0]->norm0(component, 0, false)
            );
         }
         update_norm = std::max({
            update_norm,
            previous_conformal.norm0(0, 0, false),
            previous_lapse.norm0(0, 0, false),
            previous_shift.norm0(0, 0, false),
         });
         solution_norm = std::max({
            solution_norm,
            conformal_factor.norm0(0, 0, false),
            lapse_times_conformal_factor.norm0(0, 0, false),
            azimuthal_shift.norm0(0, 0, false),
         });
         result.relative_matter_update = update_norm / solution_norm;
         step_iterations = matter_iteration;
         if (amrex::ParallelDescriptor::IOProcessor()
             && update_rotating_matter) {
            amrex::Print()
               << "axisymmetric rotating continuation step="
               << continuation_step << "/" << rotation_targets.size()
               << " omega=" << continued.angular_velocity
               << " matter_iteration=" << matter_iteration
               << " relative_update=" << result.relative_matter_update
               << '\n';
         }
         if (result.relative_matter_update <= matter_tolerance) {
            break;
         }
      }
      result.matter_iterations += step_iterations;
      if (result.relative_matter_update > matter_tolerance) {
         amrex::Abort(
            "axisymmetric rotating-star matter iteration did not converge"
         );
      }
      materialize_current_geometry();
      fill_sources();
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
      if (update_rotating_matter) {
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
         continuation_observer(
            continuation_step,
            static_cast<int>(rotation_targets.size()),
            continued.angular_velocity,
            result.rotating_star
         );
      }
   }
   materialize_current_geometry();
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
   ConformalConstraintFormulation formulation,
   RotatingStarParameters const& rotating_star,
   bool seed_from_spacetime,
   ConformalContinuationObserver const& continuation_observer
)
{
#if (AMREX_SPACEDIM == 2)
   amrex::ignore_unused(
      fields, seed_from_spacetime
   );
   if (system != PhysicalSystem::RHD
       || formulation != ConformalConstraintFormulation::xcts) {
      amrex::Abort(
         "reduced axisymmetric spherical conformal data currently require "
         "RHD and XCTS"
      );
   }
   if (eos_method == EOSPolytrope) {
      return materialize_axisymmetric_spherical_xcts_initial_data_method<
         EOSPolytrope
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, rotating_star, continuation_observer
      );
   }
   if (eos_method == EOSHybridPolytrope) {
      return materialize_axisymmetric_spherical_xcts_initial_data_method<
         EOSHybridPolytrope
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, rotating_star, continuation_observer
      );
   }
   amrex::Abort("reduced axisymmetric spherical XCTS requires a cold EOS");
   return {};
#elif (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, primitive_state, spacetime_state,
      cell_spacetime, system, eos_method, eos, fields, formulation,
      rotating_star, seed_from_spacetime, continuation_observer
   );
   amrex::Abort("conformal constraint initial data requires 2D or 3D");
   return {};
#else
   if (system == PhysicalSystem::RMHD) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RMHD,
         EOSIdealGas
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star,
         seed_from_spacetime, continuation_observer
      );
   }
   if (system != PhysicalSystem::RHD) {
      amrex::Abort("conformal constraint initial data requires RHD or RMHD");
   }
   if (eos_method == EOSPolytrope) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RHD,
         EOSPolytrope
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star,
         seed_from_spacetime, continuation_observer
      );
   }
   if (eos_method == EOSHybridPolytrope) {
      return materialize_conformal_initial_data_method<
         PhysicalSystem::RHD,
         EOSHybridPolytrope
      >(
         hierarchy, primitive_state, spacetime_state,
         cell_spacetime, eos, fields, formulation, rotating_star,
         seed_from_spacetime, continuation_observer
      );
   }
   return materialize_conformal_initial_data_method<
      PhysicalSystem::RHD,
      EOSIdealGas
   >(
      hierarchy, primitive_state, spacetime_state,
      cell_spacetime, eos, fields, formulation, rotating_star,
      seed_from_spacetime, continuation_observer
   );
#endif
}

} // namespace pyfluid_amrex
