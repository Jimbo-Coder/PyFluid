#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/output_metadata.hpp"

namespace pyfluid_amrex {

struct ScalarDiagnostics {
   int step = 0;
   amrex::Real time = 0.0;
   amrex::Real dt = 0.0;
   amrex::Real mass = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real angular_mom1 = 0.0;
   amrex::Real angular_mom2 = 0.0;
   amrex::Real angular_mom3 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real gravity_potential_energy = 0.0;
   amrex::Real gravity_total_energy = 0.0;
   amrex::Real gravity_total_energy_rel_drift = 0.0;
   amrex::Real passive_mass = 0.0;
   amrex::Real passive_mass_rel_drift = 0.0;
   amrex::Real wall_work = 0.0;
   amrex::Real wall_power = 0.0;
   amrex::Real energy_minus_wall_work = 0.0;
   amrex::Real wall_work_rel_error = 0.0;
   int external_budget_valid = 1;
   amrex::Real boundary_mass_change = 0.0;
   amrex::Real boundary_energy_change = 0.0;
   amrex::Real source_mass_change = 0.0;
   amrex::Real source_energy_change = 0.0;
   amrex::Real external_mass_change = 0.0;
   amrex::Real external_energy_change = 0.0;
   amrex::Real external_mass_residual = 0.0;
   amrex::Real external_energy_residual = 0.0;
   amrex::Real external_mass_rel_error = 0.0;
   amrex::Real external_energy_rel_error = 0.0;
   amrex::Real mass_rel_drift = 0.0;
   amrex::Real energy_rel_drift = 0.0;
   amrex::Real min_rho = 0.0;
   amrex::Real min_pressure = 0.0;
   amrex::Real magnetic_energy = 0.0;
   amrex::Real divb_l1 = 0.0;
   amrex::Real diffusion_residual = 0.0;
   int diffusion_iterations = 0;
   amrex::Long amr_total_cells = 0;
   int amr_active_finest_level = 0;
   int amr_finest_headroom = 0;
   int amr_touches_max_level = 0;
   amrex::Real amr_total_budget_fraction = -1.0;
   amrex::Real amr_peak_level_budget_fraction = -1.0;
   amrex::Real mcells_per_s = 0.0;
   amrex::Real layout_mcells_per_s = 0.0;
   amrex::Real sim_time_per_s = 0.0;
   amrex::Real cfl_limiter_frequency = 0.0;
   int cfl_limiter_level = -1;
   int cfl_limiter_i = 0;
   int cfl_limiter_j = 0;
   int cfl_limiter_k = 0;
   amrex::Real cfl_limiter_q1 = 0.0;
   amrex::Real cfl_limiter_q2 = 0.0;
   amrex::Real cfl_limiter_q3 = 0.0;
   amrex::Real z4c_hamiltonian_linf = 0.0;
   amrex::Real z4c_hamiltonian_l2 = 0.0;
   amrex::Real z4c_momentum_linf = 0.0;
   amrex::Real z4c_z_linf = 0.0;
   amrex::Real z4c_theta_linf = 0.0;
   amrex::Real z4c_log_conformal_determinant_linf = 0.0;
   amrex::Real z4c_trace_a_tilde_linf = 0.0;
};

struct SpacetimeConstraintDiagnostics {
   amrex::Real hamiltonian_linf = 0.0;
   amrex::Real hamiltonian_l2 = 0.0;
   amrex::Real momentum_linf = 0.0;
   amrex::Real z_linf = 0.0;
   amrex::Real theta_linf = 0.0;
   amrex::Real log_conformal_determinant_linf = 0.0;
   amrex::Real trace_a_tilde_linf = 0.0;
};

inline constexpr std::array<std::string_view, 63> ScalarDiagnosticColumns = {{
   "step",
   "time",
   "dt",
   "mass",
   "mom1",
   "mom2",
   "mom3",
   "angular_mom1",
   "angular_mom2",
   "angular_mom3",
   "energy",
   "gravity_potential_energy",
   "gravity_total_energy",
   "gravity_total_energy_rel_drift",
   "passive_mass",
   "passive_mass_rel_drift",
   "wall_work",
   "wall_power",
   "energy_minus_wall_work",
   "wall_work_rel_error",
   "external_budget_valid",
   "boundary_mass_change",
   "boundary_energy_change",
   "source_mass_change",
   "source_energy_change",
   "external_mass_change",
   "external_energy_change",
   "external_mass_residual",
   "external_energy_residual",
   "external_mass_rel_error",
   "external_energy_rel_error",
   "mass_rel_drift",
   "energy_rel_drift",
   "min_rho",
   "min_pressure",
   "magnetic_energy",
   "divb_l1",
   "diffusion_residual",
   "diffusion_iterations",
   "amr_total_cells",
   "amr_active_finest_level",
   "amr_finest_headroom",
   "amr_touches_max_level",
   "amr_total_budget_fraction",
   "amr_peak_level_budget_fraction",
   "mcells_per_s",
   "layout_mcells_per_s",
   "sim_time_per_s",
   "cfl_limiter_frequency",
   "cfl_limiter_level",
   "cfl_limiter_i",
   "cfl_limiter_j",
   "cfl_limiter_k",
   "cfl_limiter_q1",
   "cfl_limiter_q2",
   "cfl_limiter_q3",
   "z4c_hamiltonian_linf",
   "z4c_hamiltonian_l2",
   "z4c_momentum_linf",
   "z4c_z_linf",
   "z4c_theta_linf",
   "z4c_log_conformal_determinant_linf",
   "z4c_trace_a_tilde_linf",
}};

template <std::size_t N>
inline void write_tsv_header(std::ofstream& out, std::array<std::string_view, N> const& columns)
{
   for (std::size_t n = 0; n < columns.size(); ++n) {
      if (n > 0) {
         out << '\t';
      }
      out << columns[n];
   }
   out << '\n';
}

template <std::size_t ExpectedColumns, typename... Values>
inline void write_tsv_row_checked(std::ofstream& out, Values const&... values)
{
   static_assert(sizeof...(Values) == ExpectedColumns, "diagnostic row does not match header columns");
   std::size_t column = 0;
   ((out << (column++ == 0 ? "" : "\t") << values), ...);
   out << '\n';
}

struct AmrLevelDiagnostics {
   amrex::Long total_cells = 0;
   int active_finest_level = 0;
   int max_level = 0;
   int finest_headroom = 0;
   int touches_max_level = 0;
   amrex::Real total_budget_fraction = -1.0;
   amrex::Real peak_level_budget_fraction = -1.0;
};

struct CflLimiterLocation {
   amrex::Real frequency = 0.0;
   int level = -1;
   int i = 0;
   int j = 0;
   int k = 0;
   amrex::Real q1 = 0.0;
   amrex::Real q2 = 0.0;
   amrex::Real q3 = 0.0;
};

inline void set_cfl_limiter_diagnostics(ScalarDiagnostics& d, CflLimiterLocation const& limiter)
{
   d.cfl_limiter_frequency = limiter.frequency;
   d.cfl_limiter_level = limiter.level;
   d.cfl_limiter_i = limiter.i;
   d.cfl_limiter_j = limiter.j;
   d.cfl_limiter_k = limiter.k;
   d.cfl_limiter_q1 = limiter.q1;
   d.cfl_limiter_q2 = limiter.q2;
   d.cfl_limiter_q3 = limiter.q3;
}

enum ScalarReductionComponent {
   ScalarReductionMass = 0,
   ScalarReductionMom1,
   ScalarReductionMom2,
   ScalarReductionMom3,
   ScalarReductionEnergy,
   ScalarReductionMinRho,
   ScalarReductionMinPressure,
   ScalarReductionMagneticEnergy,
   ScalarReductionDivB,
   ScalarReductionAngularMom1,
   ScalarReductionAngularMom2,
   ScalarReductionAngularMom3,
   ScalarReductionGravityPotentialEnergy,
   ScalarReductionPassiveMass,
   ScalarReductionNComp,
};

static constexpr int ScalarReductionMinComponentCount =
   ScalarReductionMinPressure - ScalarReductionMinRho + 1;

inline AmrLevelDiagnostics summarize_amr_levels(
   std::vector<amrex::Long> const& level_cells,
   int max_level,
   amrex::Long total_cell_budget,
   amrex::Long level_cell_budget
)
{
   AmrLevelDiagnostics summary;
   summary.max_level = max_level;
   summary.peak_level_budget_fraction = level_cell_budget > 0 ? 0.0 : -1.0;
   for (int lev = 0; lev < static_cast<int>(level_cells.size()); ++lev) {
      const amrex::Long cells = level_cells[lev];
      summary.total_cells += cells;
      if (cells > 0) {
         summary.active_finest_level = lev;
      }
      if (level_cell_budget > 0) {
         summary.peak_level_budget_fraction = std::max(
            summary.peak_level_budget_fraction,
            static_cast<amrex::Real>(cells) / static_cast<amrex::Real>(level_cell_budget)
         );
      }
   }
   summary.finest_headroom = max_level - summary.active_finest_level;
   summary.touches_max_level = (summary.active_finest_level == max_level && max_level > 0) ? 1 : 0;
   summary.total_budget_fraction = total_cell_budget > 0
      ? static_cast<amrex::Real>(summary.total_cells) / static_cast<amrex::Real>(total_cell_budget)
      : -1.0;
   return summary;
}

template <bool MHD, int EOSMethod>
inline void fill_pressure(amrex::MultiFab& pressure, amrex::MultiFab const& state, EOSParams const& eos)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto p = pressure.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if constexpr (MHD) {
            p(i, j, k, 0) = pressure_from_conserved<EOSMethod>(load_mhd_state(s, i, j, k), eos);
         } else {
            p(i, j, k, 0) = pressure_from_state<EOSMethod>(s, i, j, k, eos);
         }
      });
   }
   }
}

inline void fill_pressure_from_primitive(
   amrex::MultiFab& pressure,
   const amrex::MultiFab& primitive
)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto output = pressure.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         output(i, j, k, 0) = cached(i, j, k, ENERGY);
      });
   }
}

template <bool MHD>
inline void fill_scalar_reduction_fields(
   amrex::MultiFab& reduction,
   amrex::MultiFab const& state,
   amrex::MultiFab const& pressure,
   amrex::iMultiFab const& valid_mask,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   FaceMagneticField const* face_b = nullptr,
   amrex::MultiFab const* volume_fraction = nullptr,
   amrex::MultiFab const* gravity_phi = nullptr,
   amrex::Real gravity_density_offset = 0.0,
   amrex::Real gravity_energy_factor = 0.0,
   int passive_component = -1
)
{
   std::unique_ptr<amrex::MultiFab> divb;
   if constexpr (MHD) {
      divb = std::make_unique<amrex::MultiFab>(state.boxArray(), state.DistributionMap(), 1, 0);
      if (face_b != nullptr) {
         fill_face_divergence(*divb, *face_b, geom, mesh_coordinate, mesh_axes);
      } else {
         fill_cell_divergence(*divb, state, geom, true);
      }
   }

   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   reduction.setVal(0.0);
   reduction.setVal(invalid_min, ScalarReductionMinRho, ScalarReductionMinComponentCount, 0);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto p = pressure.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      const bool use_volume_fraction = volume_fraction != nullptr;
      const bool use_gravity_energy = gravity_phi != nullptr && gravity_energy_factor != 0.0;
      const auto vf = use_volume_fraction ? volume_fraction->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto phi = use_gravity_energy ? gravity_phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto out = reduction.array(mfi);
      if constexpr (MHD) {
         const auto div = divb->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool valid = mask(i, j, k, 0) != 0;
            const amrex::Real cell_measure = mesh_cell_volume_for_reduction(mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            const amrex::Real weight = valid ? cell_measure * (use_volume_fraction ? vf(i, j, k, 0) : 1.0) : 0.0;
            const bool active = valid && weight > 0.0;
            const auto q = mesh_coordinate_triplet_at_location(
               mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
            );
            const StateVector3 momentum = mesh_basis_vector_to_cartesian(
               mesh_coordinate, mesh_axes, q, load_momentum_vector3(s, i, j, k)
            );
            out(i, j, k, ScalarReductionMass) = active ? weight * s(i, j, k, RHO) : 0.0;
            out(i, j, k, ScalarReductionMom1) = active ? weight * momentum.x : 0.0;
            out(i, j, k, ScalarReductionMom2) = active ? weight * momentum.y : 0.0;
            out(i, j, k, ScalarReductionMom3) = active ? weight * momentum.z : 0.0;
            out(i, j, k, ScalarReductionEnergy) = active ? weight * s(i, j, k, ENERGY) : 0.0;
            out(i, j, k, ScalarReductionMinRho) = active ? s(i, j, k, RHO) : invalid_min;
            out(i, j, k, ScalarReductionMinPressure) = active ? p(i, j, k, 0) : invalid_min;
            out(i, j, k, ScalarReductionMagneticEnergy) =
               active ? weight * magnetic_energy_density(load_magnetic_vector3(s, i, j, k)) : 0.0;
            out(i, j, k, ScalarReductionDivB) = active ? weight * div(i, j, k, 0) : 0.0;
            const auto physical = mesh_coordinate_triplet_to_cartesian(mesh_coordinate, q);
            const StateVector3 radius = {physical[0], physical[1], physical[2]};
            const StateVector3 angular_momentum = cross_product(radius, momentum);
            out(i, j, k, ScalarReductionAngularMom1) =
               active ? weight * angular_momentum.x : 0.0;
            out(i, j, k, ScalarReductionAngularMom2) =
               active ? weight * angular_momentum.y : 0.0;
            out(i, j, k, ScalarReductionAngularMom3) =
               active ? weight * angular_momentum.z : 0.0;
            out(i, j, k, ScalarReductionGravityPotentialEnergy) = active && use_gravity_energy
               ? gravity_energy_factor * weight * (s(i, j, k, RHO) - gravity_density_offset) * phi(i, j, k, 0)
               : 0.0;
            out(i, j, k, ScalarReductionPassiveMass) =
               active && passive_component >= 0 ? weight * s(i, j, k, passive_component) : 0.0;
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool valid = mask(i, j, k, 0) != 0;
            const amrex::Real cell_measure = mesh_cell_volume_for_reduction(mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            const amrex::Real weight = valid ? cell_measure * (use_volume_fraction ? vf(i, j, k, 0) : 1.0) : 0.0;
            const bool active = valid && weight > 0.0;
            const auto q = mesh_coordinate_triplet_at_location(
               mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
            );
            const StateVector3 momentum = mesh_basis_vector_to_cartesian(
               mesh_coordinate, mesh_axes, q, load_momentum_vector3(s, i, j, k)
            );
            out(i, j, k, ScalarReductionMass) = active ? weight * s(i, j, k, RHO) : 0.0;
            out(i, j, k, ScalarReductionMom1) = active ? weight * momentum.x : 0.0;
            out(i, j, k, ScalarReductionMom2) = active ? weight * momentum.y : 0.0;
            out(i, j, k, ScalarReductionMom3) = active ? weight * momentum.z : 0.0;
            out(i, j, k, ScalarReductionEnergy) = active ? weight * s(i, j, k, ENERGY) : 0.0;
            out(i, j, k, ScalarReductionMinRho) = active ? s(i, j, k, RHO) : invalid_min;
            out(i, j, k, ScalarReductionMinPressure) = active ? p(i, j, k, 0) : invalid_min;
            out(i, j, k, ScalarReductionMagneticEnergy) = 0.0;
            out(i, j, k, ScalarReductionDivB) = 0.0;
            const auto physical = mesh_coordinate_triplet_to_cartesian(mesh_coordinate, q);
            const StateVector3 radius = {physical[0], physical[1], physical[2]};
            const StateVector3 angular_momentum = cross_product(radius, momentum);
            out(i, j, k, ScalarReductionAngularMom1) =
               active ? weight * angular_momentum.x : 0.0;
            out(i, j, k, ScalarReductionAngularMom2) =
               active ? weight * angular_momentum.y : 0.0;
            out(i, j, k, ScalarReductionAngularMom3) =
               active ? weight * angular_momentum.z : 0.0;
            out(i, j, k, ScalarReductionGravityPotentialEnergy) = active && use_gravity_energy
               ? gravity_energy_factor * weight * (s(i, j, k, RHO) - gravity_density_offset) * phi(i, j, k, 0)
               : 0.0;
            out(i, j, k, ScalarReductionPassiveMass) =
               active && passive_component >= 0 ? weight * s(i, j, k, passive_component) : 0.0;
         });
      }
   }
   }
}

inline void fill_scalar_reduction_fields_mapped_hydro(
   amrex::MultiFab& reduction,
   amrex::MultiFab const& state,
   amrex::MultiFab const& pressure,
   amrex::iMultiFab const& valid_mask,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout,
   amrex::MultiFab const* gravity_phi = nullptr,
   amrex::Real gravity_density_offset = 0.0,
   amrex::Real gravity_energy_factor = 0.0,
   int passive_component = -1
)
{
   validate_patch_grid_system(grid_system);
   validate_patch_layout(layout);

   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   reduction.setVal(0.0);
   reduction.setVal(invalid_min, ScalarReductionMinRho, ScalarReductionMinComponentCount, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      PatchDescriptor const& layout_patch_ref = patch_layout_descriptor_containing_box(layout, mfi.validbox());
      PatchGridDescriptor const& grid_patch_ref = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch_ref);
      const PatchDescriptor layout_patch = layout_patch_ref;
      const PatchGridDescriptor grid_patch = grid_patch_ref;
      const PatchGridDescriptor central_patch = grid_system.patches.front();
      const auto s = state.const_array(mfi);
      const auto p = pressure.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      const bool use_gravity_energy = gravity_phi != nullptr && gravity_energy_factor != 0.0;
      const auto phi = use_gravity_energy ? gravity_phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto out = reduction.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const bool valid = mask(i, j, k, 0) != 0;
         const PatchCoordinate local = patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
         const PatchCoordinate global = patch_local_to_global(grid_patch.map, local);
         const bool physical_owner =
            layout_patch.nominal_patch_id == central_patch.patch_id
            || !patch_grid_contains_global_point(central_patch, global, 1.0e-12);
         const amrex::Real weight = (valid && physical_owner) ? patch_cell_volume(grid_patch, local) : 0.0;
         const bool active = valid && weight > 0.0;
         const StateVector3 momentum = load_momentum_vector3(s, i, j, k);
         out(i, j, k, ScalarReductionMass) = active ? weight * s(i, j, k, RHO) : 0.0;
         out(i, j, k, ScalarReductionMom1) = active ? weight * momentum.x : 0.0;
         out(i, j, k, ScalarReductionMom2) = active ? weight * momentum.y : 0.0;
         out(i, j, k, ScalarReductionMom3) = active ? weight * momentum.z : 0.0;
         out(i, j, k, ScalarReductionEnergy) = active ? weight * s(i, j, k, ENERGY) : 0.0;
         out(i, j, k, ScalarReductionMinRho) = active ? s(i, j, k, RHO) : invalid_min;
         out(i, j, k, ScalarReductionMinPressure) = active ? p(i, j, k, 0) : invalid_min;
         out(i, j, k, ScalarReductionMagneticEnergy) = 0.0;
         out(i, j, k, ScalarReductionDivB) = 0.0;
         const StateVector3 position = {global.q1, global.q2, global.q3};
         const StateVector3 angular_momentum = cross_product(position, momentum);
         out(i, j, k, ScalarReductionAngularMom1) =
            active ? weight * angular_momentum.x : 0.0;
         out(i, j, k, ScalarReductionAngularMom2) =
            active ? weight * angular_momentum.y : 0.0;
         out(i, j, k, ScalarReductionAngularMom3) =
            active ? weight * angular_momentum.z : 0.0;
         out(i, j, k, ScalarReductionGravityPotentialEnergy) = active && use_gravity_energy
            ? gravity_energy_factor * weight * (s(i, j, k, RHO) - gravity_density_offset) * phi(i, j, k, 0)
            : 0.0;
         out(i, j, k, ScalarReductionPassiveMass) =
            active && passive_component >= 0 ? weight * s(i, j, k, passive_component) : 0.0;
      });
   }
   }
}

inline std::pair<amrex::Real, amrex::Real> valid_min_rho_pressure(
   amrex::MultiFab const& state,
   amrex::MultiFab const& pressure,
   amrex::iMultiFab const& valid_mask
)
{
   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   amrex::Real rho_min = invalid_min;
   amrex::Real pressure_min = invalid_min;
   long valid_count = 0;
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto s = state.const_array(mfi);
      const auto p = pressure.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               if (mask(i, j, k, 0) == 0) {
                  continue;
               }
               rho_min = std::min(rho_min, s(i, j, k, RHO));
               pressure_min = std::min(pressure_min, p(i, j, k, 0));
               ++valid_count;
            }
         }
      }
   }
   if (valid_count == 0) {
      rho_min = invalid_min;
      pressure_min = invalid_min;
   }
   amrex::ParallelDescriptor::ReduceRealMin(rho_min);
   amrex::ParallelDescriptor::ReduceRealMin(pressure_min);
   return {rho_min, pressure_min};
}

inline void write_scalar_header(std::ofstream& out, RunConfig const& cfg, int step, amrex::Real time)
{
   append_output_metadata(out, "scalars", cfg, step, time);
   write_tsv_header(out, ScalarDiagnosticColumns);
}

inline void append_scalar_row(std::ofstream& out, ScalarDiagnostics const& d)
{
   out << std::setprecision(17);
   write_tsv_row_checked<ScalarDiagnosticColumns.size()>(
      out,
      d.step,
      d.time,
      d.dt,
      d.mass,
      d.mom1,
      d.mom2,
      d.mom3,
      d.angular_mom1,
      d.angular_mom2,
      d.angular_mom3,
      d.energy,
      d.gravity_potential_energy,
      d.gravity_total_energy,
      d.gravity_total_energy_rel_drift,
      d.passive_mass,
      d.passive_mass_rel_drift,
      d.wall_work,
      d.wall_power,
      d.energy_minus_wall_work,
      d.wall_work_rel_error,
      d.external_budget_valid,
      d.boundary_mass_change,
      d.boundary_energy_change,
      d.source_mass_change,
      d.source_energy_change,
      d.external_mass_change,
      d.external_energy_change,
      d.external_mass_residual,
      d.external_energy_residual,
      d.external_mass_rel_error,
      d.external_energy_rel_error,
      d.mass_rel_drift,
      d.energy_rel_drift,
      d.min_rho,
      d.min_pressure,
      d.magnetic_energy,
      d.divb_l1,
      d.diffusion_residual,
      d.diffusion_iterations,
      d.amr_total_cells,
      d.amr_active_finest_level,
      d.amr_finest_headroom,
      d.amr_touches_max_level,
      d.amr_total_budget_fraction,
      d.amr_peak_level_budget_fraction,
      d.mcells_per_s,
      d.layout_mcells_per_s,
      d.sim_time_per_s,
      d.cfl_limiter_frequency,
      d.cfl_limiter_level,
      d.cfl_limiter_i,
      d.cfl_limiter_j,
      d.cfl_limiter_k,
      d.cfl_limiter_q1,
      d.cfl_limiter_q2,
      d.cfl_limiter_q3,
      d.z4c_hamiltonian_linf,
      d.z4c_hamiltonian_l2,
      d.z4c_momentum_linf,
      d.z4c_z_linf,
      d.z4c_theta_linf,
      d.z4c_log_conformal_determinant_linf,
      d.z4c_trace_a_tilde_linf
   );
}

inline void write_amr_level_header(
   std::ofstream& out,
   int max_level,
   RunConfig const& cfg,
   int step,
   amrex::Real time
)
{
   append_output_metadata(out, "amr_levels", cfg, step, time, {}, max_level);
   out << "step\ttime\ttotal_cells\tactive_finest_level\tmax_level\tfinest_headroom\ttouches_max_level"
       << "\tfinest_dx";
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      out << "\tfinest_dx" << (dir + 1);
   }
   out << "\tmin_feature_length";
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      out << "\tmin_feature_length" << (dir + 1);
   }
   out << "\tmin_refine_radius";
   out << "\tfinest_dx_over_min_feature_length";
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      out << "\tfinest_dx" << (dir + 1) << "_over_min_feature_length" << (dir + 1);
   }
   out << "\tresolution_limited_by_max_level"
       << "\tsolid_curvature_error\tsolid_curvature_tolerance\tsolid_curvature_resolved"
       << "\ttotal_budget_fraction\tpeak_level_budget_fraction";
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\tcells_L" << lev;
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\tsteps_L" << lev;
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\tboxes_L" << lev;
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\tmax_box_aspect_L" << lev;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         out << "\tref_ratio" << (dir + 1) << "_L" << lev;
      }
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\tlatest_auto_tags_L" << lev;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         out << "\tlatest_auto_dir" << (dir + 1) << "_tags_L" << lev;
      }
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         out << "\tlatest_auto_dir" << (dir + 1) << "_mean_over_threshold_L" << lev;
      }
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << "\trefinement_decision_L" << lev;
   }
   out << '\n';
}

inline void append_amr_level_row(
   std::ofstream& out,
   int step,
   amrex::Real time,
   std::vector<amrex::Long> const& level_cells,
   amrex::Vector<int> const& level_steps,
   int max_level,
   amrex::Long total_cell_budget,
   amrex::Long level_cell_budget,
   amrex::Real finest_dx,
   std::array<amrex::Real, AMREX_SPACEDIM> const& finest_dx_by_dir,
   amrex::Real min_feature_length,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& min_feature_lengths,
   amrex::Real min_refine_radius,
   amrex::Real solid_curvature_error,
   amrex::Real solid_curvature_tolerance,
   std::vector<int> const& level_box_counts,
   std::vector<amrex::Real> const& level_max_box_aspects,
   std::array<std::vector<int>, AMREX_SPACEDIM> const& level_ref_ratios,
   std::vector<amrex::Long> const& auto_tagged_cells,
   std::array<std::vector<amrex::Long>, AMREX_SPACEDIM> const& auto_direction_tagged_cells,
   std::array<std::vector<amrex::Real>, AMREX_SPACEDIM> const& auto_direction_strength,
   std::vector<std::string> const& refinement_decisions
)
{
   const AmrLevelDiagnostics summary = summarize_amr_levels(
      level_cells,
      max_level,
      total_cell_budget,
      level_cell_budget
   );
   const amrex::Real finest_dx_over_min_feature_length =
      min_feature_length > 0.0 ? finest_dx / min_feature_length : -1.0;
   std::array<amrex::Real, AMREX_SPACEDIM> finest_dx_over_min_feature_length_by_dir{};
   bool min_feature_enabled = false;
   bool min_feature_resolved = true;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::Real directional_min_length = min_feature_lengths[dir];
      if (directional_min_length > 0.0) {
         min_feature_enabled = true;
         finest_dx_over_min_feature_length_by_dir[dir] = finest_dx_by_dir[dir] / directional_min_length;
         min_feature_resolved =
            min_feature_resolved && finest_dx_by_dir[dir] <= directional_min_length;
      } else {
         finest_dx_over_min_feature_length_by_dir[dir] = -1.0;
      }
   }
   const bool curvature_enabled = solid_curvature_tolerance > 0.0;
   const bool curvature_resolved = !curvature_enabled || solid_curvature_error <= solid_curvature_tolerance;
   const bool has_resolution_cutoff = min_feature_enabled || curvature_enabled;
   const int solid_curvature_resolved = curvature_enabled ? (curvature_resolved ? 1 : 0) : -1;
   const int resolution_limited_by_max_level = summary.touches_max_level != 0
      && (!has_resolution_cutoff || !min_feature_resolved || !curvature_resolved) ? 1 : 0;
   out << step << '\t'
       << std::setprecision(17) << time << '\t'
       << summary.total_cells << '\t'
       << summary.active_finest_level << '\t'
       << max_level << '\t'
       << summary.finest_headroom << '\t'
       << summary.touches_max_level << '\t'
       << finest_dx;
   for (amrex::Real directional_dx : finest_dx_by_dir) {
      out << '\t' << directional_dx;
   }
   out << '\t'
       << min_feature_length;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      out << '\t' << min_feature_lengths[dir];
   }
   out << '\t' << min_refine_radius;
   out << '\t'
       << finest_dx_over_min_feature_length;
   for (amrex::Real directional_ratio : finest_dx_over_min_feature_length_by_dir) {
      out << '\t' << directional_ratio;
   }
   out << '\t'
       << resolution_limited_by_max_level << '\t'
       << solid_curvature_error << '\t'
       << solid_curvature_tolerance << '\t'
       << solid_curvature_resolved << '\t'
       << summary.total_budget_fraction << '\t'
       << summary.peak_level_budget_fraction;
   for (amrex::Long cells : level_cells) {
      out << '\t' << cells;
   }
   for (int steps : level_steps) {
      out << '\t' << steps;
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      const int boxes = lev < static_cast<int>(level_box_counts.size()) ? level_box_counts[lev] : 0;
      out << '\t' << boxes;
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      const amrex::Real aspect =
         lev < static_cast<int>(level_max_box_aspects.size()) ? level_max_box_aspects[lev] : 0.0;
      out << '\t' << aspect;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         const auto& dir_ratios = level_ref_ratios[dir];
         const int ratio = lev < static_cast<int>(dir_ratios.size()) ? dir_ratios[lev] : 1;
         out << '\t' << ratio;
      }
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      const amrex::Long tags =
         lev < static_cast<int>(auto_tagged_cells.size()) ? auto_tagged_cells[lev] : 0;
      out << '\t' << tags;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         const auto& dir_tags = auto_direction_tagged_cells[dir];
         const amrex::Long tags = lev < static_cast<int>(dir_tags.size()) ? dir_tags[lev] : 0;
         out << '\t' << tags;
      }
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      for (int lev = 0; lev <= max_level; ++lev) {
         const auto& dir_tags = auto_direction_tagged_cells[dir];
         const auto& dir_strength = auto_direction_strength[dir];
         const amrex::Long tags = lev < static_cast<int>(dir_tags.size()) ? dir_tags[lev] : 0;
         const amrex::Real strength = lev < static_cast<int>(dir_strength.size()) ? dir_strength[lev] : 0.0;
         out << '\t' << (tags > 0 ? strength / static_cast<amrex::Real>(tags) : 0.0);
      }
   }
   for (int lev = 0; lev <= max_level; ++lev) {
      out << '\t' << (lev < static_cast<int>(refinement_decisions.size())
         ? refinement_decisions[lev]
         : "inactive");
   }
   out << '\n';
}

} // namespace pyfluid_amrex
