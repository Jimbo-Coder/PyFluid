#include "pyfluid_amrex/metric_reductions.hpp"
#include "amr/amr_solver_base.hpp"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuLaunch.H>
#ifdef AMREX_USE_GPU
#include <AMReX_Reduce.H>
#endif
#include "pyfluid_amrex/diagnostic_reductions.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/apparent_horizon_tracker.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/diagnostics_types.hpp"
#include "pyfluid_amrex/generated/diagnostic_catalog.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"
#include "pyfluid_amrex/kernels_generated/diagnostics.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/surface_geometry.hpp"
#include "pyfluid_amrex/output_metadata.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

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
   ScalarReductionMagneticEnergy,
   ScalarReductionDivB,
   ScalarReductionDivBSquared,
   ScalarReductionDivBAbs,
   ScalarReductionAngularMom1,
   ScalarReductionAngularMom2,
   ScalarReductionAngularMom3,
   ScalarReductionGravityPotentialEnergy,
   ScalarReductionPassiveMass,
   ScalarReductionNComp,
};

template <typename Array>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void store_conserved_diagnostic_integrands(
   Array const& output,
   int i,
   int j,
   int k,
   amrex::Real weight,
   bool active,
   ConservedDiagnosticIntegrands const& integrands
) noexcept
{
   const amrex::Real weighted = active ? weight : 0.0_rt;
   output(i, j, k, ScalarReductionMass) = weighted * integrands.mass;
   output(i, j, k, ScalarReductionMom1) = weighted * integrands.momentum1;
   output(i, j, k, ScalarReductionMom2) = weighted * integrands.momentum2;
   output(i, j, k, ScalarReductionMom3) = weighted * integrands.momentum3;
   output(i, j, k, ScalarReductionEnergy) = weighted * integrands.energy;
   output(i, j, k, ScalarReductionMagneticEnergy) = weighted * integrands.magnetic_energy;
   output(i, j, k, ScalarReductionDivB) = weighted * integrands.divb;
   output(i, j, k, ScalarReductionDivBSquared) =
      weighted * integrands.divb * integrands.divb;
   output(i, j, k, ScalarReductionDivBAbs) =
      active ? amrex::Math::abs(integrands.divb) : 0.0_rt;
   output(i, j, k, ScalarReductionAngularMom1) = weighted * integrands.angular_momentum1;
   output(i, j, k, ScalarReductionAngularMom2) = weighted * integrands.angular_momentum2;
   output(i, j, k, ScalarReductionAngularMom3) = weighted * integrands.angular_momentum3;
   output(i, j, k, ScalarReductionGravityPotentialEnergy) =
      weighted * integrands.gravity_potential_energy;
   output(i, j, k, ScalarReductionPassiveMass) = weighted * integrands.passive_mass;
}

enum RelativisticDiagnosticReductionComponent : int {
   RelativisticReductionElectromagneticEnergy = 0,
   RelativisticReductionMaxBSquared,
   RelativisticReductionMaxEnthalpyMagnetization,
   RelativisticReductionMinPlasmaBeta,
   RelativisticReductionMaxLorentzFactor,
   RelativisticReductionMinLapse,
   RelativisticReductionAtmosphereVolume,
   RelativisticReductionTotalVolume,
   RelativisticReductionNComp,
};

template <bool MHD>
inline void fill_relativistic_diagnostic_fields(
   amrex::MultiFab& reduction,
   amrex::MultiFab const& primitive,
   amrex::MultiFab const* cell_spacetime,
   amrex::iMultiFab const& valid_mask,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   EOSParams const& eos,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::Real time,
   bool analytic_fixed_spacetime
)
{
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      analytic_fixed_spacetime || cell_spacetime != nullptr,
      "relativistic diagnostics require analytic or stored spacetime"
   );
   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   reduction.setVal(0.0_rt);
   reduction.setVal(invalid_min, RelativisticReductionMinPlasmaBeta, 1, 0);
   reduction.setVal(invalid_min, RelativisticReductionMinLapse, 1, 0);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime != nullptr
         ? cell_spacetime->const_array(mfi)
         : amrex::Array4<amrex::Real const>{};
      const auto mask = valid_mask.const_array(mfi);
      const auto output = reduction.array(mfi);
      amrex::ParallelFor(box, [mask, analytic_fixed_spacetime, fixed_spacetime, mesh_coordinate,
          mesh_axes, problo, dx, time, spacetime, cached, eos, output, invalid_min] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (mask(i, j, k, 0) == 0) {
            return;
         }
         const SpacetimeMetric metric = analytic_fixed_spacetime
            ? reference_metric_fixed_spacetime_geometry_at_cell(
                 fixed_spacetime, mesh_coordinate, mesh_axes,
                 problo, dx, i, j, k, time
              ).metric
            : generated::z4c_spacetime_metric(
                 load_z4c_state(spacetime, i, j, k)
              );
         const amrex::Real coordinate_volume = mesh_cell_volume_for_reduction(
            mesh_coordinate, mesh_axes, problo, dx, i, j, k
         );
         if constexpr (MHD) {
            const RMHDDiagnosticIntegrands integrands =
               generated::rmhd_diagnostic_integrands(
                  rmhd_primitive_at(cached, i, j, k, eos), metric
               );
            output(i, j, k, RelativisticReductionElectromagneticEnergy) =
               coordinate_volume * integrands.electromagnetic_energy;
            output(i, j, k, RelativisticReductionMaxBSquared) = integrands.b_squared;
            output(i, j, k, RelativisticReductionMaxEnthalpyMagnetization) =
               integrands.enthalpy_magnetization;
            output(i, j, k, RelativisticReductionMinPlasmaBeta) =
               integrands.b_squared > 0.0_rt ? integrands.plasma_beta : invalid_min;
            output(i, j, k, RelativisticReductionMaxLorentzFactor) =
               integrands.lorentz_factor;
         } else {
            const RHDPrimitive local{
               cached(i, j, k, RHO),
               0.0_rt,
               cached(i, j, k, ENERGY),
               cached(i, j, k, MOM1),
               cached(i, j, k, MOM2),
               cached(i, j, k, MOM3),
            };
            output(i, j, k, RelativisticReductionMaxLorentzFactor) =
               generated::rhd_diagnostic_integrands(local, metric).lorentz_factor;
         }
         const bool atmosphere = primitive_is_atmosphere(
            cached(i, j, k, RHO), cached(i, j, k, ENERGY), eos
         );
         output(i, j, k, RelativisticReductionAtmosphereVolume) =
            atmosphere ? coordinate_volume : 0.0_rt;
         output(i, j, k, RelativisticReductionTotalVolume) = coordinate_volume;
         output(i, j, k, RelativisticReductionMinLapse) = metric.lapse;
      });
   }
}

AmrLevelDiagnostics summarize_amr_levels(
   std::vector<amrex::Long> const& level_cells,
   int max_level,
   amrex::Long total_cell_budget,
   amrex::Long level_cell_budget
)
{
   AmrLevelDiagnostics summary;
   summary.max_level = max_level;
   summary.peak_level_budget_fraction = level_cell_budget > 0 ? 0.0_rt : -1.0_rt;
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
      : -1.0_rt;
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
      amrex::ParallelFor(bx, [p, s, eos] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
   amrex::iMultiFab const& valid_mask,
   amrex::Geometry const& geom,
   int mesh_coordinate,
   MeshAxes const& mesh_axes,
   FaceMagneticField const* face_b = nullptr,
   amrex::MultiFab const* volume_fraction = nullptr,
   amrex::MultiFab const* gravity_phi = nullptr,
   amrex::Real gravity_density_offset = 0.0_rt,
   amrex::Real gravity_energy_factor = 0.0_rt,
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

   reduction.setVal(0.0_rt);
   const auto problo = geom.ProbLoArray();
   const auto dx = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto s = state.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      const bool use_volume_fraction = volume_fraction != nullptr;
      const bool use_gravity_energy = gravity_phi != nullptr && gravity_energy_factor != 0.0_rt;
      const auto vf = use_volume_fraction ? volume_fraction->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto phi = use_gravity_energy ? gravity_phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto out = reduction.array(mfi);
      if constexpr (MHD) {
         const auto div = divb->const_array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool valid = mask(i, j, k, 0) != 0;
            const amrex::Real cell_measure = mesh_cell_volume_for_reduction(mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            const amrex::Real weight = valid ? cell_measure * (use_volume_fraction ? vf(i, j, k, 0) : 1.0_rt) : 0.0_rt;
            const bool active = valid && weight > 0.0_rt;
            const auto q = mesh_coordinate_triplet_at_location(
               mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
            );
            const StateVector3 momentum = mesh_basis_vector_to_cartesian(
               mesh_coordinate, mesh_axes, q, load_momentum_vector3(s, i, j, k)
            );
            const auto physical = mesh_coordinate_triplet_to_cartesian(mesh_coordinate, q);
            const StateVector3 magnetic = load_magnetic_vector3(s, i, j, k);
            const ConservedDiagnosticIntegrands integrands = generated::conserved_diagnostic_integrands(
               s(i, j, k, RHO), momentum.x, momentum.y, momentum.z, s(i, j, k, ENERGY),
               magnetic.x, magnetic.y, magnetic.z, physical[0], physical[1], physical[2],
               use_gravity_energy ? phi(i, j, k, 0) : 0.0_rt, gravity_density_offset,
               gravity_energy_factor,
               passive_component >= 0 ? s(i, j, k, passive_component) : 0.0_rt,
               div(i, j, k, 0)
            );
            store_conserved_diagnostic_integrands(out, i, j, k, weight, active, integrands);
         });
      } else {
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool valid = mask(i, j, k, 0) != 0;
            const amrex::Real cell_measure = mesh_cell_volume_for_reduction(mesh_coordinate, mesh_axes, problo, dx, i, j, k);
            const amrex::Real weight = valid ? cell_measure * (use_volume_fraction ? vf(i, j, k, 0) : 1.0_rt) : 0.0_rt;
            const bool active = valid && weight > 0.0_rt;
            const auto q = mesh_coordinate_triplet_at_location(
               mesh_coordinate, mesh_axes, problo, dx, i, j, k, false, false, false
            );
            const StateVector3 momentum = mesh_basis_vector_to_cartesian(
               mesh_coordinate, mesh_axes, q, load_momentum_vector3(s, i, j, k)
            );
            const auto physical = mesh_coordinate_triplet_to_cartesian(mesh_coordinate, q);
            const ConservedDiagnosticIntegrands integrands = generated::conserved_diagnostic_integrands(
               s(i, j, k, RHO), momentum.x, momentum.y, momentum.z, s(i, j, k, ENERGY),
               0.0_rt, 0.0_rt, 0.0_rt, physical[0], physical[1], physical[2],
               use_gravity_energy ? phi(i, j, k, 0) : 0.0_rt, gravity_density_offset,
               gravity_energy_factor,
               passive_component >= 0 ? s(i, j, k, passive_component) : 0.0_rt, 0.0_rt
            );
            store_conserved_diagnostic_integrands(out, i, j, k, weight, active, integrands);
         });
      }
   }
   }
}

inline void fill_scalar_reduction_fields_mapped_hydro(
   amrex::MultiFab& reduction,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout,
   amrex::MultiFab const* gravity_phi = nullptr,
   amrex::Real gravity_density_offset = 0.0_rt,
   amrex::Real gravity_energy_factor = 0.0_rt,
   int passive_component = -1
)
{
   validate_patch_grid_system(grid_system);
   validate_patch_layout(layout);

   reduction.setVal(0.0_rt);
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
      const auto mask = valid_mask.const_array(mfi);
      const bool use_gravity_energy = gravity_phi != nullptr && gravity_energy_factor != 0.0_rt;
      const auto phi = use_gravity_energy ? gravity_phi->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto out = reduction.array(mfi);
         amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const bool valid = mask(i, j, k, 0) != 0;
         const PatchCoordinate local = patch_layout_cell_center_coordinate(grid_patch, layout_patch, i, j, k);
         const PatchCoordinate global = patch_local_to_global(grid_patch.map, local);
         const bool physical_owner =
            layout_patch.nominal_patch_id == central_patch.patch_id
            || !patch_grid_contains_global_point(central_patch, global, 1.0e-12_rt);
         const amrex::Real weight = (valid && physical_owner) ? patch_cell_volume(grid_patch, local) : 0.0_rt;
         const bool active = valid && weight > 0.0_rt;
         const StateVector3 momentum = load_momentum_vector3(s, i, j, k);
         const ConservedDiagnosticIntegrands integrands = generated::conserved_diagnostic_integrands(
            s(i, j, k, RHO), momentum.x, momentum.y, momentum.z, s(i, j, k, ENERGY),
            0.0_rt, 0.0_rt, 0.0_rt, global.q1, global.q2, global.q3,
            use_gravity_energy ? phi(i, j, k, 0) : 0.0_rt, gravity_density_offset,
            gravity_energy_factor,
            passive_component >= 0 ? s(i, j, k, passive_component) : 0.0_rt, 0.0_rt
         );
         store_conserved_diagnostic_integrands(out, i, j, k, weight, active, integrands);
      });
   }
   }
}

struct DensityPressureExtrema {
   amrex::Real min_rho = 0.0_rt;
   amrex::Real max_rho = 0.0_rt;
   amrex::Real min_pressure = 0.0_rt;
};

inline DensityPressureExtrema valid_density_pressure_extrema(
   amrex::MultiFab const& density,
   amrex::MultiFab const& pressure,
   amrex::iMultiFab const& valid_mask
)
{
   const amrex::Real invalid_min = std::numeric_limits<amrex::Real>::max();
   amrex::Real rho_min = invalid_min;
   amrex::Real rho_max = -invalid_min;
   amrex::Real pressure_min = invalid_min;
#ifdef AMREX_USE_GPU
   using ExtremaReducer = amrex::Reducer<
      amrex::TypeList<
         amrex::ReduceOpMin,
         amrex::ReduceOpMax,
         amrex::ReduceOpMin
      >,
      amrex::TypeList<amrex::Real, amrex::Real, amrex::Real>
   >;
   ExtremaReducer reducer;
   const auto density_values = density.const_arrays();
   const auto pressure_values = pressure.const_arrays();
   const auto masks = valid_mask.const_arrays();
   reducer.eval(
      density,
      amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
         -> ExtremaReducer::Result_t {
         if (masks[box_no](i, j, k, 0) == 0) {
            return {invalid_min, -invalid_min, invalid_min};
         }
         return {
            density_values[box_no](i, j, k, RHO),
            density_values[box_no](i, j, k, RHO),
            pressure_values[box_no](i, j, k, 0),
         };
      }
   );
   const auto extrema = amrex::tupleToArray(reducer.getResult());
   rho_min = extrema[0];
   rho_max = extrema[1];
   pressure_min = extrema[2];
#else
   for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.validbox();
      const amrex::Dim3 lo = amrex::lbound(box);
      const amrex::Dim3 hi = amrex::ubound(box);
      const auto rho = density.const_array(mfi);
      const auto p = pressure.const_array(mfi);
      const auto mask = valid_mask.const_array(mfi);
      for (int k = lo.z; k <= hi.z; ++k) {
         for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               if (mask(i, j, k, 0) == 0) {
                  continue;
               }
               rho_min = std::min(rho_min, rho(i, j, k, RHO));
               rho_max = std::max(rho_max, rho(i, j, k, RHO));
               pressure_min = std::min(pressure_min, p(i, j, k, 0));
            }
         }
      }
   }
#endif
   amrex::Vector<std::reference_wrapper<amrex::Real>> minimum_values{
      rho_min,
      pressure_min,
   };
   amrex::ParallelDescriptor::ReduceRealMin(minimum_values);
   amrex::ParallelDescriptor::ReduceRealMax(rho_max);
   return {rho_min, rho_max, pressure_min};
}

bool ScalarDiagnosticSelection::contains_any(ScalarDiagnosticGroup group) const noexcept
{
   auto const& members = ScalarDiagnosticGroups[static_cast<std::size_t>(group)].members;
   for (std::size_t index = 0; index < enabled.size(); ++index) {
      if (enabled[index] && members[index]) {
         return true;
      }
   }
   return false;
}

void ScalarDiagnosticSelection::enable_group(ScalarDiagnosticGroup group) noexcept
{
   auto const& members = ScalarDiagnosticGroups[static_cast<std::size_t>(group)].members;
   for (std::size_t index = 0; index < enabled.size(); ++index) {
      enabled[index] = enabled[index] || members[index];
   }
}

std::size_t ScalarDiagnosticSelection::size() const noexcept
{
   return static_cast<std::size_t>(std::count(enabled.begin(), enabled.end(), true));
}

ScalarDiagnosticSelection scalar_diagnostic_selection(std::string const& selectors)
{
   ScalarDiagnosticSelection selection;
   std::stringstream stream(selectors);
   std::string token;
   while (std::getline(stream, token, ',')) {
      token = trim_copy(token);
      if (token.empty() || token == "none") {
         continue;
      }
      if (token == "all") {
         selection.enabled.fill(true);
         continue;
      }
      bool found = false;
      for (ScalarDiagnosticGroupInfo const& group : ScalarDiagnosticGroups) {
         if (token != group.name) {
            continue;
         }
         for (std::size_t index = 0; index < selection.enabled.size(); ++index) {
            selection.enabled[index] = selection.enabled[index] || group.members[index];
         }
         found = true;
         break;
      }
      if (found) {
         continue;
      }
      for (ScalarDiagnosticInfo const& diagnostic : ScalarDiagnosticCatalog) {
         if (token == diagnostic.name) {
            selection.enabled[static_cast<std::size_t>(diagnostic.id)] = true;
            found = true;
            break;
         }
      }
      if (!found) {
         for (ScalarDiagnosticInfo const& diagnostic : ScalarDiagnosticCatalog) {
            if (token != diagnostic.family) {
               continue;
            }
            selection.enabled[static_cast<std::size_t>(diagnostic.id)] = true;
            found = true;
         }
      }
      if (!found) {
         amrex::Abort("unknown pyfluid.scalar_diagnostics selector: " + token);
      }
   }
   if (selection.size() == 0) {
      amrex::Abort("pyfluid.scalar_diagnostics must select at least one diagnostic");
   }
   selection.enabled[static_cast<std::size_t>(ScalarDiagnosticId::Step)] = true;
   selection.enabled[static_cast<std::size_t>(ScalarDiagnosticId::Time)] = true;
   selection.enabled[static_cast<std::size_t>(ScalarDiagnosticId::Dt)] = true;
   return selection;
}

void apply_scalar_diagnostic_requirements(
   ScalarDiagnosticSelection& selection,
   RunConfig const& config
)
{
   const PhysicalSystem system = physical_system(config);
   const auto requirement_is_satisfied = [&](std::string_view requirement) {
      if (requirement.empty()) {
         return true;
      }
      if (requirement == "mhd") {
         return physical_system_has_magnetic_field(system);
      }
      if (requirement == "newtonian_mhd") {
         return system == PhysicalSystem::MHD;
      }
      if (requirement == "rmhd") {
         return system == PhysicalSystem::RMHD;
      }
      if (requirement == "relativistic") {
         return physical_system_is_relativistic(system);
      }
      if (requirement == "spacetime") {
         return spacetime_stores_state(config);
      }
      if (requirement == "z4c_constraints") {
         return spacetime_stores_state(config)
            && (config.mesh_coordinate == "cartesian" || config.spacetime_formulation == "fccz4");
      }
      if (requirement == "constraint_regions") {
         return config.diagnostic_constraint_region_extent > 0.0_rt;
      }
      if (requirement == "black_hole") {
         const FixedSpacetimeMetric metric =
            fixed_spacetime_parameters(config).metric;
         return metric == FixedSpacetimeMetric::KerrSchild
            || metric == FixedSpacetimeMetric::StaticTrumpet;
      }
      if (requirement == "gravity") {
         return gravity_enabled(config);
      }
      if (requirement == "passive") {
         return config.passive_scalars > 0;
      }
      if (requirement == "diffusion") {
         return config.diffusion != "none";
      }
      amrex::Abort(
         "unknown generated scalar diagnostic requirement: "
         + std::string(requirement)
      );
      return false;
   };

   for (ScalarDiagnosticInfo const& diagnostic : ScalarDiagnosticCatalog) {
      if (!selection.contains(diagnostic.id)) {
         continue;
      }
      std::stringstream requirements(std::string(diagnostic.requirements));
      std::string requirement;
      while (std::getline(requirements, requirement, ',')) {
         requirement = trim_copy(requirement);
         if (!requirement_is_satisfied(requirement)) {
            selection.enabled[static_cast<std::size_t>(diagnostic.id)] = false;
            break;
         }
      }
   }
}

void write_scalar_header(
   std::ofstream& out,
   RunConfig const& cfg,
   int step,
   amrex::Real time,
   ScalarDiagnosticSelection const& selection
)
{
   append_output_metadata(out, "scalars", cfg, step, time);
   bool first = true;
   for (ScalarDiagnosticInfo const& diagnostic : ScalarDiagnosticCatalog) {
      if (!selection.contains(diagnostic.id)) {
         continue;
      }
      out << (first ? "" : "\t") << diagnostic.name;
      first = false;
   }
   out << '\n';
}

void append_scalar_row(
   std::ofstream& out,
   ScalarDiagnostics const& diagnostics,
   ScalarDiagnosticSelection const& selection
)
{
   out << std::setprecision(17);
   bool first = true;
   visit_scalar_diagnostics(
      diagnostics,
      [&](ScalarDiagnosticId id, std::string_view, auto const& value) {
         if (!selection.contains(id)) {
            return;
         }
         out << (first ? "" : "\t") << value;
         first = false;
      }
   );
   out << '\n';
}

inline void fill_pressure_runtime(
   amrex::MultiFab& pressure,
   amrex::MultiFab const& state,
   EOSParams const& eos,
   bool mhd,
   int eos_method
)
{
   static_cast<void>(eos_method);
   if (mhd) {
      fill_pressure<true, EOSRuntime>(pressure, state, eos);
      return;
   }
   fill_pressure<false, EOSRuntime>(pressure, state, eos);
}

void write_amr_level_header(
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

void append_amr_level_row(
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
      min_feature_length > 0.0_rt ? finest_dx / min_feature_length : -1.0_rt;
   std::array<amrex::Real, AMREX_SPACEDIM> finest_dx_over_min_feature_length_by_dir{};
   bool min_feature_enabled = false;
   bool min_feature_resolved = true;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::Real directional_min_length = min_feature_lengths[dir];
      if (directional_min_length > 0.0_rt) {
         min_feature_enabled = true;
         finest_dx_over_min_feature_length_by_dir[dir] = finest_dx_by_dir[dir] / directional_min_length;
         min_feature_resolved =
            min_feature_resolved && finest_dx_by_dir[dir] <= directional_min_length;
      } else {
         finest_dx_over_min_feature_length_by_dir[dir] = -1.0_rt;
      }
   }
   const bool curvature_enabled = solid_curvature_tolerance > 0.0_rt;
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
         lev < static_cast<int>(level_max_box_aspects.size()) ? level_max_box_aspects[lev] : 0.0_rt;
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
         const amrex::Real strength = lev < static_cast<int>(dir_strength.size()) ? dir_strength[lev] : 0.0_rt;
         out << '\t' << (tags > 0 ? strength / static_cast<amrex::Real>(tags) : 0.0_rt);
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
namespace pyfluid_amrex {

bool AmrSolverBase::symmetry_axis_active(int dir) const
{
   if (dir == 0) {
      return side_boundary_kind(cfg_, 0, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 0, true) == BoundaryKind::Symmetry;
   }
#if (AMREX_SPACEDIM >= 2)
   if (dir == 1) {
      return side_boundary_kind(cfg_, 1, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 1, true) == BoundaryKind::Symmetry;
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (dir == 2) {
      return side_boundary_kind(cfg_, 2, false) == BoundaryKind::Symmetry ||
             side_boundary_kind(cfg_, 2, true) == BoundaryKind::Symmetry;
   }
#endif
   return false;
}

amrex::Real AmrSolverBase::symmetry_volume_factor() const
{
   amrex::Real factor = 1.0_rt;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (symmetry_axis_active(dir)) {
         factor *= 2.0_rt;
      }
   }
   return factor;
}

amrex::Real AmrSolverBase::symmetry_momentum_factor(int dir, amrex::Real volume_factor) const
{
   return symmetry_axis_active(dir) ? 0.0_rt : volume_factor;
}

amrex::Real AmrSolverBase::symmetry_angular_momentum_factor(int dir, amrex::Real volume_factor) const
{
   for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
      if (symmetry_axis_active(axis) && axis != dir) {
         return 0.0_rt;
      }
   }
   return volume_factor;
}

CflLimiterLocation AmrSolverBase::current_cfl_limiter_location() const
{
   amrex::Real local_frequency = -std::numeric_limits<amrex::Real>::infinity();
   int local_level = std::numeric_limits<int>::max();
   amrex::Long local_index = std::numeric_limits<amrex::Long>::max();
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (!frequency_[lev]) continue;
      const amrex::iMultiFab mask = valid_cell_mask(lev);
      const CellMaximum maximum = diagnostic_masked_maximum(
         *frequency_[lev], mask, Geom(lev).Domain(), 0
      );
      if (maximum.value > local_frequency) {
         local_frequency = maximum.value;
         local_level = lev;
         local_index = maximum.cell_index;
      }
   }

   amrex::Real global_frequency = local_frequency;
   amrex::ParallelDescriptor::ReduceRealMax(global_frequency);
   if (!std::isfinite(global_frequency) || global_frequency <= 0.0_rt) return {};

   // Select an actual maximizer, not a cell merely within a rounding tolerance.
   // Exact ties choose the lowest level and then domain-relative cell index,
   // independently of box decomposition, MPI ownership or device scheduling.
   int global_level = local_frequency == global_frequency
      ? local_level : std::numeric_limits<int>::max();
   amrex::ParallelDescriptor::ReduceIntMin(global_level);
   amrex::Long global_index =
      local_frequency == global_frequency && local_level == global_level
         ? local_index : std::numeric_limits<amrex::Long>::max();
   amrex::ParallelDescriptor::ReduceLongMin(global_index);

   const auto location = Geom(global_level).Domain().atOffset3d(global_index);
   const auto coordinates = mesh_coordinate_triplet_at_location(
      mesh_coordinate_kind(), active_mesh_axes(), Geom(global_level).ProbLoArray(),
      Geom(global_level).CellSizeArray(), location[0], location[1], location[2],
      false, false, false
   );
   CflLimiterLocation result;
   result.frequency = global_frequency;
   result.level = global_level;
   result.i = location[0];
   result.j = location[1];
   result.k = location[2];
   result.q1 = coordinates[0];
   result.q2 = coordinates[1];
   result.q3 = coordinates[2];
   return result;
}

ScalarDiagnostics AmrSolverBase::compute_current_scalars(int step, amrex::Real time, amrex::Real dt, amrex::Real mass0,
                                                         amrex::Real energy0, amrex::Real gravity_total_energy0,
                                                         amrex::Real passive_mass0, amrex::Real max_rho0,
                                                         amrex::Real mcells_per_s, amrex::Real layout_mcells_per_s,
                                                         amrex::Real sim_time_per_s, DiffusionStats diffusion,
                                                         int eos_method,
                                                         ScalarDiagnosticSelection const& selection)
{
   ScalarDiagnostics d;
   d.step = step;
   d.time = time;
   d.dt = dt;
   d.min_rho = std::numeric_limits<amrex::Real>::max();
   d.max_rho = -std::numeric_limits<amrex::Real>::max();
   d.min_pressure = std::numeric_limits<amrex::Real>::max();
   d.min_lapse = std::numeric_limits<amrex::Real>::max();
   d.rmhd_min_plasma_beta = std::numeric_limits<amrex::Real>::max();
   amrex::Real relativistic_diagnostic_volume = 0.0_rt;
   amrex::Real gravity_density_offset = 0.0_rt;
   amrex::Real gravity_energy_factor = 0.0_rt;
   if (gravity_enabled(cfg_)) {
      prepare_gravity_fields(fluid_state_.state(), false);
      if (self_gravity_enabled(cfg_)) {
         const DensityTotal density = valid_density_total(fluid_state_.state());
         gravity_density_offset = density.volume > 0.0_rt ? density.mass / density.volume : 0.0_rt;
         gravity_energy_factor = 0.5_rt;
      } else {
         gravity_energy_factor = 1.0_rt;
      }
   }

   const int coordinate = mesh_coordinate_kind();
   const MeshAxes axes = active_mesh_axes();
   const bool analytic_fixed_spacetime =
      uses_analytic_fixed_spacetime_kernels();
   const FixedSpacetimeParameters fixed_spacetime =
      fixed_spacetime_parameters(cfg_);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      if (physical_system_is_relativistic(physics_system_)) {
         fill_level_primitive_cache(lev, time);
         fill_pressure_from_primitive(*pressure_[lev], *primitive_[lev]);
      } else {
         fill_pressure_runtime(*pressure_[lev], *fluid_state_.state()[lev], eos_params(cfg_), is_mhd(cfg_), eos_method);
      }
      amrex::iMultiFab mask = valid_cell_mask(lev);
      amrex::MultiFab reduction(fluid_state_.state()[lev]->boxArray(), fluid_state_.state()[lev]->DistributionMap(),
                                ScalarReductionNComp, 0);
      amrex::MultiFab const* gravity_phi = gravity_enabled(cfg_) ? gravity_fields_.potential(lev) : nullptr;
      const int passive_component = cfg_.passive_scalars > 0 ? passive_component_begin(is_mhd(cfg_)) : -1;
      if (mixed_patch_system_enabled() && !is_mhd(cfg_)) {
         fill_scalar_reduction_fields_mapped_hydro(reduction, *fluid_state_.state()[lev], mask,
                                                   production_patch_grid_system_for_level(lev),
                                                   production_patch_layout_for_level(lev), gravity_phi,
                                                   gravity_density_offset, gravity_energy_factor, passive_component);
      } else if (is_mhd(cfg_)) {
         fill_scalar_reduction_fields<true>(
            reduction, *fluid_state_.state()[lev], mask, Geom(lev), coordinate,
            axes, face_b_ptr(lev), nullptr, gravity_phi,
            gravity_density_offset, gravity_energy_factor, passive_component
         );
      } else {
         fill_scalar_reduction_fields<false>(
            reduction, *fluid_state_.state()[lev], mask, Geom(lev), coordinate,
            axes, nullptr,
            solid_active(lev) ? solid_volume_fraction_[lev].get() : nullptr, gravity_phi,
            gravity_density_offset, gravity_energy_factor, passive_component
         );
      }
      d.mass += diagnostic_component_sum(reduction, ScalarReductionMass);
      d.mom1 += diagnostic_component_sum(reduction, ScalarReductionMom1);
      d.mom2 += diagnostic_component_sum(reduction, ScalarReductionMom2);
      d.mom3 += diagnostic_component_sum(reduction, ScalarReductionMom3);
      d.energy += diagnostic_component_sum(reduction, ScalarReductionEnergy);
      amrex::MultiFab const& density =
         physical_system_is_relativistic(physics_system_)
            ? *primitive_[lev]
            : *fluid_state_.state()[lev];
      const DensityPressureExtrema extrema =
         valid_density_pressure_extrema(density, *pressure_[lev], mask);
      d.min_rho = amrex::min(d.min_rho, extrema.min_rho);
      d.max_rho = amrex::max(d.max_rho, extrema.max_rho);
      d.min_pressure = amrex::min(d.min_pressure, extrema.min_pressure);
      if (physics_system_ == PhysicalSystem::MHD) {
         d.newtonian_magnetic_energy += diagnostic_component_sum(reduction, ScalarReductionMagneticEnergy);
      }
      d.divb_l1 += diagnostic_component_sum(reduction, ScalarReductionDivB);
      d.divb_l2 += diagnostic_component_sum(reduction, ScalarReductionDivBSquared);
      d.divb_linf = amrex::max(
         d.divb_linf,
         reduction.norm0(ScalarReductionDivBAbs, 0, false)
      );
      d.angular_mom1 += diagnostic_component_sum(reduction, ScalarReductionAngularMom1);
      d.angular_mom2 += diagnostic_component_sum(reduction, ScalarReductionAngularMom2);
      d.angular_mom3 += diagnostic_component_sum(reduction, ScalarReductionAngularMom3);
      d.gravity_potential_energy +=
         diagnostic_component_sum(reduction, ScalarReductionGravityPotentialEnergy);
      d.passive_mass += diagnostic_component_sum(reduction, ScalarReductionPassiveMass);
      if (physical_system_is_relativistic(physics_system_)) {
         amrex::MultiFab relativistic_reduction(
            fluid_state_.state()[lev]->boxArray(),
            fluid_state_.state()[lev]->DistributionMap(),
            RelativisticReductionNComp,
            0
         );
         if (physics_system_ == PhysicalSystem::RMHD) {
            fill_relativistic_diagnostic_fields<true>(
               relativistic_reduction,
               *primitive_[lev],
               spacetime_coupling_.cell_spacetime()[lev].get(),
               mask,
               Geom(lev),
               coordinate,
               axes,
               eos_params(cfg_),
               fixed_spacetime,
               time,
               false
            );
            d.rmhd_electromagnetic_energy += diagnostic_component_sum(relativistic_reduction,
               RelativisticReductionElectromagneticEnergy
            );
            d.rmhd_max_b_squared = amrex::max(
               d.rmhd_max_b_squared,
               relativistic_reduction.norm0(
                  RelativisticReductionMaxBSquared, 0, false
               )
            );
            d.rmhd_max_enthalpy_magnetization = amrex::max(
               d.rmhd_max_enthalpy_magnetization,
               relativistic_reduction.norm0(
                  RelativisticReductionMaxEnthalpyMagnetization, 0, false
               )
            );
            d.rmhd_min_plasma_beta = amrex::min(
               d.rmhd_min_plasma_beta,
               relativistic_reduction.min(
                  RelativisticReductionMinPlasmaBeta, 0, false
               )
            );
         } else {
            fill_relativistic_diagnostic_fields<false>(
               relativistic_reduction,
               *primitive_[lev],
               analytic_fixed_spacetime
                  ? nullptr
                  : spacetime_coupling_.cell_spacetime()[lev].get(),
               mask,
               Geom(lev),
               coordinate,
               axes,
               eos_params(cfg_),
               fixed_spacetime,
               time,
               analytic_fixed_spacetime
            );
         }
         d.max_lorentz_factor = amrex::max(
            d.max_lorentz_factor,
            relativistic_reduction.norm0(
               RelativisticReductionMaxLorentzFactor, 0, false
            )
         );
         d.relativistic_atmosphere_volume_fraction += diagnostic_component_sum(relativistic_reduction,
            RelativisticReductionAtmosphereVolume
         );
         d.min_lapse = amrex::min(
            d.min_lapse,
            relativistic_reduction.min(
               RelativisticReductionMinLapse, 0, false
            )
         );
         relativistic_diagnostic_volume += diagnostic_component_sum(relativistic_reduction,
            RelativisticReductionTotalVolume
         );
      }
   }

   const amrex::Real symmetry_factor = symmetry_volume_factor();
   d.mass *= symmetry_factor;
   d.passive_mass *= symmetry_factor;
   d.mom1 *= symmetry_momentum_factor(0, symmetry_factor);
   d.mom2 *= symmetry_momentum_factor(1, symmetry_factor);
   d.mom3 *= symmetry_momentum_factor(2, symmetry_factor);
   d.angular_mom1 *= symmetry_angular_momentum_factor(0, symmetry_factor);
   d.angular_mom2 *= symmetry_angular_momentum_factor(1, symmetry_factor);
   d.angular_mom3 *= symmetry_angular_momentum_factor(2, symmetry_factor);
   d.energy *= symmetry_factor;
   d.gravity_potential_energy *= symmetry_factor;
   d.gravity_total_energy = d.energy + d.gravity_potential_energy;
   d.newtonian_magnetic_energy *= symmetry_factor;
   d.rmhd_electromagnetic_energy *= symmetry_factor;
   d.divb_l1 *= symmetry_factor;
   d.divb_l2 = std::sqrt(d.divb_l2 * symmetry_factor);
   if (!physical_system_is_relativistic(physics_system_)
       && spacetime_state_.active()) {
      for (int lev = 0; lev <= finestLevel(); ++lev) {
         d.min_lapse = amrex::min(
            d.min_lapse,
            spacetime_state_.state()[lev]->min(Z4C_LAPSE, 0, false)
         );
      }
   } else if (!physical_system_is_relativistic(physics_system_)) {
      d.min_lapse = 0.0_rt;
   }
   if (relativistic_diagnostic_volume > 0.0_rt) {
      d.relativistic_atmosphere_volume_fraction /= relativistic_diagnostic_volume;
   }
   if (physics_system_ != PhysicalSystem::RMHD
       || d.rmhd_max_b_squared == 0.0_rt
       || d.rmhd_min_plasma_beta == std::numeric_limits<amrex::Real>::max()) {
      d.rmhd_min_plasma_beta = 0.0_rt;
   }
   d.mass_rel_drift = amrex::Math::abs(d.mass - mass0) / amrex::max(amrex::Math::abs(mass0), 1.0_rt);
   d.energy_rel_drift = amrex::Math::abs(d.energy - energy0) / amrex::max(amrex::Math::abs(energy0), 1.0_rt);
   d.max_rho_rel_change = std::isfinite(max_rho0) && max_rho0 > 0.0_rt ? (d.max_rho - max_rho0) / max_rho0 : 0.0_rt;
   d.passive_mass_rel_drift = cfg_.passive_scalars > 0 ? amrex::Math::abs(d.passive_mass - passive_mass0) /
                                                            amrex::max(amrex::Math::abs(passive_mass0), 1.0_rt)
                                                       : 0.0_rt;
   const amrex::Real gravity_total_energy_scale = amrex::Math::abs(gravity_total_energy0);
   d.gravity_total_energy_rel_drift = amrex::Math::abs(d.gravity_total_energy - gravity_total_energy0) /
                                      (gravity_total_energy_scale > 0.0_rt ? gravity_total_energy_scale : 1.0_rt);
   const amrex::Real energy_delta = d.energy - wall_work_reference_energy_;
   d.wall_work = cumulative_wall_work_;
   d.wall_power = latest_wall_power_;
   d.energy_minus_wall_work = energy_delta - cumulative_wall_work_;
   const amrex::Real work_scale =
      amrex::max(amrex::max(amrex::Math::abs(energy_delta), amrex::Math::abs(cumulative_wall_work_)), 1.0_rt);
   d.wall_work_rel_error = amrex::Math::abs(d.energy_minus_wall_work) / work_scale;
   d.external_mass_budget_valid = external_conservation_budget_supported() ? 1 : 0;
   d.external_budget_valid = d.external_mass_budget_valid && !spacetime_evolves(cfg_) ? 1 : 0;
   if (d.external_mass_budget_valid) {
      d.boundary_mass_change = conservation_ledger_.boundary_change.mass;
      d.boundary_energy_change = conservation_ledger_.boundary_change.energy;
      d.source_mass_change = conservation_ledger_.source_change.mass;
      d.source_energy_change = conservation_ledger_.source_change.energy;
      amrex::Vector<std::reference_wrapper<amrex::Real>> ledger_changes{
         d.boundary_mass_change,
         d.boundary_energy_change,
         d.source_mass_change,
         d.source_energy_change,
      };
      amrex::ParallelDescriptor::ReduceRealSum(ledger_changes);
      d.boundary_mass_change *= symmetry_factor;
      d.boundary_energy_change *= symmetry_factor;
      d.source_mass_change *= symmetry_factor;
      d.source_energy_change *= symmetry_factor;
      d.external_mass_change = d.boundary_mass_change + d.source_mass_change;
      d.external_energy_change = d.boundary_energy_change + d.source_energy_change + d.wall_work;
      const amrex::Real measured_mass_change = d.mass - conservation_ledger_.reference.mass;
      const amrex::Real measured_energy_change = d.energy - conservation_ledger_.reference.energy;
      d.external_mass_residual = measured_mass_change - d.external_mass_change;
      d.external_energy_residual = measured_energy_change - d.external_energy_change;
      const amrex::Real mass_scale =
         amrex::max(amrex::max(amrex::Math::abs(measured_mass_change), amrex::Math::abs(d.external_mass_change)), 1.0_rt);
      const amrex::Real energy_scale = amrex::max(
         amrex::max(amrex::Math::abs(measured_energy_change), amrex::Math::abs(d.external_energy_change)), 1.0_rt);
      d.external_mass_rel_error = amrex::Math::abs(d.external_mass_residual) / mass_scale;
      d.external_energy_rel_error = amrex::Math::abs(d.external_energy_residual) / energy_scale;
   }
   d.diffusion_residual = diffusion.residual;
   d.diffusion_iterations = diffusion.iterations;
   const AmrLevelDiagnostics amr = current_amr_level_diagnostics();
   d.amr_total_cells = amr.total_cells;
   d.amr_active_finest_level = amr.active_finest_level;
   d.amr_finest_headroom = amr.finest_headroom;
   d.amr_touches_max_level = amr.touches_max_level;
   d.amr_total_budget_fraction = amr.total_budget_fraction;
   d.amr_peak_level_budget_fraction = amr.peak_level_budget_fraction;
   d.mcells_per_s = mcells_per_s;
   d.layout_mcells_per_s = layout_mcells_per_s;
   d.sim_time_per_s = sim_time_per_s;
   if (cfg_.cfl_limiter_diagnostics) {
      set_cfl_limiter_diagnostics(d, current_cfl_limiter_location());
   }
   bool spacetime_matter_current = false;
   const bool constraint_regions_requested =
      selection.contains_any(ScalarDiagnosticGroup::ConstraintRegions);
   if (spacetime_state_.active()
       && (selection.contains_any(ScalarDiagnosticGroup::Constraints)
           || constraint_regions_requested)) {
      if (spacetime_includes_matter(cfg_)) {
         fill_spacetime_matter_range(0, finestLevel() + 1);
         spacetime_matter_current = true;
      }
      const SpacetimeConstraintDiagnostics constraints =
         this->compute_spacetime_constraint_diagnostics(time, constraint_regions_requested);
      d.z4c_hamiltonian_linf = constraints.hamiltonian_linf;
      d.z4c_hamiltonian_l2 = constraints.hamiltonian_l2;
      d.z4c_hamiltonian_interior_linf = constraints.hamiltonian_interior_linf;
      d.z4c_hamiltonian_interior_l2 = constraints.hamiltonian_interior_l2;
      d.z4c_hamiltonian_cf_linf = constraints.hamiltonian_cf_linf;
      d.z4c_hamiltonian_cf_l2 = constraints.hamiltonian_cf_l2;
      d.z4c_hamiltonian_bh_exterior_linf =
         constraints.hamiltonian_bh_exterior_linf;
      d.z4c_hamiltonian_bh_exterior_l2 =
         constraints.hamiltonian_bh_exterior_l2;
      d.z4c_momentum_bh_exterior_linf =
         constraints.momentum_bh_exterior_linf;
      d.z4c_momentum_linf = constraints.momentum_linf;
      d.z4c_z_linf = constraints.z_linf;
      d.z4c_theta_linf = constraints.theta_linf;
      d.z4c_log_conformal_determinant_linf = constraints.log_conformal_determinant_linf;
      d.z4c_trace_a_tilde_linf = constraints.trace_a_tilde_linf;
      d.z4c_hamiltonian_region_l2 = constraints.regional_hamiltonian_l2[0];
      d.z4c_hamiltonian_region_linf = constraints.regional_hamiltonian_linf[0];
      d.z4c_momentum_region_linf = constraints.regional_momentum_linf[0];
      d.z4c_region_volume = constraints.regional_volume[0];
      d.z4c_hamiltonian_near_l2 = constraints.regional_hamiltonian_l2[1];
      d.z4c_hamiltonian_near_linf = constraints.regional_hamiltonian_linf[1];
      d.z4c_momentum_near_linf = constraints.regional_momentum_linf[1];
      d.z4c_near_volume = constraints.regional_volume[1];
      d.z4c_hamiltonian_far_l2 = constraints.regional_hamiltonian_l2[2];
      d.z4c_hamiltonian_far_linf = constraints.regional_hamiltonian_linf[2];
      d.z4c_momentum_far_linf = constraints.regional_momentum_linf[2];
      d.z4c_far_volume = constraints.regional_volume[2];
   }
   if (selection.contains_any(ScalarDiagnosticGroup::BlackHole)) {
      d.black_hole_horizon_radius = black_hole_coordinate_horizon_radius(
         fixed_spacetime_parameters(cfg_)
      );
   }
   const bool horizon_geometry_requested =
      selection.contains_any(ScalarDiagnosticGroup::Horizon);
   const bool horizon_flux_requested = selection.contains(
      ScalarDiagnosticId::BlackHoleHorizonBaryonAccretionRate
   );
   if (horizon_geometry_requested || horizon_flux_requested) {
      const ApparentHorizonResult* horizon =
         apparent_horizon_tracker_->result();
      if (horizon == nullptr) {
         if (apparent_horizon_tracker_->abort_on_failure()) {
            amrex::Abort(
               "horizon scalar diagnostics require horizon.enabled=1 and a "
               "successful apparent-horizon search"
            );
         }
      } else {
         d.black_hole_horizon_found = 1.0_rt;
         d.black_hole_horizon_area = horizon->area;
         d.black_hole_horizon_irreducible_mass = horizon->irreducible_mass;
         d.black_hole_horizon_equatorial_circumference =
            horizon->equatorial_circumference;
         d.black_hole_horizon_polar_circumference =
            horizon->polar_circumference;
         d.black_hole_horizon_kerr_mass = horizon->kerr_horizon_mass;
         d.black_hole_horizon_kerr_spin = horizon->kerr_dimensionless_spin;
         d.black_hole_horizon_coordinate_angular_momentum1 =
            horizon->angular_momentum[0];
         d.black_hole_horizon_coordinate_angular_momentum2 =
            horizon->angular_momentum[1];
         d.black_hole_horizon_coordinate_angular_momentum3 =
            horizon->angular_momentum[2];
         d.black_hole_horizon_coordinate_angular_momentum_magnitude =
            horizon->angular_momentum_magnitude;
         d.black_hole_horizon_coordinate_christodoulou_mass =
            horizon->christodoulou_mass;
         d.black_hole_horizon_coordinate_christodoulou_spin1 =
            horizon->christodoulou_dimensionless_spin[0];
         d.black_hole_horizon_coordinate_christodoulou_spin2 =
            horizon->christodoulou_dimensionless_spin[1];
         d.black_hole_horizon_coordinate_christodoulou_spin3 =
            horizon->christodoulou_dimensionless_spin[2];
         d.black_hole_horizon_coordinate_christodoulou_spin_magnitude =
            horizon->christodoulou_dimensionless_spin_magnitude;
         if (horizon_flux_requested) {
            d.black_hole_horizon_baryon_accretion_rate =
               compute_baryon_surface_accretion_rate(
                  diagnostic_apparent_horizon_surface(), time
               );
         }
      }
   }
   const bool global_gr_requested =
      selection.contains_any(ScalarDiagnosticGroup::GlobalGr);
   if (global_gr_requested) {
      if (!spacetime_state_.active()) {
         amrex::Abort("global_gr diagnostics require an active spacetime state");
      }
      const bool surface_requested =
         selection.contains(ScalarDiagnosticId::AdmMassSurface)
         || selection.contains(ScalarDiagnosticId::KomarMassSurface)
         || selection.contains(ScalarDiagnosticId::AdmMomentum1Surface)
         || selection.contains(ScalarDiagnosticId::AdmMomentum2Surface)
         || selection.contains(ScalarDiagnosticId::AdmMomentum3Surface)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum1Surface)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum2Surface)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum3Surface);
      const bool volume_requested =
         selection.contains(ScalarDiagnosticId::AdmMassVolume)
         || selection.contains(ScalarDiagnosticId::ProperMassVolume)
         || selection.contains(ScalarDiagnosticId::KomarMassVolume)
         || selection.contains(ScalarDiagnosticId::BindingEnergyVolume)
         || selection.contains(ScalarDiagnosticId::AdmMomentum1Volume)
         || selection.contains(ScalarDiagnosticId::AdmMomentum2Volume)
         || selection.contains(ScalarDiagnosticId::AdmMomentum3Volume)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum1Volume)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum2Volume)
         || selection.contains(ScalarDiagnosticId::AdmAngularMomentum3Volume);
      if (volume_requested && spacetime_includes_matter(cfg_)
          && !spacetime_matter_current) {
         fill_spacetime_matter_range(0, finestLevel() + 1);
      }
      const SpacetimeGlobalDiagnostics global = compute_spacetime_global_diagnostics(
         time, surface_requested, volume_requested
      );
      d.adm_mass_surface = global.adm_mass_surface;
      d.adm_mass_volume = global.adm_mass_volume;
      d.proper_mass_volume = global.proper_mass_volume;
      d.komar_mass_surface = global.komar_mass_surface;
      d.komar_mass_volume = global.komar_mass_volume;
      d.binding_energy_volume = d.mass - global.adm_mass_volume;
      d.adm_momentum1_surface = global.adm_momentum_surface[0];
      d.adm_momentum2_surface = global.adm_momentum_surface[1];
      d.adm_momentum3_surface = global.adm_momentum_surface[2];
      d.adm_momentum1_volume = global.adm_momentum_volume[0];
      d.adm_momentum2_volume = global.adm_momentum_volume[1];
      d.adm_momentum3_volume = global.adm_momentum_volume[2];
      d.adm_angular_momentum1_surface = global.adm_angular_momentum_surface[0];
      d.adm_angular_momentum2_surface = global.adm_angular_momentum_surface[1];
      d.adm_angular_momentum3_surface = global.adm_angular_momentum_surface[2];
      d.adm_angular_momentum1_volume = global.adm_angular_momentum_volume[0];
      d.adm_angular_momentum2_volume = global.adm_angular_momentum_volume[1];
      d.adm_angular_momentum3_volume = global.adm_angular_momentum_volume[2];
   }
   return d;
}

} // namespace pyfluid_amrex
