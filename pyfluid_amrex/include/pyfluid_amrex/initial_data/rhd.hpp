#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/initial_data/fluid.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ReferenceMetricSpacetimeGeometry initial_data_reference_spacetime_geometry_at_cell(
   FixedSpacetimeParameters const& spacetime,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lower,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
   int i,
   int j,
   int k,
   amrex::Real time
) noexcept
{
   const auto q = mesh_coordinate_triplet_at_location(
      coordinate, axes, lower, spacing, i, j, k, false, false, false
   );
   const auto position = mesh_coordinate_triplet_to_cartesian(coordinate, q);
   const FixedSpacetimeGeometry cartesian = initial_data_spacetime_geometry(
      spacetime, position[0], position[1], position[2], time
   );
   return reference_metric_spacetime_geometry_from_cartesian(
      cartesian, coordinate, axes, q
   );
}

inline amrex::Real initial_data_reference_lapse_at_mesh_center(
   FixedSpacetimeParameters const& spacetime,
   int coordinate,
   MeshAxes const& axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi
)
{
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center{};
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero_spacing{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      center[direction] = 0.5_rt * (problo[direction] + probhi[direction]);
   }
   const auto q = mesh_coordinate_triplet_at_location(
      coordinate,
      axes,
      center,
      zero_spacing,
      0,
      0,
      0,
      false,
      false,
      false
   );
   const auto position = mesh_coordinate_triplet_to_cartesian(coordinate, q);
   return initial_data_spacetime_geometry(
      spacetime, position[0], position[1], position[2], 0.0_rt
   ).metric.lapse;
}

template <int EOSMethod>
inline void initialize_relativistic_state_method(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   const auto problo = geometry.ProbLoArray();
   const auto probhi = geometry.ProbHiArray();
   const auto dx = geometry.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   validate_reference_metric_fixed_spacetime_domain(
      spacetime, geometry, mesh_coordinate, axes
   );
   const PhysicalSystem system = physical_system(cfg);
   const bool magnetized = physical_system_has_magnetic_field(system);
   const bool initialize_magnetic_from_id = initial_data.fields.magnetic_from_id;
   const FluidInitialDataParameters parameters = fluid_initial_data_parameters(
      cfg, initial_data
   );
   const FluidInitialDataContext context{
      problo,
      probhi,
      dx,
      AMREX_SPACEDIM >= 2 && geometry.isPeriodic(1),
      initial_data_reference_lapse_at_mesh_center(
         spacetime, mesh_coordinate, axes, problo, probhi
      ),
   };

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
            mesh_coordinate, axes, problo, dx, i, j, k
         );
         const ReferenceMetricSpacetimeGeometry local_geometry =
            initial_data_reference_spacetime_geometry_at_cell(
               spacetime,
               mesh_coordinate,
               axes,
               problo,
               dx,
               i,
               j,
               k,
               0.0_rt
            );
         const FixedSpacetimeGeometry cartesian_geometry =
            initial_data_spacetime_geometry(
               spacetime,
               sample.physical.q1,
               sample.physical.q2,
               sample.physical.q3,
               0.0_rt
            );
         const SpacetimeMetric local_metric = magnetized
            ? cartesian_geometry.metric
            : local_geometry.metric;
         FluidPrimitiveSample point = sample_fluid_initial_data<EOSMethod>(
            parameters, context, sample, local_metric, initial_data
         );
         if (!initialize_magnetic_from_id) {
            point.b1 = 0.0_rt;
            point.b2 = 0.0_rt;
            point.b3 = 0.0_rt;
         }
         if (magnetized) {
            const RMHDPrimitive primitive{
               point.fluid.rho,
               point.fluid.eps,
               point.fluid.pressure,
               point.fluid.v1,
               point.fluid.v2,
               point.fluid.v3,
               point.b1,
               point.b2,
               point.b3,
            };
            store_rmhd_conserved(
               conserved,
               i,
               j,
               k,
               generated::rmhd_primitive_to_conserved(primitive, point.metric)
            );
         } else {
            store_rhd_conserved(
               conserved,
               i,
               j,
               k,
               generated::rhd_primitive_to_conserved(point.fluid, point.metric)
            );
         }
      });
   }
}

inline void initialize_relativistic_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data = {}
)
{
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
      initialize_relativistic_state_method<EOSIdealGas>(
         state, geometry, cfg, initial_data
      );
      return;
   }
   initialize_relativistic_state_method<EOSRuntime>(
      state, geometry, cfg, initial_data
   );
}

template <int EOSMethod>
inline void convert_newtonian_initial_state_to_relativistic_method(
   amrex::MultiFab& relativistic_state,
   amrex::MultiFab const& newtonian_state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   const EOSParams eos = eos_params(cfg);
   const bool magnetized = physical_system(cfg) == PhysicalSystem::RMHD;
   const bool magnetic_from_id = initial_data.fields.magnetic_from_id;
   const bool exact_vacuum = relativistic_exterior_state(cfg).is_vacuum();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(
           relativistic_state, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto source = newtonian_state.const_array(mfi);
      const auto destination = relativistic_state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const ReferenceMetricSpacetimeGeometry geometry_at_cell =
            initial_data_reference_spacetime_geometry_at_cell(
               spacetime,
               mesh_coordinate,
               axes,
               lower,
               spacing,
               i,
               j,
               k,
               0.0_rt
            );
         if (magnetized) {
            const MHDPrimitive primitive = primitive_from_conserved<EOSMethod>(
               load_mhd_state(source, i, j, k), eos
            );
            const RMHDPrimitive relativistic_primitive{
               primitive.rho,
               eos_internal_energy_density_from_pressure<EOSMethod>(
                  primitive.rho, primitive.pressure, eos
               ) / primitive.rho,
               primitive.pressure,
               primitive.v1,
               primitive.v2,
               primitive.v3,
               magnetic_from_id ? primitive.b1 : 0.0_rt,
               magnetic_from_id ? primitive.b2 : 0.0_rt,
               magnetic_from_id ? primitive.b3 : 0.0_rt,
            };
            store_rmhd_conserved(
               destination,
               i,
               j,
               k,
               generated::rmhd_primitive_to_conserved(
                  relativistic_primitive, geometry_at_cell.metric
               )
            );
         } else {
            const HydroState source_conserved = load_hydro_state(
               source, i, j, k
            );
            if (exact_vacuum && source_conserved.rho == 0.0_rt
                && source_conserved.mom1 == 0.0_rt
                && source_conserved.mom2 == 0.0_rt
                && source_conserved.mom3 == 0.0_rt
                && source_conserved.energy == 0.0_rt) {
               store_rhd_conserved(
                  destination, i, j, k, RHDConserved{}
               );
               return;
            }
            const HydroPrimitive primitive = primitive_from_conserved<EOSMethod>(
               source_conserved, eos
            );
            const RHDPrimitive relativistic_primitive{
               primitive.rho,
               eos_internal_energy_density_from_pressure<EOSMethod>(
                  primitive.rho, primitive.pressure, eos
               ) / primitive.rho,
               primitive.pressure,
               primitive.v1,
               primitive.v2,
               primitive.v3,
            };
            store_rhd_conserved(
               destination,
               i,
               j,
               k,
               generated::rhd_primitive_to_conserved(
                  relativistic_primitive, geometry_at_cell.metric
               )
            );
         }
      });
   }
}

inline void convert_newtonian_initial_state_to_relativistic(
   amrex::MultiFab& relativistic_state,
   amrex::MultiFab const& newtonian_state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   convert_newtonian_initial_state_to_relativistic_method<EOSRuntime>(
      relativistic_state, newtonian_state, geometry, cfg, initial_data
   );
}

} // namespace pyfluid_amrex
