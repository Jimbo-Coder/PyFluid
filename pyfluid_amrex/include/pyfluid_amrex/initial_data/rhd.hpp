#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/initial_data/fluid.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

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
   const MeshAxes axes = mesh_axes_layout(cfg);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   validate_fixed_spacetime_domain(spacetime, geometry);
   const PhysicalSystem system = physical_system(cfg);
   const bool magnetized = physical_system_has_magnetic_field(system);
   const bool initialize_magnetic_from_id = initial_data.fields.magnetic_from_id;
   const FluidInitialDataParameters parameters = fluid_initial_data_parameters(cfg);
   const FixedSpacetimeGeometry reference_geometry = fixed_spacetime_geometry(
      spacetime,
      0.5 * (problo[0] + probhi[0]),
      AMREX_SPACEDIM >= 2 ? 0.5 * (problo[1] + probhi[1]) : 0.0,
      AMREX_SPACEDIM == 3 ? 0.5 * (problo[2] + probhi[2]) : 0.0,
      0.0
   );
   const FluidInitialDataContext context{
      problo,
      probhi,
      dx,
      AMREX_SPACEDIM >= 2 && geometry.isPeriodic(1),
      reference_geometry.metric.lapse,
   };

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
            MeshCartesian, axes, problo, dx, i, j, k
         );
         const FixedSpacetimeGeometry local_geometry = fixed_spacetime_geometry(
            spacetime,
            sample.physical.q1,
            AMREX_SPACEDIM >= 2 ? sample.physical.q2 : 0.0,
            AMREX_SPACEDIM == 3 ? sample.physical.q3 : 0.0,
            0.0
         );
         FluidPrimitiveSample point = sample_fluid_initial_data<EOSMethod>(
            parameters, context, sample, local_geometry, initial_data
         );
         if (!initialize_magnetic_from_id) {
            point.b1 = 0.0;
            point.b2 = 0.0;
            point.b3 = 0.0;
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
   const int eos_method = eos_method_code(cfg);
   if (eos_method == EOSPolytrope) {
      initialize_relativistic_state_method<EOSPolytrope>(
         state, geometry, cfg, initial_data
      );
   } else if (eos_method == EOSHybridPolytrope) {
      initialize_relativistic_state_method<EOSHybridPolytrope>(
         state, geometry, cfg, initial_data
      );
   } else {
      initialize_relativistic_state_method<EOSIdealGas>(
         state, geometry, cfg, initial_data
      );
   }
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
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   const EOSParams eos = eos_params(cfg);
   const bool magnetized = physical_system(cfg) == PhysicalSystem::RMHD;
   const bool magnetic_from_id = initial_data.fields.magnetic_from_id;

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
         const FixedSpacetimeGeometry geometry_at_cell =
            fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
               spacetime, lower, spacing, i, j, k, 0.0
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
               magnetic_from_id ? primitive.b1 : 0.0,
               magnetic_from_id ? primitive.b2 : 0.0,
               magnetic_from_id ? primitive.b3 : 0.0,
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
            const HydroPrimitive primitive = primitive_from_conserved<EOSMethod>(
               load_hydro_state(source, i, j, k), eos
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
   const int eos_method = eos_method_code(cfg);
   if (eos_method == EOSPolytrope) {
      convert_newtonian_initial_state_to_relativistic_method<EOSPolytrope>(
         relativistic_state, newtonian_state, geometry, cfg, initial_data
      );
   } else if (eos_method == EOSHybridPolytrope) {
      convert_newtonian_initial_state_to_relativistic_method<EOSHybridPolytrope>(
         relativistic_state, newtonian_state, geometry, cfg, initial_data
      );
   } else {
      convert_newtonian_initial_state_to_relativistic_method<EOSIdealGas>(
         relativistic_state, newtonian_state, geometry, cfg, initial_data
      );
   }
}

} // namespace pyfluid_amrex
