#include "pyfluid_amrex/initial_data/rhd.hpp"

#include <AMReX_GpuAsyncArray.H>

#include "pyfluid_amrex/initial_data/elliptic.hpp"

namespace pyfluid_amrex {
namespace {

template <int EOSMethod>
void initialize_relativistic_primitive_cache_from_initial_data_method(
   amrex::MultiFab& primitive,
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
   const FixedSpacetimeGeometry reference_geometry = initial_data_spacetime_geometry(
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
   const FluidInitialDataParameters parameters = fluid_initial_data_parameters(
      cfg, initial_data
   );
   const bool magnetized = physical_system_has_magnetic_field(physical_system(cfg));
   const int base_components = magnetized ? NVAR_MHD : NVAR_HYDRO;
   primitive.setVal(
      0.0, 0, base_components, primitive.nGrowVect()
   );
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
            mesh_coordinate, axes, problo, dx, i, j, k
         );
         const FixedSpacetimeGeometry local_geometry = initial_data_spacetime_geometry(
            spacetime,
            sample.physical.q1,
            AMREX_SPACEDIM >= 2 ? sample.physical.q2 : 0.0,
            AMREX_SPACEDIM == 3 ? sample.physical.q3 : 0.0,
            0.0
         );
         const FluidPrimitiveSample point = sample_fluid_initial_data<EOSMethod>(
            parameters, context, sample, local_geometry, initial_data
         );
         cached(i, j, k, RHO) = point.fluid.rho;
         cached(i, j, k, MOM1) = point.fluid.v1;
         cached(i, j, k, MOM2) = point.fluid.v2;
         cached(i, j, k, MOM3) = point.fluid.v3;
         cached(i, j, k, ENERGY) = point.fluid.pressure;
         if (magnetized) {
            cached(i, j, k, B1) = point.b1;
            cached(i, j, k, B2) = point.b2;
            cached(i, j, k, B3) = point.b3;
         }
      });
   }
}

template <int EOSMethod>
void initialize_relativistic_primitive_cache_from_mapped_provider_method(
   amrex::MultiFab& primitive,
   InitialDataEllipticHierarchy const& hierarchy,
   int level,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   if (!initial_data.has_fluid() || level < 0
       || level >= static_cast<int>(hierarchy.geometry.size())) {
      amrex::Abort("mapped relativistic primitive initialization is invalid");
   }
   amrex::Geometry const& geometry = hierarchy.geometry[level];
   const auto problo = geometry.ProbLoArray();
   const auto probhi = geometry.ProbHiArray();
   const auto dx = geometry.CellSizeArray();
   const amrex::Dim3 domain_lo = amrex::lbound(geometry.Domain());
   const int mesh_coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const InitialDataCoordinateLine coordinate1 =
      hierarchy.coordinate_line(level, 0);
   amrex::Gpu::AsyncArray<amrex::Real> coordinate1_storage(
      coordinate1.centers.data(), coordinate1.centers.size()
   );
   const amrex::Real* coordinate1_values = coordinate1_storage.data();
#if (AMREX_SPACEDIM >= 2)
   const InitialDataCoordinateLine coordinate2 =
      hierarchy.coordinate_line(level, 1);
   amrex::Gpu::AsyncArray<amrex::Real> coordinate2_storage(
      coordinate2.centers.data(), coordinate2.centers.size()
   );
   const amrex::Real* coordinate2_values = coordinate2_storage.data();
#endif
#if (AMREX_SPACEDIM == 3)
   const InitialDataCoordinateLine coordinate3 =
      hierarchy.coordinate_line(level, 2);
   amrex::Gpu::AsyncArray<amrex::Real> coordinate3_storage(
      coordinate3.centers.data(), coordinate3.centers.size()
   );
   const amrex::Real* coordinate3_values = coordinate3_storage.data();
#endif
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   const FixedSpacetimeGeometry reference_geometry = initial_data_spacetime_geometry(
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
   const FluidInitialDataParameters parameters = fluid_initial_data_parameters(
      cfg, initial_data
   );
   const bool magnetized = physical_system_has_magnetic_field(
      physical_system(cfg)
   );
   primitive.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         PatchCoordinate mesh{};
         set_patch_coordinate_component(
            mesh, 0, coordinate1_values[i - domain_lo.x]
         );
#if (AMREX_SPACEDIM >= 2)
         set_patch_coordinate_component(
            mesh, 1, coordinate2_values[j - domain_lo.y]
         );
#else
         amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM == 3)
         set_patch_coordinate_component(
            mesh, 2, coordinate3_values[k - domain_lo.z]
         );
#else
         amrex::ignore_unused(k);
#endif
         const InitialDataSamplePoint sample{
            mesh,
            mesh_coordinate_to_physical_point(mesh_coordinate, axes, mesh),
         };
         const FixedSpacetimeGeometry local_geometry = initial_data_spacetime_geometry(
            spacetime,
            sample.physical.q1,
            AMREX_SPACEDIM >= 2 ? sample.physical.q2 : 0.0,
            AMREX_SPACEDIM == 3 ? sample.physical.q3 : 0.0,
            0.0
         );
         const FluidPrimitiveSample point = sample_fluid_initial_data<EOSMethod>(
            parameters, context, sample, local_geometry, initial_data
         );
         cached(i, j, k, RHO) = point.fluid.rho;
         cached(i, j, k, MOM1) = point.fluid.v1;
         cached(i, j, k, MOM2) = point.fluid.v2;
         cached(i, j, k, MOM3) = point.fluid.v3;
         cached(i, j, k, ENERGY) = point.fluid.pressure;
         if (magnetized) {
            cached(i, j, k, B1) = point.b1;
            cached(i, j, k, B2) = point.b2;
            cached(i, j, k, B3) = point.b3;
         }
      });
   }
}

} // namespace

void initialize_relativistic_primitive_cache_from_initial_data(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
      initialize_relativistic_primitive_cache_from_initial_data_method<EOSIdealGas>(
         primitive, geometry, cfg, initial_data
      );
      return;
   }
   initialize_relativistic_primitive_cache_from_initial_data_method<EOSRuntime>(
      primitive, geometry, cfg, initial_data
   );
}

void initialize_relativistic_primitive_cache_from_initial_data(
   amrex::MultiFab& primitive,
   InitialDataEllipticHierarchy const& hierarchy,
   int level,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
      initialize_relativistic_primitive_cache_from_mapped_provider_method<
         EOSIdealGas
      >(primitive, hierarchy, level, cfg, initial_data);
      return;
   }
   initialize_relativistic_primitive_cache_from_mapped_provider_method<
      EOSRuntime
   >(primitive, hierarchy, level, cfg, initial_data);
}

} // namespace pyfluid_amrex
