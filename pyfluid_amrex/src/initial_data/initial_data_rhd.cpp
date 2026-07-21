#include "pyfluid_amrex/initial_data/rhd.hpp"

namespace pyfluid_amrex {
namespace {

template <int EOSMethod>
void initialize_relativistic_primitive_cache_from_provider_method(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   if (!initial_data.has_fluid()) {
      amrex::Abort("relativistic primitive initialization requires a fluid provider");
   }
   const auto problo = geometry.ProbLoArray();
   const auto probhi = geometry.ProbHiArray();
   const auto dx = geometry.CellSizeArray();
   const MeshAxes axes = mesh_axes_layout(cfg);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
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
   const FluidInitialDataParameters parameters = fluid_initial_data_parameters(
      cfg, initial_data
   );
   const bool magnetized = physical_system_has_magnetic_field(physical_system(cfg));
   primitive.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(primitive, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.array(mfi);
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

void initialize_relativistic_primitive_cache_from_provider(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   RelativisticInitialDataView const& initial_data
)
{
   if (physical_system(cfg) == PhysicalSystem::RMHD) {
      initialize_relativistic_primitive_cache_from_provider_method<EOSIdealGas>(
         primitive, geometry, cfg, initial_data
      );
      return;
   }
   const int eos_method = eos_method_code(cfg);
   if (eos_method == EOSPolytrope) {
      initialize_relativistic_primitive_cache_from_provider_method<EOSPolytrope>(
         primitive, geometry, cfg, initial_data
      );
   } else if (eos_method == EOSHybridPolytrope) {
      initialize_relativistic_primitive_cache_from_provider_method<EOSHybridPolytrope>(
         primitive, geometry, cfg, initial_data
      );
   } else {
      initialize_relativistic_primitive_cache_from_provider_method<EOSIdealGas>(
         primitive, geometry, cfg, initial_data
      );
   }
}

} // namespace pyfluid_amrex
