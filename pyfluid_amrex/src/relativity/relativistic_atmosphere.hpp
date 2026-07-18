#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_atmosphere_primitive(EOSParams const& eos) noexcept
{
   const amrex::Real rho = eos.rho_floor;
   const amrex::Real pressure = amrex::max(
      eos.pressure_floor, eos_cold_pressure<EOSMethod>(rho, eos)
   );
   const amrex::Real internal_energy_density =
      eos_internal_energy_density_from_pressure<EOSMethod>(rho, pressure, eos);
   return {
      rho,
      internal_energy_density / rho,
      eos_pressure_from_density_energy<EOSMethod>(rho, internal_energy_density, eos),
      0.0,
      0.0,
      0.0,
   };
}

template <PhysicalSystem System, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto recover_relativistic_primitive_with_atmosphere(
   amrex::Array4<amrex::Real> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   SpacetimeMetric const& metric
) noexcept
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   if constexpr (System == PhysicalSystem::RHD) {
      RHDRecoveryResult recovered = recover_rhd<EOSMethod>(
         load_rhd_conserved(state, i, j, k), metric, eos
      );
      if (recovered.attempt.succeeded()
          && recovered.primitive.rho < eos.rho_floor) {
         store_rhd_conserved(
            state,
            i,
            j,
            k,
            generated::rhd_primitive_to_conserved(
               rhd_atmosphere_primitive<EOSMethod>(eos), metric
            )
         );
         recovered.primitive = rhd_atmosphere_primitive<EOSMethod>(eos);
      }
      return recovered;
   } else {
      static_assert(EOSMethod == EOSIdealGas);
      RMHDRecoveryResult recovered = recover_srmhd(
         load_rmhd_conserved(state, i, j, k), metric, eos.gamma
      );
      if (recovered.succeeded()
          && recovered.primitive.rho < eos.rho_floor) {
         const RHDPrimitive fluid = rhd_atmosphere_primitive<EOSIdealGas>(eos);
         const RMHDPrimitive atmosphere{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0,
            0.0,
            0.0,
            recovered.primitive.b1,
            recovered.primitive.b2,
            recovered.primitive.b3,
         };
         store_rmhd_conserved(
            state,
            i,
            j,
            k,
            generated::rmhd_primitive_to_conserved(atmosphere, metric)
         );
         recovered.primitive = atmosphere;
      }
      return recovered;
   }
}

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
inline void project_relativistic_atmosphere(
   amrex::MultiFab& state,
   EOSParams const& eos,
   MetricAccessorFactory&& metric_accessor_factory,
   int ghost_cells = 0
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box box = amrex::grow(mfi.tilebox(), ghost_cells);
      box &= state[mfi].box();
      const auto values = state.array(mfi);
      const auto metric_at = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         static_cast<void>(recover_relativistic_primitive_with_atmosphere<System, EOSMethod>(
            values, i, j, k, eos, metric_at(i, j, k)
         ));
      });
   }
}

} // namespace pyfluid_amrex
