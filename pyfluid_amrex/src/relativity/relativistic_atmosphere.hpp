#pragma once

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct RelativisticRecoveryCellEvents {
   int c2p_fallback_count = 0;
   int atmosphere_reset_count = 0;
};

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_atmosphere_primitive(
   EOSThermodynamicState const& atmosphere
) noexcept
{
   return {
      atmosphere.rho,
      atmosphere.specific_internal_energy,
      atmosphere.pressure,
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
   SpacetimeMetric const& metric,
   EOSThermodynamicState const& atmosphere,
   RelativisticRecoveryCellEvents* events = nullptr
) noexcept
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   if constexpr (System == PhysicalSystem::RHD) {
      const RHDConserved conserved = load_rhd_conserved(state, i, j, k);
      if (std::isfinite(conserved.dens)
          && conserved.dens <= metric.sqrt_det_gamma * atmosphere.rho) {
         store_rhd_conserved(
            state,
            i,
            j,
            k,
            generated::rhd_primitive_to_conserved(
               rhd_atmosphere_primitive<EOSMethod>(atmosphere), metric
            )
         );
      }
      RHDConserved floor_candidate = load_rhd_conserved(state, i, j, k);
      if (finite_rhd_conserved(floor_candidate)) {
         const RHDRecoveryState recovery = generated::rhd_recovery_state(
            floor_candidate, metric
         );
         const amrex::Real admissibility_margin =
            rhd_conserved_admissibility_margin(floor_candidate, metric);
         if (recovery.dens > 0.0
             && amrex::Math::isfinite(admissibility_margin)
             && admissibility_margin < atmosphere.pressure) {
            floor_candidate.tau += metric.sqrt_det_gamma
               * (atmosphere.pressure - admissibility_margin);
            store_rhd_conserved(state, i, j, k, floor_candidate);
         }
      }
      RHDRecoveryResult recovered = recover_rhd<EOSMethod>(
         load_rhd_conserved(state, i, j, k), metric, eos
      );
      if (!recovered.attempt.succeeded()) {
         const RHDPrimitive atmosphere_primitive =
            rhd_atmosphere_primitive<EOSMethod>(atmosphere);
         store_rhd_conserved(
            state,
            i,
            j,
            k,
            generated::rhd_primitive_to_conserved(atmosphere_primitive, metric)
         );
         recovered = recover_rhd<EOSMethod>(
            load_rhd_conserved(state, i, j, k), metric, eos
         );
         if (events != nullptr) {
            events->atmosphere_reset_count = 1;
         }
      }
      if (recovered.attempt.succeeded()
          && recovered.primitive.rho < atmosphere.rho) {
         const RHDPrimitive atmosphere_primitive =
            rhd_atmosphere_primitive<EOSMethod>(atmosphere);
         store_rhd_conserved(
            state,
            i,
            j,
            k,
            generated::rhd_primitive_to_conserved(atmosphere_primitive, metric)
         );
         recovered.primitive = atmosphere_primitive;
      }
      return recovered;
   } else {
      const RMHDConserved conserved = load_rmhd_conserved(state, i, j, k);
      if (std::isfinite(conserved.dens)
          && conserved.dens <= metric.sqrt_det_gamma * atmosphere.rho) {
         const RHDPrimitive fluid =
            rhd_atmosphere_primitive<EOSMethod>(atmosphere);
         const amrex::Real inverse_sqrt_det_gamma = 1.0 / metric.sqrt_det_gamma;
         const RMHDPrimitive atmosphere_primitive{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0,
            0.0,
            0.0,
            conserved.b1 * inverse_sqrt_det_gamma,
            conserved.b2 * inverse_sqrt_det_gamma,
            conserved.b3 * inverse_sqrt_det_gamma,
         };
         store_rmhd_conserved(
            state,
            i,
            j,
            k,
            generated::rmhd_primitive_to_conserved(atmosphere_primitive, metric)
         );
      }
      RMHDRecoveryResult recovered = recover_srmhd(
         load_rmhd_conserved(state, i, j, k), metric, eos
      );
      const bool finite_magnetic_state =
         amrex::Math::isfinite(conserved.b1)
         && amrex::Math::isfinite(conserved.b2)
         && amrex::Math::isfinite(conserved.b3);
      if (!recovered.succeeded() && finite_magnetic_state) {
         const RMHDConserved failed = load_rmhd_conserved(state, i, j, k);
         const amrex::Real inverse_sqrt_det_gamma = 1.0 / metric.sqrt_det_gamma;
         const RHDPrimitive fluid =
            rhd_atmosphere_primitive<EOSMethod>(atmosphere);
         const RMHDPrimitive atmosphere_primitive{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0,
            0.0,
            0.0,
            failed.b1 * inverse_sqrt_det_gamma,
            failed.b2 * inverse_sqrt_det_gamma,
            failed.b3 * inverse_sqrt_det_gamma,
         };
         store_rmhd_conserved(
            state,
            i,
            j,
            k,
            generated::rmhd_primitive_to_conserved(atmosphere_primitive, metric)
         );
         recovered = recover_srmhd(
            load_rmhd_conserved(state, i, j, k), metric, eos
         );
         if (events != nullptr) {
            events->atmosphere_reset_count = 1;
         }
      } else if (events != nullptr && recovered.attempt_count > 1) {
         events->c2p_fallback_count = 1;
      }
      if (recovered.succeeded()
          && recovered.primitive.rho < atmosphere.rho) {
         const RHDPrimitive fluid =
            rhd_atmosphere_primitive<EOSMethod>(atmosphere);
         const RMHDPrimitive atmosphere_primitive{
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
            generated::rmhd_primitive_to_conserved(atmosphere_primitive, metric)
         );
         recovered.primitive = atmosphere_primitive;
      }
      return recovered;
   }
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
   const EOSThermodynamicState atmosphere = eos_project_to_floors<EOSMethod>(
      eos.rho_floor, eos.pressure_floor, eos
   );
   return recover_relativistic_primitive_with_atmosphere<System, EOSMethod>(
      state, i, j, k, eos, metric, atmosphere
   );
}

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
inline void project_relativistic_atmosphere(
   amrex::MultiFab& state,
   EOSParams const& eos,
   EOSThermodynamicState const& atmosphere,
   MetricAccessorFactory&& metric_accessor_factory,
   int ghost_cells = 0,
   amrex::Long* c2p_fallback_count = nullptr,
   amrex::Long* atmosphere_reset_count = nullptr
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box box = mfi.growntilebox(ghost_cells);
      box &= state[mfi].box();
      const amrex::Box valid_box = mfi.tilebox();
      const auto values = state.array(mfi);
      const auto metric_at = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RelativisticRecoveryCellEvents events;
         static_cast<void>(
            recover_relativistic_primitive_with_atmosphere<System, EOSMethod>(
               values, i, j, k, eos, metric_at(i, j, k), atmosphere, &events
            )
         );
         if (valid_box.contains(i, j, k)) {
            if (c2p_fallback_count != nullptr
                && events.c2p_fallback_count != 0) {
               amrex::HostDevice::Atomic::Add(
                  c2p_fallback_count,
                  static_cast<amrex::Long>(events.c2p_fallback_count)
               );
            }
            if (atmosphere_reset_count != nullptr
                && events.atmosphere_reset_count != 0) {
               amrex::HostDevice::Atomic::Add(
                  atmosphere_reset_count,
                  static_cast<amrex::Long>(events.atmosphere_reset_count)
               );
            }
         }
      });
   }
}

} // namespace pyfluid_amrex
