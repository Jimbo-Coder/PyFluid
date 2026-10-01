#pragma once

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include "pyfluid_amrex/black_hole_matter.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/relativistic_exterior.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RelativisticRecoveryCellEvents {
   int c2p_fallback_count = 0;
   int atmosphere_reset_count = 0;
   int interior_bound_count = 0;
   int interior_reset_count = 0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool limit_relativistic_velocity(
   amrex::Real& v1,
   amrex::Real& v2,
   amrex::Real& v3,
   SpacetimeMetric const& metric,
   amrex::Real maximum_lorentz_factor
) noexcept
{
   const amrex::Real velocity_squared = spatial_vector_norm_squared(
      v1, v2, v3, metric
   );
   const amrex::Real maximum_velocity_squared = 1.0_rt
      - 1.0_rt / (maximum_lorentz_factor * maximum_lorentz_factor);
   if (!(velocity_squared > maximum_velocity_squared)
       || !amrex::Math::isfinite(velocity_squared)) {
      return false;
   }
   const amrex::Real scale = std::sqrt(
      maximum_velocity_squared / velocity_squared
   );
   v1 *= scale;
   v2 *= scale;
   v3 *= scale;
   return true;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool bound_black_hole_primitive(
   RHDPrimitive& primitive,
   EOSParams const& eos,
   SpacetimeMetric const& metric,
   BlackHoleMatterParameters const& parameters
) noexcept
{
   bool changed = limit_relativistic_velocity(
      primitive.v1,
      primitive.v2,
      primitive.v3,
      metric,
      parameters.maximum_lorentz_factor
   );
   if (parameters.maximum_density > 0.0_rt
       && primitive.rho > parameters.maximum_density) {
      primitive.rho = parameters.maximum_density;
      changed = true;
   }
   if (primitive.eps > parameters.maximum_specific_internal_energy) {
      primitive.eps = parameters.maximum_specific_internal_energy;
      changed = true;
   }
   if (changed) {
      primitive.pressure = relativistic_pressure<EOSMethod>(
         primitive.rho, primitive.eps, eos
      );
   }
   return changed;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool bound_black_hole_primitive(
   RMHDPrimitive& primitive,
   EOSParams const& eos,
   SpacetimeMetric const& metric,
   BlackHoleMatterParameters const& parameters
) noexcept
{
   bool changed = limit_relativistic_velocity(
      primitive.v1,
      primitive.v2,
      primitive.v3,
      metric,
      parameters.maximum_lorentz_factor
   );
   if (parameters.maximum_density > 0.0_rt
       && primitive.rho > parameters.maximum_density) {
      primitive.rho = parameters.maximum_density;
      changed = true;
   }
   if (primitive.eps > parameters.maximum_specific_internal_energy) {
      primitive.eps = parameters.maximum_specific_internal_energy;
      changed = true;
   }
   primitive.pressure = relativistic_pressure<EOSMethod>(
      primitive.rho, primitive.eps, eos
   );
   for (int iteration = 0; iteration < 4; ++iteration) {
      const amrex::Real specific_enthalpy = 1.0_rt + primitive.eps
         + primitive.pressure / primitive.rho;
      const amrex::Real magnetic_squared =
         generated::rmhd_comoving_magnetic_squared(primitive, metric);
      const amrex::Real magnetization = magnetic_squared
         / (primitive.rho * specific_enthalpy);
      if (!(magnetization > parameters.maximum_magnetization)
          || !amrex::Math::isfinite(magnetization)) {
         break;
      }
      primitive.rho *= magnetization / parameters.maximum_magnetization;
      primitive.pressure = relativistic_pressure<EOSMethod>(
         primitive.rho, primitive.eps, eos
      );
      changed = true;
   }
   return changed;
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive rhd_exterior_primitive(
   EOSThermodynamicState const& atmosphere
) noexcept
{
   return {
      atmosphere.rho,
      atmosphere.specific_internal_energy,
      atmosphere.pressure,
      0.0_rt,
      0.0_rt,
      0.0_rt,
   };
}

template <PhysicalSystem System, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto recover_relativistic_primitive_with_exterior(
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
               rhd_exterior_primitive<EOSMethod>(atmosphere), metric
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
         if (recovery.dens > 0.0_rt
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
            rhd_exterior_primitive<EOSMethod>(atmosphere);
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
            rhd_exterior_primitive<EOSMethod>(atmosphere);
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
            rhd_exterior_primitive<EOSMethod>(atmosphere);
         const amrex::Real inverse_sqrt_det_gamma = 1.0_rt / metric.sqrt_det_gamma;
         const RMHDPrimitive atmosphere_primitive{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0_rt,
            0.0_rt,
            0.0_rt,
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
         const amrex::Real inverse_sqrt_det_gamma = 1.0_rt / metric.sqrt_det_gamma;
         const RHDPrimitive fluid =
            rhd_exterior_primitive<EOSMethod>(atmosphere);
         const RMHDPrimitive atmosphere_primitive{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0_rt,
            0.0_rt,
            0.0_rt,
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
            rhd_exterior_primitive<EOSMethod>(atmosphere);
         const RMHDPrimitive atmosphere_primitive{
            fluid.rho,
            fluid.eps,
            fluid.pressure,
            0.0_rt,
            0.0_rt,
            0.0_rt,
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

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDRecoveryResult recover_rhd_with_vacuum_exterior(
   amrex::Array4<amrex::Real> const& state,
   int i,
   int j,
   int k,
   EOSParams const& eos,
   SpacetimeMetric const& metric,
   RelativisticRecoveryCellEvents* events = nullptr
) noexcept
{
   RHDRecoveryResult recovered = recover_rhd<EOSMethod>(
      load_rhd_conserved(state, i, j, k), metric, eos
   );
   if (recovered.attempt.succeeded()) {
      return recovered;
   }
   store_rhd_conserved(state, i, j, k, RHDConserved{});
   recovered = recover_rhd<EOSMethod>(
      load_rhd_conserved(state, i, j, k), metric, eos
   );
   if (events != nullptr) {
      events->atmosphere_reset_count = 1;
   }
   return recovered;
}

template <PhysicalSystem System, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto recover_relativistic_primitive_with_exterior(
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
   return recover_relativistic_primitive_with_exterior<System, EOSMethod>(
      state, i, j, k, eos, metric, atmosphere
   );
}

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
inline void project_relativistic_exterior(
   amrex::MultiFab& state,
   EOSParams const& eos,
   RelativisticExteriorState const& exterior,
   MetricAccessorFactory&& metric_accessor_factory,
   int ghost_cells = 0,
   amrex::Long* c2p_fallback_count = nullptr,
   amrex::Long* atmosphere_reset_count = nullptr,
   amrex::iMultiFab const* black_hole_mask = nullptr,
   BlackHoleMatterParameters black_hole_matter = {},
   amrex::Long* interior_bound_count = nullptr,
   amrex::Long* interior_reset_count = nullptr
)
{
   static_assert(System == PhysicalSystem::RHD || System == PhysicalSystem::RMHD);
   const bool vacuum_exterior = exterior.is_vacuum();
   const EOSThermodynamicState thermodynamic_exterior = exterior;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Box box = mfi.growntilebox(ghost_cells);
      box &= state[mfi].box();
      const amrex::Box valid_box = mfi.tilebox();
      const auto values = state.array(mfi);
      const auto metric_at = metric_accessor_factory(mfi);
      amrex::Array4<int const> interior{};
      if (black_hole_mask != nullptr) {
         interior = black_hole_mask->const_array(mfi);
      }
      const bool use_interior = black_hole_mask != nullptr;
      amrex::ParallelFor(box,
         [metric_at, use_interior, interior, black_hole_matter, values,
          thermodynamic_exterior, vacuum_exterior, eos, valid_box,
          c2p_fallback_count, atmosphere_reset_count, interior_bound_count,
          interior_reset_count]
         AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         RelativisticRecoveryCellEvents events;
         const SpacetimeMetric metric = metric_at(i, j, k);
         const bool in_black_hole = use_interior
            && interior(i, j, k, 0) != 0;
         if (in_black_hole
             && black_hole_matter.treatment
                == BlackHoleMatterTreatment::Excision) {
            if constexpr (System == PhysicalSystem::RHD) {
               store_rhd_conserved(
                  values,
                  i,
                  j,
                  k,
                  generated::rhd_primitive_to_conserved(
                     rhd_exterior_primitive<EOSMethod>(thermodynamic_exterior),
                     metric
                  )
               );
            } else {
               const RMHDConserved current = load_rmhd_conserved(
                  values, i, j, k
               );
               const amrex::Real inverse_sqrt_det_gamma =
                  1.0_rt / metric.sqrt_det_gamma;
               const RHDPrimitive fluid =
                  rhd_exterior_primitive<EOSMethod>(thermodynamic_exterior);
               const RMHDPrimitive interior_primitive{
                  fluid.rho,
                  fluid.eps,
                  fluid.pressure,
                  0.0_rt,
                  0.0_rt,
                  0.0_rt,
                  amrex::Math::isfinite(current.b1)
                     ? current.b1 * inverse_sqrt_det_gamma : 0.0_rt,
                  amrex::Math::isfinite(current.b2)
                     ? current.b2 * inverse_sqrt_det_gamma : 0.0_rt,
                  amrex::Math::isfinite(current.b3)
                     ? current.b3 * inverse_sqrt_det_gamma : 0.0_rt,
               };
               store_rmhd_conserved(
                  values,
                  i,
                  j,
                  k,
                  generated::rmhd_primitive_to_conserved(
                     interior_primitive, metric
                  )
               );
            }
            return;
         }
         auto recovered = [&]() {
            if constexpr (System == PhysicalSystem::RHD) {
               if (vacuum_exterior) {
                  return recover_rhd_with_vacuum_exterior<EOSMethod>(
                     values, i, j, k, eos, metric, &events
                  );
               }
            }
            return recover_relativistic_primitive_with_exterior<
               System,
               EOSMethod
            >(
               values,
               i,
               j,
               k,
               eos,
               metric,
               thermodynamic_exterior,
               &events
            );
         }();
         if (in_black_hole
             && black_hole_matter.treatment
                == BlackHoleMatterTreatment::Bounded) {
            if (events.atmosphere_reset_count != 0) {
               events.interior_reset_count = events.atmosphere_reset_count;
               events.atmosphere_reset_count = 0;
            }
            if constexpr (System == PhysicalSystem::RHD) {
               RHDPrimitive primitive = recovered.primitive;
               if (recovered.attempt.succeeded()
                   && bound_black_hole_primitive<EOSMethod>(
                      primitive, eos, metric, black_hole_matter
                   )) {
                  store_rhd_conserved(
                     values,
                     i,
                     j,
                     k,
                     generated::rhd_primitive_to_conserved(primitive, metric)
                  );
                  events.interior_bound_count = 1;
               }
            } else {
               RMHDPrimitive primitive = recovered.primitive;
               if (recovered.succeeded()
                   && bound_black_hole_primitive<EOSMethod>(
                      primitive, eos, metric, black_hole_matter
                   )) {
                  store_rmhd_conserved(
                     values,
                     i,
                     j,
                     k,
                     generated::rmhd_primitive_to_conserved(primitive, metric)
                  );
                  events.interior_bound_count = 1;
               }
            }
         }
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
            if (interior_bound_count != nullptr
                && events.interior_bound_count != 0) {
               amrex::HostDevice::Atomic::Add(
                  interior_bound_count,
                  static_cast<amrex::Long>(events.interior_bound_count)
               );
            }
            if (interior_reset_count != nullptr
                && events.interior_reset_count != 0) {
               amrex::HostDevice::Atomic::Add(
                  interior_reset_count,
                  static_cast<amrex::Long>(events.interior_reset_count)
               );
            }
         }
      });
   }
}

} // namespace pyfluid_amrex
