#include "amr/amr_solver_base.hpp"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Reduce.H>

#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace {

// Detailed host formatting runs only after a failed bulk check. Copy that
// FAB explicitly instead of dereferencing device storage in BoxIterator.
amrex::FArrayBox copy_fab_for_debug(amrex::FArrayBox const& source, int num_components)
{
   amrex::FArrayBox host(source.box(), num_components, amrex::The_Pinned_Arena());
   amrex::Gpu::dtoh_memcpy(host.dataPtr(), source.dataPtr(), host.nBytes());
   return host;
}

// Recovery and reconversion include metric contractions and normalization.
// Allow their accumulated roundoff independently of the physical error
// budgets. The FP64 acceptance bounds remain unchanged.
constexpr amrex::Real conversion_roundoff =
   64.0_rt * std::numeric_limits<amrex::Real>::epsilon();
constexpr amrex::Real preflight_conserved_round_trip_tolerance =
   amrex::max(1.0e-7_rt, conversion_roundoff);
// Near a cold, low-density atmosphere the primitive variables are more
// ill-conditioned than the conserved state.  Require a tight conserved
// round trip, while allowing the two successful C2P inversions to differ at
// their repeatability bound or the arithmetic resolution, whichever is larger.
constexpr amrex::Real preflight_primitive_round_trip_tolerance =
   amrex::max(1.0e-6_rt, conversion_roundoff);

// Accepted states must be physical and reproduce the conserved state.
// Repeated primitive inversions near a cold admissibility boundary are
// ill-conditioned; their stricter repeatability check belongs to IC preflight.
enum class RelativisticCheck {
   Recovery,
   ConservedRoundTrip,
   InitialDataRoundTrip,
};

struct RelativisticRoundTripResult {
   int failure_reason = 0;
   amrex::Real conserved_error = 0.0_rt;
   amrex::Real primitive_error = 0.0_rt;
   int nonfinite_conserved = 0;
   int nonfinite_metric_volume = 0;
   int nonpositive_density = 0;
   int nonpositive_energy = 0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real preflight_scaled_error(
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real scale_floor
) noexcept
{
   if (!amrex::Math::isfinite(actual) || !amrex::Math::isfinite(expected)) {
      return std::numeric_limits<amrex::Real>::infinity();
   }
   const amrex::Real scale = amrex::max(
      scale_floor,
      amrex::max(amrex::Math::abs(actual), amrex::Math::abs(expected))
   );
   return amrex::Math::abs(actual - expected) / scale;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rhd_conserved_round_trip_error(
   RHDConserved const& actual,
   RHDConserved const& expected,
   amrex::Real scale_floor
) noexcept
{
   amrex::Real error = preflight_scaled_error(
      actual.dens, expected.dens, scale_floor
   );
   // Momentum is one covector. Lowering a recovered velocity mixes its
   // components, so cancellation in a nearly zero transverse component must
   // be measured against the whole vector, not that component's tiny value.
   // Keep density and energy independently scaled to retain cold-state checks.
   const amrex::GpuArray<amrex::Real, 6> momentum_components{
      actual.mom1, actual.mom2, actual.mom3,
      expected.mom1, expected.mom2, expected.mom3,
   };
   amrex::Real momentum_scale = scale_floor;
   for (amrex::Real component : momentum_components) {
      momentum_scale = amrex::max(momentum_scale, amrex::Math::abs(component));
   }
   error = amrex::max(error, preflight_scaled_error(
      actual.mom1, expected.mom1, momentum_scale
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.mom2, expected.mom2, momentum_scale
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.mom3, expected.mom3, momentum_scale
   ));
   return amrex::max(error, preflight_scaled_error(
      actual.tau, expected.tau, scale_floor
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rhd_primitive_round_trip_error(
   RHDPrimitive const& actual,
   RHDPrimitive const& expected,
   EOSParams const& eos
) noexcept
{
   amrex::Real error = preflight_scaled_error(
      actual.rho, expected.rho, eos.rho_floor
   );
   error = amrex::max(error, preflight_scaled_error(
      actual.eps, expected.eps, 1.0e-14_rt
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.pressure, expected.pressure, eos.pressure_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.v1, expected.v1, 1.0_rt
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.v2, expected.v2, 1.0_rt
   ));
   return amrex::max(error, preflight_scaled_error(
      actual.v3, expected.v3, 1.0_rt
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_conserved_round_trip_error(
   RMHDConserved const& actual,
   RMHDConserved const& expected,
   amrex::Real scale_floor
) noexcept
{
   amrex::Real error = rhd_conserved_round_trip_error(
      {actual.dens, actual.mom1, actual.mom2, actual.mom3, actual.tau},
      {expected.dens, expected.mom1, expected.mom2, expected.mom3, expected.tau},
      scale_floor
   );
   error = amrex::max(error, preflight_scaled_error(
      actual.b1, expected.b1, scale_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.b2, expected.b2, scale_floor
   ));
   return amrex::max(error, preflight_scaled_error(
      actual.b3, expected.b3, scale_floor
   ));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rmhd_primitive_round_trip_error(
   RMHDPrimitive const& actual,
   RMHDPrimitive const& expected,
   EOSParams const& eos
) noexcept
{
   amrex::Real error = rhd_primitive_round_trip_error(
      {
         actual.rho,
         actual.eps,
         actual.pressure,
         actual.v1,
         actual.v2,
         actual.v3,
      },
      {
         expected.rho,
         expected.eps,
         expected.pressure,
         expected.v1,
         expected.v2,
         expected.v3,
      },
      eos
   );
   error = amrex::max(error, preflight_scaled_error(
      actual.b1, expected.b1, eos.rho_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.b2, expected.b2, eos.rho_floor
   ));
   return amrex::max(error, preflight_scaled_error(
      actual.b3, expected.b3, eos.rho_floor
   ));
}

template <int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RelativisticRoundTripResult evaluate_rhd_round_trip(
   RHDConserved const& initial_conserved,
   SpacetimeMetric const& metric,
   EOSParams const& eos,
   RelativisticCheck check
) noexcept
{
   RelativisticRoundTripResult result;
   const RHDRecoveryResult initial_recovery = recover_rhd<EOSMethod>(
      initial_conserved, metric, eos
   );
   if (!initial_recovery.attempt.succeeded()) {
      result.failure_reason = 10
         + static_cast<int>(initial_recovery.attempt.status);
      const RHDRecoveryState recovery = generated::rhd_recovery_state(
         initial_conserved, metric
      );
      result.nonfinite_conserved = finite_rhd_conserved(initial_conserved)
         ? 0 : 1;
      result.nonfinite_metric_volume =
         amrex::Math::isfinite(metric.sqrt_det_gamma) ? 0 : 1;
      result.nonpositive_density = recovery.dens > 0.0_rt ? 0 : 1;
      result.nonpositive_energy = recovery.dens + recovery.tau > 0.0_rt ? 0 : 1;
      return result;
   }

   const RHDConserved regenerated_conserved =
      generated::rhd_primitive_to_conserved(initial_recovery.primitive, metric);
   const RHDRecoveryResult regenerated_recovery = recover_rhd<EOSMethod>(
      regenerated_conserved, metric, eos
   );
   if (!regenerated_recovery.attempt.succeeded()) {
      result.failure_reason = 20
         + static_cast<int>(regenerated_recovery.attempt.status);
      return result;
   }

   const amrex::Real conserved_scale_floor = amrex::max(
      1.0e-30_rt, eos.rho_floor * 1.0e-6_rt
   );
   result.conserved_error = rhd_conserved_round_trip_error(
      regenerated_conserved, initial_conserved, conserved_scale_floor
   );
   result.primitive_error = rhd_primitive_round_trip_error(
      regenerated_recovery.primitive, initial_recovery.primitive, eos
   );
   if (check != RelativisticCheck::Recovery
       && result.conserved_error > preflight_conserved_round_trip_tolerance) {
      result.failure_reason = 30;
   }
   if (check == RelativisticCheck::InitialDataRoundTrip
       && result.primitive_error > preflight_primitive_round_trip_tolerance) {
      result.failure_reason = 31;
   }
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RelativisticRoundTripResult evaluate_rmhd_round_trip(
   RMHDConserved const& initial_conserved,
   SpacetimeMetric const& metric,
   EOSParams const& eos,
   RelativisticCheck check
) noexcept
{
   RelativisticRoundTripResult result;
   const RMHDRecoveryResult initial_recovery = recover_srmhd(
      initial_conserved, metric, eos
   );
   if (!initial_recovery.succeeded()) {
      result.failure_reason = 10
         + static_cast<int>(initial_recovery.attempt.status);
      const RMHDRecoveryState recovery = generated::rmhd_recovery_state(
         initial_conserved, metric
      );
      result.nonfinite_conserved = finite_rmhd_conserved(initial_conserved)
         ? 0 : 1;
      result.nonfinite_metric_volume =
         amrex::Math::isfinite(metric.sqrt_det_gamma) ? 0 : 1;
      result.nonpositive_density = recovery.dens > 0.0_rt ? 0 : 1;
      result.nonpositive_energy = recovery.dens + recovery.tau > 0.0_rt ? 0 : 1;
      return result;
   }

   const RMHDConserved regenerated_conserved =
      generated::rmhd_primitive_to_conserved(initial_recovery.primitive, metric);
   const RMHDRecoveryResult regenerated_recovery = recover_srmhd(
      regenerated_conserved, metric, eos
   );
   if (!regenerated_recovery.succeeded()) {
      result.failure_reason = 20
         + static_cast<int>(regenerated_recovery.attempt.status);
      return result;
   }

   const amrex::Real conserved_scale_floor = amrex::max(
      1.0e-30_rt, eos.rho_floor * 1.0e-6_rt
   );
   result.conserved_error = rmhd_conserved_round_trip_error(
      regenerated_conserved, initial_conserved, conserved_scale_floor
   );
   result.primitive_error = rmhd_primitive_round_trip_error(
      regenerated_recovery.primitive, initial_recovery.primitive, eos
   );
   if (check != RelativisticCheck::Recovery
       && result.conserved_error > preflight_conserved_round_trip_tolerance) {
      result.failure_reason = 30;
   }
   if (check == RelativisticCheck::InitialDataRoundTrip
       && result.primitive_error > preflight_primitive_round_trip_tolerance) {
      result.failure_reason = 31;
   }
   return result;
}

struct PhysicalFailure {
   int found = 0;
   int i = std::numeric_limits<int>::min();
   int j = std::numeric_limits<int>::min();
   int k = std::numeric_limits<int>::min();
   amrex::Real rho = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real pressure = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::GpuArray<amrex::Real, NVAR_HYDRO> conserved{};
};

struct RelativisticPreflightSummary {
   amrex::Long cells = 0;
   amrex::Long failures = 0;
   int failure_reason = 0;
   amrex::Real conserved_error = 0.0_rt;
   amrex::Real primitive_error = 0.0_rt;
   amrex::Real minimum_density = std::numeric_limits<amrex::Real>::max();
   amrex::Real minimum_energy = std::numeric_limits<amrex::Real>::max();
   amrex::Real minimum_admissibility_margin =
      std::numeric_limits<amrex::Real>::max();
   amrex::Long nonfinite_conserved = 0;
   amrex::Long nonfinite_metric_volume = 0;
   amrex::Long nonpositive_density = 0;
   amrex::Long nonpositive_energy = 0;
};

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
RelativisticPreflightSummary evaluate_relativistic_preflight_level(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   EOSParams const& eos,
   RelativisticCheck check,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   amrex::MultiFab result(
      state.boxArray(), state.DistributionMap(), 12, 0
   );
   result.setVal(0.0_rt);
   result.setVal(std::numeric_limits<amrex::Real>::max(), 5, 3);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(result, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.const_array(mfi);
      const auto valid = valid_mask.const_array(mfi);
      const auto values = result.array(mfi);
      const auto metric_at = metric_accessor_factory(mfi);
      amrex::ParallelFor(box, [valid, conserved, metric_at, eos, check, values] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (valid(i, j, k, 0) == 0) {
            return;
         }
         RelativisticRoundTripResult cell_result;
         if constexpr (System == PhysicalSystem::RHD) {
            const RHDConserved cell_conserved = load_rhd_conserved(
               conserved, i, j, k
            );
            const SpacetimeMetric cell_metric = metric_at(i, j, k);
            cell_result = evaluate_rhd_round_trip<EOSMethod>(
               cell_conserved, cell_metric, eos, check
            );
            if (cell_result.failure_reason != 0) {
               const RHDRecoveryState recovery = generated::rhd_recovery_state(
                  cell_conserved, cell_metric
               );
               values(i, j, k, 5) = recovery.dens;
               values(i, j, k, 6) = recovery.dens + recovery.tau;
               values(i, j, k, 7) = rhd_conserved_admissibility_margin(
                  cell_conserved, cell_metric
               );
            }
         } else {
            cell_result = evaluate_rmhd_round_trip(
               load_rmhd_conserved(conserved, i, j, k), metric_at(i, j, k), eos,
               check
            );
         }
         values(i, j, k, 0) = cell_result.failure_reason == 0 ? 0.0_rt : 1.0_rt;
         values(i, j, k, 1) = cell_result.failure_reason;
         values(i, j, k, 2) = cell_result.conserved_error;
         values(i, j, k, 3) = cell_result.primitive_error;
         values(i, j, k, 4) = 1.0_rt;
         values(i, j, k, 8) = cell_result.nonfinite_conserved;
         values(i, j, k, 9) = cell_result.nonfinite_metric_volume;
         values(i, j, k, 10) = cell_result.nonpositive_density;
         values(i, j, k, 11) = cell_result.nonpositive_energy;
      });
   }

   RelativisticPreflightSummary summary;
   summary.failures = static_cast<amrex::Long>(result.sum(0));
   summary.failure_reason = static_cast<int>(result.max(1));
   summary.conserved_error = result.max(2);
   summary.primitive_error = result.max(3);
   summary.cells = static_cast<amrex::Long>(result.sum(4));
   summary.minimum_density = result.min(5);
   summary.minimum_energy = result.min(6);
   summary.minimum_admissibility_margin = result.min(7);
   summary.nonfinite_conserved = static_cast<amrex::Long>(result.sum(8));
   summary.nonfinite_metric_volume = static_cast<amrex::Long>(result.sum(9));
   summary.nonpositive_density = static_cast<amrex::Long>(result.sum(10));
   summary.nonpositive_energy = static_cast<amrex::Long>(result.sum(11));
   return summary;
}

template <PhysicalSystem System, int EOSMethod>
RelativisticPreflightSummary evaluate_relativistic_preflight_level(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   amrex::MultiFab const* cell_spacetime,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   EOSParams const& eos,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes,
   RelativisticCheck check
)
{
   if (cell_spacetime != nullptr) {
      return evaluate_relativistic_preflight_level<System, EOSMethod>(
         state,
         valid_mask,
         eos,
         check,
         StoredSpacetimeCellMetricFactory{*cell_spacetime}
      );
   }
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   return evaluate_relativistic_preflight_level<System, EOSMethod>(
      state,
      valid_mask,
      eos,
      check,
      FixedSpacetimeCellMetricFactory{
         {fixed_spacetime, lower, spacing, coordinate, axes, time}
      }
   );
}

RelativisticPreflightSummary evaluate_relativistic_preflight_level_runtime(
   PhysicalSystem system,
   int eos_method,
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   amrex::MultiFab const* cell_spacetime,
   amrex::Geometry const& geometry,
   FixedSpacetimeParameters const& fixed_spacetime,
   EOSParams const& eos,
   amrex::Real time,
   int coordinate,
   MeshAxes const& axes,
   RelativisticCheck check
)
{
   if (system == PhysicalSystem::RMHD) {
      return evaluate_relativistic_preflight_level<
         PhysicalSystem::RMHD, EOSRuntime
      >(state, valid_mask, cell_spacetime, geometry, fixed_spacetime, eos, time,
        coordinate, axes, check);
   }
   static_cast<void>(eos_method);
   return evaluate_relativistic_preflight_level<
      PhysicalSystem::RHD, EOSRuntime
   >(state, valid_mask, cell_spacetime, geometry, fixed_spacetime, eos, time,
     coordinate, axes, check);
}

} // namespace

bool AmrSolverBase::debug_contracts_enabled() const
{
   return cfg_.debug_poison || cfg_.debug_check_finite || cfg_.debug_check_physical || cfg_.physical_check_int > 0;
}

bool AmrSolverBase::accepted_physical_check_due(int step, bool force) const
{
   if (cfg_.debug_check_physical || force) {
      return true;
   }
   if (cfg_.physical_check_int <= 0) {
      return false;
   }
   return step % cfg_.physical_check_int == 0;
}

void AmrSolverBase::debug_poison_mfab(amrex::MultiFab& mf) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   mf.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
}

void AmrSolverBase::debug_poison_leveldata(LevelData& data, int lev_begin, int lev_end) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      if (data[lev]) {
         debug_poison_mfab(*data[lev]);
      }
   }
}

void AmrSolverBase::debug_poison_fluxes(int lev) const
{
   if (!cfg_.debug_poison) {
      return;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (fluxes_[lev][dir]) {
         debug_poison_mfab(*fluxes_[lev][dir]);
      }
   }
}

void AmrSolverBase::debug_check_mfab_finite(amrex::MultiFab const& mf, int ncomp, amrex::IntVect const& ngrow,
                                            char const* context) const
{
   if (!cfg_.debug_check_finite) {
      return;
   }
   if (mf.is_finite(0, ncomp, ngrow, false)) {
      return;
   }
   int first_i = std::numeric_limits<int>::min();
   int first_j = std::numeric_limits<int>::min();
   int first_k = std::numeric_limits<int>::min();
   int first_n = -1;
   amrex::Real first_value = std::numeric_limits<amrex::Real>::quiet_NaN();
   for (amrex::MFIter mfi(mf); mfi.isValid() && first_n < 0; ++mfi) {
      amrex::Box bx = amrex::grow(mfi.validbox(), ngrow);
      bx &= mfi.fabbox();
      const auto host = copy_fab_for_debug(mf[mfi], ncomp);
      const auto values = host.const_array();
      for (amrex::BoxIterator bit(bx); bit.ok() && first_n < 0; ++bit) {
         const amrex::IntVect cell = bit();
         const int i = cell[0];
         int j = 0;
         int k = 0;
#if (AMREX_SPACEDIM >= 2)
         j = cell[1];
#endif
#if (AMREX_SPACEDIM == 3)
         k = cell[2];
#endif
         for (int n = 0; n < ncomp; ++n) {
            const amrex::Real value = values(i, j, k, n);
            if (!std::isfinite(value)) {
               first_i = i;
               first_j = j;
               first_k = k;
               first_n = n;
               first_value = value;
               break;
            }
         }
      }
   }
   std::ostringstream message;
   message << "Debug finite contract failed in " << context << ": detected NaN/Inf in " << ncomp << " component(s)"
           << "; first_bad=(" << first_i << "," << first_j << "," << first_k << ") component=" << first_n
           << " value=" << first_value << ".";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::debug_check_reconstruction_input(amrex::MultiFab const& state, int lev, amrex::Real time,
                                                     char const* context) const
{
   if (!cfg_.debug_check_finite) {
      return;
   }
   debug_check_mfab_finite(state, num_state_components(), state.nGrowVect(), context);
   const amrex::Real min_rho = state.min(RHO, state.nGrow());
   if (min_rho > 0.0_rt) {
      return;
   }
   const bool allows_vacuum = physics_system_ == PhysicalSystem::RHD
      && recovery_exterior_state_.is_vacuum();
   int first_i = std::numeric_limits<int>::min();
   int first_j = std::numeric_limits<int>::min();
   int first_k = std::numeric_limits<int>::min();
   amrex::Real first_rho = std::numeric_limits<amrex::Real>::quiet_NaN();
   std::array<amrex::Real, MaxStateComponents> first_state{};
   first_state.fill(std::numeric_limits<amrex::Real>::quiet_NaN());
   for (amrex::MFIter mfi(state); mfi.isValid() && !std::isfinite(first_rho); ++mfi) {
      amrex::Box bx = amrex::grow(mfi.validbox(), state.nGrowVect());
      bx &= mfi.fabbox();
      const auto host = copy_fab_for_debug(state[mfi], num_state_components());
      const auto values = host.const_array();
      for (amrex::BoxIterator bit(bx); bit.ok(); ++bit) {
         const amrex::IntVect cell = bit();
         const int i = cell[0];
#if (AMREX_SPACEDIM >= 2)
         const int j = cell[1];
#else
         const int j = 0;
#endif
#if (AMREX_SPACEDIM == 3)
         const int k = cell[2];
#else
         const int k = 0;
#endif
         const amrex::Real rho = values(i, j, k, RHO);
         const bool canonical_vacuum = allows_vacuum
            && canonical_rhd_vacuum(load_rhd_conserved(values, i, j, k));
         if (rho <= 0.0_rt && !canonical_vacuum) {
            first_i = i;
            first_j = j;
            first_k = k;
            first_rho = rho;
            for (int component = 0; component < num_state_components(); ++component) {
               first_state[component] = values(i, j, k, component);
            }
            break;
         }
      }
   }
   if (!std::isfinite(first_rho)) {
      return;
   }
   std::ostringstream message;
   message << "Reconstruction input contract failed in " << context << " at time " << time << " on level " << lev
           << ": minimum ghost-inclusive density=" << min_rho << " first_local_bad=(" << first_i << "," << first_j
           << "," << first_k << ") first_local_state=[";
   for (int component = 0; component < num_state_components(); ++component) {
      if (component != 0) {
         message << ",";
      }
      message << first_state[component];
   }
   message << "].";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::debug_check_valid_cell_finite(amrex::MultiFab const& mf, amrex::iMultiFab const& valid_mask,
                                                  int ncomp, char const* context, int lev) const
{
   if (!cfg_.debug_check_finite) {
      return;
   }
   amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
   amrex::ReduceData<amrex::Long> reduce_data(reduce_ops);
   using ReduceTuple = decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const auto values = mf.const_array(mfi);
         const auto mask = valid_mask.const_array(mfi);
         reduce_ops.eval(mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
               amrex::Long invalid_values = 0;
               if (mask(i, j, k, 0) != 0) {
                  for (int component = 0; component < ncomp; ++component) {
                     invalid_values += !amrex::Math::isfinite(values(i, j, k, component));
                  }
               }
               return {invalid_values};
            });
      }
   }
   amrex::Long bad_count = amrex::get<0>(reduce_data.value());
   amrex::ParallelDescriptor::ReduceLongSum(bad_count);
   if (bad_count == 0) {
      return;
   }
   std::ostringstream message;
   message << "Debug finite contract failed in " << context << " on level " << lev << ": " << bad_count
           << " invalid active-cell value(s).";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::abort_bad_physical_state(char const* context, int step, amrex::Real time, int lev,
                                             amrex::Long bad_count, amrex::Real min_rho, amrex::Real min_pressure,
                                             int first_bad_i, int first_bad_j, int first_bad_k,
                                             amrex::Real first_bad_rho, amrex::Real first_bad_pressure,
                                             std::array<amrex::Real, NVAR_HYDRO> const& first_bad_conserved) const
{
   std::ostringstream message;
   message << "Physical-state check failed in " << context << " at step " << step << " time " << time << " on level "
           << lev << ": bad_active_cells=" << bad_count << " min_rho=" << min_rho << " min_pressure=" << min_pressure
           << " first_local_bad=(" << first_bad_i << "," << first_bad_j << "," << first_bad_k << ")"
           << " first_local_rho=" << first_bad_rho << " first_local_pressure=" << first_bad_pressure
           << " first_local_conserved=[";
   for (int component = 0; component < NVAR_HYDRO; ++component) {
      if (component != 0) {
         message << ",";
      }
      message << first_bad_conserved[component];
   }
   message << "].";
   const std::string text = message.str();
   amrex::Abort(text.c_str());
}

void AmrSolverBase::debug_check_rhs_level(int lev, char const* context) const
{
   if (!cfg_.debug_check_finite || !fluid_state_.rhs()[lev]) {
      return;
   }
   amrex::iMultiFab mask = valid_cell_mask(lev);
   debug_check_valid_cell_finite(*fluid_state_.rhs()[lev], mask, num_state_components(), context, lev);
}

void AmrSolverBase::debug_check_fluxes(int lev, bool solid_masked, char const* context) const
{
   if (!cfg_.debug_check_finite || solid_masked) {
      return;
   }
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (fluxes_[lev][dir]) {
         std::ostringstream label;
         label << context << " level " << lev << " dir " << dir;
         debug_check_mfab_finite(*fluxes_[lev][dir], num_state_components(), amrex::IntVect(0), label.str().c_str());
      }
   }
}

template <bool MHD, int EOSMethod>
void AmrSolverBase::check_active_cell_physical_state_method(int lev, char const* context, int step, amrex::Real time)
{
   if (physical_system_is_relativistic(physics_system_)) {
      if (spacetime_state_.active()) {
         prepare_spacetime_coupling_range(
            spacetime_state_.state(), lev, lev + 1, time, -1.0_rt, 0
         );
      }
      // Use the same non-mutating C2P/P2C contract as stage validation.
      // In particular, a finite cold ideal-gas state is not canonical vacuum
      // and is admissible at p=0; negative pressure remains inadmissible.
      check_relativistic_state_range(
         fluid_state_.state(), lev, lev + 1, context, time, true
      );
      return;
   }
   amrex::iMultiFab mask = valid_cell_mask(lev);
   const EOSParams eos = eos_params(cfg_);
   const auto maximum = std::numeric_limits<amrex::Real>::max();
   amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                    amrex::ReduceOpMin, amrex::ReduceOpMin> reduce_ops;
   amrex::ReduceData<amrex::Long, amrex::Long, amrex::Real, amrex::Real> reduce_data(reduce_ops);
   using ReduceTuple = decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const auto state = fluid_state_.state()[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         reduce_ops.eval(mfi.tilebox(), reduce_data,
            [valid, maximum, state, eos] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
               if (valid(i, j, k, 0) == 0) return {0, 0, maximum, maximum};
               const amrex::Real rho = state(i, j, k, RHO);
               amrex::Real pressure;
               if constexpr (MHD) {
                  pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
               } else {
                  pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
               }
               if (!amrex::Math::isfinite(rho) || !amrex::Math::isfinite(pressure)
                   || rho <= 0.0_rt || pressure <= 0.0_rt) {
                  return {1, 1, maximum, maximum};
               }
               return {1, 0, rho, pressure};
            });
      }
   }
   const auto totals = reduce_data.value();
   amrex::Long active_count = amrex::get<0>(totals);
   amrex::Long bad_count = amrex::get<1>(totals);
   amrex::Real min_rho = amrex::get<2>(totals);
   amrex::Real min_pressure = amrex::get<3>(totals);
   amrex::ParallelDescriptor::ReduceLongSum(active_count);
   amrex::ParallelDescriptor::ReduceLongSum(bad_count);
   amrex::ParallelDescriptor::ReduceRealMin(min_rho);
   amrex::ParallelDescriptor::ReduceRealMin(min_pressure);
   if (active_count == 0 || bad_count == 0) {
      return;
   }


   PhysicalFailure first;
   for (amrex::MFIter mfi(*fluid_state_.state()[lev]); mfi.isValid() && !first.found; ++mfi) {
      const auto state = fluid_state_.state()[lev]->const_array(mfi);
      const auto valid = mask.const_array(mfi);
      amrex::Gpu::DeviceScalar<PhysicalFailure> failure(PhysicalFailure{});
      auto* output = failure.dataPtr();
      amrex::ParallelFor(mfi.validbox(), [valid, state, eos, output] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         if (valid(i, j, k, 0) == 0) return;
         const amrex::Real rho = state(i, j, k, RHO);
         amrex::Real pressure;
         if constexpr (MHD) {
            pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
         } else {
            pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
         }
         if ((!amrex::Math::isfinite(rho) || !amrex::Math::isfinite(pressure)
              || rho <= 0.0_rt || pressure <= 0.0_rt)
             && amrex::Gpu::Atomic::CAS(&output->found, 0, 1) == 0) {
            output->i = i;
            output->j = j;
            output->k = k;
            output->rho = rho;
            output->pressure = pressure;
            for (int component = 0; component < NVAR_HYDRO; ++component) {
               output->conserved[component] = state(i, j, k, component);
            }
         }
      });
      first = failure.dataValue();
   }
   std::array<amrex::Real, NVAR_HYDRO> first_conserved;
   std::copy(first.conserved.begin(), first.conserved.end(), first_conserved.begin());
   abort_bad_physical_state(context, step, time, lev, bad_count, min_rho, min_pressure,
      first.i, first.j, first.k, first.rho, first.pressure, first_conserved);
}

void AmrSolverBase::check_active_cell_physical_state(int lev, char const* context, int step, amrex::Real time)
{
   const bool magnetic = is_mhd(cfg_);
#define PYFLUID_CHECK_STATE(EOS)                                                                                       \
   do {                                                                                                                \
      if (magnetic) {                                                                                                  \
         check_active_cell_physical_state_method<true, EOS>(lev, context, step, time);                                 \
      } else {                                                                                                         \
         check_active_cell_physical_state_method<false, EOS>(lev, context, step, time);                                \
      }                                                                                                                \
   } while (false)

   PYFLUID_CHECK_STATE(EOSRuntime);
#undef PYFLUID_CHECK_STATE
}

void AmrSolverBase::debug_check_state_level(int lev, char const* context, int step, amrex::Real time,
                                            bool force_physical_check)
{
   const bool check_physical = accepted_physical_check_due(step, force_physical_check);
   if (!cfg_.debug_check_finite && !check_physical) {
      return;
   }
   if (!fluid_state_.state()[lev]) {
      return;
   }
   if (cfg_.debug_check_finite) {
      amrex::iMultiFab mask = valid_cell_mask(lev);
      debug_check_valid_cell_finite(*fluid_state_.state()[lev], mask, num_state_components(), context, lev);
   }
   if (check_physical) {
      check_active_cell_physical_state(lev, context, step, time);
   }
}

void AmrSolverBase::debug_check_state_all_levels(char const* context, int step, amrex::Real time,
                                                 bool force_physical_check)
{
   const bool check_physical = accepted_physical_check_due(step, force_physical_check);
   if (!cfg_.debug_check_finite && !check_physical) {
      return;
   }
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      debug_check_state_level(lev, context, step, time, force_physical_check);
   }
}

void AmrSolverBase::debug_check_relativistic_stage_range(
   LevelData const& state,
   int lev_begin,
   int lev_end,
   char const* context,
   amrex::Real time
)
{
   if (cfg_.debug_check_physical && physical_system_is_relativistic(physics_system_)) {
      check_relativistic_state_range(state, lev_begin, lev_end, context, time);
   }
}

void AmrSolverBase::check_relativistic_state_range(
   LevelData const& state,
   int lev_begin,
   int lev_end,
   char const* context,
   amrex::Real time,
   bool require_round_trip
)
{
   AMREX_ALWAYS_ASSERT(physical_system_is_relativistic(physics_system_));
   const EOSParams eos = eos_params(cfg_);
   const FixedSpacetimeParameters fixed_spacetime = fixed_spacetime_parameters(cfg_);
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      const amrex::iMultiFab valid_mask = valid_cell_mask(lev);
      amrex::MultiFab const* cell_spacetime =
         spacetime_state_.active() && !uses_analytic_fixed_spacetime_kernels()
         ? spacetime_coupling_.cell_spacetime()[lev].get()
         : nullptr;
      const RelativisticPreflightSummary summary =
         evaluate_relativistic_preflight_level_runtime(
            physics_system_, eos_method_, *state[lev], valid_mask, cell_spacetime,
            Geom(lev), fixed_spacetime, eos, time,
            mesh_coordinate_code(cfg_.mesh_coordinate), mesh_axes_layout(cfg_),
            require_round_trip ? RelativisticCheck::ConservedRoundTrip
                               : RelativisticCheck::Recovery
         );
      if (summary.failures == 0) {
         continue;
      }
      std::ostringstream message;
      message << "Relativistic stage contract failed in " << context
              << " at time " << time << " on level " << lev
              << ": bad_active_cells=" << summary.failures
              << " failure_reason=" << summary.failure_reason
              << " max_conserved_error=" << summary.conserved_error
              << " max_primitive_error=" << summary.primitive_error
              << " min_D=" << summary.minimum_density
              << " min_E=" << summary.minimum_energy
              << " min_q=" << summary.minimum_admissibility_margin << ".";
      const std::string text = message.str();
      amrex::Abort(text.c_str());
   }
}

void AmrSolverBase::run_evolution_preflight(amrex::Real time)
{
   if (!physical_system_is_relativistic(physics_system_)) {
      return;
   }

   if (spacetime_state_.active()) {
      prepare_spacetime_coupling_range(
         spacetime_state_.state(), 0, finestLevel() + 1, time, -1.0_rt, 0
      );
   }

   const EOSParams eos = eos_params(cfg_);
   const FixedSpacetimeParameters fixed_spacetime =
      fixed_spacetime_parameters(cfg_);
   RelativisticPreflightSummary total;
   int first_failing_level = -1;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      const amrex::iMultiFab valid_mask = valid_cell_mask(lev);
      amrex::MultiFab const* cell_spacetime =
         spacetime_state_.active() && !uses_analytic_fixed_spacetime_kernels()
         ? spacetime_coupling_.cell_spacetime()[lev].get()
         : nullptr;
      const RelativisticPreflightSummary level =
         evaluate_relativistic_preflight_level_runtime(
            physics_system_, eos_method_, *fluid_state_.state()[lev], valid_mask,
            cell_spacetime, Geom(lev), fixed_spacetime, eos, time,
            mesh_coordinate_code(cfg_.mesh_coordinate), mesh_axes_layout(cfg_),
            RelativisticCheck::InitialDataRoundTrip
         );

      total.cells += level.cells;
      total.failures += level.failures;
      total.failure_reason = std::max(
         total.failure_reason, level.failure_reason
      );
      total.conserved_error = std::max(
         total.conserved_error, level.conserved_error
      );
      total.primitive_error = std::max(
         total.primitive_error, level.primitive_error
      );
      total.minimum_density = std::min(
         total.minimum_density, level.minimum_density
      );
      total.minimum_energy = std::min(
         total.minimum_energy, level.minimum_energy
      );
      total.minimum_admissibility_margin = std::min(
         total.minimum_admissibility_margin,
         level.minimum_admissibility_margin
      );
      total.nonfinite_conserved += level.nonfinite_conserved;
      total.nonfinite_metric_volume += level.nonfinite_metric_volume;
      total.nonpositive_density += level.nonpositive_density;
      total.nonpositive_energy += level.nonpositive_energy;
      if (first_failing_level < 0 && level.failures > 0) {
         first_failing_level = lev;
      }
   }

   if (total.failures > 0) {
      std::ostringstream message;
      message
         << "Evolution preflight C2P/P2C round trip failed: bad_active_cells="
         << total.failures
         << " first_failing_level=" << first_failing_level
         << " failure_reason=" << total.failure_reason
         << " max_conserved_error=" << total.conserved_error
         << " max_primitive_error=" << total.primitive_error
         << " min_D=" << total.minimum_density
         << " min_E=" << total.minimum_energy
         << " min_q=" << total.minimum_admissibility_margin
         << " nonfinite_conserved=" << total.nonfinite_conserved
         << " nonfinite_metric_volume=" << total.nonfinite_metric_volume
         << " nonpositive_density=" << total.nonpositive_density
         << " nonpositive_energy=" << total.nonpositive_energy
         << " conserved_tolerance="
         << preflight_conserved_round_trip_tolerance
         << " primitive_tolerance="
         << preflight_primitive_round_trip_tolerance
         << ". Failure reasons 10+status and 20+status identify the first and "
            "second C2P attempts; 30 identifies a round-trip mismatch.";
      const std::string text = message.str();
      amrex::Abort(text.c_str());
   }

   if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
         << "Evolution preflight C2P/P2C: active_cells=" << total.cells
         << " max_conserved_error=" << total.conserved_error
         << " max_primitive_error=" << total.primitive_error << '\n';
   }
}

} // namespace pyfluid_amrex
