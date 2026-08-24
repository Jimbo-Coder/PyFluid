#include "amr/amr_solver_base.hpp"

#include <AMReX_BoxIterator.H>

#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

namespace {

constexpr amrex::Real preflight_round_trip_tolerance = 1.0e-7;

struct RelativisticRoundTripResult {
   int failure_reason = 0;
   amrex::Real conserved_error = 0.0;
   amrex::Real primitive_error = 0.0;
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
   error = amrex::max(error, preflight_scaled_error(
      actual.mom1, expected.mom1, scale_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.mom2, expected.mom2, scale_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.mom3, expected.mom3, scale_floor
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
      actual.eps, expected.eps, 1.0e-14
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.pressure, expected.pressure, eos.pressure_floor
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.v1, expected.v1, 1.0
   ));
   error = amrex::max(error, preflight_scaled_error(
      actual.v2, expected.v2, 1.0
   ));
   return amrex::max(error, preflight_scaled_error(
      actual.v3, expected.v3, 1.0
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
   bool require_round_trip
) noexcept
{
   RelativisticRoundTripResult result;
   const RHDRecoveryResult initial_recovery = recover_rhd<EOSMethod>(
      initial_conserved, metric, eos
   );
   if (!initial_recovery.attempt.succeeded()) {
      result.failure_reason = 10
         + static_cast<int>(initial_recovery.attempt.status);
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
      1.0e-30, eos.rho_floor * 1.0e-6
   );
   result.conserved_error = rhd_conserved_round_trip_error(
      regenerated_conserved, initial_conserved, conserved_scale_floor
   );
   result.primitive_error = rhd_primitive_round_trip_error(
      regenerated_recovery.primitive, initial_recovery.primitive, eos
   );
   if (require_round_trip
       && (result.conserved_error > preflight_round_trip_tolerance
           || result.primitive_error > preflight_round_trip_tolerance)) {
      result.failure_reason = 30;
   }
   return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RelativisticRoundTripResult evaluate_rmhd_round_trip(
   RMHDConserved const& initial_conserved,
   SpacetimeMetric const& metric,
   EOSParams const& eos,
   bool require_round_trip
) noexcept
{
   RelativisticRoundTripResult result;
   const RMHDRecoveryResult initial_recovery = recover_srmhd(
      initial_conserved, metric, eos
   );
   if (!initial_recovery.succeeded()) {
      result.failure_reason = 10
         + static_cast<int>(initial_recovery.attempt.status);
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
      1.0e-30, eos.rho_floor * 1.0e-6
   );
   result.conserved_error = rmhd_conserved_round_trip_error(
      regenerated_conserved, initial_conserved, conserved_scale_floor
   );
   result.primitive_error = rmhd_primitive_round_trip_error(
      regenerated_recovery.primitive, initial_recovery.primitive, eos
   );
   if (require_round_trip
       && (result.conserved_error > preflight_round_trip_tolerance
           || result.primitive_error > preflight_round_trip_tolerance)) {
      result.failure_reason = 30;
   }
   return result;
}

struct RelativisticPreflightSummary {
   amrex::Long cells = 0;
   amrex::Long failures = 0;
   int failure_reason = 0;
   amrex::Real conserved_error = 0.0;
   amrex::Real primitive_error = 0.0;
   amrex::Real minimum_density = std::numeric_limits<amrex::Real>::max();
   amrex::Real minimum_energy = std::numeric_limits<amrex::Real>::max();
   amrex::Real minimum_admissibility_margin =
      std::numeric_limits<amrex::Real>::max();
};

template <PhysicalSystem System, int EOSMethod, typename MetricAccessorFactory>
RelativisticPreflightSummary evaluate_relativistic_preflight_level(
   amrex::MultiFab const& state,
   amrex::iMultiFab const& valid_mask,
   EOSParams const& eos,
   bool require_round_trip,
   MetricAccessorFactory&& metric_accessor_factory
)
{
   amrex::MultiFab result(
      state.boxArray(), state.DistributionMap(), 8, 0
   );
   result.setVal(0.0);
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
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
               cell_conserved, cell_metric, eos, require_round_trip
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
               require_round_trip
            );
         }
         values(i, j, k, 0) = cell_result.failure_reason == 0 ? 0.0 : 1.0;
         values(i, j, k, 1) = cell_result.failure_reason;
         values(i, j, k, 2) = cell_result.conserved_error;
         values(i, j, k, 3) = cell_result.primitive_error;
         values(i, j, k, 4) = 1.0;
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
   bool require_round_trip
)
{
   if (cell_spacetime != nullptr) {
      return evaluate_relativistic_preflight_level<System, EOSMethod>(
         state,
         valid_mask,
         eos,
         require_round_trip,
         [&](amrex::MFIter const& mfi) {
            return StoredSpacetimeCellMetricAccessor{
               cell_spacetime->const_array(mfi)
            };
         }
      );
   }
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   return evaluate_relativistic_preflight_level<System, EOSMethod>(
      state,
      valid_mask,
      eos,
      require_round_trip,
      [=](amrex::MFIter const&) {
         return FixedSpacetimeCellMetricAccessor{
            fixed_spacetime, lower, spacing, time
         };
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
   bool require_round_trip
)
{
   if (system == PhysicalSystem::RMHD) {
      return evaluate_relativistic_preflight_level<
         PhysicalSystem::RMHD, EOSRuntime
      >(state, valid_mask, cell_spacetime, geometry, fixed_spacetime, eos, time,
        require_round_trip);
   }
   static_cast<void>(eos_method);
   return evaluate_relativistic_preflight_level<
      PhysicalSystem::RHD, EOSRuntime
   >(state, valid_mask, cell_spacetime, geometry, fixed_spacetime, eos, time,
     require_round_trip);
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
      const auto values = mf.const_array(mfi);
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
   if (min_rho > 0.0) {
      return;
   }
   int first_i = std::numeric_limits<int>::min();
   int first_j = std::numeric_limits<int>::min();
   int first_k = std::numeric_limits<int>::min();
   amrex::Real first_rho = std::numeric_limits<amrex::Real>::quiet_NaN();
   std::array<amrex::Real, MaxStateComponents> first_state{};
   first_state.fill(std::numeric_limits<amrex::Real>::quiet_NaN());
   for (amrex::MFIter mfi(state); mfi.isValid() && !std::isfinite(first_rho); ++mfi) {
      amrex::Box bx = amrex::grow(mfi.validbox(), state.nGrowVect());
      bx &= mfi.fabbox();
      const auto values = state.const_array(mfi);
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
         if (rho <= 0.0) {
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
   amrex::Long bad_count = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+ : bad_count) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto lo = amrex::lbound(bx);
         const auto hi = amrex::ubound(bx);
         const auto values = mf.const_array(mfi);
         const auto mask = valid_mask.const_array(mfi);
         for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = lo.x; i <= hi.x; ++i) {
                  if (mask(i, j, k, 0) == 0) {
                     continue;
                  }
                  for (int n = 0; n < ncomp; ++n) {
                     if (!std::isfinite(values(i, j, k, n))) {
                        ++bad_count;
                     }
                  }
               }
            }
         }
      }
   }
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
   amrex::iMultiFab mask = valid_cell_mask(lev);
   amrex::Long active_count = 0;
   amrex::Long bad_count = 0;
   amrex::Real min_rho = std::numeric_limits<amrex::Real>::max();
   amrex::Real min_pressure = std::numeric_limits<amrex::Real>::max();
   const EOSParams eos = eos_params(cfg_);
   if (physical_system_is_relativistic(physics_system_)) {
      fill_level_primitive_cache(lev, time);
   }
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+ : active_count, bad_count)                                                            \
   reduction(min : min_rho, min_pressure) if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(*fluid_state_.state()[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& bx = mfi.tilebox();
         const auto lo = amrex::lbound(bx);
         const auto hi = amrex::ubound(bx);
         const auto state = fluid_state_.state()[lev]->const_array(mfi);
         const auto primitive = primitive_[lev]->const_array(mfi);
         const auto valid = mask.const_array(mfi);
         for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
               for (int i = lo.x; i <= hi.x; ++i) {
                  if (valid(i, j, k, 0) == 0) {
                     continue;
                  }
                  ++active_count;
                  amrex::Real rho = state(i, j, k, RHO);
                  amrex::Real pressure = 0.0;
                  if (physical_system_is_relativistic(physics_system_)) {
                     rho = primitive(i, j, k, RHO);
                     pressure = primitive(i, j, k, ENERGY);
                  } else if constexpr (MHD) {
                     pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
                  } else {
                     pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
                  }
                  if (!std::isfinite(rho) || !std::isfinite(pressure) || rho <= 0.0 || pressure <= 0.0) {
                     ++bad_count;
                     continue;
                  }
                  min_rho = std::min(min_rho, rho);
                  min_pressure = std::min(min_pressure, pressure);
               }
            }
         }
      }
   }
   amrex::ParallelDescriptor::ReduceLongSum(active_count);
   amrex::ParallelDescriptor::ReduceLongSum(bad_count);
   amrex::ParallelDescriptor::ReduceRealMin(min_rho);
   amrex::ParallelDescriptor::ReduceRealMin(min_pressure);
   if (active_count == 0 || bad_count == 0) {
      return;
   }

   int first_bad_i = std::numeric_limits<int>::min();
   int first_bad_j = std::numeric_limits<int>::min();
   int first_bad_k = std::numeric_limits<int>::min();
   amrex::Real first_bad_rho = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real first_bad_pressure = std::numeric_limits<amrex::Real>::quiet_NaN();
   std::array<amrex::Real, NVAR_HYDRO> first_bad_conserved{};
   first_bad_conserved.fill(std::numeric_limits<amrex::Real>::quiet_NaN());
   for (amrex::MFIter mfi(*fluid_state_.state()[lev]); mfi.isValid() && first_bad_i == std::numeric_limits<int>::min();
        ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const auto lo = amrex::lbound(bx);
      const auto hi = amrex::ubound(bx);
      const auto state = fluid_state_.state()[lev]->const_array(mfi);
      const auto primitive = primitive_[lev]->const_array(mfi);
      const auto valid = mask.const_array(mfi);
      for (int k = lo.z; k <= hi.z && first_bad_i == std::numeric_limits<int>::min(); ++k) {
         for (int j = lo.y; j <= hi.y && first_bad_i == std::numeric_limits<int>::min(); ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               if (valid(i, j, k, 0) == 0) {
                  continue;
               }
               amrex::Real rho = state(i, j, k, RHO);
               amrex::Real pressure = 0.0;
               if (physical_system_is_relativistic(physics_system_)) {
                  rho = primitive(i, j, k, RHO);
                  pressure = primitive(i, j, k, ENERGY);
               } else if constexpr (MHD) {
                  pressure = pressure_from_conserved<EOSMethod>(load_mhd_state(state, i, j, k), eos);
               } else {
                  pressure = pressure_from_state<EOSMethod>(state, i, j, k, eos);
               }
               if (!std::isfinite(rho) || !std::isfinite(pressure) || rho <= 0.0 || pressure <= 0.0) {
                  first_bad_i = i;
                  first_bad_j = j;
                  first_bad_k = k;
                  first_bad_rho = rho;
                  first_bad_pressure = pressure;
                  for (int component = 0; component < NVAR_HYDRO; ++component) {
                     first_bad_conserved[component] = state(i, j, k, component);
                  }
                  break;
               }
            }
         }
      }
   }
   abort_bad_physical_state(context, step, time, lev, bad_count, min_rho, min_pressure, first_bad_i, first_bad_j,
                            first_bad_k, first_bad_rho, first_bad_pressure, first_bad_conserved);
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
   if (!cfg_.debug_check_physical || !physical_system_is_relativistic(physics_system_)) {
      return;
   }
   const EOSParams eos = eos_params(cfg_);
   const FixedSpacetimeParameters fixed_spacetime = fixed_spacetime_parameters(cfg_);
   for (int lev = lev_begin; lev < lev_end; ++lev) {
      const amrex::iMultiFab valid_mask = valid_cell_mask(lev);
      amrex::MultiFab const* cell_spacetime = spacetime_state_.active()
         ? spacetime_coupling_.cell_spacetime()[lev].get()
         : nullptr;
      const RelativisticPreflightSummary summary =
         evaluate_relativistic_preflight_level_runtime(
            physics_system_, eos_method_, *state[lev], valid_mask, cell_spacetime,
            Geom(lev), fixed_spacetime, eos, time, false
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
         spacetime_state_.state(), 0, finestLevel() + 1, time, -1.0, 0
      );
   }

   const EOSParams eos = eos_params(cfg_);
   const FixedSpacetimeParameters fixed_spacetime =
      fixed_spacetime_parameters(cfg_);
   RelativisticPreflightSummary total;
   int first_failing_level = -1;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      const amrex::iMultiFab valid_mask = valid_cell_mask(lev);
      amrex::MultiFab const* cell_spacetime = spacetime_state_.active()
         ? spacetime_coupling_.cell_spacetime()[lev].get()
         : nullptr;
      const RelativisticPreflightSummary level =
         evaluate_relativistic_preflight_level_runtime(
            physics_system_, eos_method_, *fluid_state_.state()[lev], valid_mask,
            cell_spacetime, Geom(lev), fixed_spacetime, eos, time, true
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
         << " tolerance=" << preflight_round_trip_tolerance
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
