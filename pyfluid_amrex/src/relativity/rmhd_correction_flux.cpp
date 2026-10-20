#include "relativity/rmhd_correction_flux.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex {
namespace {

template <typename MetricAccessor>
void fill_rmhd_correction_fluxes_impl(
   int riemann_method,
   int direction,
   RmhdCorrectionFluxContext const& context,
   MetricAccessor const& metric,
   bool pwl_only
)
{
   switch (riemann_method) {
   case RiemannLLF:
      launch_rmhd_llf_correction_fluxes(direction, context, metric, pwl_only);
      return;
   case RiemannHLL:
   case RiemannHLLE:
      launch_rmhd_hll_correction_fluxes(direction, context, metric, pwl_only);
      return;
   case RiemannHLLC:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
      launch_rmhd_hllc_correction_fluxes(direction, context, metric, pwl_only);
#else
      amrex::Abort("RMHD HLLC candidate correction is disabled in this build");
#endif
      return;
   case RiemannHLLD:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
      launch_rmhd_hlld_correction_fluxes(direction, context, metric, pwl_only);
#else
      amrex::Abort("RMHD HLLD candidate correction is disabled in this build");
#endif
      return;
   default:
      amrex::Abort("invalid RMHD candidate-correction Riemann solver");
   }
}

} // namespace

void fill_rmhd_correction_fluxes(
   int riemann_method,
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
)
{
   fill_rmhd_correction_fluxes_impl(riemann_method, direction, context, metric, pwl_only);
}

void fill_rmhd_correction_fluxes(
   int riemann_method,
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
)
{
   fill_rmhd_correction_fluxes_impl(riemann_method, direction, context, metric, pwl_only);
}

} // namespace pyfluid_amrex
