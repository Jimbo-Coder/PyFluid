#include "relativity/rmhd_correction_flux_impl.hpp"

namespace pyfluid_amrex {

void launch_rmhd_hllc_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
)
{
   limiter_detail::launch_rmhd_correction_fluxes<RiemannHLLC>(
      direction, context, metric, pwl_only
   );
}

void launch_rmhd_hllc_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
)
{
   limiter_detail::launch_rmhd_correction_fluxes<RiemannHLLC>(
      direction, context, metric, pwl_only
   );
}

} // namespace pyfluid_amrex
