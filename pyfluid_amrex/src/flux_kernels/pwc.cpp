#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void pwc(FluxKernelLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconPWC>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
