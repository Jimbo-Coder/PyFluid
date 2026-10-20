#include "pyfluid_amrex/reconstruction/teno5.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void teno5(FluxKernelLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconTENO5>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
