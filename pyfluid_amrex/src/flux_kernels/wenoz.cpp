#include "pyfluid_amrex/reconstruction/wenoz.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void wenoz(FluxKernelLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconWENOZ>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
