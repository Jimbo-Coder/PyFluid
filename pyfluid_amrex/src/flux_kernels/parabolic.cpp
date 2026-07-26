#include "pyfluid_amrex/reconstruction/parabolic.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void parabolic(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconParabolic>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
