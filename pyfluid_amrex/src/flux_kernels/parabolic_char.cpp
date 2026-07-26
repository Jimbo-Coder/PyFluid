#include "pyfluid_amrex/reconstruction/parabolic.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void parabolic_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconParabolicChar>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
