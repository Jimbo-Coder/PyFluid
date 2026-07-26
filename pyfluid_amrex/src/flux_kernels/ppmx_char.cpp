#include "pyfluid_amrex/reconstruction/ppmx.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void ppmx_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconPPMXChar>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
