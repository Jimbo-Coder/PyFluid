#include "pyfluid_amrex/reconstruction/ppmx.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void ppmx(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconPPMX>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
