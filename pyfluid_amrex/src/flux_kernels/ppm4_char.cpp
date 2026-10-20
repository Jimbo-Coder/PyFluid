#include "pyfluid_amrex/reconstruction/ppm4.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void ppm4_char(FluxKernelLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconPPM4Char>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
