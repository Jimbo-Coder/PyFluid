#include "pyfluid_amrex/reconstruction/ppm3.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {

void ppm3_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconPPM3Char>(launch);
}

} // namespace pyfluid_amrex::flux_factory_entries
