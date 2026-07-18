#include "pyfluid_amrex/reconstruction/teno5.hpp"

#include "../flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void teno5_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconTENO5Char>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
