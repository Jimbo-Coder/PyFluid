#include "pyfluid_amrex/reconstruction/wenoz.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void wenoz_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconWENOZChar>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
