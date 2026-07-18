#include "pyfluid_amrex/reconstruction/mp5.hpp"

#include "../numerics/flux_factory_internal.hpp"

namespace pyfluid_amrex::flux_factory_entries {
void mp5_char(FluxLaunch const& launch)
{
   flux_factory_detail::dispatch_layout<ReconMP5Char>(launch);
}
} // namespace pyfluid_amrex::flux_factory_entries
