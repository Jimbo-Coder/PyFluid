#include "pyfluid_amrex/flux_factory.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex {
namespace flux_factory_entries {

void pwc(FluxLaunch const& launch);
void pwl(FluxLaunch const& launch);
void wenoz(FluxLaunch const& launch);
void wenoz_char(FluxLaunch const& launch);
void mp5(FluxLaunch const& launch);
void mp5_char(FluxLaunch const& launch);
void teno5(FluxLaunch const& launch);
void teno5_char(FluxLaunch const& launch);

} // namespace flux_factory_entries

void fill_fluxes(FluxLaunch const& launch)
{
   switch (launch.reconstruction_method) {
   case ReconPWC:
#if PYFLUID_AMREX_ENABLE_RECON_PWC
      flux_factory_entries::pwc(launch);
      return;
#endif
      break;
   case ReconPWL:
#if PYFLUID_AMREX_ENABLE_RECON_PWL
      flux_factory_entries::pwl(launch);
      return;
#endif
      break;
   case ReconWENOZ:
#if PYFLUID_AMREX_ENABLE_RECON_WENOZ
      flux_factory_entries::wenoz(launch);
      return;
#endif
      break;
   case ReconWENOZChar:
#if PYFLUID_AMREX_ENABLE_RECON_WENOZ_CHAR
      flux_factory_entries::wenoz_char(launch);
      return;
#endif
      break;
   case ReconMP5:
#if PYFLUID_AMREX_ENABLE_RECON_MP5
      flux_factory_entries::mp5(launch);
      return;
#endif
      break;
   case ReconMP5Char:
#if PYFLUID_AMREX_ENABLE_RECON_MP5_CHAR
      flux_factory_entries::mp5_char(launch);
      return;
#endif
      break;
   case ReconTENO5:
#if PYFLUID_AMREX_ENABLE_RECON_TENO5
      flux_factory_entries::teno5(launch);
      return;
#endif
      break;
   case ReconTENO5Char:
#if PYFLUID_AMREX_ENABLE_RECON_TENO5_CHAR
      flux_factory_entries::teno5_char(launch);
      return;
#endif
      break;
   default:
      amrex::Abort("invalid reconstruction method in flux factory");
      return;
   }
   amrex::Abort("selected reconstruction method is disabled in this build");
}

} // namespace pyfluid_amrex
