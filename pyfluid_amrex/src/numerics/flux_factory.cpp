#include "pyfluid_amrex/flux_factory.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#if PYFLUID_AMREX_ENABLE_MHD
#include "pyfluid_amrex/rmhd_advanced_flux.hpp"
#endif

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
void parabolic(FluxLaunch const& launch);
void parabolic_char(FluxLaunch const& launch);
void ppm4(FluxLaunch const& launch);
void ppm4_char(FluxLaunch const& launch);
void ppmx(FluxLaunch const& launch);
void ppmx_char(FluxLaunch const& launch);
void ppm3(FluxLaunch const& launch);
void ppm3_char(FluxLaunch const& launch);

} // namespace flux_factory_entries

void fill_fluxes(FluxLaunch const& launch)
{
#if PYFLUID_AMREX_ENABLE_MHD
   if (physical_system(launch.config) == PhysicalSystem::RMHD
       && (launch.riemann_method == RiemannHLLC
           || launch.riemann_method == RiemannHLLD)) {
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC || PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
      fill_rmhd_advanced_fluxes(launch);
      return;
#else
      amrex::Abort("advanced RMHD Riemann solvers are disabled in this build");
#endif
   }
#endif
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
   case ReconParabolic:
#if PYFLUID_AMREX_ENABLE_RECON_PARABOLIC
      flux_factory_entries::parabolic(launch);
      return;
#endif
      break;
   case ReconParabolicChar:
#if PYFLUID_AMREX_ENABLE_RECON_PARABOLIC_CHAR
      flux_factory_entries::parabolic_char(launch);
      return;
#endif
      break;
   case ReconPPM4:
#if PYFLUID_AMREX_ENABLE_RECON_PPM4
      flux_factory_entries::ppm4(launch);
      return;
#endif
      break;
   case ReconPPM4Char:
#if PYFLUID_AMREX_ENABLE_RECON_PPM4_CHAR
      flux_factory_entries::ppm4_char(launch);
      return;
#endif
      break;
   case ReconPPMX:
#if PYFLUID_AMREX_ENABLE_RECON_PPMX
      flux_factory_entries::ppmx(launch);
      return;
#endif
      break;
   case ReconPPMXChar:
#if PYFLUID_AMREX_ENABLE_RECON_PPMX_CHAR
      flux_factory_entries::ppmx_char(launch);
      return;
#endif
      break;
   case ReconPPM3:
#if PYFLUID_AMREX_ENABLE_RECON_PPM3
      flux_factory_entries::ppm3(launch);
      return;
#endif
      break;
   case ReconPPM3Char:
#if PYFLUID_AMREX_ENABLE_RECON_PPM3_CHAR
      flux_factory_entries::ppm3_char(launch);
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
