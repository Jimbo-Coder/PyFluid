#include "pyfluid_amrex/flux_factory.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/config.hpp"
#include "pyfluid_amrex/solid_geometry.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "numerics/flux_kernel_parameters.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#if PYFLUID_AMREX_ENABLE_MHD
#include "pyfluid_amrex/rmhd_advanced_flux.hpp"
#endif

namespace pyfluid_amrex {
namespace flux_factory_entries {

void pwc(FluxKernelLaunch const& launch);
void pwl(FluxKernelLaunch const& launch);
void wenoz(FluxKernelLaunch const& launch);
void wenoz_char(FluxKernelLaunch const& launch);
void mp5(FluxKernelLaunch const& launch);
void mp5_char(FluxKernelLaunch const& launch);
void teno5(FluxKernelLaunch const& launch);
void teno5_char(FluxKernelLaunch const& launch);
void parabolic(FluxKernelLaunch const& launch);
void parabolic_char(FluxKernelLaunch const& launch);
void ppm4(FluxKernelLaunch const& launch);
void ppm4_char(FluxKernelLaunch const& launch);
void ppmx(FluxKernelLaunch const& launch);
void ppmx_char(FluxKernelLaunch const& launch);
void ppm3(FluxKernelLaunch const& launch);
void ppm3_char(FluxKernelLaunch const& launch);

} // namespace flux_factory_entries

namespace {

FluxKernelLaunch resolve_flux_launch(FluxLaunch const& launch)
{
   auto const& config = launch.config;
   FluxKernelParameters parameters{};
   parameters.system = physical_system(config);
   parameters.eos = eos_params(config);
   parameters.mesh_coordinate = launch.geometry == nullptr
      ? MeshCartesian : mesh_coordinate_code(config.mesh_coordinate);
   parameters.mesh_axes = mesh_axes_layout(parameters.mesh_coordinate, config.mesh_axes);
   const bool relativistic = physical_system_is_relativistic(parameters.system);
   if (relativistic) {
      parameters.spacetime = fixed_spacetime_parameters(config);
   } else {
      parameters.solid = solid_geometry_evaluator(config, launch.time);
   }
   if (relativistic && launch.geometry != nullptr) {
      parameters.excision = excision_grid_view(
         spacetime_excision_region(config), *launch.geometry, launch.primitive.ixType()
      );
      parameters.diode = make_relativistic_diode_boundary(config, *launch.geometry);
   }
   return {launch, parameters};
}

} // namespace

void fill_fluxes(FluxLaunch const& launch)
{
   const auto kernel_launch = resolve_flux_launch(launch);
#if PYFLUID_AMREX_ENABLE_MHD
   if (kernel_launch.parameters.system == PhysicalSystem::RMHD
       && (launch.riemann_method == RiemannHLLC
           || launch.riemann_method == RiemannHLLD)) {
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC || PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
      fill_rmhd_advanced_fluxes(kernel_launch);
      return;
#else
      amrex::Abort("advanced RMHD Riemann solvers are disabled in this build");
#endif
   }
#endif
   switch (launch.reconstruction_method) {
   case ReconPWC:
#if PYFLUID_AMREX_ENABLE_RECON_PWC
      flux_factory_entries::pwc(kernel_launch);
      return;
#endif
      break;
   case ReconPWL:
#if PYFLUID_AMREX_ENABLE_RECON_PWL
      flux_factory_entries::pwl(kernel_launch);
      return;
#endif
      break;
   case ReconWENOZ:
#if PYFLUID_AMREX_ENABLE_RECON_WENOZ
      flux_factory_entries::wenoz(kernel_launch);
      return;
#endif
      break;
   case ReconWENOZChar:
#if PYFLUID_AMREX_ENABLE_RECON_WENOZ_CHAR
      flux_factory_entries::wenoz_char(kernel_launch);
      return;
#endif
      break;
   case ReconMP5:
#if PYFLUID_AMREX_ENABLE_RECON_MP5
      flux_factory_entries::mp5(kernel_launch);
      return;
#endif
      break;
   case ReconMP5Char:
#if PYFLUID_AMREX_ENABLE_RECON_MP5_CHAR
      flux_factory_entries::mp5_char(kernel_launch);
      return;
#endif
      break;
   case ReconTENO5:
#if PYFLUID_AMREX_ENABLE_RECON_TENO5
      flux_factory_entries::teno5(kernel_launch);
      return;
#endif
      break;
   case ReconTENO5Char:
#if PYFLUID_AMREX_ENABLE_RECON_TENO5_CHAR
      flux_factory_entries::teno5_char(kernel_launch);
      return;
#endif
      break;
   case ReconParabolic:
#if PYFLUID_AMREX_ENABLE_RECON_PARABOLIC
      flux_factory_entries::parabolic(kernel_launch);
      return;
#endif
      break;
   case ReconParabolicChar:
#if PYFLUID_AMREX_ENABLE_RECON_PARABOLIC_CHAR
      flux_factory_entries::parabolic_char(kernel_launch);
      return;
#endif
      break;
   case ReconPPM4:
#if PYFLUID_AMREX_ENABLE_RECON_PPM4
      flux_factory_entries::ppm4(kernel_launch);
      return;
#endif
      break;
   case ReconPPM4Char:
#if PYFLUID_AMREX_ENABLE_RECON_PPM4_CHAR
      flux_factory_entries::ppm4_char(kernel_launch);
      return;
#endif
      break;
   case ReconPPMX:
#if PYFLUID_AMREX_ENABLE_RECON_PPMX
      flux_factory_entries::ppmx(kernel_launch);
      return;
#endif
      break;
   case ReconPPMXChar:
#if PYFLUID_AMREX_ENABLE_RECON_PPMX_CHAR
      flux_factory_entries::ppmx_char(kernel_launch);
      return;
#endif
      break;
   case ReconPPM3:
#if PYFLUID_AMREX_ENABLE_RECON_PPM3
      flux_factory_entries::ppm3(kernel_launch);
      return;
#endif
      break;
   case ReconPPM3Char:
#if PYFLUID_AMREX_ENABLE_RECON_PPM3_CHAR
      flux_factory_entries::ppm3_char(kernel_launch);
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
