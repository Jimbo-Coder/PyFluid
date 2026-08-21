#pragma once

#include "pyfluid_amrex/flux_factory.hpp"

#include "pyfluid_amrex/config.hpp"
#include "numerics/rhs.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex::flux_factory_detail {

template <int NumComponents, int ReconstructionMethod, int EosMethod, int Direction>
void fill_direction(FluxLaunch const& launch)
{
   switch (launch.topology) {
   case FluxTopology::Standard:
      fill_flux_direction_for_riemann<
         NumComponents, ReconstructionMethod, Direction, EosMethod
      >(
         launch.fluxes,
         launch.primitive,
         launch.config,
         launch.riemann_method,
         nullptr,
         launch.geometry,
         nullptr,
         launch.time,
         nullptr,
         nullptr,
         launch.cell_spacetime,
         launch.atmosphere
      );
      return;
   case FluxTopology::SolidWall:
      if constexpr (has_mhd_components<NumComponents>()) {
         amrex::Abort("solid-wall flux topology is implemented for hydro layouts only");
      } else {
         fill_flux_direction_for_riemann<
            NumComponents, ReconstructionMethod, Direction, EosMethod, true
         >(
            launch.fluxes,
            launch.primitive,
            launch.config,
            launch.riemann_method,
            launch.solid_mask,
            launch.geometry,
            launch.directional_solid_flux_kind,
            launch.time
         );
      }
      return;
   case FluxTopology::MappedPatch:
      amrex::Abort("mapped-patch fluxes are launched for all directions together");
      return;
   }
}

template <int NumComponents, int ReconstructionMethod, int EosMethod>
void fill_all_directions(FluxLaunch const& launch)
{
   switch (launch.topology) {
   case FluxTopology::Standard:
      fill_fluxes_for_riemann<NumComponents, ReconstructionMethod, EosMethod>(
         launch.fluxes,
         launch.primitive,
         launch.config,
         launch.riemann_method,
         nullptr,
         launch.geometry,
         nullptr,
         launch.time,
         nullptr,
         nullptr,
         launch.atmosphere
      );
      return;
   case FluxTopology::SolidWall:
      if constexpr (has_mhd_components<NumComponents>()) {
         amrex::Abort("solid-wall flux topology is implemented for hydro layouts only");
      } else {
         fill_fluxes_for_riemann<
            NumComponents, ReconstructionMethod, EosMethod, true
         >(
            launch.fluxes,
            launch.primitive,
            launch.config,
            launch.riemann_method,
            launch.solid_mask,
            launch.geometry,
            launch.solid_flux_kinds,
            launch.time,
            nullptr,
            nullptr,
            launch.atmosphere
         );
      }
      return;
   case FluxTopology::MappedPatch:
      if constexpr (has_mhd_components<NumComponents>()) {
         amrex::Abort("mapped-patch flux topology is implemented for hydro layouts only");
      } else {
         fill_fluxes_for_riemann<
            NumComponents, ReconstructionMethod, EosMethod, false, true
         >(
            launch.fluxes,
            launch.primitive,
            launch.config,
            launch.riemann_method,
            nullptr,
            nullptr,
            nullptr,
            launch.time,
            launch.grid_system,
            launch.patch_layout
         );
      }
      return;
   }
}

template <int NumComponents, int ReconstructionMethod, int EosMethod>
void fill_layout(FluxLaunch const& launch)
{
   if (launch.direction < 0) {
      fill_all_directions<NumComponents, ReconstructionMethod, EosMethod>(launch);
      return;
   }
   switch (launch.direction) {
   case 0:
      fill_direction<NumComponents, ReconstructionMethod, EosMethod, 0>(launch);
      return;
#if (AMREX_SPACEDIM >= 2)
   case 1:
      fill_direction<NumComponents, ReconstructionMethod, EosMethod, 1>(launch);
      return;
#endif
#if (AMREX_SPACEDIM == 3)
   case 2:
      fill_direction<NumComponents, ReconstructionMethod, EosMethod, 2>(launch);
      return;
#endif
   default:
      amrex::Abort("invalid direction in flux factory");
   }
}

template <int NumComponents, int ReconstructionMethod>
void dispatch_eos(FluxLaunch const& launch)
{
   switch (launch.eos_method) {
   case EOSIdealGas:
      fill_layout<NumComponents, ReconstructionMethod, EOSIdealGas>(launch);
      return;
   case EOSPolytrope:
      fill_layout<NumComponents, ReconstructionMethod, EOSPolytrope>(launch);
      return;
   case EOSHybridPolytrope:
      fill_layout<NumComponents, ReconstructionMethod, EOSHybridPolytrope>(launch);
      return;
   case EOSCompose:
      fill_layout<NumComponents, ReconstructionMethod, EOSCompose>(launch);
      return;
   default:
      amrex::Abort("invalid EOS method in flux factory");
   }
}

template <int ReconstructionMethod>
void dispatch_layout(FluxLaunch const& launch)
{
   switch (launch.num_components) {
   case NVAR_HYDRO:
      dispatch_eos<NVAR_HYDRO, ReconstructionMethod>(launch);
      return;
#if PYFLUID_AMREX_ENABLE_HYDRO
   case NVAR_HYDRO + 1:
      dispatch_eos<NVAR_HYDRO + 1, ReconstructionMethod>(launch);
      return;
#endif
#if PYFLUID_AMREX_ENABLE_MHD
   case NVAR_MHD:
      dispatch_eos<NVAR_MHD, ReconstructionMethod>(launch);
      return;
   case NVAR_MHD + 1:
      dispatch_eos<NVAR_MHD + 1, ReconstructionMethod>(launch);
      return;
#endif
   default:
      amrex::Abort("unsupported state layout in flux factory");
   }
}

} // namespace pyfluid_amrex::flux_factory_detail
