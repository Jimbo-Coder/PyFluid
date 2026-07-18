#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <array>

namespace pyfluid_amrex {

struct PatchGridSystem;
struct PatchLayout;
struct RunConfig;

enum class FluxTopology {
   Standard,
   SolidWall,
   MappedPatch,
};

struct FluxLaunch {
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes;
   amrex::MultiFab const& primitive;
   RunConfig const& config;
   int num_components = 0;
   int eos_method = 0;
   int reconstruction_method = 0;
   int riemann_method = 0;
   FluxTopology topology = FluxTopology::Standard;
   int direction = -1;
   amrex::Real time = 0.0;
   amrex::iMultiFab const* solid_mask = nullptr;
   amrex::Geometry const* geometry = nullptr;
   amrex::Array<amrex::Vector<int>, AMREX_SPACEDIM> const* solid_flux_kinds = nullptr;
   amrex::Vector<int> const* directional_solid_flux_kind = nullptr;
   PatchGridSystem const* grid_system = nullptr;
   PatchLayout const* patch_layout = nullptr;
   amrex::MultiFab const* cell_spacetime = nullptr;
};

void fill_fluxes(FluxLaunch const& launch);

template <int NumComponents, int EosMethod, bool SolidWalls = false, bool MappedPatches = false>
inline void launch_fluxes_runtime(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   RunConfig const& config,
   int reconstruction_method,
   int riemann_method,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geometry = nullptr,
   amrex::Array<amrex::Vector<int>, AMREX_SPACEDIM> const* solid_flux_kinds = nullptr,
   amrex::Real time = 0.0,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* patch_layout = nullptr
)
{
   constexpr FluxTopology topology = MappedPatches
      ? FluxTopology::MappedPatch
      : (SolidWalls ? FluxTopology::SolidWall : FluxTopology::Standard);
   fill_fluxes({
      fluxes,
      primitive,
      config,
      NumComponents,
      EosMethod,
      reconstruction_method,
      riemann_method,
      topology,
      -1,
      time,
      solid_mask,
      geometry,
      solid_flux_kinds,
      nullptr,
      grid_system,
      patch_layout,
      nullptr,
   });
}

template <
   int NumComponents,
   int Direction,
   int EosMethod,
   bool SolidWalls = false,
   bool MappedPatches = false
>
inline void launch_flux_direction_runtime(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   RunConfig const& config,
   int reconstruction_method,
   int riemann_method,
   amrex::iMultiFab const* solid_mask = nullptr,
   amrex::Geometry const* geometry = nullptr,
   amrex::Vector<int> const* solid_flux_kind = nullptr,
   amrex::Real time = 0.0,
   PatchGridSystem const* grid_system = nullptr,
   PatchLayout const* patch_layout = nullptr,
   amrex::MultiFab const* cell_spacetime = nullptr
)
{
   constexpr FluxTopology topology = MappedPatches
      ? FluxTopology::MappedPatch
      : (SolidWalls ? FluxTopology::SolidWall : FluxTopology::Standard);
   fill_fluxes({
      fluxes,
      primitive,
      config,
      NumComponents,
      EosMethod,
      reconstruction_method,
      riemann_method,
      topology,
      Direction,
      time,
      solid_mask,
      geometry,
      nullptr,
      solid_flux_kind,
      grid_system,
      patch_layout,
      cell_spacetime,
   });
}

} // namespace pyfluid_amrex
