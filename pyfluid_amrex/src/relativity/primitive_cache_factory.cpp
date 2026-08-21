#include "pyfluid_amrex/primitive_cache_factory.hpp"

#include "relativity/primitive_cache.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"

namespace pyfluid_amrex {
namespace {

template <int NumComponents, int EosMethod>
void fill_newtonian_cache(PrimitiveCacheLaunch const& launch)
{
   if (launch.needed_mask != nullptr) {
      fill_primitive_cache_masked<NumComponents, EosMethod>(
         launch.primitive,
         launch.conserved,
         *launch.needed_mask,
         launch.eos,
         launch.needed_kind
      );
      return;
   }
   fill_primitive_cache<NumComponents, EosMethod>(
      launch.primitive, launch.conserved, launch.eos
   );
}

template <int NumComponents>
void dispatch_newtonian_eos(PrimitiveCacheLaunch const& launch)
{
   switch (launch.eos_method) {
   case EOSPolytrope:
      fill_newtonian_cache<NumComponents, EOSPolytrope>(launch);
      return;
   case EOSHybridPolytrope:
      fill_newtonian_cache<NumComponents, EOSHybridPolytrope>(launch);
      return;
   case EOSCompose:
      fill_newtonian_cache<NumComponents, EOSCompose>(launch);
      return;
   case EOSIdealGas:
      fill_newtonian_cache<NumComponents, EOSIdealGas>(launch);
      return;
   default:
      amrex::Abort("invalid EOS method in primitive-cache factory");
   }
}

template <int EosMethod>
void fill_rhd_cache(PrimitiveCacheLaunch const& launch)
{
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      launch.cell_spacetime != nullptr,
      "RHD primitive recovery requires cell-centered spacetime"
   );
   fill_rhd_primitive_cache<EosMethod>(
      launch.primitive,
      launch.conserved,
      launch.eos,
      *launch.cell_spacetime
   );
}

template <PhysicalSystem System, int EosMethod>
void project_atmosphere(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   EOSParams const& eos,
   EOSThermodynamicState const& atmosphere,
   int ghost_cells,
   amrex::Long* c2p_fallback_count,
   amrex::Long* atmosphere_reset_count
)
{
   project_relativistic_atmosphere<System, EosMethod>(
      conserved,
      eos,
      atmosphere,
      [&](amrex::MFIter const& mfi) {
         return StoredSpacetimeCellMetricAccessor{
            cell_spacetime.const_array(mfi)
         };
      },
      ghost_cells,
      c2p_fallback_count,
      atmosphere_reset_count
   );
}

} // namespace

void fill_primitive_cache(PrimitiveCacheLaunch const& launch)
{
   if (
      physical_system_is_relativistic(launch.physical_system)
      && launch.cell_spacetime == nullptr
   ) {
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         launch.geometry != nullptr,
         "analytic fixed-spacetime primitive recovery requires AMReX geometry"
      );
      fill_base_primitive_cache_runtime(
         launch.primitive,
         launch.conserved,
         launch.eos,
         launch.eos_method,
         launch.physical_system,
         *launch.geometry,
         launch.fixed_spacetime,
         launch.time
      );
      return;
   }
   switch (launch.physical_system) {
   case PhysicalSystem::HD:
      if (launch.num_components == NVAR_HYDRO) {
         dispatch_newtonian_eos<NVAR_HYDRO>(launch);
         return;
      }
      if (launch.num_components == NVAR_HYDRO + 1) {
         dispatch_newtonian_eos<NVAR_HYDRO + 1>(launch);
         return;
      }
      break;
   case PhysicalSystem::MHD:
      if (launch.num_components == NVAR_MHD) {
         dispatch_newtonian_eos<NVAR_MHD>(launch);
         return;
      }
      if (launch.num_components == NVAR_MHD + 1) {
         dispatch_newtonian_eos<NVAR_MHD + 1>(launch);
         return;
      }
      break;
   case PhysicalSystem::RHD:
      switch (launch.eos_method) {
      case EOSPolytrope:
         fill_rhd_cache<EOSPolytrope>(launch);
         return;
      case EOSHybridPolytrope:
         fill_rhd_cache<EOSHybridPolytrope>(launch);
         return;
      case EOSCompose:
         fill_rhd_cache<EOSCompose>(launch);
         return;
      case EOSIdealGas:
         fill_rhd_cache<EOSIdealGas>(launch);
         return;
      default:
         amrex::Abort("invalid EOS method in RHD primitive-cache factory");
      }
   case PhysicalSystem::RMHD:
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         launch.cell_spacetime != nullptr,
         "RMHD primitive recovery requires cell-centered spacetime"
      );
      fill_rmhd_primitive_cache(
         launch.primitive,
         launch.conserved,
         launch.eos,
         *launch.cell_spacetime
      );
      return;
   }
   amrex::Abort("unsupported state layout in primitive-cache factory");
}

void apply_relativistic_atmosphere(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   PhysicalSystem physical_system,
   int eos_method,
   EOSParams const& eos,
   EOSThermodynamicState const& atmosphere,
   int ghost_cells,
   amrex::Long* c2p_fallback_count,
   amrex::Long* atmosphere_reset_count
)
{
   if (physical_system == PhysicalSystem::RMHD) {
      project_atmosphere<PhysicalSystem::RMHD, EOSIdealGas>(
         conserved,
         cell_spacetime,
         eos,
         atmosphere,
         ghost_cells,
         c2p_fallback_count,
         atmosphere_reset_count
      );
      return;
   }
   if (physical_system != PhysicalSystem::RHD) {
      return;
   }
   switch (eos_method) {
   case EOSPolytrope:
      project_atmosphere<PhysicalSystem::RHD, EOSPolytrope>(
         conserved, cell_spacetime, eos, atmosphere, ghost_cells,
         c2p_fallback_count, atmosphere_reset_count
      );
      return;
   case EOSHybridPolytrope:
      project_atmosphere<PhysicalSystem::RHD, EOSHybridPolytrope>(
         conserved, cell_spacetime, eos, atmosphere, ghost_cells,
         c2p_fallback_count, atmosphere_reset_count
      );
      return;
   case EOSCompose:
      project_atmosphere<PhysicalSystem::RHD, EOSCompose>(
         conserved, cell_spacetime, eos, atmosphere, ghost_cells,
         c2p_fallback_count, atmosphere_reset_count
      );
      return;
   case EOSIdealGas:
      project_atmosphere<PhysicalSystem::RHD, EOSIdealGas>(
         conserved, cell_spacetime, eos, atmosphere, ghost_cells,
         c2p_fallback_count, atmosphere_reset_count
      );
      return;
   default:
      amrex::Abort("invalid EOS method in atmosphere factory");
   }
}

} // namespace pyfluid_amrex
