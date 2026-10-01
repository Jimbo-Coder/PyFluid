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
   if constexpr (has_mhd_components<NumComponents>()) {
      if (launch.canonical_face_b != nullptr) {
         AMREX_ALWAYS_ASSERT(launch.geometry != nullptr);
         AMREX_ALWAYS_ASSERT(launch.canonical_face_b_cell_mask != nullptr);
         fill_primitive_cache_from_ct_faces<NumComponents, EosMethod>(
            launch.primitive,
            launch.conserved,
            launch.eos,
            *launch.canonical_face_b,
            *launch.canonical_face_b_cell_mask,
            *launch.geometry
         );
         return;
      }
   }
   fill_primitive_cache<NumComponents, EosMethod>(
      launch.primitive, launch.conserved, launch.eos
   );
}

template <int NumComponents>
void fill_newtonian_selected_cache(PrimitiveCacheLaunch const& launch)
{
   if (launch.eos.method == EOSIdealGas) {
      fill_newtonian_cache<NumComponents, EOSIdealGas>(launch);
      return;
   }
   fill_newtonian_cache<NumComponents, EOSRuntime>(launch);
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
      *launch.cell_spacetime,
      launch.excision,
      launch.exterior
   );
}

template <PhysicalSystem System, int EosMethod>
void project_selected_exterior(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   EOSParams const& eos,
   RelativisticExteriorState const& exterior,
   int ghost_cells,
   amrex::Long* c2p_fallback_count,
   amrex::Long* atmosphere_reset_count,
   amrex::iMultiFab const* black_hole_mask,
   BlackHoleMatterParameters black_hole_matter,
   amrex::Long* interior_bound_count,
   amrex::Long* interior_reset_count
)
{
   project_relativistic_exterior<System, EosMethod>(
      conserved,
      eos,
      exterior,
      StoredSpacetimeCellMetricFactory{cell_spacetime},
      ghost_cells,
      c2p_fallback_count,
      atmosphere_reset_count,
      black_hole_mask,
      black_hole_matter,
      interior_bound_count,
      interior_reset_count
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
      if (launch.eos.method == EOSIdealGas) {
         fill_base_primitive_cache<EOSIdealGas>(
            launch.primitive,
            launch.conserved,
            launch.eos,
            launch.physical_system,
            *launch.geometry,
            launch.fixed_spacetime,
            launch.time,
            launch.mesh_coordinate,
            launch.mesh_axes,
            launch.canonical_face_b,
            launch.canonical_face_b_cell_mask
         );
      } else {
         fill_base_primitive_cache<EOSRuntime>(
            launch.primitive,
            launch.conserved,
            launch.eos,
            launch.physical_system,
            *launch.geometry,
            launch.fixed_spacetime,
            launch.time,
            launch.mesh_coordinate,
            launch.mesh_axes,
            launch.canonical_face_b,
            launch.canonical_face_b_cell_mask
         );
      }
      return;
   }
   switch (launch.physical_system) {
   case PhysicalSystem::HD:
      if (launch.num_components == NVAR_HYDRO) {
         fill_newtonian_selected_cache<NVAR_HYDRO>(launch);
         return;
      }
      if (launch.num_components == NVAR_HYDRO + 1) {
         fill_newtonian_selected_cache<NVAR_HYDRO + 1>(launch);
         return;
      }
      break;
   case PhysicalSystem::MHD:
      if (launch.num_components == NVAR_MHD) {
         fill_newtonian_selected_cache<NVAR_MHD>(launch);
         return;
      }
      if (launch.num_components == NVAR_MHD + 1) {
         fill_newtonian_selected_cache<NVAR_MHD + 1>(launch);
         return;
      }
      break;
   case PhysicalSystem::RHD:
      if (launch.eos.method == EOSIdealGas) {
         fill_rhd_cache<EOSIdealGas>(launch);
         return;
      }
      fill_rhd_cache<EOSRuntime>(launch);
      return;
   case PhysicalSystem::RMHD:
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
         launch.cell_spacetime != nullptr,
         "RMHD primitive recovery requires cell-centered spacetime"
      );
      fill_rmhd_primitive_cache(
         launch.primitive,
         launch.conserved,
         launch.eos,
         *launch.cell_spacetime,
         launch.geometry,
         launch.canonical_face_b,
         launch.canonical_face_b_cell_mask,
         launch.excision,
         launch.exterior
      );
      return;
   }
   amrex::Abort("unsupported state layout in primitive-cache factory");
}

void apply_relativistic_exterior(
   amrex::MultiFab& conserved,
   amrex::MultiFab const& cell_spacetime,
   PhysicalSystem physical_system,
   EOSParams const& eos,
   RelativisticExteriorState const& exterior,
   int ghost_cells,
   amrex::Long* c2p_fallback_count,
   amrex::Long* atmosphere_reset_count,
   amrex::iMultiFab const* black_hole_mask,
   BlackHoleMatterParameters black_hole_matter,
   amrex::Long* interior_bound_count,
   amrex::Long* interior_reset_count
)
{
   if (physical_system == PhysicalSystem::RMHD) {
      project_selected_exterior<PhysicalSystem::RMHD, EOSRuntime>(
         conserved,
         cell_spacetime,
         eos,
         exterior,
         ghost_cells,
         c2p_fallback_count,
         atmosphere_reset_count,
         black_hole_mask,
         black_hole_matter,
         interior_bound_count,
         interior_reset_count
      );
      return;
   }
   if (physical_system != PhysicalSystem::RHD) {
      return;
   }
   project_selected_exterior<PhysicalSystem::RHD, EOSRuntime>(
      conserved, cell_spacetime, eos, exterior, ghost_cells,
      c2p_fallback_count, atmosphere_reset_count, black_hole_mask,
      black_hole_matter, interior_bound_count, interior_reset_count
   );
}

} // namespace pyfluid_amrex
