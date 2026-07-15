#pragma once

#include <AMReX_Box.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cmath>

#include "pyfluid_amrex/mapped_patch_fill.hpp"
#include "pyfluid_amrex/patch_metrics.hpp"
#include "pyfluid_amrex/riemann.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct MappedPatchBoundaryFace2D {
   int patch_id = -1;
   int neighbor_patch_id = -1;
   int dir = 0;
   int side = 0;
   int face_i = 0;
   int face_j = 0;
   amrex::IntVect inside_cell{AMREX_D_DECL(0, 0, 0)};
   amrex::IntVect inside_local_cell{AMREX_D_DECL(0, 0, 0)};
   PatchCoordinate outside_global{};
   PhysicalVector p0{};
   PhysicalVector p1{};
   PhysicalVector normal{};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real distance_to_line(PhysicalVector point, PhysicalVector line_point, PhysicalVector line_unit) noexcept
{
   return norm(cross(point - line_point, line_unit));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState scale_flux(HydroState state, amrex::Real factor) noexcept
{
   return {
      factor * state.rho,
      factor * state.mom1,
      factor * state.mom2,
      factor * state.mom3,
      factor * state.energy,
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState add_flux(HydroState a, HydroState b) noexcept
{
   return {
      a.rho + b.rho,
      a.mom1 + b.mom1,
      a.mom2 + b.mom2,
      a.mom3 + b.mom3,
      a.energy + b.energy,
   };
}

inline PhysicalVector patch_layout_face_endpoint_2d(
   PatchGridDescriptor const& grid_patch,
   PatchDescriptor const& layout_patch,
   int dir,
   int face_i,
   int face_j,
   amrex::Real transverse_offset
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(grid_patch, layout_patch, dir, face_i, face_j, transverse_offset);
   amrex::Abort("mapped patch interface endpoint helper is implemented for 2D only");
   return physical_vector(0.0, 0.0, 0.0);
#else
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> offsets{};
   offsets[dir] = 0.0;
   offsets[1 - dir] = transverse_offset;
   const PatchCoordinate local = patch_layout_local_coordinate(
      grid_patch,
      layout_patch,
      amrex::IntVect(AMREX_D_DECL(face_i, face_j, 0)),
      offsets
   );
   const PatchCoordinate global = patch_local_to_global(grid_patch.map, local);
   return physical_vector(global.q1, global.q2, global.q3);
#endif
}

inline HydroPrimitive mapped_patch_source_hydro_primitive(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell
)
{
   return {
      mapped_patch_source_value(table, grid_system, patch_id, local_cell, RHO),
      mapped_patch_source_value(table, grid_system, patch_id, local_cell, MOM1),
      mapped_patch_source_value(table, grid_system, patch_id, local_cell, MOM2),
      mapped_patch_source_value(table, grid_system, patch_id, local_cell, MOM3),
      mapped_patch_source_value(table, grid_system, patch_id, local_cell, ENERGY),
   };
}

inline void resolve_mapped_patch_sample_cell(
   PatchGridSystem const& grid_system,
   int source_patch_id,
   amrex::IntVect const& requested_local_cell,
   int& resolved_patch_id,
   amrex::IntVect& resolved_local_cell
)
{
   PatchGridDescriptor const& source_patch = grid_system.patches[source_patch_id];
   const PatchCoordinate requested_local =
      patch_grid_cell_center(source_patch, requested_local_cell);
   const PatchCoordinate requested_global =
      patch_local_to_global(source_patch.map, requested_local);
   const int owner_patch_id =
      patch_grid_system_owner_patch_id(grid_system, requested_global);
   if (patch_grid_contains_local_cell(source_patch, requested_local_cell)) {
      if (owner_patch_id == source_patch_id) {
         resolved_patch_id = source_patch_id;
         resolved_local_cell = requested_local_cell;
         return;
      }
      if (owner_patch_id >= 0 && owner_patch_id < static_cast<int>(grid_system.patches.size())) {
         PatchGridDescriptor const& owner_patch = grid_system.patches[owner_patch_id];
         PatchCoordinate const owner_local =
            patch_global_to_local(owner_patch.map, requested_global);
         resolved_patch_id = owner_patch_id;
         resolved_local_cell = patch_grid_nearest_cell(owner_patch, owner_local);
         return;
      }
   }

   if (owner_patch_id >= 0 && owner_patch_id < static_cast<int>(grid_system.patches.size())) {
      PatchGridDescriptor const& owner_patch = grid_system.patches[owner_patch_id];
      PatchCoordinate const owner_local =
         patch_global_to_local(owner_patch.map, requested_global);
      resolved_patch_id = owner_patch_id;
      resolved_local_cell = patch_grid_nearest_cell(owner_patch, owner_local);
      return;
   }

   resolved_patch_id = source_patch_id;
   resolved_local_cell = patch_grid_clamped_local_cell(source_patch, requested_local_cell);
}

inline amrex::Real mapped_patch_source_component_at_offset(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int dir,
   int offset,
   int component
)
{
   amrex::IntVect requested_cell = local_cell;
   requested_cell[dir] += offset;
   int resolved_patch_id = patch_id;
   amrex::IntVect resolved_cell(AMREX_D_DECL(0, 0, 0));
   resolve_mapped_patch_sample_cell(
      grid_system,
      patch_id,
      requested_cell,
      resolved_patch_id,
      resolved_cell
   );
   return mapped_patch_source_value(table, grid_system, resolved_patch_id, resolved_cell, component);
}

inline bool mapped_patch_source_valid_at_offset(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int dir,
   int offset
)
{
   amrex::IntVect requested_cell = local_cell;
   requested_cell[dir] += offset;
   int resolved_patch_id = patch_id;
   amrex::IntVect resolved_cell(AMREX_D_DECL(0, 0, 0));
   resolve_mapped_patch_sample_cell(
      grid_system,
      patch_id,
      requested_cell,
      resolved_patch_id,
      resolved_cell
   );
   return mapped_patch_source_valid(table, grid_system, resolved_patch_id, resolved_cell);
}

template <int ReconMethod>
inline bool mapped_patch_reconstruction_stencil_valid(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int dir,
   int side
)
{
   if constexpr (is_five_point_reconstruction(ReconMethod)) {
      amrex::IntVect anchor_cell = local_cell;
      if (side < 0) {
         anchor_cell[dir] -= 1;
      }
      for (int offset = -2; offset <= 3; ++offset) {
         if (!mapped_patch_source_valid_at_offset(table, grid_system, patch_id, anchor_cell, dir, offset)) {
            return false;
         }
      }
      return true;
   } else if constexpr (ReconMethod == ReconPWL) {
      for (int offset = -1; offset <= 1; ++offset) {
         if (!mapped_patch_source_valid_at_offset(table, grid_system, patch_id, local_cell, dir, offset)) {
            return false;
         }
      }
      return true;
   } else {
      return mapped_patch_source_valid_at_offset(table, grid_system, patch_id, local_cell, dir, 0);
   }
}

inline HydroPrimitive mapped_patch_source_hydro_primitive_at_offset(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int dir,
   int offset
)
{
   return {
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, offset, RHO),
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, offset, MOM1),
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, offset, MOM2),
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, offset, MOM3),
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, offset, ENERGY),
   };
}

template <int ReconMethod>
inline ScalarFaceValues mapped_patch_five_point_component_face_values(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& local_cell,
   int dir,
   int component
)
{
   const amrex::Real vmm =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, -2, component);
   const amrex::Real vm =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, -1, component);
   const amrex::Real v0 =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, 0, component);
   const amrex::Real vp =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, 1, component);
   const amrex::Real vpp =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, 2, component);
   const amrex::Real vppp =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, local_cell, dir, 3, component);
   return {
      reconstruct_left<ReconMethod>(vmm, vm, v0, vp, vpp),
      reconstruct_right<ReconMethod>(vm, v0, vp, vpp, vppp),
   };
}

template <int ReconMethod>
inline HydroPrimitive mapped_patch_reconstructed_hydro_primitive(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& inside_local_cell,
   int dir,
   int side,
   EOSParams const& eos
)
{
   HydroPrimitive face{};
   HydroPrimitive const center =
      mapped_patch_source_hydro_primitive_at_offset(table, grid_system, patch_id, inside_local_cell, dir, 0);
   if constexpr (ReconMethod == ReconPWL) {
      const HydroPrimitive qm =
         mapped_patch_source_hydro_primitive_at_offset(table, grid_system, patch_id, inside_local_cell, dir, -1);
      const HydroPrimitive qp =
         mapped_patch_source_hydro_primitive_at_offset(table, grid_system, patch_id, inside_local_cell, dir, 1);
      face = pwl_hydro_primitive(qm, center, qp, static_cast<amrex::Real>(side));
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      amrex::IntVect anchor_cell = inside_local_cell;
      if (side < 0) {
         anchor_cell[dir] -= 1;
      }
      const ScalarFaceValues rho =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, RHO);
      const ScalarFaceValues v1 =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, MOM1);
      const ScalarFaceValues v2 =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, MOM2);
      const ScalarFaceValues v3 =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, MOM3);
      const ScalarFaceValues pressure =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, ENERGY);
      if (side > 0) {
         face = {rho.left, v1.left, v2.left, v3.left, pressure.left};
      } else {
         face = {rho.right, v1.right, v2.right, v3.right, pressure.right};
      }
   } else {
      face = center;
   }
   return positivity_limit_hydro_primitive(face, center, eos);
}

template <int ReconMethod>
inline amrex::Real mapped_patch_reconstructed_scalar(
   MappedPatchSourceTable const& table,
   PatchGridSystem const& grid_system,
   int patch_id,
   amrex::IntVect const& inside_local_cell,
   int dir,
   int side,
   int component
)
{
   const amrex::Real center =
      mapped_patch_source_component_at_offset(table, grid_system, patch_id, inside_local_cell, dir, 0, component);
   if constexpr (ReconMethod == ReconPWL) {
      const amrex::Real qm =
         mapped_patch_source_component_at_offset(table, grid_system, patch_id, inside_local_cell, dir, -1, component);
      const amrex::Real qp =
         mapped_patch_source_component_at_offset(table, grid_system, patch_id, inside_local_cell, dir, 1, component);
      return center + 0.5 * static_cast<amrex::Real>(side) * mc_slope(qm, center, qp);
   } else if constexpr (is_five_point_reconstruction(ReconMethod)) {
      amrex::IntVect anchor_cell = inside_local_cell;
      if (side < 0) {
         anchor_cell[dir] -= 1;
      }
      const ScalarFaceValues value =
         mapped_patch_five_point_component_face_values<ReconMethod>(table, grid_system, patch_id, anchor_cell, dir, component);
      return side > 0 ? value.left : value.right;
   }
   return center;
}

template <int NComp>
struct MappedPatchFluxCorrection2D {
   int patch_id = -1;
   int dir = 0;
   int face_i = 0;
   int face_j = 0;
   std::array<amrex::Real, NComp> value{};
};

template <int NComp>
inline void add_mapped_patch_flux_correction(
   amrex::Vector<MappedPatchFluxCorrection2D<NComp>>& corrections,
   int patch_id,
   int dir,
   int face_i,
   int face_j,
   HydroState const& hydro_flux,
   amrex::Vector<amrex::Real> const& passive_flux,
   amrex::Real sign = 1.0
)
{
   for (MappedPatchFluxCorrection2D<NComp>& correction : corrections) {
      if (
         correction.patch_id == patch_id
         && correction.dir == dir
         && correction.face_i == face_i
         && correction.face_j == face_j
      ) {
         correction.value[RHO] += sign * hydro_flux.rho;
         correction.value[MOM1] += sign * hydro_flux.mom1;
         correction.value[MOM2] += sign * hydro_flux.mom2;
         correction.value[MOM3] += sign * hydro_flux.mom3;
         correction.value[ENERGY] += sign * hydro_flux.energy;
         if constexpr (NComp > NVAR_HYDRO) {
            for (int n = NVAR_HYDRO; n < NComp; ++n) {
               correction.value[n] += sign * passive_flux[n - NVAR_HYDRO];
            }
         }
         return;
      }
   }

   MappedPatchFluxCorrection2D<NComp> correction;
   correction.patch_id = patch_id;
   correction.dir = dir;
   correction.face_i = face_i;
   correction.face_j = face_j;
   correction.value[RHO] = sign * hydro_flux.rho;
   correction.value[MOM1] = sign * hydro_flux.mom1;
   correction.value[MOM2] = sign * hydro_flux.mom2;
   correction.value[MOM3] = sign * hydro_flux.mom3;
   correction.value[ENERGY] = sign * hydro_flux.energy;
   if constexpr (NComp > NVAR_HYDRO) {
      for (int n = NVAR_HYDRO; n < NComp; ++n) {
         correction.value[n] = sign * passive_flux[n - NVAR_HYDRO];
      }
   }
   corrections.push_back(correction);
}

template <int NComp>
inline bool mapped_patch_flux_correction_exists(
   amrex::Vector<MappedPatchFluxCorrection2D<NComp>> const& corrections,
   int patch_id,
   int dir,
   int face_i,
   int face_j
)
{
   for (MappedPatchFluxCorrection2D<NComp> const& correction : corrections) {
      if (
         correction.patch_id == patch_id
         && correction.dir == dir
         && correction.face_i == face_i
         && correction.face_j == face_j
      ) {
         return true;
      }
   }
   return false;
}

template <int RiemannMethod, int EOSMethod>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState mapped_patch_hydro_flux_from_primitives(
   HydroPrimitive const& primitive_l,
   HydroPrimitive const& primitive_r,
   EOSParams const& eos,
   PhysicalVector normal
) noexcept
{
   if constexpr (RiemannMethod == RiemannHLL) {
      return hll_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   } else if constexpr (RiemannMethod == RiemannHLLE) {
      return hlle_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   } else if constexpr (RiemannMethod == RiemannHLLC) {
      return hllc_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   } else {
      return llf_flux_from_primitives_normal<EOSMethod>(primitive_l, primitive_r, eos, normal);
   }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState mapped_patch_reflecting_wall_flux(
   HydroPrimitive const& primitive,
   PhysicalVector normal
) noexcept
{
   return {
      0.0,
      primitive.pressure * normal.x,
      primitive.pressure * normal.y,
      primitive.pressure * normal.z,
      0.0,
   };
}

inline amrex::Vector<MappedPatchBoundaryFace2D> mapped_patch_boundary_faces_2d(
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(grid_system, layout);
   amrex::Abort("mapped patch boundary-face table is implemented for 2D only");
   return {};
#else
   amrex::Vector<MappedPatchBoundaryFace2D> faces;
   constexpr amrex::Real epsilon = 1.0e-10;
   for (PatchDescriptor const& layout_patch : layout.patches) {
      PatchGridDescriptor const& grid_patch = patch_grid_descriptor_for_layout_patch(grid_system, layout_patch);
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         const int transverse_dir = 1 - dir;
         for (int side = -1; side <= 1; side += 2) {
            const int face_index = side < 0
               ? layout_patch.owned_box.smallEnd(dir)
               : layout_patch.owned_box.bigEnd(dir) + 1;
            for (
               int transverse_index = layout_patch.owned_box.smallEnd(transverse_dir);
               transverse_index <= layout_patch.owned_box.bigEnd(transverse_dir);
               ++transverse_index
            ) {
               int face_i = dir == 0 ? face_index : transverse_index;
               int face_j = dir == 0 ? transverse_index : face_index;
               const PatchCoordinate face_local = patch_layout_face_coordinate(
                  grid_patch,
                  layout_patch,
                  dir,
                  face_i,
                  face_j,
                  0
               );
               const PatchCoordinate outside_local = patch_shifted_coordinate(
                  grid_patch,
                  face_local,
                  dir,
                  side * epsilon
               );
               const PatchCoordinate outside_global = patch_local_to_global(grid_patch.map, outside_local);
               const int neighbor_patch_id = patch_grid_system_owner_patch_id(grid_system, outside_global);
               if (neighbor_patch_id == layout_patch.nominal_patch_id) {
                  continue;
               }
               amrex::IntVect inside_cell(AMREX_D_DECL(face_i, face_j, 0));
               if (side > 0) {
                  inside_cell[dir] -= 1;
               }
               const amrex::IntVect inside_local_cell =
                  patch_layout_cell_to_patch_local_cell(layout_patch, inside_cell);
               const PatchFaceMetric metric = patch_face_metric(grid_patch, face_local, dir);
               faces.push_back(MappedPatchBoundaryFace2D{
                  layout_patch.nominal_patch_id,
                  neighbor_patch_id,
                  dir,
                  side,
                  face_i,
                  face_j,
                  inside_cell,
                  inside_local_cell,
                  outside_global,
                  patch_layout_face_endpoint_2d(grid_patch, layout_patch, dir, face_i, face_j, 0.0),
                  patch_layout_face_endpoint_2d(grid_patch, layout_patch, dir, face_i, face_j, 1.0),
                  metric.unit_normal,
               });
            }
         }
      }
   }
   return faces;
#endif
}

template <int NComp>
inline bool mapped_patch_flux_is_finite(
   amrex::Array4<amrex::Real const> const& flux,
   int i,
   int j,
   int k
)
{
   for (int n = 0; n < NComp; ++n) {
      if (!std::isfinite(flux(i, j, k, n))) {
         return false;
      }
   }
   return true;
}

template <int NComp>
inline void store_mapped_patch_boundary_flux(
   amrex::Array4<amrex::Real> const& flux,
   int i,
   int j,
   int k,
   HydroState const& hydro_flux
)
{
   flux(i, j, k, RHO) = hydro_flux.rho;
   flux(i, j, k, MOM1) = hydro_flux.mom1;
   flux(i, j, k, MOM2) = hydro_flux.mom2;
   flux(i, j, k, MOM3) = hydro_flux.mom3;
   flux(i, j, k, ENERGY) = hydro_flux.energy;
   if constexpr (NComp > NVAR_HYDRO) {
      for (int n = NVAR_HYDRO; n < NComp; ++n) {
         flux(i, j, k, n) = 0.0;
      }
   }
}

template <int NComp, int ReconMethod, int RiemannMethod, int EOSMethod>
inline void correct_mapped_patch_hydro_interface_fluxes(
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   EOSParams const& eos,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(fluxes, primitive, eos, grid_system, layout);
   amrex::Abort("mapped patch conservative interface flux correction is currently implemented for 2D hydro");
#else
   MappedPatchSourceTable const primitive_table =
      make_mapped_patch_source_table(primitive, grid_system, layout);
   amrex::Vector<MappedPatchBoundaryFace2D> const faces =
      mapped_patch_boundary_faces_2d(grid_system, layout);
   amrex::Vector<MappedPatchFluxCorrection2D<NComp>> corrections;

   for (MappedPatchBoundaryFace2D const& face : faces) {
      if (face.neighbor_patch_id < 0) {
         const PhysicalVector tangent = face.p1 - face.p0;
         const amrex::Real face_length = norm(tangent);
         const HydroPrimitive inside =
            mapped_patch_source_hydro_primitive(primitive_table, grid_system, face.patch_id, face.inside_local_cell);
         add_mapped_patch_flux_correction<NComp>(
            corrections,
            face.patch_id,
            face.dir,
            face.face_i,
            face.face_j,
            scale_flux(mapped_patch_reflecting_wall_flux(inside, face.normal), face_length),
            amrex::Vector<amrex::Real>{}
         );
         continue;
      }
      for (MappedPatchBoundaryFace2D const& neighbor : faces) {
         if (
            neighbor.patch_id != face.neighbor_patch_id
            || neighbor.neighbor_patch_id != face.patch_id
         ) {
            continue;
         }
         if (face.patch_id > neighbor.patch_id) {
            continue;
         }

         const PhysicalVector tangent = face.p1 - face.p0;
         const amrex::Real face_length = norm(tangent);
         const PhysicalVector neighbor_tangent = neighbor.p1 - neighbor.p0;
         const amrex::Real neighbor_length = norm(neighbor_tangent);
         if (face_length <= 0.0 || neighbor_length <= 0.0) {
            continue;
         }
         const PhysicalVector tangent_unit = scale(tangent, 1.0 / face_length);
         const amrex::Real line_tolerance = 1.0e-10 * amrex::max(face_length, amrex::max(neighbor_length, 1.0));
         if (
            distance_to_line(neighbor.p0, face.p0, tangent_unit) > line_tolerance
            || distance_to_line(neighbor.p1, face.p0, tangent_unit) > line_tolerance
         ) {
            continue;
         }

         const amrex::Real neighbor_t0 = dot(neighbor.p0 - face.p0, tangent_unit);
         const amrex::Real neighbor_t1 = dot(neighbor.p1 - face.p0, tangent_unit);
         const amrex::Real overlap_lo = amrex::max(0.0, amrex::min(neighbor_t0, neighbor_t1));
         const amrex::Real overlap_hi = amrex::min(face_length, amrex::max(neighbor_t0, neighbor_t1));
         const amrex::Real overlap_length = overlap_hi - overlap_lo;
         if (overlap_length <= 1.0e-12 * amrex::max(face_length, 1.0)) {
            continue;
         }
         const bool requested_stencil_available =
            mapped_patch_reconstruction_stencil_valid<ReconMethod>(
               primitive_table,
               grid_system,
               face.patch_id,
               face.inside_local_cell,
               face.dir,
               face.side
            )
            && mapped_patch_reconstruction_stencil_valid<ReconMethod>(
               primitive_table,
               grid_system,
               neighbor.patch_id,
               neighbor.inside_local_cell,
               neighbor.dir,
               neighbor.side
            );
         if (!requested_stencil_available) {
            amrex::Abort("mapped-patch interface correction cannot assemble the selected reconstruction stencil");
         }
         HydroPrimitive current = mapped_patch_reconstructed_hydro_primitive<ReconMethod>(
            primitive_table,
            grid_system,
            face.patch_id,
            face.inside_local_cell,
            face.dir,
            face.side,
            eos
         );
         HydroPrimitive other = mapped_patch_reconstructed_hydro_primitive<ReconMethod>(
            primitive_table,
            grid_system,
            neighbor.patch_id,
            neighbor.inside_local_cell,
            neighbor.dir,
            neighbor.side,
            eos
         );
         HydroPrimitive primitive_l = current;
         HydroPrimitive primitive_r = other;
         if (face.side < 0) {
            primitive_l = other;
            primitive_r = current;
         }
         const HydroState overlap_flux = mapped_patch_hydro_flux_from_primitives<RiemannMethod, EOSMethod>(
            primitive_l,
            primitive_r,
            eos,
            face.normal
         );
         const HydroState integrated_flux = scale_flux(overlap_flux, overlap_length);

         amrex::Vector<amrex::Real> passive_flux;
         if constexpr (NComp > NVAR_HYDRO) {
            passive_flux.assign(NComp - NVAR_HYDRO, 0.0);
            const bool use_left_passive = overlap_flux.rho >= 0.0;
            for (int n = NVAR_HYDRO; n < NComp; ++n) {
               const amrex::Real passive_l = mapped_patch_reconstructed_scalar<ReconMethod>(
                  primitive_table,
                  grid_system,
                  face.side < 0 ? neighbor.patch_id : face.patch_id,
                  face.side < 0 ? neighbor.inside_local_cell : face.inside_local_cell,
                  face.side < 0 ? neighbor.dir : face.dir,
                  face.side < 0 ? neighbor.side : face.side,
                  n
               );
               const amrex::Real passive_r = mapped_patch_reconstructed_scalar<ReconMethod>(
                  primitive_table,
                  grid_system,
                  face.side < 0 ? face.patch_id : neighbor.patch_id,
                  face.side < 0 ? face.inside_local_cell : neighbor.inside_local_cell,
                  face.side < 0 ? face.dir : neighbor.dir,
                  face.side < 0 ? face.side : neighbor.side,
                  n
               );
               passive_flux[n - NVAR_HYDRO] =
                  overlap_flux.rho * (use_left_passive ? passive_l : passive_r) * overlap_length;
            }
         }

         add_mapped_patch_flux_correction<NComp>(
            corrections,
            face.patch_id,
            face.dir,
            face.face_i,
            face.face_j,
            integrated_flux,
            passive_flux
         );
         const amrex::Real neighbor_sign = dot(face.normal, neighbor.normal) >= 0.0 ? 1.0 : -1.0;
         add_mapped_patch_flux_correction<NComp>(
            corrections,
            neighbor.patch_id,
            neighbor.dir,
            neighbor.face_i,
            neighbor.face_j,
            integrated_flux,
            passive_flux,
            neighbor_sign
         );
      }
   }

   for (MappedPatchBoundaryFace2D const& face : faces) {
      if (
         mapped_patch_flux_correction_exists<NComp>(
            corrections, face.patch_id, face.dir, face.face_i, face.face_j
         )
      ) {
         continue;
      }
      const PhysicalVector tangent = face.p1 - face.p0;
      const amrex::Real face_length = norm(tangent);
      const HydroPrimitive inside =
         mapped_patch_source_hydro_primitive(primitive_table, grid_system, face.patch_id, face.inside_local_cell);
      if (face.neighbor_patch_id >= 0) {
         amrex::Abort("unmatched internal mapped-patch face; patch maps or interface overlap reconciliation are inconsistent");
      }
      add_mapped_patch_flux_correction<NComp>(
         corrections,
         face.patch_id,
         face.dir,
         face.face_i,
         face.face_j,
         scale_flux(mapped_patch_reflecting_wall_flux(inside, face.normal), face_length),
         amrex::Vector<amrex::Real>{}
      );
   }

   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::MultiFab& flux = *fluxes[dir];
      for (amrex::MFIter mfi(flux); mfi.isValid(); ++mfi) {
         const amrex::Box& face_box = mfi.validbox();
         const amrex::Box cell_box = amrex::enclosedCells(face_box, dir);
         PatchDescriptor const& layout_patch = patch_layout_descriptor_containing_box(layout, cell_box);
         const auto out = flux.array(mfi);
         for (MappedPatchFluxCorrection2D<NComp> const& correction : corrections) {
            if (correction.dir != dir || correction.patch_id != layout_patch.nominal_patch_id) {
               continue;
            }
            const amrex::IntVect face_index(AMREX_D_DECL(correction.face_i, correction.face_j, 0));
            if (!face_box.contains(face_index)) {
               continue;
            }
            for (int n = 0; n < NComp; ++n) {
               out(correction.face_i, correction.face_j, 0, n) = correction.value[n];
            }
         }
         for (MappedPatchBoundaryFace2D const& face : faces) {
            if (face.dir != dir || face.patch_id != layout_patch.nominal_patch_id) {
               continue;
            }
            const amrex::IntVect face_index(AMREX_D_DECL(face.face_i, face.face_j, 0));
            if (!face_box.contains(face_index)) {
               continue;
            }
            if (mapped_patch_flux_is_finite<NComp>(flux.const_array(mfi), face.face_i, face.face_j, 0)) {
               continue;
            }
            const PhysicalVector tangent = face.p1 - face.p0;
            const amrex::Real face_length = norm(tangent);
            const HydroPrimitive inside =
               mapped_patch_source_hydro_primitive(primitive_table, grid_system, face.patch_id, face.inside_local_cell);
            store_mapped_patch_boundary_flux<NComp>(
               out,
               face.face_i,
               face.face_j,
               0,
               scale_flux(mapped_patch_reflecting_wall_flux(inside, face.normal), face_length)
            );
         }
      }
   }
#endif
}

} // namespace pyfluid_amrex
