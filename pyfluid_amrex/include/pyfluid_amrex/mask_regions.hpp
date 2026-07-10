#pragma once

#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/method_codes.hpp"

namespace pyfluid_amrex {

constexpr int MaskKindMixed = 0;
constexpr int MaskKindZero = 1;
constexpr int MaskKindNonzero = 2;

inline amrex::Box cell_box_for_face_tile(amrex::Box const& face_box, int direction)
{
   amrex::IntVect small = face_box.smallEnd();
   amrex::IntVect big = face_box.bigEnd();
   small[direction] -= 1;
   return amrex::Box(small, big);
}

inline amrex::Box clipped_to_domain(amrex::Box region, amrex::Box const& domain)
{
   region &= domain;
   return region;
}

inline int reconstruction_solid_mask_radius(int recon_method)
{
   if (recon_method == ReconWENOZ || recon_method == ReconWENOZChar) {
      return 2;
   }
   if (recon_method == ReconPWL) {
      return 1;
   }
   return 0;
}

inline amrex::Box solid_flux_stencil_region(amrex::Box const& face_box, int direction, int recon_method, amrex::Box const& domain)
{
   amrex::Box region = cell_box_for_face_tile(face_box, direction);
   const int radius = reconstruction_solid_mask_radius(recon_method);
   if (radius > 0) {
      region.grow(direction, radius);
   }
   return clipped_to_domain(region, domain);
}

inline bool mask_region_all_zero(
   amrex::iMultiFab const& solid_mask,
   amrex::MFIter const& mfi,
   amrex::Box const& region
)
{
   amrex::Box clipped = region;
   clipped &= solid_mask[mfi].box();
   return clipped.ok() && clipped.numPts() == region.numPts() && solid_mask[mfi].max(clipped, 0) == 0;
}

inline bool mask_region_all_nonzero(
   amrex::iMultiFab const& solid_mask,
   amrex::MFIter const& mfi,
   amrex::Box const& region
)
{
   amrex::Box clipped = region;
   clipped &= solid_mask[mfi].box();
   return clipped.ok() && clipped.numPts() == region.numPts() && solid_mask[mfi].min(clipped, 0) != 0;
}

inline int classify_mask_region(amrex::iMultiFab const& mask, amrex::MFIter const& mfi, amrex::Box const& region)
{
   if (mask_region_all_zero(mask, mfi, region)) {
      return MaskKindZero;
   }
   if (mask_region_all_nonzero(mask, mfi, region)) {
      return MaskKindNonzero;
   }
   return MaskKindMixed;
}

inline int cached_mask_kind(amrex::Vector<int> const* kinds, amrex::MFIter const& mfi, amrex::iMultiFab const& mask, amrex::Box const& region)
{
   if (kinds == nullptr) {
      return classify_mask_region(mask, mfi, region);
   }
   const int index = mfi.index();
   if (index < 0 || index >= static_cast<int>(kinds->size())) {
      return classify_mask_region(mask, mfi, region);
   }
   return (*kinds)[index];
}

inline int cached_or_region_mask_kind(
   amrex::Vector<int> const* kinds,
   amrex::MFIter const& mfi,
   amrex::iMultiFab const& mask,
   amrex::Box const& region
)
{
   const int kind = cached_mask_kind(kinds, mfi, mask, region);
   return kind == MaskKindMixed ? classify_mask_region(mask, mfi, region) : kind;
}

inline bool solid_region_all_fluid(amrex::iMultiFab const& solid_mask, amrex::MFIter const& mfi, amrex::Box const& region)
{
   return mask_region_all_zero(solid_mask, mfi, region);
}

inline bool solid_region_all_solid(amrex::iMultiFab const& solid_mask, amrex::MFIter const& mfi, amrex::Box const& region)
{
   return mask_region_all_nonzero(solid_mask, mfi, region);
}

} // namespace pyfluid_amrex
