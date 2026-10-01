#pragma once

#include <AMReX_MFInterpolater.H>
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/generated/spatial_order_catalog.hpp"

namespace pyfluid_amrex {

class NodalInterpolater final : public amrex::MFInterpolater {
public:
   explicit NodalInterpolater(
      SphericalExcisionRegion region = {}, int spatial_order = 4
   );
   amrex::Box CoarseBox(amrex::Box const& fine, int ratio) override;
   amrex::Box CoarseBox(
      amrex::Box const& fine,
      amrex::IntVect const& ratio
   ) override;

   void interp(
      amrex::MultiFab const& coarse,
      int coarse_component,
      amrex::MultiFab& fine,
      int fine_component,
      int num_components,
      amrex::IntVect const& num_ghost,
      amrex::Geometry const& coarse_geometry,
      amrex::Geometry const& fine_geometry,
      amrex::Box const& destination_domain,
      amrex::IntVect const& ratio,
      amrex::Vector<amrex::BCRec> const& boundary_conditions,
      int boundary_component
   ) override;

private:
   SphericalExcisionRegion excision_region_;
   int spatial_order_ = 4;
};

amrex::MFInterpolater* nodal_interpolater(int spatial_order = 4) noexcept;

} // namespace pyfluid_amrex
