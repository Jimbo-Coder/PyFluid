#pragma once

#include <AMReX_MFInterpolater.H>

namespace pyfluid_amrex {

class FourthOrderNodalInterpolater final : public amrex::MFInterpolater {
public:
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
};

amrex::MFInterpolater* fourth_order_nodal_interpolater() noexcept;

} // namespace pyfluid_amrex
