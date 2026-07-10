#pragma once

#include <AMReX_Config.H>

#if (AMREX_SPACEDIM >= 2)
#include <AMReX_iMultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>

#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

class CtEdgeFluxRegister {
public:
   CtEdgeFluxRegister() = default;

   CtEdgeFluxRegister(
      amrex::BoxArray const& fine_ba,
      amrex::BoxArray const& coarse_ba,
      amrex::DistributionMapping const& fine_dm,
      amrex::DistributionMapping const& coarse_dm,
      amrex::Geometry const& fine_geom,
      amrex::Geometry const& coarse_geom,
      int ncomp = 1
   );

   void define(
      amrex::BoxArray const& fine_ba,
      amrex::BoxArray const& coarse_ba,
      amrex::DistributionMapping const& fine_dm,
      amrex::DistributionMapping const& coarse_dm,
      amrex::Geometry const& fine_geom,
      amrex::Geometry const& coarse_geom,
      int ncomp = 1
   );

   void reset();

#if (AMREX_SPACEDIM == 3)
   void CrseAdd(amrex::MFIter const& mfi, amrex::Array<amrex::FArrayBox const*, 3> const& edge_line, amrex::Real dt);
   void FineAdd(amrex::MFIter const& mfi, amrex::Array<amrex::FArrayBox const*, 3> const& edge_line, amrex::Real dt);
#else
   void CrseAdd(amrex::MFIter const& mfi, amrex::FArrayBox const& edge_line, amrex::Real dt);
   void FineAdd(amrex::MFIter const& mfi, amrex::FArrayBox const& edge_line, amrex::Real dt);
#endif

   void Reflux(
      amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& coarse_face_b,
      int mesh_coordinate,
      MeshAxes mesh_axes
   ) const;

private:
   amrex::Geometry fine_geom_;
   amrex::Geometry coarse_geom_;
   amrex::IntVect ratio_;
   int ncomp_ = 1;

#if (AMREX_SPACEDIM == 3)
   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> coarse_edge_line_;
   amrex::Array<amrex::Array<amrex::MultiFab, 2>, AMREX_SPACEDIM * 2> fine_edge_line_;
   amrex::Array<amrex::iMultiFab, AMREX_SPACEDIM> fine_mask_;
#else
   amrex::MultiFab coarse_edge_line_;
   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM * 2> fine_edge_line_;
   amrex::iMultiFab fine_mask_;
#endif

   amrex::LayoutData<int> has_coarse_fine_;
};

} // namespace pyfluid_amrex
#endif
