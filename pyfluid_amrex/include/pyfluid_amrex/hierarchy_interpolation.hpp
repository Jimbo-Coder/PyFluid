#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <vector>

#include "pyfluid_amrex/mesh_geometry.hpp"

namespace pyfluid_amrex {

// Reusable interpolation geometry for arbitrary physical points. The plan owns
// no field data, so one surface construction can evaluate many diagnostics.
class HierarchyInterpolationPlan {
public:
   HierarchyInterpolationPlan(
      std::vector<amrex::GpuArray<amrex::Real, 3>> physical_points,
      int coordinate_kind,
      MeshAxes axes,
      std::vector<amrex::Geometry> const& geometry,
      std::vector<amrex::BoxArray> const& grids,
      std::vector<amrex::DistributionMapping> const& distribution,
      amrex::IndexType centering
   );

   std::vector<amrex::Real> interpolate(
      std::vector<std::unique_ptr<amrex::MultiFab>> const& hierarchy,
      int first_component,
      int num_components
   ) const;

   std::size_t size() const noexcept
   {
      return points_.size();
   }

private:
   struct Point {
      int level = -1;
      int box = -1;
      int owner = -1;
      amrex::IntVect anchor{};
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> fraction{};
   };

   std::vector<Point> points_;
};

} // namespace pyfluid_amrex
