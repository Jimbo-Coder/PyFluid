#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TagBox.H>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pyfluid_amrex/apparent_horizon_finder.hpp"

namespace pyfluid_amrex {

struct RunConfig;

// Owns the runtime search policy, persisted single-horizon history, and
// geometry-only masks.  Neither mask changes finite-volume ownership or
// boundary conditions; true excision remains a separate evolution contract.
class ApparentHorizonTracker {
public:
   ApparentHorizonTracker(RunConfig const& config, int num_levels);

   bool enabled() const noexcept;
   bool refinement_tracking_active() const noexcept;
   bool search_due(int step) const noexcept;
   int radial_points() const noexcept;
   int polar_points() const noexcept;
   amrex::Real minimum_search_radius() const noexcept;
   amrex::Real maximum_search_radius() const noexcept;
   amrex::Real initial_radius() const noexcept;
   std::array<amrex::Real, 3> const& center() const noexcept;
   ApparentHorizonFinderParameters const& finder_parameters() const noexcept;
   std::optional<ApparentHorizonSurface> previous_surface() const;
   void restore_latest_surface(int step, std::string const& output_dir);

   void accept_result(
      int step,
      amrex::Real time,
      ApparentHorizonResult result,
      std::string const& output_dir
   );

   ApparentHorizonResult const* result() const noexcept;
   amrex::iMultiFab const* apparent_horizon_mask(
      int level,
      amrex::MultiFab const& state,
      amrex::Geometry const& geometry
   );
   amrex::iMultiFab const* excision_mask(
      int level,
      amrex::MultiFab const& state,
      amrex::Geometry const& geometry
   );
   void tag_refinement_band(
      int level,
      amrex::TagBoxArray& tags,
      amrex::MultiFab const& state,
      amrex::Geometry const& geometry
   );
   void invalidate_level(int level) noexcept;

private:
   void refresh_level(
      int level,
      amrex::MultiFab const& state,
      amrex::Geometry const& geometry
   );
   void write_result(
      int step,
      amrex::Real time,
      ApparentHorizonResult const& result,
      std::string const& output_dir
   ) const;

   bool enabled_ = false;
   int find_interval_ = 8;
   int radial_points_ = 48;
   int polar_points_ = 32;
   amrex::Real minimum_search_radius_ = 0.0;
   amrex::Real maximum_search_radius_ = 0.0;
   amrex::Real initial_radius_ = 0.0;
   std::array<amrex::Real, 3> center_{};
   ApparentHorizonFinderParameters finder_parameters_{};
   amrex::Real refine_band_cells_ = 0.0;
   amrex::Real excision_fraction_ = 0.8;
   std::optional<ApparentHorizonSurface> restored_surface_;
   std::optional<ApparentHorizonResult> result_;
   amrex::Gpu::DeviceVector<amrex::Real> device_radius_;
   std::vector<std::unique_ptr<amrex::MultiFab>> signed_distance_;
   std::vector<std::unique_ptr<amrex::iMultiFab>> apparent_horizon_mask_;
   std::vector<std::unique_ptr<amrex::iMultiFab>> excision_mask_;
   std::vector<int> level_revision_;
   int surface_revision_ = 0;
};

} // namespace pyfluid_amrex
