#include "pyfluid_amrex/apparent_horizon_tracker.hpp"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

constexpr amrex::Real pi = std::numbers::pi_v<amrex::Real>;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int wrap_device_index(int index, int count) noexcept
{
   index %= count;
   return index < 0 ? index + count : index;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int map_device_theta_index(
   int& theta_index,
   int num_theta,
   int num_phi
) noexcept
{
   int phi_shift = 0;
   while (theta_index < 0 || theta_index >= num_theta) {
      if (theta_index < 0) {
         theta_index = -theta_index - 1;
         phi_shift += num_phi / 2;
      }
      if (theta_index >= num_theta) {
         theta_index = 2 * num_theta - theta_index - 1;
         phi_shift += num_phi / 2;
      }
   }
   return phi_shift;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real device_surface_radius(
   amrex::Real const* radius,
   int num_theta,
   int num_phi,
   amrex::Real theta,
   amrex::Real phi
) noexcept
{
   phi = std::atan2(std::sin(phi), std::cos(phi));
   if (phi < 0.0_rt) {
      phi += 2.0_rt * pi;
   }
   const amrex::Real theta_grid = theta * num_theta / pi - 0.5_rt;
   const amrex::Real phi_grid = phi * num_phi / (2.0_rt * pi) - 0.5_rt;
   const int theta_lower = static_cast<int>(amrex::Math::floor(theta_grid));
   const int phi_lower = static_cast<int>(amrex::Math::floor(phi_grid));
   const amrex::Real theta_fraction = theta_grid - theta_lower;
   const amrex::Real phi_fraction = phi_grid - phi_lower;
   amrex::Real result = 0.0_rt;
   for (int theta_offset = 0; theta_offset < 2; ++theta_offset) {
      int mapped_theta = theta_lower + theta_offset;
      const int phi_shift = map_device_theta_index(
         mapped_theta, num_theta, num_phi
      );
      const amrex::Real theta_weight = theta_offset == 0
         ? 1.0_rt - theta_fraction : theta_fraction;
      for (int phi_offset = 0; phi_offset < 2; ++phi_offset) {
         const int mapped_phi = wrap_device_index(
            phi_lower + phi_offset + phi_shift, num_phi
         );
         const amrex::Real phi_weight = phi_offset == 0
            ? 1.0_rt - phi_fraction : phi_fraction;
         result += theta_weight * phi_weight
            * radius[mapped_theta * num_phi + mapped_phi];
      }
   }
   return result;
}

void fill_horizon_masks(
   amrex::Box const& box,
   amrex::Array4<amrex::Real> const& distance,
   amrex::Array4<int> const& horizon_mask,
   amrex::Array4<int> const& excision,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lower,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
   std::array<amrex::Real, 3> const& center,
   int num_theta,
   int num_phi,
   amrex::Real excision_fraction,
   SphericalExcisionRegion const& spacetime_excision,
   amrex::Real const* radius
)
{
   amrex::ParallelFor(
      box,
      [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real x = lower[0]
            + (static_cast<amrex::Real>(i) + 0.5_rt) * spacing[0];
         const amrex::Real y = lower[1]
            + (static_cast<amrex::Real>(j) + 0.5_rt) * spacing[1];
         const amrex::Real z = lower[2]
            + (static_cast<amrex::Real>(k) + 0.5_rt) * spacing[2];
         const amrex::Real relative_x = x - center[0];
         const amrex::Real relative_y = y - center[1];
         const amrex::Real relative_z = z - center[2];
         const amrex::Real coordinate_radius = std::sqrt(
            relative_x * relative_x + relative_y * relative_y
            + relative_z * relative_z
         );
         amrex::Real theta = 0.0_rt;
         amrex::Real phi = 0.0_rt;
         if (coordinate_radius > 0.0_rt) {
            theta = std::acos(amrex::max(
               -1.0_rt,
               amrex::min(1.0_rt, relative_z / coordinate_radius)
            ));
            phi = std::atan2(relative_y, relative_x);
         }
         const amrex::Real surface_radius = device_surface_radius(
            radius, num_theta, num_phi, theta, phi
         );
         distance(i, j, k) = coordinate_radius - surface_radius;
         horizon_mask(i, j, k) = coordinate_radius <= surface_radius
            ? 1 : 0;
         excision(i, j, k) = spacetime_excision.enabled()
            ? (spacetime_excision.contains({{x, y, z}}) ? 1 : 0)
            : (coordinate_radius <= excision_fraction * surface_radius ? 1 : 0);
      }
   );
}

} // namespace

ApparentHorizonTracker::ApparentHorizonTracker(
   RunConfig const& config,
   int num_levels
)
{
   mask_ghost_cells_ = config.ng;
   spacetime_excision_ = spacetime_excision_region(config);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(config);
   center_ = {spacetime.origin1, spacetime.origin2, spacetime.origin3};
   finder_parameters_.mass_scale = spacetime.mass;
   minimum_search_radius_ = 0.5_rt * spacetime.mass;
   maximum_search_radius_ = 3.0_rt * spacetime.mass;
   initial_radius_ = 0.8_rt * maximum_search_radius_;

   amrex::ParmParse parameters("horizon");
   parameters.queryAdd("enabled", enabled_);
   parameters.queryAdd("abort_on_failure", abort_on_failure_);
   parameters.queryAdd("find_int", find_interval_);
   parameters.queryAdd("radial_points", radial_points_);
   parameters.queryAdd("polar_points", polar_points_);
   parameters.queryAdd(
      "coarsest_polar_points", finder_parameters_.coarsest_num_theta
   );
   parameters.queryAdd("center1", center_[0]);
   parameters.queryAdd("center2", center_[1]);
   parameters.queryAdd("center3", center_[2]);
   parameters.queryAdd("minimum_search_radius", minimum_search_radius_);
   parameters.queryAdd("maximum_search_radius", maximum_search_radius_);
   parameters.queryAdd("initial_radius", initial_radius_);
   parameters.queryAdd("mass_scale", finder_parameters_.mass_scale);
   parameters.queryAdd("cfl", finder_parameters_.cfl);
   parameters.queryAdd("eta_times_mass", finder_parameters_.eta_times_mass);
   parameters.queryAdd(
      "theta_linf_times_mass_tolerance",
      finder_parameters_.theta_linf_times_mass_tolerance
   );
   parameters.queryAdd(
      "theta_l2_times_mass_tolerance",
      finder_parameters_.theta_l2_times_mass_tolerance
   );
   parameters.queryAdd(
      "max_iterations_per_level",
      finder_parameters_.max_iterations_per_level
   );
   parameters.queryAdd("over_relaxation", finder_parameters_.over_relaxation);
   parameters.queryAdd(
      "over_relaxation_interval",
      finder_parameters_.over_relaxation_interval
   );
   parameters.queryAdd("refine_band_cells", refine_band_cells_);
   parameters.queryAdd("excision_fraction", excision_fraction_);
   finder_parameters_.target_num_theta = polar_points_;

   if (find_interval_ <= 0 || radial_points_ < 8 || polar_points_ < 8
       || polar_points_ % 2 != 0
       || finder_parameters_.coarsest_num_theta < 8
       || finder_parameters_.coarsest_num_theta % 2 != 0
       || finder_parameters_.coarsest_num_theta > polar_points_
       || !(minimum_search_radius_ > 0.0_rt)
       || !(maximum_search_radius_ > minimum_search_radius_)
       || !(initial_radius_ >= minimum_search_radius_)
       || !(initial_radius_ <= maximum_search_radius_)
       || refine_band_cells_ < 0.0_rt
       || !(excision_fraction_ > 0.0_rt && excision_fraction_ < 1.0_rt)) {
      amrex::Abort("invalid horizon.* apparent-horizon configuration");
   }
   if (enabled_) {
#if (AMREX_SPACEDIM != 3)
      amrex::Abort("horizon.enabled currently requires the 3D executable");
#else
      if (config.mesh_coordinate != "cartesian") {
         amrex::Abort(
            "horizon.enabled currently requires Cartesian evolution geometry"
         );
      }
      if (!spacetime_stores_state(config)) {
         amrex::Abort(
            "horizon.enabled requires stored fixed or evolved spacetime"
         );
      }
#endif
   }
   if (!enabled_ && (
          aux_gfs_contains(config, AuxGridField::ApparentHorizonMask)
          || aux_gfs_contains(config, AuxGridField::ExcisionMask)
       )) {
      amrex::Abort(
         "apparent_horizon_mask and excision_mask output require "
         "horizon.enabled=1"
      );
   }
   signed_distance_.resize(num_levels);
   apparent_horizon_mask_.resize(num_levels);
   excision_mask_.resize(num_levels);
   level_revision_.resize(num_levels, -1);
}

bool ApparentHorizonTracker::enabled() const noexcept
{
   return enabled_;
}

bool ApparentHorizonTracker::abort_on_failure() const noexcept
{
   return abort_on_failure_;
}

bool ApparentHorizonTracker::failed_search_is_fatal() const noexcept
{
   return abort_on_failure_ || result_.has_value()
      || restored_surface_.has_value();
}

bool ApparentHorizonTracker::refinement_tracking_active() const noexcept
{
   return result_.has_value() && refine_band_cells_ > 0.0_rt;
}

bool ApparentHorizonTracker::search_due(int step) const noexcept
{
   return enabled_ && step >= 0 && step % find_interval_ == 0;
}

int ApparentHorizonTracker::radial_points() const noexcept
{
   return radial_points_;
}

int ApparentHorizonTracker::polar_points() const noexcept
{
   return polar_points_;
}

amrex::Real ApparentHorizonTracker::minimum_search_radius() const noexcept
{
   return minimum_search_radius_;
}

amrex::Real ApparentHorizonTracker::maximum_search_radius() const noexcept
{
   return maximum_search_radius_;
}

amrex::Real ApparentHorizonTracker::initial_radius() const noexcept
{
   return initial_radius_;
}

std::array<amrex::Real, 3> const& ApparentHorizonTracker::center() const noexcept
{
   return center_;
}

ApparentHorizonFinderParameters const&
ApparentHorizonTracker::finder_parameters() const noexcept
{
   return finder_parameters_;
}

std::optional<ApparentHorizonSurface>
ApparentHorizonTracker::previous_surface() const
{
   if (!result_.has_value()) {
      return restored_surface_;
   }
   return result_->surface;
}

void ApparentHorizonTracker::restore_latest_surface(
   int step,
   std::string const& output_dir
)
{
   if (!enabled_ || result_.has_value() || step <= 0) {
      return;
   }
   const int root = amrex::ParallelDescriptor::IOProcessorNumber();
   int status = 0;
   int restored_step = -1;
   std::vector<amrex::Real> radius(
      static_cast<std::size_t>(polar_points_) * 2 * polar_points_
   );
   if (amrex::ParallelDescriptor::IOProcessor()) {
      try {
         const std::filesystem::path shape_dir =
            std::filesystem::path(output_dir) / "apparent_horizon";
         int latest_step = -1;
         std::filesystem::path latest_path;
         if (std::filesystem::exists(shape_dir)) {
            for (auto const& entry :
                 std::filesystem::directory_iterator(shape_dir)) {
               const std::string name = entry.path().filename().string();
               if (!entry.is_regular_file() || name.size() != 14
                   || name.starts_with("ah") == false
                   || name.ends_with(".tsv") == false) {
                  continue;
               }
               const int candidate_step = std::stoi(name.substr(2, 8));
               if (candidate_step <= step && candidate_step > latest_step) {
                  latest_step = candidate_step;
                  latest_path = entry.path();
               }
            }
         }
         if (latest_step >= 0) {
            restored_step = latest_step;
            std::ifstream input(latest_path);
            std::string line;
            std::getline(input, line);
            std::size_t count = 0;
            while (std::getline(input, line) && count < radius.size()) {
               std::istringstream row(line);
               amrex::Real theta = 0.0_rt;
               amrex::Real phi = 0.0_rt;
               amrex::Real x = 0.0_rt;
               amrex::Real y = 0.0_rt;
               amrex::Real z = 0.0_rt;
               row >> theta >> phi >> radius[count] >> x >> y >> z;
               if (!row || !std::isfinite(radius[count])
                   || !(radius[count] > 0.0_rt)) {
                  status = -1;
                  break;
               }
               ++count;
            }
            if (status == 0) {
               status = count == radius.size() ? 1 : -1;
            }
         }
      } catch (std::exception const&) {
         status = -1;
      }
   }
   amrex::ParallelDescriptor::Bcast(&status, 1, root);
   amrex::ParallelDescriptor::Bcast(&restored_step, 1, root);
   if (status < 0) {
      amrex::Abort(
         "failed to restore the persisted apparent-horizon surface"
      );
   }
   if (status == 0) {
      return;
   }
   amrex::ParallelDescriptor::Bcast(
      radius.data(), static_cast<int>(radius.size()), root
   );
   ApparentHorizonSurface restored_surface(
      center_, polar_points_, 2 * polar_points_, std::move(radius)
   );
   if (restored_step < step) {
      restored_surface_ = std::move(restored_surface);
      amrex::Print()
         << "restored apparent-horizon search seed from step "
         << restored_step << " for restart step " << step << '\n';
      return;
   }
   ApparentHorizonResult restored;
   restored.converged = true;
   restored.surface = std::move(restored_surface);
   restored.minimum_radius = *std::min_element(
      restored.surface.radius().begin(), restored.surface.radius().end()
   );
   restored.maximum_radius = *std::max_element(
      restored.surface.radius().begin(), restored.surface.radius().end()
   );
   restored.mean_radius = 0.0_rt;
   for (amrex::Real value : restored.surface.radius()) {
      restored.mean_radius += value;
   }
   restored.mean_radius /= restored.surface.radius().size();
   result_ = std::move(restored);
   ++surface_revision_;
   std::fill(level_revision_.begin(), level_revision_.end(), -1);
   device_radius_.resize(result_->surface.radius().size());
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      result_->surface.radius().begin(),
      result_->surface.radius().end(),
      device_radius_.begin()
   );
   amrex::Print()
      << "restored apparent-horizon surface from step " << restored_step
      << " for restart step " << step << '\n';
}

void ApparentHorizonTracker::accept_result(
   int step,
   amrex::Real time,
   ApparentHorizonResult result,
   std::string const& output_dir
)
{
   if (!result.converged) {
      throw std::invalid_argument(
         "cannot track a non-converged apparent-horizon result"
      );
   }
   if (spacetime_excision_.enabled()) {
      amrex::Real center_distance_squared = 0.0_rt;
      for (int direction = 0; direction < 3; ++direction) {
         const amrex::Real delta = center_[direction] - spacetime_excision_.center[direction];
         center_distance_squared += delta * delta;
      }
      const amrex::Real enclosing_radius = spacetime_excision_.radius + std::sqrt(center_distance_squared);
      if (*std::min_element(result.surface.radius().begin(), result.surface.radius().end()) <= enclosing_radius) {
         amrex::Abort("the apparent horizon no longer encloses the fixed spacetime excision region");
      }
   }
   result_ = std::move(result);
   restored_surface_.reset();
   ++surface_revision_;
   std::fill(level_revision_.begin(), level_revision_.end(), -1);
   device_radius_.resize(result_->surface.radius().size());
   amrex::Gpu::copy(
      amrex::Gpu::hostToDevice,
      result_->surface.radius().begin(),
      result_->surface.radius().end(),
      device_radius_.begin()
   );
   write_result(step, time, *result_, output_dir);
}

ApparentHorizonResult const* ApparentHorizonTracker::result() const noexcept
{
   return result_ ? &*result_ : nullptr;
}

void ApparentHorizonTracker::refresh_level(
   int level,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry
)
{
   if (!result_.has_value()) {
      return;
   }
   const bool layout_matches = signed_distance_[level]
      && signed_distance_[level]->boxArray() == state.boxArray()
      && signed_distance_[level]->DistributionMap().ProcessorMap()
         == state.DistributionMap().ProcessorMap();
   if (level_revision_[level] == surface_revision_ && layout_matches) {
      return;
   }
   signed_distance_[level] = std::make_unique<amrex::MultiFab>(
      state.boxArray(), state.DistributionMap(), 1, mask_ghost_cells_
   );
   apparent_horizon_mask_[level] = std::make_unique<amrex::iMultiFab>(
      state.boxArray(), state.DistributionMap(), 1, mask_ghost_cells_
   );
   excision_mask_[level] = std::make_unique<amrex::iMultiFab>(
      state.boxArray(), state.DistributionMap(), 1, mask_ghost_cells_
   );
   signed_distance_[level]->setVal(0.0_rt);
   apparent_horizon_mask_[level]->setVal(0);
   excision_mask_[level]->setVal(0);
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const auto center = center_;
   const int num_theta = result_->surface.num_theta();
   const int num_phi = result_->surface.num_phi();
   const amrex::Real excision_fraction = excision_fraction_;
   const auto spacetime_excision = spacetime_excision_;
   const amrex::Real* radius = device_radius_.data();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(
           *signed_distance_[level], amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.growntilebox(mask_ghost_cells_)
         & (*signed_distance_[level])[mfi].box();
      const auto distance = signed_distance_[level]->array(mfi);
      const auto horizon_mask = apparent_horizon_mask_[level]->array(mfi);
      const auto excision = excision_mask_[level]->array(mfi);
      fill_horizon_masks(
         box, distance, horizon_mask, excision, lower, spacing, center,
         num_theta, num_phi, excision_fraction, spacetime_excision, radius
      );
   }
   signed_distance_[level]->FillBoundary(geometry.periodicity());
   amrex::Vector<amrex::iMultiFab*> masks{
      apparent_horizon_mask_[level].get(), excision_mask_[level].get()
   };
   amrex::FillBoundary(masks, geometry.periodicity());
   level_revision_[level] = surface_revision_;
}

amrex::iMultiFab const* ApparentHorizonTracker::apparent_horizon_mask(
   int level,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry
)
{
   refresh_level(level, state, geometry);
   return apparent_horizon_mask_[level].get();
}

amrex::iMultiFab const* ApparentHorizonTracker::excision_mask(
   int level,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry
)
{
   refresh_level(level, state, geometry);
   return excision_mask_[level].get();
}

void ApparentHorizonTracker::tag_refinement_band(
   int level,
   amrex::TagBoxArray& tags,
   amrex::MultiFab const& state,
   amrex::Geometry const& geometry
)
{
   if (!result_.has_value() || refine_band_cells_ <= 0.0_rt) {
      return;
   }
   refresh_level(level, state, geometry);
   const auto spacing = geometry.CellSizeArray();
   amrex::Real maximum_spacing = spacing[0];
   maximum_spacing = amrex::max(maximum_spacing, spacing[1]);
   maximum_spacing = amrex::max(maximum_spacing, spacing[2]);
   const amrex::Real band = refine_band_cells_ * maximum_spacing;
   const char tag_value = amrex::TagBox::SET;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto distance = signed_distance_[level]->const_array(mfi);
      const auto tag = tags.array(mfi);
      amrex::ParallelFor(
         box,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (amrex::Math::abs(distance(i, j, k)) <= band) {
               tag(i, j, k) = tag_value;
            }
         }
      );
   }
}

void ApparentHorizonTracker::invalidate_level(int level) noexcept
{
   if (level < 0 || level >= static_cast<int>(level_revision_.size())) {
      return;
   }
   level_revision_[level] = -1;
   signed_distance_[level].reset();
   apparent_horizon_mask_[level].reset();
   excision_mask_[level].reset();
}

void ApparentHorizonTracker::write_result(
   int step,
   amrex::Real time,
   ApparentHorizonResult const& result,
   std::string const& output_dir
) const
{
   if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
   }
   const std::filesystem::path root(output_dir);
   std::filesystem::create_directories(root);
   const std::filesystem::path diagnostics = root / "apparent_horizon.tsv";
   const bool write_header = !std::filesystem::exists(diagnostics)
      || std::filesystem::file_size(diagnostics) == 0;
   std::ofstream summary(diagnostics, std::ios::app);
   summary << std::setprecision(17);
   if (write_header) {
      summary
         << "step\ttime\titerations\tpseudo_time\ttheta_linf_times_mass"
         << "\ttheta_l2_times_mass\tarea\tirreducible_mass"
         << "\tequatorial_circumference\tpolar_circumference"
         << "\tkerr_horizon_mass\tkerr_dimensionless_spin"
         << "\tcoordinate_angular_momentum1"
         << "\tcoordinate_angular_momentum2"
         << "\tcoordinate_angular_momentum3"
         << "\tcoordinate_angular_momentum_magnitude"
         << "\tcoordinate_christodoulou_mass"
         << "\tcoordinate_christodoulou_dimensionless_spin1"
         << "\tcoordinate_christodoulou_dimensionless_spin2"
         << "\tcoordinate_christodoulou_dimensionless_spin3"
         << "\tcoordinate_christodoulou_dimensionless_spin_magnitude"
         << "\tcentroid1\tcentroid2\tcentroid3"
         << "\tminimum_radius\tmaximum_radius\tmean_radius\n";
   }
   summary << step << '\t' << time << '\t' << result.iterations << '\t'
           << result.pseudo_time << '\t' << result.theta_linf_times_mass
           << '\t' << result.theta_l2_times_mass << '\t' << result.area
           << '\t' << result.irreducible_mass << '\t'
           << result.equatorial_circumference << '\t'
           << result.polar_circumference << '\t'
           << result.kerr_horizon_mass << '\t'
           << result.kerr_dimensionless_spin << '\t'
           << result.angular_momentum[0] << '\t'
           << result.angular_momentum[1] << '\t'
           << result.angular_momentum[2] << '\t'
           << result.angular_momentum_magnitude << '\t'
           << result.christodoulou_mass << '\t'
           << result.christodoulou_dimensionless_spin[0] << '\t'
           << result.christodoulou_dimensionless_spin[1] << '\t'
           << result.christodoulou_dimensionless_spin[2] << '\t'
           << result.christodoulou_dimensionless_spin_magnitude << '\t'
           << result.centroid[0]
           << '\t' << result.centroid[1] << '\t' << result.centroid[2]
           << '\t' << result.minimum_radius << '\t'
           << result.maximum_radius << '\t' << result.mean_radius << '\n';

   const std::filesystem::path shape_dir = root / "apparent_horizon";
   std::filesystem::create_directories(shape_dir);
   std::ostringstream filename;
   filename << "ah" << std::setw(8) << std::setfill('0') << step << ".tsv";
   std::ofstream shape(shape_dir / filename.str());
   shape << std::setprecision(17)
         << "theta\tphi\tradius\tx\ty\tz\n";
   const auto center = result.surface.center();
   const int num_theta = result.surface.num_theta();
   const int num_phi = result.surface.num_phi();
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0_rt * pi / num_phi;
   for (int theta_index = 0; theta_index < num_theta; ++theta_index) {
      const amrex::Real theta = (theta_index + 0.5_rt) * dtheta;
      for (int phi_index = 0; phi_index < num_phi; ++phi_index) {
         const amrex::Real phi = (phi_index + 0.5_rt) * dphi;
         const amrex::Real radius = result.surface.radius()[
            theta_index * num_phi + phi_index
         ];
         shape << theta << '\t' << phi << '\t' << radius << '\t'
               << center[0] + radius * std::sin(theta) * std::cos(phi)
               << '\t'
               << center[1] + radius * std::sin(theta) * std::sin(phi)
               << '\t' << center[2] + radius * std::cos(theta) << '\n';
      }
   }
}

} // namespace pyfluid_amrex
