#include "pyfluid_amrex/hierarchy_interpolation.hpp"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "pyfluid_amrex/coordinate_locations.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

struct DeviceInterpolationPoint {
   int output = 0;
   int i = 0;
   int j = 0;
   int k = 0;
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> fraction{};
};

amrex::Real wrap_periodic_coordinate(
   amrex::Real coordinate,
   amrex::Real lower,
   amrex::Real upper
)
{
   const amrex::Real period = upper - lower;
   if (!(period > 0.0_rt)) {
      throw std::invalid_argument("periodic interpolation coordinate has non-positive extent");
   }
   coordinate = lower + std::fmod(coordinate - lower, period);
   if (coordinate < lower) {
      coordinate += period;
   }
   return coordinate;
}

} // namespace

HierarchyInterpolationPlan::HierarchyInterpolationPlan(
   std::vector<amrex::GpuArray<amrex::Real, 3>> physical_points,
   int coordinate_kind,
   MeshAxes axes,
   std::vector<amrex::Geometry> const& geometry,
   std::vector<amrex::BoxArray> const& grids,
   std::vector<amrex::DistributionMapping> const& distribution,
   amrex::IndexType centering,
   std::array<int, 3> const& reflection_lo,
   std::array<int, 3> const& reflection_hi
)
{
   if (geometry.empty() || geometry.size() != grids.size()
       || geometry.size() != distribution.size()) {
      throw std::invalid_argument("interpolation hierarchy metadata must have one entry per AMR level");
   }
   points_.reserve(physical_points.size());
   for (auto const& physical_point : physical_points) {
      const auto mesh_triplet = cartesian_to_mesh_coordinate_triplet(
         coordinate_kind, physical_point
      );
      auto runtime_coordinates = mesh_triplet_to_runtime_coordinates(
         coordinate_kind, axes, mesh_triplet
      );
      Point point;
      for (int level = static_cast<int>(geometry.size()) - 1; level >= 0; --level) {
         const amrex::Geometry& level_geometry = geometry[level];
         const auto lower = level_geometry.ProbLoArray();
         const auto upper = level_geometry.ProbHiArray();
         const auto spacing = level_geometry.CellSizeArray();
         const amrex::Box centered_domain = amrex::convert(
            level_geometry.Domain(), centering
         );
         amrex::IntVect anchor;
         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> fraction{};
         bool inside = true;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            amrex::Real coordinate = runtime_coordinates[direction];
            if (level_geometry.isPeriodic(direction)) {
               coordinate = wrap_periodic_coordinate(
                  coordinate, lower[direction], upper[direction]
               );
            }
            const amrex::Real centering_offset = centering.nodeCentered(direction)
               ? 0.0_rt
               : 0.5_rt;
            const amrex::Real continuous_index =
               centered_domain.smallEnd(direction)
               + (coordinate - lower[direction]) / spacing[direction]
               - centering_offset;
            int lower_index = static_cast<int>(std::floor(continuous_index));
            amrex::Real local_fraction = continuous_index - lower_index;
            if (level_geometry.isPeriodic(direction)) {
               const int domain_length = centered_domain.length(direction);
               if (lower_index < centered_domain.smallEnd(direction)) {
                  lower_index += domain_length;
               } else if (lower_index > centered_domain.bigEnd(direction)) {
                  lower_index -= domain_length;
               }
               anchor[direction] = lower_index;
               fraction[direction] = local_fraction;
               continue;
            }
            const bool crosses_lower_reflection =
               lower_index == centered_domain.smallEnd(direction) - 1
               && reflection_lo[direction] != 0;
            const bool crosses_upper_reflection =
               lower_index == centered_domain.bigEnd(direction)
               && reflection_hi[direction] != 0;
            if ((lower_index < centered_domain.smallEnd(direction)
                 || lower_index + 1 > centered_domain.bigEnd(direction))
                && !crosses_lower_reflection
                && !crosses_upper_reflection) {
               // Cartesian-to-curvilinear round trips can amplify floating-
               // point error near an axis.  Clamp only when the point differs
               // from the first/last interpolation location by coordinate
               // roundoff; express that tolerance in index units.
               const amrex::Real coordinate_scale = std::max({
                  amrex::Real(1.0_rt),
                  std::abs(coordinate),
                  std::abs(lower[direction]),
                  std::abs(upper[direction]),
               });
               const amrex::Real index_tolerance =
                  1024.0_rt * std::numeric_limits<amrex::Real>::epsilon()
                  * coordinate_scale / spacing[direction];
               if (lower_index == centered_domain.smallEnd(direction) - 1
                   && std::abs(1.0_rt - local_fraction) < index_tolerance) {
                  lower_index = centered_domain.smallEnd(direction);
                  local_fraction = 0.0_rt;
               } else if (lower_index == centered_domain.bigEnd(direction)
                          && std::abs(local_fraction) < index_tolerance) {
                  --lower_index;
                  local_fraction = 1.0_rt;
               } else {
                  inside = false;
                  break;
               }
            }
            anchor[direction] = lower_index;
            fraction[direction] = local_fraction;
         }
         if (!inside) {
            continue;
         }
         const amrex::BoxArray centered_grids = amrex::convert(grids[level], centering);
         amrex::Box interpolation_cell(anchor, anchor + amrex::IntVect::TheUnitVector());
         const amrex::Box interior_stencil =
            interpolation_cell & centered_domain;
         const amrex::BoxArray uncovered = amrex::complementIn(
            interior_stencil, centered_grids
         );
         bool exterior_is_supported = true;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            if (!level_geometry.isPeriodic(direction)
                && ((interpolation_cell.smallEnd(direction)
                        < centered_domain.smallEnd(direction)
                     && reflection_lo[direction] == 0)
                    || (interpolation_cell.bigEnd(direction)
                          > centered_domain.bigEnd(direction)
                        && reflection_hi[direction] == 0))) {
               exterior_is_supported = false;
            }
         }
         const bool stencil_covered = uncovered.empty()
            && exterior_is_supported;
         if (!stencil_covered) {
            continue;
         }
         amrex::IntVect owner_index = anchor;
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            owner_index[direction] = std::max(
               centered_domain.smallEnd(direction),
               std::min(
                  centered_domain.bigEnd(direction), owner_index[direction]
               )
            );
         }
         for (int box = 0; box < centered_grids.size(); ++box) {
            if (centered_grids[box].contains(owner_index)) {
               point.level = level;
               point.box = box;
               point.owner = distribution[level][box];
               point.anchor = anchor;
               point.fraction = fraction;
               break;
            }
         }
         if (point.level >= 0) {
            break;
         }
      }
      if (point.level < 0) {
         throw std::out_of_range(
            "hierarchy interpolation point is outside the AMR hierarchy: x=("
            + std::to_string(physical_point[0]) + ","
            + std::to_string(physical_point[1]) + ","
            + std::to_string(physical_point[2]) + ") q=("
            + std::to_string(runtime_coordinates[0])
#if (AMREX_SPACEDIM >= 2)
            + "," + std::to_string(runtime_coordinates[1])
#endif
#if (AMREX_SPACEDIM == 3)
            + "," + std::to_string(runtime_coordinates[2])
#endif
            + ")"
         );
      }
      points_.push_back(point);
   }
}

std::vector<amrex::Real> HierarchyInterpolationPlan::interpolate(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& hierarchy,
   int first_component,
   int num_components
) const
{
   if (num_components <= 0 || first_component < 0) {
      throw std::invalid_argument("interpolation component range must be non-empty");
   }
   for (Point const& point : points_) {
      if (point.level >= static_cast<int>(hierarchy.size())
          || hierarchy[point.level] == nullptr
          || first_component + num_components > hierarchy[point.level]->nComp()) {
         throw std::out_of_range(
            "interpolation component range is absent from the selected hierarchy level"
         );
      }
      const amrex::Box interpolation_cell(
         point.anchor, point.anchor + amrex::IntVect::TheUnitVector()
      );
      amrex::Box storage_box = hierarchy[point.level]->boxArray()[point.box];
      storage_box.grow(hierarchy[point.level]->nGrowVect());
      if (!storage_box.contains(interpolation_cell)) {
         throw std::out_of_range(
            "interpolation stencil extends outside the selected FAB storage; "
            "provide and fill one ghost layer"
         );
      }
   }
   std::vector<amrex::Real> host_values(points_.size() * num_components, 0.0_rt);
   amrex::Gpu::DeviceVector<amrex::Real> device_values(host_values.size(), 0.0_rt);
   amrex::Real* values = device_values.data();
   const int rank = amrex::ParallelDescriptor::MyProc();
   std::map<std::pair<int, int>, std::vector<DeviceInterpolationPoint>> groups;
   for (std::size_t point_index = 0; point_index < points_.size(); ++point_index) {
      Point const& point = points_[point_index];
      if (point.owner != rank) {
         continue;
      }
      groups[{point.level, point.box}].push_back({
         static_cast<int>(point_index),
         point.anchor[0],
#if (AMREX_SPACEDIM >= 2)
         point.anchor[1],
#else
         0,
#endif
#if (AMREX_SPACEDIM == 3)
         point.anchor[2],
#else
         0,
#endif
         point.fraction,
      });
   }
   for (auto const& [location, host_points] : groups) {
      const int level = location.first;
      const int box = location.second;
      amrex::Gpu::DeviceVector<DeviceInterpolationPoint> device_points(host_points.size());
      amrex::Gpu::copy(
         amrex::Gpu::hostToDevice,
         host_points.begin(),
         host_points.end(),
         device_points.begin()
      );
      DeviceInterpolationPoint const* interpolation_points = device_points.data();
      const auto field = hierarchy[level]->const_array(box);
      const amrex::Long work = static_cast<amrex::Long>(host_points.size()) * num_components;
      amrex::ParallelFor(work, [=] AMREX_GPU_DEVICE(amrex::Long work_index) noexcept {
         const int component = static_cast<int>(work_index % num_components);
         const int local_point = static_cast<int>(work_index / num_components);
         const DeviceInterpolationPoint point = interpolation_points[local_point];
         const amrex::Real wx[2] = {1.0_rt - point.fraction[0], point.fraction[0]};
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real wy[2] = {1.0_rt - point.fraction[1], point.fraction[1]};
#else
         const amrex::Real wy[2] = {1.0_rt, 0.0_rt};
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real wz[2] = {1.0_rt - point.fraction[2], point.fraction[2]};
#else
         const amrex::Real wz[2] = {1.0_rt, 0.0_rt};
#endif
         amrex::Real value = 0.0_rt;
         for (int dk = 0; dk < (AMREX_SPACEDIM == 3 ? 2 : 1); ++dk) {
            for (int dj = 0; dj < (AMREX_SPACEDIM >= 2 ? 2 : 1); ++dj) {
               for (int di = 0; di < 2; ++di) {
                  value += wx[di] * wy[dj] * wz[dk]
                     * field(
                        point.i + di,
                        point.j + dj,
                        point.k + dk,
                        first_component + component
                     );
               }
            }
         }
         values[point.output * num_components + component] = value;
      });
   }
   amrex::Gpu::copy(
      amrex::Gpu::deviceToHost,
      device_values.begin(),
      device_values.end(),
      host_values.begin()
   );
   amrex::ParallelDescriptor::ReduceRealSum(
      host_values.data(), static_cast<int>(host_values.size())
   );
   return host_values;
}

} // namespace pyfluid_amrex
