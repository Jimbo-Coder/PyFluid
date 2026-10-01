#pragma once

#include <AMReX_REAL.H>

#include <array>
#include <string>
#include <vector>

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/multilevel_field_io.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Coordinate metadata carried by a portable physical-ID artifact.  Unlike
// MeshAxes this representation describes the source file, so its dimension is
// a runtime value and need not match the executable materializing the data.
struct PhysicalIdChart {
   int dimension = 0;
   int coordinate_kind = 0;
   std::array<int, 3> axes{};
   std::array<int, 3> periodic{};
   std::array<int, 3> reflection_lo{};
   std::array<int, 3> reflection_hi{};
};

PhysicalIdChart make_physical_id_chart(
   int dimension,
   std::string const& coordinate,
   std::string const& axes,
   std::array<int, 3> const& periodic = {},
   std::array<int, 3> const& reflection_lo = {},
   std::array<int, 3> const& reflection_hi = {}
);

PhysicalIdChart current_physical_id_chart(
   std::string const& coordinate,
   std::string const& axes
);

struct PortablePhysicalIdSample {
   std::vector<amrex::Real> values;
   std::array<amrex::Real, 3> source_cartesian{};
   bool exterior = false;
};

// One proper rigid transform maps a portable source object's Cartesian frame
// into the target Cartesian frame.  Coordinate charts remain responsible for
// curvilinear components; this transform owns only object translation and
// orientation, so it does not create coordinate- or dimension-specific
// materializers.
struct PhysicalIdRigidTransform {
   std::array<amrex::Real, 3> source_origin{};
   std::array<amrex::Real, 3> target_origin{};
   std::array<std::array<amrex::Real, 3>, 3> source_to_target{{
      {{1.0_rt, 0.0_rt, 0.0_rt}},
      {{0.0_rt, 1.0_rt, 0.0_rt}},
      {{0.0_rt, 0.0_rt, 1.0_rt}},
   }};
};

PhysicalIdRigidTransform make_physical_id_rigid_transform(
   std::array<amrex::Real, 3> const& source_origin,
   std::array<amrex::Real, 3> const& target_origin,
   std::array<amrex::Real, 3> const& source_axis,
   std::array<amrex::Real, 3> const& target_axis
);

bool physical_id_rigid_transform_is_identity(
   PhysicalIdRigidTransform const& transform
);

using RuntimePhysicalIdCoordinateFaces =
   std::vector<std::array<std::vector<amrex::Real>, 3>>;

class PortablePhysicalIdInterpolator {
public:
   PortablePhysicalIdInterpolator(
      RuntimeDimensionHdf5Field field,
      PhysicalIdChart source_chart,
      GridFieldSetInfo field_set,
      bool node_centered,
      RuntimePhysicalIdCoordinateFaces coordinate_faces = {},
      PhysicalIdRigidTransform rigid_transform = {}
   );

   PortablePhysicalIdSample sample(
      std::array<amrex::Real, 3> const& target_cartesian,
      PhysicalIdChart const& target_chart
   ) const;

private:
   RuntimeDimensionHdf5Field field_;
   PhysicalIdChart source_chart_;
   GridFieldSetInfo field_set_;
   bool node_centered_ = false;
   PhysicalIdRigidTransform rigid_transform_;
};

} // namespace pyfluid_amrex
