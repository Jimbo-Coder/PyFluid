#pragma once

#include <AMReX_REAL.H>

#include <array>
#include <string>
#include <vector>

#include "pyfluid_amrex/fields.hpp"
#include "pyfluid_amrex/multilevel_field_io.hpp"

namespace pyfluid_amrex {

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

class PortablePhysicalIdInterpolator {
public:
   PortablePhysicalIdInterpolator(
      RuntimeDimensionHdf5Field field,
      PhysicalIdChart source_chart,
      GridFieldSetInfo field_set,
      bool node_centered
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
};

} // namespace pyfluid_amrex
