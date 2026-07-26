#include <iostream>
#include <string>
#include <string_view>

#include "pyfluid_amrex/reconstruction/parabolic.hpp"
#include "pyfluid_amrex/reconstruction/ppm4.hpp"
#include "pyfluid_amrex/reconstruction/ppmx.hpp"

namespace {

using namespace pyfluid_amrex;

bool close_enough(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = amrex::max(
      1.0,
      amrex::max(amrex::Math::abs(actual), amrex::Math::abs(expected))
   );
   return amrex::Math::abs(actual - expected) <= 2.0e-14 * scale;
}

bool require_close(
   std::string_view label,
   amrex::Real actual,
   amrex::Real expected
)
{
   if (close_enough(actual, expected)) {
      return true;
   }
   std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
   return false;
}

template <int Method>
bool validate_constant_and_linear(std::string_view label)
{
   bool passed = true;
   passed = require_close(
      std::string(label) + " constant",
      reconstruct_left<Method>(3.25, 3.25, 3.25, 3.25, 3.25),
      3.25
   ) && passed;
   passed = require_close(
      std::string(label) + " linear",
      reconstruct_left<Method>(-2.0, -1.0, 0.0, 1.0, 2.0),
      0.5
   ) && passed;
   return passed;
}

bool validate_parabolic_family()
{
   static_assert(is_five_point_reconstruction(ReconParabolic));
   static_assert(is_five_point_reconstruction(ReconPPM4));
   static_assert(is_five_point_reconstruction(ReconPPMX));
   static_assert(is_characteristic_reconstruction(ReconParabolicChar));
   static_assert(is_characteristic_reconstruction(ReconPPM4Char));
   static_assert(is_characteristic_reconstruction(ReconPPMXChar));

   bool passed = true;
   passed = validate_constant_and_linear<ReconParabolic>("parabolic") && passed;
   passed = validate_constant_and_linear<ReconPPM4>("ppm4") && passed;
   passed = validate_constant_and_linear<ReconPPMX>("ppmx") && passed;

   constexpr amrex::Real cell_average_offset = 1.0 / 12.0;
   const amrex::Real quadratic_parabolic = reconstruct_left<ReconParabolic>(
      4.0 + cell_average_offset,
      1.0 + cell_average_offset,
      cell_average_offset,
      1.0 + cell_average_offset,
      4.0 + cell_average_offset
   );
   const amrex::Real quadratic_ppm4 = reconstruct_left<ReconPPM4>(
      4.0 + cell_average_offset,
      1.0 + cell_average_offset,
      cell_average_offset,
      1.0 + cell_average_offset,
      4.0 + cell_average_offset
   );
   const amrex::Real quadratic_ppmx = reconstruct_left<ReconPPMX>(
      4.0 + cell_average_offset,
      1.0 + cell_average_offset,
      cell_average_offset,
      1.0 + cell_average_offset,
      4.0 + cell_average_offset
   );
   passed = require_close(
      "parabolic quadratic interface",
      quadratic_parabolic,
      0.25
   ) && passed;
   passed = require_close(
      "ppm4 clips quadratic extremum",
      quadratic_ppm4,
      cell_average_offset
   ) && passed;
   passed = require_close(
      "ppmx preserves quadratic extremum",
      quadratic_ppmx,
      0.25
   ) && passed;

   const amrex::Real discontinuity_parabolic =
      reconstruct_left<ReconParabolic>(0.0, 1.0, 1.0, 1.0, 0.0);
   const amrex::Real discontinuity_ppm4 =
      reconstruct_left<ReconPPM4>(0.0, 1.0, 1.0, 1.0, 0.0);
   const amrex::Real discontinuity_ppmx =
      reconstruct_left<ReconPPMX>(0.0, 1.0, 1.0, 1.0, 0.0);
   passed = require_close(
      "unlimited parabolic discontinuity overshoot",
      discontinuity_parabolic,
      13.0 / 12.0
   ) && passed;
   passed = require_close(
      "ppm4 bounds discontinuity",
      discontinuity_ppm4,
      1.0
   ) && passed;
   passed = require_close(
      "ppmx bounds discontinuity",
      discontinuity_ppmx,
      1.0
   ) && passed;

   passed = require_close(
      "parabolic characteristic formula",
      reconstruct_left<ReconParabolicChar>(-2.0, -1.0, 0.0, 1.0, 2.0),
      0.5
   ) && passed;
   passed = require_close(
      "ppm4 characteristic formula",
      reconstruct_left<ReconPPM4Char>(-2.0, -1.0, 0.0, 1.0, 2.0),
      0.5
   ) && passed;
   passed = require_close(
      "ppmx characteristic formula",
      reconstruct_left<ReconPPMXChar>(-2.0, -1.0, 0.0, 1.0, 2.0),
      0.5
   ) && passed;
   return passed;
}

} // namespace

int main()
{
   if (!validate_parabolic_family()) {
      return 1;
   }
   std::cout << "reconstruction formulas passed\n";
   return 0;
}
