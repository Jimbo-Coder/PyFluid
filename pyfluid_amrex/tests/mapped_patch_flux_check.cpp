#include "numerics/mapped_patch_interfaces.hpp"

#include <AMReX_Print.H>
#include <cmath>

namespace pyfluid_amrex {

int check_mapped_patch_flux_corrections()
{
   using namespace amrex::literals;
   constexpr int Components = NVAR_HYDRO + 1;
   amrex::Vector<MappedPatchFluxCorrection2D<Components>> walls;
   const HydroState wall_flux{0.0_rt, 2.0_rt, -3.0_rt, 4.0_rt, 0.0_rt};
   // External reflecting faces supply no mass, energy or passive transport.
   // Exercise both creation and accumulation of the same face correction.
   add_mapped_patch_flux_correction<Components>(walls, 0, 0, 0, 0, wall_flux, {});
   add_mapped_patch_flux_correction<Components>(walls, 0, 0, 0, 0, wall_flux, {});
   bool passed = walls.size() == 1;
   if (passed) {
      const auto& value = walls[0].value;
      passed = value[RHO] == 0.0_rt && value[ENERGY] == 0.0_rt
         && value[NVAR_HYDRO] == 0.0_rt && value[MOM1] == 4.0_rt
         && value[MOM2] == -6.0_rt && value[MOM3] == 8.0_rt;
   }

   amrex::Vector<MappedPatchFluxCorrection2D<Components>> seams;
   const HydroState outward{2.0_rt, 3.0_rt, -4.0_rt, 5.0_rt, 6.0_rt};
   const HydroState inward = scale_flux(outward, -1.0_rt);
   // Two overlap contributions cancel mass but transport different dye.
   for (int side = 0; side < 2; ++side) {
      const amrex::Real sign = side == 0 ? 1.0_rt : -1.0_rt;
      add_mapped_patch_flux_correction<Components>(
         seams, side, 0, side, 0, outward, {1.0_rt}, sign);
      add_mapped_patch_flux_correction<Components>(
         seams, side, 0, side, 0, inward, {-2.0_rt}, sign);
   }
   passed = passed && seams.size() == 2;
   if (seams.size() == 2) {
      passed = passed && seams[0].value[RHO] == 0.0_rt
         && seams[1].value[RHO] == 0.0_rt
         && seams[0].value[NVAR_HYDRO] == -1.0_rt
         && seams[1].value[NVAR_HYDRO] == 1.0_rt;
      for (int component = 0; component < Components; ++component) {
         passed = passed && std::isfinite(seams[0].value[component])
            && std::isfinite(seams[1].value[component])
            && seams[0].value[component] + seams[1].value[component] == 0.0_rt;
      }
   }

   amrex::Vector<MappedPatchFluxCorrection2D<NVAR_HYDRO>> untraced;
   add_mapped_patch_flux_correction<NVAR_HYDRO>(untraced, 0, 0, 0, 0, wall_flux, {});
   passed = passed && untraced.size() == 1 && untraced[0].value[MOM1] == wall_flux.mom1;
   amrex::Print() << (passed ? "PASS" : "FAIL")
      << " mapped wall and overlap passive flux corrections\n";
   return passed ? 0 : 1;
}

} // namespace pyfluid_amrex
