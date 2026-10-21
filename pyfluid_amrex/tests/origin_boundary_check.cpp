#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/constants.hpp"
#include "amr/state_interpolation.hpp"
#include "pyfluid_amrex/fields.hpp"

#include <cmath>
#include <limits>

namespace {

using namespace amrex::literals;
using namespace pyfluid_amrex;

constexpr int NumGhost = 3;

AMREX_GPU_HOST_DEVICE
amrex::Real analytic_component(
   int i, int j, int k, int component,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& spacing,
   bool half_theta, bool south_theta, bool axisymmetric, bool magnetic
) noexcept
{
   const amrex::Real radius = (static_cast<amrex::Real>(i) + 0.5_rt) * spacing[0];
   const amrex::Real theta = (south_theta ? constants::pi / 2.0_rt : 0.0_rt)
      + (static_cast<amrex::Real>(j) + 0.5_rt) * spacing[1];
#if AMREX_SPACEDIM == 3
   const amrex::Real phi = (static_cast<amrex::Real>(k) + 0.5_rt) * spacing[2];
#else
   static_cast<void>(k);
   const amrex::Real phi = 0.0_rt;
#endif
   const amrex::Real sint = std::sin(theta);
   const amrex::Real cost = std::cos(theta);
   const amrex::Real sinp = std::sin(phi);
   const amrex::Real cosp = std::cos(phi);
   const amrex::Real x = radius * sint * cosp;
   const amrex::Real y = radius * sint * sinp;
   const amrex::Real z = radius * cost;
   if (component == RHO) {
      // The full-theta scalar is deliberately not equatorially symmetric.
      // The half-theta control must obey its declared reflection symmetry.
      return 1.0_rt + 0.1_rt * (half_theta ? z * z : z);
   }
   if (component == ENERGY) return 5.0_rt;
   if (component == (magnetic ? NVAR_MHD : NVAR_HYDRO)) {
      return 0.2_rt * (1.0_rt + 0.1_rt * (half_theta ? z * z : z));
   }

   amrex::Real vector1;
   amrex::Real vector2;
   amrex::Real vector3;
   int local_component;
   if (component >= MOM1 && component <= MOM3) {
      // A singleton phi grid represents axisymmetric data. The other 3D
      // controls deliberately exercise nonaxisymmetric Cartesian momentum.
      vector1 = axisymmetric ? (half_theta ? x : 0.0_rt) : 0.3_rt;
      vector2 = axisymmetric ? (half_theta ? y : 0.0_rt) : -0.2_rt;
      vector3 = half_theta ? 0.0_rt : 0.4_rt;
      local_component = component - MOM1;
   } else {
      // This smooth axial field obeys equatorial reflection. Its divergence
      // is identically zero; it also exercises the toroidal magnetic sign.
      vector1 = -y * z;
      vector2 = x * z;
      vector3 = 0.4_rt;
      local_component = component - B1;
   }
   if (local_component == 0) {
      return sint * cosp * vector1 + sint * sinp * vector2 + cost * vector3;
   }
   if (local_component == 1) {
      return cost * cosp * vector1 + cost * sinp * vector2 - sint * vector3;
   }
   return -sinp * vector1 + cosp * vector2;
}

bool run_case(bool half_theta, bool south_theta, bool magnetic, int theta_cells, int phi_cells)
{
   const int num_components = (magnetic ? NVAR_MHD : NVAR_HYDRO) + 1;
   const bool axisymmetric = AMREX_SPACEDIM == 2 || phi_cells == 1;
   const amrex::Box domain(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(7, theta_cells - 1, phi_cells - 1))
   );
   const amrex::RealBox physical(
      AMREX_D_DECL(0.0_rt, south_theta ? constants::pi / 2.0_rt : 0.0_rt, 0.0_rt),
      AMREX_D_DECL(1.0_rt, half_theta && !south_theta ? constants::pi / 2.0_rt : constants::pi,
                   constants::two_pi)
   );
   const amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 1)};
   const amrex::Geometry geometry(domain, physical, amrex::CoordSys::cartesian, periodic);
   amrex::BoxArray boxes(domain);
   boxes.maxSize(4);
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, num_components, NumGhost);
   state.setVal(
      std::numeric_limits<amrex::Real>::quiet_NaN(), 0, num_components, state.nGrowVect()
   );
   const auto spacing = geometry.CellSizeArray();
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto values = state.array(mfi);
      amrex::ParallelFor(mfi.validbox(), num_components,
         [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
            values(i, j, k, component) = analytic_component(
               i, j, k, component, spacing, half_theta, south_theta, axisymmetric, magnetic
            );
         });
   }
   state.FillBoundary(geometry.periodicity());
   BoundaryData boundary;
   boundary.mesh_coordinate = MeshSpherical;
#if AMREX_SPACEDIM == 3
   boundary.mesh_axes = mesh_axes_layout(MeshSpherical, "r theta phi");
#else
   boundary.mesh_axes = mesh_axes_layout(MeshSpherical, "r theta");
#endif
   boundary.mhd = magnetic ? 1 : 0;
   boundary.passive_scalars = 1;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      boundary.lo_kind[direction] = static_cast<int>(BoundaryKind::Outflow);
      boundary.hi_kind[direction] = static_cast<int>(BoundaryKind::Outflow);
   }
   boundary.lo_kind[0] = static_cast<int>(BoundaryKind::Origin);
   boundary.origin_lo[0] = 1;
   boundary.lo_kind[1] = static_cast<int>(BoundaryKind::Symmetry);
   boundary.hi_kind[1] = static_cast<int>(BoundaryKind::Symmetry);
#if AMREX_SPACEDIM == 3
   boundary.lo_kind[2] = static_cast<int>(BoundaryKind::Periodic);
   boundary.hi_kind[2] = static_cast<int>(BoundaryKind::Periodic);
#endif
   if (half_theta && !magnetic) {
      // In hydro, reflecting and symmetry prescribe the same equatorial signs.
      if (south_theta) boundary.lo_kind[1] = static_cast<int>(BoundaryKind::Reflecting);
      else boundary.hi_kind[1] = static_cast<int>(BoundaryKind::Reflecting);
   }
   for (int component = 0; component < num_components; ++component) {
      boundary.origin_state_parity[component] = state_grid_field_origin_parity(
         MeshSpherical, boundary.mesh_axes,
         magnetic ? PhysicalSystem::MHD : PhysicalSystem::HD, 1, component
      );
   }
   apply_configured_state_boundary(state, geometry, boundary);

   amrex::Box radial_ghosts = amrex::grow(domain, NumGhost);
   radial_ghosts.setSmall(0, -NumGhost);
   radial_ghosts.setBig(0, -1);
   amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                    amrex::ReduceOpSum, amrex::ReduceOpSum> operations;
   amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Long,
                     amrex::Long> data(operations);
   using Result = decltype(data)::Type;
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto values = state.const_array(mfi);
      const amrex::Box region = state[mfi].box() & radial_ghosts;
      if (!region.ok()) continue;
      operations.eval(region, data,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Result {
            amrex::Real scalar_error = 0.0_rt;
            amrex::Real polar_error = 0.0_rt;
            amrex::Real axial_error = 0.0_rt;
            amrex::Long nonfinite = 0;
            for (int component = 0; component < num_components; ++component) {
               const amrex::Real actual = values(i, j, k, component);
               if (!std::isfinite(actual)) ++nonfinite;
               const amrex::Real error = std::isfinite(actual)
                  ? std::abs(actual - analytic_component(
                       i, j, k, component, spacing, half_theta, south_theta, axisymmetric, magnetic))
                  : std::numeric_limits<amrex::Real>::max();
               if (component >= MOM1 && component <= MOM3) {
                  polar_error = amrex::max(polar_error, error);
               } else if (magnetic && component >= B1 && component <= B3) {
                  axial_error = amrex::max(axial_error, error);
               } else {
                  scalar_error = amrex::max(scalar_error, error);
               }
            }
            return {scalar_error, polar_error, axial_error, nonfinite, 1};
         });
   }
   const Result local = data.value();
   amrex::Real scalar_error = amrex::get<0>(local);
   amrex::Real polar_error = amrex::get<1>(local);
   amrex::Real axial_error = amrex::get<2>(local);
   amrex::Long nonfinite = amrex::get<3>(local);
   amrex::Long ghosts = amrex::get<4>(local);
   amrex::ParallelDescriptor::ReduceRealMax(scalar_error);
   amrex::ParallelDescriptor::ReduceRealMax(polar_error);
   amrex::ParallelDescriptor::ReduceRealMax(axial_error);
   amrex::ParallelDescriptor::ReduceLongSum(nonfinite);
   amrex::ParallelDescriptor::ReduceLongSum(ghosts);
   const amrex::Real tolerance = 128.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   const bool passed = ghosts > 0 && nonfinite == 0
      && scalar_error < tolerance && polar_error < tolerance && axial_error < tolerance;
   amrex::Print() << AMREX_SPACEDIM << '\t' << (south_theta ? "south_half" : (half_theta ? "north_half" : "full_theta"))
      << ":nt=" << theta_cells << ":np=" << phi_cells
      << '\t' << (magnetic ? "mhd" : "hydro") << '\t' << boxes.size()
      << '\t' << ghosts << '\t' << nonfinite << '\t' << scalar_error
      << '\t' << polar_error << '\t' << axial_error << '\t'
      << (passed ? "PASS" : "FAIL") << '\n';
   return passed;
}

// A deliberately incomplete fine band forces both live fine donors and
// temporally interpolated coarse donors. This isolates the origin contract;
// it is not a physical-error-controlled AMR evolution test.
bool check_partial_hierarchy(bool half_theta, amrex::Interpolater* base_interpolater)
{
   constexpr int num_components = NVAR_HYDRO + 1;
   const amrex::Box fine_domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                                amrex::IntVect(AMREX_D_DECL(7, 15, 15)));
   const amrex::Box coarse_domain = amrex::coarsen(fine_domain, 2);
   const amrex::RealBox physical(AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt),
      AMREX_D_DECL(1.0_rt, half_theta ? constants::pi / 2.0_rt : constants::pi, constants::two_pi));
   const amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 1)};
   amrex::Vector<amrex::Geometry> geometry{
      amrex::Geometry(coarse_domain, physical, amrex::CoordSys::cartesian, periodic),
      amrex::Geometry(fine_domain, physical, amrex::CoordSys::cartesian, periodic)};
   const amrex::BoxArray coarse_boxes(coarse_domain);
   const amrex::DistributionMapping coarse_distribution(coarse_boxes);
   amrex::MultiFab coarse_old(coarse_boxes, coarse_distribution, num_components, NumGhost);
   amrex::MultiFab coarse_new(coarse_boxes, coarse_distribution, num_components, NumGhost);
   for (amrex::MultiFab* state : {&coarse_old, &coarse_new}) {
      const amrex::Real density = state == &coarse_old ? 4.0_rt : 6.0_rt;
      state->setVal(0.0_rt);
      state->setVal(density, RHO, 1, NumGhost);
      state->setVal(20.0_rt, ENERGY, 1, NumGhost);
      state->setVal(0.2_rt * density, NVAR_HYDRO, 1, NumGhost);
   }
   amrex::Box fine_region = fine_domain;
   fine_region.setBig(0, 1); // The third radial ghost must use a coarse donor.
   fine_region.setBig(1, 11);
#if AMREX_SPACEDIM == 3
   fine_region.setBig(2, 11);
#endif
   amrex::BoxArray fine_boxes(fine_region);
   fine_boxes.maxSize(4);
   amrex::MultiFab fine(fine_boxes, amrex::DistributionMapping(fine_boxes), num_components, NumGhost);
   fine.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
   for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi) {
      const auto values = fine.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real density = 1.0_rt + 0.1_rt * i + 0.01_rt * j + 0.001_rt * k;
         for (int q = 0; q < num_components; ++q) values(i, j, k, q) = 0.0_rt;
         values(i, j, k, RHO) = density;
         values(i, j, k, ENERGY) = 5.0_rt;
         values(i, j, k, NVAR_HYDRO) = 0.2_rt * density;
      });
   }
   BoundaryData boundary;
   boundary.mesh_coordinate = MeshSpherical;
   boundary.mesh_axes = mesh_axes_layout(MeshSpherical,
#if AMREX_SPACEDIM == 3
      "r theta phi"
#else
      "r theta"
#endif
   );
   boundary.passive_scalars = 1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      boundary.lo_kind[dir] = static_cast<int>(BoundaryKind::Symmetry);
      boundary.hi_kind[dir] = static_cast<int>(BoundaryKind::Symmetry);
   }
   boundary.lo_kind[0] = static_cast<int>(BoundaryKind::Origin);
   boundary.origin_lo[0] = 1;
   boundary.hi_kind[0] = static_cast<int>(BoundaryKind::Outflow);
   for (int q = 0; q < num_components; ++q) {
      boundary.origin_state_parity[q] = state_grid_field_origin_parity(
         MeshSpherical, boundary.mesh_axes, PhysicalSystem::HD, 1, q);
   }
   amrex::Vector<amrex::BCRec> bcs(num_components);
   for (int q = 0; q < num_components; ++q) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
         bcs[q].setLo(dir, amrex_state_bc_type(static_cast<BoundaryKind>(boundary.lo_kind[dir]), q, dir, false));
         bcs[q].setHi(dir, amrex_state_bc_type(static_cast<BoundaryKind>(boundary.hi_kind[dir]), q, dir, false));
         if (periodic[dir]) bcs[q].setLo(dir, amrex::BCType::int_dir);
         if (periodic[dir]) bcs[q].setHi(dir, amrex::BCType::int_dir);
      }
   }
   using BoundaryFunc = amrex::GpuBndryFuncFab<StateFillPatchBoundary>;
   using PhysicalBC = amrex::PhysBCFunct<BoundaryFunc>;
   const BoundaryFunc boundary_func(StateFillPatchBoundary{boundary});
   amrex::Vector<PhysicalBC> physical_bc;
   for (auto const& geom : geometry) physical_bc.emplace_back(geom, bcs, boundary_func);
   MeshBasisStateInterpolater interpolater;
   interpolater.configure(base_interpolater, MeshSpherical, boundary.mesh_axes, true,
      PhysicalSystem::HD, EOSIdealGas, EOSParams{}, EOSThermodynamicState{}, FixedSpacetimeParameters{}, false);
   const amrex::BoxArray source_boxes(radial_origin_source_box(geometry[1], boundary, fine.nGrowVect()));
   amrex::MultiFab source_values(source_boxes, amrex::DistributionMapping(source_boxes), num_components, 0);
   source_values.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
   amrex::Vector<amrex::Vector<amrex::MultiFab*>> states{{&coarse_old, &coarse_new}, {&fine}};
   amrex::Vector<amrex::Vector<amrex::Real>> times{{0.0_rt, 1.0_rt}, {0.25_rt}};
   amrex::FillPatchNLevels(source_values, 1, amrex::IntVect::TheZeroVector(), 0.25_rt, states, times,
      0, 0, num_components, geometry, physical_bc, 0, amrex::Vector<amrex::IntVect>{amrex::IntVect(2)},
      &interpolater, bcs, 0);
   fill_radial_origin_from_source(fine, source_values, geometry[1], boundary);
   // Local physical-BC refreshes must preserve the nonlocal hierarchy donors.
   apply_configured_state_boundary(fine, geometry[1], boundary);

   amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpSum, amrex::ReduceOpSum> operations;
   amrex::ReduceData<amrex::Real, amrex::Long, amrex::Long> data(operations);
   using Result = decltype(data)::Type;
   amrex::Box ghosts = fine_domain;
   ghosts.setSmall(0, -NumGhost);
   ghosts.setBig(0, -1);
   for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi) {
      const auto values = fine.const_array(mfi);
      operations.eval(fine[mfi].box() & ghosts, data,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> Result {
            const int donor_r = -1 - i;
            const int donor_theta = half_theta ? j : 15 - j;
#if AMREX_SPACEDIM == 3
            const int donor_phi = (k + 8) % 16;
#else
            const int donor_phi = 0;
#endif
            const bool live_fine = fine_region.contains(amrex::IntVect(AMREX_D_DECL(donor_r, donor_theta, donor_phi)));
            const amrex::Real density = live_fine
               ? 1.0_rt + 0.1_rt * donor_r + 0.01_rt * donor_theta + 0.001_rt * donor_phi : 4.5_rt;
            amrex::Real error = 0.0_rt;
            for (int q = 0; q < num_components; ++q) {
               const amrex::Real expected = q == RHO ? density : (q == ENERGY ? (live_fine ? 5.0_rt : 20.0_rt)
                  : (q == NVAR_HYDRO ? 0.2_rt * density : 0.0_rt));
               const amrex::Real actual = values(i, j, k, q);
               error = amrex::max(error, std::isfinite(actual) ? std::abs(actual - expected)
                  : std::numeric_limits<amrex::Real>::max());
            }
            return {error, live_fine ? 1 : 0, live_fine ? 0 : 1};
         });
   }
   const auto result = data.value();
   amrex::Real error = amrex::get<0>(result);
   amrex::Long live_fine = amrex::get<1>(result);
   amrex::Long from_coarse = amrex::get<2>(result);
   amrex::ParallelDescriptor::ReduceRealMax(error);
   amrex::ParallelDescriptor::ReduceLongSum(live_fine);
   amrex::ParallelDescriptor::ReduceLongSum(from_coarse);
   const bool passed = error < 128.0_rt * std::numeric_limits<amrex::Real>::epsilon()
      && live_fine > 0 && from_coarse > 0;
   amrex::Print() << "partial_hierarchy " << (half_theta ? "half" : "full")
      << " fine=" << live_fine << " coarse=" << from_coarse << " error=" << error
      << " " << (passed ? "PASS" : "FAIL") << '\n';
   return passed;
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   bool passed = true;
   {
      amrex::Print() << "dimension\tangular_domain\tstate\tboxes\tghost_instances"
         "\tnonfinite\tscalar_error\tpolar_error\taxial_error\tstatus\n";
      for (int angular_domain = 0; angular_domain < 3; ++angular_domain) {
         for (bool magnetic : {false, true}) {
            passed = run_case(angular_domain != 0, angular_domain == 2, magnetic, 16, 16) && passed;
            passed = run_case(angular_domain != 0, angular_domain == 2, magnetic, 1, 1) && passed;
#if AMREX_SPACEDIM == 3
            passed = run_case(angular_domain != 0, angular_domain == 2, magnetic, 4, 2) && passed;
#endif
         }
      }
      for (bool half_theta : {false, true}) {
         passed = check_partial_hierarchy(half_theta, &amrex::cell_cons_interp) && passed;
         passed = check_partial_hierarchy(half_theta, &amrex::protected_interp) && passed;
      }
      if (passed) amrex::Print() << "PASS spherical-origin topology check\n";
   }
   amrex::Finalize();
   return passed ? 0 : 1;
}
