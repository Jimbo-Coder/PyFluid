#include "pyfluid_amrex/boundary.hpp"

namespace pyfluid_amrex {
namespace {

bool same_angle(amrex::Real first, amrex::Real second)
{
   return std::abs(first - second) <= 1.0e-12_rt * constants::pi;
}

bool equatorial_boundary_kind(BoundaryData const& bc, int kind)
{
   const int symmetry = static_cast<int>(BoundaryKind::Symmetry);
   const int reflecting = static_cast<int>(BoundaryKind::Reflecting);
   // Hydro reflecting walls have exactly the equatorial vector parity. MHD
   // requires the explicit symmetry BC because its axial signs differ.
   return kind == symmetry || (bc.mhd == 0 && kind == reflecting);
}

bool equatorial_half_domain(BoundaryData const& bc, amrex::Geometry const& geom, int theta_dir)
{
   return (same_angle(geom.ProbHi(theta_dir), constants::pi / 2.0_rt)
           && equatorial_boundary_kind(bc, bc.hi_kind[theta_dir]))
      || (same_angle(geom.ProbLo(theta_dir), constants::pi / 2.0_rt)
          && equatorial_boundary_kind(bc, bc.lo_kind[theta_dir]));
}

// All indices refer to cell averages in the runtime axis order. The component
// signs come from the field catalog, including axial-vector reflection signs.
struct RadialOriginMap {
   amrex::GpuArray<int, AMREX_SPACEDIM> lo{};
   amrex::GpuArray<int, AMREX_SPACEDIM> hi{};
   amrex::GpuArray<int, MaxStateComponents> antipodal_parity{};
   amrex::GpuArray<int, MaxStateComponents> pole_parity{};
   amrex::GpuArray<int, MaxStateComponents> equatorial_parity{};
   int radial_dir = -1;
   int theta_dir = -1;
   int phi_dir = -1;
   int half_theta = 0;
   int lo_pole = 0;
   int hi_pole = 0;
   int lo_equator = 0;
   int hi_equator = 0;

   AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real value(amrex::Array4<amrex::Real const> const& source,
                     int i, int j, int k, int component) const noexcept
   {
      int donor[3] = {i, j, k};
      donor[radial_dir] = 2 * lo[radial_dir] - 1 - donor[radial_dir];
      int parity = antipodal_parity[component];
      int phi_shift = phi_dir >= 0 ? (hi[phi_dir] - lo[phi_dir] + 1) / 2 : 0;
      if (theta_dir >= 0) {
         int& theta = donor[theta_dir];
         if (half_theta == 0) {
            theta = lo[theta_dir] + hi[theta_dir] - theta;
         }
         // Normalize coordinate identifications before reading the donor. This
         // also handles origin/pole/equator corners and more than one crossing.
         while (theta < lo[theta_dir] || theta > hi[theta_dir]) {
            const bool low = theta < lo[theta_dir];
            const bool pole = low ? lo_pole != 0 : hi_pole != 0;
            const bool equator = low ? lo_equator != 0 : hi_equator != 0;
            if (!pole && !equator) {
               break; // Other angular boundaries use their supplied BC ghosts.
            }
            theta = low ? 2 * lo[theta_dir] - 1 - theta : 2 * hi[theta_dir] + 1 - theta;
            if (pole) {
               parity *= pole_parity[component];
               phi_shift += phi_dir >= 0 ? (hi[phi_dir] - lo[phi_dir] + 1) / 2 : 0;
            } else {
               parity *= equatorial_parity[component];
            }
         }
      }
      if (phi_dir >= 0) {
         donor[phi_dir] = wrapped_index(donor[phi_dir] + phi_shift, lo[phi_dir], hi[phi_dir]);
      }
      return static_cast<amrex::Real>(parity) * source(donor[0], donor[1], donor[2], component);
   }
};

RadialOriginMap make_radial_origin_map(BoundaryData const& bc, amrex::Geometry const& geom, int num_components)
{
   RadialOriginMap mapping;
   mapping.radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   mapping.phi_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisPhi);
   mapping.theta_dir = bc.mesh_coordinate == MeshSpherical
      ? mesh_direction_for_axis(bc.mesh_axes, MeshAxisTheta) : -1;
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      mapping.lo[dir] = geom.Domain().smallEnd(dir);
      mapping.hi[dir] = geom.Domain().bigEnd(dir);
   }
   int theta_component = -1;
   int phi_component = -1;
   for (int component = 0; component < 3; ++component) {
      const int axis = mesh_vector_axis_for_component(bc.mesh_coordinate, bc.mesh_axes, component);
      if (axis == MeshAxisTheta) theta_component = component;
      if (axis == MeshAxisPhi) phi_component = component;
   }
   if (mapping.theta_dir >= 0) {
      const int theta_dir = mapping.theta_dir;
      mapping.half_theta = equatorial_half_domain(bc, geom, theta_dir) ? 1 : 0;
      mapping.lo_pole = same_angle(geom.ProbLo(theta_dir), 0.0_rt) ? 1 : 0;
      mapping.hi_pole = same_angle(geom.ProbHi(theta_dir), constants::pi) ? 1 : 0;
      mapping.lo_equator = same_angle(geom.ProbLo(theta_dir), constants::pi / 2.0_rt)
         && equatorial_boundary_kind(bc, bc.lo_kind[theta_dir]);
      mapping.hi_equator = same_angle(geom.ProbHi(theta_dir), constants::pi / 2.0_rt)
         && equatorial_boundary_kind(bc, bc.hi_kind[theta_dir]);
   }
   const PhysicalSystem system = bc.mhd != 0 ? PhysicalSystem::MHD : PhysicalSystem::HD;
   for (int component = 0; component < num_components; ++component) {
      mapping.antipodal_parity[component] = origin_state_parity(bc, mapping.radial_dir, component);
      if (mapping.theta_dir >= 0) {
         const auto symmetry = state_grid_field_info(system, bc.passive_scalars, component).symmetry;
         const int equatorial = grid_field_reflection_parity(symmetry, theta_component);
         mapping.equatorial_parity[component] = equatorial;
         mapping.pole_parity[component] = equatorial * grid_field_reflection_parity(symmetry, phi_component);
         if (mapping.half_theta != 0) mapping.antipodal_parity[component] *= equatorial;
      }
   }
   return mapping;
}

} // namespace

void validate_origin_boundary_geometry(BoundaryData const& bc, amrex::Geometry const& geom)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (bc.origin_lo[dir] != 0 && geom.ProbLo(dir) != 0.0_rt) {
         amrex::Abort("pyfluid.bc_* = origin requires the lower radial coordinate bound to be 0");
      }
   }
   if (!needs_radial_antipodal_origin_boundary(bc)) return;
   const int phi_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisPhi);
   if (phi_dir >= 0) {
      if (!geom.isPeriodic(phi_dir)) {
         amrex::Abort("radial origin boundary with active phi requires geometry.is_periodic = 1 in phi");
      }
      const int nphi = geom.Domain().length(phi_dir);
      if (nphi != 1 && nphi % 2 != 0) {
         amrex::Abort("radial origin boundary with active phi requires one or an even number of phi cells");
      }
      if (!same_angle(geom.ProbHi(phi_dir) - geom.ProbLo(phi_dir), constants::two_pi)) {
         amrex::Abort("radial origin boundary with active phi requires a full 2*pi phi domain");
      }
   }
   const int theta_dir = bc.mesh_coordinate == MeshSpherical
      ? mesh_direction_for_axis(bc.mesh_axes, MeshAxisTheta) : -1;
   if (theta_dir >= 0
       && !same_angle(geom.ProbLo(theta_dir) + geom.ProbHi(theta_dir), constants::pi)
       && !equatorial_half_domain(bc, geom, theta_dir)) {
      amrex::Abort("spherical radial origin requires an antipodally closed theta interval or declared equatorial symmetry");
   }
}

amrex::Box radial_origin_source_box(amrex::Geometry const& geom, BoundaryData const& bc,
                                    amrex::IntVect const& num_ghosts)
{
   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   const int phi_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisPhi);
   amrex::Box source = amrex::grow(geom.Domain(), num_ghosts);
   source.setSmall(radial_dir, geom.Domain().smallEnd(radial_dir));
   source.setBig(radial_dir, geom.Domain().smallEnd(radial_dir) + num_ghosts[radial_dir] - 1);
   if (phi_dir >= 0) {
      source.setSmall(phi_dir, geom.Domain().smallEnd(phi_dir));
      source.setBig(phi_dir, geom.Domain().bigEnd(phi_dir));
   }
   return source;
}

void fill_radial_origin_from_source(amrex::MultiFab& state, amrex::MultiFab const& source_values,
                                   amrex::Geometry const& geom, BoundaryData const& bc)
{
   validate_origin_boundary_geometry(bc, geom);
   AMREX_ALWAYS_ASSERT(source_values.boxArray().size() == 1 && source_values.nComp() == state.nComp());
   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   const int radial_lo = geom.Domain().smallEnd(radial_dir);
   amrex::Box destination = amrex::grow(geom.Domain(), state.nGrowVect());
   destination.setSmall(radial_dir, radial_lo - state.nGrowVect()[radial_dir]);
   destination.setBig(radial_dir, radial_lo - 1);
   amrex::MultiFab origin_values(amrex::BoxArray(destination), source_values.DistributionMap(), state.nComp(), 0);
   const auto mapping = make_radial_origin_map(bc, geom, state.nComp());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
      for (amrex::MFIter mfi(origin_values, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const auto dst = origin_values.array(mfi);
         const auto src = source_values.const_array(mfi);
         amrex::ParallelFor(mfi.tilebox(), state.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               dst(i, j, k, component) = mapping.value(src, i, j, k, component);
            });
      }
   }
   state.ParallelCopy(origin_values, 0, 0, state.nComp(), amrex::IntVect::TheZeroVector(), state.nGrowVect());
}

void apply_radial_antipodal_origin_boundary(amrex::MultiFab& state, amrex::Geometry const& geom,
                                            BoundaryData const& bc)
{
   if (!needs_radial_antipodal_origin_boundary(bc)) return;
   const int radial_dir = mesh_direction_for_axis(bc.mesh_axes, MeshAxisR);
   if (state.nGrowVect()[radial_dir] <= 0) return;
   const amrex::Box source_box = radial_origin_source_box(geom, bc, state.nGrowVect());
   // A partial fine level has no complete donor band. Its hierarchy owner fills
   // this same band with FillPatchNLevels after the ordinary local physical BC.
   if (!state.boxArray().contains(source_box & geom.Domain())) return;
   const amrex::BoxArray source_boxes(source_box);
   amrex::MultiFab source_values(source_boxes, amrex::DistributionMapping(source_boxes), state.nComp(), 0);
   source_values.ParallelCopy(state, 0, 0, state.nComp(), state.nGrowVect(), amrex::IntVect::TheZeroVector(),
                             geom.periodicity());
   fill_radial_origin_from_source(state, source_values, geom, bc);
}

} // namespace pyfluid_amrex
