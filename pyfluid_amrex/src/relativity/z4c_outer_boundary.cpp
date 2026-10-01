#include "pyfluid_amrex/z4c_runtime.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_BC_TYPES.H>
#include "pyfluid_amrex/spacetime_fields.hpp"
#include "z4c_sommerfeld.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace {

struct OpenBoundaryFaces {
   amrex::GpuArray<int, 3> low{}, high{};
   bool any = false;
};

OpenBoundaryFaces open_boundary_faces(RunConfig const& cfg, amrex::Geometry const& geometry)
{
   OpenBoundaryFaces faces;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      const auto is_open = [&](bool high) {
         const BoundaryKind kind = side_boundary_kind(cfg, direction, high);
         return !geometry.isPeriodic(direction) && kind != BoundaryKind::Reflecting
            && kind != BoundaryKind::Symmetry;
      };
      faces.low[direction] = is_open(false);
      faces.high[direction] = is_open(true);
      faces.any = faces.any || faces.low[direction] || faces.high[direction];
   }
   return faces;
}

} // namespace

amrex::Vector<amrex::BCRec> z4c_boundary_records(
   RunConfig const& cfg, amrex::Geometry const& geometry)
{
   const auto faces = open_boundary_faces(cfg, geometry);
   if (spacetime_evolves(cfg) && cfg.spacetime_outer_boundary != "extrapolate") {
      const amrex::Real origin[3]{cfg.spacetime_origin1, cfg.spacetime_origin2, cfg.spacetime_origin3};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const amrex::Real margin = cfg.spacetime_sommerfeld_power < 0
            ? 0.0_rt : geometry.CellSize(direction);
         if ((faces.low[direction] && !(origin[direction] > geometry.ProbLo(direction) + margin))
             || (faces.high[direction] && !(origin[direction] < geometry.ProbHi(direction) - margin))) {
            amrex::Abort("radiative spacetime boundaries require spacetime.origin "
               "strictly inside every open face (by more than one cell when sommerfeld_power>=0); "
               "a plane through the origin may be a symmetry plane");
         }
      }
   }
   amrex::Vector<amrex::BCRec> records(Z4C_NUM_COMPONENTS);
   int component = 0;
   for (GridFieldInfo const& field : z4c_state_grid_fields().fields) {
      for (GridFieldComponentInfo const& info : field.components) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            const int reflection = grid_field_reflection_parity(info.symmetry, direction) < 0
               ? amrex::BCType::reflect_odd : amrex::BCType::reflect_even;
            records[component].setLo(direction, geometry.isPeriodic(direction)
               ? amrex::BCType::int_dir : faces.low[direction] ? amrex::BCType::hoextrap : reflection);
            records[component].setHi(direction, geometry.isPeriodic(direction)
               ? amrex::BCType::int_dir : faces.high[direction] ? amrex::BCType::hoextrap : reflection);
         }
         ++component;
      }
   }
   return records;
}

void apply_z4c_outer_boundary_rhs(amrex::MultiFab const& state,
   amrex::MultiFab& rhs,
   amrex::Geometry const& geometry, RunConfig const& cfg)
{
   if (cfg.spacetime_outer_boundary == "extrapolate") return;
   const auto faces = open_boundary_faces(cfg, geometry);
   if (!faces.any) return;
   AMREX_ALWAYS_ASSERT(state.is_nodal());
   const amrex::Box domain = amrex::surroundingNodes(geometry.Domain());
   const auto node_lo = amrex::lbound(domain);
   const auto node_hi = amrex::ubound(domain);
   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::GpuArray<amrex::Real, 3> origin{{
      cfg.spacetime_origin1, cfg.spacetime_origin2, cfg.spacetime_origin3}};
   amrex::GpuArray<amrex::Real, 3> inv_spacing{};
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      inv_spacing[direction] = 1.0_rt / spacing[direction];
   }
   const auto parameters = z4c_evolution_parameters(cfg);
   const auto lapse_gauge = z4c_lapse_gauge(cfg);
   const auto shift_gauge = z4c_shift_gauge(cfg);
   const int nonradiative_power = cfg.spacetime_sommerfeld_power;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      bool touches = false;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         touches = touches || (faces.low[direction] && box.smallEnd(direction) == domain.smallEnd(direction))
            || (faces.high[direction] && box.bigEnd(direction) == domain.bigEnd(direction));
      }
      if (!touches) continue;
      const auto input = state.const_array(mfi);
      const auto output = rhs.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const int point[3]{i, j, k};
         const int low[3]{node_lo.x, node_lo.y, node_lo.z};
         const int high[3]{node_hi.x, node_hi.y, node_hi.z};
         bool boundary = false;
         amrex::GpuArray<int, 3> donor{{i, j, k}};
         amrex::GpuArray<amrex::Real, 3> position{};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            boundary = boundary || (faces.low[direction] && point[direction] == low[direction])
               || (faces.high[direction] && point[direction] == high[direction]);
            if (faces.low[direction] && point[direction] == low[direction]) {
               ++donor[direction];
            }
            if (faces.high[direction] && point[direction] == high[direction]) {
               --donor[direction];
            }
            position[direction] = lower[direction] + (point[direction] - low[direction])
               * spacing[direction] - origin[direction];
         }
         // Shift off EVERY incident open face. The donor is then a bulk RHS
         // node, never a concurrently overwritten boundary value. It lies in
         // this nodal FAB even for a one-cell-wide boundary block: no RHS
         // halo exchange, snapshot, ordered face sweep or corner race.
         if (!boundary) return;
         z4c_radiative_rhs_at_point(input, output, i, j, k, donor,
            position, inv_spacing, parameters, lapse_gauge, shift_gauge,
            nonradiative_power);
      });
   }
}

} // namespace pyfluid_amrex
