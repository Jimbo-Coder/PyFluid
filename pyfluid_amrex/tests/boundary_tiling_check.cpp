#include <AMReX.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_OpenMP.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_RealBox.H>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"
#include "relativity/z4c_boundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace pyfluid_amrex;

constexpr int NumCells = 16;
constexpr int NumGhost = 2;

void require(bool condition, std::string const& message)
{
   if (!condition) {
      throw std::runtime_error(message);
   }
}

bool close_enough(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = std::max(
      1.0, std::max(std::abs(actual), std::abs(expected))
   );
   return std::abs(actual - expected) <= 2.0e-12 * scale;
}

amrex::Box domain_box()
{
   return {
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(
         NumCells - 1, NumCells - 1, NumCells - 1
      )),
   };
}

amrex::Geometry nonperiodic_geometry()
{
   const amrex::RealBox physical_domain(
      AMREX_D_DECL(0.0, 0.0, 0.0),
      AMREX_D_DECL(1.0, 1.0, 1.0)
   );
   const amrex::Array<int, AMREX_SPACEDIM> periodic{
      AMREX_D_DECL(0, 0, 0)
   };
   return {
      domain_box(), physical_domain, amrex::CoordSys::cartesian, periodic
   };
}

BoundaryData outflow_boundaries()
{
   BoundaryData boundary;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      boundary.lo_kind[direction] = static_cast<int>(BoundaryKind::Outflow);
      boundary.hi_kind[direction] = static_cast<int>(BoundaryKind::Outflow);
   }
   boundary.mesh_coordinate = MeshCartesian;
   return boundary;
}

template <std::size_t NumComponents>
void initialize_constant_valid_state(
   amrex::MultiFab& state,
   std::array<amrex::Real, NumComponents> const& values
)
{
   const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();
   state.setVal(nan, 0, static_cast<int>(NumComponents), state.nGrowVect());
   for (int component = 0; component < static_cast<int>(NumComponents);
        ++component) {
      state.setVal(values[component], component, 1, 0);
   }
}

void require_constant_component(
   amrex::MultiFab const& state,
   int component,
   amrex::Real expected,
   int ghost_cells,
   std::string const& label
)
{
   const amrex::Real minimum = state.min(component, ghost_cells);
   const amrex::Real maximum = state.max(component, ghost_cells);
   require(
      close_enough(minimum, expected) && close_enough(maximum, expected),
      label + " is not constant through the requested ghost region"
   );
}

void require_constant_component(
   amrex::MultiFab const& state,
   amrex::Box const& region,
   int component,
   amrex::Real expected,
   int ghost_cells,
   std::string const& label
)
{
   const amrex::Real minimum = state.min(region, component, ghost_cells);
   const amrex::Real maximum = state.max(region, component, ghost_cells);
   require(
      close_enough(minimum, expected) && close_enough(maximum, expected),
      label + " is not constant through the requested region"
   );
}

int tile_count(amrex::MultiFab const& state)
{
   int count = 0;
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid();
        ++mfi) {
      ++count;
   }
   return count;
}

void check_ordinary_boundary()
{
   const amrex::BoxArray boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const std::array<amrex::Real, NVAR_HYDRO> expected{
      1.0, 2.0, 3.0, 4.0, 5.0
   };
   initialize_constant_valid_state(state, expected);
   require(
      amrex::Gpu::inLaunchRegion() || tile_count(state) > 1,
      "ordinary boundary fixture did not create multiple CPU tiles"
   );
   apply_state_boundary<NVAR_HYDRO>(
      state, nonperiodic_geometry(), outflow_boundaries()
   );
   require(
      !state.contains_nan(0, NVAR_HYDRO, NumGhost),
      "ordinary boundary left a nonfinite ghost cell"
   );
   for (int component = 0; component < NVAR_HYDRO; ++component) {
      require_constant_component(
         state, component, expected[component], NumGhost,
         "ordinary boundary component"
      );
   }
}

void check_patch_union_reflecting_boundary()
{
   amrex::Box patch_box = domain_box();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      patch_box.setSmall(direction, NumGhost);
      patch_box.setBig(direction, NumCells - NumGhost - 1);
   }
   const amrex::BoxArray boxes(patch_box);
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const std::array<amrex::Real, NVAR_HYDRO> expected{
      1.0, 2.0, 3.0, 4.0, 5.0
   };
   initialize_constant_valid_state(state, expected);

   PatchDomainBoxes patch_domain;
   patch_domain.count = 1;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      patch_domain.small[direction] = NumGhost;
      patch_domain.big[direction] = NumCells - NumGhost - 1;
   }
   apply_patch_union_reflecting_boundary(
      state, nonperiodic_geometry(), patch_domain
   );

   amrex::Box low_x_strip = patch_box;
   low_x_strip.setSmall(0, 0);
   low_x_strip.setBig(0, NumGhost - 1);
   require_constant_component(
      state, low_x_strip, RHO, expected[RHO], NumGhost,
      "patch-union reflected density"
   );
   require_constant_component(
      state, low_x_strip, MOM1, -expected[MOM1], NumGhost,
      "patch-union reflected normal momentum"
   );
}

struct MinkowskiMetricAccessor {
   AMREX_GPU_HOST_DEVICE
   SpacetimeMetric operator()(int, int, int) const noexcept
   {
      return minkowski_spacetime_metric();
   }
};

void check_relativistic_boundary()
{
   const amrex::BoxArray boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const EOSParams eos{};
   const RHDPrimitive primitive{
      1.0, 1.5, 1.0, 0.1, -0.05, 0.02
   };
   const RHDConserved conserved = generated::rhd_primitive_to_conserved(
      primitive, minkowski_spacetime_metric()
   );
   const std::array<amrex::Real, NVAR_HYDRO> expected{
      conserved.dens,
      conserved.mom1,
      conserved.mom2,
      conserved.mom3,
      conserved.tau,
   };
   initialize_constant_valid_state(state, expected);
   apply_metric_state_boundary<PhysicalSystem::RHD, EOSIdealGas>(
      state,
      nonperiodic_geometry(),
      outflow_boundaries(),
      eos,
      [](amrex::MFIter const&) { return MinkowskiMetricAccessor{}; }
   );
   require(
      !state.contains_nan(0, NVAR_HYDRO, NumGhost),
      "relativistic boundary left a nonfinite ghost cell"
   );
   for (int component = 0; component < NVAR_HYDRO; ++component) {
      require_constant_component(
         state, component, expected[component], NumGhost,
         "relativistic boundary component"
      );
   }
}

void check_ct_face_boundary()
{
   const amrex::BoxArray boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   FaceMagneticField face_b;
   allocate_face_magnetic_field(face_b, boxes, distribution, NumGhost);
   const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      face_b.b[direction]->setVal(
         nan, 0, 1, face_b.b[direction]->nGrowVect()
      );
      face_b.b[direction]->setVal(
         static_cast<amrex::Real>(direction + 1), 0, 1, 0
      );
   }
   apply_face_b_boundary(
      face_b, nonperiodic_geometry(), outflow_boundaries()
   );
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      require(
         !face_b.b[direction]->contains_nan(0, 1, NumGhost),
         "CT boundary left a nonfinite face ghost"
      );
      require_constant_component(
         *face_b.b[direction], 0,
         static_cast<amrex::Real>(direction + 1), NumGhost,
         "CT face boundary component"
      );
   }
}

void check_nodal_z4c_reflection_boundary()
{
   const amrex::Geometry geometry = nonperiodic_geometry();
   const amrex::BoxArray boxes = amrex::convert(
      amrex::BoxArray(domain_box()), amrex::IntVect::TheNodeVector()
   );
   const amrex::DistributionMapping distribution(boxes);
   constexpr int num_components = 2;
   amrex::MultiFab state(boxes, distribution, num_components, NumGhost);
   state.setVal(
      std::numeric_limits<amrex::Real>::quiet_NaN(),
      0,
      num_components,
      state.nGrowVect()
   );
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto values = state.array(mfi);
      amrex::ParallelFor(
         box,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real value = 100.0 * i + 10.0 * j + k;
            values(i, j, k, 0) = value;
            values(i, j, k, 1) = value;
         }
      );
   }
   amrex::Gpu::streamSynchronize();

   amrex::Vector<amrex::BCRec> records(num_components);
   for (int component = 0; component < num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         records[component].setLo(
            direction,
            component == 0
               ? amrex::BCType::reflect_even
               : amrex::BCType::reflect_odd
         );
         records[component].setHi(direction, amrex::BCType::foextrap);
      }
   }
   using NodalBoundary = amrex::GpuBndryFuncFab<Z4cFillPatchBoundary>;
   amrex::PhysBCFunct<NodalBoundary> physical_boundary(
      geometry, records, NodalBoundary{Z4cFillPatchBoundary{}}
   );
   physical_boundary(
      state,
      0,
      num_components,
      amrex::IntVect(NumGhost),
      0.0,
      0
   );
   amrex::Gpu::streamSynchronize();

   const amrex::FArrayBox& values = state[0];
   const auto sample = [&](int i, int j, int k, int component) {
      return values(amrex::IntVect(AMREX_D_DECL(i, j, k)), component);
   };
   require(
      close_enough(sample(-1, 2, 2, 0), sample(1, 2, 2, 0)),
      "nodal even reflection used the wrong mirror point"
   );
   require(
      close_enough(sample(-1, 2, 2, 1), -sample(1, 2, 2, 1)),
      "nodal odd reflection used the wrong parity"
   );
#if (AMREX_SPACEDIM >= 2)
   require(
      close_enough(sample(-1, -2, 2, 0), sample(1, 2, 2, 0)),
      "nodal corner reflection used the wrong mirror point"
   );
   require(
      close_enough(sample(-1, -2, 2, 1), sample(1, 2, 2, 1)),
      "nodal corner reflection did not compose odd parities"
   );
#endif
#if (AMREX_SPACEDIM == 3)
   require(
      close_enough(sample(-1, -2, -1, 1), -sample(1, 2, 1, 1)),
      "nodal three-plane reflection did not compose odd parities"
   );
#endif
}

} // namespace

int main(int argc, char* argv[])
{
   amrex::Initialize(argc, argv);
   int status = 0;
   const amrex::IntVect previous_tile_size =
      amrex::FabArrayBase::mfiter_tile_size;
   const int previous_thread_count = amrex::OpenMP::get_max_threads();
   try {
      amrex::FabArrayBase::mfiter_tile_size = amrex::IntVect(4);
      amrex::OpenMP::set_num_threads(2);
      check_ordinary_boundary();
      check_patch_union_reflecting_boundary();
      check_relativistic_boundary();
      check_ct_face_boundary();
      check_nodal_z4c_reflection_boundary();
      amrex::Print() << "PASS boundary grown-tile ownership check\n";
   } catch (std::exception const& error) {
      amrex::Print() << "FAIL " << error.what() << '\n';
      status = 1;
   }
   amrex::OpenMP::set_num_threads(previous_thread_count);
   amrex::FabArrayBase::mfiter_tile_size = previous_tile_size;
   amrex::Finalize();
   return status;
}
