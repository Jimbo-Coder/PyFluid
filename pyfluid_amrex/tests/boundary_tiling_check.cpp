#include <AMReX.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_OpenMP.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_RealBox.H>

#include "pyfluid_amrex/boundary.hpp"
#include "pyfluid_amrex/patch_domain.hpp"
#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/relativistic_boundary.hpp"
#include "pyfluid_amrex/rmhd_riemann.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"
#include "pyfluid_amrex/z4c_centering_representation.hpp"
#include "relativity/z4c_boundary.hpp"
#include "relativity/z4c_sommerfeld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace amrex::literals;

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
      1.0_rt, std::max(std::abs(actual), std::abs(expected))
   );
   return std::abs(actual - expected) <= 2.0e-12_rt * scale;
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

amrex::BoxArray split_boxes(amrex::Box const& box)
{
   amrex::BoxArray boxes(box);
   boxes.maxSize(8);
   return boxes;
}

amrex::Geometry nonperiodic_geometry()
{
   const amrex::RealBox physical_domain(
      AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt),
      AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt)
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
   const amrex::BoxArray boxes = split_boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const std::array<amrex::Real, NVAR_HYDRO> expected{
      1.0_rt, 2.0_rt, 3.0_rt, 4.0_rt, 5.0_rt
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
   const amrex::BoxArray boxes = split_boxes(patch_box);
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const std::array<amrex::Real, NVAR_HYDRO> expected{
      1.0_rt, 2.0_rt, 3.0_rt, 4.0_rt, 5.0_rt
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
   const amrex::BoxArray boxes = split_boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, NVAR_HYDRO, NumGhost);
   const EOSParams eos{};
   const RHDPrimitive primitive{
      1.0_rt, 1.5_rt, 1.0_rt, 0.1_rt, -0.05_rt, 0.02_rt
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
   state.FillBoundary(nonperiodic_geometry().periodicity());
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

void check_shifted_diode_kinematics()
{
   SpacetimeMetric metric;
   metric.lapse = 0.8_rt;
   metric.shift1 = 0.4_rt;
   RHDPrimitive primitive{1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.95_rt, 0.0_rt};
   apply_relativistic_diode_velocity(
      primitive, metric, false, true, false, false, false, false
   );
   require(close_enough(metric.lapse * primitive.v1 - metric.shift1, 0.0_rt),
           "shifted diode retained incoming transport");
   require(close_enough(spatial_vector_norm_squared(
      primitive.v1, primitive.v2, primitive.v3, metric), 0.95_rt * 0.95_rt),
      "shifted diode failed causal tangential completion");
   require(primitive.rho == 1.0_rt && primitive.pressure == 1.0_rt,
           "diode changed thermodynamics");

   // A ghost diode constrains its physical flux, not a two-state Riemann
   // flux. Retain this counterexample so reports cannot claim exact no-inflow.
   metric.shift1 = 0.16_rt;
   const RHDPrimitive interior{1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
   RHDPrimitive ghost = interior;
   apply_relativistic_diode_velocity(
      ghost, metric, false, true, false, false, false, false
   );
   const auto flux_left = generated::rhd_flux<0>(interior, metric);
   const auto flux_right = generated::rhd_flux<0>(ghost, metric);
   const auto state_left = generated::rhd_primitive_to_conserved(interior, metric);
   const auto state_right = generated::rhd_primitive_to_conserved(ghost, metric);
   const auto speed_left = generated::rhd_wave_speeds<0>(interior, metric, 4.0_rt / 9.0_rt);
   const auto speed_right = generated::rhd_wave_speeds<0>(ghost, metric, 4.0_rt / 9.0_rt);
   const auto numerical_flux = generated::rhd_hll_flux(
      flux_left, flux_right, state_left, state_right,
      std::min(speed_left.minus, speed_right.minus),
      std::max(speed_left.plus, speed_right.plus)
   );
   require(std::abs(flux_right.dens) < 1e-14_rt,
           "diode ghost physical baryon flux is not zero");
   amrex::Print() << "diode ghost physical mass flux=" << flux_right.dens
                  << " two-state HLL mass flux=" << numerical_flux.dens << '\n';
   RelativisticDiodeBoundary boundary;
   boundary.high_faces = 1;
   boundary.upper[0] = 4;
   RHDPrimitive face_left = interior;
   RHDPrimitive face_right = ghost;
   require(apply_relativistic_diode_face<0>(
      face_left, face_right, metric, boundary, 4, 0, 0),
      "diode did not recognize its high face");
   const auto face_flux = generated::rhd_flux<0>(face_left, metric);
   const auto face_state = generated::rhd_primitive_to_conserved(face_left, metric);
   const auto closed_flux = generated::rhd_hll_flux(
      face_flux, generated::rhd_flux<0>(face_right, metric),
      face_state, generated::rhd_primitive_to_conserved(face_right, metric),
      speed_right.minus, speed_right.plus
   );
   require(std::abs(closed_flux.dens) < 1e-14_rt,
           "face diode retained numerical HLL mass influx");
   require(close_enough(closed_flux.mom1, metric.lapse * interior.pressure),
           "face diode discarded boundary pressure stress");
   amrex::Print() << "diode face HLL mass flux=" << closed_flux.dens << '\n';
}

template <int Direction>
void check_rmhd_diode_face()
{
   const EOSParams eos{};
   for (int side : {-1, 1}) {
      SpacetimeMetric metric;
      metric.lapse = 0.8_rt;
      metric.gamma12 = 0.2_rt;
      metric.gamma22 = 1.04_rt;
      metric.inv_gamma11 = 1.04_rt;
      metric.inv_gamma12 = -0.2_rt;
      amrex::Real* shifts[]{&metric.shift1, &metric.shift2, &metric.shift3};
      *shifts[Direction] = 0.16_rt * side;
      RelativisticDiodeBoundary boundary;
      boundary.low_faces = 1 << Direction;
      boundary.high_faces = 1 << Direction;
      boundary.lower[Direction] = 0;
      boundary.upper[Direction] = 4;
      int point[3]{1, 1, 1};
      point[Direction] = side < 0 ? 0 : 4;
      RMHDPrimitive left{1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.1_rt, 0.2_rt, 0.3_rt};
      RMHDPrimitive right = left;
      // Make the external state different: no numerical diffusion from an
      // external density/magnetic jump may supply mass through a diode.
      if (side < 0) left.rho = 0.8_rt;
      else right.rho = 0.8_rt;
      require(apply_relativistic_diode_face<Direction>(
         left, right, metric, boundary, point[0], point[1], point[2]),
         "RMHD diode missed a selected boundary face");
      const auto physical_flux = generated::rmhd_flux<Direction>(left, metric);
      const auto hllc = rmhd_hllc_flux<Direction>(left, right, eos, metric);
      const auto hlld = rmhd_hlld_flux<Direction>(left, right, eos, metric);
      require(std::abs(physical_flux.dens) < 1e-14_rt
              && std::abs(hllc.flux.dens) < 1e-12_rt
              && std::abs(hlld.flux.dens) < 1e-12_rt,
              "RMHD diode admitted mass through its Riemann solver");
      require(close_enough(hllc.flux.mom1, physical_flux.mom1)
              && close_enough(hlld.flux.mom1, physical_flux.mom1),
              "RMHD diode did not retain physical magnetic/pressure stresses");
      // Outgoing transport is retained, and non-boundary faces are untouched.
      amrex::Real* velocities[]{&left.v1, &left.v2, &left.v3};
      *velocities[Direction] = 0.4_rt * side;
      right = left;
      apply_relativistic_diode_face<Direction>(
         left, right, metric, boundary, point[0], point[1], point[2]);
      require(close_enough(*velocities[Direction], 0.4_rt * side),
              "diode changed outgoing transport");
      point[Direction] = 2;
      require(!apply_relativistic_diode_face<Direction>(
         left, right, metric, boundary, point[0], point[1], point[2]),
         "diode modified an internal face");
   }
}

struct ConstantMetricAccessor {
   SpacetimeMetric metric;

   AMREX_GPU_HOST_DEVICE
   SpacetimeMetric operator()(int, int, int) const noexcept { return metric; }
};

template <PhysicalSystem System>
void check_relativistic_diode_boundary()
{
   constexpr bool magnetic = System == PhysicalSystem::RMHD;
   constexpr int num_components = magnetic ? NVAR_MHD : NVAR_HYDRO;
   const amrex::BoxArray boxes = split_boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, num_components, NumGhost);
   const EOSParams eos{};
   BoundaryData boundary = outflow_boundaries();
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      boundary.lo_kind[direction] = static_cast<int>(BoundaryKind::Diode);
      boundary.hi_kind[direction] = static_cast<int>(BoundaryKind::Diode);
   }
   // Opposite shifts cover the same physical face with both transport signs;
   // zero shift retains the existing special-relativistic diode limit.
   for (amrex::Real shift_sign : {-1.0_rt, 0.0_rt, 1.0_rt}) {
      SpacetimeMetric metric;
      metric.lapse = 0.8_rt;
      metric.shift1 = 0.16_rt * shift_sign;
      metric.shift2 = -0.12_rt * shift_sign;
      metric.shift3 = 0.08_rt * shift_sign;
      metric.gamma11 = 1.2_rt;
      metric.gamma12 = 0.15_rt;
      metric.gamma22 = 0.9_rt;
      metric.gamma33 = 1.1_rt;
      const amrex::Real determinant12 = 1.2_rt * 0.9_rt - 0.15_rt * 0.15_rt;
      metric.inv_gamma11 = metric.gamma22 / determinant12;
      metric.inv_gamma12 = -metric.gamma12 / determinant12;
      metric.inv_gamma22 = metric.gamma11 / determinant12;
      metric.inv_gamma33 = 1.0_rt / metric.gamma33;
      metric.sqrt_det_gamma = std::sqrt(determinant12 * 1.1_rt);
      const RMHDPrimitive primitive{
         1.0_rt, 1.5_rt, 1.0_rt, 0.04_rt, -0.03_rt, 0.02_rt, 0.1_rt, -0.07_rt, 0.03_rt
      };
      const auto conserved_values = [&](RMHDPrimitive const& sample) {
         std::array<amrex::Real, num_components> result{};
         if constexpr (magnetic) {
            const auto conserved = generated::rmhd_primitive_to_conserved(sample, metric);
            result = {conserved.dens, conserved.mom1, conserved.mom2,
                      conserved.mom3, conserved.tau, conserved.b1,
                      conserved.b2, conserved.b3};
         } else {
            const RHDPrimitive hydro{sample.rho, sample.eps, sample.pressure,
                                     sample.v1, sample.v2, sample.v3};
            const auto conserved = generated::rhd_primitive_to_conserved(hydro, metric);
            result = {conserved.dens, conserved.mom1, conserved.mom2,
                      conserved.mom3, conserved.tau};
         }
         return result;
      };
      initialize_constant_valid_state(state, conserved_values(primitive));
      state.FillBoundary(nonperiodic_geometry().periodicity());
      apply_metric_state_boundary<System, EOSIdealGas>(
         state, nonperiodic_geometry(), boundary, eos,
         [=](amrex::MFIter const&) { return ConstantMetricAccessor{metric}; }
      );
      require(!state.contains_nan(0, num_components, NumGhost),
              "diode created nonfinite ghost data");
      // Faces and every edge/corner combination. Expected states are built
      // independently from the production clipping helper.
      for (int code = 0; code < (AMREX_SPACEDIM == 3 ? 27 : 9); ++code) {
         int remaining = code;
         amrex::IntVect point(NumCells / 2);
         RMHDPrimitive expected = primitive;
         amrex::Real* velocities[]{&expected.v1, &expected.v2, &expected.v3};
         const amrex::Real shifts[]{metric.shift1, metric.shift2, metric.shift3};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            const int side = remaining % 3 - 1;
            remaining /= 3;
            point[direction] = side < 0 ? -1 : (side > 0 ? NumCells : NumCells / 2);
            const amrex::Real rest_velocity = shifts[direction] / metric.lapse;
            if (side < 0) *velocities[direction] = std::min(*velocities[direction], rest_velocity);
            if (side > 0) *velocities[direction] = std::max(*velocities[direction], rest_velocity);
         }
         const auto expected_values = conserved_values(expected);
         for (int component = 0; component < num_components; ++component) {
            require_constant_component(
               state, amrex::Box(point, point), component,
               expected_values[component], NumGhost, "relativistic diode"
            );
         }
      }
   }
}

void check_ct_face_boundary()
{
   const amrex::BoxArray boxes = split_boxes(domain_box());
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

void check_ct_edge_sync()
{
   const amrex::BoxArray boxes = split_boxes(domain_box());
   const amrex::DistributionMapping distribution(boxes);
   EdgeElectricField edge_e;
   allocate_edge_electric_field(edge_e, boxes, distribution);
   for (int component = 0; component < 3; ++component) {
      edge_e.e[component]->setVal(static_cast<amrex::Real>(component + 1));
   }
   enforce_edge_electric_periodicity(edge_e, nonperiodic_geometry());
   for (int component = 0; component < 3; ++component) {
      require_constant_component(
         *edge_e.e[component],
         0,
         static_cast<amrex::Real>(component + 1),
         0,
         "CT edge component"
      );
   }
}

void check_nodal_z4c_reflection_boundary()
{
   const amrex::Geometry geometry = nonperiodic_geometry();
   const amrex::BoxArray boxes = amrex::convert(
      split_boxes(domain_box()), amrex::IntVect::TheNodeVector()
   );
   const amrex::DistributionMapping distribution(boxes);
   constexpr int first_component = Z4C_K_HAT;
   constexpr int num_components = 2;
   amrex::MultiFab state(boxes, distribution, first_component + num_components, NumGhost);
   state.setVal(
      std::numeric_limits<amrex::Real>::quiet_NaN(),
      first_component,
      num_components,
      state.nGrowVect()
   );
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto values = state.array(mfi);
      amrex::ParallelFor(
         box,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real value = 100.0_rt * i + 10.0_rt * j + k;
            values(i, j, k, first_component) = value;
            values(i, j, k, first_component + 1) = value;
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
      first_component,
      num_components,
      amrex::IntVect(NumGhost),
      0.0_rt,
      0
   );
   amrex::Gpu::streamSynchronize();

   const auto sample = [&](int i, int j, int k, int component) {
      const amrex::IntVect point(AMREX_D_DECL(i, j, k));
      return state.min(amrex::Box(point, point), component, NumGhost);
   };
   require(
      close_enough(sample(-1, 2, 2, first_component), sample(1, 2, 2, first_component)),
      "nodal even reflection used the wrong mirror point"
   );
   require(
      close_enough(sample(-1, 2, 2, first_component + 1), -sample(1, 2, 2, first_component + 1)),
      "nodal odd reflection used the wrong parity"
   );
#if (AMREX_SPACEDIM >= 2)
   require(
      close_enough(sample(-1, -2, 2, first_component), sample(1, 2, 2, first_component)),
      "nodal corner reflection used the wrong mirror point"
   );
   require(
      close_enough(sample(-1, -2, 2, first_component + 1), sample(1, 2, 2, first_component + 1)),
      "nodal corner reflection did not compose odd parities"
   );
#endif
#if (AMREX_SPACEDIM == 3)
   require(
      close_enough(sample(-1, -2, -1, first_component + 1), -sample(1, 2, 1, first_component + 1)),
      "nodal three-plane reflection did not compose odd parities"
   );
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real nodal_polynomial(int i, int j, int k) noexcept
{
   // Mixed terms require a genuine tensor-product corner fill.
   const amrex::Real x = i * 0.125_rt;
   const amrex::Real y = j * 0.125_rt;
   const amrex::Real z = k * 0.125_rt;
   // Dyadic coefficients isolate polynomial exactness from amplification
   // of rounded input data by signed corner-extrapolation weights.
   return 1.0_rt + x + (x * x * x * x * x + x * y * y * y * y
      + y * z * z * z * z + z * z * z * z * z) / 64.0_rt;
}

void check_nodal_z4c_extrapolation(bool degenerate = false)
{
   constexpr int first_component = Z4C_K_HAT;
   amrex::Box cells = domain_box();
   if (degenerate) cells.setBig(AMREX_SPACEDIM - 1, 0);
   const amrex::Geometry reference_geometry = nonperiodic_geometry();
   const amrex::Geometry geometry(cells, reference_geometry.ProbDomain(),
      amrex::CoordSys::cartesian, amrex::Array<int, AMREX_SPACEDIM>{});
   const amrex::BoxArray boxes = amrex::convert(
      split_boxes(cells), amrex::IntVect::TheNodeVector());
   const amrex::DistributionMapping distribution(boxes);
   amrex::MultiFab state(boxes, distribution, first_component + 2, NumGhost);
   state.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto values = state.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real value = nodal_polynomial(i,
            degenerate && AMREX_SPACEDIM == 2 ? 0 : j,
            degenerate && AMREX_SPACEDIM == 3 ? 0 : k);
         values(i, j, k, first_component) = value;
         values(i, j, k, first_component + 1) = value;
      });
   }
   state.FillBoundary(geometry.periodicity());
   amrex::Vector<amrex::BCRec> records(2);
   for (int component = 0; component < 2; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         records[component].setLo(direction, component == 0
            || (degenerate && direction == AMREX_SPACEDIM - 1)
            ? amrex::BCType::hoextrap : amrex::BCType::reflect_odd);
         records[component].setHi(direction, amrex::BCType::hoextrap);
         if (degenerate && component == 1 && direction == AMREX_SPACEDIM - 1) {
            records[component].setLo(direction, amrex::BCType::reflect_even);
            records[component].setHi(direction, amrex::BCType::reflect_even);
         }
      }
   }
   if (degenerate) {
      amrex::IntVect source(2);
      source[AMREX_SPACEDIM - 1] = 0;
      amrex::IntVect point = source;
      point[AMREX_SPACEDIM - 1] = -2;
      const auto geometry_data = geometry.data();
      const amrex::GpuArray<amrex::BCRec, 2> point_records{{records[0], records[1]}};
      for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
         if (!mfi.validbox().contains(source)) continue;
         // Call before physical ghosts are initialized. A single reflection
         // lands on the opposite ghost, so only a complete fold is valid.
         const auto values = state.array(mfi);
         amrex::ParallelFor(amrex::Box(point, point),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::IntVect index(AMREX_D_DECL(i, j, k));
               Z4cFillPatchBoundary{}(index, values, first_component, 2, geometry_data, 0.0_rt,
                  point_records.data(), 0, 0);
               AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(values(i, j, k, first_component + 1)),
                  "thin reflected nodal line read a physical ghost");
            });
      }
      amrex::Gpu::streamSynchronize();
   }
   using NodalBoundary = amrex::GpuBndryFuncFab<Z4cFillPatchBoundary>;
   amrex::PhysBCFunct<NodalBoundary> physical_boundary(
      geometry, records, NodalBoundary{Z4cFillPatchBoundary{}});
   physical_boundary(state, first_component, 2, state.nGrowVect(), 0.0_rt, 0);
   require(!state.contains_nan(first_component, 2, NumGhost), "nodal extrapolation left NaN ghosts");
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto values = state.array(mfi);
      amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const int a = i, b = j, c = k;
         if (degenerate && AMREX_SPACEDIM == 2) j = 0;
         if (degenerate && AMREX_SPACEDIM == 3) k = 0;
         values(a, b, c, first_component) -= nodal_polynomial(i, j, k);
         amrex::Real parity = 1.0_rt;
         if (i < 0) { i = -i; parity = -parity; }
         if (j < 0) { j = -j; parity = -parity; }
         if (k < 0) { k = -k; parity = -parity; }
         values(a, b, c, first_component + 1) -= parity * nodal_polynomial(i, j, k);
      });
   }
   amrex::Print() << "nodal quintic ghost errors=" << state.norm0(first_component, NumGhost)
      << '/' << state.norm0(first_component + 1, NumGhost) << '\n';
   require(state.norm0(first_component, NumGhost) < 1e-10_rt, "nodal quintic extrapolation is not exact");
   require(state.norm0(first_component + 1, NumGhost) < 1e-10_rt,
      "nodal mixed reflection/extrapolation did not compose at corners");
}

void check_z4c_sommerfeld_point()
{
   // A fixed sqrt(2)/2 gauge multiplier underbounds these strong-field
   // lapse/shift modes. CFL selection and the excision/boundary guards must
   // use the same state-dependent principal speeds.
   Z4cState strong_field;
   strong_field.lapse = 0.1_rt;
   strong_field.chi = 0.05_rt;
   Z4cEvolutionParameters parameters;
   parameters.shift_mu = 0.75_rt;
   require(close_enough(z4c_maximum_speed_squared(strong_field, parameters,
      Z4cLapseGauge::OnePlusLog, Z4cShiftGauge::Frozen), 20.0_rt),
      "strong-field lapse speed is underestimated");
   require(close_enough(z4c_maximum_speed_squared(strong_field, parameters,
      Z4cLapseGauge::Harmonic, Z4cShiftGauge::GammaDriver), 20.0_rt),
      "strong-field shift speed is underestimated");
   require(close_enough(z4c_maximum_speed_squared(strong_field, parameters,
      Z4cLapseGauge::Trumpet, Z4cShiftGauge::Frozen), 9.0_rt),
      "trumpet speed does not follow its gauge");
   require(close_enough(z4c_maximum_speed_squared(strong_field, parameters,
      Z4cLapseGauge::Frozen, Z4cShiftGauge::Frozen), 1.0_rt),
      "frozen gauge lost the physical light-speed bound");
   // Analytic outgoing spherical wave at a point on the positive first axis.
   // Other directions are locally constant, so this checks the same operator
   // in every executable dimension. No exact solution enters production.
   const amrex::Box box(amrex::IntVect(-2), amrex::IntVect(2));
   amrex::FArrayBox rhs_storage(box, Z4C_NUM_COMPONENTS, amrex::The_Managed_Arena());
   const auto rhs = rhs_storage.array();
   for (const auto lapse_gauge : {Z4cLapseGauge::Harmonic, Z4cLapseGauge::OnePlusLog,
        Z4cLapseGauge::SlowStart, Z4cLapseGauge::Trumpet, Z4cLapseGauge::Frozen}) {
      amrex::Real previous_error = 0.0_rt;
      for (amrex::Real dx : {0.1_rt, 0.05_rt}) {
         const amrex::Real lapse = lapse_gauge == Z4cLapseGauge::Trumpet ? 0.5_rt : 1.0_rt;
         const auto input = [=] AMREX_GPU_HOST_DEVICE(int i, int, int, int component) noexcept {
            if (component == Z4C_CHI || component == Z4C_GAMMA_TILDE11
                || component == Z4C_GAMMA_TILDE22 || component == Z4C_GAMMA_TILDE33) return amrex::Real(1.0_rt);
            if (component == Z4C_LAPSE) return lapse;
            if (component == Z4C_SHIFT1) return amrex::Real(0.2_rt);
            const amrex::Real radius = 2.0_rt + i * dx;
            if (component == Z4C_K_HAT || component == Z4C_THETA || component == Z4C_A_TILDE12
                || component == Z4C_GAMMA_TILDE1 || component == Z4C_GAMMA_TILDE2) {
               return std::sin(radius) / radius;
            }
            return amrex::Real(0.0_rt);
         };
         const auto local = load_z4c_state(input, 0, 0, 0);
         const amrex::Real value = std::sin(2.0_rt) / 2.0_rt;
         const amrex::Real derivative = std::cos(2.0_rt) / 2.0_rt - std::sin(2.0_rt) / 4.0_rt;
         const auto expected = [&](amrex::Real speed) {
            return (0.2_rt - speed) * derivative - speed * value / 2.0_rt;
         };
         rhs_storage.setVal(17.0_rt);
         amrex::Gpu::streamSynchronize();
         z4c_sommerfeld_rhs_at_point(input, rhs, 0, 0, 0, {{2.0_rt, 0.0_rt, 0.0_rt}},
            {{1.0_rt / dx, 1.0_rt / dx, 1.0_rt / dx}}, {}, lapse_gauge, Z4cShiftGauge::GammaDriver);
         const amrex::Real error = std::abs(rhs(0, 0, 0, Z4C_THETA) - expected(lapse));
         require(error < 1e-3_rt, "Sommerfeld outgoing-wave residual too large");
         if (previous_error > 0.0_rt) require(previous_error / error > 3.9_rt,
            "Sommerfeld derivative does not converge at second order");
         previous_error = error;
         require(close_enough(rhs(0, 0, 0, Z4C_A_TILDE12), rhs(0, 0, 0, Z4C_THETA)),
            "Sommerfeld scalar/tensor operators disagree");
         require(std::abs(rhs(0, 0, 0, Z4C_GAMMA_TILDE1)
            - expected(lapse * std::sqrt(4.0_rt / 3.0_rt))) < 1e-3_rt,
            "Sommerfeld longitudinal shift speed is wrong");
         require(std::abs(rhs(0, 0, 0, Z4C_GAMMA_TILDE2) - expected(lapse)) < 1e-3_rt,
            "Sommerfeld transverse shift speed is wrong");
         if (lapse_gauge == Z4cLapseGauge::Frozen) {
            require(rhs(0, 0, 0, Z4C_K_HAT) == 17.0_rt, "Sommerfeld overwrote frozen-lapse K RHS");
         } else {
            require(std::abs(rhs(0, 0, 0, Z4C_K_HAT)
               - expected(lapse * std::sqrt(z4c_lapse_speed_squared(local, lapse_gauge)))) < 1e-3_rt,
               "Sommerfeld lapse characteristic speed is wrong");
         }
         require(rhs(0, 0, 0, Z4C_CHI) == 17.0_rt && rhs(0, 0, 0, Z4C_LAPSE) == 17.0_rt
            && rhs(0, 0, 0, Z4C_SHIFT1) == 17.0_rt, "Sommerfeld replaced a bulk metric/gauge equation");
         rhs_storage.setVal(17.0_rt);
         amrex::Gpu::streamSynchronize();
         z4c_sommerfeld_rhs_at_point(input, rhs, 0, 0, 0, {{2.0_rt, 0.0_rt, 0.0_rt}},
            {{1.0_rt / dx, 1.0_rt / dx, 1.0_rt / dx}}, {}, lapse_gauge, Z4cShiftGauge::Frozen);
         require(rhs(0, 0, 0, Z4C_GAMMA_TILDE1) == 17.0_rt,
            "Sommerfeld overwrote frozen-shift Gamma RHS");
      }
   }
}

void characterize_stationary_kerr_boundary()
{
#if (AMREX_SPACEDIM == 3)
   // Characterize the finite-radius model error, not an accuracy acceptance
   // test: stationary Kerr is not an outgoing 1/r perturbation. Refining the
   // derivative cannot remove the continuum residual of a naive Sommerfeld BC.
   const amrex::Box point(amrex::IntVect(-1), amrex::IntVect(0));
   amrex::FArrayBox storage(point, Z4C_NUM_COMPONENTS, amrex::The_Managed_Arena());
   const auto rhs = storage.array();
   FixedSpacetimeParameters parameters;
   parameters.mass = 1.0_rt;
   parameters.dimensionless_spin3 = 0.5_rt;
   for (amrex::Real radius : {4.0_rt, 6.0_rt, 12.0_rt}) {
      amrex::Real first_residual = 0.0_rt;
      amrex::Real previous_corrected = 0.0_rt;
      for (amrex::Real dx : {0.125_rt, 0.0625_rt, 0.03125_rt}) {
         const auto input = [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k, int component) noexcept {
            const auto geometry = exact_kerr_schild_initial_data_geometry(
               parameters, radius + i * dx, j * dx, k * dx);
            const auto state = z4c_state_from_spacetime_geometry(geometry);
            // Use the catalog-owned component bridge, not another GF map.
            amrex::Real values[Z4C_NUM_COMPONENTS]{};
            const auto output = [&] AMREX_GPU_HOST_DEVICE(int, int, int, int n) -> amrex::Real& {
               return values[n];
            };
            store_z4c_state(output, 0, 0, 0, state);
            return values[component];
         };
         z4c_sommerfeld_rhs_at_point(input, rhs, 0, 0, 0, {{radius, 0.0_rt, 0.0_rt}},
            {{1.0_rt / dx, 1.0_rt / dx, 1.0_rt / dx}}, {},
            Z4cLapseGauge::Frozen, Z4cShiftGauge::Frozen);
         const amrex::Real residual = rhs(0, 0, 0, Z4C_A_TILDE11);
         require(std::isfinite(residual), "Kerr boundary-model residual is nonfinite");
         if (first_residual == 0.0_rt) first_residual = residual;
         require(std::abs(residual / first_residual - 1.0_rt) < 0.02_rt,
            "Kerr boundary-model probe is not in its resolved derivative regime");
         amrex::Print() << "stationary-Kerr Sommerfeld model: radius=" << radius
            << " dx=" << dx << " A11_rhs=" << residual << " (exact stationary RHS=0)\n";
         storage.setVal(0.0_rt); // Exact stationary bulk RHS at the donor.
         amrex::Gpu::streamSynchronize();
         z4c_radiative_rhs_at_point(input, rhs, 0, 0, 0, {{-1, 0, 0}},
            {{radius, 0.0_rt, 0.0_rt}}, {{1.0_rt / dx, 1.0_rt / dx, 1.0_rt / dx}}, {},
            Z4cLapseGauge::Frozen, Z4cShiftGauge::Frozen, 3);
         const amrex::Real corrected = std::abs(rhs(0, 0, 0, Z4C_A_TILDE11));
         require(corrected < std::abs(residual), "nonradiative correction increased stationary error");
         if (previous_corrected > 0.0_rt) require(previous_corrected / corrected > 1.8_rt,
            "stationary corrected boundary residual does not converge");
         previous_corrected = corrected;
         amrex::Print() << "stationary-Kerr corrected model: radius=" << radius
            << " dx=" << dx << " abs_A11_rhs=" << corrected << '\n';
      }
   }
#endif
}

void check_z4c_nonradiative_point()
{
   const amrex::Box box(amrex::IntVect(-2), amrex::IntVect(2));
   amrex::FArrayBox storage(box, Z4C_NUM_COMPONENTS, amrex::The_Managed_Arena());
   const auto rhs = storage.array();
   amrex::Real previous_error = 0.0_rt;
   for (amrex::Real dx : {0.1_rt, 0.05_rt, 0.025_rt}) {
      // Outgoing wave plus an independently stationary 1/r^2 field. The
      // non-wave residual scales as 1/r^3, so n=3 must preserve the wave while
      // removing the background forcing, without knowledge of either profile.
      const auto input = [=] AMREX_GPU_HOST_DEVICE(int i, int, int, int component) noexcept {
         if (component == Z4C_CHI || component == Z4C_LAPSE
             || component == Z4C_GAMMA_TILDE11 || component == Z4C_GAMMA_TILDE22
             || component == Z4C_GAMMA_TILDE33) return amrex::Real(1.0_rt);
         const amrex::Real radius = 2.0_rt + i * dx;
         if (component == Z4C_THETA) return std::sin(radius) / radius + 0.2_rt / (radius * radius);
         return amrex::Real(0.0_rt);
      };
      storage.setVal(0.0_rt);
      amrex::Gpu::streamSynchronize();
      rhs(-1, 0, 0, Z4C_THETA) = -std::cos(2.0_rt - dx) / (2.0_rt - dx);
      rhs(-1, 0, 0, Z4C_CHI) = 7.0_rt;
      rhs(0, 0, 0, Z4C_CHI) = 13.0_rt;
      rhs(-1, 0, 0, Z4C_K_HAT) = 19.0_rt;
      rhs(0, 0, 0, Z4C_K_HAT) = 23.0_rt;
      z4c_radiative_rhs_at_point(input, rhs, 0, 0, 0, {{-1, 0, 0}},
         {{2.0_rt, 0.0_rt, 0.0_rt}}, {{1.0_rt / dx, 1.0_rt / dx, 1.0_rt / dx}}, {},
         Z4cLapseGauge::Frozen, Z4cShiftGauge::Frozen, 3);
      const amrex::Real error = std::abs(rhs(0, 0, 0, Z4C_THETA) + std::cos(2.0_rt) / 2.0_rt);
      require(error < 1e-3_rt, "corrected boundary corrupted outgoing wave on stationary background");
      if (previous_error > 0.0_rt) require(previous_error / error > 3.8_rt,
         "corrected outgoing wave fails second-order consistency");
      previous_error = error;
      require(rhs(0, 0, 0, Z4C_CHI) == 13.0_rt && rhs(0, 0, 0, Z4C_K_HAT) == 23.0_rt,
         "nonradiative correction changed an unowned bulk/frozen-gauge component");
   }
}

void check_z4c_positive_centering_representation_scope()
{
   constexpr int source_ghosts = 3;
   constexpr int destination_ghosts = 1;
   const amrex::BoxArray cells = split_boxes(domain_box());
   const amrex::BoxArray nodes = amrex::convert(
      cells, amrex::IntVect::TheNodeVector()
   );
   const amrex::DistributionMapping distribution(cells);
   const amrex::IntVect exchanged_point(AMREX_D_DECL(7, 4, 4));
   const amrex::IntVect exterior_point(AMREX_D_DECL(-1, 4, 4));
   const auto sample = [](amrex::MultiFab const& field,
                          amrex::IntVect const& point,
                          int component) {
      return field.min(amrex::Box(point, point), component, destination_ghosts);
   };
   const auto initialize = [](amrex::MultiFab& field, amrex::Real chi_slope,
                               amrex::Real lapse_slope) {
      field.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
      for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
         const auto values = field.array(mfi);
         amrex::ParallelFor(
            mfi.validbox(), field.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               if (component == Z4C_CHI) {
                  values(i, j, k, component) = std::exp(chi_slope * i);
               } else if (component == Z4C_LAPSE) {
                  values(i, j, k, component) = std::exp(lapse_slope * i);
               } else {
                  values(i, j, k, component) = 2.0_rt + 0.05_rt * i;
               }
            }
         );
      }
   };

   // A physical target at a box interface must consume the exchanged source
   // ghost in the same log representation as every other physical point.
   {
      const amrex::Geometry geometry = nonperiodic_geometry();
      amrex::MultiFab source(nodes, distribution, Z4C_NUM_COMPONENTS, source_ghosts);
      amrex::MultiFab destination(cells, distribution, Z4C_NUM_COMPONENTS,
                                  destination_ghosts);
      initialize(source, 0.01_rt, -0.015_rt);
      source.FillBoundary(geometry.periodicity());

      // This deliberately signed exterior polynomial is not input repair:
      // only the nonphysical destination below may retain it.
      for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
         if (mfi.validbox().smallEnd(0) != 0) continue;
         amrex::Box slab = mfi.fabbox();
         slab.setSmall(0, -2);
         slab.setBig(0, 1);
         const auto values = source.array(mfi);
         amrex::ParallelFor(
            slab, source.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               const amrex::Real value = (i == -2 || i == 1) ? 1.0_rt : 0.01_rt;
               values(i, j, k, component) = value;
            }
         );
      }
      transfer_grid_field_centering<
         4, AMREX_SPACEDIM, Z4cPositiveScalarCenteringRepresentation
      >(destination, source, geometry, true);
      amrex::Gpu::streamSynchronize();

      require(close_enough(sample(destination, exchanged_point, Z4C_CHI),
         std::exp(0.01_rt * 7.5_rt)),
         "physical exchanged chi transfer did not interpolate log chi");
      require(close_enough(sample(destination, exchanged_point, Z4C_LAPSE),
         std::exp(-0.015_rt * 7.5_rt)),
         "physical exchanged lapse transfer did not interpolate log lapse");
      require(close_enough(sample(destination, exchanged_point, Z4C_K_HAT),
         2.0_rt + 0.05_rt * 7.5_rt),
         "positive-scalar representation changed an ordinary Z4c component");

      constexpr amrex::Real raw_signed_value = -0.11375_rt;
      require(close_enough(sample(destination, exterior_point, Z4C_CHI),
         raw_signed_value),
         "nonphysical exterior chi was log-transformed or repaired");
      require(close_enough(sample(destination, exterior_point, Z4C_LAPSE),
         raw_signed_value),
         "nonphysical exterior lapse was log-transformed or repaired");
      require(close_enough(sample(destination, exterior_point, Z4C_K_HAT),
         raw_signed_value),
         "nonphysical exterior ordinary component changed");
   }

   // Periodic destination ghosts are physical copies, unlike nonperiodic
   // extrapolation halos, and therefore retain the selected representation.
   {
      const amrex::RealBox physical_domain(
         AMREX_D_DECL(0.0_rt, 0.0_rt, 0.0_rt), AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt)
      );
      const amrex::Array<int, AMREX_SPACEDIM> periodic{
         AMREX_D_DECL(1, 0, 0)
      };
      const amrex::Geometry geometry(
         domain_box(), physical_domain, amrex::CoordSys::cartesian, periodic
      );
      amrex::MultiFab source(nodes, distribution, Z4C_NUM_COMPONENTS, source_ghosts);
      amrex::MultiFab destination(cells, distribution, Z4C_NUM_COMPONENTS,
                                  destination_ghosts);
      initialize(source, 0.04_rt, -0.03_rt);
      source.FillBoundary(geometry.periodicity());
      transfer_grid_field_centering<
         4, AMREX_SPACEDIM, Z4cPositiveScalarCenteringRepresentation
      >(destination, source, geometry, true);
      amrex::Gpu::streamSynchronize();

      constexpr std::array<amrex::Real, 4> weights{{
         -1.0_rt / 16.0_rt, 9.0_rt / 16.0_rt, 9.0_rt / 16.0_rt, -1.0_rt / 16.0_rt
      }};
      constexpr std::array<int, 4> wrapped_nodes{{14, 15, 0, 1}};
      const auto represented_value = [&](amrex::Real slope) {
         amrex::Real logarithm = 0.0_rt;
         for (int ordinal = 0; ordinal < 4; ++ordinal) {
            logarithm += weights[ordinal] * slope * wrapped_nodes[ordinal];
         }
         return std::exp(logarithm);
      };
      const auto raw_value = [&](amrex::Real slope) {
         amrex::Real value = 0.0_rt;
         for (int ordinal = 0; ordinal < 4; ++ordinal) {
            value += weights[ordinal] * std::exp(slope * wrapped_nodes[ordinal]);
         }
         return value;
      };
      const amrex::Real chi = sample(destination, exterior_point, Z4C_CHI);
      const amrex::Real lapse = sample(destination, exterior_point, Z4C_LAPSE);
      require(close_enough(chi, represented_value(0.04_rt))
         && std::abs(chi - raw_value(0.04_rt)) > 1e-5_rt,
         "periodic chi ghost did not retain the log representation");
      require(close_enough(lapse, represented_value(-0.03_rt))
         && std::abs(lapse - raw_value(-0.03_rt)) > 1e-5_rt,
         "periodic lapse ghost did not retain the log representation");
   }
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
      check_shifted_diode_kinematics();
      check_rmhd_diode_face<0>();
      check_rmhd_diode_face<1>();
      check_rmhd_diode_face<2>();
      check_relativistic_diode_boundary<PhysicalSystem::RHD>();
      check_relativistic_diode_boundary<PhysicalSystem::RMHD>();
      check_ct_face_boundary();
      check_ct_edge_sync();
      check_nodal_z4c_reflection_boundary();
      check_nodal_z4c_extrapolation();
      check_nodal_z4c_extrapolation(true);
      check_z4c_sommerfeld_point();
      characterize_stationary_kerr_boundary();
      check_z4c_nonradiative_point();
      check_z4c_positive_centering_representation_scope();
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
