#include "pyfluid_amrex/diffusion_timestep.hpp"

#include <AMReX.H>

#include "relativistic_vacuum_check.hpp"

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/characteristic_transforms.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/diagnostics.hpp"
#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/kernels_generated/z4c.hpp"
#include "pyfluid_amrex/kernels_generated/spacetime_geometry.hpp"
#include "pyfluid_amrex/kernels_generated/reference_metric_evolution.hpp"
#include "pyfluid_amrex/kernels_generated/fccz4.hpp"
#include "pyfluid_amrex/kernels_generated/coordinate_metrics.hpp"
#include "pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp"
#include "pyfluid_amrex/kernels_generated/patch_maps.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/apparent_horizon.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/eos/presets.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/initial_data/atmosphere.hpp"
#include "pyfluid_amrex/initial_data/kerr_schild.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/static_trumpet.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/reference_metric_valencia_pointwise.hpp"
#include "relativity/rhd_system.hpp"
#include "pyfluid_amrex/reconstruction/mp5.hpp"
#include "pyfluid_amrex/reconstruction/teno5.hpp"
#include "pyfluid_amrex/reconstruction/wenoz.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/spacetime_constraint_region.hpp"
#include "pyfluid_amrex/z4c_constraint_pointwise.hpp"
#include "pyfluid_amrex/z4c_evolution_pointwise.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "pyfluid_amrex/kernels_generated/boundary_differential_ops.hpp"

namespace {

using namespace amrex::literals;

struct QuadraticDerivativeGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int, int) const noexcept
   {
      return i * i + 2.0_rt * i * j + 3.0_rt * j;
   }
};

bool check_catalogued_derivative_dispatch()
{
   using namespace pyfluid_amrex;
   const QuadraticDerivativeGrid field{};
   for (int order = 2; order <= maximum_spatial_order; order += 2) {
      if (!is_supported_spatial_order(order)) continue;
      const bool passed = dispatch_spatial_order(order, [&](auto order_tag) {
         constexpr int degree = decltype(order_tag)::value;
         for (amrex::Real value : {0.0_rt, 1.0_rt, 0.123_rt, 1000.0_rt}) {
            const auto constant = [=](int, int, int, int) { return value; };
            if (generated::centered_first<degree, 0>(constant, 0, 0, 0, 0, 64.0_rt) != 0.0_rt
                || generated::centered_second<degree, 0>(constant, 0, 0, 0, 0, 4096.0_rt) != 0.0_rt
                || generated::centered_mixed<degree, 0, 1>(constant, 0, 0, 0, 0, 4096.0_rt) != 0.0_rt
                || generated::upwind_first<degree, 0>(constant, 0, 0, 0, 0, 0.7_rt, 64.0_rt) != 0.0_rt
                || generated::kreiss_oliger<degree, 0>(constant, 0, 0, 0, 0, 64.0_rt) != 0.0_rt
                || generated::one_sided_first<degree, 0>(constant, 0, 0, 0, 0, 1, 64.0_rt) != 0.0_rt
                || generated::one_sided_second<degree, 0>(constant, 0, 0, 0, 0, -1, 4096.0_rt) != 0.0_rt
                || generated::one_sided_mixed<degree, 0, 1>(constant, 0, 0, 0, 0, 1, -1, 4096.0_rt) != 0.0_rt) {
               return false;
            }
         }
         const amrex::Real first = generated::centered_first<degree, 0>(field, 2, 1, 0, 0, 1.0_rt);
         const amrex::Real second = generated::centered_second<degree, 0>(field, 2, 1, 0, 0, 1.0_rt);
         const amrex::Real mixed = generated::centered_mixed<degree, 0, 1>(field, 2, 1, 0, 0, 1.0_rt);
         const amrex::Real upwind = generated::upwind_first<degree, 0>(field, 2, 1, 0, 0, 0.7_rt, 1.0_rt);
         const amrex::Real ko = generated::kreiss_oliger<degree, 0>(field, 2, 1, 0, 0, 1.0_rt);
         const amrex::Real roundoff = amrex::max(
            1.0e-11_rt, 64.0_rt * 6.0_rt * std::numeric_limits<amrex::Real>::epsilon());
         return std::abs(first - 6.0_rt) < roundoff
            && std::abs(second - 2.0_rt) < roundoff
            && std::abs(mixed - 2.0_rt) < roundoff
            && std::abs(upwind - 6.0_rt) < roundoff && std::abs(ko) < roundoff;
      });
      if (!passed) {
         std::cerr << "generated derivative dispatch failed order=" << order << '\n';
         return false;
      }
   }
   return true;
}

struct PolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      const amrex::Real x = static_cast<amrex::Real>(i);
      const amrex::Real y = static_cast<amrex::Real>(j);
      const amrex::Real z = static_cast<amrex::Real>(k);
      return x * x * x * x + 2.0_rt * x * y + 3.0_rt * y * y + z + component;
   }
};

bool close_enough(amrex::Real actual, amrex::Real expected);

bool close_centering_transfer(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = std::max<amrex::Real>(1.0_rt, std::abs(expected));
   return std::isfinite(actual)
      && std::abs(actual - expected)
         <= 8192.0_rt * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

template <int Degree>
struct CenteringPolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   static amrex::Real power(amrex::Real value) noexcept
   {
      amrex::Real result = 1.0_rt;
      for (int exponent = 0; exponent < Degree; ++exponent) result *= value;
      return result;
   }

   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      return power(static_cast<amrex::Real>(i))
         + 2.0_rt * power(static_cast<amrex::Real>(j))
         + 3.0_rt * power(static_cast<amrex::Real>(k))
         + component;
   }
};

template <int Degree, int CoordinateScale>
struct ScaledCenteringPolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   static amrex::Real power(amrex::Real value) noexcept
   {
      return CenteringPolynomialGrid<Degree>::power(
         value / static_cast<amrex::Real>(CoordinateScale)
      );
   }

   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      return power(static_cast<amrex::Real>(i))
         + 2.0_rt * power(static_cast<amrex::Real>(j))
         + 3.0_rt * power(static_cast<amrex::Real>(k))
         + component;
   }
};

template <int Order, int Dim>
bool check_centering_transfer()
{
   const CenteringPolynomialGrid<Order - 1> field{};
   constexpr int i = 2;
   constexpr int j = -1;
   constexpr int k = 3;
   constexpr int component = 0;
   auto expected = [&](amrex::Real offset) {
      amrex::Real value = CenteringPolynomialGrid<Order - 1>::power(i + offset);
      if constexpr (Dim >= 2) {
         value += 2.0_rt * CenteringPolynomialGrid<Order - 1>::power(j + offset);
      } else {
         value += 2.0_rt * CenteringPolynomialGrid<Order - 1>::power(j);
      }
      if constexpr (Dim >= 3) {
         value += 3.0_rt * CenteringPolynomialGrid<Order - 1>::power(k + offset);
      } else {
         value += 3.0_rt * CenteringPolynomialGrid<Order - 1>::power(k);
      }
      return value;
   };
   const amrex::Real vertex_value = pyfluid_amrex::generated::cell_to_vertex<Order, Dim>(
      field, i, j, k, component
   );
   const amrex::Real cell_value = pyfluid_amrex::generated::vertex_to_cell<Order, Dim>(
      field, i, j, k, component
   );
   const amrex::Real expected_vertex = expected(-0.5_rt);
   const amrex::Real expected_cell = expected(0.5_rt);
   if (!close_centering_transfer(vertex_value, expected_vertex)
       || !close_centering_transfer(cell_value, expected_cell)) {
      std::cerr << "centering transfer order=" << Order << " dim=" << Dim
                << " vertex=" << vertex_value << "/" << expected_vertex
                << " cell=" << cell_value << "/" << expected_cell << '\n';
      return false;
   }
   return true;
}

template <int Degree, bool CellAverages>
struct AveragePolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int, int, int) const noexcept
   {
      const amrex::Real x = static_cast<amrex::Real>(i);
      if constexpr (!CellAverages) {
         return CenteringPolynomialGrid<Degree>::power(x);
      }
      const amrex::Real hi = CenteringPolynomialGrid<Degree + 1>::power(x + 0.5_rt);
      const amrex::Real lo = CenteringPolynomialGrid<Degree + 1>::power(x - 0.5_rt);
      return (hi - lo) / static_cast<amrex::Real>(Degree + 1);
   }
};

template <int Order>
bool check_point_average_conversion()
{
   constexpr int degree = Order - 1;
   const amrex::Real point_to_average =
      pyfluid_amrex::generated::point_to_cell_average_1d<Order, 0>(
         AveragePolynomialGrid<degree, false>{}, 0, 0, 0, 0
      );
   const amrex::Real expected_average = degree % 2 == 0
      ? 1.0_rt / (static_cast<amrex::Real>(degree + 1)
               * std::pow(2.0_rt, degree))
      : 0.0_rt;
   const amrex::Real average_to_point =
      pyfluid_amrex::generated::cell_average_to_point_1d<Order, 0>(
         AveragePolynomialGrid<degree, true>{}, 0, 0, 0, 0
      );
   return close_centering_transfer(point_to_average, expected_average)
      && close_centering_transfer(average_to_point, degree == 0 ? 1.0_rt : 0.0_rt);
}

template <int Order, int Dim>
bool check_topology_centering_transfer()
{
   const CenteringPolynomialGrid<Order - 1> field{};
   constexpr int i = 2;
   constexpr int j = -1;
   constexpr int k = 3;
   constexpr int component = 0;
   constexpr int num_centerings = 1 << Dim;
   for (int source_mask = 0; source_mask < num_centerings; ++source_mask) {
      for (int destination_mask = 0;
           destination_mask < num_centerings;
           ++destination_mask) {
         const auto offset = [=](int direction) {
            const bool source_is_nodal = (source_mask & (1 << direction)) != 0;
            const bool destination_is_nodal =
               (destination_mask & (1 << direction)) != 0;
            if (source_is_nodal == destination_is_nodal) return 0.0_rt;
            return source_is_nodal ? 0.5_rt : -0.5_rt;
         };
         amrex::Real expected = CenteringPolynomialGrid<Order - 1>::power(
            i + offset(0)
         );
         expected += 2.0_rt * CenteringPolynomialGrid<Order - 1>::power(
            j + (Dim >= 2 ? offset(1) : 0.0_rt)
         );
         expected += 3.0_rt * CenteringPolynomialGrid<Order - 1>::power(
            k + (Dim >= 3 ? offset(2) : 0.0_rt)
         );
         const amrex::Real actual =
            pyfluid_amrex::generated::topology_centering_transfer<Order, Dim>(
               field,
               i,
               j,
               k,
               component,
               source_mask,
               destination_mask
            );
         if (!close_centering_transfer(actual, expected)) {
            std::cerr << "topology centering transfer order=" << Order
                      << " dim=" << Dim << " source_mask=" << source_mask
                      << " destination_mask=" << destination_mask
                      << " actual=" << actual << " expected=" << expected
                      << '\n';
            return false;
         }
      }
   }
   return true;
}

template <int Order, int Dim>
bool check_bounded_topology_centering_transfer()
{
   constexpr int num_cells = Order + 4;
   const ScaledCenteringPolynomialGrid<Order - 1, num_cells> field{};
   constexpr int num_centerings = 1 << Dim;
   for (int source_mask = 0; source_mask < num_centerings; ++source_mask) {
      amrex::GpuArray<int, 3> source_lo{{0, 0, 0}};
      amrex::GpuArray<int, 3> source_hi{{0, 0, 0}};
      for (int direction = 0; direction < Dim; ++direction) {
         source_hi[direction] = num_cells - 1
            + ((source_mask & (1 << direction)) != 0 ? 1 : 0);
      }
      for (int destination_mask = 0;
           destination_mask < num_centerings;
           ++destination_mask) {
         for (int side = 0; side < 2; ++side) {
            int destination_index[3] = {0, 0, 0};
            for (int direction = 0; direction < Dim; ++direction) {
               destination_index[direction] = side == 0
                  ? 0
                  : num_cells - 1
                     + ((destination_mask & (1 << direction)) != 0 ? 1 : 0);
            }
            const auto target_index = [=](int direction) {
               const amrex::Real source_offset =
                  (source_mask & (1 << direction)) != 0 ? 0.0_rt : 0.5_rt;
               const amrex::Real destination_offset =
                  (destination_mask & (1 << direction)) != 0 ? 0.0_rt : 0.5_rt;
               return destination_index[direction]
                  + destination_offset - source_offset;
            };
            amrex::Real expected = ScaledCenteringPolynomialGrid<
               Order - 1, num_cells
            >::power(
               target_index(0)
            );
            expected += 2.0_rt * ScaledCenteringPolynomialGrid<
               Order - 1, num_cells
            >::power(
               Dim >= 2 ? target_index(1) : 0.0_rt
            );
            expected += 3.0_rt * ScaledCenteringPolynomialGrid<
               Order - 1, num_cells
            >::power(
               Dim >= 3 ? target_index(2) : 0.0_rt
            );
            const amrex::Real actual = pyfluid_amrex::generated::
               bounded_topology_centering_transfer<Order, Dim>(
                  field,
                  destination_index[0],
                  destination_index[1],
                  destination_index[2],
                  0,
                  source_mask,
                  destination_mask,
                  source_lo,
                  source_hi
               );
            if (!close_centering_transfer(actual, expected)) {
               std::cerr << "bounded topology centering transfer order="
                         << Order << " dim=" << Dim
                         << " source_mask=" << source_mask
                         << " destination_mask=" << destination_mask
                         << " side=" << side << std::setprecision(17)
                         << " actual=" << actual
                         << " expected=" << expected
                         << " difference=" << actual - expected << '\n';
               return false;
            }
         }
      }
   }
   return true;
}

struct ComponentGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int, int, int, int component) const noexcept
   {
      return 0.5_rt + static_cast<amrex::Real>(component);
   }
};

struct MinkowskiZ4cGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int, int, int, int component) const noexcept
   {
      return component == pyfluid_amrex::Z4C_CHI
            || component == pyfluid_amrex::Z4C_GAMMA_TILDE11
            || component == pyfluid_amrex::Z4C_GAMMA_TILDE22
            || component == pyfluid_amrex::Z4C_GAMMA_TILDE33
            || component == pyfluid_amrex::Z4C_LAPSE
         ? 1.0_rt
         : 0.0_rt;
   }
};

struct MutableComponentGrid {
   mutable std::array<amrex::Real, pyfluid_amrex::Z4C_NUM_COMPONENTS> values{};

   amrex::Real& operator()(int, int, int, int component) const noexcept
   {
      return values[component];
   }
};

pyfluid_amrex::HydroState hydro_flux_reference(
   pyfluid_amrex::HydroState const& state,
   amrex::Real pressure,
   int direction
)
{
   const amrex::Real momentum[3] = {state.mom1, state.mom2, state.mom3};
   const amrex::Real vn = momentum[direction] / state.rho;
   return {
      state.rho * vn,
      state.mom1 * vn + (direction == 0 ? pressure : 0.0_rt),
      state.mom2 * vn + (direction == 1 ? pressure : 0.0_rt),
      state.mom3 * vn + (direction == 2 ? pressure : 0.0_rt),
      (state.energy + pressure) * vn,
   };
}

pyfluid_amrex::MHDState mhd_flux_reference(
   pyfluid_amrex::MHDState const& state,
   amrex::Real pressure,
   int direction
)
{
   const pyfluid_amrex::StateVector3 velocity = {
      state.mom1 / state.rho,
      state.mom2 / state.rho,
      state.mom3 / state.rho,
   };
   const pyfluid_amrex::StateVector3 magnetic = {
      state.b1,
      state.b2,
      state.b3,
   };
   const amrex::Real vn = direction == 0 ? velocity.x : direction == 1 ? velocity.y : velocity.z;
   const amrex::Real bn = direction == 0 ? magnetic.x : direction == 1 ? magnetic.y : magnetic.z;
   const amrex::Real b2 = state.b1 * state.b1 + state.b2 * state.b2 + state.b3 * state.b3;
   const amrex::Real ptot = pressure + 0.5_rt * b2;
   const amrex::Real vdotb = (state.mom1 * state.b1 + state.mom2 * state.b2 + state.mom3 * state.b3) / state.rho;
   pyfluid_amrex::MHDState flux{
      state.rho * vn,
      state.mom1 * vn - state.b1 * bn,
      state.mom2 * vn - state.b2 * bn,
      state.mom3 * vn - state.b3 * bn,
      (state.energy + ptot) * vn - bn * vdotb,
      state.b1 * vn - bn * velocity.x,
      state.b2 * vn - bn * velocity.y,
      state.b3 * vn - bn * velocity.z,
   };
   if (direction == 0) {
      flux.mom1 += ptot;
      flux.b1 = 0.0_rt;
   } else if (direction == 1) {
      flux.mom2 += ptot;
      flux.b2 = 0.0_rt;
   } else {
      flux.mom3 += ptot;
      flux.b3 = 0.0_rt;
   }
   return flux;
}

bool close_enough(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = std::max<amrex::Real>(1.0_rt, std::abs(expected));
   return std::isfinite(actual) && std::abs(actual - expected) <= 8.0e-15_rt * scale;
}

bool validate_reference_metric_valencia_basis()
{
   const pyfluid_amrex::RHDPrimitive static_fluid{
      1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt
   };
   const amrex::GpuArray<amrex::Real, 3> location{{
      2.0_rt, 0.5_rt * pyfluid_amrex::constants::pi, 0.3_rt
   }};
   for (int coordinate : {
           pyfluid_amrex::MeshCartesian,
           pyfluid_amrex::MeshCylindrical,
           pyfluid_amrex::MeshSpherical,
        }) {
      const auto basis_divergence =
         pyfluid_amrex::rhd_reference_metric_momentum_basis_divergence(
            static_fluid,
            coordinate,
            pyfluid_amrex::mesh_default_axes_layout(coordinate),
            location
         );
      for (amrex::Real value : basis_divergence) {
         if (!close_enough(value, 0.0_rt)) {
            std::cerr << "reference-metric constant-pressure balance mismatch\n";
            return false;
         }
      }
   }

   const pyfluid_amrex::RHDPrimitive spherical_rotation{
      1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.0_rt, 0.2_rt
   };
   const auto spherical_axes = pyfluid_amrex::mesh_default_axes_layout(
      pyfluid_amrex::MeshSpherical
   );
   const auto nonpressure_flux =
      pyfluid_amrex::rhd_reference_metric_nonpressure_mixed_flux(
         spherical_rotation, pyfluid_amrex::MeshSpherical, spherical_axes
      );
   const auto spherical_basis =
      pyfluid_amrex::rhd_reference_metric_momentum_basis_divergence(
         spherical_rotation,
         pyfluid_amrex::MeshSpherical,
         spherical_axes,
         location
      );
   if (!close_enough(spherical_basis[0], -nonpressure_flux[8] / location[0])
       || !close_enough(spherical_basis[1], 0.0_rt)
       || !close_enough(spherical_basis[2], 0.0_rt)) {
      std::cerr << "reference-metric spherical rotating-stress mismatch\n";
      return false;
   }

   const pyfluid_amrex::RHDPrimitive cylindrical_rotation{
      1.0_rt, 1.5_rt, 1.0_rt, 0.0_rt, 0.2_rt, 0.0_rt
   };
   const auto cylindrical_axes = pyfluid_amrex::mesh_default_axes_layout(
      pyfluid_amrex::MeshCylindrical
   );
   const auto cylindrical_flux =
      pyfluid_amrex::rhd_reference_metric_nonpressure_mixed_flux(
         cylindrical_rotation, pyfluid_amrex::MeshCylindrical, cylindrical_axes
      );
   const auto cylindrical_basis =
      pyfluid_amrex::rhd_reference_metric_momentum_basis_divergence(
         cylindrical_rotation,
         pyfluid_amrex::MeshCylindrical,
         cylindrical_axes,
         location
      );
   if (!close_enough(cylindrical_basis[0], -cylindrical_flux[4] / location[0])
       || !close_enough(cylindrical_basis[1], 0.0_rt)
       || !close_enough(cylindrical_basis[2], 0.0_rt)) {
      std::cerr << "reference-metric cylindrical rotating-stress mismatch\n";
      return false;
   }
   return true;
}

bool validate_reference_metric_fixed_spacetime_geometry()
{
   using pyfluid_amrex::FixedSpacetimeMetric;
   using pyfluid_amrex::FixedSpacetimeParameters;

   FixedSpacetimeParameters rindler{};
   rindler.metric = FixedSpacetimeMetric::Rindler;
   rindler.acceleration1 = 0.04_rt;
   rindler.acceleration2 = -0.03_rt;
   rindler.acceleration3 = 0.02_rt;
   rindler.origin1 = -0.2_rt;
   rindler.origin2 = 0.1_rt;
   rindler.origin3 = -0.15_rt;

   FixedSpacetimeParameters schwarzschild{};
   schwarzschild.metric = FixedSpacetimeMetric::SchwarzschildIsotropic;
   schwarzschild.origin1 = -0.3_rt;
   schwarzschild.origin2 = 0.15_rt;
   schwarzschild.origin3 = -0.2_rt;
   schwarzschild.mass = 0.45_rt;

   FixedSpacetimeParameters trumpet{};
   trumpet.metric = FixedSpacetimeMetric::StaticTrumpet;
   trumpet.origin1 = -0.1_rt;
   trumpet.origin2 = 0.2_rt;
   trumpet.origin3 = -0.05_rt;
   trumpet.mass = 0.55_rt;
   trumpet.black_hole_interior = pyfluid_amrex::BlackHoleInterior::Puncture;

   const pyfluid_amrex::RHDPrimitive cartesian_primitive{
      1.1_rt, 0.35_rt, 0.18_rt, 0.13_rt, -0.08_rt, 0.05_rt
   };
   const auto tensor_close = [](amrex::Real actual, amrex::Real expected) {
      const amrex::Real scale = std::max<amrex::Real>(1.0_rt, std::abs(expected));
      return std::isfinite(actual)
         && std::abs(actual - expected) <= 5.0e-12_rt * scale;
   };

   for (int coordinate : {
           pyfluid_amrex::MeshCartesian,
           pyfluid_amrex::MeshCylindrical,
           pyfluid_amrex::MeshSpherical,
        }) {
      auto axes = pyfluid_amrex::mesh_default_axes_layout(coordinate);
      if (coordinate == pyfluid_amrex::MeshCartesian) {
#if (AMREX_SPACEDIM == 1)
         axes.axis[0] = pyfluid_amrex::MeshAxisY;
#elif (AMREX_SPACEDIM == 2)
         axes.axis[0] = pyfluid_amrex::MeshAxisX;
         axes.axis[1] = pyfluid_amrex::MeshAxisZ;
#endif
      }
      const amrex::GpuArray<amrex::Real, 3> q =
         coordinate == pyfluid_amrex::MeshCartesian
         ? amrex::GpuArray<amrex::Real, 3>{{1.5_rt, 0.8_rt, -0.35_rt}}
         : coordinate == pyfluid_amrex::MeshCylindrical
         ? amrex::GpuArray<amrex::Real, 3>{{2.1_rt, 0.73_rt, -0.35_rt}}
         : amrex::GpuArray<amrex::Real, 3>{{2.1_rt, 1.07_rt, 0.73_rt}};
      const auto cartesian_position =
         pyfluid_amrex::mesh_coordinate_triplet_to_cartesian(coordinate, q);
      const pyfluid_amrex::StateVector3 cartesian_velocity{
         cartesian_primitive.v1,
         cartesian_primitive.v2,
         cartesian_primitive.v3,
      };
      const auto mesh_velocity = pyfluid_amrex::cartesian_vector_to_mesh_basis(
         coordinate, axes, q, cartesian_velocity
      );
      const pyfluid_amrex::RHDPrimitive mesh_primitive{
         cartesian_primitive.rho,
         cartesian_primitive.eps,
         cartesian_primitive.pressure,
         mesh_velocity.x,
         mesh_velocity.y,
         mesh_velocity.z,
      };

      for (FixedSpacetimeParameters const& parameters : {
              rindler, schwarzschild, trumpet,
           }) {
         const auto cartesian_geometry = pyfluid_amrex::fixed_spacetime_geometry(
            parameters,
            cartesian_position[0],
            cartesian_position[1],
            cartesian_position[2],
            0.0_rt
         );
         const auto mesh_geometry =
            pyfluid_amrex::reference_metric_fixed_spacetime_geometry(
               parameters, coordinate, axes, q, 0.0_rt
            );

         const pyfluid_amrex::RHDConserved cartesian_flux[3]{
            pyfluid_amrex::generated::rhd_flux<0>(
               cartesian_primitive, cartesian_geometry.metric
            ),
            pyfluid_amrex::generated::rhd_flux<1>(
               cartesian_primitive, cartesian_geometry.metric
            ),
            pyfluid_amrex::generated::rhd_flux<2>(
               cartesian_primitive, cartesian_geometry.metric
            ),
         };
         const auto mesh_nonpressure_flux =
            pyfluid_amrex::rhd_reference_metric_nonpressure_mixed_flux(
               mesh_primitive, mesh_geometry.metric, coordinate, axes
            );
         const auto canonical_frame =
            pyfluid_amrex::reference_metric_orthonormal_frame(
               coordinate,
               pyfluid_amrex::mesh_default_axes_layout(coordinate),
               q
            );
         const amrex::Real pressure_flux =
            cartesian_geometry.metric.lapse
            * cartesian_geometry.metric.sqrt_det_gamma
            * cartesian_primitive.pressure;
         for (int component = 0; component < 3; ++component) {
            for (int normal = 0; normal < 3; ++normal) {
               amrex::Real expected_flux = 0.0_rt;
               for (int cartesian_component = 0; cartesian_component < 3;
                    ++cartesian_component) {
                  for (int cartesian_normal = 0; cartesian_normal < 3;
                       ++cartesian_normal) {
                     const auto& flux = cartesian_flux[cartesian_normal];
                     const amrex::Real momentum = cartesian_component == 0
                        ? flux.mom1
                        : (cartesian_component == 1 ? flux.mom2 : flux.mom3);
                     expected_flux +=
                        canonical_frame.mesh_to_cartesian[component][cartesian_component]
                        * canonical_frame.mesh_to_cartesian[normal][cartesian_normal]
                        * momentum;
                  }
               }
               if (component == normal) {
                  expected_flux -= pressure_flux;
               }
               if (!tensor_close(
                      mesh_nonpressure_flux[3 * component + normal],
                      expected_flux
                   )) {
                  std::cerr
                     << "reference-metric nonpressure-flux rotation mismatch\n";
                  return false;
               }
            }
         }

         const amrex::Real gamma[3][3]{
            {
               mesh_geometry.metric.gamma11,
               mesh_geometry.metric.gamma12,
               mesh_geometry.metric.gamma13,
            },
            {
               mesh_geometry.metric.gamma12,
               mesh_geometry.metric.gamma22,
               mesh_geometry.metric.gamma23,
            },
            {
               mesh_geometry.metric.gamma13,
               mesh_geometry.metric.gamma23,
               mesh_geometry.metric.gamma33,
            },
         };
         const amrex::Real inverse_gamma[3][3]{
            {
               mesh_geometry.metric.inv_gamma11,
               mesh_geometry.metric.inv_gamma12,
               mesh_geometry.metric.inv_gamma13,
            },
            {
               mesh_geometry.metric.inv_gamma12,
               mesh_geometry.metric.inv_gamma22,
               mesh_geometry.metric.inv_gamma23,
            },
            {
               mesh_geometry.metric.inv_gamma13,
               mesh_geometry.metric.inv_gamma23,
               mesh_geometry.metric.inv_gamma33,
            },
         };
         for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
               amrex::Real product = 0.0_rt;
               for (int c = 0; c < 3; ++c) {
                  product += gamma[a][c] * inverse_gamma[c][b];
               }
               if (!tensor_close(product, a == b ? 1.0_rt : 0.0_rt)) {
                  std::cerr << "reference-metric inverse identity mismatch\n";
                  return false;
               }
            }
         }

         const auto cartesian_conserved =
            pyfluid_amrex::generated::rhd_primitive_to_conserved(
               cartesian_primitive, cartesian_geometry.metric
            );
         const auto mesh_conserved =
            pyfluid_amrex::generated::rhd_primitive_to_conserved(
               mesh_primitive, mesh_geometry.metric
            );
         const auto expected_momentum =
            pyfluid_amrex::cartesian_vector_to_mesh_basis(
               coordinate,
               axes,
               q,
               {
                  cartesian_conserved.mom1,
                  cartesian_conserved.mom2,
                  cartesian_conserved.mom3,
               }
            );
         if (!tensor_close(mesh_conserved.dens, cartesian_conserved.dens)
             || !tensor_close(mesh_conserved.mom1, expected_momentum.x)
             || !tensor_close(mesh_conserved.mom2, expected_momentum.y)
             || !tensor_close(mesh_conserved.mom3, expected_momentum.z)
             || !tensor_close(mesh_conserved.tau, cartesian_conserved.tau)) {
            std::cerr << "reference-metric conserved-state rotation mismatch\n";
            return false;
         }

         const auto cartesian_source =
            pyfluid_amrex::generated::rhd_geometric_source(
               cartesian_primitive,
               cartesian_geometry.metric,
               cartesian_geometry.derivatives
            );
         const auto mesh_source = pyfluid_amrex::generated::rhd_geometric_source(
            mesh_primitive,
            mesh_geometry.metric,
            mesh_geometry.reference_derivatives
         );
         const auto expected_source_momentum =
            pyfluid_amrex::cartesian_vector_to_mesh_basis(
               coordinate,
               axes,
               q,
               {
                  cartesian_source.mom1,
                  cartesian_source.mom2,
                  cartesian_source.mom3,
               }
            );
         if (!tensor_close(mesh_source.dens, cartesian_source.dens)
             || !tensor_close(mesh_source.mom1, expected_source_momentum.x)
             || !tensor_close(mesh_source.mom2, expected_source_momentum.y)
             || !tensor_close(mesh_source.mom3, expected_source_momentum.z)
             || !tensor_close(mesh_source.tau, cartesian_source.tau)) {
            std::cerr << "reference-metric geometric-source rotation mismatch\n";
            return false;
         }
      }
   }
   return true;
}

bool has_stationary_adm_identity(
   pyfluid_amrex::FixedSpacetimeGeometry const& geometry
)
{
   const auto& metric = geometry.metric;
   const auto& derivatives = geometry.derivatives;
   const amrex::Real spatial_metric[3][3]{
      {metric.gamma11, metric.gamma12, metric.gamma13},
      {metric.gamma12, metric.gamma22, metric.gamma23},
      {metric.gamma13, metric.gamma23, metric.gamma33},
   };
   const amrex::Real spatial_metric_derivative[3][3][3]{
      {
         {derivatives.d1_gamma11, derivatives.d1_gamma12, derivatives.d1_gamma13},
         {derivatives.d1_gamma12, derivatives.d1_gamma22, derivatives.d1_gamma23},
         {derivatives.d1_gamma13, derivatives.d1_gamma23, derivatives.d1_gamma33},
      },
      {
         {derivatives.d2_gamma11, derivatives.d2_gamma12, derivatives.d2_gamma13},
         {derivatives.d2_gamma12, derivatives.d2_gamma22, derivatives.d2_gamma23},
         {derivatives.d2_gamma13, derivatives.d2_gamma23, derivatives.d2_gamma33},
      },
      {
         {derivatives.d3_gamma11, derivatives.d3_gamma12, derivatives.d3_gamma13},
         {derivatives.d3_gamma12, derivatives.d3_gamma22, derivatives.d3_gamma23},
         {derivatives.d3_gamma13, derivatives.d3_gamma23, derivatives.d3_gamma33},
      },
   };
   const amrex::Real shift[3]{metric.shift1, metric.shift2, metric.shift3};
   const amrex::Real shift_derivative[3][3]{
      {derivatives.d1_shift1, derivatives.d1_shift2, derivatives.d1_shift3},
      {derivatives.d2_shift1, derivatives.d2_shift2, derivatives.d2_shift3},
      {derivatives.d3_shift1, derivatives.d3_shift2, derivatives.d3_shift3},
   };
   const amrex::Real extrinsic_curvature[3][3]{
      {derivatives.extrinsic11, derivatives.extrinsic12, derivatives.extrinsic13},
      {derivatives.extrinsic12, derivatives.extrinsic22, derivatives.extrinsic23},
      {derivatives.extrinsic13, derivatives.extrinsic23, derivatives.extrinsic33},
   };
   // Independent Lie-derivative form: no inverse/Christoffel implementation
   // shared with the provider's covariant-shift construction of K_ij.
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         amrex::Real residual = -2.0_rt * metric.lapse * extrinsic_curvature[first][second];
         for (int direction = 0; direction < 3; ++direction) {
            residual += shift[direction] * spatial_metric_derivative[direction][first][second]
               + spatial_metric[first][direction] * shift_derivative[second][direction]
               + spatial_metric[second][direction] * shift_derivative[first][direction];
         }
         if (!std::isfinite(residual) || std::abs(residual) > 5.0e-13_rt) {
            return false;
         }
      }
   }
   return true;
}

bool validate_positive_spacetime_transfer()
{
   using namespace pyfluid_amrex;
   using Representation = Z4cPositiveScalarCenteringRepresentation;
   const auto steep = [](int i, int, int, int) -> amrex::Real {
      return i == 0 || i == 1 ? 0.01_rt : 1.0_rt;
   };
   const CenteringTransferSource<Representation, decltype(steep)> encoded{steep};
   const amrex::Real raw = generated::topology_centering_transfer<4, 3>(
      steep, 0, 0, 0, Z4C_LAPSE, 7, 0);
   const amrex::Real expected = std::pow(0.01_rt, 9.0_rt / 8.0_rt);
   for (int component : {Z4C_LAPSE, Z4C_CHI}) {
      const amrex::Real value = Representation::decode(component,
         generated::topology_centering_transfer<4, 3>(encoded, 0, 0, 0, component, 7, 0));
      const amrex::Real face = Z4cCellToFaceReader<4, 0, decltype(steep)>{
         steep, 1, 0, 0}(component);
      const amrex::Real bounded = Z4cExteriorCellToFaceReader<4, 0, decltype(steep)>{
         steep, 1, 0, 0, {{-1, 0, 0}}, {{2, 0, 0}}}(component);
      if (!(raw < 0.0_rt && value > 0.0_rt)
          || !close_enough(value, expected) || !close_enough(face, expected)
          || !close_enough(bounded, expected)
          || std::isfinite(Representation::encode(component, -1.0_rt))
          || std::isfinite(Representation::encode(component, 0.0_rt))) {
         std::cerr << "positive-scalar transfer/face contract failed\n";
         return false;
      }
   }
   const amrex::Real untransformed = Representation::decode(Z4C_K_HAT,
      generated::topology_centering_transfer<4, 3>(encoded, 0, 0, 0, Z4C_K_HAT, 7, 0));
   if (untransformed != raw) {
      std::cerr << "positive-scalar representation changed another component\n";
      return false;
   }
   const auto cubic_log = [](int i, int, int, int) -> amrex::Real {
      return std::exp(0.2_rt * i + 0.01_rt * i * i * i);
   };
   for (int first : {-3, 1}) {
      const amrex::Real face = Z4cExteriorCellToFaceReader<4, 0, decltype(cubic_log)>{
         cubic_log, 1, 0, 0, {{first, 0, 0}}, {{first + 3, 0, 0}}}(Z4C_LAPSE);
      if (!close_enough(face, std::exp(0.2_rt * 0.5_rt + 0.01_rt * 0.125_rt))) {
         std::cerr << "one-sided logarithmic face transfer lost cubic exactness\n";
         return false;
      }
   }
   // A quartic logarithm gives a nonzero order-4 error. Rotate the varying
   // axis, including inactive-axis limits of the tensor-product stencil.
   for (int axis = 0; axis < 3; ++axis) {
      amrex::Real previous = 0.0_rt;
      for (amrex::Real spacing : {0.2_rt, 0.1_rt, 0.05_rt}) {
         const auto smooth = [=](int i, int j, int k, int) -> amrex::Real {
            const int index[3]{i, j, k};
            const amrex::Real x = 0.3_rt + spacing * index[axis];
            return std::exp(0.2_rt * x + 0.1_rt * x * x * x * x);
         };
         const CenteringTransferSource<Representation, decltype(smooth)> source{smooth};
         const amrex::Real x = 0.3_rt + 0.5_rt * spacing;
         const amrex::Real exact = std::exp(0.2_rt * x + 0.1_rt * x * x * x * x);
         const amrex::Real actual = Representation::decode(Z4C_LAPSE,
            generated::topology_centering_transfer<4, 3>(source, 0, 0, 0, Z4C_LAPSE, 7, 0));
         const amrex::Real error = std::abs(actual - exact);
         if (!(error > 0.0_rt && std::isfinite(error))
             || (previous > 0.0_rt && !(previous / error > 15.0_rt && previous / error < 17.0_rt))) {
            std::cerr << "positive-scalar transfer lost fourth-order convergence\n";
            return false;
         }
         previous = error;
      }
   }
   return true;
}

bool validate_initial_atmosphere_profiles()
{
   pyfluid_amrex::EOSParams ideal_gas{};
   ideal_gas.gamma = 5.0_rt / 3.0_rt;
   ideal_gas.rho_floor = 1.0e-8_rt;
   ideal_gas.pressure_floor = 1.0e-12_rt;
   pyfluid_amrex::InitialAtmosphereParameters atmosphere{
      pyfluid_amrex::InitialAtmosphereProfile::Flat,
      1.0e-4_rt,
      1.0e-6_rt,
      1.0_rt,
      4.0_rt,
      ideal_gas,
   };
   pyfluid_amrex::InitialDataSamplePoint point{};
   point.physical = {4.0_rt, 0.0_rt, 0.0_rt};
   const auto flat = pyfluid_amrex::initial_atmosphere_state<
      pyfluid_amrex::EOSIdealGas
   >(atmosphere, point);
   if (!close_enough(flat.rho, atmosphere.density)
       || !close_enough(flat.pressure, atmosphere.pressure)) {
      std::cerr << "flat initial atmosphere mismatch\n";
      return false;
   }

   atmosphere.profile = pyfluid_amrex::InitialAtmosphereProfile::PowerLaw;
   const auto graded = pyfluid_amrex::initial_atmosphere_state<
      pyfluid_amrex::EOSIdealGas
   >(atmosphere, point);
   const amrex::Real expected_density = atmosphere.density / 256.0_rt;
   const amrex::Real expected_pressure = atmosphere.pressure * std::pow(
      expected_density / atmosphere.density, ideal_gas.gamma
   );
   if (!close_enough(graded.rho, expected_density)
       || !close_enough(graded.pressure, expected_pressure)) {
      std::cerr << "power-law initial atmosphere mismatch\n";
      return false;
   }

   point.physical = {20.0_rt, 0.0_rt, 0.0_rt};
   const auto endpoint = pyfluid_amrex::initial_atmosphere_state<
      pyfluid_amrex::EOSIdealGas
   >(atmosphere, point);
   const auto recovery_floor = pyfluid_amrex::eos_project_to_floors<
      pyfluid_amrex::EOSIdealGas
   >(ideal_gas.rho_floor, ideal_gas.pressure_floor, ideal_gas);
   if (!close_enough(endpoint.rho, recovery_floor.rho)
       || !close_enough(endpoint.pressure, recovery_floor.pressure)
       || !close_enough(
          endpoint.specific_internal_energy,
          recovery_floor.specific_internal_energy
       )) {
      std::cerr << "initial atmosphere recovery-floor endpoint mismatch\n";
      return false;
   }

   pyfluid_amrex::EOSParams polytrope = ideal_gas;
   polytrope.segment_k[0] = 2.0_rt;
   polytrope.segment_gamma[0] = 2.0_rt;
   atmosphere.eos = polytrope;
   atmosphere.density = 1.0e-3_rt;
   atmosphere.pressure = pyfluid_amrex::eos_cold_pressure<
      pyfluid_amrex::EOSPolytrope
   >(atmosphere.density, polytrope);
   point.physical = {2.0_rt, 0.0_rt, 0.0_rt};
   const auto cold = pyfluid_amrex::initial_atmosphere_state<
      pyfluid_amrex::EOSPolytrope
   >(atmosphere, point);
   return close_enough(
      cold.pressure,
      pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(
         cold.rho, polytrope
      )
   );
}

AMREX_GPU_HOST_DEVICE
std::array<amrex::Real, pyfluid_amrex::Z4C_NUM_COMPONENTS> z4c_state_components(
   pyfluid_amrex::Z4cState const& state
)
{
   return {
      state.chi,
      state.gamma_tilde11,
      state.gamma_tilde12,
      state.gamma_tilde13,
      state.gamma_tilde22,
      state.gamma_tilde23,
      state.gamma_tilde33,
      state.k_hat,
      state.a_tilde11,
      state.a_tilde12,
      state.a_tilde13,
      state.a_tilde22,
      state.a_tilde23,
      state.a_tilde33,
      state.gamma_tilde1,
      state.gamma_tilde2,
      state.gamma_tilde3,
      state.theta,
      state.lapse,
      state.shift1,
      state.shift2,
      state.shift3,
   };
}

struct StaticTrumpetZ4cGrid {
   pyfluid_amrex::FixedSpacetimeParameters parameters{};
   amrex::Real spacing = 1.0_rt;
   std::array<amrex::Real, 3> evaluation_point{};

   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      const pyfluid_amrex::Z4cState state =
         pyfluid_amrex::static_trumpet_z4c_initial_data_state(
            parameters,
            evaluation_point[0] + spacing * static_cast<amrex::Real>(i),
            evaluation_point[1] + spacing * static_cast<amrex::Real>(j),
            evaluation_point[2] + spacing * static_cast<amrex::Real>(k)
         );
      return z4c_state_components(state)[component];
   }
};

amrex::Real static_trumpet_rhs_norm(amrex::Real spacing)
{
   pyfluid_amrex::FixedSpacetimeParameters parameters{};
   parameters.metric = pyfluid_amrex::FixedSpacetimeMetric::StaticTrumpet;
   parameters.mass = 0.8_rt;
   parameters.black_hole_interior =
      pyfluid_amrex::BlackHoleInterior::Puncture;
   const StaticTrumpetZ4cGrid grid{
      parameters,
      spacing,
      {2.3_rt, -1.1_rt, 0.7_rt},
   };
   const std::array<amrex::Real, 3> inverse_spacing{
      1.0_rt / spacing,
      1.0_rt / spacing,
      1.0_rt / spacing,
   };
   const pyfluid_amrex::Z4cRhs rhs =
      pyfluid_amrex::evaluate_vacuum_z4c_rhs_at<4, 3>(
         grid,
         0,
         0,
         0,
         inverse_spacing,
         0.0_rt,
         pyfluid_amrex::Z4cEvolutionParameters{},
         pyfluid_amrex::Z4cLapseGauge::Trumpet,
         pyfluid_amrex::Z4cShiftGauge::Frozen
      );
   const amrex::Real values[]{
      rhs.geometry.chi,
      rhs.geometry.gamma_tilde11,
      rhs.geometry.gamma_tilde12,
      rhs.geometry.gamma_tilde13,
      rhs.geometry.gamma_tilde22,
      rhs.geometry.gamma_tilde23,
      rhs.geometry.gamma_tilde33,
      rhs.geometry.k_hat,
      rhs.geometry.a_tilde11,
      rhs.geometry.a_tilde12,
      rhs.geometry.a_tilde13,
      rhs.geometry.a_tilde22,
      rhs.geometry.a_tilde23,
      rhs.geometry.a_tilde33,
      rhs.geometry.gamma_tilde1,
      rhs.geometry.gamma_tilde2,
      rhs.geometry.gamma_tilde3,
      rhs.geometry.theta,
      rhs.lapse,
      rhs.shift.shift1,
      rhs.shift.shift2,
      rhs.shift.shift3,
   };
   amrex::Real maximum = 0.0_rt;
   for (amrex::Real value : values) {
      maximum = std::max(maximum, std::abs(value));
   }
   return maximum;
}

bool validate_static_trumpet_stationarity()
{
   if constexpr (AMREX_SPACEDIM != 3) {
      return true;
   }
   const amrex::Real coarse = static_trumpet_rhs_norm(0.04_rt);
   const amrex::Real fine = static_trumpet_rhs_norm(0.02_rt);
   if (!std::isfinite(coarse) || !std::isfinite(fine)
       || !(fine < coarse / 8.0_rt) || !(fine < 2.0e-6_rt)) {
      std::cerr << "static-trumpet stationary RHS mismatch: coarse="
                << coarse << " fine=" << fine << '\n';
      return false;
   }
   return true;
}

bool validate_z4c_pointwise_bridge()
{
   const auto loaded = pyfluid_amrex::load_z4c_state(ComponentGrid{}, 3, -2, 1);
   const auto loaded_components = z4c_state_components(loaded);
   for (int component = 0; component < pyfluid_amrex::Z4C_NUM_COMPONENTS; ++component) {
      if (!close_enough(loaded_components[component], 0.5_rt + component)) {
         std::cerr << "Z4c pointwise state mapping mismatch at component "
                   << component << '\n';
         return false;
      }
   }

   constexpr int i = 2;
   constexpr int j = -1;
   constexpr int k = 3;
   const std::array<amrex::Real, 3> inv_spacing = {1.0_rt, 1.0_rt, 1.0_rt};
   pyfluid_amrex::Z4cState advection_state{};
   advection_state.shift1 = 0.7_rt;
   advection_state.shift2 = -0.4_rt;
   advection_state.shift3 = 0.2_rt;
   const auto derivatives =
      pyfluid_amrex::gather_z4c_spatial_derivatives<4, AMREX_SPACEDIM>(
         PolynomialGrid{}, i, j, k, inv_spacing, advection_state
      );
   const auto grid_constraints =
      pyfluid_amrex::evaluate_vacuum_z4c_constraints_at<4, AMREX_SPACEDIM>(
         MinkowskiZ4cGrid{}, i, j, k, inv_spacing
      );
   const amrex::Real flat_constraints[] = {
      grid_constraints.theta,
      grid_constraints.z_tilde1,
      grid_constraints.z_tilde2,
      grid_constraints.z_tilde3,
      grid_constraints.hamiltonian,
      grid_constraints.momentum1,
      grid_constraints.momentum2,
      grid_constraints.momentum3,
      grid_constraints.log_conformal_determinant,
      grid_constraints.trace_a_tilde,
   };
   for (const amrex::Real value : flat_constraints) {
      if (!close_enough(value, 0.0_rt)) {
         std::cerr << "Z4c grid-level Minkowski constraint mismatch\n";
         return false;
      }
   }
   const auto flat_rhs = pyfluid_amrex::evaluate_vacuum_z4c_rhs_at<
      4,
      AMREX_SPACEDIM
   >(
      MinkowskiZ4cGrid{},
      i,
      j,
      k,
      inv_spacing,
      0.25_rt,
      pyfluid_amrex::Z4cEvolutionParameters{},
      pyfluid_amrex::Z4cLapseGauge::OnePlusLog,
      pyfluid_amrex::Z4cShiftGauge::GammaDriver
   );
   const amrex::Real flat_rhs_values[] = {
      flat_rhs.geometry.chi,
      flat_rhs.geometry.gamma_tilde11,
      flat_rhs.geometry.gamma_tilde12,
      flat_rhs.geometry.gamma_tilde13,
      flat_rhs.geometry.gamma_tilde22,
      flat_rhs.geometry.gamma_tilde23,
      flat_rhs.geometry.gamma_tilde33,
      flat_rhs.geometry.k_hat,
      flat_rhs.geometry.a_tilde11,
      flat_rhs.geometry.a_tilde12,
      flat_rhs.geometry.a_tilde13,
      flat_rhs.geometry.a_tilde22,
      flat_rhs.geometry.a_tilde23,
      flat_rhs.geometry.a_tilde33,
      flat_rhs.geometry.gamma_tilde1,
      flat_rhs.geometry.gamma_tilde2,
      flat_rhs.geometry.gamma_tilde3,
      flat_rhs.geometry.theta,
      flat_rhs.lapse,
      flat_rhs.shift.shift1,
      flat_rhs.shift.shift2,
      flat_rhs.shift.shift3,
   };
   for (const amrex::Real value : flat_rhs_values) {
      if (!close_enough(value, 0.0_rt)) {
         std::cerr << "Z4c grid-level Minkowski RHS mismatch\n";
         return false;
      }
   }
   const amrex::Real expected_first[3] = {
      4.0_rt * i * i * i + 2.0_rt * j,
      2.0_rt * i + 6.0_rt * j,
      1.0_rt,
   };
   const amrex::Real expected_second[3] = {12.0_rt * i * i, 6.0_rt, 0.0_rt};
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real expected = direction < AMREX_SPACEDIM
         ? expected_first[direction]
         : 0.0_rt;
      const amrex::Real expected_dd = direction < AMREX_SPACEDIM
         ? expected_second[direction]
         : 0.0_rt;
      const amrex::Real first_values[] = {
         derivatives.d_chi[direction],
         derivatives.advection_chi[direction],
         derivatives.d_k_hat[direction],
         derivatives.advection_k_hat[direction],
         derivatives.d_theta[direction],
         derivatives.advection_theta[direction],
         derivatives.d_lapse[direction],
         derivatives.advection_lapse[direction],
      };
      for (const amrex::Real value : first_values) {
         if (std::abs(value - expected) > 2.0e-12_rt) {
            std::cerr << "Z4c pointwise scalar derivative mismatch in direction "
                      << direction << '\n';
            return false;
         }
      }
      if (std::abs(derivatives.dd_chi[direction][direction] - expected_dd) > 2.0e-12_rt
          || std::abs(derivatives.dd_lapse[direction][direction] - expected_dd)
             > 2.0e-12_rt) {
         std::cerr << "Z4c pointwise scalar second derivative mismatch in direction "
                   << direction << '\n';
         return false;
      }
      for (int component = 0; component < 6; ++component) {
         const amrex::Real tensor_values[] = {
            derivatives.d_gamma_tilde[direction][component],
            derivatives.advection_gamma_tilde[direction][component],
            derivatives.d_a_tilde[direction][component],
            derivatives.advection_a_tilde[direction][component],
         };
         for (const amrex::Real value : tensor_values) {
            if (std::abs(value - expected) > 2.0e-12_rt) {
               std::cerr << "Z4c pointwise tensor derivative mismatch\n";
               return false;
            }
         }
         if (std::abs(
                derivatives.dd_gamma_tilde[direction][direction][component]
                - expected_dd
             ) > 2.0e-12_rt) {
            std::cerr << "Z4c pointwise tensor second derivative mismatch\n";
            return false;
         }
      }
      for (int component = 0; component < 3; ++component) {
         const amrex::Real vector_values[] = {
            derivatives.d_gamma_tilde_vector[direction][component],
            derivatives.advection_gamma_tilde_vector[direction][component],
            derivatives.d_shift[direction][component],
            derivatives.advection_shift[direction][component],
         };
         for (const amrex::Real value : vector_values) {
            if (std::abs(value - expected) > 2.0e-12_rt) {
               std::cerr << "Z4c pointwise vector derivative mismatch\n";
               return false;
            }
         }
         if (std::abs(
                derivatives.dd_shift[direction][direction][component] - expected_dd
             ) > 2.0e-12_rt) {
            std::cerr << "Z4c pointwise vector second derivative mismatch\n";
            return false;
         }
      }
   }

   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         if (first == second) {
            continue;
         }
         const amrex::Real expected = first < AMREX_SPACEDIM
               && second < AMREX_SPACEDIM
            ? ((first == 0 && second == 1) || (first == 1 && second == 0)
                  ? 2.0_rt
                  : 0.0_rt)
            : 0.0_rt;
         if (std::abs(derivatives.dd_chi[first][second] - expected) > 2.0e-12_rt
             || std::abs(derivatives.dd_lapse[first][second] - expected)
                > 2.0e-12_rt) {
            std::cerr << "Z4c pointwise mixed scalar derivative mismatch\n";
            return false;
         }
         for (int component = 0; component < 6; ++component) {
            if (std::abs(
                   derivatives.dd_gamma_tilde[first][second][component] - expected
                ) > 2.0e-12_rt) {
               std::cerr << "Z4c pointwise mixed tensor derivative mismatch\n";
               return false;
            }
         }
         for (int component = 0; component < 3; ++component) {
            if (std::abs(derivatives.dd_shift[first][second][component] - expected)
                > 2.0e-12_rt) {
               std::cerr << "Z4c pointwise mixed vector derivative mismatch\n";
               return false;
            }
         }
      }
   }

   pyfluid_amrex::Z4cGeometricRhs geometric_rhs{};
   geometric_rhs.chi = 1.0_rt;
   geometric_rhs.gamma_tilde11 = 2.0_rt;
   geometric_rhs.gamma_tilde12 = 3.0_rt;
   geometric_rhs.gamma_tilde13 = 4.0_rt;
   geometric_rhs.gamma_tilde22 = 5.0_rt;
   geometric_rhs.gamma_tilde23 = 6.0_rt;
   geometric_rhs.gamma_tilde33 = 7.0_rt;
   geometric_rhs.k_hat = 8.0_rt;
   geometric_rhs.a_tilde11 = 9.0_rt;
   geometric_rhs.a_tilde12 = 10.0_rt;
   geometric_rhs.a_tilde13 = 11.0_rt;
   geometric_rhs.a_tilde22 = 12.0_rt;
   geometric_rhs.a_tilde23 = 13.0_rt;
   geometric_rhs.a_tilde33 = 14.0_rt;
   geometric_rhs.gamma_tilde1 = 15.0_rt;
   geometric_rhs.gamma_tilde2 = 16.0_rt;
   geometric_rhs.gamma_tilde3 = 17.0_rt;
   geometric_rhs.theta = 18.0_rt;
   const pyfluid_amrex::Z4cRhs z4c_rhs{
      geometric_rhs,
      19.0_rt,
      pyfluid_amrex::Z4cShiftRhs{20.0_rt, 21.0_rt, 22.0_rt},
   };
   MutableComponentGrid stored{};
   pyfluid_amrex::store_z4c_rhs(stored, 0, 0, 0, z4c_rhs);
   for (int component = 0; component < pyfluid_amrex::Z4C_NUM_COMPONENTS; ++component) {
      if (!close_enough(stored.values[component], component + 1.0_rt)) {
         std::cerr << "Z4c pointwise RHS mapping mismatch at component "
                   << component << '\n';
         return false;
      }
   }
   return true;
}

bool compare_hydro(pyfluid_amrex::HydroState const& actual, pyfluid_amrex::HydroState const& expected)
{
   return close_enough(actual.rho, expected.rho)
      && close_enough(actual.mom1, expected.mom1)
      && close_enough(actual.mom2, expected.mom2)
      && close_enough(actual.mom3, expected.mom3)
      && close_enough(actual.energy, expected.energy);
}

bool compare_mhd(pyfluid_amrex::MHDState const& actual, pyfluid_amrex::MHDState const& expected)
{
   return close_enough(actual.rho, expected.rho)
      && close_enough(actual.mom1, expected.mom1)
      && close_enough(actual.mom2, expected.mom2)
      && close_enough(actual.mom3, expected.mom3)
      && close_enough(actual.energy, expected.energy)
      && close_enough(actual.b1, expected.b1)
      && close_enough(actual.b2, expected.b2)
      && close_enough(actual.b3, expected.b3);
}

template <std::size_t N>
amrex::Real scaled_max_error(
   std::array<amrex::Real, N> const& actual,
   std::array<amrex::Real, N> const& expected
)
{
   amrex::Real error = 0.0_rt;
   for (std::size_t component = 0; component < N; ++component) {
      const amrex::Real scale = std::max({
         amrex::Real{1.0_rt},
         std::abs(actual[component]),
         std::abs(expected[component]),
      });
      error = std::max(error, std::abs(actual[component] - expected[component]) / scale);
   }
   return error;
}

std::array<amrex::Real, 5> components(pyfluid_amrex::HydroPrimitive const& state)
{
   return {state.rho, state.v1, state.v2, state.v3, state.pressure};
}

std::array<amrex::Real, 5> components(pyfluid_amrex::HydroState const& state)
{
   return {state.rho, state.mom1, state.mom2, state.mom3, state.energy};
}

std::array<amrex::Real, 8> components(pyfluid_amrex::MHDPrimitive const& state)
{
   return {state.rho, state.v1, state.v2, state.v3, state.pressure, state.b1, state.b2, state.b3};
}

std::array<amrex::Real, 8> components(pyfluid_amrex::MHDState const& state)
{
   return {state.rho, state.mom1, state.mom2, state.mom3, state.energy, state.b1, state.b2, state.b3};
}

std::array<amrex::Real, 6> components(pyfluid_amrex::RHDPrimitive const& state)
{
   return {state.rho, state.eps, state.pressure, state.v1, state.v2, state.v3};
}

std::array<amrex::Real, 5> components(pyfluid_amrex::RHDConserved const& state)
{
   return {state.dens, state.mom1, state.mom2, state.mom3, state.tau};
}

std::array<amrex::Real, 9> components(pyfluid_amrex::RMHDPrimitive const& state)
{
   return {
      state.rho,
      state.eps,
      state.pressure,
      state.v1,
      state.v2,
      state.v3,
      state.b1,
      state.b2,
      state.b3,
   };
}

std::array<amrex::Real, 8> components(pyfluid_amrex::RMHDConserved const& state)
{
   return {
      state.dens,
      state.mom1,
      state.mom2,
      state.mom3,
      state.tau,
      state.b1,
      state.b2,
      state.b3,
   };
}

template <int EOSMethod>
bool check_newtonian_round_trip(
   pyfluid_amrex::HydroPrimitive primitive,
   pyfluid_amrex::MHDPrimitive mhd_primitive,
   pyfluid_amrex::EOSParams const& eos
)
{
   if constexpr (EOSMethod == pyfluid_amrex::EOSPolytrope) {
      primitive.pressure = pyfluid_amrex::eos_cold_pressure<EOSMethod>(primitive.rho, eos);
      mhd_primitive.pressure = pyfluid_amrex::eos_cold_pressure<EOSMethod>(mhd_primitive.rho, eos);
   }

   const auto conserved = pyfluid_amrex::conserved_from_primitive<EOSMethod>(primitive, eos);
   const auto recovered_primitive = pyfluid_amrex::primitive_from_conserved<EOSMethod>(conserved, eos);
   const auto conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(recovered_primitive, eos);

   const auto mhd_conserved = pyfluid_amrex::conserved_from_primitive<EOSMethod>(mhd_primitive, eos);
   const auto recovered_mhd_primitive =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(mhd_conserved, eos);
   const auto mhd_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(recovered_mhd_primitive, eos);

   auto independent_conserved = conserved;
   auto independent_mhd_conserved = mhd_conserved;
   if constexpr (EOSMethod != pyfluid_amrex::EOSPolytrope) {
      independent_conserved.mom1 *= -0.81_rt;
      independent_conserved.mom2 += 0.23_rt * independent_conserved.rho;
      independent_conserved.energy += 0.37_rt * std::max(amrex::Real{1.0_rt}, independent_conserved.rho);
      independent_mhd_conserved.mom1 *= -0.81_rt;
      independent_mhd_conserved.mom2 += 0.23_rt * independent_mhd_conserved.rho;
      independent_mhd_conserved.energy += 0.37_rt * std::max(amrex::Real{1.0_rt}, independent_mhd_conserved.rho);
   }
   const auto primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_conserved, eos);
   const auto independent_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(primitive_from_independent, eos);
   const auto mhd_primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_mhd_conserved, eos);
   const auto independent_mhd_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(mhd_primitive_from_independent, eos);

   constexpr amrex::Real tolerance = 256.0_rt * std::numeric_limits<amrex::Real>::epsilon();
   const std::array<amrex::Real, 6> errors{
      scaled_max_error(components(recovered_primitive), components(primitive)),
      scaled_max_error(components(conserved_round_trip), components(conserved)),
      scaled_max_error(components(recovered_mhd_primitive), components(mhd_primitive)),
      scaled_max_error(components(mhd_conserved_round_trip), components(mhd_conserved)),
      scaled_max_error(components(independent_conserved_round_trip), components(independent_conserved)),
      scaled_max_error(components(independent_mhd_conserved_round_trip), components(independent_mhd_conserved)),
   };
   if (*std::max_element(errors.begin(), errors.end()) > tolerance) {
      std::cerr << "Newtonian round-trip residuals EOS=" << EOSMethod;
      for (amrex::Real const error : errors) {
         std::cerr << ' ' << error;
      }
      std::cerr << '\n';
      return false;
   }
   return true;
}

bool validate_newtonian_round_trips()
{
   pyfluid_amrex::EOSParams eos{};
   eos.gamma = 1.4_rt;
   eos.segment_k[0] = 0.73_rt;
   eos.segment_gamma[0] = 1.65_rt;
   eos.thermal_gamma = 1.37_rt;
   const std::array<pyfluid_amrex::HydroPrimitive, 5> hydro_corpus{{
      {0.071_rt, 2.3_rt, -0.47_rt, 0.19_rt, 0.013_rt},
      {0.82_rt, -1.7_rt, 0.93_rt, -0.31_rt, 4.6_rt},
      {1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.41_rt},
      {7.4_rt, 0.017_rt, -3.2_rt, 1.1_rt, 13.0_rt},
      {19.0_rt, -0.004_rt, 0.007_rt, -0.009_rt, 0.26_rt},
   }};
   const std::array<pyfluid_amrex::MHDPrimitive, 5> mhd_corpus{{
      {0.071_rt, 2.3_rt, -0.47_rt, 0.19_rt, 0.013_rt, 4.0_rt, -0.07_rt, 0.8_rt},
      {0.82_rt, -1.7_rt, 0.93_rt, -0.31_rt, 4.6_rt, -2.1_rt, 3.7_rt, -0.4_rt},
      {1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.41_rt, 0.0_rt, 0.0_rt, 0.0_rt},
      {7.4_rt, 0.017_rt, -3.2_rt, 1.1_rt, 13.0_rt, 0.003_rt, -8.0_rt, 2.5_rt},
      {19.0_rt, -0.004_rt, 0.007_rt, -0.009_rt, 0.26_rt, 15.0_rt, 0.2_rt, -0.5_rt},
   }};
   for (std::size_t sample = 0; sample < hydro_corpus.size(); ++sample) {
      if (!check_newtonian_round_trip<pyfluid_amrex::EOSIdealGas>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )
          || !check_newtonian_round_trip<pyfluid_amrex::EOSPolytrope>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )
          || !check_newtonian_round_trip<pyfluid_amrex::EOSHybridPolytrope>(
             hydro_corpus[sample], mhd_corpus[sample], eos
          )) {
         std::cerr << "Newtonian primitive/conserved round trip failed for sample=" << sample << "\n";
         return false;
      }
   }
   return true;
}

amrex::Real scaled_component_error(
   amrex::Real actual,
   amrex::Real expected,
   amrex::Real scale_floor
)
{
   if (!std::isfinite(actual)) {
      return std::numeric_limits<amrex::Real>::infinity();
   }
   const amrex::Real scale = std::max(
      {std::abs(actual), std::abs(expected), scale_floor}
   );
   return std::abs(actual - expected) / scale;
}

struct ProjectedKerrZ4cNodeGrid {
   pyfluid_amrex::FixedSpacetimeParameters parameters;
   amrex::Real spacing;

   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      const auto geometry =
         pyfluid_amrex::kerr_schild_initial_data_geometry(
            parameters, spacing * i, spacing * j, spacing * k
         );
      const auto state =
         pyfluid_amrex::generated::z4c_project_algebraic_constraints(
            pyfluid_amrex::z4c_state_from_spacetime_geometry(geometry)
         );
      const std::array<amrex::Real, pyfluid_amrex::Z4C_NUM_COMPONENTS>
         components{{
            state.chi,
            state.gamma_tilde11,
            state.gamma_tilde12,
            state.gamma_tilde13,
            state.gamma_tilde22,
            state.gamma_tilde23,
            state.gamma_tilde33,
            state.k_hat,
            state.a_tilde11,
            state.a_tilde12,
            state.a_tilde13,
            state.a_tilde22,
            state.a_tilde23,
            state.a_tilde33,
            state.gamma_tilde1,
            state.gamma_tilde2,
            state.gamma_tilde3,
            state.theta,
            state.lapse,
            state.shift1,
            state.shift2,
            state.shift3,
         }};
      return components[static_cast<std::size_t>(component)];
   }
};

struct CellCenteredKerrZ4cGrid {
   ProjectedKerrZ4cNodeGrid nodes;

   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      using Representation = pyfluid_amrex::Z4cPositiveScalarCenteringRepresentation;
      const pyfluid_amrex::CenteringTransferSource<Representation, ProjectedKerrZ4cNodeGrid>
         source{nodes};
      return Representation::decode(component,
         pyfluid_amrex::generated::topology_centering_transfer<4, 3>(
            source, i, j, k, component, 7, 0));
   }
};

template <int EOSMethod>
bool validate_rhd_c2p_round_trip_for_eos(
   pyfluid_amrex::EOSParams const& eos
)
{
   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.87_rt;
   curved_metric.shift1 = 0.05_rt;
   curved_metric.shift2 = -0.02_rt;
   curved_metric.shift3 = 0.01_rt;
   curved_metric.gamma11 = 1.44_rt;
   curved_metric.gamma22 = 1.21_rt;
   curved_metric.gamma33 = 0.81_rt;
   curved_metric.inv_gamma11 = 1.0_rt / curved_metric.gamma11;
   curved_metric.inv_gamma22 = 1.0_rt / curved_metric.gamma22;
   curved_metric.inv_gamma33 = 1.0_rt / curved_metric.gamma33;
   curved_metric.sqrt_det_gamma = 1.188_rt;

   pyfluid_amrex::FixedSpacetimeParameters kerr_schild{};
   kerr_schild.metric = pyfluid_amrex::FixedSpacetimeMetric::KerrSchild;
   kerr_schild.mass = 1.0_rt;
   kerr_schild.dimensionless_spin3 = 0.5_rt;
   kerr_schild.black_hole_interior =
      pyfluid_amrex::BlackHoleInterior::Filled;
   kerr_schild.black_hole_fill_radius = 1.0_rt;
   const auto kerr_inner =
      pyfluid_amrex::kerr_schild_initial_data_geometry(
         kerr_schild, 0.25_rt, 0.25_rt, 0.25_rt
      ).metric;
   const auto kerr_transition =
      pyfluid_amrex::kerr_schild_initial_data_geometry(
         kerr_schild, 0.75_rt, 0.25_rt, 0.25_rt
      ).metric;
   const CellCenteredKerrZ4cGrid centered_kerr{{kerr_schild, 0.5_rt}};
   const std::array<pyfluid_amrex::SpacetimeMetric, 2> metrics{{
      pyfluid_amrex::minkowski_spacetime_metric(),
      curved_metric,
   }};
   constexpr std::array<amrex::Real, 6> densities{{
      1.0e-12_rt, 1.0e-10_rt, 1.0e-8_rt, 1.0e-5_rt, 1.0e-2_rt, 1.0_rt,
   }};
   constexpr std::array<std::array<amrex::Real, 3>, 4> velocities{{
      {{0.0_rt, 0.0_rt, 0.0_rt}},
      {{3.0e-7_rt, -2.0e-7_rt, 1.0e-7_rt}},
      {{0.21_rt, -0.08_rt, 0.04_rt}},
      {{0.75_rt, 0.0_rt, 0.0_rt}},
   }};
   constexpr std::array<amrex::Real, 3> thermal_specific_energies{{
      1.0e-6_rt, 1.0e-3_rt, 1.0e-1_rt,
   }};
   const int thermal_samples = EOSMethod == pyfluid_amrex::EOSPolytrope
      ? 1
      : static_cast<int>(thermal_specific_energies.size());

   amrex::Real worst_primitive_error = 0.0_rt;
   amrex::Real worst_conserved_error = 0.0_rt;
   for (std::size_t metric_index = 0; metric_index < metrics.size(); ++metric_index) {
      auto const& metric = metrics[metric_index];
      for (amrex::Real const rho : densities) {
         const amrex::Real cold_specific_energy =
            pyfluid_amrex::eos_cold_energy_density<EOSMethod>(rho, eos) / rho;
         for (int thermal_index = 0;
              thermal_index < thermal_samples;
              ++thermal_index) {
            const amrex::Real thermal_specific_energy =
               EOSMethod == pyfluid_amrex::EOSPolytrope
                  ? 0.0_rt
                  : thermal_specific_energies[thermal_index];
            const amrex::Real eps = cold_specific_energy
               + thermal_specific_energy;
            const amrex::Real pressure =
               pyfluid_amrex::relativistic_pressure<EOSMethod>(rho, eps, eos);
            for (auto const& velocity : velocities) {
               const pyfluid_amrex::RHDPrimitive primitive{
                  rho,
                  eps,
                  pressure,
                  velocity[0],
                  velocity[1],
                  velocity[2],
               };
               const auto conserved =
                  pyfluid_amrex::generated::rhd_primitive_to_conserved(
                     primitive, metric
                  );
               const auto recovered = pyfluid_amrex::recover_rhd<EOSMethod>(
                  conserved, metric, eos
               );
               if (!recovered.attempt.succeeded()) {
                  std::cerr
                     << "RHD C2P(P2C(P)) failed EOS=" << EOSMethod
                     << " metric=" << metric_index
                     << " rho=" << rho
                     << " eps=" << eps
                     << " v1=" << velocity[0]
                     << " status=" << static_cast<int>(recovered.attempt.status)
                     << " residual=" << recovered.attempt.residual << '\n';
                  return false;
               }

               const auto conserved_round_trip =
                  pyfluid_amrex::generated::rhd_primitive_to_conserved(
                     recovered.primitive, metric
                  );
               const std::array<amrex::Real, 6> primitive_errors{{
                  scaled_component_error(recovered.primitive.rho, rho, eos.rho_floor),
                  scaled_component_error(recovered.primitive.eps, eps, 1.0e-12_rt),
                  scaled_component_error(
                     recovered.primitive.pressure, pressure, eos.pressure_floor
                  ),
                  scaled_component_error(recovered.primitive.v1, velocity[0], 1.0_rt),
                  scaled_component_error(recovered.primitive.v2, velocity[1], 1.0_rt),
                  scaled_component_error(recovered.primitive.v3, velocity[2], 1.0_rt),
               }};
               const auto expected_conserved = components(conserved);
               const auto actual_conserved = components(conserved_round_trip);
               amrex::Real conserved_error = 0.0_rt;
               for (std::size_t component = 0;
                    component < expected_conserved.size();
                    ++component) {
                  conserved_error = std::max(
                     conserved_error,
                     scaled_component_error(
                        actual_conserved[component],
                        expected_conserved[component],
                        eos.rho_floor * 1.0e-6_rt
                     )
                  );
               }
               const amrex::Real primitive_error = *std::max_element(
                  primitive_errors.begin(), primitive_errors.end()
               );
               worst_primitive_error = std::max(
                  worst_primitive_error, primitive_error
               );
               worst_conserved_error = std::max(
                  worst_conserved_error, conserved_error
               );
               if (primitive_error > 1.0e-5_rt || conserved_error > 1.0e-7_rt) {
                  std::cerr
                     << "RHD round-trip mismatch EOS=" << EOSMethod
                     << " metric=" << metric_index
                     << " rho=" << rho
                     << " eps=" << eps
                     << " v1=" << velocity[0]
                     << " primitive_error=" << primitive_error
                     << " conserved_error=" << conserved_error << '\n';
                  return false;
               }
            }
         }
      }
   }

   std::vector<pyfluid_amrex::SpacetimeMetric> kerr_metrics{
      kerr_inner,
      kerr_transition,
   };
   for (int k = -2; k <= 1; ++k) {
      for (int j = -2; j <= 1; ++j) {
         for (int i = -2; i <= 1; ++i) {
            kerr_metrics.push_back(
               pyfluid_amrex::generated::z4c_spacetime_metric(
                  pyfluid_amrex::load_z4c_state(centered_kerr, i, j, k)
               )
            );
         }
      }
   }
   for (std::size_t metric_index = 0;
        metric_index < kerr_metrics.size();
        ++metric_index) {
      const amrex::Real rho = 1.0e-4_rt;
      const amrex::Real eps = 1.0e-2_rt;
      const amrex::Real pressure =
         pyfluid_amrex::relativistic_pressure<EOSMethod>(rho, eps, eos);
      const pyfluid_amrex::RHDPrimitive primitive{
         rho, eps, pressure, 0.0_rt, 0.0_rt, 0.0_rt,
      };
      const auto conserved =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            primitive, kerr_metrics[metric_index]
         );
      const auto recovered = pyfluid_amrex::recover_rhd<EOSMethod>(
         conserved, kerr_metrics[metric_index], eos
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "static Kerr RHD C2P failed EOS=" << EOSMethod
                   << " metric=" << metric_index << '\n';
         return false;
      }
      const auto regenerated =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            recovered.primitive, kerr_metrics[metric_index]
         );
      const auto expected_components = components(conserved);
      const auto actual_components = components(regenerated);
      amrex::Real conserved_error = 0.0_rt;
      for (std::size_t component = 0;
           component < expected_components.size();
           ++component) {
         conserved_error = std::max(
            conserved_error,
            scaled_component_error(
               actual_components[component],
               expected_components[component],
               eos.rho_floor * 1.0e-6_rt
            )
         );
      }
      if (conserved_error > 1.0e-7_rt) {
         std::cerr << "static Kerr RHD round-trip mismatch EOS="
                   << EOSMethod << " metric=" << metric_index
                   << " conserved_error=" << conserved_error << '\n';
         return false;
      }
   }

   const std::array<pyfluid_amrex::RHDConserved, 3> invalid_states{{
      {-1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 1.0_rt},
      {1.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, -1.0_rt},
      {std::numeric_limits<amrex::Real>::quiet_NaN(), 0.0_rt, 0.0_rt, 0.0_rt, 1.0_rt},
   }};
   for (auto const& invalid_state : invalid_states) {
      const auto invalid_recovery = pyfluid_amrex::recover_rhd<EOSMethod>(
         invalid_state, metrics[0], eos
      );
      if (invalid_recovery.attempt.status
          != pyfluid_amrex::C2PStatus::InvalidConservedState) {
         std::cerr << "invalid RHD conserved state was not rejected EOS="
                   << EOSMethod << " status="
                   << static_cast<int>(invalid_recovery.attempt.status) << '\n';
         return false;
      }
   }

   std::cout << "RHD C2P matrix EOS=" << EOSMethod
             << " primitive_error=" << worst_primitive_error
             << " conserved_error=" << worst_conserved_error << '\n';
   return true;
}

bool validate_rhd_c2p_round_trip_matrix()
{
   pyfluid_amrex::EOSParams eos{};
   eos.gamma = 2.0_rt;
   eos.thermal_gamma = 1.8_rt;
   eos.rho_floor = 1.0e-14_rt;
   eos.pressure_floor = 1.0e-28_rt;
   eos.segment_k[0] = 100.0_rt;
   eos.segment_gamma[0] = 2.0_rt;
   if (!validate_rhd_c2p_round_trip_for_eos<pyfluid_amrex::EOSIdealGas>(eos)) {
      return false;
   }
   eos.gamma = 5.0_rt / 3.0_rt;
   if (!validate_rhd_c2p_round_trip_for_eos<pyfluid_amrex::EOSIdealGas>(eos)) {
      return false;
   }
   eos.gamma = 2.0_rt;
   if (!(validate_rhd_c2p_round_trip_for_eos<pyfluid_amrex::EOSPolytrope>(eos)
         && validate_rhd_c2p_round_trip_for_eos<
            pyfluid_amrex::EOSHybridPolytrope
         >(eos))) {
      return false;
   }
   for (int method : {
           pyfluid_amrex::EOSIdealGas,
           pyfluid_amrex::EOSPolytrope,
           pyfluid_amrex::EOSHybridPolytrope,
        }) {
      eos.method = method;
      if (!validate_rhd_c2p_round_trip_for_eos<pyfluid_amrex::EOSRuntime>(eos)) {
         return false;
      }
   }
   return true;
}

bool validate_piecewise_polytrope()
{
   pyfluid_amrex::EOSParams eos{};
   eos.gamma = 1.4_rt;
   eos.thermal_gamma = 1.6_rt;
   eos.polytrope_segments = 2;
   eos.segment_k[0] = 0.5_rt;
   eos.segment_gamma[0] = 2.0_rt;
   eos.segment_density_break[0] = 2.0_rt;
   eos.segment_gamma[1] = 1.5_rt;
   eos.segment_k[1] = eos.segment_k[0] * std::pow(
      eos.segment_density_break[0],
      eos.segment_gamma[0] - eos.segment_gamma[1]
   );
   eos.segment_energy_offset[1] =
      eos.segment_k[0] * std::pow(
         eos.segment_density_break[0], eos.segment_gamma[0] - 1.0_rt
      ) / (eos.segment_gamma[0] - 1.0_rt)
      - eos.segment_k[1] * std::pow(
         eos.segment_density_break[0], eos.segment_gamma[1] - 1.0_rt
      ) / (eos.segment_gamma[1] - 1.0_rt);

   const amrex::Real below = eos.segment_density_break[0] * (1.0_rt - 1.0e-10_rt);
   const amrex::Real above = eos.segment_density_break[0] * (1.0_rt + 1.0e-10_rt);
   const amrex::Real pressure_below =
      pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(below, eos);
   const amrex::Real pressure_above =
      pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(above, eos);
   const amrex::Real energy_below =
      pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSPolytrope>(below, eos);
   const amrex::Real energy_above =
      pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSPolytrope>(above, eos);
   if (std::abs(pressure_below - pressure_above)
          > 1.0e-9_rt * std::max(amrex::Real{1.0_rt}, std::abs(pressure_below))
       || std::abs(energy_below - energy_above)
          > 1.0e-9_rt * std::max(amrex::Real{1.0_rt}, std::abs(energy_below))) {
      std::cerr << "piecewise-polytrope continuity mismatch\n";
      return false;
   }

   const amrex::Real ideal_rho = 1.7_rt;
   const amrex::Real ideal_pressure = 0.43_rt;
   const amrex::Real expected_ideal_entropy = ideal_pressure / std::pow(
      ideal_rho, eos.gamma
   );
   if (!close_enough(
          pyfluid_amrex::eos_thermal_entropy_proxy<
             pyfluid_amrex::EOSIdealGas
          >(ideal_rho, ideal_pressure, eos),
          expected_ideal_entropy
       )
       || pyfluid_amrex::eos_thermal_entropy_proxy<
             pyfluid_amrex::EOSPolytrope
          >(ideal_rho, ideal_pressure, eos) != 0.0_rt) {
      std::cerr << "ideal/cold thermal-entropy proxy mismatch\n";
      return false;
   }

   for (amrex::Real const rho : {0.5_rt, 1.5_rt, 2.5_rt, 8.0_rt}) {
      const amrex::Real cold_energy =
         pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSHybridPolytrope>(rho, eos);
      const amrex::Real thermal_energy = 0.17_rt * rho;
      const amrex::Real pressure =
         pyfluid_amrex::eos_pressure_from_density_energy<pyfluid_amrex::EOSHybridPolytrope>(
            rho, cold_energy + thermal_energy, eos
         );
      const amrex::Real expected_thermal_entropy =
         (eos.thermal_gamma - 1.0_rt) * thermal_energy
         / std::pow(rho, eos.thermal_gamma);
      if (!close_enough(
             pyfluid_amrex::eos_thermal_entropy_proxy<
                pyfluid_amrex::EOSHybridPolytrope
             >(rho, pressure, eos),
             expected_thermal_entropy
          )) {
         std::cerr << "piecewise hybrid thermal-entropy proxy mismatch\n";
         return false;
      }
      const amrex::Real recovered_energy =
         pyfluid_amrex::eos_internal_energy_density_from_pressure<pyfluid_amrex::EOSHybridPolytrope>(
            rho, pressure, eos
         );
      if (std::abs(recovered_energy - cold_energy - thermal_energy)
          > 1.0e-13_rt * std::max(
             amrex::Real{1.0_rt}, std::abs(cold_energy + thermal_energy)
          )) {
         std::cerr << "piecewise hybrid-EOS pressure/energy inverse mismatch\n";
         return false;
      }

      const pyfluid_amrex::RHDPrimitive primitive{
         rho,
         (cold_energy + thermal_energy) / rho,
         pressure,
         0.21_rt,
         -0.08_rt,
         0.04_rt,
      };
      const pyfluid_amrex::RHDConserved conserved =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            primitive, pyfluid_amrex::minkowski_spacetime_metric()
         );
      const auto recovered = pyfluid_amrex::recover_rhd<
         pyfluid_amrex::EOSHybridPolytrope
      >(conserved, pyfluid_amrex::minkowski_spacetime_metric(), eos);
      const auto c2p_close = [](amrex::Real actual, amrex::Real expected) {
         return std::isfinite(actual)
            && std::abs(actual - expected)
               <= 2.0e-10_rt * std::max(amrex::Real{1.0_rt}, std::abs(expected));
      };
      if (recovered.attempt.status != pyfluid_amrex::C2PStatus::Success
          || !c2p_close(recovered.primitive.rho, primitive.rho)
          || !c2p_close(recovered.primitive.eps, primitive.eps)
          || !c2p_close(recovered.primitive.pressure, primitive.pressure)
          || !c2p_close(recovered.primitive.v1, primitive.v1)
          || !c2p_close(recovered.primitive.v2, primitive.v2)
          || !c2p_close(recovered.primitive.v3, primitive.v3)) {
         std::cerr << "piecewise hybrid-EOS RHD round-trip mismatch\n";
         return false;
      }
   }
   return true;
}

bool validate_read_piecewise_polytropes()
{
   using pyfluid_amrex::eos::PiecewisePolytropePreset;
   constexpr std::array<PiecewisePolytropePreset, 4> presets{{
      PiecewisePolytropePreset::sly,
      PiecewisePolytropePreset::apr4,
      PiecewisePolytropePreset::h4,
      PiecewisePolytropePreset::ms1,
   }};
   // Independent code-unit anchors for Read et al. Table III at
   // rho = 10^14.7 g/cm^3; catch overflow or loss of the CGS conversion scale.
   constexpr amrex::Real reference_density = 0.0008115303644041098_rt;
   constexpr std::array<amrex::Real, 4> reference_pressures{{
      4.3617760979015695e-5_rt, 3.347059002748516e-5_rt,
      8.407432094465957e-5_rt, 1.2991621769091265e-4_rt,
   }};
   const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
   const amrex::Real continuity_tolerance = std::max(2.0e-13_rt, 32.0_rt * epsilon);
   const amrex::Real anchor_tolerance = std::max(2.0e-13_rt, 128.0_rt * epsilon);
   for (std::size_t index = 0; index < presets.size(); ++index) {
      const PiecewisePolytropePreset preset = presets[index];
      const pyfluid_amrex::EOSParams eos =
         pyfluid_amrex::eos::make_read_piecewise_polytrope(
            preset, 1.8_rt, 1.0e-12_rt, 1.0e-14_rt
         );
      if (eos.polytrope_segments != 7) {
         std::cerr << "Read piecewise-polytrope preset does not have seven segments\n";
         return false;
      }
      const amrex::Real reference_pressure =
         pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(
            reference_density, eos);
      if (!std::isfinite(reference_pressure)
          || std::abs(reference_pressure - reference_pressures[index])
             > anchor_tolerance * reference_pressures[index]) {
         std::cerr << "Read piecewise-polytrope pressure anchor mismatch\n";
         return false;
      }
      for (int boundary = 0; boundary < eos.polytrope_segments - 1; ++boundary) {
         const amrex::Real rho = eos.segment_density_break[boundary];
         if (!(rho > 0.0_rt)
             || (boundary > 0
                 && !(rho > eos.segment_density_break[boundary - 1]))) {
            std::cerr << "Read piecewise-polytrope density breaks are not ordered\n";
            return false;
         }
         const amrex::Real lower_pressure = eos.segment_k[boundary]
            * std::pow(rho, eos.segment_gamma[boundary]);
         const amrex::Real upper_pressure = eos.segment_k[boundary + 1]
            * std::pow(rho, eos.segment_gamma[boundary + 1]);
         const amrex::Real lower_energy =
            eos.segment_energy_offset[boundary] * rho
            + lower_pressure / (eos.segment_gamma[boundary] - 1.0_rt);
         const amrex::Real upper_energy =
            eos.segment_energy_offset[boundary + 1] * rho
            + upper_pressure / (eos.segment_gamma[boundary + 1] - 1.0_rt);
         const amrex::Real pressure_scale = std::max(
            amrex::Real(1.0e-30_rt), std::abs(lower_pressure)
         );
         const amrex::Real energy_scale = std::max(
            amrex::Real(1.0e-30_rt), std::abs(lower_energy)
         );
         if (std::abs(upper_pressure - lower_pressure) > continuity_tolerance * pressure_scale
             || std::abs(upper_energy - lower_energy) > continuity_tolerance * energy_scale) {
            std::cerr << "Read piecewise-polytrope continuity mismatch\n";
            return false;
         }
      }
      const amrex::Real stellar_density = 1.0e-3_rt;
      const amrex::Real pressure =
         pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(
            stellar_density, eos
         );
      const amrex::Real sound_speed_squared =
         pyfluid_amrex::relativistic_sound_speed_squared<
            pyfluid_amrex::EOSPolytrope
         >(
            stellar_density,
            pyfluid_amrex::eos_cold_energy_density<
               pyfluid_amrex::EOSPolytrope
            >(stellar_density, eos) / stellar_density,
            eos
         );
      if (!(pressure > 0.0_rt) || !std::isfinite(sound_speed_squared)
          || !(sound_speed_squared > 0.0_rt)) {
         std::cerr << "Read piecewise-polytrope stellar state is invalid\n";
         return false;
      }
   }
   return true;
}

bool validate_srhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4_rt;
   constexpr amrex::Real tolerance = 2.0e-11_rt;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RHDPrimitive, 7> primitive_corpus{{
      {1.0e-3_rt, 1.0e-4_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt},
      {0.071_rt, 0.23_rt, 0.0_rt, 0.31_rt, -0.17_rt, 0.09_rt},
      {0.82_rt, 4.7_rt, 0.0_rt, -0.71_rt, 0.23_rt, -0.11_rt},
      {1.0_rt, 0.7_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt},
      {7.4_rt, 0.017_rt, 0.0_rt, 0.82_rt, -0.31_rt, 0.18_rt},
      {19.0_rt, 13.0_rt, 0.0_rt, -0.91_rt, 0.21_rt, -0.07_rt},
      {0.3_rt, 2.1_rt, 0.0_rt, 0.965_rt, 0.08_rt, -0.03_rt},
   }};

   amrex::Real worst_primitive_error = 0.0_rt;
   amrex::Real worst_conserved_error = 0.0_rt;
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho,
         primitive.eps,
         gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(primitive, minkowski);
      const auto recovered = pyfluid_amrex::recover_srhd_palenzuela_1d(
         conserved,
         gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "SRHD primitive round trip failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            recovered.primitive,
            minkowski
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );

      auto independent_conserved = conserved;
      independent_conserved.mom1 *= -0.83_rt;
      independent_conserved.mom2 += 0.07_rt * independent_conserved.dens;
      independent_conserved.mom3 *= 0.91_rt;
      independent_conserved.tau += 0.19_rt * (
         independent_conserved.dens + independent_conserved.tau
      );
      const auto independent_recovered =
         pyfluid_amrex::recover_srhd_palenzuela_1d(independent_conserved, gamma);
      if (!independent_recovered.attempt.succeeded()) {
         std::cerr << "SRHD conserved round trip failed sample=" << sample
                   << " status=" << static_cast<int>(independent_recovered.attempt.status)
                   << " residual=" << independent_recovered.attempt.residual << '\n';
         return false;
      }
      const auto independent_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            independent_recovered.primitive,
            minkowski
         );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(
            components(independent_round_trip),
            components(independent_conserved)
         )
      );

      const amrex::Real velocity_squared = primitive.v1 * primitive.v1
         + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0_rt / std::sqrt(1.0_rt - velocity_squared);
      const amrex::Real x = (1.0_rt + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6_rt * std::max(amrex::Real{1.0_rt}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0_rt * step);
      const amrex::Real analytic =
         pyfluid_amrex::srhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 2.0e-6_rt * std::max(amrex::Real{1.0_rt}, std::abs(finite_difference))) {
         std::cerr << "SRHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }
   }

   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.83_rt;
   curved_metric.shift1 = 0.07_rt;
   curved_metric.shift2 = -0.03_rt;
   curved_metric.shift3 = 0.02_rt;
   curved_metric.gamma12 = 0.1_rt;
   curved_metric.inv_gamma11 = 1.0_rt / 0.99_rt;
   curved_metric.inv_gamma12 = -0.1_rt / 0.99_rt;
   curved_metric.inv_gamma22 = 1.0_rt / 0.99_rt;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99_rt);
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho, primitive.eps, gamma
      );
      const auto conserved = pyfluid_amrex::generated::rhd_primitive_to_conserved(
         primitive, curved_metric
      );
      const auto recovered = pyfluid_amrex::recover_srhd_palenzuela_1d(
         conserved, curved_metric, gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "Curved RHD recovery failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rhd_primitive_to_conserved(
            recovered.primitive, curved_metric
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );
   }

   if (worst_primitive_error > tolerance || worst_conserved_error > tolerance) {
      std::cerr << "SRHD round-trip residuals primitive=" << worst_primitive_error
                << " conserved=" << worst_conserved_error << '\n';
      return false;
   }
   return true;
}

bool validate_fixed_spacetime_provider()
{
   using pyfluid_amrex::BlackHoleInterior;
   using pyfluid_amrex::FixedSpacetimeMetric;
   using pyfluid_amrex::FixedSpacetimeParameters;
   using pyfluid_amrex::fixed_spacetime_geometry;

   const FixedSpacetimeParameters rindler{
      FixedSpacetimeMetric::Rindler,
      0.11_rt,
      -0.07_rt,
      0.03_rt,
      0.2_rt,
      -0.1_rt,
      0.4_rt,
      1.0_rt,
   };
   const auto rindler_geometry = fixed_spacetime_geometry(
      rindler, 0.8_rt, -0.4_rt, 0.6_rt, 0.0_rt
   );
   const amrex::Real expected_lapse = 1.0_rt
      + rindler.acceleration1 * (0.8_rt - rindler.origin1)
      + rindler.acceleration2 * (-0.4_rt - rindler.origin2)
      + rindler.acceleration3 * (0.6_rt - rindler.origin3);
   if (!close_enough(rindler_geometry.metric.lapse, expected_lapse)
       || !close_enough(
          rindler_geometry.derivatives.d1_lapse, rindler.acceleration1
       )
       || !close_enough(
          rindler_geometry.derivatives.d2_lapse, rindler.acceleration2
       )
       || !close_enough(
          rindler_geometry.derivatives.d3_lapse, rindler.acceleration3
       )) {
      std::cerr << "Rindler fixed-spacetime provider mismatch\n";
      return false;
   }

   pyfluid_amrex::RHDPrimitive rest{1.0_rt, 0.5_rt, 0.2_rt, 0.0_rt, 0.0_rt, 0.0_rt};
   pyfluid_amrex::SpacetimeMetricDerivatives acceleration{};
   acceleration.d1_lapse = 0.1_rt;
   const auto source = pyfluid_amrex::generated::rhd_geometric_source(
      rest, pyfluid_amrex::minkowski_spacetime_metric(), acceleration
   );
   const amrex::Real eulerian_energy = rest.rho * (1.0_rt + rest.eps);
   if (!close_enough(source.dens, 0.0_rt)
       || !close_enough(source.mom1, -0.1_rt * eulerian_energy)
       || !close_enough(source.mom2, 0.0_rt)
       || !close_enough(source.mom3, 0.0_rt)
       || !close_enough(source.tau, 0.0_rt)) {
      std::cerr << "Rindler geometric source mismatch\n";
      return false;
   }

   const FixedSpacetimeParameters schwarzschild{
      FixedSpacetimeMetric::SchwarzschildIsotropic,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      -0.3_rt,
      0.2_rt,
      -0.1_rt,
      0.4_rt,
   };
   constexpr amrex::Real position1 = 1.7_rt;
   constexpr amrex::Real position2 = 0.6_rt;
   constexpr amrex::Real position3 = -0.25_rt;
   const auto schwarzschild_geometry = fixed_spacetime_geometry(
      schwarzschild, position1, position2, position3, 0.0_rt
   );
   const auto& metric = schwarzschild_geometry.metric;
   if (!close_enough(metric.gamma11 * metric.inv_gamma11, 1.0_rt)
       || !close_enough(metric.gamma22 * metric.inv_gamma22, 1.0_rt)
       || !close_enough(metric.gamma33 * metric.inv_gamma33, 1.0_rt)
       || !close_enough(
          metric.sqrt_det_gamma * metric.sqrt_det_gamma,
          metric.gamma11 * metric.gamma22 * metric.gamma33
       )) {
      std::cerr << "isotropic-Schwarzschild metric identity mismatch\n";
      return false;
   }
   constexpr amrex::Real step = 1.0e-6_rt;
   const auto plus = fixed_spacetime_geometry(
      schwarzschild, position1 + step, position2, position3, 0.0_rt
   );
   const auto minus = fixed_spacetime_geometry(
      schwarzschild, position1 - step, position2, position3, 0.0_rt
   );
   const amrex::Real lapse_derivative = (
      plus.metric.lapse - minus.metric.lapse
   ) / (2.0_rt * step);
   const amrex::Real gamma_derivative = (
      plus.metric.gamma11 - minus.metric.gamma11
   ) / (2.0_rt * step);
   if (std::abs(lapse_derivative - schwarzschild_geometry.derivatives.d1_lapse)
          > 2.0e-9_rt
       || std::abs(gamma_derivative - schwarzschild_geometry.derivatives.d1_gamma11)
          > 2.0e-9_rt) {
      std::cerr << "isotropic-Schwarzschild metric derivative mismatch\n";
      return false;
   }

   FixedSpacetimeParameters trumpet{};
   trumpet.metric = FixedSpacetimeMetric::StaticTrumpet;
   trumpet.origin1 = -0.2_rt;
   trumpet.origin2 = 0.1_rt;
   trumpet.origin3 = 0.05_rt;
   trumpet.mass = 0.8_rt;
   trumpet.black_hole_interior = BlackHoleInterior::Puncture;
   constexpr amrex::Real trumpet_position1 = 1.4_rt;
   constexpr amrex::Real trumpet_position2 = -0.5_rt;
   constexpr amrex::Real trumpet_position3 = 0.65_rt;
   const auto static_trumpet = fixed_spacetime_geometry(
      trumpet,
      trumpet_position1,
      trumpet_position2,
      trumpet_position3,
      0.0_rt
   );
   const amrex::Real trumpet_displacement[3]{
      trumpet_position1 - trumpet.origin1,
      trumpet_position2 - trumpet.origin2,
      trumpet_position3 - trumpet.origin3,
   };
   const amrex::Real trumpet_radius = std::sqrt(
      trumpet_displacement[0] * trumpet_displacement[0]
      + trumpet_displacement[1] * trumpet_displacement[1]
      + trumpet_displacement[2] * trumpet_displacement[2]
   );
   const amrex::Real trumpet_radius_plus_mass =
      trumpet_radius + trumpet.mass;
   const amrex::Real expected_trumpet_chi =
      trumpet_radius * trumpet_radius
      / (trumpet_radius_plus_mass * trumpet_radius_plus_mass);
   const auto trumpet_state =
      pyfluid_amrex::z4c_state_from_spacetime_geometry(static_trumpet);
   const auto direct_trumpet_state =
      pyfluid_amrex::static_trumpet_z4c_initial_data_state(
         trumpet,
         trumpet_position1,
         trumpet_position2,
         trumpet_position3
      );
   const auto puncture_trumpet_state =
      pyfluid_amrex::static_trumpet_z4c_initial_data_state(
         trumpet, trumpet.origin1, trumpet.origin2, trumpet.origin3
      );
   const auto& trumpet_metric = static_trumpet.metric;
   const auto& trumpet_derivatives = static_trumpet.derivatives;
   const amrex::Real trumpet_trace =
      trumpet_metric.inv_gamma11 * trumpet_derivatives.extrinsic11
      + 2.0_rt * trumpet_metric.inv_gamma12
         * trumpet_derivatives.extrinsic12
      + 2.0_rt * trumpet_metric.inv_gamma13
         * trumpet_derivatives.extrinsic13
      + trumpet_metric.inv_gamma22 * trumpet_derivatives.extrinsic22
      + 2.0_rt * trumpet_metric.inv_gamma23
         * trumpet_derivatives.extrinsic23
      + trumpet_metric.inv_gamma33 * trumpet_derivatives.extrinsic33;
   pyfluid_amrex::Z4cSpatialDerivatives trumpet_spatial_derivatives{};
   trumpet_spatial_derivatives.advection_lapse[0] =
      trumpet_derivatives.d1_lapse;
   trumpet_spatial_derivatives.advection_lapse[1] =
      trumpet_derivatives.d2_lapse;
   trumpet_spatial_derivatives.advection_lapse[2] =
      trumpet_derivatives.d3_lapse;
   const amrex::Real trumpet_lapse_rhs =
      pyfluid_amrex::generated::z4c_trumpet_lapse_rhs(
         trumpet_state, trumpet_spatial_derivatives
      );
   if (!close_enough(
          trumpet_metric.lapse,
          trumpet_radius / trumpet_radius_plus_mass
       )
       || !close_enough(trumpet_state.chi, expected_trumpet_chi)
       || !close_enough(trumpet_state.gamma_tilde11, 1.0_rt)
       || !close_enough(trumpet_state.gamma_tilde22, 1.0_rt)
       || !close_enough(trumpet_state.gamma_tilde33, 1.0_rt)
       || !close_enough(direct_trumpet_state.chi, trumpet_state.chi)
       || !close_enough(
            direct_trumpet_state.a_tilde12, trumpet_state.a_tilde12
          )
       || !close_enough(puncture_trumpet_state.chi, 0.0_rt)
       || !close_enough(puncture_trumpet_state.lapse, 0.0_rt)
       || !close_enough(puncture_trumpet_state.gamma_tilde11, 1.0_rt)
       || !close_enough(puncture_trumpet_state.a_tilde11, 0.0_rt)
       || std::abs(
            trumpet_trace
            - trumpet.mass
               / (trumpet_radius_plus_mass * trumpet_radius_plus_mass)
          ) > 3.0e-13_rt
       || std::abs(trumpet_lapse_rhs) > 3.0e-15_rt) {
      std::cerr << "static-trumpet identity mismatch\n";
      return false;
   }
   const auto trumpet_plus = fixed_spacetime_geometry(
      trumpet,
      trumpet_position1 + step,
      trumpet_position2,
      trumpet_position3,
      0.0_rt
   );
   const auto trumpet_minus = fixed_spacetime_geometry(
      trumpet,
      trumpet_position1 - step,
      trumpet_position2,
      trumpet_position3,
      0.0_rt
   );
   const amrex::Real trumpet_lapse_derivative =
      (trumpet_plus.metric.lapse - trumpet_minus.metric.lapse)
      / (2.0_rt * step);
   const amrex::Real trumpet_gamma_derivative =
      (trumpet_plus.metric.gamma11 - trumpet_minus.metric.gamma11)
      / (2.0_rt * step);
   const amrex::Real trumpet_shift_derivative =
      (trumpet_plus.metric.shift2 - trumpet_minus.metric.shift2)
      / (2.0_rt * step);
   if (std::abs(
          trumpet_lapse_derivative - trumpet_derivatives.d1_lapse
       ) > 2.0e-9_rt
       || std::abs(
            trumpet_gamma_derivative - trumpet_derivatives.d1_gamma11
          ) > 2.0e-8_rt
       || std::abs(
            trumpet_shift_derivative - trumpet_derivatives.d1_shift2
          ) > 2.0e-9_rt) {
      std::cerr << "static-trumpet derivative mismatch\n";
      return false;
   }

   const FixedSpacetimeParameters schwarzschild_kerr_schild{
      FixedSpacetimeMetric::KerrSchild,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      0.7_rt,
   };
   constexpr amrex::Real ks_position1 = 2.3_rt;
   constexpr amrex::Real ks_position2 = -0.8_rt;
   constexpr amrex::Real ks_position3 = 0.45_rt;
   const auto schwarzschild_ks = kerr_schild_initial_data_geometry(
      schwarzschild_kerr_schild,
      ks_position1,
      ks_position2,
      ks_position3
   );
   const amrex::Real ks_radius = std::sqrt(
      ks_position1 * ks_position1 + ks_position2 * ks_position2
      + ks_position3 * ks_position3
   );
   const amrex::Real ks_profile = schwarzschild_kerr_schild.mass / ks_radius;
   const amrex::Real ks_lapse_denominator = 1.0_rt + 2.0_rt * ks_profile;
   const amrex::Real ks_null1 = ks_position1 / ks_radius;
   const amrex::Real expected_ks_trace =
      2.0_rt * schwarzschild_kerr_schild.mass
      * (ks_radius + 3.0_rt * schwarzschild_kerr_schild.mass)
      / (
           std::pow(ks_radius, 1.5_rt)
           * std::pow(
                ks_radius + 2.0_rt * schwarzschild_kerr_schild.mass,
                1.5_rt
             )
        );
   const auto& ks_metric = schwarzschild_ks.metric;
   const auto& ks_derivatives = schwarzschild_ks.derivatives;
   const amrex::Real ks_trace =
      ks_metric.inv_gamma11 * ks_derivatives.extrinsic11
      + 2.0_rt * ks_metric.inv_gamma12 * ks_derivatives.extrinsic12
      + 2.0_rt * ks_metric.inv_gamma13 * ks_derivatives.extrinsic13
      + ks_metric.inv_gamma22 * ks_derivatives.extrinsic22
      + 2.0_rt * ks_metric.inv_gamma23 * ks_derivatives.extrinsic23
      + ks_metric.inv_gamma33 * ks_derivatives.extrinsic33;
   if (!close_enough(
          ks_metric.lapse, 1.0_rt / std::sqrt(ks_lapse_denominator)
       )
       || !close_enough(
          ks_metric.shift1,
          2.0_rt * ks_profile * ks_null1 / ks_lapse_denominator
       )
       || !close_enough(
          ks_metric.gamma11,
          1.0_rt + 2.0_rt * ks_profile * ks_null1 * ks_null1
       )
       || std::abs(ks_trace - expected_ks_trace) > 3.0e-13_rt) {
      std::cerr << "Schwarzschild Kerr-Schild identity mismatch\n";
      return false;
   }

   const FixedSpacetimeParameters spinning_kerr_schild{
      FixedSpacetimeMetric::KerrSchild,
      0.0_rt,
      0.0_rt,
      0.0_rt,
      -0.15_rt,
      0.07_rt,
      0.11_rt,
      0.9_rt,
      0.2_rt,
      -0.1_rt,
      0.35_rt,
   };
   const amrex::Real spinning_position[3]{2.4_rt, -1.3_rt, 0.75_rt};
   const auto spinning = kerr_schild_initial_data_geometry(
      spinning_kerr_schild,
      spinning_position[0],
      spinning_position[1],
      spinning_position[2]
   );
   if (!has_stationary_adm_identity(spinning)) {
      std::cerr << "exact Kerr-Schild data violates the stationary ADM identity\n";
      return false;
   }
   const amrex::Real analytic_lapse_derivatives[3]{
      spinning.derivatives.d1_lapse,
      spinning.derivatives.d2_lapse,
      spinning.derivatives.d3_lapse,
   };
   const amrex::Real analytic_shift_derivatives[3][3]{
      {
         spinning.derivatives.d1_shift1,
         spinning.derivatives.d1_shift2,
         spinning.derivatives.d1_shift3,
      },
      {
         spinning.derivatives.d2_shift1,
         spinning.derivatives.d2_shift2,
         spinning.derivatives.d2_shift3,
      },
      {
         spinning.derivatives.d3_shift1,
         spinning.derivatives.d3_shift2,
         spinning.derivatives.d3_shift3,
      },
   };
   const amrex::Real analytic_gamma_derivatives[3][6]{
      {
         spinning.derivatives.d1_gamma11,
         spinning.derivatives.d1_gamma12,
         spinning.derivatives.d1_gamma13,
         spinning.derivatives.d1_gamma22,
         spinning.derivatives.d1_gamma23,
         spinning.derivatives.d1_gamma33,
      },
      {
         spinning.derivatives.d2_gamma11,
         spinning.derivatives.d2_gamma12,
         spinning.derivatives.d2_gamma13,
         spinning.derivatives.d2_gamma22,
         spinning.derivatives.d2_gamma23,
         spinning.derivatives.d2_gamma33,
      },
      {
         spinning.derivatives.d3_gamma11,
         spinning.derivatives.d3_gamma12,
         spinning.derivatives.d3_gamma13,
         spinning.derivatives.d3_gamma22,
         spinning.derivatives.d3_gamma23,
         spinning.derivatives.d3_gamma33,
      },
   };
   for (int direction = 0; direction < 3; ++direction) {
      amrex::Real plus_position[3]{
         spinning_position[0], spinning_position[1], spinning_position[2]
      };
      amrex::Real minus_position[3]{
         spinning_position[0], spinning_position[1], spinning_position[2]
      };
      plus_position[direction] += step;
      minus_position[direction] -= step;
      const auto spinning_plus = kerr_schild_initial_data_geometry(
         spinning_kerr_schild,
         plus_position[0], plus_position[1], plus_position[2]
      );
      const auto spinning_minus = kerr_schild_initial_data_geometry(
         spinning_kerr_schild,
         minus_position[0], minus_position[1], minus_position[2]
      );
      const amrex::Real numerical_lapse = (
         spinning_plus.metric.lapse - spinning_minus.metric.lapse
      ) / (2.0_rt * step);
      const amrex::Real numerical_shift[3]{
         (spinning_plus.metric.shift1 - spinning_minus.metric.shift1)
            / (2.0_rt * step),
         (spinning_plus.metric.shift2 - spinning_minus.metric.shift2)
            / (2.0_rt * step),
         (spinning_plus.metric.shift3 - spinning_minus.metric.shift3)
            / (2.0_rt * step),
      };
      const amrex::Real numerical_gamma[6]{
         (spinning_plus.metric.gamma11 - spinning_minus.metric.gamma11)
            / (2.0_rt * step),
         (spinning_plus.metric.gamma12 - spinning_minus.metric.gamma12)
            / (2.0_rt * step),
         (spinning_plus.metric.gamma13 - spinning_minus.metric.gamma13)
            / (2.0_rt * step),
         (spinning_plus.metric.gamma22 - spinning_minus.metric.gamma22)
            / (2.0_rt * step),
         (spinning_plus.metric.gamma23 - spinning_minus.metric.gamma23)
            / (2.0_rt * step),
         (spinning_plus.metric.gamma33 - spinning_minus.metric.gamma33)
            / (2.0_rt * step),
      };
      if (std::abs(numerical_lapse - analytic_lapse_derivatives[direction])
             > 2.0e-9_rt) {
         std::cerr << "Kerr-Schild lapse derivative mismatch\n";
         return false;
      }
      for (int component = 0; component < 3; ++component) {
         if (std::abs(
                numerical_shift[component]
                - analytic_shift_derivatives[direction][component]
             ) > 2.0e-9_rt) {
            std::cerr << "Kerr-Schild shift derivative mismatch\n";
            return false;
         }
      }
      for (int component = 0; component < 6; ++component) {
         if (std::abs(
                numerical_gamma[component]
                - analytic_gamma_derivatives[direction][component]
             ) > 3.0e-9_rt) {
            std::cerr << "Kerr-Schild metric derivative mismatch\n";
            return false;
         }
      }
   }
   FixedSpacetimeParameters rotated_kerr_schild = spinning_kerr_schild;
   rotated_kerr_schild.origin1 = spinning_kerr_schild.origin3;
   rotated_kerr_schild.origin2 = spinning_kerr_schild.origin1;
   rotated_kerr_schild.origin3 = spinning_kerr_schild.origin2;
   rotated_kerr_schild.dimensionless_spin1 =
      spinning_kerr_schild.dimensionless_spin3;
   rotated_kerr_schild.dimensionless_spin2 =
      spinning_kerr_schild.dimensionless_spin1;
   rotated_kerr_schild.dimensionless_spin3 =
      spinning_kerr_schild.dimensionless_spin2;
   const auto rotated = kerr_schild_initial_data_geometry(
      rotated_kerr_schild,
      spinning_position[2], spinning_position[0], spinning_position[1]
   );
   if (!close_enough(rotated.metric.lapse, spinning.metric.lapse)
       || !close_enough(rotated.metric.shift1, spinning.metric.shift3)
       || !close_enough(rotated.metric.shift2, spinning.metric.shift1)
       || !close_enough(rotated.metric.shift3, spinning.metric.shift2)
       || !close_enough(rotated.metric.gamma11, spinning.metric.gamma33)
       || !close_enough(rotated.metric.gamma12, spinning.metric.gamma13)
       || !close_enough(rotated.metric.gamma13, spinning.metric.gamma23)
       || !close_enough(rotated.metric.gamma22, spinning.metric.gamma11)
       || !close_enough(rotated.metric.gamma23, spinning.metric.gamma12)
       || !close_enough(rotated.metric.gamma33, spinning.metric.gamma22)
       || !close_enough(
          rotated.derivatives.extrinsic11,
          spinning.derivatives.extrinsic33
       )
       || !close_enough(
          rotated.derivatives.extrinsic12,
          spinning.derivatives.extrinsic13
       )
       || !close_enough(
          rotated.derivatives.extrinsic23,
          spinning.derivatives.extrinsic12
       )) {
      std::cerr << "Kerr-Schild rotation covariance mismatch\n";
      return false;
   }
   FixedSpacetimeParameters filled_kerr_schild = spinning_kerr_schild;
   filled_kerr_schild.black_hole_interior = BlackHoleInterior::Filled;
   filled_kerr_schild.black_hole_fill_radius = 1.0_rt;
   const auto filled_center = kerr_schild_initial_data_geometry(
      filled_kerr_schild,
      filled_kerr_schild.origin1,
      filled_kerr_schild.origin2,
      filled_kerr_schild.origin3
   );
   if (!close_enough(filled_center.metric.lapse, 1.0_rt)
       || !close_enough(filled_center.metric.gamma11, 1.0_rt)
       || !close_enough(filled_center.metric.gamma22, 1.0_rt)
       || !close_enough(filled_center.metric.gamma33, 1.0_rt)
       || !close_enough(filled_center.metric.shift1, 0.0_rt)
       || !close_enough(filled_center.derivatives.d1_lapse, 0.0_rt)
       || !close_enough(filled_center.derivatives.d2_shift3, 0.0_rt)
       || !close_enough(filled_center.derivatives.d3_gamma22, 0.0_rt)
       || !close_enough(filled_center.derivatives.extrinsic11, 0.0_rt)
       || !has_stationary_adm_identity(filled_center)) {
      std::cerr << "filled Kerr-Schild center is not regular Minkowski data\n";
      return false;
   }
   const amrex::Real exterior_position[3]{
      filled_kerr_schild.origin1 + 1.1_rt,
      filled_kerr_schild.origin2,
      filled_kerr_schild.origin3,
   };
   const auto filled_exterior = kerr_schild_initial_data_geometry(
      filled_kerr_schild,
      exterior_position[0], exterior_position[1], exterior_position[2]
   );
   const auto exact_exterior = exact_kerr_schild_initial_data_geometry(
      filled_kerr_schild,
      exterior_position[0], exterior_position[1], exterior_position[2]
   );
   if (!close_enough(filled_exterior.metric.lapse, exact_exterior.metric.lapse)
       || !close_enough(
          filled_exterior.derivatives.extrinsic13,
          exact_exterior.derivatives.extrinsic13
       )
       || !close_enough(
          filled_exterior.derivatives.d2_gamma23,
          exact_exterior.derivatives.d2_gamma23
       )
       || !has_stationary_adm_identity(filled_exterior)) {
      std::cerr << "filled Kerr-Schild data changed its exact exterior\n";
      return false;
   }

   const amrex::Real filled_transition_position[3]{
      filled_kerr_schild.origin1 + 0.65_rt,
      filled_kerr_schild.origin2,
      filled_kerr_schild.origin3,
   };
   const auto filled_transition = kerr_schild_initial_data_geometry(
      filled_kerr_schild,
      filled_transition_position[0],
      filled_transition_position[1],
      filled_transition_position[2]
   );
   if (!has_stationary_adm_identity(filled_transition)) {
      std::cerr << "filled Kerr-Schild transition violates the stationary ADM identity\n";
      return false;
   }
   const amrex::Real filled_lapse_derivatives[3]{
      filled_transition.derivatives.d1_lapse,
      filled_transition.derivatives.d2_lapse,
      filled_transition.derivatives.d3_lapse,
   };
   const amrex::Real filled_shift_derivatives[3][3]{
      {
         filled_transition.derivatives.d1_shift1,
         filled_transition.derivatives.d1_shift2,
         filled_transition.derivatives.d1_shift3,
      },
      {
         filled_transition.derivatives.d2_shift1,
         filled_transition.derivatives.d2_shift2,
         filled_transition.derivatives.d2_shift3,
      },
      {
         filled_transition.derivatives.d3_shift1,
         filled_transition.derivatives.d3_shift2,
         filled_transition.derivatives.d3_shift3,
      },
   };
   const amrex::Real filled_gamma_derivatives[3][6]{
      {
         filled_transition.derivatives.d1_gamma11,
         filled_transition.derivatives.d1_gamma12,
         filled_transition.derivatives.d1_gamma13,
         filled_transition.derivatives.d1_gamma22,
         filled_transition.derivatives.d1_gamma23,
         filled_transition.derivatives.d1_gamma33,
      },
      {
         filled_transition.derivatives.d2_gamma11,
         filled_transition.derivatives.d2_gamma12,
         filled_transition.derivatives.d2_gamma13,
         filled_transition.derivatives.d2_gamma22,
         filled_transition.derivatives.d2_gamma23,
         filled_transition.derivatives.d2_gamma33,
      },
      {
         filled_transition.derivatives.d3_gamma11,
         filled_transition.derivatives.d3_gamma12,
         filled_transition.derivatives.d3_gamma13,
         filled_transition.derivatives.d3_gamma22,
         filled_transition.derivatives.d3_gamma23,
         filled_transition.derivatives.d3_gamma33,
      },
   };
   for (int direction = 0; direction < 3; ++direction) {
      amrex::Real plus_position[3]{
         filled_transition_position[0],
         filled_transition_position[1],
         filled_transition_position[2],
      };
      amrex::Real minus_position[3]{
         filled_transition_position[0],
         filled_transition_position[1],
         filled_transition_position[2],
      };
      plus_position[direction] += step;
      minus_position[direction] -= step;
      const auto plus = kerr_schild_initial_data_geometry(
         filled_kerr_schild, plus_position[0], plus_position[1], plus_position[2]
      );
      const auto minus = kerr_schild_initial_data_geometry(
         filled_kerr_schild, minus_position[0], minus_position[1], minus_position[2]
      );
      const amrex::Real numerical_lapse = (
         plus.metric.lapse - minus.metric.lapse
      ) / (2.0_rt * step);
      const amrex::Real numerical_shift[3]{
         (plus.metric.shift1 - minus.metric.shift1) / (2.0_rt * step),
         (plus.metric.shift2 - minus.metric.shift2) / (2.0_rt * step),
         (plus.metric.shift3 - minus.metric.shift3) / (2.0_rt * step),
      };
      const amrex::Real numerical_gamma[6]{
         (plus.metric.gamma11 - minus.metric.gamma11) / (2.0_rt * step),
         (plus.metric.gamma12 - minus.metric.gamma12) / (2.0_rt * step),
         (plus.metric.gamma13 - minus.metric.gamma13) / (2.0_rt * step),
         (plus.metric.gamma22 - minus.metric.gamma22) / (2.0_rt * step),
         (plus.metric.gamma23 - minus.metric.gamma23) / (2.0_rt * step),
         (plus.metric.gamma33 - minus.metric.gamma33) / (2.0_rt * step),
      };
      if (std::abs(numerical_lapse - filled_lapse_derivatives[direction])
             > 2.0e-8_rt) {
         std::cerr << "filled Kerr-Schild lapse derivative mismatch\n";
         return false;
      }
      for (int component = 0; component < 3; ++component) {
         if (std::abs(
                numerical_shift[component]
                - filled_shift_derivatives[direction][component]
             ) > 2.0e-8_rt) {
            std::cerr << "filled Kerr-Schild shift derivative mismatch\n";
            return false;
         }
      }
      for (int component = 0; component < 6; ++component) {
         if (std::abs(
                numerical_gamma[component]
                - filled_gamma_derivatives[direction][component]
             ) > 3.0e-8_rt) {
            std::cerr << "filled Kerr-Schild metric derivative mismatch\n";
            return false;
         }
      }
   }
   FixedSpacetimeParameters filled_rotated_kerr_schild = rotated_kerr_schild;
   filled_rotated_kerr_schild.black_hole_interior = BlackHoleInterior::Filled;
   filled_rotated_kerr_schild.black_hole_fill_radius = 1.0_rt;
   const auto filled_rotated = kerr_schild_initial_data_geometry(
      filled_rotated_kerr_schild,
      filled_transition_position[2],
      filled_transition_position[0],
      filled_transition_position[1]
   );
   if (!close_enough(filled_rotated.metric.lapse, filled_transition.metric.lapse)
       || !close_enough(filled_rotated.metric.shift1, filled_transition.metric.shift3)
       || !close_enough(filled_rotated.metric.shift2, filled_transition.metric.shift1)
       || !close_enough(filled_rotated.metric.shift3, filled_transition.metric.shift2)
       || !close_enough(
          filled_rotated.derivatives.extrinsic12,
          filled_transition.derivatives.extrinsic13
       )
       || !close_enough(
          filled_rotated.derivatives.d1_lapse,
          filled_transition.derivatives.d3_lapse
       )
       || !has_stationary_adm_identity(filled_rotated)) {
      std::cerr << "filled Kerr-Schild rotation covariance mismatch\n";
      return false;
   }

   constexpr amrex::Real flat_sphere_radius = 1.7_rt;
   const amrex::Real flat_sphere_normal[3]{0.6_rt, -0.8_rt, 0.0_rt};
   amrex::Real flat_sphere_hessian[3][3]{};
   for (int first = 0; first < 3; ++first) {
      for (int second = 0; second < 3; ++second) {
         flat_sphere_hessian[first][second] = (
            (first == second ? 1.0_rt : 0.0_rt)
            - flat_sphere_normal[first] * flat_sphere_normal[second]
         ) / flat_sphere_radius;
      }
   }
   const amrex::Real flat_sphere_expansion = apparent_horizon_expansion(
      pyfluid_amrex::minkowski_spacetime_metric(),
      {},
      flat_sphere_normal,
      flat_sphere_hessian
   );
   if (!close_enough(flat_sphere_expansion, 2.0_rt / flat_sphere_radius)) {
      std::cerr << "flat-sphere outgoing expansion mismatch: "
                << flat_sphere_expansion << '\n';
      return false;
   }

   FixedSpacetimeParameters horizon_kerr = spinning_kerr_schild;
   horizon_kerr.dimensionless_spin1 = 0.0_rt;
   horizon_kerr.dimensionless_spin2 = 0.0_rt;
   const amrex::Real horizon_radius =
      kerr_schild_outer_horizon_radius(horizon_kerr);
   const amrex::Real spin_length = horizon_kerr.mass
      * horizon_kerr.dimensionless_spin3;
   const amrex::Real radial_scale = std::sqrt(
      horizon_radius * horizon_radius + spin_length * spin_length
   );
   constexpr amrex::Real surface_theta = 1.1_rt;
   constexpr amrex::Real surface_phi = 0.7_rt;
   const amrex::Real horizon_position[3]{
      horizon_kerr.origin1
         + radial_scale * std::sin(surface_theta) * std::cos(surface_phi),
      horizon_kerr.origin2
         + radial_scale * std::sin(surface_theta) * std::sin(surface_phi),
      horizon_kerr.origin3
         + horizon_radius * std::cos(surface_theta),
   };
   constexpr amrex::Real derivative_step = 2.0e-4_rt;
   const auto level_set = [&](amrex::Real x, amrex::Real y, amrex::Real z) {
      return std::sqrt(kerr_schild_radius_squared(
         horizon_kerr, x, y, z
      )) - horizon_radius;
   };
   amrex::Real level_set_gradient[3]{};
   amrex::Real level_set_hessian[3][3]{};
   const amrex::Real center_value = level_set(
      horizon_position[0], horizon_position[1], horizon_position[2]
   );
   for (int first = 0; first < 3; ++first) {
      amrex::Real plus[3]{
         horizon_position[0], horizon_position[1], horizon_position[2]
      };
      amrex::Real minus[3]{
         horizon_position[0], horizon_position[1], horizon_position[2]
      };
      plus[first] += derivative_step;
      minus[first] -= derivative_step;
      const amrex::Real plus_value = level_set(plus[0], plus[1], plus[2]);
      const amrex::Real minus_value = level_set(minus[0], minus[1], minus[2]);
      level_set_gradient[first] = (plus_value - minus_value)
         / (2.0_rt * derivative_step);
      level_set_hessian[first][first] = (
         plus_value - 2.0_rt * center_value + minus_value
      ) / (derivative_step * derivative_step);
      for (int second = first + 1; second < 3; ++second) {
         amrex::Real first_plus_second_plus[3]{
            horizon_position[0], horizon_position[1], horizon_position[2]
         };
         amrex::Real first_plus_second_minus[3]{
            horizon_position[0], horizon_position[1], horizon_position[2]
         };
         amrex::Real first_minus_second_plus[3]{
            horizon_position[0], horizon_position[1], horizon_position[2]
         };
         amrex::Real first_minus_second_minus[3]{
            horizon_position[0], horizon_position[1], horizon_position[2]
         };
         first_plus_second_plus[first] += derivative_step;
         first_plus_second_plus[second] += derivative_step;
         first_plus_second_minus[first] += derivative_step;
         first_plus_second_minus[second] -= derivative_step;
         first_minus_second_plus[first] -= derivative_step;
         first_minus_second_plus[second] += derivative_step;
         first_minus_second_minus[first] -= derivative_step;
         first_minus_second_minus[second] -= derivative_step;
         const amrex::Real mixed = (
            level_set(
               first_plus_second_plus[0],
               first_plus_second_plus[1],
               first_plus_second_plus[2]
            )
            - level_set(
               first_plus_second_minus[0],
               first_plus_second_minus[1],
               first_plus_second_minus[2]
            )
            - level_set(
               first_minus_second_plus[0],
               first_minus_second_plus[1],
               first_minus_second_plus[2]
            )
            + level_set(
               first_minus_second_minus[0],
               first_minus_second_minus[1],
               first_minus_second_minus[2]
            )
         ) / (4.0_rt * derivative_step * derivative_step);
         level_set_hessian[first][second] = mixed;
         level_set_hessian[second][first] = mixed;
      }
   }
   const auto horizon_geometry = exact_kerr_schild_initial_data_geometry(
      horizon_kerr,
      horizon_position[0],
      horizon_position[1],
      horizon_position[2]
   );
   const amrex::Real horizon_expansion = apparent_horizon_expansion(
      horizon_geometry.metric,
      horizon_geometry.derivatives,
      level_set_gradient,
      level_set_hessian
   );
   if (std::abs(center_value) > 2.0e-14_rt
       || std::abs(horizon_expansion) > 2.0e-7_rt) {
      std::cerr << "Kerr apparent-horizon expansion mismatch: F="
                << center_value << " expansion=" << horizon_expansion << '\n';
      return false;
   }
   return true;
}

bool validate_srmhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4_rt;
   constexpr amrex::Real tolerance = 5.0e-10_rt;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RMHDPrimitive, 8> primitive_corpus{{
      {1.0e-3_rt, 1.0e-4_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt},
      {0.071_rt, 0.23_rt, 0.0_rt, 0.31_rt, -0.17_rt, 0.09_rt, 0.04_rt, -0.02_rt, 0.01_rt},
      {0.82_rt, 4.7_rt, 0.0_rt, -0.71_rt, 0.23_rt, -0.11_rt, -2.1_rt, 3.7_rt, -0.4_rt},
      {1.0_rt, 0.7_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 8.0_rt, -3.0_rt, 1.5_rt},
      {7.4_rt, 0.017_rt, 0.0_rt, 0.82_rt, -0.31_rt, 0.18_rt, 0.003_rt, -8.0_rt, 2.5_rt},
      {19.0_rt, 13.0_rt, 0.0_rt, -0.91_rt, 0.21_rt, -0.07_rt, 15.0_rt, 0.2_rt, -0.5_rt},
      {0.3_rt, 2.1_rt, 0.0_rt, 0.965_rt, 0.08_rt, -0.03_rt, 0.9_rt, -1.4_rt, 0.7_rt},
      {2.7_rt, 0.08_rt, 0.0_rt, -0.44_rt, -0.51_rt, 0.37_rt, -9.0_rt, 4.0_rt, 6.0_rt},
   }};

   amrex::Real worst_primitive_error = 0.0_rt;
   amrex::Real worst_conserved_error = 0.0_rt;
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho,
         primitive.eps,
         gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(primitive, minkowski);
      const auto recovered = pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
         conserved,
         gamma
      );
      if (!recovered.attempt.succeeded()) {
         std::cerr << "SRMHD primitive round trip failed sample=" << sample
                   << " status=" << static_cast<int>(recovered.attempt.status)
                   << " residual=" << recovered.attempt.residual << '\n';
         return false;
      }
      const auto conserved_round_trip =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            recovered.primitive,
            minkowski
         );
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(components(recovered.primitive), components(primitive))
      );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(components(conserved_round_trip), components(conserved))
      );
      const amrex::Real velocity_squared = primitive.v1 * primitive.v1
         + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0_rt / std::sqrt(1.0_rt - velocity_squared);
      const pyfluid_amrex::RMHDEntropyInput entropy{
         lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0_rt),
      };
      const auto default_policy = pyfluid_amrex::recover_srmhd<true>(
         conserved, gamma
      );
      if (!default_policy.succeeded()
          || default_policy.attempt.method != pyfluid_amrex::C2PMethod::RePrimAnd
          || default_policy.attempt_count != 1) {
         std::cerr << "SRMHD default policy ordering failed sample=" << sample
                   << '\n';
         return false;
      }
      const std::array independent_methods{
         pyfluid_amrex::recover_srmhd_reprimand<true>(conserved, gamma),
         pyfluid_amrex::recover_srmhd_noble_2d<true>(
            conserved, primitive, gamma
         ),
         pyfluid_amrex::recover_srmhd<true>(conserved, primitive, gamma),
         pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy<true>(
            conserved, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_1d_entropy<true>(
            conserved, primitive, entropy, gamma
         ),
         default_policy,
      };
      for (const auto& method_recovery : independent_methods) {
         if (!method_recovery.succeeded()) {
            std::cerr << "SRMHD alternate recovery failed sample=" << sample
                      << " method=" << static_cast<int>(method_recovery.attempt.method)
                      << " status=" << static_cast<int>(method_recovery.attempt.status)
                      << " residual=" << method_recovery.attempt.residual << '\n';
            return false;
         }
         const auto method_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               method_recovery.primitive,
               minkowski
            );
         worst_primitive_error = std::max(
            worst_primitive_error,
            scaled_max_error(
               components(method_recovery.primitive), components(primitive)
            )
         );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(components(method_round_trip), components(conserved))
         );
      }

      auto independent_conserved = conserved;
      independent_conserved.mom1 *= 0.997_rt;
      independent_conserved.mom2 += 0.001_rt * independent_conserved.dens;
      independent_conserved.mom3 *= 1.002_rt;
      independent_conserved.tau += 0.003_rt * (
         independent_conserved.dens + independent_conserved.tau
      );
      const auto independent_recovered =
         pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
            independent_conserved, gamma
         );
      if (!independent_recovered.attempt.succeeded()) {
         std::cerr << "SRMHD conserved round trip failed sample=" << sample
                   << " status=" << static_cast<int>(independent_recovered.attempt.status)
                   << " residual=" << independent_recovered.attempt.residual << '\n';
         return false;
      }
      const auto independent_round_trip =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            independent_recovered.primitive,
            minkowski
         );
      worst_conserved_error = std::max(
         worst_conserved_error,
         scaled_max_error(
            components(independent_round_trip),
            components(independent_conserved)
         )
      );
      const auto independent_reprimand =
         pyfluid_amrex::recover_srmhd_reprimand<true>(
            independent_conserved, gamma
         );
      const auto independent_policy =
         pyfluid_amrex::recover_srmhd<true>(independent_conserved, gamma);
      for (const auto& method_recovery : std::array{
              independent_reprimand,
              independent_policy,
           }) {
         if (!method_recovery.succeeded()) {
            std::cerr << "SRMHD alternate conserved recovery failed sample=" << sample
                      << " method=" << static_cast<int>(method_recovery.attempt.method)
                      << " status=" << static_cast<int>(method_recovery.attempt.status)
                      << " residual=" << method_recovery.attempt.residual << '\n';
            return false;
         }
         const auto method_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               method_recovery.primitive,
               minkowski
            );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(
               components(method_round_trip), components(independent_conserved)
            )
         );
      }

      const amrex::Real x = (1.0_rt + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6_rt * std::max(amrex::Real{1.0_rt}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srmhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srmhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0_rt * step);
      const amrex::Real analytic =
         pyfluid_amrex::srmhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 3.0e-6_rt * std::max(amrex::Real{1.0_rt}, std::abs(finite_difference))) {
         std::cerr << "SRMHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }

      constexpr pyfluid_amrex::ColdPolytropeEquationOfState cold_eos{
         0.2_rt, 1.8_rt,
      };
      auto cold_primitive = primitive;
      cold_primitive.pressure = cold_eos.polytropic_k * std::pow(
         cold_primitive.rho, cold_eos.polytropic_gamma
      );
      cold_primitive.eps = cold_primitive.pressure / (
         cold_primitive.rho * (cold_eos.polytropic_gamma - 1.0_rt)
      );
      const auto cold_conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            cold_primitive, minkowski
         );
      const auto cold_recovered = pyfluid_amrex::recover_srmhd_font_1d<true>(
         cold_conserved, cold_eos
      );
      if (!cold_recovered.succeeded()) {
         std::cerr << "SRMHD cold recovery failed sample=" << sample
                   << " status=" << static_cast<int>(cold_recovered.attempt.status)
                   << " residual=" << cold_recovered.attempt.residual << '\n';
         return false;
      }
      worst_primitive_error = std::max(
         worst_primitive_error,
         scaled_max_error(
            components(cold_recovered.primitive), components(cold_primitive)
         )
      );
   }

   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.83_rt;
   curved_metric.shift1 = 0.07_rt;
   curved_metric.shift2 = -0.03_rt;
   curved_metric.shift3 = 0.02_rt;
   curved_metric.gamma12 = 0.1_rt;
   curved_metric.inv_gamma11 = 1.0_rt / 0.99_rt;
   curved_metric.inv_gamma12 = -0.1_rt / 0.99_rt;
   curved_metric.inv_gamma22 = 1.0_rt / 0.99_rt;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99_rt);
   for (std::size_t sample = 0; sample < primitive_corpus.size(); ++sample) {
      auto primitive = primitive_corpus[sample];
      primitive.pressure = pyfluid_amrex::generated::relativistic_ideal_gas_pressure(
         primitive.rho, primitive.eps, gamma
      );
      const auto conserved =
         pyfluid_amrex::generated::rmhd_primitive_to_conserved(
            primitive, curved_metric
         );
      const amrex::Real velocity_squared =
         pyfluid_amrex::rmhd_velocity_squared(primitive, curved_metric);
      const amrex::Real lorentz_factor = 1.0_rt / std::sqrt(
         1.0_rt - velocity_squared
      );
      const pyfluid_amrex::RMHDEntropyInput entropy{
         curved_metric.sqrt_det_gamma * lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0_rt),
      };
      const std::array recoveries{
         pyfluid_amrex::recover_srmhd_palenzuela_1d<true>(
            conserved, curved_metric, gamma
         ),
         pyfluid_amrex::recover_srmhd_reprimand<true>(
            conserved, curved_metric, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_2d<true>(
            conserved, curved_metric, primitive, gamma
         ),
         pyfluid_amrex::recover_srmhd_palenzuela_1d_entropy<true>(
            conserved, curved_metric, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd_noble_1d_entropy<true>(
            conserved, curved_metric, primitive, entropy, gamma
         ),
         pyfluid_amrex::recover_srmhd<true>(conserved, curved_metric, gamma),
      };
      for (const auto& recovered : recoveries) {
         if (!recovered.succeeded()) {
            std::cerr << "Curved RMHD recovery failed sample=" << sample
                      << " method=" << static_cast<int>(recovered.attempt.method)
                      << " status=" << static_cast<int>(recovered.attempt.status)
                      << " residual=" << recovered.attempt.residual << '\n';
            return false;
         }
         const auto conserved_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               recovered.primitive, curved_metric
            );
         worst_primitive_error = std::max(
            worst_primitive_error,
            scaled_max_error(
               components(recovered.primitive), components(primitive)
            )
         );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(
               components(conserved_round_trip), components(conserved)
            )
         );
      }
   }

   if (worst_primitive_error > tolerance || worst_conserved_error > tolerance) {
      std::cerr << "SRMHD round-trip residuals primitive=" << worst_primitive_error
                << " conserved=" << worst_conserved_error << '\n';
      return false;
   }

   return true;
}

bool validate_runtime_eos_rmhd_round_trips()
{
   pyfluid_amrex::EOSParams eos{};
   eos.gamma = 1.4_rt;
   eos.thermal_gamma = 1.6_rt;
   eos.polytrope_segments = 1;
   eos.segment_k[0] = 0.2_rt;
   eos.segment_gamma[0] = 1.8_rt;
   const pyfluid_amrex::SpacetimeMetric metric =
      pyfluid_amrex::minkowski_spacetime_metric();
   amrex::Real worst_primitive_error = 0.0_rt;
   amrex::Real worst_conserved_error = 0.0_rt;
   amrex::Real worst_characteristic_error = 0.0_rt;
   amrex::Real worst_ideal_fast_path_error = 0.0_rt;

   for (int eos_method : {
           pyfluid_amrex::EOSIdealGas,
           pyfluid_amrex::EOSPolytrope,
           pyfluid_amrex::EOSHybridPolytrope,
        }) {
      eos.method = eos_method;
      for (amrex::Real rho : {0.01_rt, 0.3_rt, 1.7_rt}) {
         const amrex::Real cold_energy =
            pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSRuntime>(
               rho, eos
            );
         const amrex::Real thermal_specific_energy =
            eos_method == pyfluid_amrex::EOSPolytrope ? 0.0_rt : 0.07_rt;
         pyfluid_amrex::RMHDPrimitive primitive{
            rho,
            cold_energy / rho + thermal_specific_energy,
            0.0_rt,
            0.21_rt,
            -0.08_rt,
            0.04_rt,
            0.3_rt,
            -0.2_rt,
            0.1_rt,
         };
         primitive.pressure = pyfluid_amrex::relativistic_pressure<
            pyfluid_amrex::EOSRuntime
         >(primitive.rho, primitive.eps, eos);
         const pyfluid_amrex::RelativisticCharacteristicState
            characteristic =
               pyfluid_amrex::relativistic_characteristic_state<
                  pyfluid_amrex::EOSRuntime
               >(primitive.rho, primitive.eps, eos);
         worst_characteristic_error = std::max(
            worst_characteristic_error,
            scaled_max_error(
               std::array<amrex::Real, 2>{
                  characteristic.specific_enthalpy,
                  characteristic.sound_speed_squared,
               },
               std::array<amrex::Real, 2>{
                  pyfluid_amrex::relativistic_specific_enthalpy<
                     pyfluid_amrex::EOSRuntime
                  >(primitive.rho, primitive.eps, eos),
                  pyfluid_amrex::relativistic_sound_speed_squared<
                     pyfluid_amrex::EOSRuntime
                  >(primitive.rho, primitive.eps, eos),
               }
            )
         );
         const pyfluid_amrex::RMHDConserved conserved =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               primitive, metric
            );
         const pyfluid_amrex::RMHDRecoveryResult recovered =
            pyfluid_amrex::recover_srmhd<true>(
               conserved, metric, eos
            );
         if (!recovered.succeeded()) {
            std::cerr << "runtime-EOS RMHD recovery failed EOS=" << eos_method
                      << " rho=" << rho
                      << " method="
                      << static_cast<int>(recovered.attempt.method)
                      << " status="
                      << static_cast<int>(recovered.attempt.status)
                      << " residual=" << recovered.attempt.residual << '\n';
            return false;
         }
         if (eos_method == pyfluid_amrex::EOSIdealGas) {
            const pyfluid_amrex::RMHDRecoveryResult ideal_control =
               pyfluid_amrex::recover_srmhd<true>(
                  conserved, metric, eos.gamma
               );
            if (!ideal_control.succeeded()
                || ideal_control.attempt.method != recovered.attempt.method
                || ideal_control.attempt.status != recovered.attempt.status
                || ideal_control.attempt.iterations
                   != recovered.attempt.iterations) {
               std::cerr << "runtime-EOS ideal RMHD fast path disagrees with "
                            "the legacy ideal-gas API control\n";
               return false;
            }
            worst_ideal_fast_path_error = std::max(
               worst_ideal_fast_path_error,
               scaled_max_error(
                  components(recovered.primitive),
                  components(ideal_control.primitive)
               )
            );
         }
         const pyfluid_amrex::RMHDConserved conserved_round_trip =
            pyfluid_amrex::generated::rmhd_primitive_to_conserved(
               recovered.primitive, metric
            );
         worst_primitive_error = std::max(
            worst_primitive_error,
            scaled_max_error(
               components(recovered.primitive), components(primitive)
            )
         );
         worst_conserved_error = std::max(
            worst_conserved_error,
            scaled_max_error(
               components(conserved_round_trip), components(conserved)
            )
         );
      }
   }
   if (worst_primitive_error > 2.0e-8_rt
       || worst_conserved_error > 5.0e-10_rt
       || worst_characteristic_error > 5.0e-15_rt
       || worst_ideal_fast_path_error != 0.0_rt) {
      std::cerr << "runtime-EOS RMHD round-trip residuals primitive="
                << worst_primitive_error
                << " conserved=" << worst_conserved_error
                << " characteristic=" << worst_characteristic_error
                << " ideal_fast_path=" << worst_ideal_fast_path_error << '\n';
      return false;
   }
   std::cout << "RMHD runtime-EOS C2P matrix primitive_error="
             << worst_primitive_error
             << " conserved_error=" << worst_conserved_error
             << " characteristic_error=" << worst_characteristic_error
             << " ideal_fast_path_error=" << worst_ideal_fast_path_error
             << '\n';
   return true;
}

pyfluid_amrex::HydroState hydro_llf_reference(
   pyfluid_amrex::HydroState const& flux_l,
   pyfluid_amrex::HydroState const& flux_r,
   pyfluid_amrex::HydroState const& state_l,
   pyfluid_amrex::HydroState const& state_r,
   amrex::Real wavespeed
)
{
   return {
      0.5_rt * (flux_l.rho + flux_r.rho) - 0.5_rt * wavespeed * (state_r.rho - state_l.rho),
      0.5_rt * (flux_l.mom1 + flux_r.mom1) - 0.5_rt * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5_rt * (flux_l.mom2 + flux_r.mom2) - 0.5_rt * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5_rt * (flux_l.mom3 + flux_r.mom3) - 0.5_rt * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5_rt * (flux_l.energy + flux_r.energy) - 0.5_rt * wavespeed * (state_r.energy - state_l.energy),
   };
}

pyfluid_amrex::HydroState hydro_hll_reference(
   pyfluid_amrex::HydroState const& flux_l,
   pyfluid_amrex::HydroState const& flux_r,
   pyfluid_amrex::HydroState const& state_l,
   pyfluid_amrex::HydroState const& state_r,
   amrex::Real s_left,
   amrex::Real s_right
)
{
   const amrex::Real inv = 1.0_rt / (s_right - s_left);
   return {
      (s_right * flux_l.rho - s_left * flux_r.rho + s_left * s_right * (state_r.rho - state_l.rho)) * inv,
      (s_right * flux_l.mom1 - s_left * flux_r.mom1 + s_left * s_right * (state_r.mom1 - state_l.mom1)) * inv,
      (s_right * flux_l.mom2 - s_left * flux_r.mom2 + s_left * s_right * (state_r.mom2 - state_l.mom2)) * inv,
      (s_right * flux_l.mom3 - s_left * flux_r.mom3 + s_left * s_right * (state_r.mom3 - state_l.mom3)) * inv,
      (s_right * flux_l.energy - s_left * flux_r.energy + s_left * s_right * (state_r.energy - state_l.energy)) * inv,
   };
}

pyfluid_amrex::MHDState mhd_llf_reference(
   pyfluid_amrex::MHDState const& flux_l,
   pyfluid_amrex::MHDState const& flux_r,
   pyfluid_amrex::MHDState const& state_l,
   pyfluid_amrex::MHDState const& state_r,
   amrex::Real wavespeed
)
{
   return {
      0.5_rt * (flux_l.rho + flux_r.rho) - 0.5_rt * wavespeed * (state_r.rho - state_l.rho),
      0.5_rt * (flux_l.mom1 + flux_r.mom1) - 0.5_rt * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5_rt * (flux_l.mom2 + flux_r.mom2) - 0.5_rt * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5_rt * (flux_l.mom3 + flux_r.mom3) - 0.5_rt * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5_rt * (flux_l.energy + flux_r.energy) - 0.5_rt * wavespeed * (state_r.energy - state_l.energy),
      0.5_rt * (flux_l.b1 + flux_r.b1) - 0.5_rt * wavespeed * (state_r.b1 - state_l.b1),
      0.5_rt * (flux_l.b2 + flux_r.b2) - 0.5_rt * wavespeed * (state_r.b2 - state_l.b2),
      0.5_rt * (flux_l.b3 + flux_r.b3) - 0.5_rt * wavespeed * (state_r.b3 - state_l.b3),
   };
}

pyfluid_amrex::MHDState mhd_hll_reference(
   pyfluid_amrex::MHDState const& flux_l,
   pyfluid_amrex::MHDState const& flux_r,
   pyfluid_amrex::MHDState const& state_l,
   pyfluid_amrex::MHDState const& state_r,
   amrex::Real s_left,
   amrex::Real s_right
)
{
   const amrex::Real inv = 1.0_rt / (s_right - s_left);
   return {
      (s_right * flux_l.rho - s_left * flux_r.rho + s_left * s_right * (state_r.rho - state_l.rho)) * inv,
      (s_right * flux_l.mom1 - s_left * flux_r.mom1 + s_left * s_right * (state_r.mom1 - state_l.mom1)) * inv,
      (s_right * flux_l.mom2 - s_left * flux_r.mom2 + s_left * s_right * (state_r.mom2 - state_l.mom2)) * inv,
      (s_right * flux_l.mom3 - s_left * flux_r.mom3 + s_left * s_right * (state_r.mom3 - state_l.mom3)) * inv,
      (s_right * flux_l.energy - s_left * flux_r.energy + s_left * s_right * (state_r.energy - state_l.energy)) * inv,
      (s_right * flux_l.b1 - s_left * flux_r.b1 + s_left * s_right * (state_r.b1 - state_l.b1)) * inv,
      (s_right * flux_l.b2 - s_left * flux_r.b2 + s_left * s_right * (state_r.b2 - state_l.b2)) * inv,
      (s_right * flux_l.b3 - s_left * flux_r.b3 + s_left * s_right * (state_r.b3 - state_l.b3)) * inv,
   };
}

amrex::Real laplacian_reference(
   int dim,
   amrex::Real center,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real inv_dx2,
   amrex::Real inv_dy2,
   amrex::Real inv_dz2
)
{
   amrex::Real value = (xm - 2.0_rt * center + xp) * inv_dx2;
   if (dim >= 2) {
      value += (ym - 2.0_rt * center + yp) * inv_dy2;
   }
   if (dim == 3) {
      value += (zm - 2.0_rt * center + zp) * inv_dz2;
   }
   return value;
}

amrex::Real relaxed_reference(
   int dim,
   amrex::Real base,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real cx,
   amrex::Real cy,
   amrex::Real cz
)
{
   amrex::Real numerator = base + cx * (xm + xp);
   amrex::Real denominator = 1.0_rt + 2.0_rt * cx;
   if (dim >= 2) {
      numerator += cy * (ym + yp);
      denominator += 2.0_rt * cy;
   }
   if (dim == 3) {
      numerator += cz * (zm + zp);
      denominator += 2.0_rt * cz;
   }
   return numerator / denominator;
}

amrex::Real residual_reference(
   int dim,
   amrex::Real base,
   amrex::Real center,
   amrex::Real xm,
   amrex::Real xp,
   amrex::Real ym,
   amrex::Real yp,
   amrex::Real zm,
   amrex::Real zp,
   amrex::Real cx,
   amrex::Real cy,
   amrex::Real cz
)
{
   amrex::Real value = center - base + cx * (2.0_rt * center - xm - xp);
   if (dim >= 2) {
      value += cy * (2.0_rt * center - ym - yp);
   }
   if (dim == 3) {
      value += cz * (2.0_rt * center - zm - zp);
   }
   return value;
}

bool validate_spacetime_constraint_region_geometry()
{
   using pyfluid_amrex::SpacetimeConstraintRegion;
   const auto point = [](amrex::Real x, amrex::Real y, amrex::Real z) {
      return amrex::GpuArray<amrex::Real, 3>{{x, y, z}};
   };
   const SpacetimeConstraintRegion disabled{};
   if (disabled.enabled() || disabled.classify(point(0.0_rt, 0.0_rt, 0.0_rt)) != -1) {
      std::cerr << "disabled fixed constraint region accepts a point\n";
      return false;
   }

   SpacetimeConstraintRegion region;
   region.extent = 3.5_rt;
   region.exclusion_radius = 2.5_rt;
   region.split_radius = 3.0_rt;
   const auto classify = [&](amrex::Real x, amrex::Real y, amrex::Real z) {
      return region.classify(point(x, y, z));
   };
   if (classify(0.0_rt, 0.0_rt, 0.0_rt) != -1
       || classify(2.5_rt - 1.0e-12_rt, 0.0_rt, 0.0_rt) != -1
       || classify(2.5_rt, 0.0_rt, 0.0_rt) != 0
       || classify(3.0_rt - 1.0e-12_rt, 0.0_rt, 0.0_rt) != 0
       || classify(3.0_rt, 0.0_rt, 0.0_rt) != 1
       || classify(3.5_rt, 0.0_rt, 0.0_rt) != 1
       || classify(3.5_rt + 1.0e-12_rt, 0.0_rt, 0.0_rt) != -1
       || classify(3.5_rt, 3.5_rt, 3.5_rt) != 1) {
      std::cerr << "fixed constraint region boundary classification mismatch\n";
      return false;
   }

   SpacetimeConstraintRegion shifted = region;
   shifted.center = point(1.25_rt, -0.75_rt, 0.5_rt);
   for (const auto& displacement : std::array<amrex::GpuArray<amrex::Real, 3>, 6>{{
            point(0.0_rt, 0.0_rt, 0.0_rt), point(2.5_rt, 0.0_rt, 0.0_rt),
            point(2.75_rt, 0.0_rt, 0.0_rt), point(3.0_rt, 0.0_rt, 0.0_rt),
            point(3.5_rt, 0.0_rt, 0.0_rt), point(3.5_rt, 2.0_rt, -1.0_rt)
         }}) {
      const int unshifted = region.classify(displacement);
      const int translated = shifted.classify(point(
         shifted.center[0] + displacement[0],
         shifted.center[1] + displacement[1],
         shifted.center[2] + displacement[2]
      ));
      if (unshifted != translated) {
         std::cerr << "fixed constraint region is not coordinate-shift invariant\n";
         return false;
      }
   }

   struct RegionIntegral {
      amrex::Real volume = 0.0_rt;
      amrex::Real hamiltonian_squared = 0.0_rt;
      int all = 0;
      int near = 0;
      int far = 0;
   };
   const auto integrate_domain = [&](int cells, bool upper_half) {
      const amrex::Real spacing = 8.0_rt / static_cast<amrex::Real>(cells);
      const amrex::Real weight = spacing * spacing * spacing;
      RegionIntegral integral;
      // This is the z >= 0 half of the Cartesian convergence domain. The
      // predicate itself has no dx input; N32 nodes coincide with even N64.
      const int k_begin = upper_half ? cells / 2 : 0;
      for (int k = k_begin; k <= cells; ++k) {
         const amrex::Real z = -4.0_rt + spacing * k;
         for (int j = 0; j <= cells; ++j) {
            const amrex::Real y = -4.0_rt + spacing * j;
            for (int i = 0; i <= cells; ++i) {
               const amrex::Real x = -4.0_rt + spacing * i;
               const int classification = region.classify(point(x, y, z));
               if (classification < 0) continue;
               const amrex::Real hamiltonian = 1.0_rt
                  + 0.01_rt * (x * x + y * y + z * z);
               integral.volume += weight;
               integral.hamiltonian_squared += weight * hamiltonian * hamiltonian;
               ++integral.all;
               if (classification == 0) ++integral.near;
               if (classification == 1) ++integral.far;
            }
         }
      }
      return integral;
   };
   const RegionIntegral n32 = integrate_domain(32, true);
   const RegionIntegral n64 = integrate_domain(64, true);
   const RegionIntegral n32_full = integrate_domain(32, false);
   if (n32.all != n32.near + n32.far || n64.all != n64.near + n64.far
       || n32.near == 0 || n32.far == 0 || n64.near == 0 || n64.far == 0) {
      std::cerr << "fixed constraint region does not partition near/far support\n";
      return false;
   }
   for (int k = 0; k <= 32; ++k) {
      for (int j = 0; j <= 32; ++j) {
         for (int i = 0; i <= 32; ++i) {
            const amrex::GpuArray<amrex::Real, 3> coarse_point{{
               -4.0_rt + 0.25_rt * i, -4.0_rt + 0.25_rt * j, -4.0_rt + 0.25_rt * k
            }};
            const amrex::GpuArray<amrex::Real, 3> fine_point{{
               -4.0_rt + 0.125_rt * (2 * i), -4.0_rt + 0.125_rt * (2 * j),
               -4.0_rt + 0.125_rt * (2 * k)
            }};
            if (region.classify(coarse_point) != region.classify(fine_point)) {
               std::cerr << "fixed constraint region depends on dx at coincident nodes\n";
               return false;
            }
         }
      }
   }
   const amrex::Real n32_l2 = std::sqrt(n32.hamiltonian_squared / n32.volume);
   const amrex::Real n64_l2 = std::sqrt(n64.hamiltonian_squared / n64.volume);
   const amrex::Real volume_difference = std::abs(n64.volume - n32.volume)
      / std::max(n64.volume, n32.volume);
   if (volume_difference > 0.12_rt || std::abs(n64_l2 - n32_l2) > 0.01_rt) {
      std::cerr << "fixed constraint-region coordinate norm does not converge under refinement\n";
      return false;
   }
   const amrex::Real half_fraction = n32.volume / n32_full.volume;
   if (std::abs(half_fraction - 0.5_rt) > 0.03_rt) {
      std::cerr << "fixed constraint region does not retain the physical half-space\n";
      return false;
   }

   // Reflection pairs have identical classification. Consequently, sampling
   // only z >= 0 retains the intended physical half-space, rather than a
   // resolution-dependent geometric region.
   for (int k = 1; k <= 14; ++k) {
      const amrex::Real z = 0.25_rt * k;
      for (int j = -14; j <= 14; ++j) {
         for (int i = -14; i <= 14; ++i) {
            if (classify(0.25_rt * i, 0.25_rt * j, z)
                != classify(0.25_rt * i, 0.25_rt * j, -z)) {
               std::cerr << "fixed constraint region breaks half-space reflection symmetry\n";
               return false;
            }
         }
      }
   }
   return true;
}

bool validate_z4c_constraints()
{
   using pyfluid_amrex::Z4cSpatialDerivatives;
   using pyfluid_amrex::Z4cState;

   const Z4cState minkowski{};
   const Z4cSpatialDerivatives zero_derivatives{};
   Z4cState unprojected = minkowski;
   unprojected.chi = 0.8_rt;
   unprojected.gamma_tilde11 = 2.0_rt;
   unprojected.gamma_tilde22 = 3.0_rt;
   unprojected.gamma_tilde33 = 4.0_rt;
   unprojected.a_tilde11 = 0.3_rt;
   unprojected.a_tilde22 = -0.1_rt;
   unprojected.a_tilde33 = 0.2_rt;
   const Z4cState projected =
      pyfluid_amrex::generated::z4c_project_algebraic_constraints(unprojected);
   const auto projected_metric =
      pyfluid_amrex::generated::z4c_conformal_metric(projected);
   const auto projected_constraints =
      pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
         projected, zero_derivatives
      );
   if (!close_enough(projected_metric.determinant, 1.0_rt)
       || !close_enough(projected_constraints.trace_a_tilde, 0.0_rt)
       || !close_enough(
          projected.gamma_tilde11 / projected.chi,
          unprojected.gamma_tilde11 / unprojected.chi
       )) {
      std::cerr << "generated Z4c algebraic projection mismatch\n";
      return 1;
   }
   const auto flat = pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
      minkowski, zero_derivatives
   );
   if (!close_enough(flat.theta, 0.0_rt)
       || !close_enough(flat.z_tilde1, 0.0_rt)
       || !close_enough(flat.z_tilde2, 0.0_rt)
       || !close_enough(flat.z_tilde3, 0.0_rt)
       || !close_enough(flat.hamiltonian, 0.0_rt)
       || !close_enough(flat.momentum1, 0.0_rt)
       || !close_enough(flat.momentum2, 0.0_rt)
       || !close_enough(flat.momentum3, 0.0_rt)
       || !close_enough(flat.log_conformal_determinant, 0.0_rt)
       || !close_enough(flat.trace_a_tilde, 0.0_rt)) {
      std::cerr << "generated Z4c Minkowski constraints are nonzero\n";
      return false;
   }
   const auto flat_metric = pyfluid_amrex::generated::z4c_conformal_metric(minkowski);
   const auto flat_spacetime_metric =
      pyfluid_amrex::generated::z4c_spacetime_metric(minkowski);
   const auto flat_spacetime_derivatives =
      pyfluid_amrex::generated::z4c_spacetime_metric_derivatives(
         minkowski, zero_derivatives
      );
   if (!close_enough(flat_spacetime_metric.lapse, 1.0_rt)
       || !close_enough(flat_spacetime_metric.gamma11, 1.0_rt)
       || !close_enough(flat_spacetime_metric.gamma22, 1.0_rt)
       || !close_enough(flat_spacetime_metric.gamma33, 1.0_rt)
       || !close_enough(flat_spacetime_metric.inv_gamma11, 1.0_rt)
       || !close_enough(flat_spacetime_metric.sqrt_det_gamma, 1.0_rt)
       || !close_enough(flat_spacetime_derivatives.d1_lapse, 0.0_rt)
       || !close_enough(flat_spacetime_derivatives.d1_gamma11, 0.0_rt)
       || !close_enough(flat_spacetime_derivatives.extrinsic11, 0.0_rt)) {
      std::cerr << "generated Z4c-to-Valencia Minkowski bridge mismatch\n";
      return false;
   }
   Z4cState bridge_state = projected;
   bridge_state.k_hat = 0.21_rt;
   bridge_state.theta = -0.015_rt;
   Z4cSpatialDerivatives bridge_derivatives{};
   bridge_derivatives.d_chi[0] = 0.07_rt;
   bridge_derivatives.d_gamma_tilde[0][0] = -0.03_rt;
   bridge_derivatives.d_lapse[0] = 0.02_rt;
   bridge_derivatives.d_shift[0][1] = -0.04_rt;
   const auto bridge_metric =
      pyfluid_amrex::generated::z4c_spacetime_metric(bridge_state);
   const auto bridge_metric_derivatives =
      pyfluid_amrex::generated::z4c_spacetime_metric_derivatives(
         bridge_state, bridge_derivatives
      );
   const amrex::Real expected_d1_gamma11 =
      bridge_derivatives.d_gamma_tilde[0][0] / bridge_state.chi
      - bridge_state.gamma_tilde11 * bridge_derivatives.d_chi[0]
         / (bridge_state.chi * bridge_state.chi);
   const amrex::Real total_curvature =
      bridge_state.k_hat + 2.0_rt * bridge_state.theta;
   const amrex::Real expected_extrinsic11 =
      bridge_state.a_tilde11 / bridge_state.chi
      + bridge_metric.gamma11 * total_curvature / 3.0_rt;
   if (!close_enough(bridge_metric.gamma11, projected_metric.gamma11)
       || !close_enough(bridge_metric.inv_gamma23, projected_metric.inv_gamma23)
       || !close_enough(
          bridge_metric.sqrt_det_gamma, projected_metric.sqrt_det_gamma
       )
       || !close_enough(bridge_metric_derivatives.d1_lapse, 0.02_rt)
       || !close_enough(bridge_metric_derivatives.d1_shift2, -0.04_rt)
       || !close_enough(
          bridge_metric_derivatives.d1_gamma11, expected_d1_gamma11
       )
       || !close_enough(
          bridge_metric_derivatives.extrinsic11, expected_extrinsic11
       )) {
      std::cerr << "generated Z4c-to-Valencia metric bridge mismatch\n";
      return false;
   }
   const auto flat_connections = pyfluid_amrex::generated::z4c_connections(
      minkowski, zero_derivatives, flat_metric
   );
   const auto flat_diagnostic_geometry =
      pyfluid_amrex::generated::z4c_diagnostic_geometry(
         minkowski, zero_derivatives, flat_metric, flat_connections
      );
   if (!close_enough(flat_diagnostic_geometry.ricci_scalar, 0.0_rt)) {
      std::cerr << "generated Z4c Minkowski diagnostic geometry is nonzero\n";
      return false;
   }
   const auto flat_evolution_geometry = pyfluid_amrex::generated::z4c_evolution_geometry(
      minkowski, zero_derivatives, flat_metric, flat_connections
   );
   const auto flat_rhs = pyfluid_amrex::generated::z4c_geometric_rhs(
      minkowski,
      zero_derivatives,
      flat_metric,
      flat_connections,
      flat_evolution_geometry,
      pyfluid_amrex::vacuum_z4c_matter(),
      0.02_rt,
      0.0_rt
   );
   const std::array<amrex::Real, 18> flat_rhs_components = {
      flat_rhs.chi,
      flat_rhs.gamma_tilde11,
      flat_rhs.gamma_tilde12,
      flat_rhs.gamma_tilde13,
      flat_rhs.gamma_tilde22,
      flat_rhs.gamma_tilde23,
      flat_rhs.gamma_tilde33,
      flat_rhs.k_hat,
      flat_rhs.a_tilde11,
      flat_rhs.a_tilde12,
      flat_rhs.a_tilde13,
      flat_rhs.a_tilde22,
      flat_rhs.a_tilde23,
      flat_rhs.a_tilde33,
      flat_rhs.gamma_tilde1,
      flat_rhs.gamma_tilde2,
      flat_rhs.gamma_tilde3,
      flat_rhs.theta,
   };
   for (const amrex::Real value : flat_rhs_components) {
      if (!close_enough(value, 0.0_rt)) {
         std::cerr << "generated Z4c Minkowski RHS is nonzero\n";
         return false;
      }
   }
   if (!close_enough(
          pyfluid_amrex::generated::z4c_harmonic_lapse_rhs(minkowski, zero_derivatives),
          0.0_rt
       )
       || !close_enough(
          pyfluid_amrex::generated::z4c_one_plus_log_lapse_rhs(minkowski, zero_derivatives),
          0.0_rt
       )) {
      std::cerr << "generated Z4c Minkowski lapse RHS is nonzero\n";
      return false;
   }
   const auto flat_shift_rhs = pyfluid_amrex::generated::z4c_gamma_driver_shift_rhs(
      minkowski, zero_derivatives, 1.0_rt, 0.5_rt
   );
   if (!close_enough(flat_shift_rhs.shift1, 0.0_rt)
       || !close_enough(flat_shift_rhs.shift2, 0.0_rt)
       || !close_enough(flat_shift_rhs.shift3, 0.0_rt)) {
      std::cerr << "generated Z4c Minkowski shift RHS is nonzero\n";
      return false;
   }

   // gamma_ij = psi^4 delta_ij has R = -8 psi^-5 Laplacian(psi).
   constexpr amrex::Real curvature = 0.03_rt;
   constexpr amrex::Real x = 0.11_rt;
   constexpr amrex::Real y = -0.07_rt;
   constexpr amrex::Real z = 0.05_rt;
   const amrex::Real radius_squared = x * x + y * y + z * z;
   const amrex::Real psi = 1.0_rt + curvature * radius_squared;
   Z4cState conformally_flat{};
   conformally_flat.chi = std::pow(psi, -4.0_rt);
   Z4cSpatialDerivatives derivatives{};
   const amrex::Real d_psi[3] = {
      2.0_rt * curvature * x,
      2.0_rt * curvature * y,
      2.0_rt * curvature * z,
   };
   for (int first = 0; first < 3; ++first) {
      derivatives.d_chi[first] = -4.0_rt * std::pow(psi, -5.0_rt) * d_psi[first];
      for (int second = 0; second < 3; ++second) {
         const amrex::Real dd_psi = first == second ? 2.0_rt * curvature : 0.0_rt;
         derivatives.dd_chi[first][second] =
            20.0_rt * std::pow(psi, -6.0_rt) * d_psi[first] * d_psi[second]
            - 4.0_rt * std::pow(psi, -5.0_rt) * dd_psi;
      }
   }
   const auto curved = pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
      conformally_flat, derivatives
   );
   const auto curved_metric = pyfluid_amrex::generated::z4c_conformal_metric(
      conformally_flat
   );
   const auto curved_connections = pyfluid_amrex::generated::z4c_connections(
      conformally_flat, derivatives, curved_metric
   );
   const auto curved_diagnostic_geometry =
      pyfluid_amrex::generated::z4c_diagnostic_geometry(
         conformally_flat, derivatives, curved_metric, curved_connections
      );
   const auto curved_evolution_geometry = pyfluid_amrex::generated::z4c_evolution_geometry(
      conformally_flat, derivatives, curved_metric, curved_connections
   );
   const amrex::Real expected_ricci = -48.0_rt * curvature * std::pow(psi, -5.0_rt);
   if (std::abs(curved.hamiltonian - expected_ricci) > 2.0e-13_rt
       || std::abs(curved_diagnostic_geometry.ricci_scalar - expected_ricci) > 2.0e-13_rt
       || std::abs(curved_evolution_geometry.ricci_scalar - expected_ricci) > 2.0e-13_rt
       || !close_enough(curved.log_conformal_determinant, 0.0_rt)
       || !close_enough(curved.trace_a_tilde, 0.0_rt)) {
      std::cerr << "generated Z4c conformally-flat curvature mismatch: "
                << curved.hamiltonian << " != " << expected_ricci << '\n';
      return false;
   }

   // det(gamma_tilde)=1 with gamma_tilde=diag(exp(2 f), exp(-2 f), 1)
   // and f=a*x exercises the evolved-Gamma form of the conformal Ricci tensor.
   constexpr amrex::Real slope = 0.17_rt;
   constexpr amrex::Real f_value = 0.04_rt;
   const amrex::Real exp_two_f = std::exp(2.0_rt * f_value);
   const amrex::Real exp_minus_two_f = std::exp(-2.0_rt * f_value);
   Z4cState anisotropic{};
   anisotropic.gamma_tilde11 = exp_two_f;
   anisotropic.gamma_tilde22 = exp_minus_two_f;
   anisotropic.gamma_tilde1 = 2.0_rt * slope * exp_minus_two_f;
   Z4cSpatialDerivatives anisotropic_derivatives{};
   anisotropic_derivatives.d_gamma_tilde[0][0] = 2.0_rt * slope * exp_two_f;
   anisotropic_derivatives.d_gamma_tilde[0][3] = -2.0_rt * slope * exp_minus_two_f;
   anisotropic_derivatives.dd_gamma_tilde[0][0][0] =
      4.0_rt * slope * slope * exp_two_f;
   anisotropic_derivatives.dd_gamma_tilde[0][0][3] =
      4.0_rt * slope * slope * exp_minus_two_f;
   anisotropic_derivatives.d_gamma_tilde_vector[0][0] =
      -4.0_rt * slope * slope * exp_minus_two_f;
   const auto anisotropic_metric = pyfluid_amrex::generated::z4c_conformal_metric(
      anisotropic
   );
   const auto anisotropic_connections = pyfluid_amrex::generated::z4c_connections(
      anisotropic, anisotropic_derivatives, anisotropic_metric
   );
   const auto anisotropic_diagnostic_geometry =
      pyfluid_amrex::generated::z4c_diagnostic_geometry(
         anisotropic,
         anisotropic_derivatives,
         anisotropic_metric,
         anisotropic_connections
      );
   const auto anisotropic_evolution_geometry =
      pyfluid_amrex::generated::z4c_evolution_geometry(
         anisotropic,
         anisotropic_derivatives,
         anisotropic_metric,
         anisotropic_connections
      );
   const std::array<amrex::Real, 7> diagnostic_ricci = {
      anisotropic_diagnostic_geometry.ricci11,
      anisotropic_diagnostic_geometry.ricci12,
      anisotropic_diagnostic_geometry.ricci13,
      anisotropic_diagnostic_geometry.ricci22,
      anisotropic_diagnostic_geometry.ricci23,
      anisotropic_diagnostic_geometry.ricci33,
      anisotropic_diagnostic_geometry.ricci_scalar,
   };
   const std::array<amrex::Real, 7> evolution_ricci = {
      anisotropic_evolution_geometry.ricci11,
      anisotropic_evolution_geometry.ricci12,
      anisotropic_evolution_geometry.ricci13,
      anisotropic_evolution_geometry.ricci22,
      anisotropic_evolution_geometry.ricci23,
      anisotropic_evolution_geometry.ricci33,
      anisotropic_evolution_geometry.ricci_scalar,
   };
   for (std::size_t component = 0; component < diagnostic_ricci.size(); ++component) {
      if (std::abs(diagnostic_ricci[component] - evolution_ricci[component]) > 3.0e-13_rt) {
         std::cerr << "generated Z4c evolved-Gamma Ricci mismatch\n";
         return false;
      }
   }
   const auto anisotropic_constraints =
      pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
         anisotropic, anisotropic_derivatives
      );
   if (std::abs(anisotropic_constraints.z_tilde1) > 2.0e-15_rt
       || std::abs(anisotropic_constraints.z_tilde2) > 2.0e-15_rt
       || std::abs(anisotropic_constraints.z_tilde3) > 2.0e-15_rt) {
      std::cerr << "generated Z4c evolved-Gamma constraint mismatch\n";
      return false;
   }

   pyfluid_amrex::Z4cMatter matter{};
   matter.rho = 0.02_rt;
   matter.momentum1 = 0.003_rt;
   matter.momentum2 = -0.004_rt;
   matter.momentum3 = 0.005_rt;
   matter.stress11 = 0.011_rt;
   matter.stress12 = -0.002_rt;
   matter.stress13 = 0.004_rt;
   matter.stress22 = 0.017_rt;
   matter.stress23 = -0.003_rt;
   matter.stress33 = 0.023_rt;
   const auto sourced = pyfluid_amrex::generated::evaluate_z4c_constraints(
      minkowski, zero_derivatives, matter
   );
   if (!close_enough(sourced.hamiltonian, -16.0_rt * pyfluid_amrex::constants::pi * matter.rho)
       || !close_enough(sourced.momentum1, -8.0_rt * pyfluid_amrex::constants::pi * matter.momentum1)
       || !close_enough(sourced.momentum2, -8.0_rt * pyfluid_amrex::constants::pi * matter.momentum2)
       || !close_enough(sourced.momentum3, -8.0_rt * pyfluid_amrex::constants::pi * matter.momentum3)) {
      std::cerr << "generated Z4c matter constraints mismatch\n";
      return false;
   }

   const auto sourced_rhs = pyfluid_amrex::generated::z4c_geometric_rhs(
      minkowski,
      zero_derivatives,
      flat_metric,
      flat_connections,
      flat_evolution_geometry,
      matter,
      0.0_rt,
      0.0_rt
   );
   const amrex::Real stress_trace = matter.stress11 + matter.stress22 + matter.stress33;
   if (!close_enough(sourced_rhs.k_hat, 4.0_rt * pyfluid_amrex::constants::pi * (matter.rho + stress_trace))
       || !close_enough(
          sourced_rhs.a_tilde11,
          -8.0_rt * pyfluid_amrex::constants::pi * (matter.stress11 - stress_trace / 3.0_rt)
       )
       || !close_enough(sourced_rhs.a_tilde12, -8.0_rt * pyfluid_amrex::constants::pi * matter.stress12)
       || !close_enough(sourced_rhs.a_tilde13, -8.0_rt * pyfluid_amrex::constants::pi * matter.stress13)
       || !close_enough(
          sourced_rhs.a_tilde22,
          -8.0_rt * pyfluid_amrex::constants::pi * (matter.stress22 - stress_trace / 3.0_rt)
       )
       || !close_enough(sourced_rhs.a_tilde23, -8.0_rt * pyfluid_amrex::constants::pi * matter.stress23)
       || !close_enough(
          sourced_rhs.a_tilde33,
          -8.0_rt * pyfluid_amrex::constants::pi * (matter.stress33 - stress_trace / 3.0_rt)
       )
       || !close_enough(sourced_rhs.gamma_tilde1, -16.0_rt * pyfluid_amrex::constants::pi * matter.momentum1)
       || !close_enough(sourced_rhs.gamma_tilde2, -16.0_rt * pyfluid_amrex::constants::pi * matter.momentum2)
       || !close_enough(sourced_rhs.gamma_tilde3, -16.0_rt * pyfluid_amrex::constants::pi * matter.momentum3)
       || !close_enough(sourced_rhs.theta, -8.0_rt * pyfluid_amrex::constants::pi * matter.rho)) {
      std::cerr << "generated Z4c matter RHS mismatch\n";
      return false;
   }

   Z4cState gauge_state{};
   gauge_state.k_hat = 0.2_rt;
   gauge_state.lapse = 0.8_rt;
   gauge_state.shift1 = 0.1_rt;
   gauge_state.shift2 = -0.2_rt;
   gauge_state.shift3 = 0.3_rt;
   gauge_state.gamma_tilde1 = 0.04_rt;
   gauge_state.gamma_tilde2 = -0.03_rt;
   gauge_state.gamma_tilde3 = 0.02_rt;
   Z4cSpatialDerivatives gauge_derivatives{};
   gauge_derivatives.advection_lapse[0] = 0.05_rt;
   gauge_derivatives.advection_lapse[1] = -0.04_rt;
   gauge_derivatives.advection_lapse[2] = 0.02_rt;
   for (int direction = 0; direction < 3; ++direction) {
      for (int component = 0; component < 3; ++component) {
         gauge_derivatives.advection_shift[direction][component] =
            0.01_rt * static_cast<amrex::Real>(1 + 3 * direction + component);
      }
   }
   const amrex::Real lapse_advection =
      gauge_state.shift1 * gauge_derivatives.advection_lapse[0]
      + gauge_state.shift2 * gauge_derivatives.advection_lapse[1]
      + gauge_state.shift3 * gauge_derivatives.advection_lapse[2];
   if (!close_enough(
          pyfluid_amrex::generated::z4c_harmonic_lapse_rhs(
             gauge_state, gauge_derivatives
          ),
          -gauge_state.lapse * gauge_state.lapse * gauge_state.k_hat + lapse_advection
       )
       || !close_enough(
          pyfluid_amrex::generated::z4c_one_plus_log_lapse_rhs(
             gauge_state, gauge_derivatives
          ),
          -2.0_rt * gauge_state.lapse * gauge_state.k_hat + lapse_advection
       )) {
      std::cerr << "generated Z4c lapse gauge mismatch\n";
      return false;
   }
   constexpr amrex::Real gauge_time = 0.7_rt;
   constexpr amrex::Real slow_start_strength = 0.6_rt;
   constexpr amrex::Real slow_start_width = 2.5_rt;
   const amrex::Real conformal_weight = std::sqrt(gauge_state.chi);
   const amrex::Real expected_slow_start =
      -2.0_rt * gauge_state.lapse * gauge_state.k_hat + lapse_advection
      - conformal_weight * slow_start_strength
         * std::exp(
            -gauge_time * gauge_time
            / (2.0_rt * slow_start_width * slow_start_width)
         )
         * (gauge_state.lapse - conformal_weight);
   if (!close_enough(
          pyfluid_amrex::generated::z4c_slow_start_lapse_rhs(
             gauge_state,
             gauge_derivatives,
             gauge_time,
             slow_start_strength,
             slow_start_width
          ),
          expected_slow_start
       )) {
      std::cerr << "generated Z4c slow-start lapse mismatch\n";
      return false;
   }
   constexpr amrex::Real shift_mu = 0.75_rt;
   constexpr amrex::Real shift_eta = 0.12_rt;
   const auto shift_rhs = pyfluid_amrex::generated::z4c_gamma_driver_shift_rhs(
      gauge_state, gauge_derivatives, shift_mu, shift_eta
   );
   const amrex::Real shift[3] = {
      gauge_state.shift1,
      gauge_state.shift2,
      gauge_state.shift3,
   };
   const amrex::Real gamma_tilde[3] = {
      gauge_state.gamma_tilde1,
      gauge_state.gamma_tilde2,
      gauge_state.gamma_tilde3,
   };
   const amrex::Real actual_shift_rhs[3] = {
      shift_rhs.shift1,
      shift_rhs.shift2,
      shift_rhs.shift3,
   };
   for (int component = 0; component < 3; ++component) {
      amrex::Real expected = gauge_state.lapse * gauge_state.lapse
         * shift_mu * gamma_tilde[component] - shift_eta * shift[component];
      for (int direction = 0; direction < 3; ++direction) {
         expected += shift[direction]
            * gauge_derivatives.advection_shift[direction][component];
      }
      if (!close_enough(actual_shift_rhs[component], expected)) {
         std::cerr << "generated Z4c shift gauge mismatch\n";
         return false;
      }
   }
   return true;
}

} // namespace

bool validate_quintic_interpolation()
{
   const amrex::Real left = 0.3_rt;
   const amrex::Real right = 1.7_rt;
   for (int degree = 0; degree <= 5; ++degree) {
      const auto value = [=](amrex::Real x) { return std::pow(x, degree); };
      const auto first = [=](amrex::Real x) { return degree == 0 ? 0.0_rt : degree * std::pow(x, degree - 1); };
      const auto second = [=](amrex::Real x) { return degree < 2 ? 0.0_rt : degree * (degree - 1) * std::pow(x, degree - 2); };
      for (amrex::Real fraction : {0.0_rt, 0.1_rt, 0.35_rt, 0.9_rt, 1.0_rt}) {
         const amrex::Real actual = pyfluid_amrex::quintic_hermite(fraction, right - left,
            value(left), value(right), first(left), first(right), second(left), second(right));
         if (std::abs(actual - value(left + fraction * (right - left))) > 2.0e-13_rt) {
            std::cerr << "quintic interpolation failed polynomial degree " << degree << '\n';
            return false;
         }
      }
   }
   return true;
}

bool validate_fccz4_exact_points()
{
   using namespace pyfluid_amrex;
   Fccz4Point point{};
   point.chi.value = point.lapse.value = 1.0_rt;
   for (int component : {0, 3, 5}) point.metric.value[component] = 1.0_rt;
   const Fccz4Damping damping{0.1_rt, 0.0_rt, 1.0_rt};
   const auto check = [&](EulerianMatter const& matter, amrex::Real expected_chi, amrex::Real expected_k) {
      const auto geometry = generated::fccz4_geometry(point);
      const auto rhs = generated::fccz4_rhs(point, matter, damping, geometry);
      amrex::Real error = std::max(std::abs(rhs.chi - expected_chi), std::abs(rhs.trace_k - expected_k));
      error = std::max(error, std::abs(rhs.theta));
      const auto constraints = generated::fccz4_constraints(point, matter, geometry);
      error = std::max(error, std::abs(constraints.hamiltonian));
      for (int direction = 0; direction < 3; ++direction) {
         error = std::max(error, std::abs(constraints.momentum[direction]));
      }
      for (int component = 0; component < 6; ++component) {
         error = std::max(error, std::abs(rhs.metric[component]));
         error = std::max(error, std::abs(rhs.a_tilde[component]));
      }
      for (int direction = 0; direction < 3; ++direction) error = std::max(error, std::abs(rhs.connection[direction]));
      return std::isfinite(error) && error < 1.0e-12_rt;
   };
   if (!check({}, 0.0_rt, 0.0_rt)) return false;
   const amrex::Real hubble = 0.2_rt;
   EulerianMatter matter{};
   matter.rho = 3.0_rt * hubble * hubble / (8.0_rt * constants::pi);
   point.trace_k.value = -3.0_rt * hubble;
   if (!check(matter, -2.0_rt * hubble, 12.0_rt * constants::pi * matter.rho)) return false;
   point.trace_k.value = 0.0_rt;
   const amrex::Real radius = 2.0_rt;
   const amrex::Real psi = 1.0_rt + 0.5_rt / radius;
   const amrex::Real d_psi = -0.5_rt / (radius * radius);
   const amrex::Real dd_psi = 1.0_rt / std::pow(radius, 3.0_rt);
   ReferenceScalarJet chi{};
   chi.value = std::pow(psi, -4);
   chi.first[0] = -4.0_rt * std::pow(psi, -5) * d_psi;
   chi.second[0][0] = 20.0_rt * std::pow(psi, -6) * d_psi * d_psi - 4.0_rt * std::pow(psi, -5) * dd_psi;
   ReferenceScalarJet lapse{};
   lapse.value = (1.0_rt - 0.5_rt / radius) / psi;
   lapse.first[0] = 1.0_rt / (radius * radius * psi * psi);
   lapse.second[0][0] = -2.0_rt / (std::pow(radius, 3.0_rt) * psi * psi) + 1.0_rt / (std::pow(radius, 4.0_rt) * std::pow(psi, 3));
   point.chi = generated::spherical_reference_scalar_jet(radius, 0.73_rt, 0.2_rt, chi);
   point.lapse = generated::spherical_reference_scalar_jet(radius, 0.73_rt, 0.2_rt, lapse);
   return check({}, 0.0_rt, 0.0_rt);
}

bool validate_reference_spacetime_metric_jets()
{
   using pyfluid_amrex::SymmetricTensorJet;
   using namespace pyfluid_amrex::generated;
   const amrex::Real radius = 2.0_rt;
   SymmetricTensorJet jet{};
   for (int component : {0, 3, 5}) {
      jet.value[component] = radius * radius * radius;
      jet.first[0][component] = 3.0_rt * radius * radius;
      jet.second[0][0][component] = 6.0_rt * radius;
   }
   const auto result = spherical_reference_metric_jet(radius, 0.73_rt, 0.41_rt, jet);
   const auto connection = reference_metric_connection(result);
   amrex::Real error = 0.0_rt;
   for (int i = 0; i < 3; ++i) {
      const amrex::Real expected = i == 0 ? -1.5_rt / std::pow(radius, 4.0_rt) : 0.0_rt;
      error = std::max(error, std::abs(connection.contracted[i] - expected));
   }
   for (int component = 0; component < 6; ++component) {
      const bool diagonal = component == 0 || component == 3 || component == 5;
      for (int first = 0; first < 3; ++first) {
         const amrex::Real expected_first = diagonal && first == 0 ? 3.0_rt * radius * radius : 0.0_rt;
         error = std::max(error, std::abs(result.first[first][component] - expected_first));
         for (int second = 0; second < 3; ++second) {
            const amrex::Real expected_second = diagonal && first == second
               ? (first == 0 ? 6.0_rt : 3.0_rt) * radius : 0.0_rt;
            error = std::max(error, std::abs(result.second[first][second][component] - expected_second));
         }
      }
   }
   const auto coordinate = spherical_coordinate_metric_jet(radius, 0.73_rt, 0.41_rt, jet);
   SymmetricTensorJet conformal{};
   pyfluid_amrex::ReferenceVectorGradient shift{};
   for (int component : {0, 3, 5}) {
      conformal.value[component] = 1.0_rt;
   }
   for (int direction = 0; direction < 3; ++direction) {
      shift.first[direction][direction] = 0.25_rt;
   }
   const amrex::Real a_tilde[6] = {1.0_rt, 0.0_rt, 0.0_rt, -1.0_rt, 0.0_rt, 0.0_rt};
   const amrex::Real d_chi[3] = {};
   const auto rhs = reference_conformal_metric_rhs(conformal, a_tilde, 1.0_rt, d_chi, 0.5_rt, 3.0_rt, shift);
   error = std::max(error, std::abs(rhs.chi - 0.5_rt));
   for (int component = 0; component < 6; ++component) {
      error = std::max(error, std::abs(rhs.metric[component] + a_tilde[component]));
   }
   // g_theta_theta = r^5 even though all theta derivatives of the input vanish.
   error = std::max(error, std::abs(coordinate.value[3] - std::pow(radius, 5.0_rt)));
   error = std::max(error, std::abs(coordinate.first[0][3] - 5.0_rt * std::pow(radius, 4.0_rt)));
   error = std::max(error, std::abs(coordinate.second[0][0][3] - 20.0_rt * std::pow(radius, 3.0_rt)));
   if (!std::isfinite(error) || error > 1.0e-12_rt) {
      std::cerr << "reference spacetime metric jet error=" << error << '\n';
      return false;
   }
   return true;
}

int main(int argc, char** argv)
{
   if (argc == 2 && std::string_view(argv[1]) == "--stencils-only") {
      return check_catalogued_derivative_dispatch() ? 0 : 1;
   }
   if (argc == 2 && std::string_view(argv[1]) == "--eos-presets-only") {
      return validate_read_piecewise_polytropes() ? 0 : 1;
   }
   if (argc != 1) {
      std::cerr << "usage: codegen_check [--stencils-only|--eos-presets-only]\n";
      return 2;
   }
   if (!validate_quintic_interpolation()
       || !validate_relativistic_vacuum_algebra()
       || !validate_fccz4_exact_points()
       || !validate_reference_spacetime_metric_jets()
       || !validate_reference_metric_valencia_basis()
       || !validate_reference_metric_fixed_spacetime_geometry()
       || !validate_newtonian_round_trips()
       || !validate_rhd_c2p_round_trip_matrix()
       || !validate_piecewise_polytrope()
       || !validate_read_piecewise_polytropes()
       || !validate_initial_atmosphere_profiles()
       || !validate_positive_spacetime_transfer()
       || !validate_srhd_round_trips()
       || !validate_srmhd_round_trips()
       || !validate_runtime_eos_rmhd_round_trips()
       || !validate_fixed_spacetime_provider()
       || !validate_static_trumpet_stationarity()
       || !validate_z4c_pointwise_bridge()
       || !validate_spacetime_constraint_region_geometry()
       || !validate_z4c_constraints()) {
      return 1;
   }
   static_assert(pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::HD,
      pyfluid_amrex::SpacetimeEvolution::None
   ));
   static_assert(pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::RHD,
      pyfluid_amrex::SpacetimeEvolution::Fixed
   ));
   static_assert(!pyfluid_amrex::valid_spacetime_evolution(
      pyfluid_amrex::PhysicalSystem::RHD,
      pyfluid_amrex::SpacetimeEvolution::None
   ));
   static_assert(pyfluid_amrex::ideal_rmhd_c2p_policy[0]
                 == pyfluid_amrex::C2PMethod::RePrimAnd);
   static_assert(pyfluid_amrex::ideal_rmhd_c2p_policy[1]
                 == pyfluid_amrex::C2PMethod::Palenzuela1D);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::RePrimAnd
   ).implemented);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Noble2D
   ).requires_seed);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Palenzuela1DEntropy
   ).requires_entropy);
   static_assert(pyfluid_amrex::c2p_method_capabilities(
      pyfluid_amrex::C2PMethod::Font1D
   ).requires_cold_equation_of_state);
   {
      const pyfluid_amrex::RHDPrimitive primitive{1.3_rt, 0.2_rt, 0.4_rt, 0.2_rt, -0.1_rt, 0.05_rt};
      const pyfluid_amrex::SpacetimeMetric metric{};
      const amrex::Real velocity_squared =
         primitive.v1 * primitive.v1 + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0_rt / std::sqrt(1.0_rt - velocity_squared);
      const amrex::Real enthalpy = 1.0_rt + primitive.eps + primitive.pressure / primitive.rho;
      const pyfluid_amrex::RHDConserved expected{
         primitive.rho * lorentz_factor,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v1,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v2,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor * primitive.v3,
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor
            - primitive.pressure - primitive.rho * lorentz_factor,
      };
      const auto conserved = pyfluid_amrex::generated::rhd_primitive_to_conserved(primitive, metric);
      if (!close_enough(conserved.dens, expected.dens)
          || !close_enough(conserved.mom1, expected.mom1)
          || !close_enough(conserved.mom2, expected.mom2)
          || !close_enough(conserved.mom3, expected.mom3)
          || !close_enough(conserved.tau, expected.tau)) {
         std::cerr << "generated RHD primitive-to-conserved mismatch\n";
         return 1;
      }
      const auto matter = pyfluid_amrex::generated::rhd_eulerian_matter(
         primitive, metric
      );
      const amrex::Real eulerian_energy =
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor
         - primitive.pressure;
      const amrex::Real inertia =
         primitive.rho * enthalpy * lorentz_factor * lorentz_factor;
      if (!close_enough(matter.rho, eulerian_energy)
          || !close_enough(matter.momentum1, inertia * primitive.v1)
          || !close_enough(matter.momentum2, inertia * primitive.v2)
          || !close_enough(matter.momentum3, inertia * primitive.v3)
          || !close_enough(
             matter.stress11,
             inertia * primitive.v1 * primitive.v1 + primitive.pressure
          )
          || !close_enough(
             matter.stress12, inertia * primitive.v1 * primitive.v2
          )
          || !close_enough(
             matter.stress13, inertia * primitive.v1 * primitive.v3
          )
          || !close_enough(
             matter.stress22,
             inertia * primitive.v2 * primitive.v2 + primitive.pressure
          )
          || !close_enough(
             matter.stress23, inertia * primitive.v2 * primitive.v3
          )
          || !close_enough(
             matter.stress33,
             inertia * primitive.v3 * primitive.v3 + primitive.pressure
          )) {
         std::cerr << "generated RHD Eulerian-matter projection mismatch\n";
         return 1;
      }
      const pyfluid_amrex::RMHDPrimitive zero_field{
         primitive.rho,
         primitive.eps,
         primitive.pressure,
         primitive.v1,
         primitive.v2,
         primitive.v3,
         0.0_rt,
         0.0_rt,
         0.0_rt,
      };
      const auto magnetic_matter =
         pyfluid_amrex::generated::rmhd_eulerian_matter(zero_field, metric);
      if (!close_enough(magnetic_matter.rho, matter.rho)
          || !close_enough(magnetic_matter.momentum1, matter.momentum1)
          || !close_enough(magnetic_matter.momentum2, matter.momentum2)
          || !close_enough(magnetic_matter.momentum3, matter.momentum3)
          || !close_enough(magnetic_matter.stress11, matter.stress11)
          || !close_enough(magnetic_matter.stress12, matter.stress12)
          || !close_enough(magnetic_matter.stress13, matter.stress13)
          || !close_enough(magnetic_matter.stress22, matter.stress22)
          || !close_enough(magnetic_matter.stress23, matter.stress23)
          || !close_enough(magnetic_matter.stress33, matter.stress33)) {
         std::cerr << "generated RMHD zero-field matter reduction mismatch\n";
         return 1;
      }
      const auto flux1 = pyfluid_amrex::generated::rhd_flux<0>(primitive, metric);
      const auto flux2 = pyfluid_amrex::generated::rhd_flux<1>(primitive, metric);
      const auto flux3 = pyfluid_amrex::generated::rhd_flux<2>(primitive, metric);
      if (!close_enough(flux1.dens, expected.dens * primitive.v1)
          || !close_enough(flux1.mom1, expected.mom1 * primitive.v1 + primitive.pressure)
          || !close_enough(flux2.mom2, expected.mom2 * primitive.v2 + primitive.pressure)
          || !close_enough(flux3.tau, expected.tau * primitive.v3 + primitive.pressure * primitive.v3)) {
         std::cerr << "generated RHD directional flux mismatch\n";
         return 1;
      }
      const pyfluid_amrex::RHDWaveSpeeds speeds =
         pyfluid_amrex::generated::rhd_wave_speeds<0>(
            primitive,
            metric,
            pyfluid_amrex::generated::relativistic_ideal_gas_sound_speed_squared(
               primitive.eps, 5.0_rt / 3.0_rt
            )
         );
      if (!std::isfinite(speeds.minus) || !std::isfinite(speeds.plus)
          || speeds.minus > speeds.plus || speeds.minus < -1.0_rt || speeds.plus > 1.0_rt) {
         std::cerr << "generated SRHD wave-speed bounds mismatch\n";
         return 1;
      }
      const pyfluid_amrex::EOSParams eos{};
      const pyfluid_amrex::RHDPrimitive consistent{
         1.3_rt,
         0.2_rt,
         (eos.gamma - 1.0_rt) * 1.3_rt * 0.2_rt,
         0.2_rt,
         -0.1_rt,
         0.05_rt,
      };
      const auto physical_flux = pyfluid_amrex::generated::rhd_flux<0>(consistent, metric);
      const auto hll_flux = pyfluid_amrex::rhd_hll_flux<0, pyfluid_amrex::EOSIdealGas>(
         consistent, consistent, eos, metric
      );
      if (!close_enough(hll_flux.dens, physical_flux.dens)
          || !close_enough(hll_flux.mom1, physical_flux.mom1)
          || !close_enough(hll_flux.mom2, physical_flux.mom2)
          || !close_enough(hll_flux.mom3, physical_flux.mom3)
          || !close_enough(hll_flux.tau, physical_flux.tau)) {
         std::cerr << "SRHD HLL consistency mismatch\n";
         return 1;
      }
   }
   const std::array<pyfluid_amrex::HydroState, 4> hydro_states{{
      {1.2_rt, 0.3_rt, -0.2_rt, 0.1_rt, 2.5_rt},
      {0.7_rt, -1.4_rt, 0.5_rt, 0.25_rt, 5.0_rt},
      {3.0_rt, 1.1_rt, 2.2_rt, -0.9_rt, 12.0_rt},
      {10.0_rt, -0.02_rt, 0.03_rt, -0.04_rt, 50.0_rt},
   }};
   const std::array<pyfluid_amrex::MHDState, 4> mhd_states{{
      {1.2_rt, 0.3_rt, -0.2_rt, 0.1_rt, 2.5_rt, 0.4_rt, -0.1_rt, 0.2_rt},
      {0.7_rt, -1.4_rt, 0.5_rt, 0.25_rt, 5.0_rt, -0.6_rt, 0.2_rt, 0.7_rt},
      {3.0_rt, 1.1_rt, 2.2_rt, -0.9_rt, 12.0_rt, 0.0_rt, 1.3_rt, -0.8_rt},
      {10.0_rt, -0.02_rt, 0.03_rt, -0.04_rt, 50.0_rt, 2.0_rt, -1.5_rt, 0.1_rt},
   }};
   const std::array<amrex::Real, 4> pressures{{0.7_rt, 2.5_rt, 0.05_rt, 20.0_rt}};
   for (int n = 0; n < 4; ++n) {
      for (int direction = 0; direction < 3; ++direction) {
         pyfluid_amrex::HydroState hydro_flux{};
         pyfluid_amrex::MHDState mhd_flux{};
         if (direction == 0) {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<0>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<0>(mhd_states[n], pressures[n]);
         } else if (direction == 1) {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<1>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<1>(mhd_states[n], pressures[n]);
         } else {
            hydro_flux = pyfluid_amrex::generated::hydro_flux_orthonormal<2>(hydro_states[n], pressures[n]);
            mhd_flux = pyfluid_amrex::generated::mhd_flux_orthonormal<2>(mhd_states[n], pressures[n]);
         }
         if (!compare_hydro(hydro_flux, hydro_flux_reference(hydro_states[n], pressures[n], direction))) {
            std::cerr << "generated hydro flux mismatch state=" << n << " direction=" << direction << "\n";
            return 1;
         }
         if (!compare_mhd(mhd_flux, mhd_flux_reference(mhd_states[n], pressures[n], direction))) {
            std::cerr << "generated mhd flux mismatch state=" << n << " direction=" << direction << "\n";
            return 1;
         }
      }
   }
   {
      const pyfluid_amrex::HydroState state_l{1.1_rt, 0.2_rt, -0.3_rt, 0.4_rt, 2.7_rt};
      const pyfluid_amrex::HydroState state_r{0.8_rt, -0.5_rt, 0.6_rt, -0.1_rt, 3.2_rt};
      const pyfluid_amrex::HydroState flux_l{0.4_rt, 1.2_rt, -0.8_rt, 0.25_rt, 2.1_rt};
      const pyfluid_amrex::HydroState flux_r{-0.2_rt, 0.7_rt, 0.9_rt, -1.1_rt, 1.6_rt};
      constexpr amrex::Real wavespeed = 2.3_rt;
      constexpr amrex::Real s_left = -1.7_rt;
      constexpr amrex::Real s_right = 2.9_rt;
      if (!compare_hydro(
             pyfluid_amrex::generated::hydro_llf_flux(flux_l, flux_r, state_l, state_r, wavespeed),
             hydro_llf_reference(flux_l, flux_r, state_l, state_r, wavespeed)
          )) {
         std::cerr << "generated hydro LLF flux mismatch\n";
         return 1;
      }
      if (!compare_hydro(
             pyfluid_amrex::generated::hydro_hll_flux(flux_l, flux_r, state_l, state_r, s_left, s_right),
             hydro_hll_reference(flux_l, flux_r, state_l, state_r, s_left, s_right)
          )) {
         std::cerr << "generated hydro HLL flux mismatch\n";
         return 1;
      }
   }
   {
      const pyfluid_amrex::MHDState state_l{1.1_rt, 0.2_rt, -0.3_rt, 0.4_rt, 2.7_rt, 0.05_rt, -0.4_rt, 0.7_rt};
      const pyfluid_amrex::MHDState state_r{0.8_rt, -0.5_rt, 0.6_rt, -0.1_rt, 3.2_rt, -0.2_rt, 0.3_rt, -0.6_rt};
      const pyfluid_amrex::MHDState flux_l{0.4_rt, 1.2_rt, -0.8_rt, 0.25_rt, 2.1_rt, -0.9_rt, 0.1_rt, 0.8_rt};
      const pyfluid_amrex::MHDState flux_r{-0.2_rt, 0.7_rt, 0.9_rt, -1.1_rt, 1.6_rt, 0.6_rt, -0.5_rt, 0.2_rt};
      constexpr amrex::Real wavespeed = 2.3_rt;
      constexpr amrex::Real s_left = -1.7_rt;
      constexpr amrex::Real s_right = 2.9_rt;
      if (!compare_mhd(
             pyfluid_amrex::generated::mhd_llf_flux(flux_l, flux_r, state_l, state_r, wavespeed),
             mhd_llf_reference(flux_l, flux_r, state_l, state_r, wavespeed)
          )) {
         std::cerr << "generated MHD LLF flux mismatch\n";
         return 1;
      }
      if (!compare_mhd(
             pyfluid_amrex::generated::mhd_hll_flux(flux_l, flux_r, state_l, state_r, s_left, s_right),
             mhd_hll_reference(flux_l, flux_r, state_l, state_r, s_left, s_right)
          )) {
         std::cerr << "generated MHD HLL flux mismatch\n";
         return 1;
      }
   }
   constexpr amrex::Real center = 1.4_rt;
   constexpr amrex::Real base = -0.2_rt;
   constexpr amrex::Real xm = 0.5_rt;
   constexpr amrex::Real xp = 2.1_rt;
   constexpr amrex::Real ym = -1.2_rt;
   constexpr amrex::Real yp = 0.9_rt;
   constexpr amrex::Real zm = 0.3_rt;
   constexpr amrex::Real zp = -0.7_rt;
   constexpr amrex::Real inv_dx2 = 2.0_rt;
   constexpr amrex::Real inv_dy2 = 3.0_rt;
   constexpr amrex::Real inv_dz2 = 5.0_rt;
   constexpr amrex::Real cx = 0.11_rt;
   constexpr amrex::Real cy = 0.07_rt;
   constexpr amrex::Real cz = 0.03_rt;
   for (int dim = 1; dim <= 3; ++dim) {
      amrex::Real laplacian{};
      amrex::Real relaxed{};
      amrex::Real residual{};
      if (dim == 1) {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<1>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<1>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<1>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      } else if (dim == 2) {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<2>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<2>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<2>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      } else {
         laplacian = pyfluid_amrex::generated::laplacian_cartesian<3>(
            center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2
         );
         relaxed = pyfluid_amrex::generated::implicit_relaxed_cartesian<3>(
            base, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
         residual = pyfluid_amrex::generated::implicit_residual_cartesian<3>(
            base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz
         );
      }
      if (!close_enough(laplacian, laplacian_reference(dim, center, xm, xp, ym, yp, zm, zp, inv_dx2, inv_dy2, inv_dz2))) {
         std::cerr << "generated laplacian mismatch dim=" << dim << "\n";
         return 1;
      }
      if (!close_enough(relaxed, relaxed_reference(dim, base, xm, xp, ym, yp, zm, zp, cx, cy, cz))) {
         std::cerr << "generated implicit relaxed mismatch dim=" << dim << "\n";
         return 1;
      }
      if (!close_enough(residual, residual_reference(dim, base, center, xm, xp, ym, yp, zm, zp, cx, cy, cz))) {
         std::cerr << "generated implicit residual mismatch dim=" << dim << "\n";
         return 1;
      }
   }
   {
      if (!check_catalogued_derivative_dispatch()) return 1;
      const PolynomialGrid field{};
      constexpr int i = 2;
      constexpr int j = -1;
      constexpr int k = 3;
      constexpr int component = 0;
      const amrex::Real dx = pyfluid_amrex::generated::centered_first<4, 0>(
         field, i, j, k, component, 1.0_rt
      );
      const amrex::Real dx_lower = pyfluid_amrex::generated::domain_first<4, 0>(
         field, 0, j, k, component, 0, 0, 8, 1.0_rt
      );
      const amrex::Real dx_near_lower =
         pyfluid_amrex::generated::domain_first<4, 0>(
            field, 1, j, k, component, 1, 0, 8, 1.0_rt
         );
      const amrex::Real dx_upper = pyfluid_amrex::generated::domain_first<4, 0>(
         field, 8, j, k, component, 8, 0, 8, 1.0_rt
      );
      const amrex::Real dxx = pyfluid_amrex::generated::centered_second<4, 0>(
         field, i, j, k, component, 1.0_rt
      );
      const amrex::Real dxy = pyfluid_amrex::generated::centered_mixed<4, 0, 1>(
         field, i, j, k, component, 1.0_rt
      );
      const amrex::Real upwind = pyfluid_amrex::generated::upwind_first<4, 0>(
         field, i, j, k, component, 0.7_rt, 1.0_rt
      );
      const amrex::Real ko = pyfluid_amrex::generated::kreiss_oliger<4, 0>(
         field, i, j, k, component, 1.0_rt
      );
      const auto boundary_close = [&](amrex::Real actual, amrex::Real expected,
                                      int position, int side) {
         // One-sided differentiation cancels polynomial samples much larger
         // than the result. Bound arithmetic by its input scale; a relative
         // derivative-only tolerance is ill-conditioned near a stationary point.
         amrex::Real sample_scale = 1.0_rt;
         for (int offset = 0; offset <= 4; ++offset) {
            sample_scale = amrex::max(sample_scale,
               std::abs(field(position + side * offset, j, k, component)));
         }
         return std::isfinite(actual)
            && std::abs(actual - expected) <= 8.0e-15_rt * sample_scale;
      };
      if (!close_enough(dx, 4.0_rt * i * i * i + 2.0_rt * j)
          || !boundary_close(dx_lower, 2.0_rt * j, 0, 1)
          || !boundary_close(dx_near_lower, 4.0_rt + 2.0_rt * j, 1, 1)
          || !boundary_close(dx_upper, 4.0_rt * 8.0_rt * 8.0_rt * 8.0_rt + 2.0_rt * j, 8, -1)
          || !close_enough(dxx, 12.0_rt * i * i)
          || !close_enough(dxy, 2.0_rt)
          || !close_enough(upwind, 4.0_rt * i * i * i + 2.0_rt * j)
          || !close_enough(ko, 0.0_rt)) {
         std::cerr.precision(std::numeric_limits<amrex::Real>::max_digits10);
         std::cerr << "generated differential-operator mismatch: dx=" << dx
                   << " lower=" << dx_lower << " near_lower=" << dx_near_lower
                   << " upper=" << dx_upper << " dxx=" << dxx << " dxy=" << dxy
                   << " upwind=" << upwind << " ko=" << ko << '\n';
         return 1;
      }
      const amrex::Real face = pyfluid_amrex::generated::cell_to_face<0>(
         field, i, j, k, component
      );
      const amrex::Real expected_face = 0.5_rt * (
         field(i - 1, j, k, component) + field(i, j, k, component)
      );
      if (!close_enough(face, expected_face)) {
         std::cerr << "generated centering-transfer mismatch\n";
         return 1;
      }
      bool centering_transfers_pass = true;
      for (int order = 2; order <= pyfluid_amrex::maximum_spatial_order; order += 2) {
         if (!pyfluid_amrex::is_supported_spatial_order(order)) continue;
         centering_transfers_pass = centering_transfers_pass
            && pyfluid_amrex::dispatch_spatial_order(order, [](auto order_tag) {
               constexpr int spatial_order = decltype(order_tag)::value;
               return check_centering_transfer<spatial_order, 1>()
                  && check_centering_transfer<spatial_order, 2>()
                  && check_centering_transfer<spatial_order, 3>()
                  && check_topology_centering_transfer<spatial_order, 1>()
                  && check_topology_centering_transfer<spatial_order, 2>()
                  && check_topology_centering_transfer<spatial_order, 3>()
                  && check_bounded_topology_centering_transfer<spatial_order, 1>()
                  && check_bounded_topology_centering_transfer<spatial_order, 2>()
                  && check_bounded_topology_centering_transfer<spatial_order, 3>()
                  && check_point_average_conversion<spatial_order>();
            });
      }
      if (!centering_transfers_pass) {
         std::cerr << "generated high-order centering-transfer mismatch\n";
         return 1;
      }
   }
   const amrex::Real update = pyfluid_amrex::generated::explicit_diffusion_update(1.25_rt, 0.125_rt, -3.0_rt);
   if (!close_enough(update, 1.25_rt + 0.125_rt * -3.0_rt)) {
      std::cerr << "generated explicit diffusion update mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_x_cartesian(3.0_rt, -2.0_rt), 5.0_rt)) {
      std::cerr << "generated cartesian curl x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_y_cartesian(0.5_rt, 2.5_rt), -2.0_rt)) {
      std::cerr << "generated cartesian curl y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_z_cartesian(-1.5_rt, 4.0_rt), -5.5_rt)) {
      std::cerr << "generated cartesian curl z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_cartesian(1.0_rt, -2.0_rt, 0.25_rt), -0.75_rt)) {
      std::cerr << "generated cartesian divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_orthogonal_1(0.25_rt, 7.0_rt, -1.0_rt), 2.0_rt)) {
      std::cerr << "generated orthogonal curl mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_orthogonal(0.5_rt, 1.0_rt, 2.0_rt, -0.5_rt), 1.25_rt)) {
      std::cerr << "generated orthogonal divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_x(2.0_rt, -3.0_rt, 0.25_rt, 4.0_rt), -8.75_rt)) {
      std::cerr << "generated ideal emf x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_y(-3.0_rt, 2.0_rt, 4.0_rt, 0.25_rt), 8.75_rt)) {
      std::cerr << "generated ideal emf y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_z(2.0_rt, -3.0_rt, 0.25_rt, 4.0_rt), -8.75_rt)) {
      std::cerr << "generated ideal emf z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::total_emf(-2.0_rt, 0.5_rt, 3.0_rt), -0.5_rt)) {
      std::cerr << "generated total emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::resistive_emf_scaled_gradient(0.5_rt, 2.0_rt, -3.0_rt, 4.0_rt), -0.75_rt)) {
      std::cerr << "generated scaled-gradient resistive emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::magnetic_energy_density_from_components(1.0_rt, 2.0_rt, 3.0_rt), 7.0_rt)) {
      std::cerr << "generated magnetic energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::plasma_beta_from_pressure(4.0_rt, 1.0e-300_rt, 1.0_rt, 2.0_rt, 1.0_rt), 4.0_rt / 3.0_rt)) {
      std::cerr << "generated plasma beta mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_pressure_from_density_energy(3.0_rt, 5.0_rt / 3.0_rt), 2.0_rt)) {
      std::cerr << "generated ideal-gas pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_internal_energy_density_from_pressure(2.0_rt, 5.0_rt / 3.0_rt), 3.0_rt)) {
      std::cerr << "generated ideal-gas energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::polytrope_cold_pressure(2.0_rt, 0.5_rt, 2.0_rt), 2.0_rt)) {
      std::cerr << "generated polytropic pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::hybrid_polytrope_pressure_from_density_energy(2.0_rt, 4.0_rt, 0.5_rt, 2.0_rt, 1.5_rt, 0.0_rt), 3.0_rt)) {
      std::cerr << "generated hybrid-polytrope pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_cell_volume(2.0_rt, 3.0_rt, 4.0_rt), 24.0_rt)) {
      std::cerr << "generated cartesian cell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_face_area_x(3.0_rt, 4.0_rt), 12.0_rt)) {
      std::cerr << "generated cartesian face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_volume_scale(2.5_rt, 0.7_rt, 0.0_rt), 2.5_rt)) {
      std::cerr << "generated cylindrical volume scale mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_shell_volume_scale(2.0_rt, 3.0_rt), 2.5_rt)) {
      std::cerr << "generated cylindrical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_pressure_source_factor(2.0_rt, 3.0_rt), 0.4_rt)) {
      std::cerr << "generated cylindrical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_cell_volume_scale(2.0_rt, 3.0_rt, 0.5_rt), 3.0_rt)) {
      std::cerr << "generated cylindrical phi-z volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_axial_face_area_scale(2.0_rt, 0.5_rt), 1.0_rt)) {
      std::cerr << "generated cylindrical phi-z axial face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_azimuthal_face_area_scale(3.0_rt), 3.0_rt)) {
      std::cerr << "generated cylindrical phi-z azimuthal face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_scale_phi(2.0_rt, 0.5_rt, 0.0_rt), 2.0_rt * std::sin(0.5_rt))) {
      std::cerr << "generated spherical scale phi mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_shell_volume_scale(2.0_rt, 3.0_rt), 19.0_rt / 3.0_rt)) {
      std::cerr << "generated spherical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_pressure_source_factor(2.0_rt, 3.0_rt), 15.0_rt / 19.0_rt)) {
      std::cerr << "generated spherical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_cell_volume_scale(2.0_rt, 0.25_rt, 0.75_rt, 0.5_rt),
          2.0_rt * (std::cos(0.25_rt) - std::cos(0.75_rt))
       )) {
      std::cerr << "generated spherical theta-phi volume mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_theta_face_area_scale(2.0_rt, 0.25_rt, 0.5_rt),
          std::sin(0.25_rt)
       )) {
      std::cerr << "generated spherical theta-phi theta face area mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_azimuthal_face_area_scale(2.0_rt, 0.25_rt, 0.75_rt),
          1.0_rt
       )) {
      std::cerr << "generated spherical theta-phi azimuthal face area mismatch\n";
      return 1;
   }
   {
      constexpr amrex::Real radial_inner = 1.0_rt;
      constexpr amrex::Real radial_outer = 4.0_rt;
      const auto point = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2_rt,
         -0.1_rt,
         2.0_rt,
         0.3_rt,
         -0.2_rt,
         0.4_rt,
         -1.0_rt,
         radial_inner,
         radial_outer
      );
      const amrex::Real tan_q1 = std::tan(0.2_rt);
      const amrex::Real tan_q2 = std::tan(-0.1_rt);
      const amrex::Real denom2 = 1.0_rt + tan_q1 * tan_q1 + tan_q2 * tan_q2;
      const amrex::Real radial_fraction = (2.0_rt - radial_inner) / (radial_outer - radial_inner);
      const amrex::Real blend = 1.0_rt - radial_fraction + radial_fraction / std::sqrt(denom2);
      const amrex::Real primary = -2.0_rt * blend;
      if (
         !close_enough(point.q1, 0.3_rt + primary)
         || !close_enough(point.q2, -0.2_rt + primary * tan_q2)
         || !close_enough(point.q3, 0.4_rt + primary * tan_q1)
      ) {
         std::cerr << "generated Thornburg patch-map transform mismatch\n";
         return 1;
      }
      const auto jacobian = pyfluid_amrex::generated::thornburg_shell_jacobian_axis0(
         0.2_rt,
         -0.1_rt,
         2.0_rt,
         -1.0_rt,
         radial_inner,
         radial_outer
      );
      constexpr amrex::Real eps = 1.0e-6_rt;
      const auto point_plus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2_rt,
         -0.1_rt + eps,
         2.0_rt,
         0.3_rt,
         -0.2_rt,
         0.4_rt,
         -1.0_rt,
         radial_inner,
         radial_outer
      );
      const auto point_minus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2_rt,
         -0.1_rt - eps,
         2.0_rt,
         0.3_rt,
         -0.2_rt,
         0.4_rt,
         -1.0_rt,
         radial_inner,
         radial_outer
      );
      const amrex::Real expected_j22 = (point_plus.q2 - point_minus.q2) / (2.0_rt * eps);
      if (std::abs(jacobian.j22 - expected_j22) > 1.0e-9_rt) {
         std::cerr << "generated Thornburg patch-map Jacobian mismatch\n";
         return 1;
      }
   }
   {
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
      dx[0] = 0.5_rt;
#if (AMREX_SPACEDIM >= 2)
      dx[1] = 0.25_rt;
#endif
#if (AMREX_SPACEDIM == 3)
      dx[2] = 0.125_rt;
#endif
      const pyfluid_amrex::CartesianUniformMetric metric = pyfluid_amrex::make_cartesian_uniform_metric(dx);
#if (AMREX_SPACEDIM == 1)
      const amrex::Real expected_volume = 0.5_rt;
      const amrex::Real expected_area_x = 1.0_rt;
#elif (AMREX_SPACEDIM == 2)
      const amrex::Real expected_volume = 0.5_rt * 0.25_rt;
      const amrex::Real expected_area_x = 0.25_rt;
#else
      const amrex::Real expected_volume = 0.5_rt * 0.25_rt * 0.125_rt;
      const amrex::Real expected_area_x = 0.25_rt * 0.125_rt;
#endif
      if (!close_enough(metric.cell_volume, expected_volume) || !close_enough(metric.face_area[0], expected_area_x)) {
         std::cerr << "cartesian metric provider mismatch\n";
         return 1;
      }
      const auto cart_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshCartesian>(2.0_rt, 0.5_rt, 0.0_rt);
      if (!close_enough(cart_scale[0], 1.0_rt) || !close_enough(cart_scale[1], 1.0_rt) || !close_enough(cart_scale[2], 1.0_rt)) {
         std::cerr << "cartesian coordinate scale provider mismatch\n";
         return 1;
      }
      const auto sph_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshSpherical>(2.0_rt, 0.5_rt, 0.0_rt);
      if (!close_enough(sph_scale[2], 2.0_rt * std::sin(0.5_rt))) {
         std::cerr << "spherical coordinate scale provider mismatch\n";
         return 1;
      }
   }
   {
      const auto integrands = pyfluid_amrex::generated::conserved_diagnostic_integrands(
         2.0_rt, 3.0_rt, -4.0_rt, 5.0_rt, 7.0_rt, 1.0_rt, 2.0_rt, -2.0_rt,
         0.5_rt, -1.0_rt, 2.0_rt, -3.0_rt, 0.25_rt, 0.5_rt, 0.75_rt, -0.125_rt
      );
      if (
         !close_enough(integrands.mass, 2.0_rt)
         || !close_enough(integrands.angular_momentum1, 3.0_rt)
         || !close_enough(integrands.angular_momentum2, 3.5_rt)
         || !close_enough(integrands.angular_momentum3, 1.0_rt)
         || !close_enough(integrands.magnetic_energy, 4.5_rt)
         || !close_enough(integrands.gravity_potential_energy, -2.625_rt)
         || !close_enough(integrands.passive_mass, 0.75_rt)
         || !close_enough(integrands.divb, 0.125_rt)
      ) {
         std::cerr << "generated conserved diagnostic integrand mismatch\n";
         return 1;
      }

      const pyfluid_amrex::RMHDPrimitive rmhd_primitive{
         1.0_rt, 1.5_rt, 1.0_rt, 0.2_rt, -0.1_rt, 0.05_rt, 0.75_rt, 0.3_rt, -0.2_rt
      };
      const pyfluid_amrex::SpacetimeMetric flat_metric{};
      const auto rmhd_integrands =
         pyfluid_amrex::generated::rmhd_diagnostic_integrands(
            rmhd_primitive, flat_metric
         );
      if (
         !close_enough(rmhd_integrands.electromagnetic_energy, 0.358378125_rt)
         || !close_enough(rmhd_integrands.b_squared, 0.66824375_rt)
         || !close_enough(
            rmhd_integrands.enthalpy_magnetization, 0.66824375_rt / 3.5_rt
         )
         || !close_enough(rmhd_integrands.plasma_beta, 2.0_rt / 0.66824375_rt)
         || !close_enough(
            rmhd_integrands.lorentz_factor, 1.0_rt / std::sqrt(1.0_rt - 0.0525_rt)
         )
      ) {
         std::cerr << "generated RMHD diagnostic integrand mismatch\n";
         return 1;
      }

      pyfluid_amrex::Z4cState minkowski{};
      minkowski.chi = 1.0_rt;
      minkowski.gamma_tilde11 = 1.0_rt;
      minkowski.gamma_tilde22 = 1.0_rt;
      minkowski.gamma_tilde33 = 1.0_rt;
      minkowski.lapse = 1.0_rt;
      const pyfluid_amrex::Z4cSpatialDerivatives zero_derivatives{};
      const pyfluid_amrex::Vector3 position{1.0_rt, -2.0_rt, 3.0_rt};
      const auto flux = pyfluid_amrex::generated::adm_flux_integrands(
         minkowski, zero_derivatives, position
      );
      const auto vacuum_volume =
         pyfluid_amrex::generated::adm_constraint_volume_integrands(
            minkowski,
            zero_derivatives,
            pyfluid_amrex::vacuum_z4c_matter(),
            position
         );
      const auto metric = pyfluid_amrex::generated::z4c_conformal_metric(minkowski);
      const auto connections = pyfluid_amrex::generated::z4c_connections(
         minkowski, zero_derivatives, metric
      );
      const auto geometry = pyfluid_amrex::generated::z4c_diagnostic_geometry(
         minkowski, zero_derivatives, metric, connections
      );
      const auto radiation = pyfluid_amrex::generated::weyl_radiation_integrands(
         minkowski,
         zero_derivatives,
         metric,
         connections,
         geometry,
         pyfluid_amrex::vacuum_z4c_matter()
      );
      const auto psi4 = pyfluid_amrex::generated::psi4_from_weyl(
         metric, radiation, position
      );
      if (
         !close_enough(flux.mass1, 0.0_rt)
         || !close_enough(flux.mass2, 0.0_rt)
         || !close_enough(flux.mass3, 0.0_rt)
         || !close_enough(flux.linear_momentum11, 0.0_rt)
         || !close_enough(flux.linear_momentum12, 0.0_rt)
         || !close_enough(flux.linear_momentum13, 0.0_rt)
         || !close_enough(flux.linear_momentum21, 0.0_rt)
         || !close_enough(flux.linear_momentum22, 0.0_rt)
         || !close_enough(flux.linear_momentum23, 0.0_rt)
         || !close_enough(flux.linear_momentum31, 0.0_rt)
         || !close_enough(flux.linear_momentum32, 0.0_rt)
         || !close_enough(flux.linear_momentum33, 0.0_rt)
         || !close_enough(flux.angular_momentum11, 0.0_rt)
         || !close_enough(flux.angular_momentum12, 0.0_rt)
         || !close_enough(flux.angular_momentum13, 0.0_rt)
         || !close_enough(flux.angular_momentum21, 0.0_rt)
         || !close_enough(flux.angular_momentum22, 0.0_rt)
         || !close_enough(flux.angular_momentum23, 0.0_rt)
         || !close_enough(flux.angular_momentum31, 0.0_rt)
         || !close_enough(flux.angular_momentum32, 0.0_rt)
         || !close_enough(flux.angular_momentum33, 0.0_rt)
         || !close_enough(vacuum_volume.mass, 0.0_rt)
         || !close_enough(vacuum_volume.linear_momentum1, 0.0_rt)
         || !close_enough(vacuum_volume.linear_momentum2, 0.0_rt)
         || !close_enough(vacuum_volume.linear_momentum3, 0.0_rt)
         || !close_enough(vacuum_volume.angular_momentum1, 0.0_rt)
         || !close_enough(vacuum_volume.angular_momentum2, 0.0_rt)
         || !close_enough(vacuum_volume.angular_momentum3, 0.0_rt)
         || !close_enough(radiation.electric11, 0.0_rt)
         || !close_enough(radiation.electric12, 0.0_rt)
         || !close_enough(radiation.electric13, 0.0_rt)
         || !close_enough(radiation.electric22, 0.0_rt)
         || !close_enough(radiation.electric23, 0.0_rt)
         || !close_enough(radiation.electric33, 0.0_rt)
         || !close_enough(radiation.magnetic11, 0.0_rt)
         || !close_enough(radiation.magnetic12, 0.0_rt)
         || !close_enough(radiation.magnetic13, 0.0_rt)
         || !close_enough(radiation.magnetic22, 0.0_rt)
         || !close_enough(radiation.magnetic23, 0.0_rt)
         || !close_enough(radiation.magnetic33, 0.0_rt)
         || !close_enough(psi4.psi4_real, 0.0_rt)
         || !close_enough(psi4.psi4_imaginary, 0.0_rt)
      ) {
         std::cerr << "generated Minkowski ADM/radiation diagnostic mismatch\n";
         return 1;
      }

      // Along positive axis 1 the tetrad is (e_phi, e_r, e_theta) =
      // (e_2, e_1, -e_3). Independent electric/magnetic contributions give
      // Psi4 = (E_33-E_22)/2 + B_23 + i(E_23-(B_33-B_22)/2).
      // Rescaling the diagonal metric and covariant Weyl components together
      // must preserve that result after tetrad normalization.
      for (bool stretched : {false, true}) {
         const amrex::Real length1 = stretched ? 2.0_rt : 1.0_rt;
         const amrex::Real length2 = stretched ? 3.0_rt : 1.0_rt;
         const amrex::Real length3 = stretched ? 4.0_rt : 1.0_rt;
         pyfluid_amrex::Z4cConformalMetric tetrad_metric{};
         tetrad_metric.gamma11 = length1 * length1;
         tetrad_metric.gamma22 = length2 * length2;
         tetrad_metric.gamma33 = length3 * length3;
         tetrad_metric.inv_gamma11 = 1.0_rt / tetrad_metric.gamma11;
         tetrad_metric.inv_gamma22 = 1.0_rt / tetrad_metric.gamma22;
         tetrad_metric.inv_gamma33 = 1.0_rt / tetrad_metric.gamma33;
         tetrad_metric.sqrt_det_gamma = length1 * length2 * length3;
         pyfluid_amrex::WeylRadiationIntegrands nonzero_weyl{};
         nonzero_weyl.electric11 = -8.0_rt * length1 * length1;
         nonzero_weyl.electric22 = 2.0_rt * length2 * length2;
         nonzero_weyl.electric33 = 6.0_rt * length3 * length3;
         nonzero_weyl.electric23 = 3.0_rt * length2 * length3;
         nonzero_weyl.magnetic11 = -6.0_rt * length1 * length1;
         nonzero_weyl.magnetic22 = 5.0_rt * length2 * length2;
         nonzero_weyl.magnetic33 = length3 * length3;
         nonzero_weyl.magnetic23 = 7.0_rt * length2 * length3;
         const auto nonzero_psi4 = pyfluid_amrex::generated::psi4_from_weyl(
            tetrad_metric, nonzero_weyl, pyfluid_amrex::Vector3{1.0_rt, 0.0_rt, 0.0_rt}
         );
         if (!close_enough(nonzero_psi4.psi4_real, 9.0_rt)
             || !close_enough(nonzero_psi4.psi4_imaginary, 5.0_rt)) {
            std::cerr << "generated nonzero Psi4 tetrad normalization mismatch\n";
            return 1;
         }
      }

      constexpr amrex::Real trace_curvature = 0.6_rt;
      pyfluid_amrex::Z4cState homogeneous = minkowski;
      homogeneous.k_hat = trace_curvature;
      pyfluid_amrex::Z4cMatter constraint_matter{};
      constraint_matter.rho = trace_curvature * trace_curvature
         / (24.0_rt * pyfluid_amrex::constants::pi);
      const auto constrained_volume =
         pyfluid_amrex::generated::adm_constraint_volume_integrands(
            homogeneous, zero_derivatives, constraint_matter, position
         );
      if (!close_enough(constrained_volume.mass, 0.0_rt)) {
         std::cerr << "Hamiltonian-substituted ADM mass integrand mismatch\n";
         return 1;
      }

      constexpr amrex::Real conformal_gradient = 0.2_rt;
      pyfluid_amrex::Z4cSpatialDerivatives conformal_derivatives{};
      conformal_derivatives.d_chi[0] = -4.0_rt * conformal_gradient;
      const auto conformal_flux = pyfluid_amrex::generated::adm_flux_integrands(
         minkowski, conformal_derivatives, position
      );
      const auto conformal_volume =
         pyfluid_amrex::generated::adm_constraint_volume_integrands(
            minkowski,
            conformal_derivatives,
            pyfluid_amrex::vacuum_z4c_matter(),
            position
         );
      if (
         !close_enough(
            conformal_flux.mass1,
            -conformal_gradient / (2.0_rt * pyfluid_amrex::constants::pi)
         )
         || !close_enough(
            conformal_volume.mass,
            -conformal_gradient * conformal_gradient
               / (2.0_rt * pyfluid_amrex::constants::pi)
         )
      ) {
         std::cerr << "constraint-assisted conformal ADM mass mismatch\n";
         return 1;
      }

      pyfluid_amrex::Z4cMatter momentum_matter{};
      momentum_matter.momentum1 = 2.0_rt;
      momentum_matter.momentum2 = 3.0_rt;
      momentum_matter.momentum3 = 5.0_rt;
      const auto momentum_volume =
         pyfluid_amrex::generated::adm_constraint_volume_integrands(
            minkowski, zero_derivatives, momentum_matter, position
         );
      const auto matter_flux = pyfluid_amrex::generated::matter_energy_flux_integrands(
         minkowski, metric, momentum_matter
      );
      if (
         !close_enough(momentum_volume.angular_momentum1, -19.0_rt)
         || !close_enough(momentum_volume.angular_momentum2, 1.0_rt)
         || !close_enough(momentum_volume.angular_momentum3, 7.0_rt)
         || !close_enough(matter_flux.flux1, 2.0_rt)
         || !close_enough(matter_flux.flux2, 3.0_rt)
         || !close_enough(matter_flux.flux3, 5.0_rt)
      ) {
         std::cerr << "momentum-constraint or matter-current diagnostic mismatch\n";
         return 1;
      }
   }
   return 0;
}
