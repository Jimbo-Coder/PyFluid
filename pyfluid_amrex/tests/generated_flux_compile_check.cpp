#include <AMReX.H>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/kernels_generated/eos.hpp"
#include "pyfluid_amrex/kernels_generated/characteristic_transforms.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/scalar_ops.hpp"
#include "pyfluid_amrex/kernels_generated/vector_ops.hpp"
#include "pyfluid_amrex/kernels_generated/z4c.hpp"
#include "pyfluid_amrex/kernels_generated/coordinate_metrics.hpp"
#include "pyfluid_amrex/kernels_generated/ideal_mhd_flux_cartesian.hpp"
#include "pyfluid_amrex/kernels_generated/patch_maps.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/kernels_generated/riemann_ops.hpp"
#include "pyfluid_amrex/c2p_policy.hpp"
#include "pyfluid_amrex/eos.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "relativity/rhd_system.hpp"
#include "pyfluid_amrex/reconstruction/mp5.hpp"
#include "pyfluid_amrex/reconstruction/teno5.hpp"
#include "pyfluid_amrex/reconstruction/wenoz.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/z4c_constraint_pointwise.hpp"
#include "pyfluid_amrex/z4c_evolution_pointwise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

struct PolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      const amrex::Real x = static_cast<amrex::Real>(i);
      const amrex::Real y = static_cast<amrex::Real>(j);
      const amrex::Real z = static_cast<amrex::Real>(k);
      return x * x * x * x + 2.0 * x * y + 3.0 * y * y + z + component;
   }
};

bool close_enough(amrex::Real actual, amrex::Real expected);

template <int Degree>
struct CenteringPolynomialGrid {
   AMREX_GPU_HOST_DEVICE
   static amrex::Real power(amrex::Real value) noexcept
   {
      amrex::Real result = 1.0;
      for (int exponent = 0; exponent < Degree; ++exponent) result *= value;
      return result;
   }

   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int i, int j, int k, int component) const noexcept
   {
      return power(static_cast<amrex::Real>(i))
         + 2.0 * power(static_cast<amrex::Real>(j))
         + 3.0 * power(static_cast<amrex::Real>(k))
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
         value += 2.0 * CenteringPolynomialGrid<Order - 1>::power(j + offset);
      } else {
         value += 2.0 * CenteringPolynomialGrid<Order - 1>::power(j);
      }
      if constexpr (Dim >= 3) {
         value += 3.0 * CenteringPolynomialGrid<Order - 1>::power(k + offset);
      } else {
         value += 3.0 * CenteringPolynomialGrid<Order - 1>::power(k);
      }
      return value;
   };
   const amrex::Real vertex_value = pyfluid_amrex::generated::cell_to_vertex<Order, Dim>(
      field, i, j, k, component
   );
   const amrex::Real cell_value = pyfluid_amrex::generated::vertex_to_cell<Order, Dim>(
      field, i, j, k, component
   );
   return close_enough(vertex_value, expected(-0.5))
      && close_enough(cell_value, expected(0.5));
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
            if (source_is_nodal == destination_is_nodal) return 0.0;
            return source_is_nodal ? 0.5 : -0.5;
         };
         amrex::Real expected = CenteringPolynomialGrid<Order - 1>::power(
            i + offset(0)
         );
         expected += 2.0 * CenteringPolynomialGrid<Order - 1>::power(
            j + (Dim >= 2 ? offset(1) : 0.0)
         );
         expected += 3.0 * CenteringPolynomialGrid<Order - 1>::power(
            k + (Dim >= 3 ? offset(2) : 0.0)
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
         if (!close_enough(actual, expected)) return false;
      }
   }
   return true;
}

struct ComponentGrid {
   AMREX_GPU_HOST_DEVICE
   amrex::Real operator()(int, int, int, int component) const noexcept
   {
      return 0.5 + static_cast<amrex::Real>(component);
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
         ? 1.0
         : 0.0;
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
      state.mom1 * vn + (direction == 0 ? pressure : 0.0),
      state.mom2 * vn + (direction == 1 ? pressure : 0.0),
      state.mom3 * vn + (direction == 2 ? pressure : 0.0),
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
   const amrex::Real ptot = pressure + 0.5 * b2;
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
      flux.b1 = 0.0;
   } else if (direction == 1) {
      flux.mom2 += ptot;
      flux.b2 = 0.0;
   } else {
      flux.mom3 += ptot;
      flux.b3 = 0.0;
   }
   return flux;
}

bool close_enough(amrex::Real actual, amrex::Real expected)
{
   const amrex::Real scale = std::max<amrex::Real>(1.0, std::abs(expected));
   return std::isfinite(actual) && std::abs(actual - expected) <= 8.0e-15 * scale;
}

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

bool validate_z4c_pointwise_bridge()
{
   const auto loaded = pyfluid_amrex::load_z4c_state(ComponentGrid{}, 3, -2, 1);
   const auto loaded_components = z4c_state_components(loaded);
   for (int component = 0; component < pyfluid_amrex::Z4C_NUM_COMPONENTS; ++component) {
      if (!close_enough(loaded_components[component], 0.5 + component)) {
         std::cerr << "Z4c pointwise state mapping mismatch at component "
                   << component << '\n';
         return false;
      }
   }

   constexpr int i = 2;
   constexpr int j = -1;
   constexpr int k = 3;
   const std::array<amrex::Real, 3> inv_spacing = {1.0, 1.0, 1.0};
   pyfluid_amrex::Z4cState advection_state{};
   advection_state.shift1 = 0.7;
   advection_state.shift2 = -0.4;
   advection_state.shift3 = 0.2;
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
      if (!close_enough(value, 0.0)) {
         std::cerr << "Z4c grid-level Minkowski constraint mismatch\n";
         return false;
      }
   }
   const auto flat_rhs = pyfluid_amrex::evaluate_vacuum_z4c_rhs_at<
      4,
      AMREX_SPACEDIM,
      pyfluid_amrex::Z4cLapseGauge::OnePlusLog,
      pyfluid_amrex::Z4cShiftGauge::GammaDriver
   >(
      MinkowskiZ4cGrid{},
      i,
      j,
      k,
      inv_spacing,
      0.25,
      pyfluid_amrex::Z4cEvolutionParameters{}
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
      if (!close_enough(value, 0.0)) {
         std::cerr << "Z4c grid-level Minkowski RHS mismatch\n";
         return false;
      }
   }
   const amrex::Real expected_first[3] = {
      4.0 * i * i * i + 2.0 * j,
      2.0 * i + 6.0 * j,
      1.0,
   };
   const amrex::Real expected_second[3] = {12.0 * i * i, 6.0, 0.0};
   for (int direction = 0; direction < 3; ++direction) {
      const amrex::Real expected = direction < AMREX_SPACEDIM
         ? expected_first[direction]
         : 0.0;
      const amrex::Real expected_dd = direction < AMREX_SPACEDIM
         ? expected_second[direction]
         : 0.0;
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
         if (std::abs(value - expected) > 2.0e-12) {
            std::cerr << "Z4c pointwise scalar derivative mismatch in direction "
                      << direction << '\n';
            return false;
         }
      }
      if (std::abs(derivatives.dd_chi[direction][direction] - expected_dd) > 2.0e-12
          || std::abs(derivatives.dd_lapse[direction][direction] - expected_dd)
             > 2.0e-12) {
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
            if (std::abs(value - expected) > 2.0e-12) {
               std::cerr << "Z4c pointwise tensor derivative mismatch\n";
               return false;
            }
         }
         if (std::abs(
                derivatives.dd_gamma_tilde[direction][direction][component]
                - expected_dd
             ) > 2.0e-12) {
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
            if (std::abs(value - expected) > 2.0e-12) {
               std::cerr << "Z4c pointwise vector derivative mismatch\n";
               return false;
            }
         }
         if (std::abs(
                derivatives.dd_shift[direction][direction][component] - expected_dd
             ) > 2.0e-12) {
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
                  ? 2.0
                  : 0.0)
            : 0.0;
         if (std::abs(derivatives.dd_chi[first][second] - expected) > 2.0e-12
             || std::abs(derivatives.dd_lapse[first][second] - expected)
                > 2.0e-12) {
            std::cerr << "Z4c pointwise mixed scalar derivative mismatch\n";
            return false;
         }
         for (int component = 0; component < 6; ++component) {
            if (std::abs(
                   derivatives.dd_gamma_tilde[first][second][component] - expected
                ) > 2.0e-12) {
               std::cerr << "Z4c pointwise mixed tensor derivative mismatch\n";
               return false;
            }
         }
         for (int component = 0; component < 3; ++component) {
            if (std::abs(derivatives.dd_shift[first][second][component] - expected)
                > 2.0e-12) {
               std::cerr << "Z4c pointwise mixed vector derivative mismatch\n";
               return false;
            }
         }
      }
   }

   pyfluid_amrex::Z4cGeometricRhs geometric_rhs{};
   geometric_rhs.chi = 1.0;
   geometric_rhs.gamma_tilde11 = 2.0;
   geometric_rhs.gamma_tilde12 = 3.0;
   geometric_rhs.gamma_tilde13 = 4.0;
   geometric_rhs.gamma_tilde22 = 5.0;
   geometric_rhs.gamma_tilde23 = 6.0;
   geometric_rhs.gamma_tilde33 = 7.0;
   geometric_rhs.k_hat = 8.0;
   geometric_rhs.a_tilde11 = 9.0;
   geometric_rhs.a_tilde12 = 10.0;
   geometric_rhs.a_tilde13 = 11.0;
   geometric_rhs.a_tilde22 = 12.0;
   geometric_rhs.a_tilde23 = 13.0;
   geometric_rhs.a_tilde33 = 14.0;
   geometric_rhs.gamma_tilde1 = 15.0;
   geometric_rhs.gamma_tilde2 = 16.0;
   geometric_rhs.gamma_tilde3 = 17.0;
   geometric_rhs.theta = 18.0;
   const pyfluid_amrex::Z4cRhs z4c_rhs{
      geometric_rhs,
      19.0,
      pyfluid_amrex::Z4cShiftRhs{20.0, 21.0, 22.0},
   };
   MutableComponentGrid stored{};
   pyfluid_amrex::store_z4c_rhs(stored, 0, 0, 0, z4c_rhs);
   for (int component = 0; component < pyfluid_amrex::Z4C_NUM_COMPONENTS; ++component) {
      if (!close_enough(stored.values[component], component + 1.0)) {
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
   amrex::Real error = 0.0;
   for (std::size_t component = 0; component < N; ++component) {
      const amrex::Real scale = std::max({
         amrex::Real{1.0},
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
      independent_conserved.mom1 *= -0.81;
      independent_conserved.mom2 += 0.23 * independent_conserved.rho;
      independent_conserved.energy += 0.37 * std::max(amrex::Real{1.0}, independent_conserved.rho);
      independent_mhd_conserved.mom1 *= -0.81;
      independent_mhd_conserved.mom2 += 0.23 * independent_mhd_conserved.rho;
      independent_mhd_conserved.energy += 0.37 * std::max(amrex::Real{1.0}, independent_mhd_conserved.rho);
   }
   const auto primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_conserved, eos);
   const auto independent_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(primitive_from_independent, eos);
   const auto mhd_primitive_from_independent =
      pyfluid_amrex::primitive_from_conserved<EOSMethod>(independent_mhd_conserved, eos);
   const auto independent_mhd_conserved_round_trip =
      pyfluid_amrex::conserved_from_primitive<EOSMethod>(mhd_primitive_from_independent, eos);

   constexpr amrex::Real tolerance = 256.0 * std::numeric_limits<amrex::Real>::epsilon();
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
   eos.gamma = 1.4;
   eos.segment_k[0] = 0.73;
   eos.segment_gamma[0] = 1.65;
   eos.thermal_gamma = 1.37;
   const std::array<pyfluid_amrex::HydroPrimitive, 5> hydro_corpus{{
      {0.071, 2.3, -0.47, 0.19, 0.013},
      {0.82, -1.7, 0.93, -0.31, 4.6},
      {1.0, 0.0, 0.0, 0.0, 0.41},
      {7.4, 0.017, -3.2, 1.1, 13.0},
      {19.0, -0.004, 0.007, -0.009, 0.26},
   }};
   const std::array<pyfluid_amrex::MHDPrimitive, 5> mhd_corpus{{
      {0.071, 2.3, -0.47, 0.19, 0.013, 4.0, -0.07, 0.8},
      {0.82, -1.7, 0.93, -0.31, 4.6, -2.1, 3.7, -0.4},
      {1.0, 0.0, 0.0, 0.0, 0.41, 0.0, 0.0, 0.0},
      {7.4, 0.017, -3.2, 1.1, 13.0, 0.003, -8.0, 2.5},
      {19.0, -0.004, 0.007, -0.009, 0.26, 15.0, 0.2, -0.5},
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

bool validate_piecewise_polytrope()
{
   pyfluid_amrex::EOSParams eos{};
   eos.thermal_gamma = 1.6;
   eos.polytrope_segments = 2;
   eos.segment_k[0] = 0.5;
   eos.segment_gamma[0] = 2.0;
   eos.segment_density_break[0] = 2.0;
   eos.segment_gamma[1] = 1.5;
   eos.segment_k[1] = eos.segment_k[0] * std::pow(
      eos.segment_density_break[0],
      eos.segment_gamma[0] - eos.segment_gamma[1]
   );
   eos.segment_energy_offset[1] =
      eos.segment_k[0] * std::pow(
         eos.segment_density_break[0], eos.segment_gamma[0] - 1.0
      ) / (eos.segment_gamma[0] - 1.0)
      - eos.segment_k[1] * std::pow(
         eos.segment_density_break[0], eos.segment_gamma[1] - 1.0
      ) / (eos.segment_gamma[1] - 1.0);

   const amrex::Real below = eos.segment_density_break[0] * (1.0 - 1.0e-10);
   const amrex::Real above = eos.segment_density_break[0] * (1.0 + 1.0e-10);
   const amrex::Real pressure_below =
      pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(below, eos);
   const amrex::Real pressure_above =
      pyfluid_amrex::eos_cold_pressure<pyfluid_amrex::EOSPolytrope>(above, eos);
   const amrex::Real energy_below =
      pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSPolytrope>(below, eos);
   const amrex::Real energy_above =
      pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSPolytrope>(above, eos);
   if (std::abs(pressure_below - pressure_above)
          > 1.0e-9 * std::max(amrex::Real{1.0}, std::abs(pressure_below))
       || std::abs(energy_below - energy_above)
          > 1.0e-9 * std::max(amrex::Real{1.0}, std::abs(energy_below))) {
      std::cerr << "piecewise-polytrope continuity mismatch\n";
      return false;
   }

   for (amrex::Real const rho : {0.5, 1.5, 2.5, 8.0}) {
      const amrex::Real cold_energy =
         pyfluid_amrex::eos_cold_energy_density<pyfluid_amrex::EOSHybridPolytrope>(rho, eos);
      const amrex::Real thermal_energy = 0.17 * rho;
      const amrex::Real pressure =
         pyfluid_amrex::eos_pressure_from_density_energy<pyfluid_amrex::EOSHybridPolytrope>(
            rho, cold_energy + thermal_energy, eos
         );
      const amrex::Real recovered_energy =
         pyfluid_amrex::eos_internal_energy_density_from_pressure<pyfluid_amrex::EOSHybridPolytrope>(
            rho, pressure, eos
         );
      if (std::abs(recovered_energy - cold_energy - thermal_energy)
          > 1.0e-13 * std::max(
             amrex::Real{1.0}, std::abs(cold_energy + thermal_energy)
          )) {
         std::cerr << "piecewise hybrid-EOS pressure/energy inverse mismatch\n";
         return false;
      }

      const pyfluid_amrex::RHDPrimitive primitive{
         rho,
         (cold_energy + thermal_energy) / rho,
         pressure,
         0.21,
         -0.08,
         0.04,
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
               <= 2.0e-10 * std::max(amrex::Real{1.0}, std::abs(expected));
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

bool validate_srhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4;
   constexpr amrex::Real tolerance = 2.0e-11;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RHDPrimitive, 7> primitive_corpus{{
      {1.0e-3, 1.0e-4, 0.0, 0.0, 0.0, 0.0},
      {0.071, 0.23, 0.0, 0.31, -0.17, 0.09},
      {0.82, 4.7, 0.0, -0.71, 0.23, -0.11},
      {1.0, 0.7, 0.0, 0.0, 0.0, 0.0},
      {7.4, 0.017, 0.0, 0.82, -0.31, 0.18},
      {19.0, 13.0, 0.0, -0.91, 0.21, -0.07},
      {0.3, 2.1, 0.0, 0.965, 0.08, -0.03},
   }};

   amrex::Real worst_primitive_error = 0.0;
   amrex::Real worst_conserved_error = 0.0;
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
      independent_conserved.mom1 *= -0.83;
      independent_conserved.mom2 += 0.07 * independent_conserved.dens;
      independent_conserved.mom3 *= 0.91;
      independent_conserved.tau += 0.19 * (
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
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const amrex::Real x = (1.0 + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6 * std::max(amrex::Real{1.0}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0 * step);
      const amrex::Real analytic =
         pyfluid_amrex::srhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 2.0e-6 * std::max(amrex::Real{1.0}, std::abs(finite_difference))) {
         std::cerr << "SRHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }
   }

   pyfluid_amrex::SpacetimeMetric curved_metric{};
   curved_metric.lapse = 0.83;
   curved_metric.shift1 = 0.07;
   curved_metric.shift2 = -0.03;
   curved_metric.shift3 = 0.02;
   curved_metric.gamma12 = 0.1;
   curved_metric.inv_gamma11 = 1.0 / 0.99;
   curved_metric.inv_gamma12 = -0.1 / 0.99;
   curved_metric.inv_gamma22 = 1.0 / 0.99;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99);
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
   using pyfluid_amrex::FixedSpacetimeMetric;
   using pyfluid_amrex::FixedSpacetimeParameters;
   using pyfluid_amrex::fixed_spacetime_geometry;

   const FixedSpacetimeParameters rindler{
      FixedSpacetimeMetric::Rindler,
      0.11,
      -0.07,
      0.03,
      0.2,
      -0.1,
      0.4,
      1.0,
   };
   const auto rindler_geometry = fixed_spacetime_geometry(
      rindler, 0.8, -0.4, 0.6, 0.0
   );
   const amrex::Real expected_lapse = 1.0
      + rindler.acceleration1 * (0.8 - rindler.origin1)
      + rindler.acceleration2 * (-0.4 - rindler.origin2)
      + rindler.acceleration3 * (0.6 - rindler.origin3);
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

   pyfluid_amrex::RHDPrimitive rest{1.0, 0.5, 0.2, 0.0, 0.0, 0.0};
   pyfluid_amrex::SpacetimeMetricDerivatives acceleration{};
   acceleration.d1_lapse = 0.1;
   const auto source = pyfluid_amrex::generated::rhd_geometric_source(
      rest, pyfluid_amrex::minkowski_spacetime_metric(), acceleration
   );
   const amrex::Real eulerian_energy = rest.rho * (1.0 + rest.eps);
   if (!close_enough(source.dens, 0.0)
       || !close_enough(source.mom1, -0.1 * eulerian_energy)
       || !close_enough(source.mom2, 0.0)
       || !close_enough(source.mom3, 0.0)
       || !close_enough(source.tau, 0.0)) {
      std::cerr << "Rindler geometric source mismatch\n";
      return false;
   }

   const FixedSpacetimeParameters schwarzschild{
      FixedSpacetimeMetric::SchwarzschildIsotropic,
      0.0,
      0.0,
      0.0,
      -0.3,
      0.2,
      -0.1,
      0.4,
   };
   constexpr amrex::Real position1 = 1.7;
   constexpr amrex::Real position2 = 0.6;
   constexpr amrex::Real position3 = -0.25;
   const auto schwarzschild_geometry = fixed_spacetime_geometry(
      schwarzschild, position1, position2, position3, 0.0
   );
   const auto& metric = schwarzschild_geometry.metric;
   if (!close_enough(metric.gamma11 * metric.inv_gamma11, 1.0)
       || !close_enough(metric.gamma22 * metric.inv_gamma22, 1.0)
       || !close_enough(metric.gamma33 * metric.inv_gamma33, 1.0)
       || !close_enough(
          metric.sqrt_det_gamma * metric.sqrt_det_gamma,
          metric.gamma11 * metric.gamma22 * metric.gamma33
       )) {
      std::cerr << "isotropic-Schwarzschild metric identity mismatch\n";
      return false;
   }
   constexpr amrex::Real step = 1.0e-6;
   const auto plus = fixed_spacetime_geometry(
      schwarzschild, position1 + step, position2, position3, 0.0
   );
   const auto minus = fixed_spacetime_geometry(
      schwarzschild, position1 - step, position2, position3, 0.0
   );
   const amrex::Real lapse_derivative = (
      plus.metric.lapse - minus.metric.lapse
   ) / (2.0 * step);
   const amrex::Real gamma_derivative = (
      plus.metric.gamma11 - minus.metric.gamma11
   ) / (2.0 * step);
   if (std::abs(lapse_derivative - schwarzschild_geometry.derivatives.d1_lapse)
          > 2.0e-9
       || std::abs(gamma_derivative - schwarzschild_geometry.derivatives.d1_gamma11)
          > 2.0e-9) {
      std::cerr << "isotropic-Schwarzschild metric derivative mismatch\n";
      return false;
   }
   return true;
}

bool validate_srmhd_round_trips()
{
   constexpr amrex::Real gamma = 1.4;
   constexpr amrex::Real tolerance = 5.0e-10;
   const pyfluid_amrex::SpacetimeMetric minkowski{};
   std::array<pyfluid_amrex::RMHDPrimitive, 8> primitive_corpus{{
      {1.0e-3, 1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.071, 0.23, 0.0, 0.31, -0.17, 0.09, 0.04, -0.02, 0.01},
      {0.82, 4.7, 0.0, -0.71, 0.23, -0.11, -2.1, 3.7, -0.4},
      {1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 8.0, -3.0, 1.5},
      {7.4, 0.017, 0.0, 0.82, -0.31, 0.18, 0.003, -8.0, 2.5},
      {19.0, 13.0, 0.0, -0.91, 0.21, -0.07, 15.0, 0.2, -0.5},
      {0.3, 2.1, 0.0, 0.965, 0.08, -0.03, 0.9, -1.4, 0.7},
      {2.7, 0.08, 0.0, -0.44, -0.51, 0.37, -9.0, 4.0, 6.0},
   }};

   amrex::Real worst_primitive_error = 0.0;
   amrex::Real worst_conserved_error = 0.0;
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
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const pyfluid_amrex::RMHDEntropyInput entropy{
         lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0),
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
      independent_conserved.mom1 *= 0.997;
      independent_conserved.mom2 += 0.001 * independent_conserved.dens;
      independent_conserved.mom3 *= 1.002;
      independent_conserved.tau += 0.003 * (
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

      const amrex::Real x = (1.0 + gamma * primitive.eps) * lorentz_factor;
      const amrex::Real step = 1.0e-6 * std::max(amrex::Real{1.0}, x);
      const amrex::Real finite_difference = (
         pyfluid_amrex::srmhd_palenzuela_residual(conserved, x + step, gamma)
         - pyfluid_amrex::srmhd_palenzuela_residual(conserved, x - step, gamma)
      ) / (2.0 * step);
      const amrex::Real analytic =
         pyfluid_amrex::srmhd_palenzuela_residual_derivative(conserved, x, gamma);
      if (!std::isfinite(analytic)
          || std::abs(analytic - finite_difference)
             > 3.0e-6 * std::max(amrex::Real{1.0}, std::abs(finite_difference))) {
         std::cerr << "SRMHD generated residual derivative mismatch sample=" << sample
                   << " analytic=" << analytic
                   << " finite_difference=" << finite_difference << '\n';
         return false;
      }

      constexpr pyfluid_amrex::ColdPolytropeEquationOfState cold_eos{
         0.2, 1.8,
      };
      auto cold_primitive = primitive;
      cold_primitive.pressure = cold_eos.polytropic_k * std::pow(
         cold_primitive.rho, cold_eos.polytropic_gamma
      );
      cold_primitive.eps = cold_primitive.pressure / (
         cold_primitive.rho * (cold_eos.polytropic_gamma - 1.0)
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
   curved_metric.lapse = 0.83;
   curved_metric.shift1 = 0.07;
   curved_metric.shift2 = -0.03;
   curved_metric.shift3 = 0.02;
   curved_metric.gamma12 = 0.1;
   curved_metric.inv_gamma11 = 1.0 / 0.99;
   curved_metric.inv_gamma12 = -0.1 / 0.99;
   curved_metric.inv_gamma22 = 1.0 / 0.99;
   curved_metric.sqrt_det_gamma = std::sqrt(0.99);
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
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(
         1.0 - velocity_squared
      );
      const pyfluid_amrex::RMHDEntropyInput entropy{
         curved_metric.sqrt_det_gamma * lorentz_factor * primitive.pressure
            / std::pow(primitive.rho, gamma - 1.0),
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

pyfluid_amrex::HydroState hydro_llf_reference(
   pyfluid_amrex::HydroState const& flux_l,
   pyfluid_amrex::HydroState const& flux_r,
   pyfluid_amrex::HydroState const& state_l,
   pyfluid_amrex::HydroState const& state_r,
   amrex::Real wavespeed
)
{
   return {
      0.5 * (flux_l.rho + flux_r.rho) - 0.5 * wavespeed * (state_r.rho - state_l.rho),
      0.5 * (flux_l.mom1 + flux_r.mom1) - 0.5 * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5 * (flux_l.mom2 + flux_r.mom2) - 0.5 * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5 * (flux_l.mom3 + flux_r.mom3) - 0.5 * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5 * (flux_l.energy + flux_r.energy) - 0.5 * wavespeed * (state_r.energy - state_l.energy),
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
   const amrex::Real inv = 1.0 / (s_right - s_left);
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
      0.5 * (flux_l.rho + flux_r.rho) - 0.5 * wavespeed * (state_r.rho - state_l.rho),
      0.5 * (flux_l.mom1 + flux_r.mom1) - 0.5 * wavespeed * (state_r.mom1 - state_l.mom1),
      0.5 * (flux_l.mom2 + flux_r.mom2) - 0.5 * wavespeed * (state_r.mom2 - state_l.mom2),
      0.5 * (flux_l.mom3 + flux_r.mom3) - 0.5 * wavespeed * (state_r.mom3 - state_l.mom3),
      0.5 * (flux_l.energy + flux_r.energy) - 0.5 * wavespeed * (state_r.energy - state_l.energy),
      0.5 * (flux_l.b1 + flux_r.b1) - 0.5 * wavespeed * (state_r.b1 - state_l.b1),
      0.5 * (flux_l.b2 + flux_r.b2) - 0.5 * wavespeed * (state_r.b2 - state_l.b2),
      0.5 * (flux_l.b3 + flux_r.b3) - 0.5 * wavespeed * (state_r.b3 - state_l.b3),
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
   const amrex::Real inv = 1.0 / (s_right - s_left);
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
   amrex::Real value = (xm - 2.0 * center + xp) * inv_dx2;
   if (dim >= 2) {
      value += (ym - 2.0 * center + yp) * inv_dy2;
   }
   if (dim == 3) {
      value += (zm - 2.0 * center + zp) * inv_dz2;
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
   amrex::Real denominator = 1.0 + 2.0 * cx;
   if (dim >= 2) {
      numerator += cy * (ym + yp);
      denominator += 2.0 * cy;
   }
   if (dim == 3) {
      numerator += cz * (zm + zp);
      denominator += 2.0 * cz;
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
   amrex::Real value = center - base + cx * (2.0 * center - xm - xp);
   if (dim >= 2) {
      value += cy * (2.0 * center - ym - yp);
   }
   if (dim == 3) {
      value += cz * (2.0 * center - zm - zp);
   }
   return value;
}

bool validate_z4c_constraints()
{
   using pyfluid_amrex::Z4cSpatialDerivatives;
   using pyfluid_amrex::Z4cState;

   const Z4cState minkowski{};
   const Z4cSpatialDerivatives zero_derivatives{};
   Z4cState unprojected = minkowski;
   unprojected.chi = 0.8;
   unprojected.gamma_tilde11 = 2.0;
   unprojected.gamma_tilde22 = 3.0;
   unprojected.gamma_tilde33 = 4.0;
   unprojected.a_tilde11 = 0.3;
   unprojected.a_tilde22 = -0.1;
   unprojected.a_tilde33 = 0.2;
   const Z4cState projected =
      pyfluid_amrex::generated::z4c_project_algebraic_constraints(unprojected);
   const auto projected_metric =
      pyfluid_amrex::generated::z4c_conformal_metric(projected);
   const auto projected_constraints =
      pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
         projected, zero_derivatives
      );
   if (!close_enough(projected_metric.determinant, 1.0)
       || !close_enough(projected_constraints.trace_a_tilde, 0.0)
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
   if (!close_enough(flat.theta, 0.0)
       || !close_enough(flat.z_tilde1, 0.0)
       || !close_enough(flat.z_tilde2, 0.0)
       || !close_enough(flat.z_tilde3, 0.0)
       || !close_enough(flat.hamiltonian, 0.0)
       || !close_enough(flat.momentum1, 0.0)
       || !close_enough(flat.momentum2, 0.0)
       || !close_enough(flat.momentum3, 0.0)
       || !close_enough(flat.log_conformal_determinant, 0.0)
       || !close_enough(flat.trace_a_tilde, 0.0)) {
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
   if (!close_enough(flat_spacetime_metric.lapse, 1.0)
       || !close_enough(flat_spacetime_metric.gamma11, 1.0)
       || !close_enough(flat_spacetime_metric.gamma22, 1.0)
       || !close_enough(flat_spacetime_metric.gamma33, 1.0)
       || !close_enough(flat_spacetime_metric.inv_gamma11, 1.0)
       || !close_enough(flat_spacetime_metric.sqrt_det_gamma, 1.0)
       || !close_enough(flat_spacetime_derivatives.d1_lapse, 0.0)
       || !close_enough(flat_spacetime_derivatives.d1_gamma11, 0.0)
       || !close_enough(flat_spacetime_derivatives.extrinsic11, 0.0)) {
      std::cerr << "generated Z4c-to-Valencia Minkowski bridge mismatch\n";
      return false;
   }
   Z4cState bridge_state = projected;
   bridge_state.k_hat = 0.21;
   bridge_state.theta = -0.015;
   Z4cSpatialDerivatives bridge_derivatives{};
   bridge_derivatives.d_chi[0] = 0.07;
   bridge_derivatives.d_gamma_tilde[0][0] = -0.03;
   bridge_derivatives.d_lapse[0] = 0.02;
   bridge_derivatives.d_shift[0][1] = -0.04;
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
      bridge_state.k_hat + 2.0 * bridge_state.theta;
   const amrex::Real expected_extrinsic11 =
      bridge_state.a_tilde11 / bridge_state.chi
      + bridge_metric.gamma11 * total_curvature / 3.0;
   if (!close_enough(bridge_metric.gamma11, projected_metric.gamma11)
       || !close_enough(bridge_metric.inv_gamma23, projected_metric.inv_gamma23)
       || !close_enough(
          bridge_metric.sqrt_det_gamma, projected_metric.sqrt_det_gamma
       )
       || !close_enough(bridge_metric_derivatives.d1_lapse, 0.02)
       || !close_enough(bridge_metric_derivatives.d1_shift2, -0.04)
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
   if (!close_enough(flat_diagnostic_geometry.ricci_scalar, 0.0)) {
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
      0.02,
      0.0
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
      if (!close_enough(value, 0.0)) {
         std::cerr << "generated Z4c Minkowski RHS is nonzero\n";
         return false;
      }
   }
   if (!close_enough(
          pyfluid_amrex::generated::z4c_harmonic_lapse_rhs(minkowski, zero_derivatives),
          0.0
       )
       || !close_enough(
          pyfluid_amrex::generated::z4c_one_plus_log_lapse_rhs(minkowski, zero_derivatives),
          0.0
       )) {
      std::cerr << "generated Z4c Minkowski lapse RHS is nonzero\n";
      return false;
   }
   const auto flat_shift_rhs = pyfluid_amrex::generated::z4c_gamma_driver_shift_rhs(
      minkowski, zero_derivatives, 1.0, 0.5
   );
   if (!close_enough(flat_shift_rhs.shift1, 0.0)
       || !close_enough(flat_shift_rhs.shift2, 0.0)
       || !close_enough(flat_shift_rhs.shift3, 0.0)) {
      std::cerr << "generated Z4c Minkowski shift RHS is nonzero\n";
      return false;
   }

   // gamma_ij = psi^4 delta_ij has R = -8 psi^-5 Laplacian(psi).
   constexpr amrex::Real curvature = 0.03;
   constexpr amrex::Real x = 0.11;
   constexpr amrex::Real y = -0.07;
   constexpr amrex::Real z = 0.05;
   const amrex::Real radius_squared = x * x + y * y + z * z;
   const amrex::Real psi = 1.0 + curvature * radius_squared;
   Z4cState conformally_flat{};
   conformally_flat.chi = std::pow(psi, -4.0);
   Z4cSpatialDerivatives derivatives{};
   const amrex::Real d_psi[3] = {
      2.0 * curvature * x,
      2.0 * curvature * y,
      2.0 * curvature * z,
   };
   for (int first = 0; first < 3; ++first) {
      derivatives.d_chi[first] = -4.0 * std::pow(psi, -5.0) * d_psi[first];
      for (int second = 0; second < 3; ++second) {
         const amrex::Real dd_psi = first == second ? 2.0 * curvature : 0.0;
         derivatives.dd_chi[first][second] =
            20.0 * std::pow(psi, -6.0) * d_psi[first] * d_psi[second]
            - 4.0 * std::pow(psi, -5.0) * dd_psi;
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
   const amrex::Real expected_ricci = -48.0 * curvature * std::pow(psi, -5.0);
   if (std::abs(curved.hamiltonian - expected_ricci) > 2.0e-13
       || std::abs(curved_diagnostic_geometry.ricci_scalar - expected_ricci) > 2.0e-13
       || std::abs(curved_evolution_geometry.ricci_scalar - expected_ricci) > 2.0e-13
       || !close_enough(curved.log_conformal_determinant, 0.0)
       || !close_enough(curved.trace_a_tilde, 0.0)) {
      std::cerr << "generated Z4c conformally-flat curvature mismatch: "
                << curved.hamiltonian << " != " << expected_ricci << '\n';
      return false;
   }

   // det(gamma_tilde)=1 with gamma_tilde=diag(exp(2 f), exp(-2 f), 1)
   // and f=a*x exercises the evolved-Gamma form of the conformal Ricci tensor.
   constexpr amrex::Real slope = 0.17;
   constexpr amrex::Real f_value = 0.04;
   const amrex::Real exp_two_f = std::exp(2.0 * f_value);
   const amrex::Real exp_minus_two_f = std::exp(-2.0 * f_value);
   Z4cState anisotropic{};
   anisotropic.gamma_tilde11 = exp_two_f;
   anisotropic.gamma_tilde22 = exp_minus_two_f;
   anisotropic.gamma_tilde1 = 2.0 * slope * exp_minus_two_f;
   Z4cSpatialDerivatives anisotropic_derivatives{};
   anisotropic_derivatives.d_gamma_tilde[0][0] = 2.0 * slope * exp_two_f;
   anisotropic_derivatives.d_gamma_tilde[0][3] = -2.0 * slope * exp_minus_two_f;
   anisotropic_derivatives.dd_gamma_tilde[0][0][0] =
      4.0 * slope * slope * exp_two_f;
   anisotropic_derivatives.dd_gamma_tilde[0][0][3] =
      4.0 * slope * slope * exp_minus_two_f;
   anisotropic_derivatives.d_gamma_tilde_vector[0][0] =
      -4.0 * slope * slope * exp_minus_two_f;
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
      if (std::abs(diagnostic_ricci[component] - evolution_ricci[component]) > 3.0e-13) {
         std::cerr << "generated Z4c evolved-Gamma Ricci mismatch\n";
         return false;
      }
   }
   const auto anisotropic_constraints =
      pyfluid_amrex::generated::evaluate_vacuum_z4c_constraints(
         anisotropic, anisotropic_derivatives
      );
   if (std::abs(anisotropic_constraints.z_tilde1) > 2.0e-15
       || std::abs(anisotropic_constraints.z_tilde2) > 2.0e-15
       || std::abs(anisotropic_constraints.z_tilde3) > 2.0e-15) {
      std::cerr << "generated Z4c evolved-Gamma constraint mismatch\n";
      return false;
   }

   pyfluid_amrex::Z4cMatter matter{};
   matter.rho = 0.02;
   matter.momentum1 = 0.003;
   matter.momentum2 = -0.004;
   matter.momentum3 = 0.005;
   matter.stress11 = 0.011;
   matter.stress12 = -0.002;
   matter.stress13 = 0.004;
   matter.stress22 = 0.017;
   matter.stress23 = -0.003;
   matter.stress33 = 0.023;
   const auto sourced = pyfluid_amrex::generated::evaluate_z4c_constraints(
      minkowski, zero_derivatives, matter
   );
   if (!close_enough(sourced.hamiltonian, -16.0 * pyfluid_amrex::constants::pi * matter.rho)
       || !close_enough(sourced.momentum1, -8.0 * pyfluid_amrex::constants::pi * matter.momentum1)
       || !close_enough(sourced.momentum2, -8.0 * pyfluid_amrex::constants::pi * matter.momentum2)
       || !close_enough(sourced.momentum3, -8.0 * pyfluid_amrex::constants::pi * matter.momentum3)) {
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
      0.0,
      0.0
   );
   const amrex::Real stress_trace = matter.stress11 + matter.stress22 + matter.stress33;
   if (!close_enough(sourced_rhs.k_hat, 4.0 * pyfluid_amrex::constants::pi * (matter.rho + stress_trace))
       || !close_enough(
          sourced_rhs.a_tilde11,
          -8.0 * pyfluid_amrex::constants::pi * (matter.stress11 - stress_trace / 3.0)
       )
       || !close_enough(sourced_rhs.a_tilde12, -8.0 * pyfluid_amrex::constants::pi * matter.stress12)
       || !close_enough(sourced_rhs.a_tilde13, -8.0 * pyfluid_amrex::constants::pi * matter.stress13)
       || !close_enough(
          sourced_rhs.a_tilde22,
          -8.0 * pyfluid_amrex::constants::pi * (matter.stress22 - stress_trace / 3.0)
       )
       || !close_enough(sourced_rhs.a_tilde23, -8.0 * pyfluid_amrex::constants::pi * matter.stress23)
       || !close_enough(
          sourced_rhs.a_tilde33,
          -8.0 * pyfluid_amrex::constants::pi * (matter.stress33 - stress_trace / 3.0)
       )
       || !close_enough(sourced_rhs.gamma_tilde1, -16.0 * pyfluid_amrex::constants::pi * matter.momentum1)
       || !close_enough(sourced_rhs.gamma_tilde2, -16.0 * pyfluid_amrex::constants::pi * matter.momentum2)
       || !close_enough(sourced_rhs.gamma_tilde3, -16.0 * pyfluid_amrex::constants::pi * matter.momentum3)
       || !close_enough(sourced_rhs.theta, -8.0 * pyfluid_amrex::constants::pi * matter.rho)) {
      std::cerr << "generated Z4c matter RHS mismatch\n";
      return false;
   }

   Z4cState gauge_state{};
   gauge_state.k_hat = 0.2;
   gauge_state.lapse = 0.8;
   gauge_state.shift1 = 0.1;
   gauge_state.shift2 = -0.2;
   gauge_state.shift3 = 0.3;
   gauge_state.gamma_tilde1 = 0.04;
   gauge_state.gamma_tilde2 = -0.03;
   gauge_state.gamma_tilde3 = 0.02;
   Z4cSpatialDerivatives gauge_derivatives{};
   gauge_derivatives.advection_lapse[0] = 0.05;
   gauge_derivatives.advection_lapse[1] = -0.04;
   gauge_derivatives.advection_lapse[2] = 0.02;
   for (int direction = 0; direction < 3; ++direction) {
      for (int component = 0; component < 3; ++component) {
         gauge_derivatives.advection_shift[direction][component] =
            0.01 * static_cast<amrex::Real>(1 + 3 * direction + component);
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
          -2.0 * gauge_state.lapse * gauge_state.k_hat + lapse_advection
       )) {
      std::cerr << "generated Z4c lapse gauge mismatch\n";
      return false;
   }
   constexpr amrex::Real gauge_time = 0.7;
   constexpr amrex::Real slow_start_strength = 0.6;
   constexpr amrex::Real slow_start_width = 2.5;
   const amrex::Real conformal_weight = std::sqrt(gauge_state.chi);
   const amrex::Real expected_slow_start =
      -2.0 * gauge_state.lapse * gauge_state.k_hat + lapse_advection
      - conformal_weight * slow_start_strength
         * std::exp(
            -gauge_time * gauge_time
            / (2.0 * slow_start_width * slow_start_width)
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
   constexpr amrex::Real shift_mu = 0.75;
   constexpr amrex::Real shift_eta = 0.12;
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

int main()
{
   if (!validate_newtonian_round_trips()
       || !validate_piecewise_polytrope()
       || !validate_srhd_round_trips()
       || !validate_srmhd_round_trips()
       || !validate_fixed_spacetime_provider()
       || !validate_z4c_pointwise_bridge()
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
      const pyfluid_amrex::RHDPrimitive primitive{1.3, 0.2, 0.4, 0.2, -0.1, 0.05};
      const pyfluid_amrex::SpacetimeMetric metric{};
      const amrex::Real velocity_squared =
         primitive.v1 * primitive.v1 + primitive.v2 * primitive.v2 + primitive.v3 * primitive.v3;
      const amrex::Real lorentz_factor = 1.0 / std::sqrt(1.0 - velocity_squared);
      const amrex::Real enthalpy = 1.0 + primitive.eps + primitive.pressure / primitive.rho;
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
         0.0,
         0.0,
         0.0,
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
               primitive.eps, 5.0 / 3.0
            )
         );
      if (!std::isfinite(speeds.minus) || !std::isfinite(speeds.plus)
          || speeds.minus > speeds.plus || speeds.minus < -1.0 || speeds.plus > 1.0) {
         std::cerr << "generated SRHD wave-speed bounds mismatch\n";
         return 1;
      }
      const pyfluid_amrex::EOSParams eos{};
      const pyfluid_amrex::RHDPrimitive consistent{
         1.3,
         0.2,
         (eos.gamma - 1.0) * 1.3 * 0.2,
         0.2,
         -0.1,
         0.05,
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
      {1.2, 0.3, -0.2, 0.1, 2.5},
      {0.7, -1.4, 0.5, 0.25, 5.0},
      {3.0, 1.1, 2.2, -0.9, 12.0},
      {10.0, -0.02, 0.03, -0.04, 50.0},
   }};
   const std::array<pyfluid_amrex::MHDState, 4> mhd_states{{
      {1.2, 0.3, -0.2, 0.1, 2.5, 0.4, -0.1, 0.2},
      {0.7, -1.4, 0.5, 0.25, 5.0, -0.6, 0.2, 0.7},
      {3.0, 1.1, 2.2, -0.9, 12.0, 0.0, 1.3, -0.8},
      {10.0, -0.02, 0.03, -0.04, 50.0, 2.0, -1.5, 0.1},
   }};
   const std::array<amrex::Real, 4> pressures{{0.7, 2.5, 0.05, 20.0}};
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
      const pyfluid_amrex::HydroState state_l{1.1, 0.2, -0.3, 0.4, 2.7};
      const pyfluid_amrex::HydroState state_r{0.8, -0.5, 0.6, -0.1, 3.2};
      const pyfluid_amrex::HydroState flux_l{0.4, 1.2, -0.8, 0.25, 2.1};
      const pyfluid_amrex::HydroState flux_r{-0.2, 0.7, 0.9, -1.1, 1.6};
      constexpr amrex::Real wavespeed = 2.3;
      constexpr amrex::Real s_left = -1.7;
      constexpr amrex::Real s_right = 2.9;
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
      const pyfluid_amrex::MHDState state_l{1.1, 0.2, -0.3, 0.4, 2.7, 0.05, -0.4, 0.7};
      const pyfluid_amrex::MHDState state_r{0.8, -0.5, 0.6, -0.1, 3.2, -0.2, 0.3, -0.6};
      const pyfluid_amrex::MHDState flux_l{0.4, 1.2, -0.8, 0.25, 2.1, -0.9, 0.1, 0.8};
      const pyfluid_amrex::MHDState flux_r{-0.2, 0.7, 0.9, -1.1, 1.6, 0.6, -0.5, 0.2};
      constexpr amrex::Real wavespeed = 2.3;
      constexpr amrex::Real s_left = -1.7;
      constexpr amrex::Real s_right = 2.9;
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
   constexpr amrex::Real center = 1.4;
   constexpr amrex::Real base = -0.2;
   constexpr amrex::Real xm = 0.5;
   constexpr amrex::Real xp = 2.1;
   constexpr amrex::Real ym = -1.2;
   constexpr amrex::Real yp = 0.9;
   constexpr amrex::Real zm = 0.3;
   constexpr amrex::Real zp = -0.7;
   constexpr amrex::Real inv_dx2 = 2.0;
   constexpr amrex::Real inv_dy2 = 3.0;
   constexpr amrex::Real inv_dz2 = 5.0;
   constexpr amrex::Real cx = 0.11;
   constexpr amrex::Real cy = 0.07;
   constexpr amrex::Real cz = 0.03;
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
      const PolynomialGrid field{};
      constexpr int i = 2;
      constexpr int j = -1;
      constexpr int k = 3;
      constexpr int component = 0;
      const amrex::Real dx = pyfluid_amrex::generated::centered_first<4, 0>(
         field, i, j, k, component, 1.0
      );
      const amrex::Real dxx = pyfluid_amrex::generated::centered_second<4, 0>(
         field, i, j, k, component, 1.0
      );
      const amrex::Real dxy = pyfluid_amrex::generated::centered_mixed<4, 0, 1>(
         field, i, j, k, component, 1.0
      );
      const amrex::Real upwind = pyfluid_amrex::generated::upwind_first<4, 0>(
         field, i, j, k, component, 0.7, 1.0
      );
      const amrex::Real ko = pyfluid_amrex::generated::kreiss_oliger<4, 0>(
         field, i, j, k, component, 1.0
      );
      if (!close_enough(dx, 4.0 * i * i * i + 2.0 * j)
          || !close_enough(dxx, 12.0 * i * i)
          || !close_enough(dxy, 2.0)
          || !close_enough(upwind, 4.0 * i * i * i + 2.0 * j)
          || !close_enough(ko, 0.0)) {
         std::cerr << "generated differential-operator mismatch\n";
         return 1;
      }
      const amrex::Real face = pyfluid_amrex::generated::cell_to_face<0>(
         field, i, j, k, component
      );
      const amrex::Real expected_face = 0.5 * (
         field(i - 1, j, k, component) + field(i, j, k, component)
      );
      if (!close_enough(face, expected_face)) {
         std::cerr << "generated centering-transfer mismatch\n";
         return 1;
      }
      const bool centering_transfers_pass =
         check_centering_transfer<2, 1>()
         && check_centering_transfer<2, 2>()
         && check_centering_transfer<2, 3>()
         && check_centering_transfer<4, 1>()
         && check_centering_transfer<4, 2>()
         && check_centering_transfer<4, 3>()
         && check_centering_transfer<6, 1>()
         && check_centering_transfer<6, 2>()
         && check_centering_transfer<6, 3>()
         && check_centering_transfer<8, 1>()
         && check_centering_transfer<8, 2>()
         && check_centering_transfer<8, 3>()
         && check_topology_centering_transfer<2, 1>()
         && check_topology_centering_transfer<2, 2>()
         && check_topology_centering_transfer<2, 3>()
         && check_topology_centering_transfer<4, 1>()
         && check_topology_centering_transfer<4, 2>()
         && check_topology_centering_transfer<4, 3>()
         && check_topology_centering_transfer<6, 1>()
         && check_topology_centering_transfer<6, 2>()
         && check_topology_centering_transfer<6, 3>()
         && check_topology_centering_transfer<8, 1>()
         && check_topology_centering_transfer<8, 2>()
         && check_topology_centering_transfer<8, 3>();
      if (!centering_transfers_pass) {
         std::cerr << "generated high-order centering-transfer mismatch\n";
         return 1;
      }
   }
   const amrex::Real update = pyfluid_amrex::generated::explicit_diffusion_update(1.25, 0.125, -3.0);
   if (!close_enough(update, 1.25 + 0.125 * -3.0)) {
      std::cerr << "generated explicit diffusion update mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_x_cartesian(3.0, -2.0), 5.0)) {
      std::cerr << "generated cartesian curl x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_y_cartesian(0.5, 2.5), -2.0)) {
      std::cerr << "generated cartesian curl y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_z_cartesian(-1.5, 4.0), -5.5)) {
      std::cerr << "generated cartesian curl z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_cartesian(1.0, -2.0, 0.25), -0.75)) {
      std::cerr << "generated cartesian divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::curl_orthogonal_1(0.25, 7.0, -1.0), 2.0)) {
      std::cerr << "generated orthogonal curl mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::divergence_orthogonal(0.5, 1.0, 2.0, -0.5), 1.25)) {
      std::cerr << "generated orthogonal divergence mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_x(2.0, -3.0, 0.25, 4.0), -8.75)) {
      std::cerr << "generated ideal emf x mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_y(-3.0, 2.0, 4.0, 0.25), 8.75)) {
      std::cerr << "generated ideal emf y mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_emf_z(2.0, -3.0, 0.25, 4.0), -8.75)) {
      std::cerr << "generated ideal emf z mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::total_emf(-2.0, 0.5, 3.0), -0.5)) {
      std::cerr << "generated total emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::resistive_emf_scaled_gradient(0.5, 2.0, -3.0, 4.0), -0.75)) {
      std::cerr << "generated scaled-gradient resistive emf mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::magnetic_energy_density_from_components(1.0, 2.0, 3.0), 7.0)) {
      std::cerr << "generated magnetic energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::plasma_beta_from_pressure(4.0, 1.0e-300, 1.0, 2.0, 1.0), 4.0 / 3.0)) {
      std::cerr << "generated plasma beta mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_pressure_from_density_energy(3.0, 5.0 / 3.0), 2.0)) {
      std::cerr << "generated ideal-gas pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::ideal_gas_internal_energy_density_from_pressure(2.0, 5.0 / 3.0), 3.0)) {
      std::cerr << "generated ideal-gas energy mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::polytrope_cold_pressure(2.0, 0.5, 2.0), 2.0)) {
      std::cerr << "generated polytropic pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::hybrid_polytrope_pressure_from_density_energy(2.0, 4.0, 0.5, 2.0, 1.5, 0.0), 3.0)) {
      std::cerr << "generated hybrid-polytrope pressure mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_cell_volume(2.0, 3.0, 4.0), 24.0)) {
      std::cerr << "generated cartesian cell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cartesian_face_area_x(3.0, 4.0), 12.0)) {
      std::cerr << "generated cartesian face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_volume_scale(2.5, 0.7, 0.0), 2.5)) {
      std::cerr << "generated cylindrical volume scale mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_shell_volume_scale(2.0, 3.0), 2.5)) {
      std::cerr << "generated cylindrical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_radial_pressure_source_factor(2.0, 3.0), 0.4)) {
      std::cerr << "generated cylindrical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_cell_volume_scale(2.0, 3.0, 0.5), 3.0)) {
      std::cerr << "generated cylindrical phi-z volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_axial_face_area_scale(2.0, 0.5), 1.0)) {
      std::cerr << "generated cylindrical phi-z axial face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::cylindrical_phiz_azimuthal_face_area_scale(3.0), 3.0)) {
      std::cerr << "generated cylindrical phi-z azimuthal face area mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_scale_phi(2.0, 0.5, 0.0), 2.0 * std::sin(0.5))) {
      std::cerr << "generated spherical scale phi mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_shell_volume_scale(2.0, 3.0), 19.0 / 3.0)) {
      std::cerr << "generated spherical radial shell volume mismatch\n";
      return 1;
   }
   if (!close_enough(pyfluid_amrex::generated::spherical_radial_pressure_source_factor(2.0, 3.0), 15.0 / 19.0)) {
      std::cerr << "generated spherical radial pressure source mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_cell_volume_scale(2.0, 0.25, 0.75, 0.5),
          2.0 * (std::cos(0.25) - std::cos(0.75))
       )) {
      std::cerr << "generated spherical theta-phi volume mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_theta_face_area_scale(2.0, 0.25, 0.5),
          std::sin(0.25)
       )) {
      std::cerr << "generated spherical theta-phi theta face area mismatch\n";
      return 1;
   }
   if (!close_enough(
          pyfluid_amrex::generated::spherical_thetaphi_azimuthal_face_area_scale(2.0, 0.25, 0.75),
          1.0
       )) {
      std::cerr << "generated spherical theta-phi azimuthal face area mismatch\n";
      return 1;
   }
   {
      constexpr amrex::Real radial_inner = 1.0;
      constexpr amrex::Real radial_outer = 4.0;
      const auto point = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const amrex::Real tan_q1 = std::tan(0.2);
      const amrex::Real tan_q2 = std::tan(-0.1);
      const amrex::Real denom2 = 1.0 + tan_q1 * tan_q1 + tan_q2 * tan_q2;
      const amrex::Real radial_fraction = (2.0 - radial_inner) / (radial_outer - radial_inner);
      const amrex::Real blend = 1.0 - radial_fraction + radial_fraction / std::sqrt(denom2);
      const amrex::Real primary = -2.0 * blend;
      if (
         !close_enough(point.q1, 0.3 + primary)
         || !close_enough(point.q2, -0.2 + primary * tan_q2)
         || !close_enough(point.q3, 0.4 + primary * tan_q1)
      ) {
         std::cerr << "generated Thornburg patch-map transform mismatch\n";
         return 1;
      }
      const auto jacobian = pyfluid_amrex::generated::thornburg_shell_jacobian_axis0(
         0.2,
         -0.1,
         2.0,
         -1.0,
         radial_inner,
         radial_outer
      );
      constexpr amrex::Real eps = 1.0e-6;
      const auto point_plus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1 + eps,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const auto point_minus = pyfluid_amrex::generated::thornburg_shell_local_to_global_axis0(
         0.2,
         -0.1 - eps,
         2.0,
         0.3,
         -0.2,
         0.4,
         -1.0,
         radial_inner,
         radial_outer
      );
      const amrex::Real expected_j22 = (point_plus.q2 - point_minus.q2) / (2.0 * eps);
      if (std::abs(jacobian.j22 - expected_j22) > 1.0e-9) {
         std::cerr << "generated Thornburg patch-map Jacobian mismatch\n";
         return 1;
      }
   }
   {
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
      dx[0] = 0.5;
#if (AMREX_SPACEDIM >= 2)
      dx[1] = 0.25;
#endif
#if (AMREX_SPACEDIM == 3)
      dx[2] = 0.125;
#endif
      const pyfluid_amrex::CartesianUniformMetric metric = pyfluid_amrex::make_cartesian_uniform_metric(dx);
#if (AMREX_SPACEDIM == 1)
      const amrex::Real expected_volume = 0.5;
      const amrex::Real expected_area_x = 1.0;
#elif (AMREX_SPACEDIM == 2)
      const amrex::Real expected_volume = 0.5 * 0.25;
      const amrex::Real expected_area_x = 0.25;
#else
      const amrex::Real expected_volume = 0.5 * 0.25 * 0.125;
      const amrex::Real expected_area_x = 0.25 * 0.125;
#endif
      if (!close_enough(metric.cell_volume, expected_volume) || !close_enough(metric.face_area[0], expected_area_x)) {
         std::cerr << "cartesian metric provider mismatch\n";
         return 1;
      }
      const auto cart_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshCartesian>(2.0, 0.5, 0.0);
      if (!close_enough(cart_scale[0], 1.0) || !close_enough(cart_scale[1], 1.0) || !close_enough(cart_scale[2], 1.0)) {
         std::cerr << "cartesian coordinate scale provider mismatch\n";
         return 1;
      }
      const auto sph_scale = pyfluid_amrex::coordinate_scale_factors<pyfluid_amrex::MeshSpherical>(2.0, 0.5, 0.0);
      if (!close_enough(sph_scale[2], 2.0 * std::sin(0.5))) {
         std::cerr << "spherical coordinate scale provider mismatch\n";
         return 1;
      }
   }
   return 0;
}
