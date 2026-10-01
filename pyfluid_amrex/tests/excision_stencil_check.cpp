#include <AMReX.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pyfluid_amrex/kernels_generated/boundary_differential_ops.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"
#include "pyfluid_amrex/z4c_excision_derivatives.hpp"
#include "pyfluid_amrex/centering_transfer.hpp"
#include "pyfluid_amrex/nodal_interpolater.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace {

using namespace amrex::literals;

using amrex::Real;
namespace fd = pyfluid_amrex::generated;

AMREX_GPU_HOST_DEVICE
Real transfer_polynomial(amrex::GpuArray<Real, 3> const& point)
{
   const Real x = point[0], y = point[1], z = point[2];
   return 1.0_rt + x + 2.0_rt * y + 3.0_rt * z + 0.2_rt * x * x + 0.3_rt * y * y
      + 0.4_rt * z * z + 0.1_rt * x * y * z;
}

struct PoisonedMetricSample {
   pyfluid_amrex::ExcisionGridView grid;

   Real operator()(int i, int j, int k, int component) const
   {
      using namespace pyfluid_amrex;
      if (!grid(i, j, k, 0)) return std::numeric_limits<Real>::quiet_NaN();
      // Face transfers interpolate log-lapse: cubic exactness belongs to
      // that representation, not to the decoded positive scalar.
      if (component == Z4C_LAPSE) {
         return std::exp(0.01_rt * transfer_polynomial(grid.position(i, j, k)));
      }
      return component == Z4C_CHI || component == Z4C_GAMMA_TILDE11
         || component == Z4C_GAMMA_TILDE22 || component == Z4C_GAMMA_TILDE33 ? 1.0_rt : 0.0_rt;
   }
};

template <int Direction>
void check_face_metric()
{
   using namespace pyfluid_amrex;
   const ExcisionGridView grid{{{{0.07_rt, -0.04_rt, 0.03_rt}}, 0.45_rt},
      {{-1.0_rt, -1.0_rt, -1.0_rt}}, {{0.0625_rt, 0.0625_rt, 0.0625_rt}}, {{0.5_rt, 0.5_rt, 0.5_rt}}};
   const PoisonedMetricSample sample{grid};
   for (int k = 0; k < 32; ++k) for (int j = 0; j < 32; ++j) for (int i = 0; i < 32; ++i) {
      int left[3]{i, j, k};
      --left[Direction];
      if (!grid(i, j, k, 0) && !grid(left[0], left[1], left[2], 0)) continue;
      const auto metric = stored_spacetime_metric_at_face<4, Direction>(sample, i, j, k, grid);
      auto point = grid.position(i, j, k);
      point[Direction] -= 0.5_rt * grid.spacing[Direction];
      const Real error = std::abs(
         metric.lapse - std::exp(0.01_rt * transfer_polynomial(point)));
      if (!(error < 1.0e-12_rt)) {
         throw std::runtime_error("face metric sampled inactive data or lost log-cubic exactness");
      }
   }
}

void fill_transfer_source(amrex::MultiFab& field, pyfluid_amrex::ExcisionGridView grid)
{
   const Real poison = std::numeric_limits<Real>::quiet_NaN();
   for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
      const auto output = field.array(mfi);
      amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         output(i, j, k) = grid(i, j, k, 0) ? transfer_polynomial(grid.position(i, j, k)) : poison;
      });
   }
}

void check_transfer_error(amrex::MultiFab const& field, pyfluid_amrex::ExcisionGridView grid)
{
   amrex::MultiFab error(field.boxArray(), field.DistributionMap(), 1, 0);
   for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
      const auto input = field.const_array(mfi);
      const auto output = error.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         Real value = 0.0_rt;
         if (grid(i, j, k, 0)) {
            value = std::abs(input(i, j, k) - transfer_polynomial(grid.position(i, j, k)));
            if (!std::isfinite(value)) value = 1.0e30_rt;
         }
         output(i, j, k) = value;
      });
   }
   const Real maximum = error.norm0(0);
   if (!(maximum < 1.0e-10_rt)) {
      throw std::runtime_error("excision transfer sampled inactive data or lost cubic exactness: "
         + std::to_string(maximum));
   }
}

void check_transfers()
{
   using namespace pyfluid_amrex;
   const SphericalExcisionRegion region{{{0.07_rt, -0.04_rt, 0.03_rt}}, 0.45_rt};
   const amrex::Box coarse_domain(amrex::IntVect(0), amrex::IntVect(15));
   const amrex::RealBox physical({AMREX_D_DECL(-1.0_rt, -1.0_rt, -1.0_rt)},
      {AMREX_D_DECL(1.0_rt, 1.0_rt, 1.0_rt)});
   const int periodic[3]{0, 0, 0};
   const amrex::Geometry coarse_geometry(coarse_domain, &physical, 0, periodic);
   const amrex::BoxArray cell_boxes(coarse_domain);
   const amrex::DistributionMapping distribution(cell_boxes);
   const auto node_boxes = amrex::convert(cell_boxes, amrex::IntVect::TheNodeVector());
   amrex::MultiFab cells(cell_boxes, distribution, 1, 4);
   amrex::MultiFab nodes(node_boxes, distribution, 1, 4);
   const auto cell_grid = excision_grid_view(region, coarse_geometry, cells.ixType());
   const auto node_grid = excision_grid_view(region, coarse_geometry, nodes.ixType());
   fill_transfer_source(cells, cell_grid);
   transfer_grid_field_centering<4, 3>(nodes, cells, coarse_geometry, false, region);
   check_transfer_error(nodes, node_grid);
   fill_transfer_source(nodes, node_grid);
   transfer_grid_field_centering<4, 3>(cells, nodes, coarse_geometry, false, region);
   check_transfer_error(cells, cell_grid);
   for (int refinement : {2, 4}) {
      const auto fine_domain = amrex::refine(coarse_domain, refinement);
      const amrex::Geometry fine_geometry(fine_domain, &physical, 0, periodic);
      const amrex::BoxArray fine_boxes(amrex::convert(fine_domain, amrex::IntVect::TheNodeVector()));
      amrex::MultiFab fine(fine_boxes, distribution, 1, 0);
      NodalInterpolater interpolater(region);
      const amrex::Vector<amrex::BCRec> boundaries(1);
      interpolater.interp(nodes, 0, fine, 0, 1, amrex::IntVect(0), coarse_geometry,
         fine_geometry, fine_boxes[0], amrex::IntVect(refinement), boundaries, 0);
      check_transfer_error(fine, excision_grid_view(region, fine_geometry, fine.ixType()));
   }
   amrex::Print() << "PASS poisoned-interior centering transfers and nodal refinement ratios 2/4\n";
}

void require_close(Real value, Real expected, Real tolerance)
{
   if (!std::isfinite(value) || std::abs(value - expected) > tolerance) {
      throw std::runtime_error("excision stencil mismatch: "
         + std::to_string(value) + " expected " + std::to_string(expected));
   }
}

struct Polynomial {
   int degree;
   int side;
   Real spacing;
   bool mixed = false;

   Real operator()(int i, int j, int, int) const
   {
      // Unavailable samples must not even be evaluated, including zero-weight
      // samples: zero multiplied by NaN remains NaN.
      if (side * i < 0 || (mixed && side * j < 0)) {
         return std::numeric_limits<Real>::quiet_NaN();
      }
      const Real x = 0.7_rt + spacing * i;
      const Real y = 0.9_rt + spacing * j;
      return mixed ? x * x * y * y * y : std::pow(x, degree);
   }
};

struct SpacetimeDerivativeSample {
   bool metric_only = false;
   bool constraints_only = false;

   Real operator()(int i, int j, int k, int component) const
   {
      using namespace pyfluid_amrex;
      bool metric_component = component == Z4C_CHI;
      for (int metric_index : z4c_gamma_tilde_components()) {
         metric_component = metric_component || component == metric_index;
      }
      if (metric_only && !metric_component) {
         throw std::runtime_error("connection gather read a non-metric component");
      }
      bool constraint_component = metric_component || component == Z4C_K_HAT || component == Z4C_THETA;
      for (int curvature_index : z4c_a_tilde_components()) {
         constraint_component = constraint_component || component == curvature_index;
      }
      if (constraints_only && !constraint_component) {
         throw std::runtime_error("constraint gather read gauge or connection derivatives");
      }
      return 0.01_rt * (component + 1) * std::sin(0.13_rt * i + 0.21_rt * j - 0.17_rt * k);
   }
};

template <typename Operator>
struct ConstraintDerivativeOperator : Operator {
   explicit ConstraintDerivativeOperator(Operator const& derivative_operator)
      : Operator(derivative_operator) {}

   // A constraint gather must compile with no transport operation available.
   template <int Direction, typename Array>
   Real advection(Array const&, int, int, int, int, Real, Real) const = delete;
};

template <int Order, int Dim, typename DerivativeOperator>
void check_derivative_subsets(DerivativeOperator const& derivative_operator)
{
   using namespace pyfluid_amrex;
   const amrex::GpuArray<Real, 3> inv_spacing{{2.0_rt, 3.0_rt, 5.0_rt}};
   const auto complete = gather_z4c_spatial_derivatives<Order, Dim>(
      SpacetimeDerivativeSample{}, 2, -1, 3, inv_spacing, Z4cState{}, derivative_operator);
   const auto metric = gather_z4c_metric_derivatives<Order, Dim>(
      SpacetimeDerivativeSample{true}, 2, -1, 3, inv_spacing, derivative_operator);
   const auto constraints = gather_z4c_spatial_derivatives<Order, Dim, Z4cDerivativePurpose::Constraints>(
      SpacetimeDerivativeSample{false, true}, 2, -1, 3, inv_spacing, Z4cState{},
      ConstraintDerivativeOperator<DerivativeOperator>{derivative_operator});
   for (int direction = 0; direction < 3; ++direction) {
      require_close(metric.d_chi[direction], complete.d_chi[direction], 0.0_rt);
      require_close(constraints.d_chi[direction], complete.d_chi[direction], 0.0_rt);
      require_close(constraints.d_k_hat[direction], complete.d_k_hat[direction], 0.0_rt);
      require_close(constraints.d_theta[direction], complete.d_theta[direction], 0.0_rt);
      for (int component = 0; component < 6; ++component) {
         require_close(metric.d_gamma_tilde[direction][component],
            complete.d_gamma_tilde[direction][component], 0.0_rt);
         require_close(constraints.d_gamma_tilde[direction][component],
            complete.d_gamma_tilde[direction][component], 0.0_rt);
         require_close(constraints.d_a_tilde[direction][component],
            complete.d_a_tilde[direction][component], 0.0_rt);
      }
      for (int second = 0; second < 3; ++second) {
         require_close(constraints.dd_chi[direction][second],
            complete.dd_chi[direction][second], 0.0_rt);
         for (int component = 0; component < 6; ++component) {
            require_close(constraints.dd_gamma_tilde[direction][second][component],
               complete.dd_gamma_tilde[direction][second][component], 0.0_rt);
         }
      }
   }
}

template <int Dim>
void check_derivative_subset_orders()
{
   using namespace pyfluid_amrex;
   // Exercise the catalog, including the highest order, without a second list.
   for (int order = 2; order <= maximum_spatial_order; ++order) {
      if (!is_supported_spatial_order(order)) continue;
      dispatch_spatial_order(order, [](auto order_tag) {
         constexpr int spatial_order = decltype(order_tag)::value;
         check_derivative_subsets<spatial_order, Dim>(CenteredZ4cDerivativeOperator<spatial_order>{});
         for (int side : {-1, 1}) {
            check_derivative_subsets<spatial_order, Dim>(
               ExcisionZ4cDerivativeOperator<spatial_order>{amrex::GpuArray<int, 3>{{side, 0, -side}}});
         }
      });
   }
}

template <int Order>
void check_polynomials()
{
   constexpr Real spacing = 0.125_rt;
   for (int side : {-1, 1}) {
      for (int degree = 0; degree <= Order; ++degree) {
         const Polynomial field{degree, side, spacing};
         const Real first = degree == 0 ? 0.0_rt
            : degree * std::pow(0.7_rt, degree - 1);
         require_close(fd::one_sided_first<Order, 0>(
            field, 0, 0, 0, 0, side, 1.0_rt / spacing), first, 2.0e-9_rt);
      }
      for (int degree = 0; degree <= Order + 1; ++degree) {
         const Polynomial field{degree, side, spacing};
         const Real second = degree < 2 ? 0.0_rt
            : degree * (degree - 1) * std::pow(0.7_rt, degree - 2);
         require_close(fd::one_sided_second<Order, 0>(
            field, 0, 0, 0, 0, side, 1.0_rt / (spacing * spacing)),
            second, 2.0e-8_rt);
      }
      if constexpr (Order >= 4) {
         const Polynomial field{0, side, spacing, true};
         require_close(fd::one_sided_mixed<Order, 0, 1>(
            field, 0, 0, 0, 0, side, side, 1.0_rt / (spacing * spacing)),
            6.0_rt * 0.7_rt * 0.9_rt * 0.9_rt, 2.0e-8_rt);
      }
   }
}

struct ActiveLine {
   Real operator()(int i, int, int, int) const
   {
      return (i >= 0 && i < 71 && !(i >= 23 && i < 31)) ? 1.0_rt : 0.0_rt;
   }
};

struct LineField {
   Real operator()(int i, int j, int k, int component) const
   {
      if (!ActiveLine{}(i, j, k, component)) {
         return std::numeric_limits<Real>::quiet_NaN();
      }
      return std::sin(1.31_rt * i) + 0.3_rt * std::cos(0.17_rt * i);
   }
};

template <int Order>
void check_dissipation()
{
   const LineField field;
   const ActiveLine active;
   Real energy_rate = 0.0_rt;
   for (int i = -5; i < 76; ++i) {
      const Real value = fd::masked_kreiss_oliger<Order, 0>(
         field, active, i, 0, 0, 0, 1.0_rt);
      if (active(i, 0, 0, 0)) {
         if (!std::isfinite(value)) {
            throw std::runtime_error("excision dissipation read poisoned data");
         }
         energy_rate += field(i, 0, 0, 0) * value;
      } else {
         require_close(value, 0.0_rt, 0.0_rt);
      }
   }
   if (!(energy_rate < 0.0_rt)) {
      throw std::runtime_error("excision dissipation amplifies energy");
   }
   require_close(fd::masked_kreiss_oliger<Order, 0>(
      field, active, 50, 0, 0, 0, 1.0_rt),
      fd::kreiss_oliger<Order, 0>(field, 50, 0, 0, 0, 1.0_rt), 1.0e-13_rt);
}

struct NyquistMode {
   Real operator()(int i, int, int, int) const
   {
      return i % 2 == 0 ? 1.0_rt : -1.0_rt;
   }
};

template <int Order>
void check_shift_advection()
{
   // The generated Z4c RHS contains +beta*d_x u. Its transport velocity
   // is -beta, so an upwind operator must damp, not amplify, this mode.
   const NyquistMode field;
   const pyfluid_amrex::CenteredZ4cDerivativeOperator<Order> derivatives;
   for (Real shift : {-0.4_rt, 0.4_rt}) {
      Real energy_rate = 0.0_rt;
      for (int i = 0; i < 64; ++i) {
         energy_rate += field(i, 0, 0, 0) * shift
            * derivatives.template advection<0>(field, i, 0, 0, 0, shift, 1.0_rt);
      }
      amrex::Print() << "shift-advection order=" << Order
         << " beta=" << shift << " energy_rate=" << energy_rate << '\n';
      if (!(energy_rate < 0.0_rt)) {
         throw std::runtime_error("Z4c shift advection is downwind");
      }
   }
}

struct WaveState {
   std::vector<Real> field;
   std::vector<Real> velocity;
};

Real pulse(Real x)
{
   return std::exp(-80.0_rt * (x - 0.5_rt) * (x - 0.5_rt));
}

// A shifted wave has the same second-order spatial principal structure as
// the geometry subsystem. Both characteristics leave through x=0 (beta>c).
// This checks a complete boundary evolution, not only polynomial exactness.
Real check_outflow_wave(int cells)
{
   constexpr Real shift = 1.2_rt;
   constexpr Real wave_speed = 0.4_rt;
   constexpr Real end_time = 1.0_rt;
   const Real spacing = 1.0_rt / cells;
   const int steps = static_cast<int>(std::ceil(
      end_time / (0.2_rt * spacing / (shift + wave_speed))));
   const Real dt = end_time / steps;
   WaveState state{std::vector<Real>(cells + 1), std::vector<Real>(cells + 1)};
   for (int i = 0; i <= cells; ++i) {
      const Real x = i * spacing;
      state.field[i] = pulse(x);
      state.velocity[i] = -160.0_rt * wave_speed * (x - 0.5_rt) * pulse(x);
   }
   const auto rhs = [&](WaveState const& input, Real time) {
      WaveState result{std::vector<Real>(cells + 1), std::vector<Real>(cells + 1)};
      const auto sample = [&](int i, int, int, int component) -> Real {
         if (i < 0) return std::numeric_limits<Real>::quiet_NaN();
         if (i <= cells) return component == 0 ? input.field[i] : input.velocity[i];
         // Exact incoming data at the unrelated right boundary.
         const Real x = i * spacing + (shift + wave_speed) * time;
         return component == 0 ? pulse(x)
            : -160.0_rt * wave_speed * (x - 0.5_rt) * pulse(x);
      };
      const auto active = [](int i, int, int, int) { return i >= 0; };
      for (int i = 0; i <= cells; ++i) {
         const pyfluid_amrex::ExcisionZ4cDerivativeOperator<4> derivatives{
            amrex::GpuArray<int, 3>{{i < 3 ? 1 : 0, 0, 0}}};
         result.field[i] = shift * derivatives.advection<0>(
            sample, i, 0, 0, 0, shift, 1.0_rt / spacing) + input.velocity[i];
         result.velocity[i] = shift * derivatives.advection<0>(
            sample, i, 0, 0, 1, shift, 1.0_rt / spacing)
            + wave_speed * wave_speed * derivatives.second<0>(
               sample, i, 0, 0, 0, 1.0_rt / (spacing * spacing));
         result.field[i] += 0.32_rt * fd::masked_kreiss_oliger<4, 0>(
            sample, active, i, 0, 0, 0, 1.0_rt / spacing);
         result.velocity[i] += 0.32_rt * fd::masked_kreiss_oliger<4, 0>(
            sample, active, i, 0, 0, 1, 1.0_rt / spacing);
      }
      return result;
   };
   const auto stage = [&](WaveState const& slope, Real factor) {
      WaveState result = state;
      for (int i = 0; i <= cells; ++i) {
         result.field[i] += factor * dt * slope.field[i];
         result.velocity[i] += factor * dt * slope.velocity[i];
      }
      return result;
   };
   Real peak_error = 0.0_rt;
   for (int step = 0; step < steps; ++step) {
      const Real time = step * dt;
      const auto k1 = rhs(state, time);
      const auto k2 = rhs(stage(k1, 0.5_rt), time + 0.5_rt * dt);
      const auto k3 = rhs(stage(k2, 0.5_rt), time + 0.5_rt * dt);
      const auto k4 = rhs(stage(k3, 1.0_rt), time + dt);
      for (int i = 0; i <= cells; ++i) {
         state.field[i] += dt / 6.0_rt * (k1.field[i] + 2.0_rt * k2.field[i]
            + 2.0_rt * k3.field[i] + k4.field[i]);
         state.velocity[i] += dt / 6.0_rt * (k1.velocity[i] + 2.0_rt * k2.velocity[i]
            + 2.0_rt * k3.velocity[i] + k4.velocity[i]);
         const Real exact = pulse(i * spacing + (shift + wave_speed) * (time + dt));
         if (!std::isfinite(state.field[i])) {
            throw std::runtime_error("outflow wave boundary became nonfinite");
         }
         peak_error = std::max(peak_error, std::abs(state.field[i] - exact));
      }
   }
   amrex::Print() << "outflow-wave cells=" << cells << " peak_error=" << peak_error << '\n';
   return peak_error;
}

Real check_round_outflow_wave(int cells)
{
   constexpr Real shift_scale = 3.0_rt;
   constexpr Real wave_speed = 0.4_rt;
   constexpr Real end_time = 0.6_rt;
   const Real spacing = 2.0_rt / cells;
   const int width = cells + 1;
   const int steps = static_cast<int>(std::ceil(end_time
      / (0.2_rt * spacing / (2.0_rt * shift_scale + wave_speed))));
   const Real dt = end_time / steps;
   const pyfluid_amrex::ExcisionGridView grid{
      {{{0.0_rt, 0.0_rt, 0.0_rt}}, 0.45_rt}, {{-1.0_rt, -1.0_rt, 0.0_rt}},
      {{spacing, spacing, spacing}}, {{0.0_rt, 0.0_rt, 0.0_rt}}};
   const auto exact = [&](int i, int j, Real time) {
      const Real x = -1.0_rt + i * spacing;
      const Real y = -1.0_rt + j * spacing;
      const Real decay = std::exp(-time);
      const Real u = decay * std::sin(2.0_rt * x) * std::cos(3.0_rt * y);
      const Real ux = 2.0_rt * decay * std::cos(2.0_rt * x) * std::cos(3.0_rt * y);
      const Real uy = -3.0_rt * decay * std::sin(2.0_rt * x) * std::sin(3.0_rt * y);
      const Real uxx = -4.0_rt * u;
      const Real uyy = -9.0_rt * u;
      const Real uxy = -6.0_rt * decay * std::cos(2.0_rt * x) * std::sin(3.0_rt * y);
      const Real v = -u - shift_scale * (x * ux + y * uy);
      const Real vx = -ux - shift_scale * (ux + x * uxx + y * uxy);
      const Real vy = -uy - shift_scale * (uy + y * uyy + x * uxy);
      return std::array<Real, 3>{{u, v, -v - shift_scale * (x * vx + y * vy)
         - wave_speed * wave_speed * (uxx + uyy)}};
   };
   WaveState state{std::vector<Real>(width * width), std::vector<Real>(width * width)};
   for (int j = 0; j <= cells; ++j) for (int i = 0; i <= cells; ++i) {
      const auto value = exact(i, j, 0.0_rt);
      state.field[i + width * j] = value[0];
      state.velocity[i + width * j] = value[1];
   }
   const auto rhs = [&](WaveState const& input, Real time) {
      WaveState result{std::vector<Real>(width * width), std::vector<Real>(width * width)};
      const auto sample = [&](int i, int j, int, int component) -> Real {
         if (!grid(i, j, 0, 0)) return std::numeric_limits<Real>::quiet_NaN();
         if (i < 0 || j < 0 || i > cells || j > cells) return exact(i, j, time)[component];
         return component == 0 ? input.field[i + width * j] : input.velocity[i + width * j];
      };
      for (int j = 0; j <= cells; ++j) for (int i = 0; i <= cells; ++i) {
         if (!grid(i, j, 0, 0)) continue;
         const int index = i + width * j;
         const Real shift1 = shift_scale * (-1.0_rt + spacing * i);
         const Real shift2 = shift_scale * (-1.0_rt + spacing * j);
         const pyfluid_amrex::ExcisionZ4cDerivativeOperator<4> derivatives{
            grid.derivative_stencil(i, j, 0, 3)};
         result.field[index] = input.velocity[index];
         result.velocity[index] = exact(i, j, time)[2]
            + wave_speed * wave_speed * (derivatives.second<0>(sample, i, j, 0, 0,
               1.0_rt / (spacing * spacing)) + derivatives.second<1>(sample, i, j, 0, 0,
               1.0_rt / (spacing * spacing)));
         for (int component = 0; component < 2; ++component) {
            Real& value = component == 0 ? result.field[index] : result.velocity[index];
            value += shift1 * derivatives.advection<0>(sample, i, j, 0, component, shift1,
               1.0_rt / spacing) + shift2 * derivatives.advection<1>(sample, i, j, 0, component,
               shift2, 1.0_rt / spacing);
            value += 0.32_rt * (fd::masked_kreiss_oliger<4, 0>(sample, grid, i, j, 0, component,
               1.0_rt / spacing) + fd::masked_kreiss_oliger<4, 1>(sample, grid, i, j, 0, component,
               1.0_rt / spacing));
         }
      }
      return result;
   };
   const auto stage = [&](WaveState const& slope, Real factor) {
      WaveState result = state;
      for (int index = 0; index < width * width; ++index) {
         result.field[index] += factor * dt * slope.field[index];
         result.velocity[index] += factor * dt * slope.velocity[index];
      }
      return result;
   };
   Real peak_error = 0.0_rt;
   for (int step = 0; step < steps; ++step) {
      const Real time = step * dt;
      const auto k1 = rhs(state, time);
      const auto k2 = rhs(stage(k1, 0.5_rt), time + 0.5_rt * dt);
      const auto k3 = rhs(stage(k2, 0.5_rt), time + 0.5_rt * dt);
      const auto k4 = rhs(stage(k3, 1.0_rt), time + dt);
      for (int j = 0; j <= cells; ++j) for (int i = 0; i <= cells; ++i) {
         const int index = i + width * j;
         state.field[index] += dt / 6.0_rt * (k1.field[index] + 2.0_rt * k2.field[index]
            + 2.0_rt * k3.field[index] + k4.field[index]);
         state.velocity[index] += dt / 6.0_rt * (k1.velocity[index] + 2.0_rt * k2.velocity[index]
            + 2.0_rt * k3.velocity[index] + k4.velocity[index]);
         if (!grid(i, j, 0, 0)) continue;
         if (!std::isfinite(state.field[index])) {
            throw std::runtime_error("round outflow wave boundary became nonfinite");
         }
         peak_error = std::max(peak_error, std::abs(state.field[index] - exact(i, j, time + dt)[0]));
      }
   }
   amrex::Print() << "round-outflow-wave cells=" << cells << " peak_error=" << peak_error << '\n';
   return peak_error;
}

} // namespace

int main(int argc, char** argv)
{
   amrex::Initialize(argc, argv);
   int result = 0;
   try {
      check_transfers();
      check_derivative_subset_orders<1>();
      check_derivative_subset_orders<2>();
      check_derivative_subset_orders<3>();
      check_face_metric<0>();
      check_face_metric<1>();
      check_face_metric<2>();
      check_polynomials<2>();
      check_polynomials<4>();
      check_polynomials<6>();
      check_polynomials<8>();
      check_dissipation<2>();
      check_dissipation<4>();
      check_dissipation<6>();
      check_dissipation<8>();
      check_shift_advection<2>();
      check_shift_advection<4>();
      check_shift_advection<6>();
      check_shift_advection<8>();
      const Real coarse_wave_error = check_outflow_wave(64);
      const Real fine_wave_error = check_outflow_wave(128);
      if (!(fine_wave_error < coarse_wave_error / 8.0_rt && coarse_wave_error < 0.01_rt)) {
         throw std::runtime_error("outflow wave boundary is unstable or nonconvergent");
      }
      const Real coarse_round_error = check_round_outflow_wave(32);
      const Real fine_round_error = check_round_outflow_wave(64);
      if (!(fine_round_error < coarse_round_error / 8.0_rt && coarse_round_error < 0.1_rt)) {
         throw std::runtime_error("round outflow wave boundary is unstable or nonconvergent");
      }
      amrex::Print() << "PASS excision stencils: polynomial exactness, poisoned "
         "interior exclusion, nonpositive KO energy, centered KO equivalence\n";
   } catch (std::exception const& error) {
      amrex::Print() << "FAIL " << error.what() << '\n';
      result = 1;
   }
   amrex::Finalize();
   return result;
}
