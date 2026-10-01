#include "pyfluid_amrex/fccz4_runtime.hpp"

#include "pyfluid_amrex/fccz4_state_access.hpp"
#include "pyfluid_amrex/eulerian_matter_access.hpp"
#include "pyfluid_amrex/kernels_generated/fccz4.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/gauge.hpp"
#include "pyfluid_amrex/spacetime_derivatives.hpp"
#include "pyfluid_amrex/z4c_runtime.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real transport_correction(
   ReferenceChartPoint const& chart,
   Fccz4Point const& point,
   amrex::Real const* upwind,
   amrex::Real const* centered
) noexcept
{
   amrex::Real result = 0.0_rt;
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const amrex::Real speed = point.shift.value[chart.canonical[d]]
         * reference_inverse_scale(chart, d);
      result += speed * (upwind[d] - centered[d]);
   }
   return result;
}

template <int spatial_order>
void fill_fccz4_connections(
   amrex::Box const& box,
   amrex::Array4<amrex::Real> values,
   ReferenceChartGeometry const& chart_geometry
)
{
   amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const auto derivatives = gather_z4c_metric_derivatives<spatial_order, AMREX_SPACEDIM>(
         values, i, j, k, chart_geometry.inverse_spacing);
      const auto chart = chart_geometry.at(i, j, k);
      const auto point = gather_fccz4_point(values, i, j, k, derivatives, chart);
      const auto connection = generated::reference_metric_connection(point.metric);
      for (int a = 0; a < 3; ++a) {
         values(i, j, k, z4c_gamma_tilde_vector_components()[a]) =
            connection.contracted[chart.canonical[a]];
      }
   });
}

} // namespace

void validate_fccz4_domain(amrex::Geometry const& geometry, RunConfig const& config)
{
   const ReferenceChartGeometry chart(geometry, config);
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const int axis = mesh_axis(chart.axes, d);
      if (axis == MeshAxisR && !(geometry.ProbLo(d) > 0.0_rt)) {
         amrex::Abort("vertex-centered fCCZ4 currently requires a regular chart with r > 0; "
                      "origin regularity is not implemented");
      }
      if (axis == MeshAxisTheta
          && !(geometry.ProbLo(d) > 0.0_rt && geometry.ProbHi(d) < constants::pi)) {
         amrex::Abort("vertex-centered fCCZ4 currently requires 0 < theta < pi; "
                      "polar regularity is not implemented");
      }
   }
}

void compute_fccz4_rhs(
   amrex::MultiFab const& state,
   amrex::MultiFab const& matter,
   amrex::MultiFab& rhs,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   amrex::Real time
)
{
   const ReferenceChartGeometry chart_geometry(geometry, config);
   const int derivative_order = config.spacetime_derivative_order;
   const auto parameters = z4c_evolution_parameters(config);
   const auto lapse_gauge = z4c_lapse_gauge(config);
   const auto shift_gauge = z4c_shift_gauge(config);
   const Fccz4Damping damping{config.spacetime_kappa1, config.spacetime_kappa2, 1.0_rt};
   const bool matter_coupled = config.spacetime_matter_coupling == "coupled";
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const auto input = state.const_array(mfi);
      const auto sources = matter.const_array(mfi);
      const auto output = rhs.array(mfi);
      amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto chart = chart_geometry.at(i, j, k);
         const auto local = load_z4c_state(input, i, j, k);
         const auto derivatives = gather_spacetime_derivatives(derivative_order,
            input, i, j, k, chart_geometry.inverse_spacing, local);
         const auto point = gather_fccz4_point(input, i, j, k, derivatives, chart);
         const auto prepared = generated::fccz4_geometry(point);
         const auto source = matter_coupled
            ? canonical_matter(load_eulerian_matter(sources, i, j, k), chart)
            : EulerianMatter{};
         const auto result = generated::fccz4_rhs(point, source, damping, prepared);
         output(i, j, k, Z4C_CHI) = result.chi
            + transport_correction(chart, point, derivatives.advection_chi, derivatives.d_chi);
         output(i, j, k, Z4C_K_HAT) = result.trace_k - 2.0_rt * result.theta
            + transport_correction(chart, point, derivatives.advection_k_hat, derivatives.d_k_hat);
         output(i, j, k, Z4C_THETA) = result.theta
            + transport_correction(chart, point, derivatives.advection_theta, derivatives.d_theta);
         for (int a = 0; a < 3; ++a) {
            amrex::Real connection_correction = 0.0_rt;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
               const amrex::Real speed = point.shift.value[chart.canonical[d]]
                  * reference_inverse_scale(chart, d);
               connection_correction += speed * (derivatives.advection_gamma_tilde_vector[d][a]
                                                  - derivatives.d_gamma_tilde_vector[d][a]);
            }
            output(i, j, k, z4c_gamma_tilde_vector_components()[a]) =
               result.connection[chart.canonical[a]] + connection_correction;
            for (int b = a; b < 3; ++b) {
               const int component = symmetric_tensor_component(a, b);
               const int canonical_component = symmetric_tensor_component(
                  chart.canonical[a], chart.canonical[b]
               );
               amrex::Real metric_correction = 0.0_rt;
               amrex::Real a_correction = 0.0_rt;
               for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                  const amrex::Real speed = point.shift.value[chart.canonical[d]]
                  * reference_inverse_scale(chart, d);
                  metric_correction += speed * (
                     derivatives.advection_gamma_tilde[d][component]
                     - derivatives.d_gamma_tilde[d][component]
                  );
                  a_correction += speed * (
                     derivatives.advection_a_tilde[d][component]
                     - derivatives.d_a_tilde[d][component]
                  );
               }
               output(i, j, k, z4c_gamma_tilde_components()[component]) =
                  result.metric[canonical_component] + metric_correction;
               output(i, j, k, z4c_a_tilde_components()[component]) =
                  result.a_tilde[canonical_component] + a_correction;
            }
         }
         // Reuse the existing scalar slicing and integrated Gamma-driver gauge
         // convention, supplying physical covariant transport derivatives.
         auto gauge_derivatives = derivatives;
         for (int d = 0; d < 3; ++d) {
            const amrex::Real inverse_scale = reference_inverse_scale(chart, d);
            gauge_derivatives.advection_lapse[d] *= inverse_scale;
            for (int a = 0; a < 3; ++a) {
               gauge_derivatives.advection_shift[d][a] = point.shift.first[chart.canonical[d]][chart.canonical[a]]
                  + inverse_scale * (derivatives.advection_shift[d][a] - derivatives.d_shift[d][a]);
            }
         }
         amrex::Real lapse_rhs = 0.0_rt;
         if (lapse_gauge == Z4cLapseGauge::Harmonic) {
            lapse_rhs = generated::z4c_harmonic_lapse_rhs(local, gauge_derivatives);
         } else if (lapse_gauge == Z4cLapseGauge::OnePlusLog) {
            lapse_rhs = generated::z4c_one_plus_log_lapse_rhs(local, gauge_derivatives);
         } else if (lapse_gauge == Z4cLapseGauge::Trumpet) {
            lapse_rhs = generated::z4c_trumpet_lapse_rhs(local, gauge_derivatives);
         } else if (lapse_gauge == Z4cLapseGauge::SlowStart) {
            lapse_rhs = generated::z4c_slow_start_lapse_rhs(
               local, gauge_derivatives, time,
               parameters.slow_start_strength, parameters.slow_start_width
            );
         }
         output(i, j, k, Z4C_LAPSE) = lapse_rhs;
         Z4cShiftRhs shift_rhs{};
         if (shift_gauge == Z4cShiftGauge::GammaDriver) {
            shift_rhs = generated::z4c_gamma_driver_shift_rhs(
               local, gauge_derivatives, parameters.shift_mu, parameters.shift_eta
            );
         }
         output(i, j, k, z4c_shift_components()[0]) = shift_rhs.shift1;
         output(i, j, k, z4c_shift_components()[1]) = shift_rhs.shift2;
         output(i, j, k, z4c_shift_components()[2]) = shift_rhs.shift3;
      });
   }
}

void complete_fccz4_connections(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   const ReferenceChartGeometry chart_geometry(geometry, config);
   const int derivative_order = config.spacetime_derivative_order;
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const auto values = state.array(mfi);
      // The chart conversion needs metric values and first derivatives only.
      // Share the narrowed gather with Cartesian connection initialization.
      dispatch_spatial_order(derivative_order, [&](auto order_tag) {
         constexpr int spatial_order = decltype(order_tag)::value;
         fill_fccz4_connections<spatial_order>(mfi.tilebox(), values, chart_geometry);
      });
   }
}

void rotate_fccz4_initial_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   const ReferenceChartGeometry chart_geometry(geometry, config);
   for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
      const auto values = state.array(mfi);
      amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const auto chart = chart_geometry.at(i, j, k);
         const auto q = chart.position;
         const auto basis = chart.coordinate == MeshSpherical
            ? generated::spherical_reference_basis(q[0], q[1], q[2])
            : (chart.coordinate == MeshCylindrical
               ? generated::cylindrical_reference_basis(q[0], q[1], q[2])
               : generated::cartesian_reference_basis(q[0], q[1], q[2]));
         for (int tensor = 0; tensor < 2; ++tensor) {
            amrex::Real source[6]{};
            for (int c = 0; c < 6; ++c) {
               const int component = tensor == 0
                  ? z4c_gamma_tilde_components()[c] : z4c_a_tilde_components()[c];
               source[c] = values(i, j, k, component);
            }
            for (int a = 0; a < 3; ++a) {
               for (int b = a; b < 3; ++b) {
                  amrex::Real rotated = 0.0_rt;
                  for (int d = 0; d < 3; ++d) {
                     for (int e = 0; e < 3; ++e) {
                        rotated += basis.to_local[chart.canonical[a]][d]
                           * basis.to_local[chart.canonical[b]][e]
                           * source[symmetric_tensor_component(d, e)];
                     }
                  }
                  const int c = symmetric_tensor_component(a, b);
                  const int component = tensor == 0
                     ? z4c_gamma_tilde_components()[c] : z4c_a_tilde_components()[c];
                  values(i, j, k, component) = rotated;
               }
            }
         }
         amrex::Real shift[3]{};
         for (int a = 0; a < 3; ++a) {
            shift[a] = values(i, j, k, z4c_shift_components()[a]);
         }
         for (int a = 0; a < 3; ++a) {
            amrex::Real rotated = 0.0_rt;
            for (int d = 0; d < 3; ++d) {
               rotated += basis.to_local[chart.canonical[a]][d] * shift[d];
            }
            values(i, j, k, z4c_shift_components()[a]) = rotated;
         }
      });
   }
}

} // namespace pyfluid_amrex
