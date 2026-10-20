#include "pyfluid_amrex/rmhd_advanced_flux.hpp"

#include "pyfluid_amrex/build_options.hpp"
#include "numerics/flux_kernel_parameters.hpp"
#include "pyfluid_amrex/flux_factory.hpp"
#include "pyfluid_amrex/spacetime_excision.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/reconstruction/mp5.hpp"
#include "pyfluid_amrex/reconstruction/parabolic.hpp"
#include "pyfluid_amrex/reconstruction/ppm3.hpp"
#include "pyfluid_amrex/reconstruction/ppm4.hpp"
#include "pyfluid_amrex/reconstruction/ppmx.hpp"
#include "pyfluid_amrex/reconstruction/teno5.hpp"
#include "pyfluid_amrex/reconstruction/wenoz.hpp"
#include "pyfluid_amrex/rmhd_riemann.hpp"
#include "relativity/rmhd_system.hpp"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <limits>

namespace pyfluid_amrex {

using namespace amrex::literals;
namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive cached_rmhd_primitive(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k
) noexcept
{
   return {
      primitive(i, j, k, RHO),
      0.0_rt,
      primitive(i, j, k, ENERGY),
      primitive(i, j, k, MOM1),
      primitive(i, j, k, MOM2),
      primitive(i, j, k, MOM3),
      primitive(i, j, k, B1),
      primitive(i, j, k, B2),
      primitive(i, j, k, B3),
   };
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDPrimitive cached_rmhd_primitive_at_offset(
   amrex::Array4<amrex::Real const> const& primitive,
   int i,
   int j,
   int k,
   int offset
) noexcept
{
   if constexpr (Direction == 0) {
      i += offset;
   } else if constexpr (Direction == 1) {
      j += offset;
   } else {
      k += offset;
   }
   return cached_rmhd_primitive(primitive, i, j, k);
}

template <int ReconMethod, int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool rmhd_hllc_shock_stabilization_needed(
   amrex::Array4<amrex::Real const> const& primitive,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   SpacetimeMetric const& face_metric
) noexcept
{
#if (AMREX_SPACEDIM == 1)
   (void)primitive;
   (void)il;
   (void)jl;
   (void)kl;
   (void)ir;
   (void)jr;
   (void)kr;
   (void)face_metric;
   return false;
#else
   constexpr int sensor_radius = reconstruction_stencil_radius(ReconMethod);
   static_assert(sensor_radius >= 0 && sensor_radius <= 2);
   const RMHDFaceFrame frame = rmhd_face_frame<Direction>(face_metric);
   amrex::Real pressure_min = std::numeric_limits<amrex::Real>::infinity();
   amrex::Real pressure_max = 0.0_rt;
   for (int offset = 0; offset <= sensor_radius; ++offset) {
      const RMHDPrimitive local_l = rmhd_primitive_to_face_frame(
         cached_rmhd_primitive_at_offset<Direction>(
            primitive, il, jl, kl, -offset
         ),
         frame
      );
      const RMHDPrimitive local_r = rmhd_primitive_to_face_frame(
         cached_rmhd_primitive_at_offset<Direction>(
            primitive, ir, jr, kr, offset
         ),
         frame
      );
      const amrex::Real total_pressure_l = srmhd_total_pressure(local_l);
      const amrex::Real total_pressure_r = srmhd_total_pressure(local_r);
      pressure_min = amrex::min(
         pressure_min, amrex::min(total_pressure_l, total_pressure_r)
      );
      pressure_max = amrex::max(
         pressure_max, amrex::max(total_pressure_l, total_pressure_r)
      );
   }
   return pressure_max > 0.0_rt && pressure_min < 0.5_rt * pressure_max;
#endif
}

template <int ReconMethod, int RiemannMethod, int Direction, typename MetricAccessor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDRiemannFluxResult rmhd_advanced_riemann_flux(
   amrex::Array4<amrex::Real const> const& primitive,
   int il,
   int jl,
   int kl,
   int ir,
   int jr,
   int kr,
   EOSParams const& eos,
   MetricAccessor const& metric,
   EOSThermodynamicState const& atmosphere
) noexcept
{
   const RMHDPrimitiveFaceStates face =
      rmhd_primitive_face_states<ReconMethod, Direction>(
         primitive,
         il,
         jl,
         kl,
         ir,
         jr,
         kr,
         eos,
         metric,
         atmosphere,
         atmosphere
      );
   const SpacetimeMetric face_metric = metric.template face<Direction>(
      ir, jr, kr
   );
   if constexpr (RiemannMethod == RiemannHLLC) {
      const bool shock_stabilize =
         !relativistic_reconstruction_stencil_is_active<ReconMethod, Direction>(
            metric, il, jl, kl, ir, jr, kr)
         || rmhd_hllc_shock_stabilization_needed<ReconMethod, Direction>(
            primitive, il, jl, kl, ir, jr, kr, face_metric
         );
      return rmhd_hllc_flux<Direction>(
         face.left, face.right, eos, face_metric, shock_stabilize
      );
   } else {
      static_assert(RiemannMethod == RiemannHLLD);
      return rmhd_hlld_flux<Direction>(
         face.left, face.right, eos, face_metric
      );
   }
}

template <int Direction>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rmhd_left_cell(
   int i,
   int j,
   int k,
   int& il,
   int& jl,
   int& kl
) noexcept
{
   il = i;
   jl = j;
   kl = k;
   if constexpr (Direction == 0) {
      --il;
   } else if constexpr (Direction == 1) {
      --jl;
   } else {
      --kl;
   }
}

template <int ReconMethod, int RiemannMethod, int Direction>
void fill_rmhd_advanced_direction(FluxKernelLaunch const& kernel_launch)
{
   const auto& launch = kernel_launch.launch;
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      launch.geometry != nullptr,
      "RMHD advanced fluxes require level geometry"
   );
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      launch.atmosphere != nullptr,
      "RMHD advanced fluxes require the recovery atmosphere state"
   );
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      launch.topology == FluxTopology::Standard,
      "RMHD advanced fluxes support the standard topology"
   );
   const auto diode = kernel_launch.parameters.diode;
   const EOSParams eos = kernel_launch.parameters.eos;
   const EOSThermodynamicState atmosphere = *launch.atmosphere;
   amrex::Long* const hll_fallback_count =
      launch.rmhd_riemann_hll_fallback_count;
   amrex::MultiFab& flux = *launch.fluxes[Direction];
   const auto excision = kernel_launch.parameters.excision;

   if (launch.cell_spacetime != nullptr) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const amrex::Box& box = mfi.tilebox();
         const auto cached = launch.primitive.const_array(mfi);
         const auto spacetime = launch.cell_spacetime->const_array(mfi);
         const auto face_flux = flux.array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int il;
               int jl;
               int kl;
               rmhd_left_cell<Direction>(i, j, k, il, jl, kl);
               const RHDStoredMetricAccessor metric{spacetime, excision, diode};
               const RMHDRiemannFluxResult result = rmhd_advanced_riemann_flux<
                  ReconMethod,
                  RiemannMethod,
                  Direction
               >(
                  cached,
                  il,
                  jl,
                  kl,
                  i,
                  j,
                  k,
                  eos,
                  metric,
                  atmosphere
               );
               if (hll_fallback_count != nullptr && result.used_hll) {
                  amrex::HostDevice::Atomic::Add(
                     hll_fallback_count, static_cast<amrex::Long>(1)
                  );
               }
               store_rmhd_conserved(face_flux, i, j, k, result.flux);
            }
         );
      }
      return;
   }

   const auto lower = launch.geometry->ProbLoArray();
   const auto spacing = launch.geometry->CellSizeArray();
   const FixedSpacetimeParameters spacetime = kernel_launch.parameters.spacetime;
   const amrex::Real time = launch.time;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto cached = launch.primitive.const_array(mfi);
      const auto face_flux = flux.array(mfi);
      amrex::ParallelFor(
         box,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int il;
            int jl;
            int kl;
            rmhd_left_cell<Direction>(i, j, k, il, jl, kl);
            const RHDFixedMetricAccessor metric{
               spacetime,
               lower,
               spacing,
               MeshCartesian,
               mesh_default_axes_layout(MeshCartesian),
               time,
               {},
               diode
            };
            const RMHDRiemannFluxResult result = rmhd_advanced_riemann_flux<
               ReconMethod,
               RiemannMethod,
               Direction
            >(
               cached,
               il,
               jl,
               kl,
               i,
               j,
               k,
               eos,
               metric,
               atmosphere
            );
            if (hll_fallback_count != nullptr && result.used_hll) {
               amrex::HostDevice::Atomic::Add(
                  hll_fallback_count, static_cast<amrex::Long>(1)
               );
            }
            store_rmhd_conserved(face_flux, i, j, k, result.flux);
         }
      );
   }
}

template <int ReconMethod, int RiemannMethod>
void fill_rmhd_advanced_layout(FluxKernelLaunch const& kernel_launch)
{
   const auto& launch = kernel_launch.launch;
   if (launch.direction < 0) {
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 0>(kernel_launch);
#if (AMREX_SPACEDIM >= 2)
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 1>(kernel_launch);
#endif
#if (AMREX_SPACEDIM == 3)
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 2>(kernel_launch);
#endif
      return;
   }
   switch (launch.direction) {
   case 0:
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 0>(kernel_launch);
      return;
#if (AMREX_SPACEDIM >= 2)
   case 1:
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 1>(kernel_launch);
      return;
#endif
#if (AMREX_SPACEDIM == 3)
   case 2:
      fill_rmhd_advanced_direction<ReconMethod, RiemannMethod, 2>(kernel_launch);
      return;
#endif
   default:
      amrex::Abort("invalid RMHD advanced flux direction");
   }
}

template <int RiemannMethod>
void dispatch_rmhd_advanced_reconstruction(FluxKernelLaunch const& kernel_launch)
{
   const auto& launch = kernel_launch.launch;
   switch (launch.reconstruction_method) {
   case ReconPWC:
#if PYFLUID_AMREX_ENABLE_RECON_PWC
      fill_rmhd_advanced_layout<ReconPWC, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconPWL:
#if PYFLUID_AMREX_ENABLE_RECON_PWL
      fill_rmhd_advanced_layout<ReconPWL, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconWENOZ:
#if PYFLUID_AMREX_ENABLE_RECON_WENOZ
      fill_rmhd_advanced_layout<ReconWENOZ, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconMP5:
#if PYFLUID_AMREX_ENABLE_RECON_MP5
      fill_rmhd_advanced_layout<ReconMP5, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconTENO5:
#if PYFLUID_AMREX_ENABLE_RECON_TENO5
      fill_rmhd_advanced_layout<ReconTENO5, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconParabolic:
#if PYFLUID_AMREX_ENABLE_RECON_PARABOLIC
      fill_rmhd_advanced_layout<ReconParabolic, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconPPM4:
#if PYFLUID_AMREX_ENABLE_RECON_PPM4
      fill_rmhd_advanced_layout<ReconPPM4, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconPPMX:
#if PYFLUID_AMREX_ENABLE_RECON_PPMX
      fill_rmhd_advanced_layout<ReconPPMX, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   case ReconPPM3:
#if PYFLUID_AMREX_ENABLE_RECON_PPM3
      fill_rmhd_advanced_layout<ReconPPM3, RiemannMethod>(kernel_launch);
      return;
#endif
      break;
   default:
      amrex::Abort("invalid RMHD reconstruction method");
      return;
   }
   amrex::Abort("selected RMHD reconstruction method is disabled in this build");
}

} // namespace

void fill_rmhd_advanced_fluxes(FluxKernelLaunch const& kernel_launch)
{
   const auto& launch = kernel_launch.launch;
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      kernel_launch.parameters.system == PhysicalSystem::RMHD,
      "advanced RMHD flux entry requires the RMHD physical system"
   );
   switch (launch.riemann_method) {
   case RiemannHLLC:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLC
      dispatch_rmhd_advanced_reconstruction<RiemannHLLC>(kernel_launch);
      return;
#else
      amrex::Abort("RMHD HLLC was disabled at compile time");
#endif
   case RiemannHLLD:
#if PYFLUID_AMREX_ENABLE_RIEMANN_HLLD
      dispatch_rmhd_advanced_reconstruction<RiemannHLLD>(kernel_launch);
      return;
#else
      amrex::Abort("RMHD HLLD was disabled at compile time");
#endif
   default:
      amrex::Abort("invalid advanced RMHD Riemann solver");
   }
}

} // namespace pyfluid_amrex
