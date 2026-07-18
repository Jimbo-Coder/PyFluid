#include "pyfluid_amrex/signal_speed_factory.hpp"

#include "pyfluid_amrex/config.hpp"
#include "signal_speed_impl.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {
namespace {

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_standard(
   SignalSpeedLaunch const& launch,
   amrex::Geometry const& geometry,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
)
{
   return adaptive_dt<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      geometry,
      launch.config,
      time,
      cell_spacetime
   );
}

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_mapped(
   SignalSpeedLaunch const& launch,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   return adaptive_dt_mapped_hydro<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      launch.config,
      grid_system,
      layout
   );
}

template <int NumComponents, int EosMethod, int DiffusionMethod>
amrex::Real launch_cutcell(
   SignalSpeedLaunch const& launch,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geometry,
   amrex::Vector<int> const* solid_valid_kind
)
{
   return adaptive_dt_cutcell_masked<NumComponents, EosMethod, DiffusionMethod>(
      launch.frequency,
      launch.state,
      solid_mask,
      volume_fraction,
      apertures,
      geometry,
      launch.config,
      solid_valid_kind
   );
}

template <int NumComponents, int EosMethod, typename Operation>
amrex::Real dispatch_diffusion(int diffusion_method, Operation&& operation)
{
   switch (diffusion_method) {
   case DiffusionNone:
      return operation.template operator()<NumComponents, EosMethod, DiffusionNone>();
   case DiffusionExplicit:
      return operation.template operator()<NumComponents, EosMethod, DiffusionExplicit>();
   case DiffusionImplicit:
      return operation.template operator()<NumComponents, EosMethod, DiffusionImplicit>();
   default:
      amrex::Abort("invalid diffusion method in signal-speed factory");
      return 0.0;
   }
}

template <int NumComponents, typename Operation>
amrex::Real dispatch_eos(SignalSpeedLaunch const& launch, Operation&& operation)
{
   switch (launch.eos_method) {
   case EOSIdealGas:
      return dispatch_diffusion<NumComponents, EOSIdealGas>(
         launch.diffusion_method, operation
      );
   case EOSPolytrope:
      return dispatch_diffusion<NumComponents, EOSPolytrope>(
         launch.diffusion_method, operation
      );
   case EOSHybridPolytrope:
      return dispatch_diffusion<NumComponents, EOSHybridPolytrope>(
         launch.diffusion_method, operation
      );
   default:
      amrex::Abort("invalid EOS method in signal-speed factory");
      return 0.0;
   }
}

template <typename Operation>
amrex::Real dispatch_layout(SignalSpeedLaunch const& launch, Operation&& operation)
{
   switch (launch.num_components) {
   case NVAR_HYDRO:
      return dispatch_eos<NVAR_HYDRO>(launch, operation);
   case NVAR_HYDRO + 1:
      return dispatch_eos<NVAR_HYDRO + 1>(launch, operation);
   case NVAR_MHD:
      return dispatch_eos<NVAR_MHD>(launch, operation);
   case NVAR_MHD + 1:
      return dispatch_eos<NVAR_MHD + 1>(launch, operation);
   default:
      amrex::Abort("unsupported state layout in signal-speed factory");
      return 0.0;
   }
}

} // namespace

amrex::Real compute_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::Geometry const& geometry,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_standard<NumComponents, EosMethod, DiffusionMethod>(
         launch, geometry, time, cell_spacetime
      );
   };
   return dispatch_layout(launch, operation);
}

amrex::Real compute_spacetime_dt(
   amrex::MultiFab& frequency,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   amrex::Real cfl,
   amrex::Real gauge_speed
)
{
   const auto spacing = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(frequency, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto output = frequency.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         amrex::Real value = (
            amrex::Math::abs(metric.shift1)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma11)
         ) / spacing[0];
#if (AMREX_SPACEDIM >= 2)
         value += (
            amrex::Math::abs(metric.shift2)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma22)
         ) / spacing[1];
#endif
#if (AMREX_SPACEDIM == 3)
         value += (
            amrex::Math::abs(metric.shift3)
            + gauge_speed * metric.lapse * std::sqrt(metric.inv_gamma33)
         ) / spacing[2];
#endif
         output(i, j, k, 0) = value;
      });
   }
   return cfl / frequency.max(0);
}

amrex::Real compute_mapped_hydro_adaptive_dt(
   SignalSpeedLaunch const& launch,
   PatchGridSystem const& grid_system,
   PatchLayout const& layout
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_mapped<NumComponents, EosMethod, DiffusionMethod>(
         launch, grid_system, layout
      );
   };
   switch (launch.num_components) {
   case NVAR_HYDRO:
      return dispatch_eos<NVAR_HYDRO>(launch, operation);
   case NVAR_HYDRO + 1:
      return dispatch_eos<NVAR_HYDRO + 1>(launch, operation);
   default:
      amrex::Abort("mapped-patch signal-speed launch requires a hydro state layout");
      return 0.0;
   }
}

amrex::Real compute_cutcell_adaptive_dt(
   SignalSpeedLaunch const& launch,
   amrex::iMultiFab const& solid_mask,
   amrex::MultiFab const& volume_fraction,
   std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& apertures,
   amrex::Geometry const& geometry,
   amrex::Vector<int> const* solid_valid_kind
)
{
   const auto operation = [&]<int NumComponents, int EosMethod, int DiffusionMethod>() {
      return launch_cutcell<NumComponents, EosMethod, DiffusionMethod>(
         launch,
         solid_mask,
         volume_fraction,
         apertures,
         geometry,
         solid_valid_kind
      );
   };
   return dispatch_layout(launch, operation);
}

} // namespace pyfluid_amrex
