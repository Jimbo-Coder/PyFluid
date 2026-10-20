#pragma once

#include <AMReX_Array4.H>
#include <AMReX_Box.H>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex {

struct StoredLimiterMetricAccessor;
struct FixedLimiterMetricAccessor;

// Host-only launch contract. Device equations stay with each compiled owner.
struct RmhdCorrectionFluxContext {
   amrex::Box box;
   amrex::Box face_box;
   amrex::Array4<amrex::Real> output;
   amrex::Array4<amrex::Real const> high_flux;
   amrex::Array4<amrex::Real const> primitive;
   amrex::Array4<int const> correction;
   EOSParams eos;
   EOSThermodynamicState atmosphere;
   amrex::Array4<int const> excision_mask;
   bool use_excision_mask;
   amrex::Long* hll_fallback_count = nullptr;
};

void fill_rmhd_correction_fluxes(
   int riemann_method, int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
);

void fill_rmhd_correction_fluxes(
   int riemann_method, int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_llf_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_llf_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hll_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hll_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hllc_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hllc_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hlld_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   StoredLimiterMetricAccessor const& metric,
   bool pwl_only
);

void launch_rmhd_hlld_correction_fluxes(
   int direction,
   RmhdCorrectionFluxContext const& context,
   FixedLimiterMetricAccessor const& metric,
   bool pwl_only
);

} // namespace pyfluid_amrex
