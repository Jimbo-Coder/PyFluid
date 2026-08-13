#pragma once

namespace pyfluid_amrex {

struct FluxLaunch;

// HLLC/HLLD are isolated from the ordinary reconstruction translation units
// because their RMHD fan algebra is substantially larger than LLF/HLL.  This
// host-level entry retains compile-time reconstruction and direction inside
// its owning translation unit without multiplying that implementation across
// every generic flux-kernel object.
void fill_rmhd_advanced_fluxes(FluxLaunch const& launch);

} // namespace pyfluid_amrex
