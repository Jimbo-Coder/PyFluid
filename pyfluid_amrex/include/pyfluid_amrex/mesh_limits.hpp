#pragma once

namespace pyfluid_amrex {

static constexpr int MaxInitialRefineBoxes = 8;
static constexpr int MaxInitialRefineBoxValues = 3 * MaxInitialRefineBoxes;
static constexpr int MaxMeshPatchBoxes = 16;
static constexpr int MaxMeshPatchBoxValues = 3 * MaxMeshPatchBoxes;
static constexpr int MaxSolidBoxes = 32;
static constexpr int MaxSolidBoxValues = 3 * MaxSolidBoxes;

} // namespace pyfluid_amrex
