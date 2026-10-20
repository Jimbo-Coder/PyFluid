#pragma once

#include "pyfluid_amrex/flux_factory.hpp"
#include "pyfluid_amrex/fixed_spacetime_types.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/mesh_geometry.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/relativistic_diode.hpp"
#include "pyfluid_amrex/solid_geometry_point.hpp"

namespace pyfluid_amrex {

// Resolve user configuration once in the compiled host factory. Numerical
// owners consume only the values their launches need, never RunConfig.
struct FluxKernelParameters {
   PhysicalSystem system;
   EOSParams eos;
   FixedSpacetimeParameters spacetime;
   int mesh_coordinate;
   MeshAxes mesh_axes;
   SolidGeometryEvaluator solid;
   ExcisionGridView excision;
   RelativisticDiodeBoundary diode;
};

struct FluxKernelLaunch {
   FluxLaunch const& launch;
   FluxKernelParameters parameters;
};

} // namespace pyfluid_amrex
