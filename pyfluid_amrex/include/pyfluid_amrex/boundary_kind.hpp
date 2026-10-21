#pragma once

namespace pyfluid_amrex {

enum class BoundaryKind {
   Periodic,
   Outflow,
   Reflecting,
   Symmetry,
   Inflow,
   Diode,
   Origin,
   Analytic
};

} // namespace pyfluid_amrex
