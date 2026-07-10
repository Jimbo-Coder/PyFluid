#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

namespace pyfluid_amrex {

struct FaceMagneticField {
   std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> b;
};

struct EdgeElectricField {
   std::array<std::unique_ptr<amrex::MultiFab>, 3> e;
};

inline void allocate_face_magnetic_field(
   FaceMagneticField& face_b,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm,
   int ng
)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      const amrex::BoxArray face_ba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(dir));
      face_b.b[dir] = std::make_unique<amrex::MultiFab>(face_ba, dm, 1, ng);
   }
}

inline void allocate_face_magnetic_field_like(FaceMagneticField& dst, FaceMagneticField const& src)
{
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      AMREX_ALWAYS_ASSERT(src.b[dir] != nullptr);
      dst.b[dir] = std::make_unique<amrex::MultiFab>(
         src.b[dir]->boxArray(),
         src.b[dir]->DistributionMap(),
         1,
         src.b[dir]->nGrowVect()
      );
   }
}

inline amrex::IndexType edge_index_type(int component)
{
   amrex::IntVect type(AMREX_D_DECL(0, 0, 0));
#if (AMREX_SPACEDIM == 2)
   if (component == 0) {
      type = amrex::IntVect(AMREX_D_DECL(0, 1, 0));
   } else if (component == 1) {
      type = amrex::IntVect(AMREX_D_DECL(1, 0, 0));
   } else {
      type = amrex::IntVect(AMREX_D_DECL(1, 1, 0));
   }
#elif (AMREX_SPACEDIM == 3)
   if (component == 0) {
      type = amrex::IntVect(AMREX_D_DECL(0, 1, 1));
   } else if (component == 1) {
      type = amrex::IntVect(AMREX_D_DECL(1, 0, 1));
   } else {
      type = amrex::IntVect(AMREX_D_DECL(1, 1, 0));
   }
#endif
   return amrex::IndexType(type);
}

inline void allocate_edge_electric_field(
   EdgeElectricField& edge_e,
   amrex::BoxArray const& ba,
   amrex::DistributionMapping const& dm
)
{
#if (AMREX_SPACEDIM == 2 || AMREX_SPACEDIM == 3)
   for (int component = 0; component < 3; ++component) {
      edge_e.e[component] = std::make_unique<amrex::MultiFab>(amrex::convert(ba, edge_index_type(component)), dm, 1, 0);
   }
#else
   (void)edge_e;
   (void)ba;
   (void)dm;
#endif
}

inline amrex::MultiFab* mutable_edge_component(EdgeElectricField& edge_e, int component)
{
   return edge_e.e[component].get();
}

template <typename EdgeField>
inline void set_edge_field_value(EdgeField& edge_e, amrex::Real value)
{
   for (int component = 0; component < 3; ++component) {
      if (amrex::MultiFab* mf = mutable_edge_component(edge_e, component)) {
         mf->setVal(value);
      }
   }
}

} // namespace pyfluid_amrex
