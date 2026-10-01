#pragma once

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include "pyfluid_amrex/generated/field_ids.hpp"
#include "pyfluid_amrex/physical_system.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

static constexpr int MaxPassiveScalars = 1;
static constexpr int MaxStateComponents = NVAR_MHD + MaxPassiveScalars;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDConserved load_rhd_conserved(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {
      state(i, j, k, RHO),
      state(i, j, k, MOM1),
      state(i, j, k, MOM2),
      state(i, j, k, MOM3),
      state(i, j, k, ENERGY),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rhd_conserved(
   amrex::Array4<amrex::Real> const& state,
   int i,
   int j,
   int k,
   const RHDConserved& value
) noexcept
{
   state(i, j, k, RHO) = value.dens;
   state(i, j, k, MOM1) = value.mom1;
   state(i, j, k, MOM2) = value.mom2;
   state(i, j, k, MOM3) = value.mom3;
   state(i, j, k, ENERGY) = value.tau;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RMHDConserved load_rmhd_conserved(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {
      state(i, j, k, RHO),
      state(i, j, k, MOM1),
      state(i, j, k, MOM2),
      state(i, j, k, MOM3),
      state(i, j, k, ENERGY),
      state(i, j, k, B1),
      state(i, j, k, B2),
      state(i, j, k, B3),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void store_rmhd_conserved(
   amrex::Array4<amrex::Real> const& state,
   int i,
   int j,
   int k,
   const RMHDConserved& value
) noexcept
{
   state(i, j, k, RHO) = value.dens;
   state(i, j, k, MOM1) = value.mom1;
   state(i, j, k, MOM2) = value.mom2;
   state(i, j, k, MOM3) = value.mom3;
   state(i, j, k, ENERGY) = value.tau;
   state(i, j, k, B1) = value.b1;
   state(i, j, k, B2) = value.b2;
   state(i, j, k, B3) = value.b3;
}

struct Vector3 {
   amrex::Real x = 0.0_rt;
   amrex::Real y = 0.0_rt;
   amrex::Real z = 0.0_rt;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3() = default;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3(amrex::Real x_, amrex::Real y_, amrex::Real z_) noexcept : x(x_), y(y_), z(z_) {}

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 operator+(Vector3 const& other) const noexcept
   {
      return {x + other.x, y + other.y, z + other.z};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 operator-(Vector3 const& other) const noexcept
   {
      return {x - other.x, y - other.y, z - other.z};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 operator-() const noexcept
   {
      return {-x, -y, -z};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 operator*(amrex::Real scalar) const noexcept
   {
      return {x * scalar, y * scalar, z * scalar};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 operator/(amrex::Real divisor) const noexcept
   {
      return {x / divisor, y / divisor, z / divisor};
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3& operator+=(Vector3 const& other) noexcept
   {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3& operator-=(Vector3 const& other) noexcept
   {
      x -= other.x;
      y -= other.y;
      z -= other.z;
      return *this;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3& operator*=(amrex::Real scalar) noexcept
   {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3& operator/=(amrex::Real divisor) noexcept
   {
      x /= divisor;
      y /= divisor;
      z /= divisor;
      return *this;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr amrex::Real dot(Vector3 const& other) const noexcept
   {
      return x * other.x + y * other.y + z * other.z;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 cross(Vector3 const& other) const noexcept
   {
      return {
         y * other.z - z * other.y,
         z * other.x - x * other.z,
         x * other.y - y * other.x,
      };
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr amrex::Real sum() const noexcept
   {
      return x + y + z;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr amrex::Real squared_norm() const noexcept
   {
      return dot(*this);
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 scaled(amrex::Real scalar) const noexcept
   {
      return (*this) * scalar;
   }

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr Vector3 divided_by(amrex::Real divisor) const noexcept
   {
      return (*this) / divisor;
   }
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr Vector3 operator*(amrex::Real scalar, Vector3 const& vector) noexcept
{
   return vector * scalar;
}

struct StateVector3 : Vector3 {
   using Vector3::Vector3;

   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
   constexpr StateVector3(Vector3 const& value) noexcept : Vector3(value) {}
};

struct HydroState;
struct HydroPrimitive;
struct MHDState;
struct MHDPrimitive;

struct PhysicalStateLayout {
   PhysicalSystem system = PhysicalSystem::HD;
   int base_components = NVAR_HYDRO;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr PhysicalStateLayout physical_state_layout(PhysicalSystem system) noexcept
{
   if (system == PhysicalSystem::MHD) {
      return {PhysicalSystem::MHD, NVAR_MHD};
   }
   if (system == PhysicalSystem::HD) {
      return {PhysicalSystem::HD, NVAR_HYDRO};
   }
   if (system == PhysicalSystem::RHD) {
      return {PhysicalSystem::RHD, NVAR_HYDRO};
   }
   if (system == PhysicalSystem::RMHD) {
      return {PhysicalSystem::RMHD, NVAR_MHD};
   }
   return {system, 0};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int base_component_count(PhysicalSystem system) noexcept
{
   return physical_state_layout(system).base_components;
}

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr bool has_mhd_components() noexcept
{
   return NComp >= NVAR_MHD;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_momentum_vector3(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {state(i, j, k, MOM1), state(i, j, k, MOM2), state(i, j, k, MOM3)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_magnetic_vector3(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {state(i, j, k, B1), state(i, j, k, B2), state(i, j, k, B3)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real dot(StateVector3 const& lhs, StateVector3 const& rhs) noexcept
{
   return lhs.dot(rhs);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 cross_product(StateVector3 const& lhs, StateVector3 const& rhs) noexcept
{
   return lhs.cross(rhs);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 scale_state_vector3(StateVector3 const& vector, amrex::Real scalar) noexcept
{
   return vector * scalar;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 divide_state_vector3(StateVector3 const& vector, amrex::Real divisor) noexcept
{
   return scale_state_vector3(vector, 1.0_rt / divisor);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real squared_norm(StateVector3 const& vector) noexcept
{
   return vector.squared_norm();
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 velocity_from_momentum(StateVector3 const& momentum, amrex::Real rho) noexcept
{
   return divide_state_vector3(momentum, rho);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_velocity_vector3(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return velocity_from_momentum(load_momentum_vector3(state, i, j, k), state(i, j, k, RHO));
}

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int base_component_count() noexcept
{
   return has_mhd_components<NComp>() ? NVAR_MHD : NVAR_HYDRO;
}

inline int base_component_count(bool mhd)
{
   return mhd ? NVAR_MHD : NVAR_HYDRO;
}

template <int NComp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int passive_component_count() noexcept
{
   return NComp - base_component_count<NComp>();
}

inline int passive_component_begin(bool mhd)
{
   return base_component_count(mhd);
}

struct HydroState {
   amrex::Real rho = 0.0_rt;
   amrex::Real mom1 = 0.0_rt;
   amrex::Real mom2 = 0.0_rt;
   amrex::Real mom3 = 0.0_rt;
   amrex::Real energy = 0.0_rt;
};

struct HydroPrimitive {
   amrex::Real rho = 0.0_rt;
   amrex::Real v1 = 0.0_rt;
   amrex::Real v2 = 0.0_rt;
   amrex::Real v3 = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
};

struct MHDState {
   amrex::Real rho = 0.0_rt;
   amrex::Real mom1 = 0.0_rt;
   amrex::Real mom2 = 0.0_rt;
   amrex::Real mom3 = 0.0_rt;
   amrex::Real energy = 0.0_rt;
   amrex::Real b1 = 0.0_rt;
   amrex::Real b2 = 0.0_rt;
   amrex::Real b3 = 0.0_rt;
};

struct MHDPrimitive {
   amrex::Real rho = 0.0_rt;
   amrex::Real v1 = 0.0_rt;
   amrex::Real v2 = 0.0_rt;
   amrex::Real v3 = 0.0_rt;
   amrex::Real pressure = 0.0_rt;
   amrex::Real b1 = 0.0_rt;
   amrex::Real b2 = 0.0_rt;
   amrex::Real b3 = 0.0_rt;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
HydroState load_hydro_state(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {
      state(i, j, k, RHO),
      state(i, j, k, MOM1),
      state(i, j, k, MOM2),
      state(i, j, k, MOM3),
      state(i, j, k, ENERGY),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
MHDState load_mhd_state(
   amrex::Array4<amrex::Real const> const& state,
   int i,
   int j,
   int k
) noexcept
{
   return {
      state(i, j, k, RHO),
      state(i, j, k, MOM1),
      state(i, j, k, MOM2),
      state(i, j, k, MOM3),
      state(i, j, k, ENERGY),
      state(i, j, k, B1),
      state(i, j, k, B2),
      state(i, j, k, B3),
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_momentum_vector3(HydroState const& state) noexcept
{
   return {state.mom1, state.mom2, state.mom3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_momentum_vector3(MHDState const& state) noexcept
{
   return {state.mom1, state.mom2, state.mom3};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_velocity_vector3(HydroState const& state) noexcept
{
   return velocity_from_momentum(load_momentum_vector3(state), state.rho);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_velocity_vector3(MHDState const& state) noexcept
{
   return velocity_from_momentum(load_momentum_vector3(state), state.rho);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
StateVector3 load_magnetic_vector3(MHDState const& state) noexcept
{
   return {state.b1, state.b2, state.b3};
}

} // namespace pyfluid_amrex
