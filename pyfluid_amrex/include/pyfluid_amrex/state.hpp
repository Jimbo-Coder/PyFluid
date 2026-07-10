#pragma once

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>
namespace pyfluid_amrex {

enum Conserved {
   RHO = 0,
   MOM1 = 1,
   MOM2 = 2,
   MOM3 = 3,
   ENERGY = 4,
   NVAR_HYDRO = 5,
   B1 = 5,
   B2 = 6,
   B3 = 7,
   NVAR_MHD = 8,
};

static constexpr int MaxPassiveScalars = 1;
static constexpr int MaxStateComponents = NVAR_MHD + MaxPassiveScalars;

enum class PhysicsSystemKind : int {
   Hydro = 0,
   MHD = 1,
};

struct Vector3 {
   amrex::Real x = 0.0;
   amrex::Real y = 0.0;
   amrex::Real z = 0.0;

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

struct PhysicsSystemTraits {
   PhysicsSystemKind kind = PhysicsSystemKind::Hydro;
   int base_components = NVAR_HYDRO;
   bool has_magnetic_field = false;
   bool supports_ct = false;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr PhysicsSystemTraits physics_system_traits(PhysicsSystemKind kind) noexcept
{
   return kind == PhysicsSystemKind::MHD
      ? PhysicsSystemTraits{PhysicsSystemKind::MHD, NVAR_MHD, true, true}
      : PhysicsSystemTraits{PhysicsSystemKind::Hydro, NVAR_HYDRO, false, false};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr bool physics_system_has_magnetic_field(PhysicsSystemKind kind) noexcept
{
   return physics_system_traits(kind).has_magnetic_field;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int base_component_count(PhysicsSystemKind kind) noexcept
{
   return physics_system_traits(kind).base_components;
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
   return scale_state_vector3(vector, 1.0 / divisor);
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
   amrex::Real rho = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real energy = 0.0;
};

struct HydroPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real pressure = 0.0;
};

struct MHDState {
   amrex::Real rho = 0.0;
   amrex::Real mom1 = 0.0;
   amrex::Real mom2 = 0.0;
   amrex::Real mom3 = 0.0;
   amrex::Real energy = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
};

struct MHDPrimitive {
   amrex::Real rho = 0.0;
   amrex::Real v1 = 0.0;
   amrex::Real v2 = 0.0;
   amrex::Real v3 = 0.0;
   amrex::Real pressure = 0.0;
   amrex::Real b1 = 0.0;
   amrex::Real b2 = 0.0;
   amrex::Real b3 = 0.0;
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
