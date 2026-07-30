"""Exact and analytic solutions used by validation and postprocessing."""

import numpy as np


def sod_wave_function(pressure, rho_k, pressure_k, velocity_k, gamma):
   if pressure > pressure_k:
      a = 2.0 / ((gamma + 1.0) * rho_k)
      b = (gamma - 1.0) * pressure_k / (gamma + 1.0)
      value = (pressure - pressure_k) * np.sqrt(a / (pressure + b))
      deriv = np.sqrt(a / (pressure + b)) * (1.0 - 0.5 * (pressure - pressure_k) / (pressure + b))
      return value, deriv
   sound_k = np.sqrt(gamma * pressure_k / rho_k)
   exponent = (gamma - 1.0) / (2.0 * gamma)
   ratio = pressure / pressure_k
   value = (2.0 * sound_k / (gamma - 1.0)) * (ratio ** exponent - 1.0)
   deriv = (1.0 / (rho_k * sound_k)) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
   return value, deriv


def solve_sod_star_state(left, right, gamma):
   rho_l, velocity_l, pressure_l = left
   rho_r, velocity_r, pressure_r = right

   def phi(pressure):
      f_l, _ = sod_wave_function(pressure, rho_l, pressure_l, velocity_l, gamma)
      f_r, _ = sod_wave_function(pressure, rho_r, pressure_r, velocity_r, gamma)
      return f_l + f_r + velocity_r - velocity_l

   lo = 1.0e-14
   hi = max(pressure_l, pressure_r)
   while phi(hi) < 0.0:
      hi *= 2.0
   for _ in range(120):
      mid = 0.5 * (lo + hi)
      if phi(mid) > 0.0:
         hi = mid
      else:
         lo = mid
   pressure_star = 0.5 * (lo + hi)
   f_l, _ = sod_wave_function(pressure_star, rho_l, pressure_l, velocity_l, gamma)
   f_r, _ = sod_wave_function(pressure_star, rho_r, pressure_r, velocity_r, gamma)
   velocity_star = 0.5 * (velocity_l + velocity_r + f_r - f_l)
   return pressure_star, velocity_star


def sod_exact_solution(x, time_value, gamma, left, right, x0=0.5):
   rho_l, velocity_l, pressure_l = left
   rho_r, velocity_r, pressure_r = right
   pressure_star, velocity_star = solve_sod_star_state(left, right, gamma)
   sound_l = np.sqrt(gamma * pressure_l / rho_l)
   sound_r = np.sqrt(gamma * pressure_r / rho_r)
   gamma_ratio = (gamma - 1.0) / (gamma + 1.0)
   xi = (x - x0) / max(time_value, 1.0e-300)
   rho = np.empty_like(x)
   velocity = np.empty_like(x)
   pressure = np.empty_like(x)

   rho_star_l = rho_l * (pressure_star / pressure_l) ** (1.0 / gamma)
   rho_star_r = rho_r * ((pressure_star / pressure_r + gamma_ratio) / (1.0 + gamma_ratio * pressure_star / pressure_r))

   for n, speed in enumerate(xi):
      if speed <= velocity_star:
         if pressure_star > pressure_l:
            shock_speed = velocity_l - sound_l * np.sqrt(
               (gamma + 1.0) * pressure_star / (2.0 * gamma * pressure_l)
               + (gamma - 1.0) / (2.0 * gamma)
            )
            if speed <= shock_speed:
               rho[n], velocity[n], pressure[n] = rho_l, velocity_l, pressure_l
            else:
               rho[n], velocity[n], pressure[n] = rho_star_l, velocity_star, pressure_star
         else:
            head_speed = velocity_l - sound_l
            sound_star_l = sound_l * (pressure_star / pressure_l) ** ((gamma - 1.0) / (2.0 * gamma))
            tail_speed = velocity_star - sound_star_l
            if speed <= head_speed:
               rho[n], velocity[n], pressure[n] = rho_l, velocity_l, pressure_l
            elif speed > tail_speed:
               rho[n], velocity[n], pressure[n] = rho_star_l, velocity_star, pressure_star
            else:
               u = 2.0 * (sound_l + 0.5 * (gamma - 1.0) * velocity_l + speed) / (gamma + 1.0)
               sound = 2.0 * (sound_l + 0.5 * (gamma - 1.0) * (velocity_l - speed)) / (gamma + 1.0)
               rho[n] = rho_l * (sound / sound_l) ** (2.0 / (gamma - 1.0))
               velocity[n] = u
               pressure[n] = pressure_l * (sound / sound_l) ** (2.0 * gamma / (gamma - 1.0))
      else:
         if pressure_star > pressure_r:
            shock_speed = velocity_r + sound_r * np.sqrt(
               (gamma + 1.0) * pressure_star / (2.0 * gamma * pressure_r)
               + (gamma - 1.0) / (2.0 * gamma)
            )
            if speed >= shock_speed:
               rho[n], velocity[n], pressure[n] = rho_r, velocity_r, pressure_r
            else:
               rho[n], velocity[n], pressure[n] = rho_star_r, velocity_star, pressure_star
         else:
            head_speed = velocity_r + sound_r
            sound_star_r = sound_r * (pressure_star / pressure_r) ** ((gamma - 1.0) / (2.0 * gamma))
            tail_speed = velocity_star + sound_star_r
            if speed >= head_speed:
               rho[n], velocity[n], pressure[n] = rho_r, velocity_r, pressure_r
            elif speed <= tail_speed:
               rho[n], velocity[n], pressure[n] = rho_star_r, velocity_star, pressure_star
            else:
               u = 2.0 * (-sound_r + 0.5 * (gamma - 1.0) * velocity_r + speed) / (gamma + 1.0)
               sound = 2.0 * (sound_r + 0.5 * (gamma - 1.0) * (speed - velocity_r)) / (gamma + 1.0)
               rho[n] = rho_r * (sound / sound_r) ** (2.0 / (gamma - 1.0))
               velocity[n] = u
               pressure[n] = pressure_r * (sound / sound_r) ** (2.0 * gamma / (gamma - 1.0))

   return rho, pressure, velocity


def sod_wave_breakpoints(
   time_value,
   gamma,
   left,
   right,
   x0=0.5,
   domain_lo=0.0,
   domain_hi=1.0,
):
   if domain_hi <= domain_lo:
      raise ValueError("Sod breakpoint domain must have positive extent")
   if time_value <= 0.0:
      return (domain_lo, min(max(x0, domain_lo), domain_hi), domain_hi)
   rho_l, velocity_l, pressure_l = left
   rho_r, velocity_r, pressure_r = right
   pressure_star, velocity_star = solve_sod_star_state(left, right, gamma)
   sound_l = np.sqrt(gamma * pressure_l / rho_l)
   sound_r = np.sqrt(gamma * pressure_r / rho_r)
   points = [domain_lo, domain_hi]

   if pressure_star > pressure_l:
      left_shock = velocity_l - sound_l * np.sqrt(
         (gamma + 1.0) * pressure_star / (2.0 * gamma * pressure_l)
         + (gamma - 1.0) / (2.0 * gamma)
      )
      points.append(x0 + left_shock * time_value)
   else:
      sound_star_l = sound_l * (pressure_star / pressure_l) ** ((gamma - 1.0) / (2.0 * gamma))
      points.append(x0 + (velocity_l - sound_l) * time_value)
      points.append(x0 + (velocity_star - sound_star_l) * time_value)

   points.append(x0 + velocity_star * time_value)

   if pressure_star > pressure_r:
      right_shock = velocity_r + sound_r * np.sqrt(
         (gamma + 1.0) * pressure_star / (2.0 * gamma * pressure_r)
         + (gamma - 1.0) / (2.0 * gamma)
      )
      points.append(x0 + right_shock * time_value)
   else:
      sound_star_r = sound_r * (pressure_star / pressure_r) ** ((gamma - 1.0) / (2.0 * gamma))
      points.append(x0 + (velocity_star + sound_star_r) * time_value)
      points.append(x0 + (velocity_r + sound_r) * time_value)

   clipped = [
      min(max(float(point), domain_lo), domain_hi)
      for point in points
   ]
   clipped.sort()
   unique = []
   for point in clipped:
      if not unique or abs(point - unique[-1]) > 1.0e-14:
         unique.append(point)
   if unique[0] > domain_lo:
      unique.insert(0, domain_lo)
   if unique[-1] < domain_hi:
      unique.append(domain_hi)
   return tuple(unique)


def sod_conserved_integrals(time_value, gamma, left, right, x0=0.5, quadrature_order=96):
   nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
   breakpoints = sod_wave_breakpoints(time_value, gamma, left, right, x0)
   mass = 0.0
   momentum = 0.0
   energy = 0.0
   for a, b in zip(breakpoints[:-1], breakpoints[1:]):
      if b <= a:
         continue
      center = 0.5 * (a + b)
      half_width = 0.5 * (b - a)
      x = center + half_width * nodes
      rho, pressure, velocity = sod_exact_solution(x, time_value, gamma, left, right, x0)
      mass += half_width * float(np.sum(weights * rho))
      momentum += half_width * float(np.sum(weights * rho * velocity))
      total_energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity
      energy += half_width * float(np.sum(weights * total_energy))
   return {"mass": mass, "energy": energy, "momx": momentum}


def resistive_mhd_wave_exact(x, time_value, gamma, pressure0, amplitude, resistivity, wave_number):
   k = 2.0 * np.pi * wave_number
   decay = np.exp(-resistivity * k * k * time_value)
   by = amplitude * decay * np.sin(k * x)
   bz = amplitude * decay * np.cos(k * x)
   energy = pressure0 / (gamma - 1.0) + 0.5 * amplitude * amplitude
   pressure = (gamma - 1.0) * (energy - 0.5 * (by * by + bz * bz))
   return by, bz, pressure


def diffusion_wave_amplitude_exact(amplitude, coefficient, wave_number, time_value):
   k = 2.0 * np.pi * wave_number
   return amplitude * np.exp(-coefficient * k * k * time_value)


def circular_alfven_wave_exact(x, time_value, rho0, pressure0, bx0, amplitude, wave_number, direction):
   ca = abs(bx0) / np.sqrt(rho0)
   speed = direction * ca
   velocity_factor = -bx0 / (rho0 * speed)
   phase = 2.0 * np.pi * wave_number * (x - speed * time_value)
   by = amplitude * np.cos(phase)
   bz = amplitude * np.sin(phase)
   vy = velocity_factor * by
   vz = velocity_factor * bz
   rho = np.full_like(x, rho0, dtype=float)
   pressure = np.full_like(x, pressure0, dtype=float)
   bx = np.full_like(x, bx0, dtype=float)
   vx = np.zeros_like(x, dtype=float)
   return rho, pressure, vx, vy, vz, bx, by, bz
