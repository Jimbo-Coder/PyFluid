"""Coordinate-basis metric differential geometry, independent of NR formulation."""

from __future__ import annotations

import sympy as sp

DIMENSION = 3


def metric_christoffel(
   inverse_metric: sp.Matrix,
   metric_derivatives: list[sp.Matrix],
) -> list[list[list[sp.Expr]]]:
   return [
      [
         [
            sp.Rational(1, 2)
            * sum(
               inverse_metric[upper, contracted]
               * (
                  metric_derivatives[first][contracted, second]
                  + metric_derivatives[second][contracted, first]
                  - metric_derivatives[contracted][first, second]
               )
               for contracted in range(DIMENSION)
            )
            for second in range(DIMENSION)
         ]
         for first in range(DIMENSION)
      ]
      for upper in range(DIMENSION)
   ]


def inverse_metric_derivatives(
   inverse_metric: sp.Matrix,
   metric_derivatives: list[sp.Matrix],
) -> list[sp.Matrix]:
   return [
      -inverse_metric * metric_derivatives[direction] * inverse_metric
      for direction in range(DIMENSION)
   ]


def metric_ricci_tensor(
   inverse_metric: sp.Matrix,
   metric_derivatives: list[sp.Matrix],
   metric_second_derivatives: list[list[sp.Matrix]],
   christoffel: list[list[list[sp.Expr]]],
) -> sp.Matrix:
   inverse_derivatives = inverse_metric_derivatives(inverse_metric, metric_derivatives)

   def christoffel_derivative(
      derivative: int,
      upper: int,
      first: int,
      second: int,
   ) -> sp.Expr:
      value = sp.Integer(0)
      for contracted in range(DIMENSION):
         first_derivative_term = (
            metric_derivatives[first][contracted, second]
            + metric_derivatives[second][contracted, first]
            - metric_derivatives[contracted][first, second]
         )
         second_derivative_term = (
            metric_second_derivatives[derivative][first][contracted, second]
            + metric_second_derivatives[derivative][second][contracted, first]
            - metric_second_derivatives[derivative][contracted][first, second]
         )
         value += sp.Rational(1, 2) * (
            inverse_derivatives[derivative][upper, contracted] * first_derivative_term
            + inverse_metric[upper, contracted] * second_derivative_term
         )
      return value

   ricci = sp.zeros(DIMENSION)
   for first in range(DIMENSION):
      for second in range(DIMENSION):
         value = sp.Integer(0)
         for contracted in range(DIMENSION):
            value += christoffel_derivative(contracted, contracted, first, second)
            value -= christoffel_derivative(second, contracted, first, contracted)
            for inner in range(DIMENSION):
               value += (
                  christoffel[contracted][first][second]
                  * christoffel[inner][contracted][inner]
                  - christoffel[inner][first][contracted]
                  * christoffel[contracted][second][inner]
               )
         ricci[first, second] = value
   return ricci
