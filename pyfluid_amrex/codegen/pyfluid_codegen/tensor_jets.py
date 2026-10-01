"""Pointwise tensor values and derivative data; no chart or evolution policy."""

from dataclasses import dataclass

import sympy as sp


SYMMETRIC_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclass(frozen=True)
class SymmetricTensorJet:
    value: sp.Matrix
    first: tuple[sp.Matrix, ...]
    second: tuple[tuple[sp.Matrix, ...], ...]

    def __post_init__(self):
        matrices = [self.value, *self.first]
        if len(self.first) != 3 or len(self.second) != 3:
            raise ValueError("tensor jets require three coordinate directions")
        for row in self.second:
            if len(row) != 3:
                raise ValueError("tensor Hessians require three directions")
            matrices.extend(row)
        for matrix in matrices:
            if matrix.shape != (3, 3):
                raise ValueError("tensor jets require symmetric 3x3 components")
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if matrix[i, j] != matrix[j, i] and sp.simplify(matrix[i, j] - matrix[j, i]) != 0:
                    raise ValueError("tensor jets require symmetric 3x3 components")

    @classmethod
    def differentiate(cls, value, coordinates):
        first = tuple(value.diff(coordinate) for coordinate in coordinates)
        second = tuple(
            tuple(derivative.diff(coordinate) for coordinate in coordinates)
            for derivative in first
        )
        return cls(value, first, second)


def symbolic_tensor_jet(prefix="jet"):
    def matrix(prefix):
        return sp.Matrix(3, 3, lambda i, j: sp.Symbol(
            f"{prefix}[{SYMMETRIC_COMPONENTS.index(tuple(sorted((i, j))))}]"
        ))
    return SymmetricTensorJet(
        matrix(f"{prefix}.value"),
        tuple(matrix(f"{prefix}.first[{d}]") for d in range(3)),
        tuple(tuple(matrix(f"{prefix}.second[{d}][{e}]") for e in range(3)) for d in range(3)),
    )


@dataclass(frozen=True)
class ScalarJet:
    value: sp.Expr
    first: sp.Matrix
    second: sp.Matrix


@dataclass(frozen=True)
class VectorJet:
    value: sp.Matrix
    first: sp.Matrix  # [derivative, component]
    second: tuple[sp.Matrix, ...]  # [first derivative][second derivative, component]


def symbolic_scalar_jet(prefix):
    return ScalarJet(sp.Symbol(f"{prefix}.value"), sp.Matrix([sp.Symbol(f"{prefix}.first[{i}]") for i in range(3)]),
                     sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"{prefix}.second[{i}][{j}]")))


def symbolic_vector_jet(prefix):
    return VectorJet(
        sp.Matrix([sp.Symbol(f"{prefix}.value[{i}]") for i in range(3)]),
        sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"{prefix}.first[{i}][{j}]")),
        tuple(sp.Matrix(3, 3, lambda j, k: sp.Symbol(f"{prefix}.second[{i}][{j}][{k}]"))
              for i in range(3)),
    )
