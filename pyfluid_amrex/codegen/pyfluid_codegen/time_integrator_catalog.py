"""Exact RK stage algebra and coefficient validation, independent of AMReX."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
import tomllib

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema/time_integrators.toml"


@dataclass(frozen=True)
class RungeKuttaMethod:
    name: str
    code: int
    enumerator: str
    order: int
    state: tuple[tuple[Fraction, ...], ...]
    rhs: tuple[tuple[Fraction, ...], ...]
    a: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    c: tuple[Fraction, ...]
    output_times: tuple[Fraction, ...]
    projection_weights: tuple[Fraction, ...]


def validate_order(a, b, c, order: int) -> None:
    """Rooted-tree conditions through order four, using exact rationals.

    Checking only sum(b) or the linear stability polynomial misses nonlinear
    order conditions such as b^T A c and b^T diag(c) A c.
    """
    def dot(x, y):
        return sum(u*v for u, v in zip(x, y, strict=True))

    ac = [dot(row, c) for row in a]
    c2 = [value**2 for value in c]
    conditions = [(1, sum(b), Fraction(1)),
                  (2, dot(b, c), Fraction(1, 2)),
                  (3, dot(b, c2), Fraction(1, 3)),
                  (3, dot(b, ac), Fraction(1, 6)),
                  (4, dot(b, [value**3 for value in c]), Fraction(1, 4)),
                  (4, dot(b, [x*y for x, y in zip(c, ac)]), Fraction(1, 8)),
                  (4, dot(b, [dot(row, c2) for row in a]), Fraction(1, 12)),
                  (4, dot(b, [dot(row, ac) for row in a]), Fraction(1, 24))]
    for degree, actual, expected in conditions:
        if degree <= order and actual != expected:
            raise ValueError(f"RK order-{degree} condition: {actual} != {expected}")


def method_from_record(record: dict) -> RungeKuttaMethod:
    if set(record) != {"name", "code", "enumerator", "order", "stages"}:
        raise ValueError("invalid time-integrator record keys")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", record["name"]):
        raise ValueError("invalid time-integrator name")
    if not re.fullmatch(r"Time[A-Z][A-Za-z0-9]*", record["enumerator"]):
        raise ValueError("invalid time-integrator enumerator")
    if type(record["code"]) is not int or record["code"] < 0:
        raise ValueError("time-integrator code must be a nonnegative integer")
    if type(record["order"]) is not int or not 1 <= record["order"] <= 4:
        raise ValueError("only orders 1 through 4 are validated")
    stages = record["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= 4:
        raise ValueError("runtime supports one through four RK stages")
    state, rhs = [], []
    for index, stage in enumerate(stages):
        if set(stage) != {"state", "rhs"}:
            raise ValueError("stage requires state and rhs coefficients")
        vectors = []
        for field in ("state", "rhs"):
            values = stage[field]
            if not isinstance(values, list) or len(values) != index + 1:
                raise ValueError("stage may reference only existing states and RHS values")
            if any(not isinstance(value, str) or not re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value)
                   for value in values):
                raise ValueError("coefficients must be exact integer/fraction strings")
            vectors.append(tuple(Fraction(value) for value in values))
        alpha, beta = vectors
        if sum(alpha) != 1:
            raise ValueError("stage state coefficients must preserve a constant state")
        # Existing low-storage owners combine the base and current input only.
        if any(alpha[1:index]):
            raise ValueError("stage needs unsupported nonadjacent state storage")
        state.append(alpha)
        rhs.append(beta)
    count = len(stages)
    expanded = [[Fraction(0)] * count]
    for index, (alpha, beta) in enumerate(zip(state, rhs)):
        expanded.append([sum(alpha[j] * expanded[j][k] for j in range(index+1))
                         + (beta[k] if k <= index else 0) for k in range(count)])
    a, b = expanded[:-1], expanded[-1]
    c = tuple(sum(row) for row in a)
    validate_order(a, b, c, record["order"])
    # Reverse only direct state dependencies. Effects through F are already
    # included in integrated boundary/source fluxes, not projection sources.
    sensitivity = [Fraction(0)] * (count+1)
    sensitivity[-1] = Fraction(1)
    for index in reversed(range(count)):
        for j, coefficient in enumerate(state[index]):
            sensitivity[j] += sensitivity[index+1] * coefficient
    if sensitivity[0] != 1:
        raise ValueError("base state must reach the accepted state with unit weight")
    return RungeKuttaMethod(
        record["name"], record["code"], record["enumerator"], record["order"],
        tuple(state), tuple(rhs), tuple(tuple(row) for row in a), tuple(b), c,
        tuple(sum(row) for row in expanded[1:]), tuple(sensitivity[:-1]),
    )


def load_time_integrators(path: Path = SCHEMA_PATH) -> tuple[RungeKuttaMethod, ...]:
    with path.open("rb") as stream:
        schema = tomllib.load(stream)
    if set(schema) != {"schema_version", "method"} or schema["schema_version"] != 1:
        raise ValueError("invalid time-integrator schema")
    methods = tuple(method_from_record(record) for record in schema["method"])
    if not methods:
        raise ValueError("time-integrator catalog must not be empty")
    for field in ("name", "code", "enumerator"):
        if len({getattr(method, field) for method in methods}) != len(methods):
            raise ValueError(f"duplicate time-integrator {field}")
    return methods


TIME_INTEGRATORS = load_time_integrators()
