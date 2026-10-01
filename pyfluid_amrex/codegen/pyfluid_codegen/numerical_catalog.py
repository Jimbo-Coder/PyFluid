"""Strict identity loader; numerical formulas belong in differential_equations."""

from pathlib import Path
import tomllib


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "numerics.toml"


def load_spatial_orders(path: Path = SCHEMA_PATH) -> tuple[int, ...]:
    with path.open("rb") as stream:
        schema = tomllib.load(stream)
    if set(schema) != {"schema_version", "spatial_accuracy"}:
        raise ValueError("numerics schema requires schema_version and spatial_accuracy")
    if type(schema["schema_version"]) is not int or schema["schema_version"] != 1:
        raise ValueError("unsupported numerics schema version")
    accuracy = schema["spatial_accuracy"]
    if not isinstance(accuracy, dict) or set(accuracy) != {"orders"}:
        raise ValueError("spatial_accuracy requires only orders")
    orders = accuracy["orders"]
    if not isinstance(orders, list) or not orders or any(
        type(order) is not int or order < 2 or order % 2 for order in orders
    ):
        raise ValueError("spatial_accuracy.orders must be positive even integers >= 2")
    if orders != sorted(set(orders)):
        raise ValueError("spatial_accuracy.orders must be strictly increasing")
    return tuple(orders)


SPATIAL_ORDERS = load_spatial_orders()
