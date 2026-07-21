import numbers

import numpy as np


def _plain_value(value):
   if isinstance(value, np.ndarray) and value.shape == ():
      return value.item()
   return value


def _values_match(left, right, tol):
   left = _plain_value(left)
   right = _plain_value(right)

   if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
      left_array = np.asarray(left)
      right_array = np.asarray(right)
      if left_array.shape != right_array.shape:
         return False
      if np.issubdtype(left_array.dtype, np.number) and np.issubdtype(right_array.dtype, np.number):
         return bool(np.allclose(left_array, right_array, rtol=tol, atol=tol))
      return bool(np.array_equal(left_array, right_array))

   if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
      return bool(np.isclose(left, right, rtol=tol, atol=tol))

   return left == right


def _dict_from_npz_or_mapping(values):
   if hasattr(values, "files"):
      return {key: values[key] for key in values.files}
   return dict(values)


def compare_parameter_dicts(reference, current, tol=1e-12, exclude=None):
   reference = _dict_from_npz_or_mapping(reference)
   current = _dict_from_npz_or_mapping(current)
   exclude = set(exclude or ())

   reference_keys = set(reference) - exclude
   current_keys = set(current) - exclude
   mismatches = []

   for key in sorted(reference_keys - current_keys):
      mismatches.append(f"{key}: missing from current parameters")
   for key in sorted(current_keys - reference_keys):
      mismatches.append(f"{key}: missing from checkpoint parameters")

   for key in sorted(reference_keys & current_keys):
      if not _values_match(reference[key], current[key], tol):
         mismatches.append(f"{key}: checkpoint={reference[key]!r}, current={current[key]!r}")

   return len(mismatches) == 0, mismatches
