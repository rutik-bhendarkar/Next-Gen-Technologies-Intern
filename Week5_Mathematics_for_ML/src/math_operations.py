"""Manual mathematical/statistical operations for Week 5.

NumPy is used for array representation and validation. Core calculations
(mean, median, mode, variance, standard deviation, probability, and matrix
operations) are implemented with explicit Python loops wherever practical.
"""
from __future__ import annotations
import math
from typing import Callable, Iterable, Sequence
import numpy as np


def _numeric_vector(values: Iterable[float], name: str = "values") -> np.ndarray:
    """Return a finite, non-empty, one-dimensional float array."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _numeric_matrix(values: Iterable[Iterable[float]], name: str = "matrix") -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values in a rectangular structure") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def manual_sum(values: Iterable[float]) -> float:
    arr = _numeric_vector(values)
    total = 0.0
    for value in arr:
        total += float(value)
    return total


def manual_mean(values: Iterable[float]) -> float:
    arr = _numeric_vector(values)
    return manual_sum(arr) / arr.size


def manual_insertion_sort(values: Iterable[float]) -> list[float]:
    """Sort values with insertion sort (no sorted()/np.sort() call)."""
    arr = _numeric_vector(values)
    result: list[float] = []
    for raw in arr:
        value = float(raw)
        index = len(result)
        result.append(value)
        while index > 0 and result[index - 1] > value:
            result[index] = result[index - 1]
            index -= 1
        result[index] = value
    return result


def manual_median(values: Iterable[float]) -> float:
    ordered = manual_insertion_sort(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def manual_mode(values: Iterable[float]) -> list[float]:
    """Return all modes, including ties, in ascending order."""
    arr = _numeric_vector(values)
    counts: dict[float, int] = {}
    for raw in arr:
        key = float(raw)
        counts[key] = counts.get(key, 0) + 1
    highest = 0
    for count in counts.values():
        if count > highest:
            highest = count
    modes = [value for value, count in counts.items() if count == highest]
    return manual_insertion_sort(modes)


def manual_min(values: Iterable[float]) -> float:
    arr = _numeric_vector(values)
    current = float(arr[0])
    for raw in arr[1:]:
        if raw < current:
            current = float(raw)
    return current


def manual_max(values: Iterable[float]) -> float:
    arr = _numeric_vector(values)
    current = float(arr[0])
    for raw in arr[1:]:
        if raw > current:
            current = float(raw)
    return current


def manual_variance(values: Iterable[float], sample: bool = False) -> float:
    """Population variance (default) or unbiased sample variance."""
    arr = _numeric_vector(values)
    n = arr.size
    if sample and n < 2:
        raise ValueError("Sample variance requires at least two observations")
    avg = manual_mean(arr)
    squared_difference_sum = 0.0
    for raw in arr:
        difference = float(raw) - avg
        squared_difference_sum += difference * difference
    denominator = n - 1 if sample else n
    return squared_difference_sum / denominator


def manual_standard_deviation(values: Iterable[float], sample: bool = False) -> float:
    return math.sqrt(manual_variance(values, sample=sample))


def empirical_probability(event_count: int, total_count: int) -> float:
    """Probability from observed counts; returns a value in [0, 1]."""
    if not isinstance(event_count, int) or not isinstance(total_count, int):
        raise TypeError("event_count and total_count must be integers")
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if event_count < 0 or event_count > total_count:
        raise ValueError("event_count must be between 0 and total_count")
    return event_count / total_count


def probability_from_data(values: Iterable[float], condition: Callable[[float], bool]) -> float:
    arr = _numeric_vector(values)
    event_count = 0
    for raw in arr:
        if condition(float(raw)):
            event_count += 1
    return empirical_probability(event_count, int(arr.size))


def vector_add(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    a, b = _numeric_vector(left, "left vector"), _numeric_vector(right, "right vector")
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same length")
    out = np.zeros(a.size, dtype=float)
    for i in range(a.size): out[i] = a[i] + b[i]
    return out


def vector_subtract(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    a, b = _numeric_vector(left, "left vector"), _numeric_vector(right, "right vector")
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same length")
    out = np.zeros(a.size, dtype=float)
    for i in range(a.size): out[i] = a[i] - b[i]
    return out


def vector_scale(vector: Iterable[float], scalar: float) -> np.ndarray:
    a = _numeric_vector(vector, "vector")
    if not math.isfinite(float(scalar)):
        raise ValueError("scalar must be finite")
    out = np.zeros(a.size, dtype=float)
    for i in range(a.size): out[i] = a[i] * float(scalar)
    return out


def vector_dot(left: Iterable[float], right: Iterable[float]) -> float:
    a, b = _numeric_vector(left, "left vector"), _numeric_vector(right, "right vector")
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same length")
    total = 0.0
    for i in range(a.size): total += float(a[i]) * float(b[i])
    return total


def vector_norm(vector: Iterable[float]) -> float:
    return math.sqrt(vector_dot(vector, vector))


def matrix_add(left: Iterable[Iterable[float]], right: Iterable[Iterable[float]]) -> np.ndarray:
    a, b = _numeric_matrix(left, "left matrix"), _numeric_matrix(right, "right matrix")
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same dimensions for addition")
    out = np.zeros(a.shape, dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]): out[i, j] = a[i, j] + b[i, j]
    return out


def matrix_transpose(matrix: Iterable[Iterable[float]]) -> np.ndarray:
    a = _numeric_matrix(matrix)
    out = np.zeros((a.shape[1], a.shape[0]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]): out[j, i] = a[i, j]
    return out


def matrix_multiply(left: Iterable[Iterable[float]], right: Iterable[Iterable[float]]) -> np.ndarray:
    """Matrix product using three explicit loops (not np.matmul)."""
    a, b = _numeric_matrix(left, "left matrix"), _numeric_matrix(right, "right matrix")
    if a.shape[1] != b.shape[0]:
        raise ValueError("For A(m x n) and B(p x q), n must equal p")
    out = np.zeros((a.shape[0], b.shape[1]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            total = 0.0
            for k in range(a.shape[1]): total += a[i, k] * b[k, j]
            out[i, j] = total
    return out


def describe_numbers(values: Iterable[float]) -> dict[str, float | list[float]]:
    arr = _numeric_vector(values)
    return {
        "count": int(arr.size),
        "minimum": manual_min(arr),
        "maximum": manual_max(arr),
        "range": manual_max(arr) - manual_min(arr),
        "mean": manual_mean(arr),
        "median": manual_median(arr),
        "mode": manual_mode(arr),
        "population_variance": manual_variance(arr),
        "population_standard_deviation": manual_standard_deviation(arr),
        "sample_variance": manual_variance(arr, sample=True) if arr.size > 1 else float("nan"),
        "sample_standard_deviation": manual_standard_deviation(arr, sample=True) if arr.size > 1 else float("nan"),
    }
