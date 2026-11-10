"""Shared numeric helpers: distances, activations, top-k, batching and input checks."""

import numpy as np
import numpy.typing as npt

from numcompute.utils import batches, logsumexp, relu, sigmoid, softmax, topk_indices

__all__ = [
    "batches",
    "check_positive_int",
    "check_supervised_chunk",
    "cosine_similarity",
    "euclidean_distance",
    "logsumexp",
    "manhattan_distance",
    "relu",
    "sigmoid",
    "softmax",
    "topk_indices",
]


def _paired_finite(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], ...]:
    """Check two arrays share a shape and are finite, then return them flattened."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"a and b must have the same shape. Received {left.shape} and {right.shape}.")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("a and b must not contain NaN or infinity.")
    return left.ravel(), right.ravel()


def _scaled_difference(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float]:
    """Return ``(left - right) / scale`` and ``scale``, avoiding overflow.

    Subtracting directly can overflow: ``1e308 - (-1e308)`` is infinity.
    Dividing both arrays by their largest magnitude first keeps the scaled
    difference in [-2, 2]. If both arrays are all zero, scale is 0.0.
    """
    scale = float(max(np.max(np.abs(left), initial=0.0), np.max(np.abs(right), initial=0.0)))
    if scale == 0.0:
        return np.zeros_like(left), 0.0
    return left / scale - right / scale, scale


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Return the Euclidean (L2) distance between two arrays of the same shape.

    Values are rescaled before squaring, so inputs near the float64 limit
    (for example ``1e200``) do not overflow or underflow. The result is
    ``inf`` only when the true distance is larger than the float range.

    Raises
    ------
    ValueError
        On a shape mismatch or NaN/infinite input.

    Complexity
    ----------
    Time: O(n). Space: O(n) temporaries.
    """
    difference, scale = _scaled_difference(*_paired_finite(a, b))
    with np.errstate(over="ignore"):
        return float(scale * np.sqrt(np.dot(difference, difference)))


def manhattan_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Return the Manhattan (L1) distance between two arrays of the same shape.

    Uses the same rescaling as :func:`euclidean_distance` to avoid overflow.

    Raises
    ------
    ValueError
        On a shape mismatch or NaN/infinite input.

    Complexity
    ----------
    Time: O(n). Space: O(n) temporaries.
    """
    difference, scale = _scaled_difference(*_paired_finite(a, b))
    with np.errstate(over="ignore"):
        return float(scale * np.sum(np.abs(difference)))


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Return the cosine similarity of two arrays of the same shape.

    Each vector is first divided by its own largest magnitude, so very large
    or very small values do not overflow or underflow. The result is clipped
    to [-1, 1] to absorb rounding error. If either vector is all zeros the
    similarity is defined as 0.0.

    Raises
    ------
    ValueError
        On a shape mismatch or NaN/infinite input.

    Complexity
    ----------
    Time: O(n). Space: O(n) temporaries.
    """
    left, right = _paired_finite(a, b)
    left_scale = np.max(np.abs(left), initial=0.0)
    right_scale = np.max(np.abs(right), initial=0.0)
    if left_scale == 0 or right_scale == 0:
        return 0.0
    left, right = left / left_scale, right / right_scale
    similarity = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(similarity, -1.0, 1.0))


def check_positive_int(
    value: object, name: str, *, allow_none: bool = False
) -> int | None:
    """Check that a configuration value is a positive integer.

    Parameters
    ----------
    value : object
        Value to check. Booleans are rejected even though they subclass int.
    name : str
        Parameter name to show in the error message.
    allow_none : bool, default=False
        Whether ``None`` is allowed (usually meaning "no limit").

    Returns
    -------
    int or None
        The value as a Python int, or None if None was passed and allowed.

    Raises
    ------
    ValueError
        If the value is not a positive integer (or None, when allowed).

    Complexity
    ----------
    Time: O(1)
    Space: O(1)
    """
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < 1
    ):
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive integer{suffix}. Received {value!r}.")
    return int(value)


def check_supervised_chunk(
    X: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
    """Check one labelled chunk before incremental training or scoring.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        Feature chunk. A 1D array is treated as a single feature.
    y : array-like of shape (n_samples,)
        Target labels aligned with the rows of ``X``.

    Returns
    -------
    tuple of numpy.ndarray
        ``X`` as a 2D array (dtype unchanged) and ``y`` as a 1D array.

    Raises
    ------
    ValueError
        If ``X`` is not 1D or 2D, ``y`` is not 1D, the sample counts differ,
        or the chunk is empty.

    Complexity
    ----------
    Time: O(1) beyond array conversion (reshape returns a view).
    Space: O(1) beyond array conversion.
    """
    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(f"X must be a 1D or 2D array. Received shape {features.shape}.")
    if targets.ndim != 1:
        raise ValueError(f"y must be a 1D array. Received shape {targets.shape}.")
    if features.shape[0] != targets.size:
        raise ValueError(
            "X and y must contain the same number of samples. "
            f"Received {features.shape[0]} and {targets.size}."
        )
    if targets.size == 0:
        raise ValueError("X and y must not be empty.")
    return features, targets
