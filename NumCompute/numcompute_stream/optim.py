"""Finite-difference gradients and Jacobians.

Step-size handling
------------------
A fixed absolute step breaks down for large inputs. At ``|x| = 1e20`` the
update ``x + 1e-5`` rounds back to ``x``, so the derivative quietly comes
out as 0. At ``|x| = 1e8`` most of the step's digits are lost to rounding.

To avoid this, both functions use ``step = max(h, sqrt(eps) * |x_i|)``.
This is exactly ``h`` at ordinary magnitudes and only grows when ``h``
would be too small to register. We also divide by the step that was
actually representable, ``(x_i + step) - (x_i - step)``, instead of the
nominal ``2 * step``. If the step vanishes in floating point, or the
function returns NaN or infinity, a ``ValueError`` is raised rather than
returning a wrong number.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))


def _validate(h: float, method: str) -> None:
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite.")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")


def _evaluate(
    func: Callable[[NDArray[np.float64]], npt.ArrayLike], point: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Call ``func`` on a copy of ``point`` and return the result flattened.

    The copy stops ``func`` from changing our working array. Raises
    ValueError if any returned value is NaN or infinite.
    """
    value = np.asarray(func(point.copy()), dtype=float).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("the function returned NaN or infinity near x; reduce h or check f.")
    return value


def _difference_matrix(
    func: Callable[[NDArray[np.float64]], npt.ArrayLike],
    x_array: NDArray[np.float64],
    h: float,
    method: str,
) -> NDArray[np.float64]:
    """Build the ``(n_outputs, x.size)`` matrix of finite-difference derivatives.

    Complexity: ``x.size + 1`` function evaluations for forward differences
    and ``2 * x.size + 1`` for central (the extra one is the baseline).
    O(n_outputs * x.size) space.
    """
    baseline = _evaluate(func, x_array)
    matrix = np.zeros((baseline.size, x_array.size), dtype=float)
    for column, index in enumerate(np.ndindex(x_array.shape)):
        centre = x_array[index]
        # Grow the step for large |x| so it survives rounding (see module docstring).
        step = max(h, _SQRT_EPS * abs(float(centre)))
        x_plus, x_minus = x_array.copy(), x_array.copy()
        x_plus[index] = centre + step
        if method == "central":
            x_minus[index] = centre - step
        # Rounding can change the step we asked for, so divide by the one we got.
        actual_step = float(x_plus[index] - x_minus[index])
        if actual_step == 0.0 or not np.isfinite(actual_step):
            raise ValueError(f"step h={h} vanishes or overflows at x={float(centre)!r}.")
        lower = _evaluate(func, x_minus) if method == "central" else baseline
        matrix[:, column] = (_evaluate(func, x_plus) - lower) / actual_step
    return matrix


def grad(
    f: Callable[[float | NDArray[np.float64]], float],
    x: npt.ArrayLike,
    h: float = 1e-5,
    method: str = "central",
) -> float | NDArray[np.float64]:
    """
    Estimate the gradient of a scalar-valued function at a point.

    Parameters
    ----------
    f : callable
        Function that takes x and returns a scalar.
    x : scalar or array-like
        Point where the gradient is estimated.
    h : float, default=1e-5
        Base step size. Coordinate ``i`` moves by ``max(h, sqrt(eps) * |x_i|)``.
    method : {"central", "forward"}, default="central"
        Finite-difference scheme. Central differences have O(h^2) error.
        Forward differences have O(h) error but need fewer evaluations.

    Returns
    -------
    float or numpy.ndarray
        A float for a scalar ``x``, otherwise a gradient array with the
        same shape as ``x``. An empty ``x`` gives an empty array.

    Raises
    ------
    ValueError
        If h is not positive and finite, the method is unknown, the step
        vanishes in floating point, f returns NaN or infinity, or f returns
        more than one value for an array input.

    Complexity: O(d) function evaluations and O(d) workspace for d inputs.
    """
    _validate(h, method)
    x_array: NDArray[np.float64] = np.asarray(x, dtype=float)

    if x_array.ndim == 0:
        def scalar_f(point: NDArray[np.float64]) -> float:
            return f(float(point[0]))

        return float(_difference_matrix(scalar_f, x_array.reshape(1), h, method)[0, 0])

    if x_array.size == 0:
        return np.zeros_like(x_array, dtype=float)
    matrix = _difference_matrix(f, x_array, h, method)
    if matrix.shape[0] != 1:
        raise ValueError("f must return a scalar; use jacobian for vector outputs.")
    return matrix[0].reshape(x_array.shape)


def jacobian(
    F: Callable[[NDArray[np.float64]], npt.ArrayLike],
    x: npt.ArrayLike,
    h: float = 1e-5,
    method: str = "central",
) -> NDArray[np.float64]:
    """
    Estimate the Jacobian of a vector-valued function at a point.

    Row ``j``, column ``i`` holds the derivative of output ``j`` with
    respect to input ``i``. Inputs and outputs are flattened first.

    Parameters
    ----------
    F : callable
        Function that takes x and returns an array-like output.
    x : array-like
        Point where the Jacobian is estimated. A scalar is treated as a
        one-element vector.
    h : float, default=1e-5
        Base step size. Coordinate ``i`` moves by ``max(h, sqrt(eps) * |x_i|)``.
    method : {"central", "forward"}, default="central"
        Finite-difference scheme.

    Returns
    -------
    numpy.ndarray
        Array of shape (number of outputs, number of inputs).

    Raises
    ------
    ValueError
        If h is not positive and finite, the method is unknown, the step
        vanishes in floating point, or F returns NaN or infinity.

    Complexity: O(d) function evaluations (two per input for central mode),
    O(m*d) output space for m outputs and d inputs.
    """
    _validate(h, method)
    x_array: NDArray[np.float64] = np.asarray(x, dtype=float)
    if x_array.ndim == 0:
        x_array = x_array.reshape(1)
    if x_array.size == 0:
        output = np.asarray(F(x_array.copy()), dtype=float).reshape(-1)
        return np.zeros((output.size, 0), dtype=float)
    return _difference_matrix(F, x_array, h, method)
