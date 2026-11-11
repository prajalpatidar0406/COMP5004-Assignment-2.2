"""Descriptive statistics for whole arrays and for data streams.

The batch functions (``mean``, ``std``, ``quantile``, ``histogram`` and so on)
work on a complete array. For data that arrives in chunks,
:class:`StreamingStats` and its functional wrapper :func:`update_stats` keep
the same statistics up to date one chunk at a time:

* mean, variance, min and max are exact and use O(1) memory per feature.
  Chunks are merged with the parallel Welford update of Chan et al.
* quantiles are exact over a sliding window. Without a window they come from
  a uniform reservoir sample, which is exact until the reservoir is full.
* histograms keep exact counts, either over a range that widens as needed or
  over a sliding window.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from .utils import check_positive_int


def _as_numeric_array(
    data: npt.ArrayLike,
) -> NDArray[np.float64]:
    """Convert the input to a float NumPy array, rejecting non-numeric data.

    Parameters
    ----------
    data : array-like
        Integer or float values.

    Returns
    -------
    numpy.ndarray
        The input as a float array.

    Raises
    ------
    TypeError
        If the input is not integer or float, for example strings, booleans
        or complex numbers.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = np.asarray(data)

    if arr.dtype.kind not in "iuf":
        raise TypeError(f"Data must contain real numeric values. Received dtype {arr.dtype}.")

    return arr.astype(float, copy=False)


def _validate_data(
    data: npt.ArrayLike,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Check that the input is numeric and, unless allowed, not empty.

    Parameters
    ----------
    data : array-like
        Input data to check.
    allow_empty : bool, default=False
        Whether an empty array is accepted.

    Returns
    -------
    numpy.ndarray
        The input as a float array.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the array is empty and ``allow_empty`` is False.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = _as_numeric_array(data)

    if arr.size == 0 and not allow_empty:
        raise ValueError("Data must not be empty")

    return arr


def mean(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the arithmetic mean.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes to average over. None averages over all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Mean value or array of mean values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmean(arr, axis=axis)

    return np.mean(arr, axis=axis)


def median(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the median.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes to take the median over. None uses all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Median value or array of median values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmedian(arr, axis=axis)

    return np.median(arr, axis=axis)


def std(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the standard deviation.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes to compute over. None uses all elements.
    ddof : int, default=0
        Delta degrees of freedom. 0 gives the population standard deviation
        and 1 gives the sample standard deviation.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Standard deviation value or array of values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty, ``ddof`` is not a non-negative integer, or
        the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")

    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)

    return np.std(arr, axis=axis, ddof=ddof)


def minimum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the smallest value.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes to search along. None uses all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Minimum value or array of minimum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmin(arr, axis=axis)

    return np.min(arr, axis=axis)


def maximum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the largest value.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes to search along. None uses all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Maximum value or array of maximum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmax(arr, axis=axis)

    return np.max(arr, axis=axis)


def quantile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return one or more quantiles of the data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Quantile level(s) between 0 and 1, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes to compute over. None uses all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Quantile value or array of quantile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, ``q`` is outside [0, 1] or not finite, or the
        axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    if skipna:
        return np.nanquantile(arr, q_arr, axis=axis)

    return np.quantile(arr, q_arr, axis=axis)


def percentile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return one or more percentiles of the data.

    This is :func:`quantile` with ``q`` given on a 0 to 100 scale.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Percentile(s) between 0 and 100, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes to compute over. None uses all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Percentile value or array of percentile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, ``q`` is outside [0, 100] or not finite, or
        the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(data, q_arr / 100.0, axis=axis, skipna=skipna)


def histogram(
    data: npt.ArrayLike,
    bins: int | str | npt.ArrayLike = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    skipna: bool = False,
) -> tuple[NDArray[np.int64] | NDArray[np.float64], NDArray[np.float64]]:
    """Count how many values fall into each bin.

    The data is flattened first, so the result covers all elements.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    bins : int, str, or array-like, default=10
        Number of equal-width bins, the name of a NumPy binning strategy
        (such as ``"auto"``), or explicit bin edges.
    range : tuple of float, optional
        ``(lower, upper)`` limits of the bins. Defaults to the minimum and
        maximum of the data.
    density : bool, default=False
        If True, return a probability density (normalised so the total area
        is 1) instead of counts.
    skipna : bool, default=False
        If True, ignore NaN values. Infinite values are always rejected.

    Returns
    -------
    tuple of numpy.ndarray
        ``(counts, edges)``. ``counts`` holds one count (or density) per
        bin, and ``edges`` has one more value than ``counts``.

    Raises
    ------
    TypeError
        If the data contains non-numeric or complex values.
    ValueError
        If no valid values remain, the data contains NaN while ``skipna`` is
        False, the data contains infinity, or the bins are invalid (a
        non-positive count, or edges that are not 1D, finite and strictly
        increasing).

    Complexity: O(n + b) time, where n is the number of observations
    and b is the number of bins.
    """
    arr = _validate_data(data, allow_empty=True).ravel()

    if np.any(np.isinf(arr)):
        raise ValueError("Data must not contain infinite values")

    if skipna:
        arr = arr[~np.isnan(arr)]
    elif np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN; use skipna=True to ignore it")

    if arr.size == 0:
        raise ValueError("Data must contain at least one valid value")

    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("Bins must be a positive integer")
    elif not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "Bin edges must be a 1D array with at least 2 values"
            )

        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Bin edges must be finite and strictly increasing"
            )

    counts, edges = np.histogram(
        arr,
        bins=bins,
        range=range,
        density=density,
    )

    return counts, edges



@contextmanager
def _suppress_nan_warnings() -> Iterator[None]:
    """Silence NumPy's RuntimeWarning for all-NaN slices, whose NaN result is expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


class StreamingHistogram:
    """Streaming histogram with a fixed number of bins and exact counts.

    Parameters
    ----------
    bins : int, default=10
        Number of equal-width bins.
    range : tuple of float or None, default=None
        Fixed ``(lower, upper)`` range. Values outside it are not binned;
        they are counted in ``n_out_of_range_`` instead. With None the range
        adapts to the data.

    Notes
    -----
    In adaptive mode the starting range is the range of the first chunk, as
    in ``numpy.histogram``. When later data falls outside it, the bin width
    doubles and neighbouring pairs of bins are merged, which extends the
    range to the left or right. Because whole bins are merged, every stored
    count is still exact for the new, coarser bins, so no observation has to
    be stored or re-binned. Each doubling costs O(bins), and the number of
    doublings grows with the logarithm of how far the data range expands.

    Every edge is stored as ``anchor + unit * m`` for an integer ``m``. A
    doubling only changes the integers, so an edge that survives it keeps
    exactly the same floating-point value, and values that were already
    counted can never drift across it. As in ``numpy.histogram``, bins are
    half-open ``[a, b)`` except the last, which also includes its upper edge.
    """

    def __init__(self, bins: int = 10, range: tuple[float, float] | None = None) -> None:
        self.bins = int(check_positive_int(bins, "bins"))
        if range is not None:
            lower, upper = (float(edge) for edge in range)
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError("range must be finite with lower < upper.")
        self.range = range
        self.reset()

    def reset(self) -> None:
        """Clear all counts. In adaptive mode the learned range is cleared too."""
        self._counts = np.zeros(self.bins, dtype=np.int64)
        # Values exactly on the upper edge are kept separately: if the range
        # later grows to the right, that edge moves inside the range and these
        # values must be folded into the bin that now contains them.
        self._upper_edge_count = 0
        self.n_out_of_range_ = 0
        self._anchor: float | None = None
        self._unit = 1.0
        self._scale = 1  # current bin width, measured in units
        self._shift = 0  # lowest edge is ``anchor - shift * unit``
        self._single_bin_extrema = (np.inf, -np.inf)
        if self.range is not None:
            self._start(float(self.range[0]), float(self.range[1]))

    def _start(self, lower: float, upper: float) -> None:
        """Set the initial edges so they cover ``[lower, upper]``."""
        unit = (upper - lower) / self.bins
        # Rounding can leave the top edge just below ``upper``, so nudge the
        # unit up until it is covered.
        while lower + unit * self.bins < upper:
            unit = float(np.nextafter(unit, np.inf))
        self._anchor, self._unit = lower, unit

    def _edges(self) -> NDArray[np.float64]:
        assert self._anchor is not None
        # ``scale`` is a power of two and ``shift`` is an integer, so these
        # products are exact in float64 for any realistic number of doublings.
        multiples = np.arange(self.bins + 1, dtype=np.float64) * self._scale - self._shift
        return self._anchor + self._unit * multiples

    def _double(self, shift: int) -> None:
        """Double the bin width by merging pairs of bins, and move the lowest edge left.

        ``shift`` is the number of new bins added on the left. Any integer in
        ``[0, bins // 2]`` keeps the merged bins lined up with the new edges
        and keeps the old range covered.
        """
        merged = np.add.reduceat(self._counts, np.arange(0, self.bins, 2))
        counts = np.zeros(self.bins, dtype=np.int64)
        counts[shift : shift + merged.size] = merged

        # Distance from the new lowest edge to the old upper edge, in old bin
        # widths.
        old_upper_offset = 2 * shift + self.bins
        if old_upper_offset < 2 * self.bins:
            # The old upper edge is now inside the range, so its values belong
            # to the bin that contains it.
            counts[old_upper_offset // 2] += self._upper_edge_count
            self._upper_edge_count = 0

        self._shift += 2 * self._scale * shift
        self._scale *= 2
        self._counts = counts
        if not np.all(np.isfinite(self._edges()[[0, -1]])):
            raise ValueError("Histogram range overflowed; supply a fixed range instead.")

    def _left_shift_for(self, low: float, edges: NDArray[np.float64]) -> int:
        """Return the smallest left shift, in new bins, that reaches ``low``.

        The result is capped at ``bins // 2``. Growing only as far as the data
        needs keeps the bins narrow: a value just below the range adds one new
        bin rather than half the histogram.
        """
        if low >= edges[0]:
            return 0
        new_width = 2.0 * (edges[1] - edges[0])
        needed = int(np.ceil((edges[0] - low) / new_width))
        return int(np.clip(needed, 1, self.bins // 2))

    def update(self, values: npt.ArrayLike) -> StreamingHistogram:
        """Add a chunk of finite values to the histogram.

        The chunk is flattened, so any shape is accepted. Empty chunks are
        ignored.

        Raises
        ------
        ValueError
            If a value is NaN or infinite, or the adaptive range grows past
            what float64 can represent.

        Complexity
        ----------
        Time: O(n log bins + bins * d), where d is the number of doublings.
        Space: O(bins).
        """
        data = np.asarray(values, dtype=np.float64).ravel()
        if data.size == 0:
            return self
        if not np.all(np.isfinite(data)):
            raise ValueError("StreamingHistogram values must be finite.")

        if self.bins == 1 and self.range is None:
            # A single bin has no pairs to merge, so its edges are simply the
            # smallest and largest values seen.
            low, high = float(np.min(data)), float(np.max(data))
            old_low, old_high = self._single_bin_extrema
            self._single_bin_extrema = (min(low, old_low), max(high, old_high))
            self._anchor = self._single_bin_extrema[0]
            self._counts[0] += data.size
            return self

        if self._anchor is None:
            lower, upper = float(np.min(data)), float(np.max(data))
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            self._start(lower, upper)
        edges = self._edges()
        if self.range is None:
            low, high = float(np.min(data)), float(np.max(data))
            while low < edges[0] or high > edges[-1]:
                self._double(self._left_shift_for(low, edges))
                edges = self._edges()
        else:
            inside = (data >= edges[0]) & (data <= edges[-1])
            self.n_out_of_range_ += int(data.size - np.count_nonzero(inside))
            data = data[inside]

        # Search the same edges that ``edges`` reports, so each value's bin
        # always agrees with the published edges.
        index = np.searchsorted(edges, data, side="right") - 1
        on_upper = index == self.bins
        self._upper_edge_count += int(np.count_nonzero(on_upper))
        self._counts += np.bincount(index[~on_upper], minlength=self.bins)
        return self

    @property
    def counts(self) -> NDArray[np.int64]:
        """Counts per bin, shape ``(bins,)``."""
        counts = self._counts.copy()
        counts[-1] += self._upper_edge_count
        return counts

    @property
    def edges(self) -> NDArray[np.float64]:
        """Bin edges, shape ``(bins + 1,)``. All NaN until the range is known."""
        if self._anchor is None:
            return np.full(self.bins + 1, np.nan)
        if self.bins == 1 and self.range is None:
            low, high = self._single_bin_extrema
            return np.array([low - 0.5, high + 0.5] if low == high else [low, high])
        return self._edges()


class StreamingStats:
    """Descriptive statistics that are updated one chunk at a time.

    Each chunk can be 1D (one feature) or 2D with shape
    ``(n_samples, n_features)``. Statistics are kept per feature, and a 1D
    stream reports scalars.

    Parameters
    ----------
    bins : int, default=10
        Number of histogram bins per feature.
    quantiles : array-like of float, default=(0.25, 0.5, 0.75)
        Quantile levels in [0, 1] to report.
    window_size : int or None, default=None
        If set, every statistic is computed exactly over just the most recent
        ``window_size`` rows (a sliding window).
    skipna : bool, default=False
        If True, ignore NaN values instead of raising an error. Infinity is
        always rejected.
    range : tuple of float or None, default=None
        Fixed histogram range for unbounded streams (see
        :class:`StreamingHistogram`). Ignored in window mode.
    reservoir_size : int, default=10_000
        Number of rows kept in the uniform reservoir sample that quantiles
        of an unbounded stream are computed from. Quantiles are exact until
        the stream has more rows than this, and approximate after that.
    ddof : int, default=0
        Delta degrees of freedom used for ``variance`` and ``std``.
    random_state : int or None, default=None
        Seed for the reservoir sampling.

    Notes
    -----
    Memory is O(features * (bins + reservoir_size)) for an unbounded stream
    and O(features * window_size) in window mode. Neither depends on how long
    the stream is. An unbounded update costs O(n_chunk * features); a window
    update recomputes over the whole window in O(window_size * features).

    Examples
    --------
    >>> stats = StreamingStats(bins=5)
    >>> _ = stats.update([1.0, 2.0, 3.0]).update([4.0, 5.0])
    >>> stats["mean"], stats["variance"]
    (3.0, 2.0)
    """

    def __init__(
        self,
        bins: int = 10,
        quantiles: npt.ArrayLike = (0.25, 0.5, 0.75),
        window_size: int | None = None,
        skipna: bool = False,
        range: tuple[float, float] | None = None,
        reservoir_size: int = 10_000,
        ddof: int = 0,
        random_state: int | None = None,
    ) -> None:
        self.bins = int(check_positive_int(bins, "bins"))
        levels = np.asarray(quantiles, dtype=float)
        if levels.ndim > 1 or not np.all(np.isfinite(levels)) or np.any((levels < 0) | (levels > 1)):
            raise ValueError("quantiles must be finite values between 0 and 1.")
        self.quantile_levels = levels
        self.window_size = check_positive_int(window_size, "window_size", allow_none=True)
        self.skipna = bool(skipna)
        self.range = range
        StreamingHistogram(self.bins, range)  # fail now on a bad range, not at the first update
        self.reservoir_size = int(check_positive_int(reservoir_size, "reservoir_size"))
        if not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer.")
        self.ddof = int(ddof)
        self.random_state = random_state
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all accumulated data. The settings are kept."""
        self._rng = np.random.default_rng(self.random_state)
        self._n_features: int | None = None
        self._scalar_output = True
        self._rows_seen = 0

    def _initialise(self, n_features: int) -> None:
        self._n_features = n_features
        self._count = np.zeros(n_features, dtype=np.int64)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)
        self._reservoir = np.empty((0, n_features))
        self._window = np.empty((0, n_features))
        self._histograms = [StreamingHistogram(self.bins, self.range) for _ in range(n_features)]

    def _validate_chunk(self, X_chunk: npt.ArrayLike) -> NDArray[np.float64]:
        raw = np.asarray(X_chunk)
        if raw.dtype.kind not in "biuf":
            raise TypeError(f"X_chunk must contain real numeric values. Received dtype {raw.dtype}.")
        if raw.ndim not in (1, 2):
            raise ValueError(f"X_chunk must be 1D or 2D. Received shape {raw.shape}.")
        chunk = raw.astype(np.float64, copy=False)
        if np.any(np.isinf(chunk)):
            raise ValueError("X_chunk must not contain infinity.")
        if not self.skipna and np.any(np.isnan(chunk)):
            raise ValueError("X_chunk contains NaN; use skipna=True to ignore it.")
        scalar_output = chunk.ndim == 1
        chunk = chunk.reshape(-1, 1) if scalar_output else chunk
        if self._n_features is None:
            self._initialise(chunk.shape[1])
            self._scalar_output = scalar_output
        elif chunk.shape[1] != self._n_features:
            raise ValueError(
                f"X_chunk has {chunk.shape[1]} features; expected {self._n_features}."
            )
        return chunk

    def _merge_moments(self, chunk: NDArray[np.float64]) -> None:
        """Merge a chunk's per-feature count, mean, M2, min and max (Chan et al.)."""
        valid = ~np.isnan(chunk)
        n_chunk = valid.sum(axis=0)
        safe_n = np.maximum(n_chunk, 1)
        chunk_mean = np.where(valid, chunk, 0.0).sum(axis=0) / safe_n
        chunk_m2 = np.where(valid, (chunk - chunk_mean) ** 2, 0.0).sum(axis=0)

        total = self._count + n_chunk
        safe_total = np.maximum(total, 1)
        delta = chunk_mean - self._mean
        self._mean = self._mean + delta * n_chunk / safe_total
        self._m2 = self._m2 + chunk_m2 + delta**2 * self._count * n_chunk / safe_total
        self._count = total
        self._min = np.minimum(self._min, np.where(valid, chunk, np.inf).min(axis=0))
        self._max = np.maximum(self._max, np.where(valid, chunk, -np.inf).max(axis=0))

    def _sample_reservoir(self, chunk: NDArray[np.float64]) -> None:
        """Keep a uniform sample of all rows seen, using a vectorised Algorithm R.

        Row ``t`` (0-based, counted over the whole stream) replaces a randomly
        chosen slot with probability ``k / (t + 1)``. These decisions do not
        depend on what is in the reservoir, so they can all be applied at once
        as a single scatter where the last write to each slot wins. This gives
        the same result as running the algorithm one row at a time.
        """
        free = self.reservoir_size - self._reservoir.shape[0]
        fill = chunk[:free]
        if fill.shape[0]:
            self._reservoir = np.concatenate((self._reservoir, fill))
        rest = chunk[fill.shape[0] :]
        if rest.shape[0]:
            positions = self._rows_seen + fill.shape[0] + np.arange(rest.shape[0])
            slots = self._rng.integers(0, positions + 1)
            accepted = slots < self.reservoir_size
            slots, rows = slots[accepted], rest[accepted]
            # np.unique on the reversed array finds the last write to each slot.
            _, last_from_end = np.unique(slots[::-1], return_index=True)
            last = slots.size - 1 - last_from_end
            self._reservoir[slots[last]] = rows[last]

    def update(self, X_chunk: npt.ArrayLike) -> StreamingStats:
        """Merge one chunk into the statistics.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
            Real numeric values. An empty chunk is ignored.

        Returns
        -------
        StreamingStats
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError
            If the chunk does not contain real numeric values.
        ValueError
            If the chunk is not 1D or 2D, contains infinity, contains NaN
            while ``skipna`` is False, or has a different number of features
            from earlier chunks.
        """
        chunk = self._validate_chunk(X_chunk)
        if chunk.shape[0] == 0:
            return self
        if self.window_size is not None:
            self._window = np.concatenate((self._window, chunk))[-self.window_size :]
        else:
            self._merge_moments(chunk)
            self._sample_reservoir(chunk)
            for feature, histogram in enumerate(self._histograms):
                column = chunk[:, feature]
                histogram.update(column[~np.isnan(column)])
        self._rows_seen += chunk.shape[0]
        return self

    # ------------------------------------------------------------------ #
    def _window_result(self) -> dict[str, Any]:
        window = self._window
        count = (~np.isnan(window)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"), _suppress_nan_warnings():
            mean = np.nanmean(window, axis=0)
            variance = np.nanvar(window, axis=0, ddof=self.ddof)
            minimum = np.nanmin(window, axis=0) if window.size else np.full(count.size, np.nan)
            maximum = np.nanmax(window, axis=0) if window.size else np.full(count.size, np.nan)
            quantiles = (
                np.nanquantile(window, self.quantile_levels, axis=0)
                if window.size
                else np.full((self.quantile_levels.size, count.size), np.nan)
            )
        histograms, edges = [], []
        for feature in np.arange(count.size):
            column = window[:, feature]
            column = column[~np.isnan(column)]
            if column.size:
                counts, feature_edges = np.histogram(column, bins=self.bins)
            else:
                counts, feature_edges = np.zeros(self.bins, np.int64), np.full(self.bins + 1, np.nan)
            histograms.append(counts)
            edges.append(feature_edges)
        return {
            "count": count,
            "mean": mean,
            "variance": variance,
            "min": minimum,
            "max": maximum,
            "quantiles": quantiles,
            "histogram": np.array(histograms),
            "bin_edges": np.array(edges),
            "quantiles_exact": True,
        }

    def _unbounded_result(self) -> dict[str, Any]:
        count = self._count
        has_data = count > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = np.where(count > self.ddof, self._m2 / (count - self.ddof), np.nan)
        with _suppress_nan_warnings():
            quantiles = (
                np.nanquantile(self._reservoir, self.quantile_levels, axis=0)
                if self._reservoir.size
                else np.full((self.quantile_levels.size, count.size), np.nan)
            )
        return {
            "count": count,
            "mean": np.where(has_data, self._mean, np.nan),
            "variance": variance,
            "min": np.where(has_data, self._min, np.nan),
            "max": np.where(has_data, self._max, np.nan),
            "quantiles": quantiles,
            "histogram": np.array([h.counts for h in self._histograms]),
            "bin_edges": np.array([h.edges for h in self._histograms]),
            "quantiles_exact": self._rows_seen <= self.reservoir_size,
        }

    def result(self) -> dict[str, Any]:
        """Return a snapshot of the current statistics.

        Returns
        -------
        dict
            Keys ``count``, ``mean``, ``variance``, ``std``, ``min``,
            ``max``, ``quantiles``, ``histogram``, ``bin_edges``,
            ``quantile_levels``, ``quantiles_exact`` and ``n_rows``. For a 1D
            stream the statistics are scalars, ``quantiles`` has shape
            ``(n_levels,)`` and ``histogram`` has shape ``(bins,)``. For a 2D
            stream a feature axis is added: last for ``quantiles``, first for
            ``histogram`` and ``bin_edges``. Values that are not defined yet,
            because no data has arrived, are NaN.
        """
        if self._n_features is None:
            stats = {
                "count": np.zeros(1, np.int64),
                "mean": np.full(1, np.nan),
                "variance": np.full(1, np.nan),
                "min": np.full(1, np.nan),
                "max": np.full(1, np.nan),
                "quantiles": np.full((self.quantile_levels.size, 1), np.nan),
                "histogram": np.zeros((1, self.bins), np.int64),
                "bin_edges": np.full((1, self.bins + 1), np.nan),
                "quantiles_exact": True,
            }
        elif self.window_size is not None:
            stats = self._window_result()
        else:
            stats = self._unbounded_result()
        stats["std"] = np.sqrt(stats["variance"])

        if self._scalar_output:
            for key in ("count", "mean", "variance", "std", "min", "max"):
                value = stats[key][0]
                stats[key] = int(value) if key == "count" else float(value)
            stats["quantiles"] = stats["quantiles"][:, 0]
            stats["histogram"] = stats["histogram"][0]
            stats["bin_edges"] = stats["bin_edges"][0]
        stats["quantile_levels"] = self.quantile_levels.copy()
        stats["n_rows"] = self._rows_seen
        return stats

    def __getitem__(self, key: str) -> Any:
        """Return one entry of :meth:`result`, so ``stats["mean"]`` works."""
        return self.result()[key]

    @property
    def count(self) -> Any:
        """Number of non-missing values per feature."""
        return self.result()["count"]

    @property
    def mean(self) -> Any:
        """Running mean per feature."""
        return self.result()["mean"]

    @property
    def variance(self) -> Any:
        """Running variance per feature, using ``ddof``."""
        return self.result()["variance"]

    def __repr__(self) -> str:
        mode = f"window_size={self.window_size}" if self.window_size else "unbounded"
        return f"StreamingStats({mode}, rows={self._rows_seen}, bins={self.bins})"


def update_stats(
    X_chunk: npt.ArrayLike,
    state: StreamingStats | None = None,
    bins: int = 10,
    *,
    quantiles: npt.ArrayLike = (0.25, 0.5, 0.75),
    window_size: int | None = None,
    skipna: bool = False,
    **options: Any,
) -> StreamingStats:
    """Add one chunk to streaming statistics, creating them on the first call.

    Parameters
    ----------
    X_chunk : array-like of shape (n_samples,) or (n_samples, n_features)
        The new data.
    state : StreamingStats or None, default=None
        The state returned by a previous call. None starts a new stream.
    bins, quantiles, window_size, skipna, **options
        Settings for a new :class:`StreamingStats`. They are ignored when
        ``state`` is given, because an existing state keeps its own settings.

    Returns
    -------
    StreamingStats
        The updated state. Read it with dictionary access such as
        ``state["mean"]`` or with :meth:`StreamingStats.result`.

    Raises
    ------
    TypeError
        If ``state`` is neither None nor a :class:`StreamingStats`.
    TypeError, ValueError
        If the chunk is invalid; see :meth:`StreamingStats.update`.

    Complexity: the same as one :meth:`StreamingStats.update` call (see the
    class Notes).

    Examples
    --------
    >>> state = update_stats([1, 2, 3])
    >>> state = update_stats([4, 5], state)
    >>> state["count"], state["mean"]
    (5, 3.0)
    """
    if state is None:
        state = StreamingStats(
            bins=bins, quantiles=quantiles, window_size=window_size, skipna=skipna, **options
        )
    elif not isinstance(state, StreamingStats):
        raise TypeError("state must be a StreamingStats returned by update_stats, or None.")
    return state.update(X_chunk)
