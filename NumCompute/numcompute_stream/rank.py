import numpy as np

from numcompute.rank import Ranker


class StreamingRanker:
    """Percentiles and percentile ranks over every value seen in a stream.

    Each chunk is merged into a sorted reference array with
    ``np.searchsorted`` and ``np.insert``, so the data never needs a full
    re-sort. Results are exact, which means every valid value is kept:
    memory is O(N) for N values. For bounded memory with approximate
    quantiles, use ``numcompute_stream.stats.StreamingStats`` instead.

    Values that cannot be read as numbers (including NaN) are treated as
    missing, as in :class:`Ranker`: they are counted in ``n_missing_`` and
    otherwise ignored.

    Examples
    --------
    >>> ranker = StreamingRanker()
    >>> _ = ranker.partial_fit([1, 2, 3]).partial_fit([4, 5])
    >>> float(ranker.percentile(50))
    3.0
    >>> ranker.percentile_rank([3])
    array([50.])
    """

    def __init__(self):
        self._ranker = Ranker()
        self.reset()

    def reset(self):
        """Forget every value seen so far."""
        self._sorted = np.empty(0, dtype=float)
        self.n_seen_ = 0
        self.n_missing_ = 0

    def partial_fit(self, values):
        """Merge one 1D chunk into the sorted reference values.

        Parameters
        ----------
        values : array-like of shape (n_samples,)
            Values to add. An empty chunk is ignored.

        Returns
        -------
        StreamingRanker
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If the chunk is not 1D, is None, or is a single string.

        Complexity
        ----------
        Time: O(m log m + N) for a chunk of m values and N values stored.
        Space: O(N + m).
        """
        if values is not None and not isinstance(values, str) and np.size(values) == 0:
            return self
        arr = self._ranker._safe_array(values)
        missing = np.isnan(arr)
        valid = np.sort(arr[~missing])
        # Insert the sorted chunk at its positions in one call, keeping order.
        self._sorted = np.insert(self._sorted, np.searchsorted(self._sorted, valid), valid)
        self.n_seen_ += int(arr.size)
        self.n_missing_ += int(np.count_nonzero(missing))
        return self

    def update(self, values):
        """Alias for :meth:`partial_fit`, matching the streaming metric API."""
        return self.partial_fit(values)

    def _require_data(self):
        if self._sorted.size == 0:
            raise ValueError("No valid values have been seen yet. Call partial_fit first.")

    def percentile(self, q, interpolation="linear"):
        """Return the q-th percentile of all valid values seen so far.

        Uses the same interpolation rules as :meth:`Ranker.percentile`.

        Raises
        ------
        ValueError
            If no valid values have been seen, or q or interpolation is invalid.

        Complexity
        ----------
        Time: O(N), since the stored values are already sorted. Space: O(N).
        """
        self._require_data()
        return self._ranker.percentile(self._sorted, q, interpolation)

    def percentile_rank(self, values):
        """Return the percentile rank (0-100) of each value among those seen.

        The rank is the percentage of stored values below the value, plus half
        of those equal to it, so ties are split evenly. NaN input gives NaN.

        Parameters
        ----------
        values : array-like
            Values to rank. Any shape; the output has the same shape.

        Returns
        -------
        numpy.ndarray
            Percentile ranks in [0, 100].

        Raises
        ------
        ValueError
            If no valid values have been seen yet.

        Complexity
        ----------
        Time: O(m log N) for m values (two binary searches each). Space: O(m).
        """
        self._require_data()
        query = np.asarray(values, dtype=float)
        below = np.searchsorted(self._sorted, query, side="left")
        at_or_below = np.searchsorted(self._sorted, query, side="right")
        ranks = 100.0 * (below + 0.5 * (at_or_below - below)) / self._sorted.size
        return np.where(np.isnan(query), np.nan, ranks)
