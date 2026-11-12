from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

from numcompute.sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)

__all__ = [
    "StreamingTopK",
    "binary_search",
    "multi_key_sort",
    "quickselect",
    "stable_sort",
    "topk",
]


class StreamingTopK:
    """Keep the k largest (or smallest) values seen across a stream of chunks.

    Only the current best k values and their stream positions are stored, so
    memory stays O(k) however long the stream is. Each update combines the
    stored values with the new chunk, uses ``np.partition`` to find the k-th
    best value, and keeps the candidates at least that good. Ties are broken
    by stream position: the value seen first wins.

    Parameters
    ----------
    k : int
        Number of values to keep. Must be a positive integer.
    largest : bool, default=True
        Keep the largest values if True, the smallest if False.
    skipna : bool, default=False
        Ignore NaN values instead of rejecting them. NaN has no order, so it
        can never be a valid top-k value.

    Raises
    ------
    ValueError
        If ``k`` is not a positive integer.

    Examples
    --------
    >>> top = StreamingTopK(k=2)
    >>> _ = top.update([5, 1, 9]).update([7, 3])
    >>> top.result()
    array([9, 7])
    """

    def __init__(self, k: int, largest: bool = True, skipna: bool = False) -> None:
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1:
            raise ValueError(f"k must be a positive integer. Received {k!r}.")
        self.k = int(k)
        self.largest = largest
        self.skipna = skipna
        self.reset()

    def reset(self) -> None:
        """Forget every value seen so far."""
        self._values: npt.NDArray[Any] | None = None
        self._positions = np.empty(0, dtype=np.int64)
        self.n_seen_ = 0

    def update(self, values: npt.ArrayLike) -> "StreamingTopK":
        """Merge one 1D chunk into the running top k.

        Parameters
        ----------
        values : array-like of shape (n_samples,)
            Real numeric values. An empty chunk is ignored.

        Returns
        -------
        StreamingTopK
            ``self``, so updates can be chained.

        Raises
        ------
        ValueError
            If the chunk is not 1D, or contains NaN while ``skipna`` is False.
        TypeError
            If the chunk is not real numeric.

        Complexity
        ----------
        Time: O(n + k) to select candidates, plus O(c log c) to order the c
        candidates tied with the k-th value (normally c = k).
        Space: O(n + k).
        """
        chunk = np.asarray(values)
        if chunk.ndim != 1:
            raise ValueError(f"values must be a 1D array. Received shape {chunk.shape}.")
        if chunk.dtype.kind not in "biuf":
            raise TypeError(f"values must be real numeric. Received dtype {chunk.dtype}.")
        positions = self.n_seen_ + np.arange(chunk.size, dtype=np.int64)
        self.n_seen_ += int(chunk.size)

        if chunk.dtype.kind == "f":
            nan_mask = np.isnan(chunk)
            if np.any(nan_mask):
                if not self.skipna:
                    raise ValueError("values contain NaN; use skipna=True to ignore them.")
                chunk, positions = chunk[~nan_mask], positions[~nan_mask]
        if chunk.size == 0:
            return self

        if self._values is None:
            self._values = chunk[:0]
        values_all = np.concatenate((self._values, chunk))
        positions_all = np.concatenate((self._positions, positions))
        # Score so that a larger score is always better; floats avoid integer overflow.
        score = values_all.astype(np.float64)
        if not self.largest:
            score = -score
        if score.size > self.k:
            kth_best = np.partition(score, -self.k)[-self.k]
            # Keep every candidate tied with the k-th best so ties can be
            # resolved by position below, not by partition order.
            keep = score >= kth_best
            values_all, positions_all, score = values_all[keep], positions_all[keep], score[keep]
        order = np.lexsort((positions_all, -score))[: self.k]
        self._values, self._positions = values_all[order], positions_all[order]
        return self

    def result(
        self, return_indices: bool = False
    ) -> npt.NDArray[Any] | Tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
        """Return the best values so far, best first.

        Parameters
        ----------
        return_indices : bool, default=False
            Also return each value's 0-based position in the whole stream.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            Up to k values (fewer if fewer have been seen), and optionally
            their stream positions.
        """
        values = np.empty(0) if self._values is None else self._values.copy()
        if return_indices:
            return values, self._positions.copy()
        return values
