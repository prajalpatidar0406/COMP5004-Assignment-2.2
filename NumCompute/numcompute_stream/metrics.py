"""Classification and regression metrics, in batch and streaming form.

The batch functions (``accuracy``, ``precision`` ... ``roc_auc_score``) score
a full set of predictions in one call. The streaming classes follow the
:class:`StreamingMetric` interface: ``update(y_true_chunk, y_pred_chunk)``,
``result()`` and ``reset()``. They keep running counts (or a confusion
matrix) instead of the raw data, and ``result()`` after any sequence of
chunks gives the same answer as the batch metric on all the chunks joined
together.
"""

from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from numcompute.metrics import _safe_divide

from .utils import check_positive_int


def _validate_labels(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[NDArray, NDArray]:
    """
    Check that true and predicted labels can be compared.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    tuple of numpy.ndarray
        The true and predicted labels as arrays.

    Raises
    ------
    ValueError
        If either input is not 1D, is empty, or the shapes differ, or if a
        float or complex array contains NaN or infinity.

    Complexity: O(n) time and O(n) space for n labels.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D arrays. Received shapes {true.shape} and {pred.shape}.")

    if true.size == 0 or pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Received {true.shape} and {pred.shape}.")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    if pred.dtype.kind in "fc" and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred must not contain NaN or infinity")

    return true, pred


def _validate_scores(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
) -> tuple[NDArray, NDArray[np.float64]]:
    """
    Check binary labels and their positive-class scores.

    Parameters
    ----------
    y_true : array-like
        Actual class labels, shape (n_samples,).
    y_score : array-like
        Scores for the positive class, shape (n_samples,).

    Returns
    -------
    tuple of numpy.ndarray
        The labels, and the scores converted to float.

    Raises
    ------
    TypeError
        If y_score is not a real numeric array.
    ValueError
        If the inputs are not 1D, are empty, differ in shape, or contain
        NaN or infinity.

    Complexity: O(n) time and O(n) space.
    """
    true = np.asarray(y_true)
    scores = np.asarray(y_score)

    if true.ndim != 1 or scores.ndim != 1:
        raise ValueError(f"y_true and y_score must be 1D arrays. Received shapes {true.shape} and {scores.shape}.")

    if true.size == 0 or scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")

    if true.shape != scores.shape:
        raise ValueError(f"y_true and y_score must have the same shape. Received {true.shape} and {scores.shape}.")

    if scores.dtype.kind not in "iuf":
        raise TypeError(f"y_score must contain real numeric values. Received dtype {scores.dtype}.")

    scores = scores.astype(float, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must not contain NaN or infinity")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    return true, scores


def _encode_labels(
    values: NDArray,
    labels: NDArray,
    missing_message: str,
) -> NDArray[np.intp]:
    """
    Replace each value with its index in ``labels``.

    Parameters
    ----------
    values : numpy.ndarray
        Observed labels, shape (n_samples,).
    labels : numpy.ndarray
        Distinct known labels in any order, shape (k,).
    missing_message : str
        Format string for the error message. It receives the first unknown
        label.

    Returns
    -------
    numpy.ndarray
        Integer indices into ``labels``, shape (n_samples,).

    Raises
    ------
    ValueError
        If a value is not one of ``labels``.

    Complexity: O(n + range) time for compact integer labels, which use a
    lookup table. Otherwise binary search, O((n + k) log k) time. O(n + k)
    space.
    """
    if values.size and labels.size and values.dtype.kind in "iu" and labels.dtype.kind in "iu":
        low = int(min(values.min(), labels.min()))
        span = int(max(values.max(), labels.max())) - low + 1
        if span <= 4 * (labels.size + values.size):
            # When integer labels fit in a small range, a lookup table is a
            # single O(n) gather and beats binary search.
            table = np.full(span, -1, dtype=np.intp)
            table[labels - low] = np.arange(labels.size)
            positions = table[values - low]
            if np.any(positions < 0):
                raise ValueError(missing_message.format(values[positions < 0][0]))
            return positions

    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    positions = np.searchsorted(sorted_labels, values)
    positions = np.minimum(positions, max(labels.size - 1, 0))
    found = (
        sorted_labels[positions] == values
        if labels.size
        else np.zeros(values.size, dtype=bool)
    )
    if not np.all(found):
        raise ValueError(missing_message.format(values[~found][0]))
    return order[positions]


def _count_pairs(
    rows: NDArray[np.intp],
    columns: NDArray[np.intp],
    n_labels: int,
) -> NDArray[np.int64]:
    """
    Count (row, column) index pairs in an ``(n_labels, n_labels)`` matrix.

    Each pair becomes one flat cell index, ``row * n_labels + column``, and
    the cells are counted with ``np.bincount``. This is several times faster
    than ``np.add.at``.

    Complexity: O(n + k^2) time and O(k^2) space.
    """
    cells = rows * n_labels + columns
    counts = np.bincount(cells, minlength=n_labels * n_labels)
    return counts.reshape(n_labels, n_labels).astype(np.int64, copy=False)


def _divide_or_default(
    numerator: NDArray,
    denominator: NDArray,
    zero_division: int,
) -> NDArray[np.float64]:
    """Element-wise ``numerator / denominator``, using ``zero_division`` wherever the denominator is 0."""
    safe = np.where(denominator == 0, 1, denominator)
    return np.where(denominator == 0, float(zero_division), numerator / safe)


def accuracy(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Return the fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels, with the same shape as y_true.

    Returns
    -------
    float
        Correct predictions divided by the total number of predictions.

    Raises
    ------
    ValueError
        If the arrays are empty, not 1D, or have different shapes.

    Complexity: O(n) time for n predictions.
    """
    true, pred = _validate_labels(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Count how often each true label was predicted as each label.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    labels : array-like, optional
        Class labels in the order they should appear. If omitted, every
        label seen in either array is used, in sorted order.

    Returns
    -------
    numpy.ndarray
        Integer matrix of shape (k, k). Rows are actual classes and columns
        are predicted classes.

    Raises
    ------
    ValueError
        If the input arrays are invalid, ``labels`` is empty or has
        duplicates, or an observed label is not in ``labels``.

    Complexity: O(n log k + k^2) time for n samples and k classes, since
    labels are encoded by binary search.
    """
    true, pred = _validate_labels(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(f"labels must be a 1D array. Received shape {labels.shape}.")

        if labels.size == 0:
            raise ValueError("labels must not be empty")

        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")

    actual_index = _encode_labels(true, labels, "actual label {!r} is not in labels")
    predicted_index = _encode_labels(pred, labels, "predicted label {!r} is not in labels")
    return _count_pairs(actual_index, predicted_index, labels.size)


def precision(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """Return binary precision for the chosen positive class.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value returned when nothing is predicted as positive. Must be 0 or 1.

    Returns
    -------
    float
        True positives divided by (true positives + false positives).

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fp = np.count_nonzero(~actual_positive & predicted_positive)

    return _safe_divide(tp, tp + fp, zero_division)


def recall(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """Return binary recall for the chosen positive class.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value returned when there are no actual positives. Must be 0 or 1.

    Returns
    -------
    float
        True positives divided by (true positives + false negatives).

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fn = np.count_nonzero(actual_positive & ~predicted_positive)

    return _safe_divide(tp, tp + fn, zero_division)


def f1(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """
    Return the binary F1 score for the chosen positive class.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value used whenever a denominator is zero, in precision, recall or
        F1 itself. Must be 0 or 1.

    Returns
    -------
    float
        Harmonic mean of precision and recall.

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    p = precision(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    r = recall(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )

    return _safe_divide(2 * p * r, p + r, zero_division)


def mse(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Return the mean squared error between targets and predictions.

    Parameters
    ----------
    y_true : array-like
        Actual numeric target values.
    y_pred : array-like
        Predicted numeric target values, same shape as y_true.

    Returns
    -------
    float
        Mean of (y_true - y_pred) ** 2.

    Raises
    ------
    TypeError
        If either input is not a real numeric array.
    ValueError
        If the arrays have different shapes, are empty, or contain
        NaN or infinity.

    Complexity: O(n) time for n target values.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.dtype.kind not in "iuf" or pred.dtype.kind not in "iuf":
        raise TypeError(f"y_true and y_pred must contain real numeric values. Received dtypes {true.dtype} and {pred.dtype}.")

    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Received {true.shape} and {pred.shape}.")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must not contain NaN or infinity")

    true = true.astype(float, copy=False)
    pred = pred.astype(float, copy=False)

    return float(np.mean((true - pred) ** 2))


def roc_curve(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the points of the ROC curve for binary labels and scores.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Scores for the positive class. Higher means more likely positive.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    tuple of numpy.ndarray
        (fpr, tpr, thresholds). Thresholds are in descending order and the
        first one is infinity, so the curve starts at (0, 0). There is one
        point per distinct score.

    Raises
    ------
    TypeError
        If y_score is not a real numeric array.
    ValueError
        If the inputs are empty, not 1D, have different shapes or contain
        NaN or infinity, if y_true does not have exactly two classes, or if
        positive_label does not appear in y_true.

    Complexity: O(n log n) time for sorting the n scores.
    """
    true, scores = _validate_scores(y_true, y_score)

    classes = np.unique(true)

    if classes.size != 2:
        raise ValueError("ROC curve requires exactly two classes")

    if not np.any(true == positive_label):
        raise ValueError("positive_label must be present in y_true")

    is_positive = true == positive_label
    n_positive = np.count_nonzero(is_positive)
    n_negative = true.size - n_positive

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_positive = is_positive[order]

    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.cumsum(~sorted_positive)

    distinct_indices = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_indices = np.r_[
        distinct_indices,
        sorted_scores.size - 1,
    ]

    tp = cumulative_tp[threshold_indices]
    fp = cumulative_fp[threshold_indices]

    tpr = np.r_[0.0, tp / n_positive]
    fpr = np.r_[0.0, fp / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    return fpr, tpr, thresholds


def auc(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> float:
    """
    Return the area under a curve using the trapezoidal rule.

    Parameters
    ----------
    x : array-like
        x-coordinates, either increasing or decreasing.
    y : array-like
        y-coordinates, same length as x.

    Returns
    -------
    float
        The area, as a non-negative number.

    Raises
    ------
    TypeError
        If x or y is not a real numeric array.
    ValueError
        If the inputs are not 1D, differ in length, have fewer than two
        points, contain NaN or infinity, or x is not monotonic.

    Complexity: O(n) time for n curve points.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError(f"x and y must be 1D arrays. Received shapes {x_arr.shape} and {y_arr.shape}.")

    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have the same length. Received {x_arr.size} and {y_arr.size}.")

    if x_arr.size < 2:
        raise ValueError("at least two points are required")

    if x_arr.dtype.kind not in "iuf" or y_arr.dtype.kind not in "iuf":
        raise TypeError(f"x and y must contain real numeric values. Received dtypes {x_arr.dtype} and {y_arr.dtype}.")

    x_arr = x_arr.astype(float, copy=False)
    y_arr = y_arr.astype(float, copy=False)

    if (
        not np.all(np.isfinite(x_arr))
        or not np.all(np.isfinite(y_arr))
    ):
        raise ValueError("x and y must not contain NaN or infinity")

    differences = np.diff(x_arr)

    if not (
        np.all(differences >= 0)
        or np.all(differences <= 0)
    ):
        raise ValueError("x values must be monotonic")

    return float(abs(np.trapezoid(y_arr, x_arr)))


def roc_auc_score(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> float:
    """
    Return the area under the ROC curve.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Numeric scores for the positive class.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        ROC AUC score between 0 and 1.

    Raises
    ------
    TypeError
        If y_score is not a real numeric array.
    ValueError
        If the inputs are invalid for :func:`roc_curve`, for example
        y_true does not have exactly two classes.

    Complexity: O(n log n) time for n prediction scores.
    """
    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        positive_label=positive_label,
    )

    return auc(fpr, tpr)


class StreamingMetric(ABC):
    """Base class for metrics that are updated one chunk at a time.

    Subclasses implement :meth:`update`, :meth:`reset` and :meth:`result`.
    Metrics that need positive-class scores rather than hard predictions,
    such as :class:`AUC`, set ``requires_scores = True``. Callers like
    ``StreamTrainer`` check this flag to decide whether to pass
    probabilities.
    """

    requires_scores: bool = False

    @abstractmethod
    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Add one chunk of 1D true labels and predictions (or scores) of equal length."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the accumulated state. Constructor settings are kept."""

    @abstractmethod
    def result(self) -> Any:
        """Return the metric over every chunk since the last reset."""


class Accuracy(StreamingMetric):
    """Running accuracy over every sample seen so far.

    Each sample counts equally, whatever chunk it came in. Only two integers
    are stored, so memory is O(1) and each update is O(n_chunk).
    ``result()`` is 0.0 before any update.
    """

    def __init__(self) -> None:
        self.reset()

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Add this chunk's number of correct and total samples.

        Raises
        ------
        ValueError
            If the arrays are empty, not 1D, or of different shapes.
        """
        true, pred = _validate_labels(y_true, y_pred)
        self._correct += int(np.count_nonzero(true == pred))
        self._total += int(true.size)

    def reset(self) -> None:
        """Set the correct and total counts back to zero."""
        self._correct = 0
        self._total = 0

    def result(self) -> float:
        """Return accuracy over all chunks, or 0.0 if nothing has been seen."""
        return _safe_divide(self._correct, self._total, 0)


class ConfusionMatrix(StreamingMetric):
    """Confusion matrix built up over chunks.

    Rows are true labels and columns are predicted labels, both in the order
    given by :attr:`labels_`.

    Parameters
    ----------
    labels : array-like or None, default=None
        Fixed label order. Seeing any other label raises ValueError.
        If None, the label set grows as new labels appear. It stays sorted,
        and existing counts are moved to their new positions rather than
        lost.

    Raises
    ------
    ValueError
        If ``labels`` is given but is not a non-empty 1D array, or has
        duplicates.

    Notes
    -----
    Labels are encoded with binary search, so an update costs
    O(n log k + k^2) for k labels. The k^2 term only applies when new labels
    are added. Memory is O(k^2).
    """

    def __init__(self, labels: npt.ArrayLike | None = None) -> None:
        self.labels = None if labels is None else np.asarray(labels)
        if self.labels is not None:
            if self.labels.ndim != 1 or self.labels.size == 0:
                raise ValueError(f"labels must be a non-empty 1D array. Received shape {self.labels.shape}.")
            if np.unique(self.labels).size != self.labels.size:
                raise ValueError("labels must not contain duplicates.")
        self.reset()

    def _grow_labels(self, observed: NDArray) -> None:
        """Add any new labels and move the existing counts to match the new order."""
        if self._labels is None:
            new_labels = np.unique(observed)
        else:
            new_labels = np.unique(np.concatenate((self._labels, observed)))
            if new_labels.size == self._labels.size:
                return
        expanded = np.zeros((new_labels.size, new_labels.size), dtype=np.int64)
        if self._labels is not None:
            old_positions = _encode_labels(self._labels, new_labels, "{!r}")
            expanded[np.ix_(old_positions, old_positions)] = self._matrix
        self._labels = new_labels
        self._matrix = expanded

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Add the (true, predicted) counts from one chunk.

        Raises
        ------
        ValueError
            If the arrays are invalid, or a label is not in the fixed
            ``labels``.
        """
        true, pred = _validate_labels(y_true, y_pred)
        if self.labels is None:
            self._grow_labels(np.concatenate((true, pred)))
        assert self._labels is not None
        message = "Observed label {!r} is not in the configured label set."
        rows = _encode_labels(true, self._labels, message)
        columns = _encode_labels(pred, self._labels, message)
        self._matrix += _count_pairs(rows, columns, self._labels.size)

    def reset(self) -> None:
        """Clear the counts. A fixed label order passed to the constructor is kept."""
        self._labels: NDArray | None = None if self.labels is None else self.labels.copy()
        size = 0 if self._labels is None else self._labels.size
        self._matrix = np.zeros((size, size), dtype=np.int64)

    def result(self) -> NDArray[np.int64]:
        """Return a copy of the matrix so far, shape ``(k, k)``."""
        return self._matrix.copy()

    @property
    def labels_(self) -> NDArray:
        """Labels for the matrix rows and columns, in order."""
        if self._labels is None:
            return np.array([])
        return self._labels.copy()


_AVERAGES = (None, "binary", "micro", "macro", "weighted")


class _ConfusionCountsMetric(StreamingMetric):
    """Shared base for streaming precision, recall and F1.

    All three are worked out from a running confusion matrix.

    Parameters
    ----------
    positive_label : object, default=1
        Positive class used when ``average`` is None or ``"binary"``.
    zero_division : {0, 1}, default=0
        Value used when a ratio has a zero denominator.
    average : {None, "binary", "micro", "macro", "weighted"}, default=None
        None or ``"binary"`` scores only ``positive_label``, the same as
        the batch functions. ``"micro"`` adds up the counts over all classes
        first. ``"macro"`` is the plain mean of the per-class scores, and
        ``"weighted"`` weights each class by its support.

    Raises
    ------
    ValueError
        If ``zero_division`` is not 0 or 1, or ``average`` is not one of
        the options above.
    """

    def __init__(
        self,
        positive_label: object = 1,
        zero_division: int = 0,
        average: str | None = None,
    ) -> None:
        _safe_divide(0, 0, zero_division)  # only called to validate zero_division
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}.")
        self.positive_label = positive_label
        self.zero_division = zero_division
        self.average = average
        self._confusion = ConfusionMatrix()

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Add the chunk to the running confusion matrix (see ConfusionMatrix)."""
        self._confusion.update(y_true, y_pred)

    def reset(self) -> None:
        """Clear the running confusion matrix."""
        self._confusion.reset()

    @abstractmethod
    def _score(
        self, tp: NDArray, fp: NDArray, fn: NDArray
    ) -> NDArray[np.float64]:
        """Return a score per class from its true positive, false positive and false negative counts."""

    def _pooled_score(self, tp: NDArray, fp: NDArray, fn: NDArray) -> float:
        """Add up the counts over the given classes, then score the totals."""
        pooled = [np.array([counts.sum()]) for counts in (tp, fp, fn)]
        return float(self._score(*pooled)[0])

    def result(self) -> float:
        """Return the score over all chunks, using the chosen ``average``."""
        matrix = self._confusion.result()
        tp = np.diag(matrix)
        fp = matrix.sum(axis=0) - tp
        fn = matrix.sum(axis=1) - tp

        if self.average in (None, "binary"):
            # If the positive class has never appeared, nothing is selected, the
            # counts are all 0 and the result falls back to zero_division.
            selected = self._confusion.labels_ == self.positive_label
            return self._pooled_score(tp[selected], fp[selected], fn[selected])
        if self.average == "micro":
            return self._pooled_score(tp, fp, fn)
        if tp.size == 0:
            return float(self.zero_division)
        per_class = self._score(tp, fp, fn)
        if self.average == "macro":
            return float(np.mean(per_class))
        support = tp + fn
        if support.sum() == 0:
            return float(self.zero_division)
        return float(np.average(per_class, weights=support))


class Precision(_ConfusionCountsMetric):
    """Streaming precision, ``tp / (tp + fp)``. Parameters are as in the base class."""

    def _score(self, tp: NDArray, fp: NDArray, fn: NDArray) -> NDArray[np.float64]:
        return _divide_or_default(tp, tp + fp, self.zero_division)


class Recall(_ConfusionCountsMetric):
    """Streaming recall, ``tp / (tp + fn)``. Parameters are as in the base class."""

    def _score(self, tp: NDArray, fp: NDArray, fn: NDArray) -> NDArray[np.float64]:
        return _divide_or_default(tp, tp + fn, self.zero_division)


class F1(_ConfusionCountsMetric):
    """Streaming F1, the harmonic mean of precision and recall.

    ``zero_division`` is applied the same way as in the batch :func:`f1`.
    Parameters are as in the base class.
    """

    def _score(self, tp: NDArray, fp: NDArray, fn: NDArray) -> NDArray[np.float64]:
        p = _divide_or_default(tp, tp + fp, self.zero_division)
        r = _divide_or_default(tp, tp + fn, self.zero_division)
        return _divide_or_default(2 * p * r, p + r, self.zero_division)


class AUC(StreamingMetric):
    """Streaming ROC AUC from true labels and positive-class scores.

    Parameters
    ----------
    positive_label : object, default=1
        Label treated as the positive class.
    num_thresholds : int or None, default=None
        If None, every score is stored and the exact AUC is returned, which
        needs O(N) memory. If an integer, scores in [0, 1] are put into that
        many equal-width bins and only the positive and negative count per
        bin is kept. That needs O(num_thresholds) memory and O(n) per
        update, and the result is the exact AUC of the binned scores.

    Raises
    ------
    ValueError
        If ``num_thresholds`` is given but is not a positive integer.

    Notes
    -----
    ``result()`` returns NaN until both classes have been seen, because
    AUC is undefined with only one class.
    """

    requires_scores = True

    def __init__(self, positive_label: object = 1, num_thresholds: int | None = None) -> None:
        check_positive_int(num_thresholds, "num_thresholds", allow_none=True)
        self.positive_label = positive_label
        self.num_thresholds = num_thresholds
        self.reset()

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Add one chunk of labels and positive-class scores (passed as ``y_pred``).

        Raises
        ------
        TypeError
            If the scores are not real numeric.
        ValueError
            If the arrays are invalid, or binning is on and a score is
            outside [0, 1].
        """
        true, scores = _validate_scores(y_true, y_pred)
        if self.num_thresholds is None:
            # Copy so later changes to the caller's arrays don't alter what we
            # stored. Joining the chunks waits until result(), because joining
            # on every update would copy O(N^2) data in total.
            self._true_chunks.append(true.copy())
            self._score_chunks.append(scores.copy())
            return
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("binned AUC requires scores in [0, 1].")
        bins = np.minimum((scores * self.num_thresholds).astype(np.intp), self.num_thresholds - 1)
        positive = true == self.positive_label
        self._positive_counts += np.bincount(bins[positive], minlength=self.num_thresholds)
        self._negative_counts += np.bincount(bins[~positive], minlength=self.num_thresholds)

    def reset(self) -> None:
        """Clear the stored labels, scores and bin counts."""
        self._true_chunks: list[NDArray] = []
        self._score_chunks: list[NDArray[np.float64]] = []
        size = self.num_thresholds or 0
        self._positive_counts = np.zeros(size, dtype=np.int64)
        self._negative_counts = np.zeros(size, dtype=np.int64)

    def result(self) -> float:
        """Return ROC AUC over all chunks, or NaN if both classes have not been seen yet."""
        if self.num_thresholds is None:
            if not self._true_chunks:
                return float("nan")
            true = np.concatenate(self._true_chunks)
            if np.unique(true).size < 2:
                return float("nan")
            scores = np.concatenate(self._score_chunks)
            return roc_auc_score(true, scores, positive_label=self.positive_label)

        n_positive = self._positive_counts.sum()
        n_negative = self._negative_counts.sum()
        if n_positive == 0 or n_negative == 0:
            return float("nan")
        # Lower the threshold one bin at a time from the top. Scores in the
        # same bin act as ties.
        tpr = np.r_[0.0, np.cumsum(self._positive_counts[::-1]) / n_positive]
        fpr = np.r_[0.0, np.cumsum(self._negative_counts[::-1]) / n_negative]
        return float(np.trapezoid(tpr, fpr))


class RollingWindowMetric(StreamingMetric):
    """Compute a streaming metric over only the most recent chunks.

    Parameters
    ----------
    metric : StreamingMetric
        Metric to compute. This wrapper resets and refills it, so don't
        update it directly.
    window_size : int, default=10
        How many of the latest chunks :meth:`result` covers.

    Raises
    ------
    ValueError
        If ``window_size`` is not a positive integer.
    TypeError
        If ``metric`` is not a StreamingMetric.

    Notes
    -----
    Chunks are kept in a deque with a maximum length, so memory is
    O(window_size * chunk). ``update`` costs O(n_chunk): the chunk is first
    checked on a throwaway copy of the metric, so bad input never gets into
    the window. ``result`` only recomputes when the window has changed, at
    O(samples in the window). Because it replays the raw chunks, this works
    for any streaming metric, including ones like AUC that cannot be
    updated by simple subtraction.
    """

    def __init__(self, metric: StreamingMetric, window_size: int = 10) -> None:
        check_positive_int(window_size, "window_size")
        if not isinstance(metric, StreamingMetric):
            raise TypeError("metric must be a StreamingMetric instance.")
        self.metric = metric
        self.window_size = window_size
        self.requires_scores = metric.requires_scores
        self._template = deepcopy(metric)
        self._template.reset()
        self._chunks: deque[tuple[NDArray, NDArray]] = deque(maxlen=window_size)
        self._stale = False
        self.metric.reset()

    def update(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> None:
        """Check and store one chunk, dropping the oldest if the window is full."""
        true = np.array(y_true, copy=True)
        pred = np.array(y_pred, copy=True)
        deepcopy(self._template).update(true, pred)  # raises if the chunk is invalid
        self._chunks.append((true, pred))
        self._stale = True

    def reset(self) -> None:
        """Empty the window and reset the wrapped metric."""
        self._chunks.clear()
        self.metric.reset()
        self._stale = False

    def result(self) -> Any:
        """Return the wrapped metric's value over the chunks in the window."""
        if self._stale:
            self.metric.reset()
            for true, pred in self._chunks:
                self.metric.update(true, pred)
            self._stale = False
        return self.metric.result()


# Older names, kept so existing code still imports.
AccuracyMetric = Accuracy
PrecisionMetric = Precision
RecallMetric = Recall
F1Metric = F1

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "confusion_matrix",
    "roc_curve",
    "auc",
    "roc_auc_score",
    "StreamingMetric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ConfusionMatrix",
    "AUC",
    "RollingWindowMetric",
    "AccuracyMetric",
    "PrecisionMetric",
    "RecallMetric",
    "F1Metric",
]
