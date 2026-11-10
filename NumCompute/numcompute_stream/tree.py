"""Depth-limited CART decision tree with buffered streaming retraining.

Algorithm
---------
The tree is grown greedily from the root. At each node we sort every
candidate feature once and score all of its split thresholds in a single
vectorised pass over cumulative class counts. For a node with ``n`` samples
and ``m`` candidate features, the split search costs ``O(m * n log n)``.
Nodes are grown from an explicit stack instead of by recursion, so a tree
with no depth limit cannot hit Python's recursion limit.

Streaming
---------
:meth:`DecisionTreeClassifier.partial_fit` does buffered retraining. Each
chunk is added to an internal buffer. Once at least ``buffer_size`` samples
are waiting, a new tree is trained on the buffered data and the buffer is
emptied. Between retrains, predictions come from the last trained tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .utils import check_positive_int, check_supervised_chunk

_CRITERIA = ("gini", "entropy")
_MAX_FEATURE_RULES = ("sqrt", "log2")
# Gains smaller than this come from floating-point rounding, not a real split.
_MIN_GAIN = 1e-12


@dataclass(slots=True)
class _Node:
    """A single tree node. It is a leaf when ``feature`` is None."""

    class_counts: npt.NDArray[np.int64]
    feature: int | None = None
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """Classification tree that splits on Gini or entropy impurity.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree, where the root has depth 0. With None,
        nodes keep splitting until they are pure or another stopping rule
        applies.
    min_samples_split : int, default=2
        Minimum number of samples a node needs before it may be split.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        How many features to sample (without replacement) at each node. An
        int is a count, a float in (0, 1] is a fraction of the features, and
        None uses every feature.
    buffer_size : int, default=1000
        Number of buffered samples that triggers a retrain in
        :meth:`partial_fit`.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used to score splits.
    random_state : int or None, default=None
        Seed for feature sampling. It only matters when ``max_features``
        picks fewer than all of the features.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        Sorted class labels from the most recent training run.
    n_features_in_ : int
        Number of features used by the most recent training run.
    retrain_count_ : int
        Number of times a tree has been trained, including retrains.
    depth_ : int
        Depth of the current tree.
    n_leaves_ : int
        Number of leaves in the current tree.

    Notes
    -----
    Prediction sends all samples down the tree together, splitting them
    with boolean masks at each node, so it costs ``O(n * depth)`` time.

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    >>> for X_chunk, y_chunk in stream:          # doctest: +SKIP
    ...     tree.partial_fit(X_chunk, y_chunk)
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | float | str | None = None,
        buffer_size: int = 1000,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        check_positive_int(max_depth, "max_depth", allow_none=True)
        check_positive_int(min_samples_split, "min_samples_split")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2.")
        self._validate_max_features(max_features)
        check_positive_int(buffer_size, "buffer_size")
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}. Received {criterion!r}.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._root: _Node | None = None
        self._buffer_X: list[npt.NDArray[np.float64]] = []
        self._buffer_y: list[npt.NDArray[Any]] = []
        self._buffer_samples = 0
        self.retrain_count_ = 0

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_max_features(max_features: object) -> None:
        if max_features is None or max_features in _MAX_FEATURE_RULES:
            return
        if isinstance(max_features, float):
            if not 0.0 < max_features <= 1.0:
                raise ValueError("max_features as a float must be in (0, 1].")
            return
        check_positive_int(max_features, "max_features", allow_none=True)

    def _n_candidate_features(self, n_features: int) -> int:
        """Turn ``max_features`` into an actual feature count for ``n_features``."""
        rule = self.max_features
        if rule is None:
            return n_features
        if rule == "sqrt":
            count = int(np.sqrt(n_features))
        elif rule == "log2":
            count = int(np.log2(n_features)) if n_features > 1 else 1
        elif isinstance(rule, float):
            count = int(np.ceil(rule * n_features))
        else:
            count = int(rule)
        return int(np.clip(count, 1, n_features))

    @staticmethod
    def _as_feature_matrix(X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Check features and return them as a finite 2D float matrix."""
        X_arr = np.asarray(X)
        if X_arr.dtype.kind not in "biuf":
            raise TypeError(f"X must contain real numeric feature values. Received dtype {X_arr.dtype}.")
        X_arr = X_arr.astype(np.float64, copy=False)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 1D or 2D array. Received shape {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X must not contain NaN or infinity.")
        return X_arr

    def _validate_xy(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
        X_checked, y_arr = check_supervised_chunk(X, y)
        X_arr = self._as_feature_matrix(X_checked)
        if y_arr.dtype.kind in "fc" and not np.all(np.isfinite(y_arr)):
            raise ValueError("y must not contain NaN or infinity.")
        return X_arr, y_arr

    def _check_fitted(self) -> _Node:
        if self._root is None:
            raise ValueError(
                "DecisionTreeClassifier has not been fit yet. Call 'fit', or "
                "'partial_fit' until buffer_size samples have been seen."
            )
        return self._root

    # ------------------------------------------------------------------ #
    # Split search
    # ------------------------------------------------------------------ #
    def _impurity(self, counts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Impurity of each row of class counts (classes on the last axis)."""
        totals = counts.sum(axis=-1, keepdims=True)
        proportions = counts / np.maximum(totals, 1)
        if self.criterion == "gini":
            return 1.0 - np.sum(proportions * proportions, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(proportions > 0, np.log2(proportions), 0.0)
        return -np.sum(proportions * log_p, axis=-1)

    def _best_threshold(
        self,
        values: npt.NDArray[np.float64],
        y_encoded: npt.NDArray[np.intp],
        n_classes: int,
        parent_impurity: float,
    ) -> tuple[float, float] | None:
        """Return ``(gain, threshold)`` for the best split on one feature.

        Returns None if the feature has no useful split. Sorting costs
        O(n log n) and scoring every boundary with cumulative one-hot counts
        costs O(n * k) for ``k`` classes.
        """
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        # We can only split between two neighbouring values that differ.
        boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
        if boundaries.size == 0:
            return None

        one_hot = np.eye(n_classes, dtype=np.int64)[y_encoded[order]]
        left_counts = np.cumsum(one_hot, axis=0)[boundaries]
        right_counts = one_hot.sum(axis=0) - left_counts
        left_sizes = boundaries + 1
        right_sizes = values.size - left_sizes
        weighted_child = (
            left_sizes * self._impurity(left_counts)
            + right_sizes * self._impurity(right_counts)
        ) / values.size
        gains = parent_impurity - weighted_child
        best = int(np.argmax(gains))
        if gains[best] <= _MIN_GAIN:
            return None

        low = sorted_values[boundaries[best]]
        high = sorted_values[boundaries[best] + 1]
        # A midpoint threshold generalises better than the left value. If
        # rounding pushes the midpoint up to ``high``, use the left value.
        midpoint = low + (high - low) / 2.0
        threshold = midpoint if low <= midpoint < high else low
        return float(gains[best]), float(threshold)

    def _find_split(
        self,
        X: npt.NDArray[np.float64],
        y_encoded: npt.NDArray[np.intp],
        node: _Node,
        depth: int,
    ) -> tuple[int, float] | None:
        """Pick the best ``(feature, threshold)``, or None if the node should be a leaf."""
        n_samples = y_encoded.size
        n_classes = node.class_counts.size
        if np.count_nonzero(node.class_counts) <= 1:
            return None
        if n_samples < self.min_samples_split:
            return None
        if self.max_depth is not None and depth >= self.max_depth:
            return None

        n_features = X.shape[1]
        n_candidates = self._n_candidate_features(n_features)
        if n_candidates < n_features:
            features = np.sort(self._rng.choice(n_features, n_candidates, replace=False))
        else:
            features = np.arange(n_features)

        parent_impurity = float(self._impurity(node.class_counts))
        best_gain = 0.0
        best_split: tuple[int, float] | None = None
        for feature in features:
            candidate = self._best_threshold(
                X[:, feature], y_encoded, n_classes, parent_impurity
            )
            # Strict ">" means ties go to the lowest-index feature, so results
            # are deterministic.
            if candidate is not None and candidate[0] > best_gain:
                best_gain = candidate[0]
                best_split = (int(feature), candidate[1])
        return best_split

    def _grow(
        self, X: npt.NDArray[np.float64], y_encoded: npt.NDArray[np.intp]
    ) -> _Node:
        """Grow the tree without recursion and record its depth and leaf count."""
        n_classes = self.classes_.size
        root = _Node(np.bincount(y_encoded, minlength=n_classes))
        stack: list[tuple[_Node, npt.NDArray[np.intp], int]] = [
            (root, np.arange(y_encoded.size), 0)
        ]
        self.depth_ = 0
        self.n_leaves_ = 0
        while stack:
            node, rows, depth = stack.pop()
            split = self._find_split(X[rows], y_encoded[rows], node, depth)
            if split is None:
                self.n_leaves_ += 1
                self.depth_ = max(self.depth_, depth)
                continue
            node.feature, node.threshold = split
            goes_left = X[rows, node.feature] <= node.threshold
            left_rows, right_rows = rows[goes_left], rows[~goes_left]
            node.left = _Node(np.bincount(y_encoded[left_rows], minlength=n_classes))
            node.right = _Node(np.bincount(y_encoded[right_rows], minlength=n_classes))
            stack.append((node.right, right_rows, depth + 1))
            stack.append((node.left, left_rows, depth + 1))
        return root

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> DecisionTreeClassifier:
        """Train a fresh tree on all supplied samples.

        Any samples waiting in the streaming buffer are dropped, because the
        new tree completely replaces the old one.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Finite numeric features. A 1D array is treated as a single feature.
        y : array-like of shape (n_samples,)
            Class labels of any sortable dtype.

        Returns
        -------
        DecisionTreeClassifier
            The fitted classifier (``self``).

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If the shapes do not match, the data is empty, or it contains
            NaN or infinity.

        Complexity
        ----------
        Time: O(m * n log n * depth) for n samples and m candidate features.
        Space: O(n * k) for k classes during split scoring, plus O(nodes).
        """
        X_arr, y_arr = self._validate_xy(X, y)
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        self.n_features_in_ = X_arr.shape[1]
        self._root = self._grow(X_arr, y_encoded.astype(np.intp))
        self.retrain_count_ += 1
        self._clear_buffer()
        return self

    def partial_fit(
        self, X_chunk: npt.ArrayLike, y_chunk: npt.ArrayLike
    ) -> DecisionTreeClassifier:
        """Buffer a chunk and retrain once ``buffer_size`` samples are waiting.

        The whole chunk goes into the buffer before the size check, so a
        retrain can use more than ``buffer_size`` samples. The buffer is
        emptied after each retrain.

        The number of features may grow over time, for example when an
        incremental one-hot encoder adds a column for a new category. If a
        chunk is wider than the chunks already buffered, the older rows are
        padded with zeros on the right. A chunk with fewer features than
        earlier data is rejected.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features) or (n_samples,)
            Finite numeric features for this chunk.
        y_chunk : array-like of shape (n_samples,)
            Labels for this chunk.

        Returns
        -------
        DecisionTreeClassifier
            ``self``, so calls can be chained.

        Raises
        ------
        TypeError, ValueError
            In the same cases as :meth:`fit`, or if the chunk has fewer
            features than earlier data.

        Complexity
        ----------
        Time: O(n_chunk) to buffer. A retrain, when triggered, costs the same
        as :meth:`fit` on the buffered samples.
        Space: O(buffer_size * n_features) for the buffer.
        """
        X_arr, y_arr = self._validate_xy(X_chunk, y_chunk)
        width = X_arr.shape[1]
        seen_width = max(
            getattr(self, "n_features_in_", 0),
            self._buffer_X[0].shape[1] if self._buffer_X else 0,
        )
        if width < seen_width:
            raise ValueError(
                f"X_chunk has {width} features but {seen_width} were seen previously."
            )
        if self._buffer_X and width > self._buffer_X[0].shape[1]:
            self._buffer_X = [
                np.pad(rows, ((0, 0), (0, width - rows.shape[1]))) for rows in self._buffer_X
            ]

        self._buffer_X.append(X_arr.copy())
        self._buffer_y.append(y_arr.copy())
        self._buffer_samples += y_arr.size
        if self._buffer_samples >= self.buffer_size:
            self.fit(np.concatenate(self._buffer_X), np.concatenate(self._buffer_y))
        return self

    def _route(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return the class counts of the leaf that each sample lands in."""
        root = self._check_fitted()
        X_arr = self._as_feature_matrix(X)
        if X_arr.shape[1] < self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, fewer than the "
                f"{self.n_features_in_} used to fit."
            )
        # Columns added after the last retrain did not exist when this tree
        # was trained, so it ignores them.
        X_arr = X_arr[:, : self.n_features_in_]

        leaf_counts = np.empty((X_arr.shape[0], self.classes_.size), dtype=np.int64)
        pending: list[tuple[_Node, npt.NDArray[np.intp]]] = [
            (root, np.arange(X_arr.shape[0]))
        ]
        while pending:
            node, rows = pending.pop()
            if rows.size == 0:
                continue
            if node.is_leaf:
                leaf_counts[rows] = node.class_counts
                continue
            assert node.left is not None and node.right is not None
            goes_left = X_arr[rows, node.feature] <= node.threshold
            pending.append((node.right, rows[~goes_left]))
            pending.append((node.left, rows[goes_left]))
        return leaf_counts

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return class probabilities based on the training class mix in each leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Finite numeric features.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_classes)
            Each row sums to 1. Columns are in the order of ``classes_``.

        Raises
        ------
        ValueError
            If the model has not been fitted or ``X`` has too few features.

        Complexity
        ----------
        Time: O(n * depth). Space: O(n * k).
        """
        counts = self._route(X)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict the majority class of the leaf each sample lands in.

        If classes tie, the smallest label wins.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Finite numeric features.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted labels with the dtype of ``classes_``.

        Raises
        ------
        ValueError
            If the model has not been fitted or ``X`` has too few features.

        Complexity
        ----------
        Time: O(n * depth). Space: O(n * k).
        """
        leaf_counts = self._route(X)  # raises here if the tree is not fitted yet
        return self.classes_[np.argmax(leaf_counts, axis=1)]

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        """Return the accuracy of the model on a labelled batch.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` are empty or do not match.
        """
        X_checked, y_arr = check_supervised_chunk(X, y)
        return float(np.mean(self.predict(X_checked) == y_arr))

    def _clear_buffer(self) -> None:
        self._buffer_X = []
        self._buffer_y = []
        self._buffer_samples = 0

    @property
    def buffered_samples(self) -> int:
        """Number of samples waiting in the buffer for the next retrain."""
        return self._buffer_samples

    @property
    def is_fitted(self) -> bool:
        """True once a trained tree is available for prediction."""
        return self._root is not None


# Shorter name used in the assignment brief, e.g. DecisionTree(max_depth=5, buffer_size=500).
DecisionTree = DecisionTreeClassifier
