"""Preprocessing transformers that can be fitted in one go or chunk by chunk.

Every transformer takes either a single feature, shape ``(n_samples,)``, or
a feature matrix, shape ``(n_samples, n_features)``, and has the same
methods::

    fit(X) -> self            # learn from scratch
    partial_fit(X) -> self    # merge one more chunk into the learned state
    transform(X) -> X_out
    fit_transform(X) -> X_out

Each column of a matrix is a separate feature with its own learned state.
The scalers handle all columns at once with vectorised NumPy code. The
imputer and encoder keep one fitted copy per column. Calling
``partial_fit`` on an unfitted transformer does the same as ``fit``. Empty
chunks are ignored by ``partial_fit``, so a stream can contain empty
batches.
"""

from copy import deepcopy
from typing import Any, Sequence, Tuple

import numpy as np
import numpy.typing as npt

_EPS = float(np.finfo(np.float64).eps)


def _check_1d_or_2d(X: npt.ArrayLike) -> npt.NDArray[Any]:
    """Return ``X`` as a 1D or 2D array, or raise ValueError naming the bad shape."""
    X_arr = np.asarray(X)
    if X_arr.ndim not in (1, 2):
        raise ValueError(f"X must be a 1D or 2D array. Received shape {X_arr.shape}.")
    return X_arr


def _check_finite_numeric(X_arr: npt.NDArray[Any], owner: str) -> None:
    """Raise TypeError for non-numeric ``X_arr`` and ValueError for NaN or inf."""
    if X_arr.dtype.kind not in "biuf":
        raise TypeError(f"{owner} requires real numeric values. Received dtype {X_arr.dtype}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError(f"{owner} input must not contain NaN or infinity.")


def _check_is_fitted(estimator: Any, attributes: Sequence[str]) -> None:
    """Raise ValueError if any of the learned ``attributes`` is not set yet."""
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(
                f"This {estimator.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' or 'partial_fit' first."
            )


def _unfitted_clone(transformer: object) -> Any:
    """Return a deep copy of a transformer with its learned state removed.

    Learned attributes end in an underscore (``mean_``, ``statistics_`` and
    so on), so deleting them leaves an unfitted transformer with the same
    settings.
    """
    clone = deepcopy(transformer)
    for attribute in list(vars(clone)):
        if attribute.endswith("_") and not attribute.startswith("__"):
            delattr(clone, attribute)
    return clone


def _update_transformer(transformer: Any, X: npt.ArrayLike, first_call: bool) -> None:
    """Call ``partial_fit`` if available, otherwise ``fit`` on the first call only."""
    partial_fit = getattr(transformer, "partial_fit", None)
    if callable(partial_fit):
        partial_fit(X)
    elif first_call:
        transformer.fit(X)


class _ColumnwiseStep:
    """Apply a 1D transformer to each column of a matrix, one copy per column.

    ``layout`` is a list of ``(feature, start, stop)`` slices that says
    which part of each feature's output goes where in the combined output.
    If a feature's output gets wider while streaming (for example an
    encoder learns a new category), only the new slice is added at the end.
    This way existing output columns never change meaning.
    """

    def __init__(self, prototype: Any, reuse_prototype: bool = True) -> None:
        self.prototype = prototype
        self.reuse_prototype = reuse_prototype
        self.transformers: list[Any] = []
        self.widths: list[int] = []
        self.layout: list[tuple[int, int, int]] = []

    def _new_transformer(self, feature: int) -> Any:
        # Reuse the caller's object for column 0 so they can inspect its learned state.
        if feature == 0 and self.reuse_prototype:
            return self.prototype
        return _unfitted_clone(self.prototype)

    @staticmethod
    def _as_2d(output: npt.ArrayLike, n_samples: int) -> npt.NDArray[np.generic]:
        result = np.asarray(output)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        if result.ndim != 2 or result.shape[0] != n_samples:
            raise ValueError("transformer outputs must preserve the sample count.")
        return result

    def _feature_outputs(self, values: npt.NDArray[Any]) -> list[npt.NDArray[np.generic]]:
        if values.shape[1] != len(self.transformers):
            raise ValueError(
                "input feature count differs from the data used to fit: "
                f"received {values.shape[1]}, expected {len(self.transformers)}."
            )
        return [
            self._as_2d(transformer.transform(values[:, feature]), values.shape[0])
            for feature, transformer in enumerate(self.transformers)
        ]

    def _assemble(self, outputs: list[npt.NDArray[np.generic]]) -> npt.NDArray[np.generic]:
        return np.concatenate(
            [outputs[feature][:, start:stop] for feature, start, stop in self.layout], axis=1
        )

    def transform(self, values: npt.NDArray[Any]) -> npt.NDArray[np.generic]:
        return self._assemble(self._feature_outputs(values))

    def fit_transform(self, values: npt.NDArray[Any]) -> npt.NDArray[np.generic]:
        self.transformers = []
        for feature in range(values.shape[1]):
            transformer = self._new_transformer(feature)
            transformer.fit(values[:, feature])
            self.transformers.append(transformer)
        outputs = self._feature_outputs(values)
        self.widths = [output.shape[1] for output in outputs]
        self.layout = [(feature, 0, width) for feature, width in enumerate(self.widths)]
        return self._assemble(outputs)

    def partial_fit_transform(self, values: npt.NDArray[Any]) -> npt.NDArray[np.generic]:
        n_known = len(self.transformers)
        if values.shape[1] < n_known:
            raise ValueError(
                f"input has {values.shape[1]} features, fewer than the {n_known} seen before."
            )
        for feature in range(values.shape[1]):
            if feature >= n_known:
                self.transformers.append(self._new_transformer(feature))
                self.widths.append(0)
            _update_transformer(self.transformers[feature], values[:, feature], True)

        outputs = self._feature_outputs(values)
        for feature, output in enumerate(outputs):
            old_width, new_width = self.widths[feature], output.shape[1]
            if new_width < old_width:
                raise ValueError("a transformer's output width must not shrink.")
            if new_width > old_width:
                self.layout.append((feature, old_width, new_width))
                self.widths[feature] = new_width
        return self._assemble(outputs)


class _PerColumnMixin:
    """Let a 1D transformer accept a 2D matrix by fitting one copy per column.

    Subclasses call :meth:`_matrix_fit` and :meth:`_matrix_transform` when
    they get 2D input. They also implement :meth:`_summarise_columns`, which
    copies the per-column learned values onto the main object.
    """

    columns_: _ColumnwiseStep | None

    def _matrix_fit(self, X_arr: npt.NDArray[Any], incremental: bool) -> Any:
        if X_arr.shape[0] == 0:
            if incremental:
                return self
            raise ValueError(f"{type(self).__name__} cannot fit on an empty array.")
        existing = getattr(self, "columns_", None)
        if not incremental or existing is None:
            self.columns_ = _ColumnwiseStep(_unfitted_clone(self), reuse_prototype=False)
            self.columns_.fit_transform(X_arr)
        else:
            self.columns_.partial_fit_transform(X_arr)
        self._summarise_columns(self.columns_.transformers)
        return self

    def _matrix_transform(self, X_arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if getattr(self, "columns_", None) is None:
            raise ValueError(
                f"This {type(self).__name__} was fitted on 1D data; pass a 1D array "
                "or refit on a 2D matrix."
            )
        assert self.columns_ is not None
        return self.columns_.transform(X_arr)

    def _summarise_columns(self, transformers: list[Any]) -> None:
        raise NotImplementedError

    def _reject_1d_after_matrix_fit(self) -> None:
        if getattr(self, "columns_", None) is not None:
            raise ValueError(
                f"This {type(self).__name__} was fitted on a 2D matrix; pass a 2D array."
            )


# --------------------------------------------------------------------------- #
# Scalers (vectorised across columns)
# --------------------------------------------------------------------------- #
class _VectorisedScaler:
    """Shape handling shared by the scalers.

    Internally every statistic is an array of shape ``(n_features,)``.
    Public attributes are floats for 1D input, to stay backwards compatible,
    and arrays for 2D input.
    """

    _owner = "Scaler"

    def _as_features(self, X: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], bool]:
        X_arr = _check_1d_or_2d(X)
        _check_finite_numeric(X_arr, self._owner)
        values = X_arr.astype(np.float64, copy=False)
        is_1d = values.ndim == 1
        return (values.reshape(-1, 1) if is_1d else values), is_1d

    def _check_width(self, values: npt.NDArray[np.float64]) -> None:
        expected = getattr(self, "n_features_in_", values.shape[1])
        if values.shape[1] != expected:
            raise ValueError(
                f"X has {values.shape[1]} features, but {self._owner} was fitted "
                f"with {expected}."
            )

    def _publish(self, name: str, values: npt.NDArray[np.float64]) -> None:
        setattr(self, name, float(values[0]) if self._is_1d else values.copy())

    def _shape_output(self, values: npt.NDArray[np.float64], is_1d: bool) -> npt.NDArray[np.float64]:
        return values[:, 0] if is_1d else values


def _constant_features(
    variance: npt.NDArray[np.float64], mean: npt.NDArray[np.float64], n_samples: int
) -> npt.NDArray[np.bool_]:
    """Flag features whose variance could be pure rounding error.

    Floating-point error alone can produce a computed variance up to about
    ``n*eps*var + (n*|mean|*eps)**2``, so anything at or below that bound
    is treated as constant. The bound is relative to the data, unlike a
    fixed additive epsilon. That means a feature with a real but tiny
    spread (std 1e-12 around 0) is still scaled to unit variance.
    """
    bound = n_samples * _EPS * variance + (n_samples * np.abs(mean) * _EPS) ** 2
    return variance <= bound


class StandardScaler(_VectorisedScaler):
    """Standardise features to zero mean and unit (population) variance.

    Parameters
    ----------
    epsilon : float, default=0.0
        Value added to the scale in :meth:`transform`. With the default of
        0, every non-constant feature is scaled exactly. Constant features
        are detected separately (see Notes) and get a scale of 1.
    ema_alpha : float in (0, 1] or None, default=None
        How :meth:`partial_fit` combines chunks. ``None`` merges them
        exactly, with every sample weighted equally. A float switches to an
        exponential moving average where each new chunk gets weight
        ``ema_alpha``, so the scaler can follow data that drifts over time.

    Raises
    ------
    ValueError
        If ``epsilon`` is negative or not finite, or ``ema_alpha`` is
        outside (0, 1].

    Attributes
    ----------
    mean_, var_, scale_ : float or numpy.ndarray of shape (n_features,)
        Learned mean, population variance and scale. The scale is
        ``sqrt(var_)``, or 1 for constant features. Floats for 1D input.
    n_samples_seen_ : int
        Number of samples merged so far.
    n_features_in_ : int
        Number of features (1 for 1D input).

    Notes
    -----
    Exact streaming uses the parallel form of Welford's algorithm (Chan et
    al.). Each chunk's mean and variance are merged into the running values
    with

    ``M2 = M2_a + M2_b + delta**2 * n_a * n_b / (n_a + n_b)``

    This is numerically stable and uses O(n_features) memory however long
    the stream is. A feature only counts as constant when its variance is
    within rounding error of zero, so small but real variation is still
    scaled properly.
    """

    _owner = "StandardScaler"

    def __init__(self, epsilon: float = 0.0, ema_alpha: float | None = None) -> None:
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative finite number.")
        if ema_alpha is not None and not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1] or None.")
        self.epsilon = epsilon
        self.ema_alpha = ema_alpha

    def _set_statistics(
        self, mean: npt.NDArray[np.float64], var: npt.NDArray[np.float64], n_samples: int
    ) -> None:
        self._mean = mean
        self._var = np.maximum(var, 0.0)
        self.n_samples_seen_ = int(n_samples)
        self.n_features_in_ = int(mean.size)
        self._scale = np.where(
            _constant_features(self._var, self._mean, n_samples), 1.0, np.sqrt(self._var)
        )
        self._publish("mean_", self._mean)
        self._publish("var_", self._var)
        self._publish("scale_", self._scale)

    def fit(self, X: npt.ArrayLike) -> "StandardScaler":
        """Learn the mean and variance of each feature from a non-empty array.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Finite numeric values.

        Returns
        -------
        StandardScaler
            The fitted scaler.

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 1D/2D, is empty, or contains NaN/infinity.

        Complexity
        ----------
        Time: O(n_samples * n_features)
        Space: O(n_features) beyond the input.
        """
        values, self._is_1d = self._as_features(X)
        if values.shape[0] == 0:
            raise ValueError("StandardScaler cannot fit on an empty array.")
        self.__dict__.pop("n_features_in_", None)  # a refit may use a different width
        self._set_statistics(values.mean(axis=0), values.var(axis=0), values.shape[0])
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "StandardScaler":
        """Merge one chunk into the running mean and variance of each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Finite numeric values. An empty chunk changes nothing.

        Returns
        -------
        StandardScaler
            The updated scaler.

        Raises
        ------
        TypeError, ValueError
            Same as :meth:`fit`, or if the number of features changes.

        Complexity
        ----------
        Time: O(n_chunk * n_features)
        Space: O(n_features). Past samples are not stored.
        """
        values, _ = self._as_features(X)
        if values.shape[0] == 0:
            return self
        if getattr(self, "n_samples_seen_", 0) == 0:
            return self.fit(X)
        self._check_width(values)

        n_old, n_new = self.n_samples_seen_, values.shape[0]
        weight_new = n_new / (n_old + n_new) if self.ema_alpha is None else self.ema_alpha
        weight_old = 1.0 - weight_new
        delta = values.mean(axis=0) - self._mean
        mean = self._mean + weight_new * delta
        # Variance of a mix of two groups = weighted within-group variance
        # plus the between-group term. With exact weights this is Chan's M2
        # merge divided by n.
        var = weight_old * self._var + weight_new * values.var(axis=0)
        var = var + weight_old * weight_new * delta**2
        self._set_statistics(mean, var, n_old + n_new)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Standardise ``X`` using the learned mean and scale.

        Returns
        -------
        numpy.ndarray
            Same shape as ``X``: ``(X - mean_) / (scale_ + epsilon)``.

        Raises
        ------
        ValueError
            If not fitted, ``X`` is not 1D or 2D, or the number of features
            differs from fit.

        Complexity
        ----------
        Time: O(n_samples * n_features). Space: same.
        """
        _check_is_fitted(self, ["mean_", "scale_"])
        values, is_1d = self._as_features(X)
        self._check_width(values)
        scaled = (values - self._mean) / (self._scale + self.epsilon)
        return self._shape_output(scaled, is_1d)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Fit on ``X`` and return it standardised (see :meth:`fit`)."""
        return self.fit(X).transform(X)


class MinMaxScaler(_VectorisedScaler):
    """Scale each feature linearly so its observed range maps onto ``feature_range``.

    Parameters
    ----------
    feature_range : tuple of float, default=(0.0, 1.0)
        Target ``(min, max)``. Both must be finite with ``min < max``.
    clip : bool, default=False
        Clip transformed values to ``feature_range``. Useful when streaming,
        because later data can fall outside the min and max seen so far.

    Raises
    ------
    ValueError
        If ``feature_range`` is not a finite ``(min, max)`` pair with
        ``min < max``.

    Attributes
    ----------
    data_min_, data_max_ : float or numpy.ndarray of shape (n_features,)
        Smallest and largest values seen so far.
    data_range_ : float or numpy.ndarray
        ``data_max_ - data_min_``, or 1.0 for a constant feature. This can
        be ``inf`` if the data spans more than the float64 range, but the
        transform still gives finite values.

    Notes
    -----
    Streaming only stores the running min and max, so memory is
    O(n_features). A constant feature maps to the low end of
    ``feature_range``.
    """

    _owner = "MinMaxScaler"

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), clip: bool = False) -> None:
        if len(feature_range) != 2:
            raise ValueError(f"feature_range must be a (min, max) pair. Received {feature_range!r}.")
        low, high = (float(bound) for bound in feature_range)
        if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
            raise ValueError(
                f"feature_range must be finite with min < max. Received {feature_range!r}."
            )
        self.feature_range = feature_range
        self.clip = clip

    def _set_extrema(self, data_min: npt.NDArray[np.float64], data_max: npt.NDArray[np.float64]) -> None:
        self._min, self._max = data_min, data_max
        self.n_features_in_ = int(data_min.size)
        with np.errstate(over="ignore"):
            span = data_max - data_min
        # Use 1.0 for constant features so transform does not divide by zero.
        self._range = np.where(span == 0, 1.0, span)
        self._publish("data_min_", self._min)
        self._publish("data_max_", self._max)
        self._publish("data_range_", self._range)

    def fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        """Learn the minimum and maximum of each feature from a non-empty array.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        MinMaxScaler
            The fitted scaler.

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 1D/2D, is empty, or contains NaN/infinity.

        Complexity
        ----------
        Time: O(n_samples * n_features). Space: O(n_features).
        """
        values, self._is_1d = self._as_features(X)
        if values.shape[0] == 0:
            raise ValueError("MinMaxScaler cannot fit on an empty array.")
        self.__dict__.pop("n_features_in_", None)
        self._set_extrema(values.min(axis=0), values.max(axis=0))
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        """Widen the learned min and max to cover one more chunk.

        Empty chunks are ignored. Raises in the same cases as :meth:`fit`,
        or if the number of features changes.

        Complexity
        ----------
        Time: O(n_chunk * n_features). Space: O(n_features).
        """
        values, _ = self._as_features(X)
        if values.shape[0] == 0:
            return self
        if not hasattr(self, "data_min_"):
            return self.fit(X)
        self._check_width(values)
        self._set_extrema(
            np.minimum(self._min, values.min(axis=0)), np.maximum(self._max, values.max(axis=0))
        )
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Scale ``X`` using the learned min and max.

        Returns
        -------
        numpy.ndarray
            Same shape as ``X``.

        Raises
        ------
        ValueError
            If not fitted, ``X`` is not 1D or 2D, or the number of features
            differs from fit.

        Complexity
        ----------
        Time: O(n_samples * n_features). Space: same.
        """
        _check_is_fitted(self, ["data_min_", "data_range_"])
        values, is_1d = self._as_features(X)
        self._check_width(values)
        if np.all(np.isfinite(self._range)):
            unit = (values - self._min) / self._range
        else:
            # The range overflowed to inf. Halving each term is exact in
            # floating point and brings max - min back into range.
            half_range = np.where(np.isfinite(self._range), self._range / 2, self._max / 2 - self._min / 2)
            unit = (values / 2 - self._min / 2) / half_range
        low, high = (float(bound) for bound in self.feature_range)
        scaled = unit * (high - low) + low
        if self.clip:
            scaled = np.clip(scaled, low, high)
        return self._shape_output(scaled, is_1d)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Fit on ``X`` and return its scaled values (see :meth:`fit`)."""
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Imputer and encoder (one fitted copy per column for 2D input)
# --------------------------------------------------------------------------- #
def _text_missing_mask(X_arr: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """Return True where a string or object array holds a missing value.

    None, float NaN, and text that is empty or "nan" (any case, surrounding
    spaces ignored) count as missing. Byte strings follow the same text rule.
    The check uses NumPy's vectorised string functions instead of a Python
    loop over the elements.

    Complexity
    ----------
    Time: O(n * L) for n elements of length up to L. Space: O(n * L).
    """
    text = np.char.lower(np.char.strip(X_arr.astype(str)))
    missing = np.isin(text, ["", "nan"])
    if X_arr.dtype.kind == "O":
        # Object arrays can hold None itself, which becomes the text "None".
        missing |= np.equal(X_arr, None)
    return missing


class Imputer(_PerColumnMixin):
    """Fill in missing values with a statistic learned for each feature.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default="mean"
        Which statistic to fill with.
    fill_value : Any, default=0.0
        Value used for ``strategy="constant"``, and as a fallback before
        any non-missing value has been seen. If the data is strings and
        ``fill_value`` is not numeric, mean and median fall back to
        constant.
    missing_values : Any, default=np.nan
        Marker for missing entries. When it is NaN, string and object
        arrays also treat None, ``""`` and ``"nan"`` (any case) as missing.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the three options.

    Attributes
    ----------
    statistics_ : Any or numpy.ndarray of shape (n_features,)
        Current fill value, one per column for 2D input.
    n_samples_seen_ : int or numpy.ndarray
        Number of non-missing numeric values merged so far.

    Notes
    -----
    When streaming, the mean only needs a count and a running mean, so
    memory is O(1). An exact median has to keep every observed value, so
    memory is O(n).
    """

    strategy: str
    fill_value: Any
    missing_values: Any
    statistics_: Any

    _STRATEGIES = ("mean", "median", "constant")

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Any = 0.0,
        missing_values: Any = np.nan,
    ) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError("Strategy must be 'mean', 'median', or 'constant'")
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

    def _summarise_columns(self, transformers: list[Any]) -> None:
        statistics = [column.statistics_ for column in transformers]
        numeric = all(isinstance(value, (int, float, np.number)) for value in statistics)
        self.statistics_ = np.array(statistics, dtype=float if numeric else object)
        self.n_samples_seen_ = np.array([column.n_samples_seen_ for column in transformers])

    def _get_missing_mask(self, X_arr: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        """Return a boolean mask that is True where ``X_arr`` has a missing value."""
        marker = self.missing_values
        if isinstance(marker, (float, np.floating)) and np.isnan(marker):
            if X_arr.dtype.kind in "biufc":
                return np.isnan(X_arr)
            return _text_missing_mask(X_arr)
        return X_arr == marker

    @staticmethod
    def _as_numeric(values: npt.NDArray[Any]) -> npt.NDArray[np.float64] | None:
        """Return ``values`` as floats, or None if they cannot be read as numbers."""
        if values.dtype.kind in "biuf":
            return values.astype(float)
        if values.dtype.kind == "O":
            # CSV readers often give object arrays that still hold plain numbers.
            try:
                return values.astype(float)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _stable_mean(values: npt.NDArray[np.float64]) -> float:
        """Return the mean, scaling first so large values cannot overflow the sum."""
        scale = float(np.max(np.abs(values)))
        if scale == 0.0:
            return 0.0
        return float(np.mean(values / scale) * scale)

    def _reset_state(self) -> None:
        self.statistics_ = self.fill_value
        self.n_samples_seen_ = 0
        self.columns_ = None
        self._observed_values_ = np.empty(0, dtype=float)
        self._effective_strategy_ = self.strategy

    def _update(self, X_arr: npt.NDArray[Any]) -> None:
        """Merge the non-missing values of one non-empty 1D chunk into the state."""
        observed = X_arr[~self._get_missing_mask(X_arr)]
        numeric = self._as_numeric(observed) if observed.size else np.empty(0)
        if numeric is not None and not np.all(np.isfinite(numeric)):
            raise ValueError("Imputer input must not contain infinite observed values.")

        if numeric is None and self._effective_strategy_ in ("mean", "median"):
            if isinstance(self.fill_value, (int, float, np.number)):
                raise ValueError(
                    f"Cannot compute strategy='{self._effective_strategy_}' "
                    "on non-numeric string data."
                )
            # Mean and median make no sense for strings, so use the caller's
            # non-numeric fill_value instead.
            self._effective_strategy_ = "constant"

        if self._effective_strategy_ == "constant" or numeric is None or numeric.size == 0:
            if self._effective_strategy_ == "constant":
                self.statistics_ = self.fill_value
            return

        if self._effective_strategy_ == "mean":
            n_old, n_new = self.n_samples_seen_, int(numeric.size)
            chunk_mean = self._stable_mean(numeric)
            if n_old == 0:
                self.statistics_ = chunk_mean
            else:
                # Combine the two means by weight rather than summing values,
                # which could overflow.
                total = n_old + n_new
                self.statistics_ = self.statistics_ * (n_old / total) + chunk_mean * (
                    n_new / total
                )
        else:
            self._observed_values_ = np.concatenate((self._observed_values_, numeric))
            self.statistics_ = float(np.median(self._observed_values_))
        self.n_samples_seen_ += int(numeric.size)

    def fit(self, X: npt.ArrayLike) -> "Imputer":
        """Learn the fill value for each feature from a non-empty array.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Values that may contain ``missing_values``.

        Returns
        -------
        Imputer
            The fitted imputer.

        Raises
        ------
        ValueError
            If ``X`` is not 1D or 2D, is empty, has infinite non-missing
            values, or mean/median is asked for on string data with a
            numeric ``fill_value``.

        Complexity
        ----------
        Time: O(n) for mean/constant, O(n log n) for median (per column).
        Space: O(1) state for mean/constant, O(n) for median (per column).
        """
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_fit(X_arr, incremental=False)
        if X_arr.size == 0:
            raise ValueError("Imputer cannot fit on an empty array.")
        self._reset_state()
        self._update(X_arr)
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "Imputer":
        """Update the fill value for each feature with one more chunk.

        A chunk with only missing values leaves the statistic unchanged.
        Empty chunks are ignored. Raises in the same cases as :meth:`fit`.

        Complexity
        ----------
        Time: O(n_chunk) for mean, O(N log N) for the median over all N
        observed values. Space: O(1) for mean, O(N) for median.
        """
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_fit(X_arr, incremental=hasattr(self, "statistics_"))
        if X_arr.size == 0:
            return self
        if not hasattr(self, "statistics_"):
            return self.fit(X_arr)
        self._update(X_arr)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Replace missing entries with the learned fill value(s).

        Returns
        -------
        numpy.ndarray
            Same shape as ``X``.

        Raises
        ------
        ValueError
            If not fitted, ``X`` is not 1D or 2D, or it does not match the
            data used to fit (1D vs 2D, or a different number of columns).

        Complexity
        ----------
        Time: O(n_samples * n_features). Space: same.
        """
        _check_is_fitted(self, ["statistics_"])
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_transform(X_arr)
        self._reject_1d_after_matrix_fit()
        return np.where(self._get_missing_mask(X_arr), self.statistics_, X_arr)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Fit on ``X`` and return it with missing values filled (see :meth:`fit`)."""
        return self.fit(X).transform(X)


class OneHotEncoder(_PerColumnMixin):
    """One-hot encode categorical features. New categories can be added while streaming.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        What to do with categories that have not been learned yet.
        ``"ignore"`` encodes them as all-zero rows, which is useful when you
        predict on a stream chunk before training on it.

    Raises
    ------
    ValueError
        If ``handle_unknown`` is not ``"error"`` or ``"ignore"``.

    Attributes
    ----------
    unique_categories_ : numpy.ndarray or list of numpy.ndarray
        Learned categories in output column order. For 2D input this is a
        list with one array per input column.
    categories_ : int
        Total number of output columns.

    Notes
    -----
    :meth:`partial_fit` adds newly seen categories as new columns at the
    end of the output, so existing columns keep their meaning.
    """

    categories_: int
    unique_categories_: Any
    _category_type: str

    def __init__(self, handle_unknown: str = "error") -> None:
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        self.handle_unknown = handle_unknown

    def _summarise_columns(self, transformers: list[Any]) -> None:
        self.unique_categories_ = [column.unique_categories_ for column in transformers]
        self.categories_ = int(sum(column.categories_ for column in transformers))
        self._category_type = "per-column"

    def _check_category_dtype(self, X_arr: npt.NDArray[Any]) -> None:
        """Raise ValueError if a chunk's dtype does not match the learned categories."""
        if self._category_type == "integer":
            if not np.issubdtype(X_arr.dtype, np.integer):
                raise ValueError("OneHotEncoder expects integer categorical data.")
            if np.any(X_arr < 0):
                raise ValueError("Categories must be non-negative integers.")
        elif X_arr.dtype.kind not in "USO":
            raise ValueError("OneHotEncoder expects string categorical data.")

    def fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        """Learn the sorted list of categories in each column.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Non-negative integers or strings.

        Returns
        -------
        OneHotEncoder
            The fitted encoder.

        Raises
        ------
        ValueError
            If ``X`` is empty, not 1D or 2D, has an unsupported dtype, or
            contains negative integers.

        Complexity
        ----------
        Time: O(n log n) per column. Space: O(k) for k categories.
        """
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_fit(X_arr, incremental=False)
        if X_arr.size == 0:
            raise ValueError("OneHotEncoder requires at least one category.")
        if np.issubdtype(X_arr.dtype, np.integer):
            self._category_type = "integer"
        elif X_arr.dtype.kind in "USO":
            self._category_type = "string"
        else:
            raise ValueError("OneHotEncoder expects integer categorical data or strings.")
        self._check_category_dtype(X_arr)
        self.columns_ = None
        self.unique_categories_ = np.unique(X_arr)
        self.categories_ = int(self.unique_categories_.size)
        return self

    def partial_fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        """Add categories first seen in this chunk as new columns at the end.

        Empty chunks are ignored. Raises in the same cases as :meth:`fit`,
        or if the chunk's dtype does not match the learned category type.

        Complexity
        ----------
        Time: O((n + k) log(n + k)) per column. Space: O(k).
        """
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_fit(X_arr, incremental=hasattr(self, "unique_categories_"))
        if X_arr.size == 0:
            return self
        if not hasattr(self, "unique_categories_"):
            return self.fit(X_arr)
        self._check_category_dtype(X_arr)
        new_categories = np.unique(X_arr[~np.isin(X_arr, self.unique_categories_)])
        if new_categories.size:
            self.unique_categories_ = np.concatenate((self.unique_categories_, new_categories))
            self.categories_ = int(self.unique_categories_.size)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Return the one-hot matrix of shape ``(n_samples, categories_)``.

        Raises
        ------
        ValueError
            If not fitted, the dtype does not match the learned type, or an
            unknown category appears while ``handle_unknown="error"``.

        Complexity
        ----------
        Time: O(n * k). Space: O(n * k).
        """
        _check_is_fitted(self, ["categories_", "unique_categories_", "_category_type"])
        X_arr = _check_1d_or_2d(X)
        if X_arr.ndim == 2:
            return self._matrix_transform(X_arr)
        self._reject_1d_after_matrix_fit()
        if X_arr.size == 0:
            return np.empty((0, self.categories_), dtype=float)
        self._check_category_dtype(X_arr)

        matches = X_arr[:, None] == self.unique_categories_[None, :]
        if self.handle_unknown == "error" and not np.all(np.any(matches, axis=1)):
            raise ValueError("Found unknown categories during transform.")
        # Each row matches at most one category, so the boolean match
        # matrix is already the encoding. Unknown rows stay all zeros.
        return matches.astype(float)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Learn the categories and return ``X`` encoded (see :meth:`fit`)."""
        return self.fit(X).transform(X)


class ColumnSelector:
    """Pick a subset of columns from a 2D array, optionally casting their dtype.

    Use it as the first step of each ``FeatureUnion`` branch to send the
    numeric and categorical columns of one table through different
    preprocessing.

    Parameters
    ----------
    columns : sequence of int
        Column indices to keep, in output order.
    dtype : data-type or None, default=None
        Dtype to cast the selected columns to, for example ``float`` for
        numeric columns read from a mixed CSV (missing cells become NaN).

    Raises
    ------
    ValueError
        If ``columns`` is not a non-empty 1D sequence of integers.

    Complexity
    ----------
    Time: O(n_samples * len(columns)) per transform. Space: same.
    """

    def __init__(self, columns: Sequence[int], dtype: npt.DTypeLike | None = None) -> None:
        indices = np.asarray(columns)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError("columns must be a non-empty sequence of integer indices.")
        self.columns = indices
        self.dtype = dtype

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "ColumnSelector":
        """Check that ``X`` is 2D and has every selected column. Learns nothing."""
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array. Received shape {X_arr.shape}.")
        if np.any(self.columns >= X_arr.shape[1]) or np.any(self.columns < -X_arr.shape[1]):
            raise ValueError(f"columns {self.columns.tolist()} out of range for {X_arr.shape[1]}.")
        return self

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "ColumnSelector":
        """Same as :meth:`fit`, since there is no state to update."""
        return self.fit(X, y)

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Return ``X[:, columns]``, cast to ``dtype`` if one was given."""
        self.fit(X)
        selected = np.asarray(X)[:, self.columns]
        return selected if self.dtype is None else selected.astype(self.dtype)

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.NDArray[Any]:
        """Check ``X`` and return the selected columns in one call."""
        return self.transform(X)
