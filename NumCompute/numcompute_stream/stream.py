"""Train, evaluate and log streaming models one chunk at a time.

:class:`StreamTrainer` holds a model, an optional preprocessing pipeline and
a set of streaming metrics. ``fit_chunk`` updates the model with one chunk.
``score_chunk`` evaluates the model on a labelled chunk and adds a log
record with per-chunk and cumulative metrics, timings and memory use.
``run`` processes a whole stream with prequential (test-then-train)
evaluation, so each chunk is scored before the model learns from it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from time import perf_counter
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from .metrics import StreamingMetric
from .utils import check_supervised_chunk

_SKIPPED_TYPES = (ModuleType, type, FunctionType, BuiltinFunctionType, MethodType)


def deep_sizeof(obj: object) -> int:
    """Estimate the memory held by ``obj`` and everything it references.

    The walk covers containers, instance ``__dict__`` and ``__slots__``
    attributes, NumPy arrays and the elements of object arrays. It uses a
    stack instead of recursion, so a deep tree cannot hit the recursion
    limit, and each object is counted once. For array views we add ``nbytes``, because ``sys.getsizeof`` only
    reports the view's header. Modules, classes and functions are shared
    code, so they are not counted.

    Parameters
    ----------
    obj : object
        Object to measure.

    Returns
    -------
    int
        Estimated size in bytes.

    Complexity
    ----------
    Time and space: O(number of reachable objects).
    """
    seen: set[int] = set()
    stack: list[object] = [obj]
    total = 0
    while stack:
        value = stack.pop()
        if id(value) in seen or isinstance(value, _SKIPPED_TYPES):
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.base is not None:
                total += value.nbytes
            if value.dtype == object:
                stack.extend(value.ravel().tolist())
        elif isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
        else:
            if hasattr(value, "__dict__"):
                stack.append(vars(value))
            for slot in getattr(type(value), "__slots__", ()):
                if hasattr(value, slot):
                    stack.append(getattr(value, slot))
    return total


def _ends_with_estimator(component: object) -> bool:
    """Return True if ``component`` can predict (a Pipeline may only transform)."""
    if hasattr(component, "final_estimator"):
        return component.final_estimator is not None
    return hasattr(component, "predict")


class StreamTrainer:
    """Train a model chunk by chunk and log how it performs on the stream.

    Parameters
    ----------
    model : object or None
        Estimator with ``predict`` and ``partial_fit``. If it only has
        ``fit``, that is called on every chunk instead. It can also be a
        ``Pipeline``.
    pipeline : object or None, default=None
        Preprocessing applied before ``model``, such as a ``Pipeline`` made
        only of transformers. It is updated with ``partial_fit`` on each
        chunk. If ``model`` is None, ``pipeline`` must end with an estimator
        and is used as the model.
    metrics : mapping of str to StreamingMetric, optional
        Extra metrics to log, for example ``{"f1": F1(), "auc": AUC()}``.
        Each one is logged per chunk under ``<name>`` and over the whole
        stream under ``cumulative_<name>``. For a ``RollingWindowMetric``,
        ``cumulative_<name>`` holds the value over its window instead.
        Metrics with ``requires_scores`` are given positive-class
        probabilities from ``predict_proba``. Accuracy is always logged.
    logger : callable or logging.Logger, optional
        Gets every record, through ``logger.info`` if it has one and
        ``logger(record)`` otherwise.
    track_memory : bool, default=True
        Whether to record the memory used by the model in each record.

    Attributes
    ----------
    history : list of dict
        One record per :meth:`score_chunk` call.

    Examples
    --------
    >>> trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=500),
    ...                         metrics={"f1": F1()})       # doctest: +SKIP
    >>> trainer.run(stream)                                  # doctest: +SKIP
    >>> trainer.metric_history("cumulative_accuracy")        # doctest: +SKIP
    """

    def __init__(
        self,
        model: Any | None = None,
        pipeline: Any | None = None,
        *,
        metrics: Mapping[str, StreamingMetric] | None = None,
        logger: Any | None = None,
        track_memory: bool = True,
    ) -> None:
        if model is None and pipeline is None:
            raise ValueError("A model or pipeline must be provided.")
        if model is None:
            model, pipeline = pipeline, None
        elif pipeline is not None and _ends_with_estimator(pipeline):
            raise ValueError(
                "pipeline must contain only preprocessing steps when a model is also given."
            )
        if not _ends_with_estimator(model):
            raise TypeError("model must implement predict.")
        if not hasattr(model, "partial_fit") and not hasattr(model, "fit"):
            raise TypeError("model must implement partial_fit or fit.")
        if logger is not None and not (hasattr(logger, "info") or callable(logger)):
            raise TypeError("logger must be callable or expose an info method.")

        self.model = model
        self.pipeline = pipeline
        self.metrics: dict[str, StreamingMetric] = dict(metrics or {})
        for name, metric in self.metrics.items():
            if not isinstance(metric, StreamingMetric):
                raise TypeError(f"metric {name!r} must be a StreamingMetric instance.")
        self.logger = logger
        self.track_memory = track_memory
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear the logs, counters and metric state. The model is not changed."""
        self.history: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._samples_trained = 0
        self._correct = 0
        self._scored_samples = 0
        self._last_fit_seconds = 0.0
        self._pipeline_updated = False
        for metric in self.metrics.values():
            metric.reset()
        self._chunk_templates = {name: deepcopy(metric) for name, metric in self.metrics.items()}

    def _preprocess(self, X: npt.ArrayLike) -> npt.ArrayLike:
        return X if self.pipeline is None else self.pipeline.transform(X)

    @property
    def is_ready(self) -> bool:
        """True if the model can make predictions yet."""
        fitted = getattr(self.model, "is_fitted", None)
        if fitted is not None:
            return bool(fitted)
        return self._chunk_index > 0

    def fit_chunk(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Any:
        """Update the pipeline and model with one labelled chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
        y : array-like of shape (n_samples,)

        Returns
        -------
        object
            Whatever the model's ``partial_fit`` or ``fit`` returns, usually
            the model itself.

        Raises
        ------
        ValueError
            If the chunk is empty or ``X`` and ``y`` do not match.
        """
        X_arr, y_arr = check_supervised_chunk(X, y)
        started = perf_counter()
        if self.pipeline is not None:
            update = getattr(self.pipeline, "partial_fit", None)
            if callable(update):
                update(X_arr)
            elif not self._pipeline_updated:
                self.pipeline.fit(X_arr)
            self._pipeline_updated = True
        features = self._preprocess(X_arr)
        if hasattr(self.model, "partial_fit"):
            result = self.model.partial_fit(features, y_arr)
        else:
            result = self.model.fit(features, y_arr)
        self._last_fit_seconds = perf_counter() - started
        self._chunk_index += 1
        self._samples_trained += int(y_arr.size)
        return result

    def _positive_scores(self, features: npt.ArrayLike, positive_label: object) -> np.ndarray:
        """Return P(positive_label) for each row, or zeros if that class was never seen."""
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("score-based metrics need a model with predict_proba.")
        probabilities = np.asarray(self.model.predict_proba(features))
        classes = np.asarray(self.model.classes_)
        column = np.flatnonzero(classes == positive_label)
        if column.size == 0:
            return np.zeros(probabilities.shape[0])
        return probabilities[:, column[0]]

    @staticmethod
    def _safe_result(metric: StreamingMetric) -> Any:
        """Return the metric's result, or NaN if it is undefined for the data so far."""
        try:
            return metric.result()
        except ValueError:
            return float("nan")

    def score_chunk(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        chunk_index: int | None = None,
    ) -> float:
        """Evaluate the current model on one labelled chunk and log the result.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
        y : array-like of shape (n_samples,)
        chunk_index : int, optional
            Chunk number stored in the record. Defaults to the number of
            chunks trained so far (at least 1).

        Returns
        -------
        float
            Accuracy on this chunk.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` do not match, ``chunk_index`` is not a
            positive integer, or the predictions have a different shape
            from ``y``.
        """
        X_arr, y_arr = check_supervised_chunk(X, y)
        if chunk_index is None:
            chunk_index = max(1, self._chunk_index)
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 1:
            raise ValueError("chunk_index must be a positive integer.")

        started = perf_counter()
        features = self._preprocess(X_arr)
        predictions = np.asarray(self.model.predict(features))
        predict_seconds = perf_counter() - started
        if predictions.shape != y_arr.shape:
            raise ValueError(
                "predictions and y must be matching 1D arrays. "
                f"Received {predictions.shape} and {y_arr.shape}."
            )

        correct = int(np.count_nonzero(predictions == y_arr))
        self._correct += correct
        self._scored_samples += int(y_arr.size)
        record: dict[str, Any] = {
            "chunk": chunk_index,
            "n_samples": int(y_arr.size),
            "accuracy": correct / y_arr.size,
            "chunk_accuracy": correct / y_arr.size,
            "cumulative_accuracy": self.cumulative_accuracy,
        }

        for name, metric in self.metrics.items():
            if metric.requires_scores:
                label = getattr(metric, "positive_label", 1)
                outputs = self._positive_scores(features, label)
            else:
                outputs = predictions
            chunk_metric = deepcopy(self._chunk_templates[name])
            chunk_metric.update(y_arr, outputs)
            metric.update(y_arr, outputs)
            record[name] = self._safe_result(chunk_metric)
            record[f"cumulative_{name}"] = self._safe_result(metric)

        record["samples_trained"] = self._samples_trained
        record["fit_seconds"] = self._last_fit_seconds
        record["predict_seconds"] = predict_seconds
        if self.track_memory:
            record["memory_bytes"] = deep_sizeof((self.model, self.pipeline))
        self._log(record)
        return float(record["accuracy"])

    def _log(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        if self.logger is None:
            return
        if hasattr(self.logger, "info"):
            self.logger.info("%s", record)
        else:
            self.logger(record)

    def run(
        self, stream: Iterable[tuple[npt.ArrayLike, npt.ArrayLike]]
    ) -> list[dict[str, Any]]:
        """Process a whole stream with prequential (test-then-train) evaluation.

        Each chunk is scored by the current model first, if the model can
        predict yet, and only then used for training. This way no chunk is
        scored by a model that has already learned from it.

        Parameters
        ----------
        stream : iterable of (X_chunk, y_chunk)

        Returns
        -------
        list of dict
            The log records produced during this run.
        """
        start = len(self.history)
        for chunk_number, (X_chunk, y_chunk) in enumerate(stream, start=self._chunk_index + 1):
            if self.is_ready:
                self.score_chunk(X_chunk, y_chunk, chunk_index=chunk_number)
            self.fit_chunk(X_chunk, y_chunk)
        return self.history[start:]

    def log_chunk(self, metric_name: str, value: float) -> dict[str, Any]:
        """Add a metric you computed yourself to the current chunk's record.

        If the latest record is for the current chunk, the value is added to
        it. Otherwise a new record is started for this chunk.
        """
        if not metric_name:
            raise ValueError("metric_name must not be empty.")
        if self.history and self.history[-1]["chunk"] == self._chunk_index:
            record = self.history[-1]
        else:
            record = {"chunk": self._chunk_index}
            self.history.append(record)
        record[metric_name] = float(value)
        return record

    def metric_history(self, name: str) -> np.ndarray:
        """Return one logged value from every record, with NaN where it is missing.

        Handy for plotting, for example
        ``plot_metric_over_time(trainer.metric_history("accuracy"), ...)``.
        """
        return np.array([record.get(name, np.nan) for record in self.history], dtype=float)

    @property
    def cumulative_accuracy(self) -> float:
        """Accuracy over every scored sample so far, or 0.0 if nothing has been scored."""
        if self._scored_samples == 0:
            return 0.0
        return self._correct / self._scored_samples

    @property
    def logs(self) -> list[dict[str, Any]]:
        """Shallow copy of the log records."""
        return list(self.history)

    def __len__(self) -> int:
        return len(self.history)
