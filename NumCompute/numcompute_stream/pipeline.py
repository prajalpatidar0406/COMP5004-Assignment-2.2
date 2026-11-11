"""Chain transformers and a final estimator, for batch or streaming data.

The main classes are:

* :class:`Pipeline` runs transformers in order, then an optional final
  estimator. The built-in 1D preprocessors (scalers, imputer, encoder) are
  applied to each column of 2D input separately.
* :class:`Compose` runs transformers in order and acts as one transformer.
* :class:`FeatureUnion` runs transformers side by side on the same input
  and joins their outputs column-wise.

All three have ``partial_fit``, so a full preprocessing and model chain can
be trained one chunk at a time::

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ])
    for X_chunk, y_chunk in stream:
        pipe.partial_fit(X_chunk, y_chunk)
"""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from numcompute.pipeline import Estimator, Transformer, _require_transformer

from .preprocessing import (
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    _ColumnwiseStep,
    _update_transformer,
)

# These only accept 1D input, so Pipeline fits one copy per column of 2D data.
_COLUMNWISE_TRANSFORMERS = (Imputer, MinMaxScaler, OneHotEncoder, StandardScaler)


def _fit_final_estimator(estimator: object, X: npt.ArrayLike, y: npt.ArrayLike) -> None:
    """Train an estimator from scratch with ``fit``, or ``partial_fit`` if it lacks fit."""
    fit = getattr(estimator, "fit", None)
    if callable(fit):
        fit(X, y)
        return
    partial_fit = getattr(estimator, "partial_fit", None)
    if callable(partial_fit):
        partial_fit(X, y)
        return
    raise TypeError("the final estimator must implement fit or partial_fit.")


def _update_final_estimator(estimator: object, X: npt.ArrayLike, y: npt.ArrayLike) -> None:
    """Update an estimator with one chunk via ``partial_fit``, or ``fit`` as a fallback."""
    partial_fit = getattr(estimator, "partial_fit", None)
    if callable(partial_fit):
        partial_fit(X, y)
        return
    _fit_final_estimator(estimator, X, y)


def _validate_steps(steps: list[tuple[str, object]], kind: str) -> None:
    if not steps:
        raise ValueError(f"{kind} must contain at least one named component.")
    names = [name for name, _ in steps]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{kind} names must be non-empty and unique.")


class Compose:
    """Run named transformers one after another as a single transformer.

    Each step's output is the next step's input. The number of samples stays
    the same, but the number of features may change. Cost is the sum of the
    step costs, and only one intermediate output is held at a time.
    """

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = list(steps)
        _validate_steps(self.steps, "steps")
        self._updated_steps: set[str] = set()

    def _transformers(self) -> Iterable[tuple[str, Transformer]]:
        for name, component in self.steps:
            yield name, _require_transformer(
                component, "each composed step must implement fit and transform."
            )

    def _fit_transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        current = X
        for _, transformer in self._transformers():
            transformer.fit(current)
            current = transformer.transform(current)
        return current

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Compose":
        """Fit each step in turn on the output of the step before it."""
        self._fit_transform(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Run X through each fitted step in order."""
        current = X
        for _, transformer in self._transformers():
            current = transformer.transform(current)
        return current

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """Fit every step and return the final transformed X."""
        return self._fit_transform(X)

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Compose":
        """Update each step with one chunk, passing it through earlier steps first.

        Steps without ``partial_fit`` are fitted on the first chunk only.
        """
        current = X
        for name, transformer in self._transformers():
            _update_transformer(transformer, current, name not in self._updated_steps)
            self._updated_steps.add(name)
            current = transformer.transform(current)
        return self


class FeatureUnion:
    """Run several transformers on the same input and join their outputs by column.

    Input has shape ``(n_samples, n_features)``. Output has shape
    ``(n_samples, sum(branch_feature_counts))``. Every branch must keep the
    sample count and return a 2D array. Joining the outputs needs memory
    the size of the result.
    """

    def __init__(self, transformer_list: Iterable[tuple[str, object]]) -> None:
        self.transformer_list = list(transformer_list)
        _validate_steps(self.transformer_list, "transformer_list")
        self._updated_steps: set[str] = set()

    def _transformers(self) -> Iterable[tuple[str, Transformer]]:
        for name, component in self.transformer_list:
            yield name, _require_transformer(
                component, "each union step must implement fit and transform."
            )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "FeatureUnion":
        """Fit every branch on the same X."""
        for _, transformer in self._transformers():
            transformer.fit(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Transform X with every branch and join the 2D outputs column-wise."""
        outputs = [np.asarray(transformer.transform(X)) for _, transformer in self._transformers()]
        if any(output.ndim != 2 for output in outputs):
            raise ValueError(
                "FeatureUnion transformers must return two-dimensional arrays. "
                f"Received shapes {[output.shape for output in outputs]}."
            )
        if len({output.shape[0] for output in outputs}) != 1:
            raise ValueError(
                "FeatureUnion outputs must have equal sample counts. "
                f"Received {[output.shape[0] for output in outputs]}."
            )
        return np.concatenate(outputs, axis=1)

    def fit_transform(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.generic]:
        """Fit every branch, then return their joined outputs for X."""
        return self.fit(X, y).transform(X)

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "FeatureUnion":
        """Update every branch with the same chunk.

        If a branch's output gets wider (for example an encoder learns a new
        category), the new columns appear inside that branch's block. That
        shifts the columns of every later branch, so put branches that can
        grow last.
        """
        for name, transformer in self._transformers():
            _update_transformer(transformer, X, name not in self._updated_steps)
            self._updated_steps.add(name)
        return self


class Pipeline:
    """Run transformers in order, then an optional final estimator.

    Parameters
    ----------
    steps : iterable of (str, object)
        At least one ``(name, component)`` pair with unique, non-empty
        names. If the last component has ``predict`` it is used as the
        estimator. Every other component must have ``fit`` and ``transform``.

    Raises
    ------
    ValueError
        If ``steps`` is empty, names are empty or repeated, or a component
        is None.

    Notes
    -----
    The built-in 1D preprocessors are applied to each column of 2D input
    separately, and each column learns its own statistics. Other
    transformers get the whole array. Each call costs the sum of the step
    costs, and the number of samples is preserved.
    """

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = list(steps)
        _validate_steps(self.steps, "steps")
        if any(component is None for _, component in self.steps):
            raise ValueError("pipeline components must not be None.")
        self.named_steps = dict(self.steps)
        self._columnwise: dict[str, _ColumnwiseStep] = {}
        self._updated_steps: set[str] = set()
        self._is_fitted = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _final_estimator(self) -> object | None:
        """Return the last component if it has ``predict``, otherwise None."""
        final = self.steps[-1][1]
        return final if hasattr(final, "predict") else None

    def _transformers(self) -> list[tuple[str, Transformer]]:
        """Return every step except the estimator, checking each can transform."""
        steps = self.steps[:-1] if self._final_estimator() is not None else self.steps
        return [
            (
                name,
                _require_transformer(
                    component, "non-final pipeline steps must implement fit and transform."
                ),
            )
            for name, component in steps
        ]

    def _validate_target(self, X: npt.ArrayLike, y: npt.ArrayLike | None) -> None:
        """Check that y is given, 1D and the right length when there is an estimator."""
        final = self._final_estimator()
        if final is None:
            return
        if not hasattr(final, "fit") and not hasattr(final, "partial_fit"):
            raise TypeError("the final estimator must implement fit or partial_fit and predict.")
        if y is None:
            raise ValueError("y is required when the final pipeline step is an estimator.")
        y_arr = np.asarray(y)
        n_samples = np.asarray(X).shape[0] if np.ndim(X) else 0
        if y_arr.ndim != 1 or y_arr.size != n_samples:
            raise ValueError(
                "y must be 1D and match the number of X samples. "
                f"Received y shape {y_arr.shape} for {n_samples} samples."
            )

    @staticmethod
    def _is_columnwise(transformer: object, values: npt.NDArray[Any]) -> bool:
        return values.ndim == 2 and isinstance(transformer, _COLUMNWISE_TRANSFORMERS)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None) -> npt.ArrayLike:
        self._is_fitted = False
        self._validate_target(X, y)
        self._columnwise = {}
        current: npt.ArrayLike = X
        for name, transformer in self._transformers():
            values = np.asarray(current)
            if self._is_columnwise(transformer, values):
                step = _ColumnwiseStep(transformer)
                self._columnwise[name] = step
                current = step.fit_transform(values)
            else:
                transformer.fit(current)
                current = np.asarray(transformer.transform(current))
            self._updated_steps.add(name)

        final = self._final_estimator()
        if final is not None:
            assert y is not None
            _fit_final_estimator(final, current, y)
        self._is_fitted = True
        return current

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Pipeline":
        """Fit the transformers in order, then the final estimator if there is one.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), optional
            Required when the final step is an estimator.

        Returns
        -------
        Pipeline
            The fitted pipeline.

        Raises
        ------
        TypeError
            If a component is missing the methods it needs.
        ValueError
            If y is missing, not 1D, or a different length from X.
        """
        self._fit(X, y)
        return self

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Pipeline":
        """Update every step with one chunk.

        Each transformer is updated with ``partial_fit``. A transformer
        without ``partial_fit`` is fitted on the first chunk and then left
        as is. The chunk is transformed with the updated state and passed to
        the next step.
        Finally the estimator is updated with its own ``partial_fit``. For a
        buffered :class:`~numcompute_stream.tree.DecisionTreeClassifier` this adds
        the chunk to the buffer and only retrains when the buffer is full.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Non-empty chunk.
        y : array-like of shape (n_samples,), optional
            Required when the final step is an estimator.

        Returns
        -------
        Pipeline
            The updated pipeline.

        Raises
        ------
        TypeError, ValueError
            Same as :meth:`fit`, or if the chunk is empty or not 1D or 2D.
        """
        values = np.asarray(X)
        if values.ndim not in (1, 2):
            raise ValueError(f"X must be a 1D or 2D array. Received shape {values.shape}.")
        if values.shape[0] == 0:
            raise ValueError("X must not be empty.")
        self._validate_target(values, y)

        current: npt.ArrayLike = values
        for name, transformer in self._transformers():
            step_input = np.asarray(current)
            if self._is_columnwise(transformer, step_input):
                step = self._columnwise.setdefault(name, _ColumnwiseStep(transformer))
                current = step.partial_fit_transform(step_input)
            else:
                _update_transformer(transformer, current, name not in self._updated_steps)
                current = np.asarray(transformer.transform(current))
            self._updated_steps.add(name)

        final = self._final_estimator()
        if final is not None:
            assert y is not None
            _update_final_estimator(final, current, y)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Run X through every transformer step (not the estimator).

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")
        current: npt.ArrayLike = X
        for name, transformer in self._transformers():
            values = np.asarray(current)
            if name in self._columnwise and values.ndim == 2:
                current = self._columnwise[name].transform(values)
            else:
                current = transformer.transform(current)
        return current

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """Fit all steps and return X as transformed by the transformer steps.

        The result has shape ``(n_samples, n_output_features)``. Raises in the
        same cases as :meth:`fit`.
        """
        return self._fit(X, y)

    def _require_estimator(self, method: str) -> Any:
        if not self._is_fitted:
            raise RuntimeError(f"Pipeline must be fitted before {method}.")
        final = self._final_estimator()
        if final is None or not hasattr(final, method):
            raise AttributeError(f"the final pipeline step does not implement {method}.")
        return final

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X, then return the final estimator's predictions.

        Raises
        ------
        RuntimeError
            Before fitting.
        AttributeError
            If the pipeline has no final estimator.
        """
        final = self._require_estimator("predict")
        return cast(Estimator, final).predict(self.transform(X))

    def predict_proba(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X, then return the final estimator's class probabilities.

        Raises
        ------
        RuntimeError
            Before fitting.
        AttributeError
            If the final estimator has no ``predict_proba``.
        """
        return self._require_estimator("predict_proba").predict_proba(self.transform(X))

    @property
    def final_estimator(self) -> object | None:
        """The final estimator, or None if every step is a transformer."""
        return self._final_estimator()

    @property
    def classes_(self) -> npt.NDArray[Any]:
        """Class labels learned by the final estimator. Raises if not fitted."""
        return self._require_estimator("predict").classes_

    @property
    def is_fitted(self) -> bool:
        """True once the pipeline, including any buffered estimator, can predict."""
        final = self._final_estimator()
        return self._is_fitted and bool(getattr(final, "is_fitted", True))
