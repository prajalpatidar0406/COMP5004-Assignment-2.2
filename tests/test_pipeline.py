"""Check that composed transformers and estimators pass data through correctly."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Compose, FeatureUnion, Pipeline
from numcompute_stream.preprocessing import ColumnSelector, Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_fit_transform_chains_transformers():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pipeline = Pipeline([("scale", StandardScaler())])

    transformed = pipeline.fit_transform(values)

    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(pipeline.transform(values), transformed)


def test_pipeline_transform_requires_fit():
    pipeline = Pipeline([("scale", StandardScaler())])

    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.transform([[1.0]])


def test_pipeline_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("step", StandardScaler()), ("step", StandardScaler())])


def test_pipeline_requires_target_for_estimator():
    class Estimator:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.zeros(len(X))

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    with pytest.raises(ValueError, match="y is required"):
        pipeline.fit([[1.0], [2.0]])


def test_pipeline_predict_applies_transformers():
    class Estimator:
        def fit(self, X, y):
            self.width = X.shape[1]

        def predict(self, X):
            return np.full(len(X), self.width)

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    pipeline.fit([[1.0], [2.0]], [0, 1])

    np.testing.assert_array_equal(pipeline.predict([[3.0], [4.0]]), [1, 1])


def test_pipeline_failed_refit_disables_prediction():
    class Estimator:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.zeros(len(X))

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    pipeline.fit([[1.0], [2.0]], [0, 1])

    with pytest.raises(ValueError, match="y is required"):
        pipeline.fit([[10.0], [20.0]])

    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.predict([[1.0]])


def test_pipeline_with_imputer_and_encoder():
    values = np.array([["red"], ["blue"], ["NaN"]], dtype=object)
    pipeline = Pipeline(
        [("impute", Imputer(fill_value="missing")), ("encode", OneHotEncoder())]
    )

    result = pipeline.fit_transform(values)

    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.sum(axis=1), [1.0, 1.0, 1.0])


class AddOffset:
    def __init__(self, offset):
        self.offset = offset

    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) + self.offset


def test_compose_fits_and_applies_steps_in_order():
    compose = Compose([("add", AddOffset(1)), ("add_again", AddOffset(2))])
    values = np.array([[1.0], [2.0]])

    assert compose.fit(values) is compose
    np.testing.assert_array_equal(compose.transform(values), [[4.0], [5.0]])

    fitted = compose.fit_transform(values)

    np.testing.assert_array_equal(fitted, [[4.0], [5.0]])
    np.testing.assert_array_equal(compose.transform(values), fitted)


def test_compose_rejects_steps_without_fit_and_transform():
    compose = Compose([("invalid", object())])

    with pytest.raises(TypeError, match="fit and transform"):
        compose.fit_transform([[1.0]])


def test_composition_helpers_validate_configuration_and_components():
    with pytest.raises(ValueError, match="at least one"):
        Compose([])
    with pytest.raises(ValueError, match="non-empty and unique"):
        FeatureUnion([("same", AddOffset(0)), ("same", AddOffset(1))])

    union = FeatureUnion([("invalid", object())])
    with pytest.raises(TypeError, match="fit and transform"):
        union.fit([[1.0]])


def test_feature_union_concatenates_branch_outputs():
    class FirstColumn(AddOffset):
        def transform(self, X):
            return np.asarray(X)[:, :1] + self.offset

    class SecondColumn(AddOffset):
        def transform(self, X):
            return np.asarray(X)[:, 1:2] + self.offset

    values = np.array([[1.0, 10.0], [2.0, 20.0]])
    union = FeatureUnion([("first", FirstColumn(1)), ("second", SecondColumn(-1))])

    result = union.fit_transform(values)

    np.testing.assert_array_equal(result, [[2.0, 9.0], [3.0, 19.0]])


@pytest.mark.parametrize(
    "transform",
    [
        lambda X: np.asarray(X)[:, 0],
        lambda X: np.asarray(X)[:-1],
    ],
)
def test_feature_union_rejects_invalid_branch_shapes(transform):
    class Transformer:
        def fit(self, X):
            return self

        def transform(self, X):
            return transform(X)

    class Identity:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)

    union = FeatureUnion([("branch", Transformer()), ("identity", Identity())])

    with pytest.raises(ValueError, match="two-dimensional|equal sample counts"):
        union.fit_transform([[1.0], [2.0]])


def test_pipeline_rejects_changed_feature_count_and_predict_without_estimator():
    pipeline = Pipeline([("scale", StandardScaler())])
    pipeline.fit([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="feature count"):
        pipeline.transform([[1.0]])

    with pytest.raises(AttributeError, match="does not implement predict"):
        pipeline.predict([[1.0, 2.0]])


def test_pipeline_rejects_invalid_transformer_and_estimator_components():
    with pytest.raises(TypeError, match="fit and transform"):
        Pipeline([("invalid", object())]).fit([[1.0]])

    class PredictOnly:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="fit and predict"):
        Pipeline([("model", PredictOnly())]).fit([[1.0]], [0])

# Streaming (partial_fit) behaviour


def test_pipeline_partial_fit_uses_tree_buffering_not_refitting():
    tree = DecisionTreeClassifier(buffer_size=4)
    pipeline = Pipeline([("scale", StandardScaler()), ("model", tree)])
    pipeline.partial_fit([[0.0], [1.0]], [0, 0])
    assert tree.buffered_samples == 2 and not tree.is_fitted
    assert not pipeline.is_fitted
    pipeline.partial_fit([[2.0], [3.0]], [1, 1])
    assert tree.retrain_count_ == 1 and pipeline.is_fitted
    np.testing.assert_array_equal(pipeline.predict([[0.0], [3.0]]), [0, 1])


def test_pipeline_partial_fit_scaler_statistics_equal_batch():
    data = np.random.default_rng(1).normal(size=(90, 2)) * [1, 50]
    streamed = Pipeline([("scale", StandardScaler())])
    for chunk in np.array_split(data, 6):
        streamed.partial_fit(chunk)
    batch = Pipeline([("scale", StandardScaler())]).fit(data)
    np.testing.assert_allclose(streamed.transform(data), batch.transform(data))


def test_pipeline_partial_fit_continues_from_batch_fit():
    pipeline = Pipeline([("scale", StandardScaler())]).fit([[0.0], [2.0]])
    pipeline.partial_fit([[4.0], [6.0]])
    assert pipeline.named_steps["scale"].mean_ == pytest.approx(3.0)


def test_pipeline_handles_incremental_category_expansion_with_tree():
    pipeline = Pipeline(
        [("encode", OneHotEncoder()), ("model", DecisionTreeClassifier(buffer_size=4))]
    )
    pipeline.partial_fit([["a"], ["b"]], [0, 0])
    pipeline.partial_fit([["c"], ["c"]], [1, 1])
    np.testing.assert_array_equal(pipeline.predict([["a"], ["c"]]), [0, 1])


def test_pipeline_appends_grown_columns_without_shifting_existing_ones():
    pipeline = Pipeline([("encode", OneHotEncoder())])
    first = pipeline.partial_fit([["a", "x"], ["b", "y"]]).transform([["a", "x"]])
    pipeline.partial_fit([["c", "x"]])  # feature 0 gains category "c"
    after = pipeline.transform([["a", "x"], ["c", "x"]])
    np.testing.assert_array_equal(after[:1, : first.shape[1]], first)
    np.testing.assert_array_equal(after[1], [0, 0, 1, 0, 1])  # "c" column is last


def test_pipeline_category_expansion_does_not_break_an_already_fitted_tree():
    pipeline = Pipeline(
        [("encode", OneHotEncoder()), ("model", DecisionTreeClassifier(buffer_size=2))]
    )
    pipeline.partial_fit([["a"], ["b"]], [0, 1])
    assert pipeline.named_steps["model"].is_fitted
    pipeline.partial_fit([["c"]], [1])
    assert pipeline.predict([["a"], ["c"]]).shape == (2,)


def test_pipeline_new_feature_columns_get_fresh_transformers():
    pipeline = Pipeline([("encode", OneHotEncoder()), ("scale", StandardScaler())])
    pipeline.partial_fit([["a"], ["a"]])
    pipeline.partial_fit([["b"], ["b"]])  # creates a second scaled column
    scalers = pipeline._columnwise["scale"].transformers
    assert len(scalers) == 2
    assert scalers[1].n_samples_seen_ == 2  # did not inherit column 0's state


def test_pipeline_predict_proba_and_classes_pass_through():
    pipeline = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())])
    pipeline.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    np.testing.assert_array_equal(pipeline.classes_, [0, 1])
    np.testing.assert_allclose(pipeline.predict_proba([[0.0], [3.0]]), [[1, 0], [0, 1]])


def test_feature_union_with_column_selectors_streams_mixed_tables():
    features = FeatureUnion(
        [
            ("numeric", Pipeline([("pick", ColumnSelector([0], dtype=float)), ("impute", Imputer())])),
            ("category", Pipeline([("pick", ColumnSelector([1])), ("encode", OneHotEncoder())])),
        ]
    )
    pipeline = Pipeline([("features", features), ("tree", DecisionTreeClassifier(buffer_size=4))])
    pipeline.partial_fit(np.array([[1.0, "a"], [np.nan, "b"]], dtype=object), [0, 1])
    pipeline.partial_fit(np.array([[3.0, "a"], [4.0, "b"]], dtype=object), [0, 1])
    np.testing.assert_array_equal(
        pipeline.predict(np.array([[2.0, "a"], [2.0, "b"]], dtype=object)), [0, 1]
    )


def test_pipeline_partial_fit_requires_targets_for_estimator():
    pipeline = Pipeline([("tree", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match="y is required"):
        pipeline.partial_fit([[0]])


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.empty((0, 1)), [], "must not be empty"),
        (np.zeros((2, 1, 1)), [0, 1], "1D or 2D"),
        ([[0], [1]], [0], "match the number"),
    ],
)
def test_pipeline_partial_fit_rejects_invalid_chunks(X, y, message):
    pipeline = Pipeline([("tree", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match=message):
        pipeline.partial_fit(X, y)


def test_compose_updates_transformers_incrementally():
    class IncrementalOffset:
        def partial_fit(self, X):
            self.seen = getattr(self, "seen", 0) + len(X)
            return self

        def fit(self, X):
            self.seen = len(X)
            return self

        def transform(self, X):
            return np.asarray(X) + self.seen

    transformer = IncrementalOffset()
    compose = Compose([("offset", transformer)])
    compose.partial_fit([1, 2])
    compose.partial_fit([3])
    assert transformer.seen == 3


def test_compose_fits_non_streaming_steps_only_once():
    class CountingFit:
        fits = 0

        def fit(self, X):
            CountingFit.fits += 1
            return self

        def transform(self, X):
            return X

    compose = Compose([("count", CountingFit())])
    compose.partial_fit([1])
    compose.partial_fit([2])
    assert CountingFit.fits == 1


def test_feature_union_updates_each_branch_incrementally():
    class IncrementalIdentity:
        def partial_fit(self, X):
            self.calls = getattr(self, "calls", 0) + 1
            return self

        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X).reshape(-1, 1)

    first = IncrementalIdentity()
    second = IncrementalIdentity()
    union = FeatureUnion([("first", first), ("second", second)])
    union.partial_fit([[1], [2]])
    union.partial_fit([[3]])
    assert first.calls == second.calls == 2


def test_pipeline_rejects_empty_or_none_steps_and_uses_generic_transformers():
    with pytest.raises(ValueError, match="at least one"):
        Pipeline([])
    with pytest.raises(ValueError, match="must not be None"):
        Pipeline([("missing", None)])

    class Offset:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X) + 1

    transformed = Pipeline([("offset", Offset())]).fit_transform([[1.0], [2.0]])
    np.testing.assert_array_equal(transformed, [[2.0], [3.0]])


def test_final_estimator_helpers_use_partial_fit_and_report_missing_methods():
    from numcompute_stream.pipeline import _fit_final_estimator, _update_final_estimator

    class IncrementalOnly:
        def partial_fit(self, X, y):
            self.last = (X, y)

    incremental = IncrementalOnly()
    _fit_final_estimator(incremental, [[1]], [0])
    assert incremental.last == ([[1]], [0])

    class FitOnly:
        def fit(self, X, y):
            self.last = (X, y)

    fallback = FitOnly()
    _update_final_estimator(fallback, [[2]], [1])
    assert fallback.last == ([[2]], [1])

    with pytest.raises(TypeError, match="fit or partial_fit"):
        _fit_final_estimator(object(), [[1]], [0])
