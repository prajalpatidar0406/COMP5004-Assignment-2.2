"""Tests for StreamTrainer, prequential evaluation and memory measurement."""

import logging
from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.io import iter_csv
from numcompute_stream.metrics import AUC, F1, Accuracy, ConfusionMatrix, RollingWindowMetric, roc_auc_score
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import ColumnSelector, Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer, deep_sizeof
from numcompute_stream.tree import DecisionTreeClassifier

TITANIC = Path(__file__).resolve().parents[1] / "demo" / "data" / "titanic.csv"


@pytest.fixture
def stream():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return list(zip(np.array_split(X, 8), np.array_split(y, 8)))


def test_stream_trainer_logs_weighted_cumulative_accuracy_and_memory():
    model = DecisionTreeClassifier(buffer_size=2).fit([[0], [1]], [0, 1])
    trainer = StreamTrainer(model=model)
    trainer.score_chunk([[0], [1]], [0, 0])
    trainer.score_chunk([[0]], [0])
    assert trainer.cumulative_accuracy == pytest.approx(2 / 3)
    record = trainer.logs[-1]
    assert record["cumulative_accuracy"] == pytest.approx(2 / 3)
    assert record["memory_bytes"] > 0
    assert record["n_samples"] == 1


def test_stream_trainer_fit_chunk_trains_incrementally():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=4))
    trainer.fit_chunk([[0], [1]], [0, 0])
    assert not trainer.is_ready
    trainer.fit_chunk([[2], [3]], [1, 1])
    assert trainer.is_ready
    assert trainer.score_chunk([[0], [3]], [0, 1]) == 1.0
    assert trainer.logs[-1]["samples_trained"] == 4


def test_run_is_prequential_test_then_train(stream):
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=3, buffer_size=100))
    records = trainer.run(stream)
    # 50 samples per chunk: the first retrain happens after chunk 2, so chunks
    # 1-2 cannot be scored and chunk 3 is the first evaluated chunk.
    assert [record["chunk"] for record in records] == [3, 4, 5, 6, 7, 8]
    assert records[0]["samples_trained"] == 100
    assert trainer.cumulative_accuracy > 0.8


def test_run_never_scores_a_chunk_the_model_has_trained_on(stream):
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=1))
    trainer.run(stream)
    for record in trainer.history:
        assert record["samples_trained"] == (record["chunk"] - 1) * 50


def test_stream_trainer_logs_extra_and_score_based_metrics(stream):
    trainer = StreamTrainer(
        DecisionTreeClassifier(max_depth=3, buffer_size=50),
        metrics={"f1": F1(), "auc": AUC()},
    )
    trainer.run(stream[:-1])
    X_last, y_last = stream[-1]
    # Score the final chunk with the current model, before it trains on that chunk.
    expected_auc = roc_auc_score(y_last, trainer.model.predict_proba(X_last)[:, 1])
    trainer.score_chunk(X_last, y_last)
    last = trainer.logs[-1]
    assert last["auc"] == pytest.approx(expected_auc)
    for key in ("f1", "cumulative_f1", "auc", "cumulative_auc"):
        assert 0.0 <= last[key] <= 1.0


def test_stream_trainer_rejects_model_pipeline_as_preprocessing():
    with pytest.raises(TypeError, match="predict"):
        StreamTrainer(Pipeline([("scale", StandardScaler())]))


def test_stream_trainer_logs_nan_for_undefined_chunk_metric():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    trainer = StreamTrainer(model, metrics={"auc": AUC()})
    trainer.score_chunk([[0], [0.2]], [0, 0])  # single class: AUC undefined
    assert np.isnan(trainer.logs[-1]["auc"])


def test_stream_trainer_logs_confusion_matrix_values():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    trainer = StreamTrainer(model, metrics={"cm": ConfusionMatrix()})
    trainer.score_chunk([[0], [1]], [0, 0])
    np.testing.assert_array_equal(trainer.logs[-1]["cumulative_cm"], [[1, 1], [0, 0]])


def test_stream_trainer_combines_preprocessing_pipeline_with_model(stream):
    trainer = StreamTrainer(
        DecisionTreeClassifier(max_depth=3, buffer_size=100),
        pipeline=Pipeline([("scale", StandardScaler())]),
    )
    trainer.run(stream)
    scaler = trainer.pipeline.named_steps["scale"]
    assert scaler.n_samples_seen_ == 400
    assert trainer.cumulative_accuracy > 0.8


def test_stream_trainer_accepts_full_pipeline_as_model(stream):
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("tree", DecisionTreeClassifier(max_depth=3, buffer_size=100))]
    )
    trainer = StreamTrainer(pipeline=pipeline)
    trainer.run(stream)
    assert trainer.model is pipeline
    assert len(trainer) == 6


def test_stream_trainer_can_send_records_to_a_callable_logger():
    records = []
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    StreamTrainer(model, logger=records.append).score_chunk([[0]], [0])
    assert len(records) == 1
    assert records[0]["accuracy"] == 1.0


def test_stream_trainer_supports_logging_logger(caplog):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    logger = logging.getLogger("numcompute_stream.test")
    with caplog.at_level(logging.INFO, logger="numcompute_stream.test"):
        StreamTrainer(model, logger=logger).score_chunk([[0]], [0])
    assert "cumulative_accuracy" in caplog.text


def test_metric_history_and_log_chunk(stream):
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=50), track_memory=False)
    trainer.run(stream[:3])
    history = trainer.metric_history("accuracy")
    assert history.shape == (2,)
    trainer.log_chunk("custom", 0.5)
    assert trainer.logs[-1]["custom"] == 0.5
    assert "memory_bytes" not in trainer.logs[0]


def test_stream_trainer_reset_clears_logs_and_metrics(stream):
    metric = F1()
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=50), metrics={"f1": metric})
    trainer.run(stream)
    trainer.reset()
    assert len(trainer) == 0 and trainer.cumulative_accuracy == 0.0
    assert metric.result() == 0.0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({}, ValueError, "model or pipeline"),
        ({"model": object()}, TypeError, "predict"),
        ({"model": DecisionTreeClassifier(), "metrics": {"x": 1}}, TypeError, "StreamingMetric"),
        ({"model": DecisionTreeClassifier(), "logger": 5}, TypeError, "logger"),
        (
            {"model": DecisionTreeClassifier(), "pipeline": Pipeline([("t", DecisionTreeClassifier())])},
            ValueError,
            "preprocessing",
        ),
    ],
)
def test_stream_trainer_rejects_invalid_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        StreamTrainer(**kwargs)


def test_stream_trainer_rejects_mismatched_chunk_shapes():
    trainer = StreamTrainer(DecisionTreeClassifier())
    with pytest.raises(ValueError, match="same number"):
        trainer.fit_chunk([[0], [1]], [0])


def test_stream_trainer_rejects_invalid_chunk_index():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="chunk_index"):
        StreamTrainer(model).score_chunk([[0]], [0], chunk_index=0)


def test_deep_sizeof_counts_array_data_views_and_slotted_tree_nodes():
    data = np.zeros(10_000)
    assert deep_sizeof(data) >= data.nbytes
    assert deep_sizeof(data[::2]) >= data[::2].nbytes  # a view
    small = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    large = DecisionTreeClassifier().fit(np.arange(200.0).reshape(-1, 1), np.arange(200) % 2)
    assert deep_sizeof(large) > deep_sizeof(small)


def test_deep_sizeof_counts_shared_objects_once():
    array = np.zeros(1000)
    assert deep_sizeof([array, array]) < 2 * array.nbytes


def test_end_to_end_csv_stream_with_mixed_columns():
    """CSV chunks -> imputation/scaling/encoding -> buffered tree -> metrics."""
    header, first = next(iter_csv(TITANIC, chunksize=5))
    survived, sex, age, fare = (header.index(name) for name in ("survived", "sex", "age", "fare"))
    features = FeatureUnion(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("pick", ColumnSelector([age, fare], dtype=float)),
                        ("impute", Imputer()),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
            (
                "category",
                Pipeline(
                    [
                        ("pick", ColumnSelector([sex])),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
            ),
        ]
    )
    model = Pipeline([("features", features), ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=200))])
    trainer = StreamTrainer(model, metrics={"f1": F1()})
    chunks = (
        (data, data[:, survived].astype(float).astype(int))
        for _, data in iter_csv(TITANIC, chunksize=100)
    )
    trainer.run(chunks)
    assert len(trainer) == 7  # 9 chunks, the first two fill the initial buffer
    assert trainer.cumulative_accuracy > 0.7


def test_rolling_metric_window_value_is_logged_as_cumulative(stream):
    trainer = StreamTrainer(
        DecisionTreeClassifier(max_depth=3, buffer_size=50),
        metrics={"recent": RollingWindowMetric(Accuracy(), window_size=3)},
    )
    trainer.run(stream)
    correct = trainer.metric_history("accuracy") * trainer.metric_history("n_samples")
    expected = correct[-3:].sum() / trainer.metric_history("n_samples")[-3:].sum()
    assert trainer.logs[-1]["cumulative_recent"] == pytest.approx(expected)
    # The entry without the "cumulative_" prefix covers only the current chunk.
    assert trainer.logs[-1]["recent"] == pytest.approx(trainer.logs[-1]["accuracy"])


def test_trainer_fallback_fit_pipeline_and_prediction_edges():
    class FitOnlyModel:
        def fit(self, X, y):
            self.fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    class FitOnlyPipeline:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, X):
            self.fit_calls += 1

        def transform(self, X):
            return np.asarray(X)

    model = FitOnlyModel()
    pipeline = FitOnlyPipeline()
    trainer = StreamTrainer(model, pipeline=pipeline, track_memory=False)
    assert trainer.is_ready is False
    trainer.fit_chunk([[0.0]], [0])
    trainer.fit_chunk([[1.0]], [1])
    assert trainer.is_ready is True
    assert pipeline.fit_calls == 1

    class PredictOnly:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    with pytest.raises(TypeError, match="partial_fit or fit"):
        StreamTrainer(PredictOnly())

    with pytest.raises(ValueError, match="matching 1D arrays"):
        StreamTrainer(
            type("WrongShape", (), {"fit": lambda self, X, y: None, "predict": lambda self, X: []})(),
            track_memory=False,
        ).score_chunk([[0.0]], [0])


def test_score_metrics_require_probabilities_and_handle_unseen_class():
    class NoProbabilityModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    trainer = StreamTrainer(NoProbabilityModel(), metrics={"auc": AUC()}, track_memory=False)
    with pytest.raises(TypeError, match="predict_proba"):
        trainer.score_chunk([[0.0], [1.0]], [0, 1])

    class UnseenPositiveModel(NoProbabilityModel):
        classes_ = np.array([0])

        def predict_proba(self, X):
            return np.ones((len(X), 1))

    trainer = StreamTrainer(UnseenPositiveModel(), metrics={"auc": AUC()}, track_memory=False)
    trainer.score_chunk([[0.0], [1.0]], [0, 1])
    assert trainer.logs[-1]["auc"] == pytest.approx(0.5)


def test_stream_log_chunk_starts_a_record_and_sizeof_skips_unset_slots():
    trainer = StreamTrainer(DecisionTreeClassifier(), track_memory=False)
    record = trainer.log_chunk("custom", 2.5)
    assert record == {"chunk": 0, "custom": 2.5}
    with pytest.raises(ValueError, match="must not be empty"):
        trainer.log_chunk("", 1.0)

    class PartiallyInitialized:
        __slots__ = ("present", "missing")

        def __init__(self):
            self.present = {"nested": 1}

    assert deep_sizeof(PartiallyInitialized()) > 0


def test_trainer_logs_nan_when_metric_result_is_undefined():
    class UndefinedAccuracy(Accuracy):
        def result(self):
            raise ValueError("not defined yet")

    model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
    trainer = StreamTrainer(
        model, metrics={"undefined": UndefinedAccuracy()}, track_memory=False
    )
    trainer.score_chunk([[0.0]], [0])
    assert np.isnan(trainer.logs[-1]["undefined"])
    assert np.isnan(trainer.logs[-1]["cumulative_undefined"])
