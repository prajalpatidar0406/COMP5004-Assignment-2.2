"""End-to-end checks that run several modules together on real data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.benchmarking import benchmark_streaming_buffer_sizes
from numcompute_stream.io import iter_csv, load_csv
from numcompute_stream.metrics import AUC, F1, Accuracy, RollingWindowMetric, accuracy
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import ColumnSelector, Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import compare_models, plot_metric_over_time

TITANIC = Path(__file__).resolve().parents[1] / "demo" / "data" / "titanic.csv"


@pytest.fixture(scope="module")
def columns():
    header, _ = load_csv(TITANIC, fill_value=np.nan)
    return {name: index for index, name in enumerate(header)}


def titanic_stream(chunksize, columns):
    for _, data in iter_csv(TITANIC, chunksize=chunksize):
        yield data, data[:, columns["survived"]].astype(float).astype(int)


def test_batch_and_streaming_packages_coexist():
    from numcompute.metrics import accuracy as batch_accuracy

    assert batch_accuracy.__module__ == "numcompute.metrics"
    assert accuracy.__module__ == "numcompute_stream.metrics"


def test_stream_package_reuses_shared_batch_implementations():
    from numcompute import benchmarking as batch_benchmarking
    from numcompute import histogram_display as batch_histogram_display
    from numcompute import io as batch_io
    from numcompute import metrics as batch_metrics
    from numcompute import pipeline as batch_pipeline
    from numcompute import rank as batch_rank
    from numcompute import sort_search as batch_sort_search
    from numcompute import utils as batch_utils
    from numcompute_stream import benchmarking as stream_benchmarking
    from numcompute_stream import histogram_display as stream_histogram_display
    from numcompute_stream import io as stream_io
    from numcompute_stream import metrics as stream_metrics
    from numcompute_stream import pipeline as stream_pipeline
    from numcompute_stream import rank as stream_rank
    from numcompute_stream import sort_search as stream_sort_search
    from numcompute_stream import utils as stream_utils

    assert stream_io.iter_csv is batch_io.iter_csv
    assert stream_io.load_csv is batch_io.load_csv
    assert stream_histogram_display.display_histogram is batch_histogram_display.display_histogram
    assert stream_pipeline.Transformer is batch_pipeline.Transformer
    assert stream_pipeline.Estimator is batch_pipeline.Estimator
    assert stream_pipeline._require_transformer is batch_pipeline._require_transformer
    assert stream_rank.Ranker is batch_rank.Ranker
    assert stream_benchmarking._positive_int is batch_benchmarking._positive_int
    assert stream_benchmarking._non_negative_int is batch_benchmarking._non_negative_int
    for name in (
        "benchmark",
        "compare",
        "format_table",
        "numpy_vectorized",
        "python_loop",
    ):
        assert getattr(stream_benchmarking, name) is getattr(batch_benchmarking, name)
    assert stream_metrics._safe_divide is batch_metrics._safe_divide
    for name in ("binary_search", "multi_key_sort", "quickselect", "stable_sort", "topk"):
        assert getattr(stream_sort_search, name) is getattr(batch_sort_search, name)
    for name in ("batches", "logsumexp", "relu", "sigmoid", "softmax", "topk_indices"):
        assert getattr(stream_utils, name) is getattr(batch_utils, name)


def make_model(columns, buffer_size):
    numeric = [columns[name] for name in ("pclass", "age", "sibsp", "parch", "fare")]
    categorical = [columns[name] for name in ("sex", "embarked")]
    features = FeatureUnion(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("select", ColumnSelector(numeric, dtype=float)),
                        ("impute", Imputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("select", ColumnSelector(categorical)),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
            ),
        ]
    )
    return Pipeline(
        [("features", features), ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size))]
    )


def test_streaming_csv_statistics_equal_whole_file_statistics(columns):
    _, table = load_csv(TITANIC, fill_value=np.nan)
    fare = table[:, columns["fare"]].astype(float)
    stats = StreamingStats(quantiles=[0.5])
    for _, chunk in iter_csv(TITANIC, chunksize=64):
        stats.update(chunk[:, columns["fare"]].astype(float))
    assert stats["count"] == fare.size
    assert stats["mean"] == pytest.approx(np.mean(fare))
    assert stats["std"] == pytest.approx(np.std(fare))
    assert stats["quantiles"][0] == pytest.approx(np.median(fare))


def test_csv_to_trainer_to_plots(columns, tmp_path):
    trainer = StreamTrainer(
        make_model(columns, buffer_size=180),
        metrics={"f1": F1(), "auc": AUC(), "rolling_accuracy": RollingWindowMetric(Accuracy(), window_size=3)},
    )
    trainer.run(titanic_stream(90, columns))

    assert len(trainer) == 8  # 10 chunks, and the first two only fill the buffer
    assert 0.7 < trainer.cumulative_accuracy <= 1.0
    assert 0.5 < trainer.logs[-1]["cumulative_auc"] <= 1.0
    assert all(record["memory_bytes"] > 0 for record in trainer.logs)

    # The size-weighted mean of the per-chunk accuracies should equal the cumulative accuracy.
    chunk_sizes = trainer.metric_history("n_samples")
    weighted = np.sum(trainer.metric_history("accuracy") * chunk_sizes) / chunk_sizes.sum()
    assert weighted == pytest.approx(trainer.cumulative_accuracy)

    output = tmp_path / "accuracy.png"
    figure, _ = plot_metric_over_time(
        trainer.metric_history("accuracy"),
        "Titanic stream",
        "Accuracy",
        x_values=trainer.metric_history("chunk"),
        save_path=str(output),
        show=False,
    )
    plt.close(figure)
    assert output.stat().st_size > 0


def test_buffer_size_comparison_runs_end_to_end(columns, tmp_path):
    histories = {}
    for buffer_size in (90, 360):
        trainer = StreamTrainer(make_model(columns, buffer_size), track_memory=False)
        for chunk_number, (X, y) in enumerate(titanic_stream(90, columns), start=1):
            if chunk_number > 4:  # evaluate the same chunks for both models
                trainer.score_chunk(X, y, chunk_index=chunk_number)
            trainer.fit_chunk(X, y)
        histories[buffer_size] = trainer.metric_history("accuracy")
    figure, axes = compare_models(
        histories[90], histories[360], ["buffer 90", "buffer 360"], show=False,
        save_path=str(tmp_path / "compare.png"),
    )
    plt.close(figure)
    assert len(axes.lines) == 2 and histories[90].size == histories[360].size == 6


def test_single_retrain_when_buffer_holds_whole_stream(columns):
    stream = list(titanic_stream(100, columns))
    n_samples = sum(y.size for _, y in stream)
    model = make_model(columns, buffer_size=n_samples)
    for X, y in stream:
        assert not model.is_fitted  # nothing to predict with until the buffer fills
        model.partial_fit(X, y)
    tree = model.named_steps["tree"]
    assert tree.retrain_count_ == 1 and tree.buffered_samples == 0
    X_all = np.concatenate([X for X, _ in stream])
    y_all = np.concatenate([y for _, y in stream])
    assert accuracy(y_all, model.predict(X_all)) > 0.75


def test_buffer_benchmark_on_real_features(columns):
    _, table = load_csv(TITANIC, fill_value=np.nan)
    X = np.nan_to_num(table[:, [columns["pclass"], columns["fare"]]].astype(float))
    y = table[:, columns["survived"]].astype(float).astype(int)
    results = benchmark_streaming_buffer_sizes(X, y, [100, 400], chunk_size=100, max_depth=4)
    assert results[100]["retrain_count"] > results[400]["retrain_count"]
    assert all(0.5 < result["accuracy"] <= 1.0 for result in results.values())
