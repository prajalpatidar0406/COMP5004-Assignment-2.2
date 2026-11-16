import numpy as np
import pytest

from numcompute_stream.benchmarking import (
    benchmark,
    benchmark_streaming_buffer_sizes,
    benchmark_toolkit,
    format_toolkit_table,
    toolkit_workloads,
    compare,
    format_table,
    main,
)


def test_benchmark_returns_timing_summary():
    result = benchmark(sum, [1, 2, 3], repeats=2, warmup=0)
    assert result["repeats"] == 2
    assert result["minimum_seconds"] >= 0
    assert result["median_seconds"] >= result["minimum_seconds"]


def test_compare_and_format_table():
    results = compare({"sum": sum, "max": max}, [1, 2, 3], repeats=1, warmup=0)
    table = format_table(results)
    assert "Implementation" in table
    assert "sum" in table and "max" in table
    assert "Environment:" in table
    assert results["sum"]["environment"]["numpy"]


def test_benchmark_validates_repeat_count():
    with pytest.raises(ValueError, match="repeats"):
        benchmark(sum, [1], repeats=0)


def test_benchmark_counts_warmups_and_timed_calls():
    calls = []

    result = benchmark(lambda: calls.append(None), repeats=2, warmup=1)

    assert result["repeats"] == 2
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("function", "kwargs", "error", "message"),
    [
        (None, {}, TypeError, "callable"),
        (sum, {"repeats": True}, ValueError, "repeats"),
        (sum, {"warmup": -1}, ValueError, "repeats"),
    ],
)
def test_benchmark_validates_callable_and_iteration_counts(
    function, kwargs, error, message
):
    with pytest.raises(error, match=message):
        benchmark(function, [1], **kwargs)


def test_compare_validates_all_implementations_before_running():
    called = []

    def implementation():
        called.append(True)

    with pytest.raises(TypeError, match="must be callable"):
        compare(
            {"valid": implementation, "invalid": None},
            repeats=1,
            warmup=0,
        )

    assert called == []


@pytest.mark.parametrize("name", ["", "  ", 1])
def test_compare_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="non-empty strings"):
        compare({name: sum}, [1], repeats=1, warmup=0)


def test_format_table_rejects_empty_results():
    with pytest.raises(ValueError, match="at least one"):
        format_table({})


def test_vectorization_benchmark_prints_comparison(capsys):
    main(["--size", "20", "--repeats", "1", "--warmup", "0"])

    output = capsys.readouterr().out
    assert "Python loop" in output
    assert "NumPy vectorised" in output
    assert "Environment:" in output
    assert "Median speedup:" in output

def test_buffer_size_benchmark_reports_accuracy_and_retraining_time():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 10).astype(int)
    results = benchmark_streaming_buffer_sizes(X, y, [2, 5], chunk_size=2, max_depth=2)
    assert results[2]["retrain_count"] == 10
    assert results[5]["retrain_count"] == 3
    assert results[2]["accuracy"] is not None
    assert results[2]["retraining_seconds"] >= 0
    assert len(results[2]["chunk_accuracies"]) == 9  # all chunks but the first


def test_buffer_size_benchmark_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        benchmark_streaming_buffer_sizes([[0]], [0], [1], chunk_size=0)


def test_buffer_size_benchmark_rejects_empty_sizes():
    with pytest.raises(ValueError, match="at least one"):
        benchmark_streaming_buffer_sizes([[0]], [0], [], chunk_size=1)


def test_toolkit_workloads_agree_with_their_references():
    for workload in toolkit_workloads(size=500, seed=1):
        workload["check"](workload["reference"](), workload["numcompute_stream"]())


def test_benchmark_toolkit_reports_every_workload():
    rows = benchmark_toolkit(size=500, repeats=1, warmup=0)
    assert len(rows) == 6
    assert all(row["reference_seconds"] > 0 and row["speedup"] > 0 for row in rows)
    table = format_toolkit_table(rows)
    assert table.count("\n") == len(rows) + 1
    assert "Decision-tree best split" in table


def test_toolkit_benchmark_rejects_tiny_sizes_and_empty_tables():
    with pytest.raises(ValueError, match="at least 100"):
        toolkit_workloads(size=10)
    with pytest.raises(ValueError, match="at least one"):
        format_toolkit_table([])


def test_main_runs_toolkit_suite(capsys):
    main(["--size", "500", "--repeats", "1", "--warmup", "0", "--suite", "toolkit"])
    assert "Speed-up" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("X", "y", "kwargs", "message"),
    [
        (np.zeros((2, 1, 1)), [0, 1], {}, "1D or 2D"),
        ([[0], [1]], [[0], [1]], {}, "y must be 1D"),
        ([[0], [1]], [0], {}, "same number"),
        (np.empty((0, 1)), [], {}, "must not be empty"),
        ([[0]], [0], {"chunk_size": True}, "chunk_size"),
        ([[0]], [0], {"buffer_sizes": [True]}, "positive integers"),
        ([[0]], [0], {"buffer_sizes": [0]}, "positive integers"),
    ],
)
def test_buffer_size_benchmark_rejects_invalid_shapes_and_sizes(X, y, kwargs, message):
    buffer_sizes = kwargs.pop("buffer_sizes", [1])
    kwargs.setdefault("chunk_size", 1)
    with pytest.raises(ValueError, match=message):
        benchmark_streaming_buffer_sizes(X, y, buffer_sizes, **kwargs)


def test_reference_split_handles_empty_child_and_module_entry_point(capsys, monkeypatch):
    import runpy
    import sys

    from numcompute_stream.benchmarking import _python_best_split

    # NaN makes the candidate threshold NaN, leaving one child empty.
    feature, threshold = _python_best_split([[np.nan], [1.0]], [0, 1])
    assert feature == 0
    assert np.isnan(threshold)

    monkeypatch.setattr(
        sys,
        "argv",
        ["numcompute_stream.benchmarking", "--size", "20", "--repeats", "1", "--warmup", "0"],
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("numcompute_stream.benchmarking", run_name="__main__")
    assert "Median speedup:" in capsys.readouterr().out


def test_buffer_size_benchmark_accepts_one_dimensional_features():
    result = benchmark_streaming_buffer_sizes(
        np.array([0.0, 1.0, 2.0]),
        np.array([0, 0, 1]),
        [1],
        chunk_size=1,
        max_depth=1,
    )
    assert result[1]["accuracy"] is not None
