"""Timing helpers and a command-line benchmark for NumCompute."""

import argparse
from collections.abc import Sequence
from time import perf_counter

import numpy as np
import numpy.typing as npt

from numcompute.benchmarking import (
    _non_negative_int,
    _positive_int,
    benchmark,
    compare,
    format_table,
    numpy_vectorized,
    python_loop,
)


def benchmark_streaming_buffer_sizes(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    buffer_sizes: Sequence[int],
    *,
    chunk_size: int,
    max_depth: int | None = 5,
) -> dict[int, dict[str, object]]:
    """Compare streaming tree accuracy and retraining cost for several buffer sizes.

    Each chunk is scored by the most recently trained tree and then added to
    the training buffer (test-then-train). Chunks that arrive before the
    first retrain cannot be scored. Accuracy is weighted by sample count over
    the scored chunks, and retraining time only counts ``partial_fit`` calls
    that actually triggered a fit.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or (n_samples,)
        Features, fed to the tree in order.
    y : array-like of shape (n_samples,)
        Class labels.
    buffer_sizes : sequence of int
        Buffer sizes to try. Each one gets a fresh tree.
    chunk_size : int
        Number of samples per chunk.
    max_depth : int or None, default=5
        Depth limit passed to each tree.

    Returns
    -------
    dict of int to dict
        For each buffer size: ``buffer_size``, ``accuracy`` (None if no chunk
        was scored), ``chunk_accuracies``, ``retraining_seconds``,
        ``retrain_count`` and ``scored_samples``.

    Raises
    ------
    ValueError
        If the shapes are wrong or do not match, the data is empty,
        ``chunk_size`` is not a positive integer, or ``buffer_sizes`` is
        empty or holds a value that is not a positive integer.
    TypeError
        If ``X`` is not numeric (raised by the tree).

    Complexity
    ----------
    Time: one full streaming run per buffer size, dominated by the tree
    retrains (see :meth:`DecisionTreeClassifier.fit`).
    Space: O(n_samples * n_features) for the chunk views and tree buffer.
    """
    from .tree import DecisionTreeClassifier

    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be 1D or 2D and y must be 1D.")
    if features.shape[0] != targets.size:
        raise ValueError("X and y must contain the same number of samples.")
    if features.shape[0] == 0:
        raise ValueError("X and y must not be empty.")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    sizes = tuple(buffer_sizes)
    if not sizes:
        raise ValueError("buffer_sizes must contain at least one value.")
    if any(
        not isinstance(size, (int, np.integer))
        or isinstance(size, (bool, np.bool_))
        or size < 1
        for size in sizes
    ):
        raise ValueError("buffer_sizes must contain positive integers.")

    results: dict[int, dict[str, object]] = {}
    chunks = [
        (features[start : start + chunk_size], targets[start : start + chunk_size])
        for start in range(0, targets.size, chunk_size)
    ]
    for raw_buffer_size in sizes:
        buffer_size = int(raw_buffer_size)
        model = DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size)
        total_correct = 0
        total_scored = 0
        scores: list[float] = []
        retraining_seconds = 0.0
        retrain_count = 0
        for chunk_X, chunk_y in chunks:
            if model.is_fitted:
                predictions = model.predict(chunk_X)
                total_correct += int(np.count_nonzero(predictions == chunk_y))
                total_scored += int(chunk_y.size)
                scores.append(float(np.mean(predictions == chunk_y)))
            previous_retrains = model.retrain_count_ if model.is_fitted else 0
            started = perf_counter()
            model.partial_fit(chunk_X, chunk_y)
            elapsed = perf_counter() - started
            if getattr(model, "retrain_count_", 0) > previous_retrains:
                retraining_seconds += elapsed
                retrain_count += 1
        results[buffer_size] = {
            "buffer_size": buffer_size,
            "accuracy": float(total_correct / total_scored) if total_scored else None,
            "chunk_accuracies": scores,
            "retraining_seconds": retraining_seconds,
            "retrain_count": retrain_count,
            "scored_samples": total_scored,
        }
    return results


# --------------------------------------------------------------------------- #
# Representative toolkit workloads
# --------------------------------------------------------------------------- #
def _python_standardise(rows: list[list[float]]) -> list[list[float]]:
    """Naive reference: standardise each column (population mean and std) in pure Python."""
    columns = list(zip(*rows))
    stats = []
    for column in columns:
        mean = sum(column) / len(column)
        std = (sum((value - mean) ** 2 for value in column) / len(column)) ** 0.5 or 1.0
        stats.append((mean, std))
    return [[(value - m) / s for value, (m, s) in zip(row, stats)] for row in rows]


def _recompute_streaming_moments(chunks: list[np.ndarray]) -> tuple[float, float]:
    """Naive reference: keep every value and recompute after each chunk (O(N^2))."""
    seen = np.empty(0)
    for chunk in chunks:
        seen = np.concatenate((seen, chunk))
        result = (float(np.mean(seen)), float(np.var(seen)))
    return result


def _incremental_streaming_moments(chunks: list[np.ndarray]) -> tuple[float, float]:
    from .stats import StreamingStats

    stats = StreamingStats(reservoir_size=1)
    for chunk in chunks:
        stats.update(chunk)
    result = stats.result()
    return float(result["mean"]), float(result["variance"])


def _python_confusion(y_true: list[int], y_pred: list[int], n_classes: int) -> list[list[int]]:
    """Naive reference: count (true, predicted) pairs one sample at a time."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def _python_best_split(rows: list[list[float]], labels: list[int]) -> tuple[int, float]:
    """Naive reference: score every threshold by re-partitioning the rows (O(d * n^2))."""

    def gini(group: list[int]) -> float:
        if not group:
            return 0.0
        share = sum(group) / len(group)
        return 1.0 - share**2 - (1.0 - share) ** 2

    best_gain, best = 0.0, (-1, 0.0)
    parent = gini(labels)
    for feature in range(len(rows[0])):
        values = sorted({row[feature] for row in rows})
        for low, high in zip(values, values[1:]):
            threshold = (low + high) / 2
            left = [y for row, y in zip(rows, labels) if row[feature] <= threshold]
            right = [y for row, y in zip(rows, labels) if row[feature] > threshold]
            gain = parent - (len(left) * gini(left) + len(right) * gini(right)) / len(labels)
            if gain > best_gain + 1e-12:
                best_gain, best = gain, (feature, threshold)
    return best


def _numcompute_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    from .tree import DecisionTreeClassifier

    root = DecisionTreeClassifier(max_depth=1).fit(X, y)._root
    return int(root.feature), float(root.threshold)


def _pairwise_auc(y_true: list[int], scores: list[float]) -> float:
    """Naive reference: AUC as the fraction of positive/negative pairs ranked correctly."""
    positives = [s for s, t in zip(scores, y_true) if t == 1]
    negatives = [s for s, t in zip(scores, y_true) if t == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def toolkit_workloads(size: int = 20_000, seed: int = 42) -> list[dict[str, object]]:
    """Build typical NumCompute operations, each paired with a naive reference.

    Each workload is a dict with ``name``, ``n`` (the problem size),
    ``reference`` and ``numcompute_stream`` (zero-argument callables that return
    comparable results) and ``check`` (asserts the two results agree). The
    quadratic references, split search and pairwise AUC, run on a smaller
    ``n`` so the suite still finishes quickly.

    Raises
    ------
    ValueError
        If ``size`` is smaller than 100.
    """
    from .metrics import confusion_matrix, roc_auc_score
    from .preprocessing import StandardScaler
    from .sort_search import topk

    if size < 100:
        raise ValueError("size must be at least 100.")
    rng = np.random.default_rng(seed)

    matrix = rng.normal(size=(size // 10, 10)) * rng.uniform(1, 100, 10)
    rows = matrix.tolist()
    # Make the stream long enough that recomputing from scratch (O(N^2))
    # visibly costs more than the incremental update.
    stream = rng.normal(size=size * 10)
    chunks = np.array_split(stream, 500)
    y_true = rng.integers(0, 5, size)
    y_pred = np.where(rng.random(size) < 0.7, y_true, rng.integers(0, 5, size))
    split_n = min(size // 25, 800)
    split_X = rng.integers(0, 200, size=(split_n, 3)).astype(float)
    split_y = (split_X[:, 1] + rng.normal(0, 30, split_n) > 100).astype(int)
    auc_n = min(size // 10, 2000)
    auc_true = rng.integers(0, 2, auc_n)
    auc_scores = np.round(0.3 * auc_true + 0.7 * rng.random(auc_n), 3)
    values = rng.random(size)

    def close(a: object, b: object) -> None:
        np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    return [
        {
            "name": "StandardScaler fit+transform (10 features)",
            "n": matrix.shape[0],
            "reference": lambda: _python_standardise(rows),
            "numcompute_stream": lambda: StandardScaler().fit_transform(matrix),
            "check": close,
        },
        {
            "name": "Streaming mean/variance over 500 chunks",
            "n": stream.size,
            "reference": lambda: _recompute_streaming_moments(chunks),
            "numcompute_stream": lambda: _incremental_streaming_moments(chunks),
            "check": close,
        },
        {
            "name": "Confusion matrix (5 classes)",
            "n": size,
            "reference": lambda: _python_confusion(y_true.tolist(), y_pred.tolist(), 5),
            "numcompute_stream": lambda: confusion_matrix(y_true, y_pred),
            "check": close,
        },
        {
            "name": "Decision-tree best split (3 features)",
            "n": split_n,
            "reference": lambda: _python_best_split(split_X.tolist(), split_y.tolist()),
            "numcompute_stream": lambda: _numcompute_best_split(split_X, split_y),
            "check": close,
        },
        {
            "name": "ROC AUC",
            "n": auc_n,
            "reference": lambda: _pairwise_auc(auc_true.tolist(), auc_scores.tolist()),
            "numcompute_stream": lambda: roc_auc_score(auc_true, auc_scores),
            "check": close,
        },
        {
            "name": "Top-10 selection",
            "n": size,
            "reference": lambda: sorted(values.tolist(), reverse=True)[:10],
            "numcompute_stream": lambda: topk(values, 10, return_indices=False),
            # topk returns its k results in no particular order (it uses O(n)
            # selection), so sort both sides before comparing.
            "check": lambda a, b: close(sorted(a), sorted(b)),
        },
    ]


def benchmark_toolkit(
    size: int = 20_000, repeats: int = 3, warmup: int = 1, seed: int = 42
) -> list[dict[str, object]]:
    """Time typical toolkit operations against naive references.

    Each pair is checked to give the same answer before it is timed, so we
    never report a speed-up for a wrong result.

    Parameters
    ----------
    size : int, default=20_000
        Base problem size (see :func:`toolkit_workloads`).
    repeats, warmup : int
        Passed to :func:`benchmark`.
    seed : int, default=42
        Seed for the synthetic inputs.

    Returns
    -------
    list of dict
        One row per workload with ``name``, ``n``, ``reference_seconds``,
        ``numcompute_seconds`` (both medians), ``speedup`` and
        ``environment``.

    Raises
    ------
    AssertionError
        If an implementation gives a different answer from its reference.
    """
    rows = []
    for workload in toolkit_workloads(size, seed):
        workload["check"](workload["reference"](), workload["numcompute_stream"]())
        reference = benchmark(workload["reference"], repeats=repeats, warmup=warmup)
        fast = benchmark(workload["numcompute_stream"], repeats=repeats, warmup=warmup)
        reference_seconds = float(reference["median_seconds"])
        numcompute_seconds = float(fast["median_seconds"])
        rows.append(
            {
                "name": workload["name"],
                "n": workload["n"],
                "reference_seconds": reference_seconds,
                "numcompute_seconds": numcompute_seconds,
                "speedup": reference_seconds / max(numcompute_seconds, 1e-12),
                "environment": fast["environment"],
            }
        )
    return rows


def format_toolkit_table(rows: list[dict[str, object]]) -> str:
    """Format :func:`benchmark_toolkit` rows as a Markdown table.

    Raises ``ValueError`` if ``rows`` is empty.
    """
    if not rows:
        raise ValueError("rows must contain at least one workload.")
    lines = [
        "| Workload | n | Naive reference (s) | NumCompute (s) | Speed-up |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['n']:,} | {float(row['reference_seconds']):.4g} | "
            f"{float(row['numcompute_seconds']):.4g} | {float(row['speedup']):.0f}x |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: run the arithmetic benchmark, toolkit suite, or both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=_positive_int, default=200_000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_non_negative_int, default=1)
    parser.add_argument(
        "--suite", choices=("arithmetic", "toolkit", "all"), default="arithmetic"
    )
    args = parser.parse_args(argv)

    if args.suite in ("arithmetic", "all"):
        data = np.random.default_rng(42).random(args.size)
        np.testing.assert_allclose(python_loop(data), numpy_vectorized(data))
        results = compare(
            {
                "Python loop": python_loop,
                "NumPy vectorised": numpy_vectorized,
            },
            data,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        print(format_table(results))
        speedup = (
            float(results["Python loop"]["median_seconds"])
            / float(results["NumPy vectorised"]["median_seconds"])
        )
        print(f"Median speedup: {speedup:.2f}x")
    if args.suite in ("toolkit", "all"):
        size = min(args.size, 20_000)
        print(format_toolkit_table(benchmark_toolkit(size, args.repeats, args.warmup)))


if __name__ == "__main__":
    main()