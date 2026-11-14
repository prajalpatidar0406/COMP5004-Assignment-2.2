"""Independent numerical checks against pure-Python reference implementations.

These tests deliberately avoid validating NumPy code with NumPy. References
are exact rational arithmetic (``fractions``), the standard library
(``statistics``, ``math``, ``collections``) or brute-force algorithms that
share no code with the implementation under test.
"""

import math
import statistics
from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from numcompute_stream.metrics import AUC, F1, ConfusionMatrix, Precision, Recall
from numcompute_stream.optim import grad, jacobian
from numcompute_stream.preprocessing import MinMaxScaler, StandardScaler
from numcompute_stream.stats import StreamingStats, quantile
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import cosine_similarity, euclidean_distance, manhattan_distance

RNG = np.random.default_rng(2024)
CHUNKS = [RNG.integers(-50, 50, size=n).astype(float) for n in (7, 1, 13, 4, 25)]
ALL_VALUES = [float(value) for chunk in CHUNKS for value in chunk]


def exact_mean_and_variance(values):
    """Exact population mean and variance using rational arithmetic."""
    exact = [Fraction(value) for value in values]
    mean = sum(exact) / len(exact)
    return mean, sum((value - mean) ** 2 for value in exact) / len(exact)


# --- Moments and quantiles ------------------------------------------------- #


def test_streaming_moments_match_exact_rational_arithmetic():
    stats = StreamingStats()
    for chunk in CHUNKS:
        stats.update(chunk)
    mean, variance = exact_mean_and_variance(ALL_VALUES)
    assert stats["mean"] == pytest.approx(float(mean), rel=1e-14, abs=1e-14)
    assert stats["variance"] == pytest.approx(float(variance), rel=1e-13)
    assert stats["min"] == min(ALL_VALUES) and stats["max"] == max(ALL_VALUES)


def test_streaming_scaler_matches_statistics_module():
    scaler = StandardScaler()
    for chunk in CHUNKS:
        scaler.partial_fit(chunk)
    assert scaler.mean_ == pytest.approx(statistics.fmean(ALL_VALUES))
    assert scaler.var_ == pytest.approx(statistics.pvariance(ALL_VALUES))


def test_streaming_quantiles_match_statistics_module():
    stats = StreamingStats(quantiles=[0.25, 0.5, 0.75])
    for chunk in CHUNKS:
        stats.update(chunk)
    # The "inclusive" method of statistics.quantiles uses the same linear interpolation.
    expected = statistics.quantiles(ALL_VALUES, n=4, method="inclusive")
    np.testing.assert_allclose(stats["quantiles"], expected)
    assert quantile(ALL_VALUES, 0.5) == pytest.approx(statistics.median(ALL_VALUES))


def test_streaming_histogram_matches_manual_bin_counting():
    stats = StreamingStats(bins=6)
    for chunk in CHUNKS:
        stats.update(chunk)
    edges = [float(edge) for edge in stats["bin_edges"]]
    manual = [0] * 6
    for value in ALL_VALUES:
        for index in range(6):
            last = index == 5
            if edges[index] <= value < edges[index + 1] or (last and value == edges[-1]):
                manual[index] += 1
                break
    assert stats["histogram"].tolist() == manual


def test_minmax_scaler_matches_hand_formula():
    scaler = MinMaxScaler(feature_range=(-1, 3))
    for chunk in CHUNKS:
        scaler.partial_fit(chunk)
    low, high = min(ALL_VALUES), max(ALL_VALUES)
    expected = [-1 + 4 * (value - low) / (high - low) for value in ALL_VALUES[:10]]
    np.testing.assert_allclose(scaler.transform(np.array(ALL_VALUES[:10])), expected)


# --- Metrics ---------------------------------------------------------------- #

LABELS_TRUE = RNG.integers(0, 3, 120).tolist()
LABELS_PRED = [t if RNG.random() < 0.65 else int(RNG.integers(0, 3)) for t in LABELS_TRUE]


def _feed(metric, y_true, y_pred, n_chunks=5):
    for true_chunk, pred_chunk in zip(
        np.array_split(np.array(y_true), n_chunks), np.array_split(np.array(y_pred), n_chunks)
    ):
        metric.update(true_chunk, pred_chunk)
    return metric


def test_confusion_matrix_matches_counter_of_pairs():
    counts = Counter(zip(LABELS_TRUE, LABELS_PRED))
    expected = [[counts[(t, p)] for p in range(3)] for t in range(3)]
    assert _feed(ConfusionMatrix(), LABELS_TRUE, LABELS_PRED).result().tolist() == expected


def test_binary_precision_recall_f1_match_hand_counts():
    y_true = [label == 2 for label in LABELS_TRUE]
    y_pred = [label == 2 for label in LABELS_PRED]
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    p, r = tp / (tp + fp), tp / (tp + fn)
    assert _feed(Precision(positive_label=2), LABELS_TRUE, LABELS_PRED).result() == pytest.approx(p)
    assert _feed(Recall(positive_label=2), LABELS_TRUE, LABELS_PRED).result() == pytest.approx(r)
    assert _feed(F1(positive_label=2), LABELS_TRUE, LABELS_PRED).result() == pytest.approx(
        2 * p * r / (p + r)
    )


def test_streaming_auc_matches_pairwise_mann_whitney_definition():
    y_true = RNG.integers(0, 2, 80).tolist()
    scores = [round(float(0.3 * t + 0.7 * RNG.random()), 1) for t in y_true]  # many ties
    positives = [s for s, t in zip(scores, y_true) if t == 1]
    negatives = [s for s, t in zip(scores, y_true) if t == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(positives, negatives))
    expected = wins / (len(positives) * len(negatives))
    assert _feed(AUC(), y_true, scores, n_chunks=4).result() == pytest.approx(expected)


# --- Decision tree ---------------------------------------------------------- #


def brute_force_best_split(X, y):
    """Exhaustively score every (feature, threshold) with pure-Python Gini."""

    def gini(labels):
        if not labels:
            return 0.0
        counts = Counter(labels)
        return 1.0 - sum((count / len(labels)) ** 2 for count in counts.values())

    parent = gini(y)
    best = (0.0, None, None)
    for feature in range(len(X[0])):
        values = sorted({row[feature] for row in X})
        for low, high in zip(values, values[1:]):
            threshold = (low + high) / 2
            left = [label for row, label in zip(X, y) if row[feature] <= threshold]
            right = [label for row, label in zip(X, y) if row[feature] > threshold]
            child = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if parent - child > best[0] + 1e-12:
                best = (parent - child, feature, threshold)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_tree_root_split_matches_brute_force_search(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, size=(30, 3)).astype(float)
    y = (X[:, 0] + rng.integers(0, 3, 30) > 4).astype(int)
    gain, feature, threshold = brute_force_best_split(X.tolist(), y.tolist())
    root = DecisionTreeClassifier(max_depth=1).fit(X, y)._root
    assert root.feature == feature
    assert root.threshold == pytest.approx(threshold)


# --- Distances and derivatives --------------------------------------------- #


def test_distances_match_math_module():
    a, b = RNG.normal(size=8).tolist(), RNG.normal(size=8).tolist()
    assert euclidean_distance(a, b) == pytest.approx(math.dist(a, b))
    assert manhattan_distance(a, b) == pytest.approx(math.fsum(abs(x - y) for x, y in zip(a, b)))
    dot = math.fsum(x * y for x, y in zip(a, b))
    norms = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    assert cosine_similarity(a, b) == pytest.approx(dot / norms)


@pytest.mark.parametrize(
    ("function", "derivative", "x"),
    [
        (math.sin, math.cos, 0.7),
        (math.exp, math.exp, -3.0),
        (lambda v: v**3, lambda v: 3 * v**2, 1e6),
        (math.log, lambda v: 1 / v, 50.0),
    ],
)
def test_grad_matches_analytic_derivatives(function, derivative, x):
    assert grad(function, x) == pytest.approx(derivative(x), rel=1e-6)


def test_jacobian_matches_analytic_matrix():
    point = np.array([0.5, 2.0])
    result = jacobian(lambda v: np.array([v[0] * v[1], math.sin(v[0]) + v[1] ** 2]), point)
    expected = [[2.0, 0.5], [math.cos(0.5), 4.0]]
    np.testing.assert_allclose(result, expected, rtol=1e-7)
