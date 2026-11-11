import numpy as np
import pytest

from numcompute_stream.metrics import (
    AUC,
    F1,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    RollingWindowMetric,
    StreamingMetric,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)

# Example: TP=3, FP=1, FN=2, TN=4 when 1 is the positive label.
Y_TRUE = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]


def test_accuracy_of_a_mixed_prediction() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.7)


def test_precision_of_a_mixed_prediction() -> None:
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(3 / 4)


def test_recall_of_a_mixed_prediction() -> None:
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(3 / 5)


def test_f1_of_a_mixed_prediction() -> None:
    p, r = 3 / 4, 3 / 5

    assert f1(Y_TRUE, Y_PRED) == pytest.approx(2 * p * r / (p + r))


def test_confusion_matrix_for_binary_labels() -> None:
    result = confusion_matrix(Y_TRUE, Y_PRED)

    # Rows are actual labels and columns are predicted labels.
    np.testing.assert_array_equal(result, [[4, 1], [2, 3]])


def test_confusion_matrix_for_multiclass_labels() -> None:
    result = confusion_matrix([0, 1, 2, 2, 1], [0, 2, 2, 1, 1])

    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 1, 1], [0, 1, 1]])


def test_mse_of_a_simple_prediction() -> None:
    assert mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([1, 0, 1], [1, 0], "must have the same shape"),
    ],
)
def test_metrics_reject_empty_or_mismatched_inputs(
    y_true: list, y_pred: list, message: str
) -> None:
    for func in (accuracy, confusion_matrix, precision, recall, f1):
        with pytest.raises(ValueError, match=message):
            func(y_true, y_pred)

    with pytest.raises(ValueError, match=message):
        mse(y_true, y_pred)


@pytest.mark.parametrize("func", [precision, recall, f1])
@pytest.mark.parametrize(
    ("zero_division", "expected"),
    [
        (0, 0.0),
        (1, 1.0),
    ],
)
def test_zero_division_fallback_when_there_are_no_positives(
    func, zero_division: int, expected: float
) -> None:
    # There are no positive labels or predictions, so the ratio is 0/0.
    assert func([0, 0], [0, 0], zero_division=zero_division) == expected


# ROC example: scores for two negatives (0.1, 0.4) and two positives (0.35, 0.8).
ROC_TRUE = [0, 0, 1, 1]
ROC_SCORES = [0.1, 0.4, 0.35, 0.8]


def test_roc_curve_returns_known_points() -> None:
    fpr, tpr, thresholds = roc_curve(ROC_TRUE, ROC_SCORES)

    np.testing.assert_allclose(fpr, [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])


def test_roc_curve_merges_tied_scores_into_one_threshold() -> None:
    fpr, tpr, thresholds = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])

    np.testing.assert_allclose(fpr, [0.0, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 1.0])
    np.testing.assert_allclose(thresholds, [np.inf, 0.5])


def test_roc_curve_uses_a_custom_positive_label() -> None:
    fpr, tpr, _ = roc_curve(
        ["a", "b", "a", "b"],
        [0.1, 0.9, 0.2, 0.8],
        positive_label="b",
    )

    np.testing.assert_allclose(fpr, [0.0, 0.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 0.5, 1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("y_true", "y_score", "kwargs", "error", "message"),
    [
        ([], [], {}, ValueError, "must not be empty"),
        ([0, 1, 1], [0.1, 0.9], {}, ValueError, "must have the same shape"),
        ([[0, 1]], [[0.1, 0.9]], {}, ValueError, "must be 1D arrays"),
        ([0, 1], [0.1, np.nan], {}, ValueError, "must not contain NaN or infinity"),
        ([0, 1], ["a", "b"], {}, TypeError, "real numeric values"),
        ([1, 1, 1], [0.1, 0.5, 0.9], {}, ValueError, "exactly two classes"),
        ([0, 1, 2], [0.1, 0.5, 0.9], {}, ValueError, "exactly two classes"),
        ([0, 1], [0.1, 0.9], {"positive_label": 5}, ValueError, "positive_label"),
    ],
)
def test_roc_curve_rejects_invalid_inputs(
    y_true: list, y_score: list, kwargs: dict, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        roc_curve(y_true, y_score, **kwargs)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([0, 1], [0, 1], 0.5),
        ([0, 0.5, 1], [0, 1, 1], 0.75),

        # x in decreasing order should give the same non-negative area.
        ([1, 0.5, 0], [1, 1, 0], 0.75),
    ],
)
def test_auc_uses_the_trapezoidal_rule(x: list, y: list, expected: float) -> None:
    assert auc(x, y) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "y", "error", "message"),
    [
        ([0, 1, 2], [0, 1], ValueError, "must have the same length"),
        ([0], [0], ValueError, "at least two points"),
        ([[0, 1]], [[0, 1]], ValueError, "must be 1D arrays"),
        (["a", "b"], [0, 1], TypeError, "real numeric values"),
        ([0, np.nan], [0, 1], ValueError, "must not contain NaN or infinity"),
        ([0, 2, 1], [0, 1, 2], ValueError, "must be monotonic"),
    ],
)
def test_auc_rejects_invalid_inputs(
    x: list, y: list, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        auc(x, y)


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected"),
    [
        (ROC_TRUE, ROC_SCORES, 0.75),
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0),
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
    ],
)
def test_roc_auc_score_for_known_rankings(
    y_true: list, y_score: list, expected: float
) -> None:
    assert roc_auc_score(y_true, y_score) == pytest.approx(expected)


def test_roc_auc_score_matches_auc_of_the_roc_curve() -> None:
    fpr, tpr, _ = roc_curve(ROC_TRUE, ROC_SCORES)

    assert roc_auc_score(ROC_TRUE, ROC_SCORES) == pytest.approx(auc(fpr, tpr))

# Streaming metrics


def _stream(y_true, y_pred, n_chunks):
    return zip(np.array_split(np.asarray(y_true), n_chunks), np.array_split(np.asarray(y_pred), n_chunks))


@pytest.fixture
def multiclass_labels():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 3, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.integers(0, 3, 300))
    return y_true, y_pred


@pytest.mark.parametrize("n_chunks", [1, 4, 11])
@pytest.mark.parametrize(
    ("metric_class", "batch_function"),
    [(Accuracy, accuracy), (Precision, precision), (Recall, recall), (F1, f1)],
)
def test_streaming_metric_equals_batch_metric_for_any_chunking(
    metric_class, batch_function, n_chunks
) -> None:
    metric = metric_class()
    for true_chunk, pred_chunk in _stream(Y_TRUE * 3, Y_PRED * 3, n_chunks):
        metric.update(true_chunk, pred_chunk)
    assert metric.result() == pytest.approx(batch_function(Y_TRUE * 3, Y_PRED * 3))


def test_streaming_accuracy_accumulates_sample_weighted_counts() -> None:
    metric = Accuracy()
    metric.update([1, 1], [1, 0])
    metric.update([0] * 8, [0] * 8)
    assert metric.result() == pytest.approx(0.9)


def test_streaming_accuracy_reset_clears_state() -> None:
    metric = Accuracy()
    metric.update([0, 1], [0, 0])
    metric.reset()
    assert metric.result() == 0.0


def test_streaming_precision_handles_positive_label_never_seen() -> None:
    metric = Precision(positive_label="yes", zero_division=1)
    metric.update(["no"], ["no"])
    assert metric.result() == 1.0


def test_streaming_averages_match_hand_computed_per_class_scores(multiclass_labels) -> None:
    y_true, y_pred = multiclass_labels
    matrix = confusion_matrix(y_true, y_pred)
    tp = np.diag(matrix)
    per_class_precision = tp / matrix.sum(axis=0)
    per_class_recall = tp / matrix.sum(axis=1)
    support = matrix.sum(axis=1)

    expected = {
        (Precision, "macro"): per_class_precision.mean(),
        (Precision, "weighted"): np.average(per_class_precision, weights=support),
        (Precision, "micro"): accuracy(y_true, y_pred),
        (Recall, "macro"): per_class_recall.mean(),
        (Recall, "micro"): accuracy(y_true, y_pred),
    }
    for (metric_class, average), value in expected.items():
        metric = metric_class(average=average)
        for true_chunk, pred_chunk in _stream(y_true, y_pred, 6):
            metric.update(true_chunk, pred_chunk)
        assert metric.result() == pytest.approx(value), (metric_class, average)


def test_streaming_metric_rejects_invalid_average_and_zero_division() -> None:
    with pytest.raises(ValueError, match="average"):
        F1(average="samples")
    with pytest.raises(ValueError, match="zero_division"):
        Recall(zero_division=2)


def test_streaming_metric_update_validates_chunks() -> None:
    with pytest.raises(ValueError, match="same shape"):
        Accuracy().update([0, 1], [0])


def test_streaming_metric_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        StreamingMetric()


def test_streaming_confusion_matrix_equals_batch(multiclass_labels) -> None:
    y_true, y_pred = multiclass_labels
    metric = ConfusionMatrix()
    for true_chunk, pred_chunk in _stream(y_true, y_pred, 9):
        metric.update(true_chunk, pred_chunk)
    np.testing.assert_array_equal(metric.result(), confusion_matrix(y_true, y_pred))


def test_streaming_confusion_matrix_remaps_labels_when_classes_expand() -> None:
    metric = ConfusionMatrix()
    metric.update(["b"], ["b"])
    metric.update(["a", "c"], ["c", "a"])
    np.testing.assert_array_equal(metric.labels_, ["a", "b", "c"])
    np.testing.assert_array_equal(metric.result(), [[0, 0, 1], [0, 1, 0], [1, 0, 0]])


def test_streaming_confusion_matrix_honours_configured_label_order() -> None:
    metric = ConfusionMatrix(labels=[1, 0])
    metric.update([1, 1, 0], [1, 0, 0])
    np.testing.assert_array_equal(metric.result(), [[1, 1], [0, 1]])


def test_streaming_confusion_matrix_reset_preserves_configured_labels() -> None:
    metric = ConfusionMatrix(labels=[1, 0])
    metric.update([0, 1], [0, 1])
    metric.reset()
    np.testing.assert_array_equal(metric.result(), np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="configured label set"):
        metric.update([2], [2])


@pytest.mark.parametrize("labels", [[], [1, 1], [[0, 1]]])
def test_streaming_confusion_matrix_rejects_invalid_labels(labels) -> None:
    with pytest.raises(ValueError, match="labels"):
        ConfusionMatrix(labels=labels)


def test_streaming_auc_equals_batch_auc_across_chunks() -> None:
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 400)
    scores = np.clip(0.4 * y_true + 0.6 * rng.random(400), 0, 1)
    metric = AUC()
    for true_chunk, score_chunk in _stream(y_true, scores, 7):
        metric.update(true_chunk, score_chunk)
    assert metric.result() == pytest.approx(roc_auc_score(y_true, scores))


def test_binned_auc_equals_exact_auc_of_quantised_scores() -> None:
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 2, 500)
    scores = np.clip(0.3 * y_true + 0.7 * rng.random(500), 0, 1)
    metric = AUC(num_thresholds=20)
    for true_chunk, score_chunk in _stream(y_true, scores, 5):
        metric.update(true_chunk, score_chunk)
    quantised = np.minimum((scores * 20).astype(int), 19)
    assert metric.result() == pytest.approx(roc_auc_score(y_true, quantised))
    assert metric._positive_counts.size == 20  # memory independent of stream length


def test_streaming_auc_is_nan_until_both_classes_seen() -> None:
    for metric in (AUC(), AUC(num_thresholds=10)):
        assert np.isnan(metric.result())
        metric.update([1, 1], [0.8, 0.9])
        assert np.isnan(metric.result())
        metric.update([0], [0.1])
        assert metric.result() == pytest.approx(1.0)


def test_streaming_auc_copies_input_arrays_before_accumulating() -> None:
    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    metric = AUC()
    metric.update(labels, scores)
    labels[:] = [1, 0]
    scores[:] = [0.9, 0.1]
    assert metric.result() == pytest.approx(1.0)


def test_binned_auc_rejects_scores_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AUC(num_thresholds=10).update([0, 1], [0.2, 1.5])


def test_rolling_metric_uses_only_recent_chunks() -> None:
    metric = RollingWindowMetric(Accuracy(), window_size=2)
    metric.update([1, 1], [1, 1])
    metric.update([1, 1], [1, 0])
    metric.update([1, 1], [0, 0])
    assert metric.result() == pytest.approx(0.25)


def test_rolling_metric_wraps_score_based_metrics() -> None:
    metric = RollingWindowMetric(AUC(), window_size=1)
    assert metric.requires_scores is True
    metric.update([0, 1], [0.9, 0.1])
    metric.update([0, 1], [0.1, 0.9])
    assert metric.result() == pytest.approx(1.0)


def test_rolling_metric_rejects_invalid_chunk_without_corrupting_window() -> None:
    metric = RollingWindowMetric(Accuracy(), window_size=3)
    metric.update([1], [1])
    with pytest.raises(ValueError, match="same shape"):
        metric.update([1, 0], [1])
    assert metric.result() == 1.0


def test_rolling_metric_reset_clears_recent_window() -> None:
    metric = RollingWindowMetric(Accuracy(), window_size=2)
    metric.update([1], [1])
    metric.reset()
    assert metric.result() == 0.0


def test_rolling_metric_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RollingWindowMetric(Accuracy(), window_size=0)
    with pytest.raises(TypeError, match="StreamingMetric"):
        RollingWindowMetric(accuracy, window_size=2)


def test_shape_mismatch_messages_report_received_shapes() -> None:
    with pytest.raises(ValueError, match=r"Received \(3,\) and \(2,\)"):
        accuracy([0, 1, 1], [0, 1])
    with pytest.raises(ValueError, match=r"Received shapes \(1, 2\) and \(1, 2\)"):
        roc_curve([[0, 1]], [[0.1, 0.9]])
