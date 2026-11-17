"""Tests for the reusable plotting helpers (rendered off-screen)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_decision_tree,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_returns_axes_and_saves(tmp_path):
    output = tmp_path / "accuracy.png"
    _, axes = plot_metric_over_time(
        [0.5, 0.75, 1.0], "Accuracy", "score", save_path=str(output), show=False
    )
    assert axes.get_ylabel() == "score"
    assert output.exists() and output.stat().st_size > 0
    np.testing.assert_array_equal(axes.lines[0].get_xdata(), [1, 2, 3])


def test_plot_metric_over_time_accepts_custom_x_and_existing_axes():
    figure, axes = plt.subplots()
    returned_figure, returned_axes = plot_metric_over_time(
        [0.1, 0.2], "t", "y", x_values=[10, 20], ax=axes, show=False
    )
    assert returned_figure is figure and returned_axes is axes
    np.testing.assert_array_equal(axes.lines[0].get_xdata(), [10, 20])


def test_plot_metric_over_time_keeps_nan_as_gap():
    _, axes = plot_metric_over_time([0.5, np.nan, 0.7], "t", "y", show=False)
    assert np.isnan(axes.lines[0].get_ydata()[1])


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ([], {}, "must not be empty"),
        ([[1, 2]], {}, "1D"),
        ([0.1, 0.2], {"x_values": [1]}, "x_values"),
    ],
)
def test_plot_metric_over_time_rejects_invalid_input(values, kwargs, message):
    with pytest.raises(ValueError, match=message):
        plot_metric_over_time(values, "t", "y", show=False, **kwargs)


def test_compare_models_plots_two_labelled_streams(tmp_path):
    _, axes = compare_models(
        [0.5, 0.75], [0.4, 0.8], ["small", "large"], show=False, save_path=str(tmp_path / "c.svg")
    )
    assert len(axes.lines) == 2
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["small", "large"]
    assert (tmp_path / "c.svg").exists()


def test_compare_models_rejects_different_chunk_counts():
    with pytest.raises(ValueError, match="same number"):
        compare_models([0.5], [0.5, 0.6], ["a", "b"], show=False)


@pytest.mark.parametrize("labels", [["a"], ["a", ""], ["a", "b", "c"]])
def test_compare_models_requires_two_names(labels):
    with pytest.raises(ValueError, match="two non-empty"):
        compare_models([0.5], [0.6], labels, show=False)


def test_predictions_plot_reports_accuracy_and_marks_errors():
    _, axes = plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 0], show=False)
    assert "75.0%" in axes.get_title(loc="left")
    assert len(axes.collections) == 3  # error connectors + two scatter series


def test_predictions_plot_supports_string_labels():
    _, axes = plot_predictions_vs_ground_truth(["no", "yes"], ["yes", "yes"], show=False)
    assert [tick.get_text() for tick in axes.get_yticklabels()] == ["no", "yes"]


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [([0, 1], [1], "same shape"), ([], [], "must not be empty"), ([[0]], [[0]], "1D")],
)
def test_predictions_plot_rejects_invalid_input(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        plot_predictions_vs_ground_truth(y_true, y_pred, show=False)


def test_plot_decision_tree_draws_named_splits_and_saves(tmp_path):
    tree = DecisionTreeClassifier(max_depth=1).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1]
    )
    output = tmp_path / "tree.png"
    _, axes = plot_decision_tree(
        tree,
        feature_names=["score"],
        class_names=["low", "high"],
        save_path=str(output),
        show=False,
    )
    labels = [text.get_text() for text in axes.texts]
    assert any("score <=" in label for label in labels)
    assert any("Predict: low" in label for label in labels)
    assert output.exists() and output.stat().st_size > 0


def test_plot_decision_tree_rejects_unfitted_trees_and_wrong_names():
    with pytest.raises(ValueError, match="fitted"):
        plot_decision_tree(DecisionTreeClassifier(), show=False)

    tree = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="feature_names"):
        plot_decision_tree(tree, feature_names=["x", "y"], show=False)


def test_plot_decision_tree_supports_multiclass_labels():
    tree = DecisionTreeClassifier(max_depth=2).fit([[0], [1], [2]], ["a", "b", "c"])
    _, axes = plot_decision_tree(tree, show=False)
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "Predicts a",
        "Predicts b",
        "Predicts c",
    ]


def test_plotting_show_empty_series_and_unavailable_matplotlib(monkeypatch):
    import numcompute_stream.visualise as visualise

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    plot_metric_over_time([0.5], "Accuracy", "score", show=True)
    assert shown == [True]

    _, axes = compare_models(
        [np.nan, np.nan], [0.2, 0.4], ["missing", "reference"], show=False
    )
    assert len(axes.texts) == 1

    monkeypatch.setattr(visualise, "plt", None)
    with pytest.raises(ImportError, match="matplotlib is required"):
        plot_metric_over_time([0.5], "Accuracy", "score", show=False)


def test_prediction_plot_handles_fractional_labels_without_category_ticks(monkeypatch):
    _, axes = plt.subplots()
    set_yticks = axes.set_yticks
    calls = []

    def track_set_yticks(*args, **kwargs):
        calls.append(args)
        return set_yticks(*args, **kwargs)

    monkeypatch.setattr(axes, "set_yticks", track_set_yticks)
    plot_predictions_vs_ground_truth(
        [0.1, 0.2], [0.2, 0.3], ax=axes, show=False
    )
    assert calls == []


def test_decision_tree_plot_rejects_missing_root_names_and_children():
    from numcompute_stream.tree import _Node
    from types import SimpleNamespace

    tree = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
    tree._root = None
    with pytest.raises(ValueError, match="fitted"):
        plot_decision_tree(tree, show=False)
    with pytest.raises(ValueError, match="fitted"):
        plot_decision_tree(SimpleNamespace(is_fitted=True, _root=None), show=False)

    tree._root = _Node(np.array([1, 1]), feature=0)
    with pytest.raises(ValueError, match="class_names"):
        plot_decision_tree(tree, class_names=["", "one"], show=False)

    with pytest.raises(RuntimeError, match="without two children"):
        plot_decision_tree(tree, show=False)


def test_decision_tree_plot_detects_children_disappearing_during_traversal():
    from numcompute_stream.tree import _Node

    leaf = _Node(np.array([1, 0]))

    class ChangingNode:
        def __init__(self):
            self.left_reads = 0
            self.right = leaf

        @property
        def is_leaf(self):
            return False

        @property
        def left(self):
            self.left_reads += 1
            return leaf if self.left_reads <= 2 else None

    tree = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
    tree._root = ChangingNode()
    with pytest.raises(RuntimeError, match="without two children"):
        plot_decision_tree(tree, show=False)
