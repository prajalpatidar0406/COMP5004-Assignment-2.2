"""Tests for the buffered streaming DecisionTreeClassifier."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTree, DecisionTreeClassifier


@pytest.fixture
def interaction_data():
    """Two informative features that interact, plus two noise features."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0.5)).astype(int)
    return X, y


# --- Batch training ------------------------------------------------------- #


def test_tree_learns_a_binary_threshold_with_gini():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_tree_places_threshold_at_midpoint_between_values():
    model = DecisionTreeClassifier().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert model._root.threshold == pytest.approx(1.5)
    np.testing.assert_array_equal(model.predict([[1.4], [1.6]]), [0, 1])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_solves_interaction_with_both_criteria(interaction_data, criterion):
    X, y = interaction_data
    model = DecisionTreeClassifier(max_depth=4, criterion=criterion).fit(X[:400], y[:400])
    assert model.score(X[400:], y[400:]) > 0.9


def test_tree_respects_max_depth(interaction_data):
    X, y = interaction_data
    for depth in (1, 2, 3):
        assert DecisionTreeClassifier(max_depth=depth).fit(X, y).depth_ <= depth


def test_tree_with_depth_one_is_a_stump():
    model = DecisionTreeClassifier(max_depth=1).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not model._root.is_leaf
    assert model._root.left.is_leaf and model._root.right.is_leaf
    assert model.n_leaves_ == 2


def test_tree_respects_min_samples_split():
    model = DecisionTreeClassifier(min_samples_split=5).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert model._root.is_leaf


def test_tree_unlimited_depth_does_not_hit_recursion_limit():
    X = np.arange(3000, dtype=float).reshape(-1, 1)
    y = np.arange(3000) % 2  # alternating labels force a maximally deep tree
    model = DecisionTreeClassifier().fit(X, y)
    assert model.depth_ > 1000
    np.testing.assert_array_equal(model.predict(X), y)


def test_tree_uses_smallest_label_for_a_majority_tie():
    model = DecisionTreeClassifier().fit([[1], [1]], [1, 0])
    np.testing.assert_array_equal(model.predict([[1]]), [0])


def test_tree_supports_string_labels_and_multiclass():
    X = [[0], [1], [5], [6], [10], [11]]
    y = ["low", "low", "mid", "mid", "high", "high"]
    model = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(model.predict([[0.5], [5.5], [10.5]]), ["low", "mid", "high"])
    np.testing.assert_array_equal(model.classes_, ["high", "low", "mid"])


def test_tree_on_constant_features_predicts_majority():
    model = DecisionTreeClassifier().fit([[1], [1], [1]], [0, 1, 1])
    assert model._root.is_leaf
    np.testing.assert_array_equal(model.predict([[7]]), [1])


def test_predict_proba_rows_sum_to_one_and_match_predict(interaction_data):
    X, y = interaction_data
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (X.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_[probabilities.argmax(axis=1)], model.predict(X))


@pytest.mark.parametrize("max_features", [1, 0.5, "sqrt", "log2"])
def test_tree_feature_subsampling_is_reproducible(interaction_data, max_features):
    X, y = interaction_data
    first = DecisionTreeClassifier(max_features=max_features, random_state=3).fit(X, y)
    second = DecisionTreeClassifier(max_features=max_features, random_state=3).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_tree_score_matches_mean_of_correct_predictions(interaction_data):
    X, y = interaction_data
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert model.score(X, y) == pytest.approx(np.mean(model.predict(X) == y))


def test_tree_predicts_empty_batch_with_zero_predictions():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    assert model.predict(np.empty((0, 1))).shape == (0,)


# --- Streaming / buffered retraining -------------------------------------- #


def test_tree_partial_fit_buffers_until_threshold():
    model = DecisionTreeClassifier(buffer_size=4)
    model.partial_fit([[0], [1]], [0, 0])
    assert model.buffered_samples == 2
    assert model.is_fitted is False
    model.partial_fit([[2], [3]], [1, 1])
    assert model.buffered_samples == 0
    assert model.is_fitted is True
    assert model.retrain_count_ == 1


def test_tree_partial_fit_retrains_on_oversized_chunk():
    model = DecisionTreeClassifier(buffer_size=3)
    model.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert model.retrain_count_ == 1
    assert model.buffered_samples == 0


def test_tree_keeps_previous_model_between_retrains():
    model = DecisionTreeClassifier(buffer_size=2).fit([[0], [1]], [0, 1])
    original = model.predict([[0], [1]])
    model.partial_fit([[4]], [0])
    assert model.buffered_samples == 1
    np.testing.assert_array_equal(model.predict([[0], [1]]), original)


def test_tree_retrains_only_on_buffered_data():
    model = DecisionTreeClassifier(buffer_size=2)
    model.partial_fit([[0], [1]], [0, 1])
    model.partial_fit([[0], [1]], [1, 0])  # the relationship flips
    np.testing.assert_array_equal(model.predict([[0], [1]]), [1, 0])
    assert model.retrain_count_ == 2


def test_tree_partial_fit_with_full_buffer_equals_batch_fit(interaction_data):
    X, y = interaction_data
    streamed = DecisionTreeClassifier(max_depth=4, buffer_size=X.shape[0])
    for X_chunk, y_chunk in zip(np.array_split(X, 7), np.array_split(y, 7)):
        streamed.partial_fit(X_chunk, y_chunk)
    batch = DecisionTreeClassifier(max_depth=4).fit(X, y)
    np.testing.assert_array_equal(streamed.predict(X), batch.predict(X))


def test_tree_fit_discards_pending_buffer():
    model = DecisionTreeClassifier(buffer_size=10)
    model.partial_fit([[0]], [0])
    model.fit([[0], [1]], [0, 1])
    assert model.buffered_samples == 0


def test_tree_pads_older_buffered_rows_when_features_grow():
    model = DecisionTreeClassifier(buffer_size=4)
    model.partial_fit([[1, 0], [0, 1]], [0, 0])
    model.partial_fit([[0, 0, 1], [0, 0, 1]], [1, 1])  # a new trailing column
    assert model.n_features_in_ == 3
    np.testing.assert_array_equal(model.predict([[1, 0, 0], [0, 0, 1]]), [0, 1])


def test_tree_rejects_chunk_narrower_than_seen_data():
    model = DecisionTreeClassifier(buffer_size=10)
    model.partial_fit([[0, 1]], [0])
    with pytest.raises(ValueError, match="features but 2 were seen"):
        model.partial_fit([[0]], [0])


def test_tree_ignores_trailing_features_added_after_last_retrain():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    np.testing.assert_array_equal(model.predict([[0, 9], [1, 9]]), [0, 1])


# --- Validation ----------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": True}, "max_depth"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"max_features": 0}, "max_features"),
        ({"max_features": 1.5}, "max_features"),
        ({"max_features": "half"}, "max_features"),
        ({"buffer_size": 0}, "buffer_size"),
        ({"criterion": "bad"}, "criterion"),
    ],
)
def test_tree_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DecisionTreeClassifier(**kwargs)


@pytest.mark.parametrize(
    ("X", "y", "error", "message"),
    [
        ([[np.nan]], [0], ValueError, "NaN or infinity"),
        ([[np.inf]], [0], ValueError, "NaN or infinity"),
        ([[0], [1]], [0], ValueError, "same number"),
        ([[0], [1]], [[0], [1]], ValueError, "y must be a 1D"),
        (np.empty((0, 1)), [], ValueError, "must not be empty"),
        ([["a"], ["b"]], [0, 1], TypeError, "numeric"),
        (np.zeros((2, 1, 1)), [0, 1], ValueError, "1D or 2D"),
    ],
)
def test_tree_rejects_invalid_training_data(X, y, error, message):
    with pytest.raises(error, match=message):
        DecisionTreeClassifier().fit(X, y)


def test_tree_requires_a_model_before_predicting():
    with pytest.raises(ValueError, match="not been fit"):
        DecisionTreeClassifier().predict([[0]])


def test_tree_rejects_feature_width_mismatch():
    model = DecisionTreeClassifier().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError, match="fewer than the 2"):
        model.predict([[0]])


def test_decision_tree_alias_matches_assignment_brief():
    tree = DecisionTree(max_depth=5, buffer_size=500)
    assert isinstance(tree, DecisionTreeClassifier)
    assert (tree.max_depth, tree.buffer_size) == (5, 500)
