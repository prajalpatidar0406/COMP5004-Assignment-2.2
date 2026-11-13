"""Check ranking and percentile behavior, including mixed and invalid values."""

import numpy as np
import pytest
from numcompute_stream.rank import Ranker, StreamingRanker


@pytest.fixture
def ranker():
    return Ranker()


def test_rank_average_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])
    assert np.allclose(result, expected)


def test_rank_dense_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="dense")
    expected = np.array([1, 2, 2, 3])
    assert np.allclose(result, expected)


def test_rank_ordinal_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="ordinal")
    expected = np.array([1, 2, 3, 4])
    assert np.allclose(result, expected)


def test_rank_ordinal_preserves_order_of_ties(ranker):
    result = ranker.rank([2, 1, 2, 1], method="ordinal")
    np.testing.assert_array_equal(result, [3, 1, 4, 2])


def test_rank_mixed_types(ranker):
    data = [10, "20", "abc", None, 5.5]
    result = ranker.rank(data, method="average")
    assert len(result) == 5
    # Values that cannot be converted are treated as missing and ranked first.
    assert result[2] == 1.0
    assert result[3] == 1.0


def test_rank_with_nan(ranker):
    data = [10, np.nan, 5]
    result = ranker.rank(data, method="dense")
    assert len(result) == 3
    assert result[1] == 1  # Missing entries receive the lowest rank.


def test_rank_invalid_method(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([1, 2, 3], method="weird")
    assert "Invalid ranking method" in str(exc.value)


def test_rank_invalid_data_none(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank(None)
    assert "cannot be None" in str(exc.value)


def test_rank_invalid_data_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank("hello")
    assert "single string" in str(exc.value)


def test_rank_invalid_data_multidimensional(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([[1, 2], [3, 4]])
    assert "must be 1D" in str(exc.value)


def test_rank_empty_array(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([])
    assert "empty" in str(exc.value)


def test_percentile_linear(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 50, interpolation="linear")
    assert result == 25.0


def test_percentile_lower(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="lower")
    assert result == 10


def test_percentile_higher(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="higher")
    assert result == 20


def test_percentile_midpoint(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="midpoint")
    assert result == 15


def test_percentile_mixed_types(ranker):
    data = [10, "20", "abc", None, 5.5]
    result = ranker.percentile(data, 50)
    assert np.isclose(result, np.median([10, 20, 5.5]))


def test_percentile_with_nan(ranker):
    data = [10, np.nan, 20]
    result = ranker.percentile(data, 50)
    assert result == 15.0


@pytest.mark.parametrize(("q", "expected"), [(0, 10.0), (100, 40.0)])
def test_percentile_at_endpoints(ranker, q, expected):
    assert ranker.percentile([10, 20, 30, 40], q) == expected


def test_percentile_rejects_data_with_no_numeric_values(ranker):
    with pytest.raises(ValueError, match="no valid numeric values"):
        ranker.percentile(["invalid", None], 50)


def test_percentile_invalid_interpolation(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], 50, interpolation="weird")
    assert "Invalid interpolation" in str(exc.value)


def test_percentile_invalid_q_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], "abc")
    assert "numeric" in str(exc.value)


def test_percentile_invalid_q_range(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], 150)
    assert "between 0 and 100" in str(exc.value)


def test_percentile_invalid_data_none(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile(None, 50)
    assert "cannot be None" in str(exc.value)


def test_percentile_invalid_data_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile("hello", 50)
    assert "single string" in str(exc.value)


def test_percentile_invalid_data_multidimensional(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([[1, 2], [3, 4]], 50)
    assert "must be 1D" in str(exc.value)


def test_percentile_empty_array(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([], 50)
    assert "empty" in str(exc.value)

# Streaming ranker


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint"])
def test_streaming_percentile_equals_batch_percentile(interpolation):
    values = np.random.default_rng(4).integers(0, 50, 101).astype(float)
    ranker = StreamingRanker()
    for chunk in np.array_split(values, 6):
        ranker.partial_fit(chunk)
    for q in (0, 10, 37.5, 50, 99, 100):
        assert ranker.percentile(q, interpolation) == pytest.approx(
            Ranker().percentile(values, q, interpolation)
        )


def test_streaming_percentile_rank_splits_ties_evenly():
    ranker = StreamingRanker().partial_fit([1, 2, 2, 3])
    np.testing.assert_allclose(ranker.percentile_rank([0, 2, 5]), [0.0, 50.0, 100.0])


def test_streaming_ranker_keeps_reference_sorted_across_chunks():
    ranker = StreamingRanker().partial_fit([5, 1]).update([3, 2, 4])
    np.testing.assert_array_equal(ranker._sorted, [1, 2, 3, 4, 5])


def test_streaming_ranker_counts_and_ignores_missing_values():
    ranker = StreamingRanker().partial_fit([1.0, np.nan, "x", 3.0])
    assert (ranker.n_seen_, ranker.n_missing_) == (4, 2)
    assert ranker.percentile(50) == pytest.approx(2.0)
    assert np.isnan(ranker.percentile_rank([np.nan])[0])


def test_streaming_ranker_ignores_empty_chunks_and_needs_data():
    ranker = StreamingRanker().partial_fit([])
    with pytest.raises(ValueError, match="No valid values"):
        ranker.percentile(50)
    with pytest.raises(ValueError, match="No valid values"):
        ranker.percentile_rank([1])


def test_streaming_ranker_rejects_2d_chunks_and_resets():
    with pytest.raises(ValueError, match="1D"):
        StreamingRanker().partial_fit([[1, 2]])
    ranker = StreamingRanker().partial_fit([1, 2])
    ranker.reset()
    assert ranker.n_seen_ == 0 and ranker._sorted.size == 0
