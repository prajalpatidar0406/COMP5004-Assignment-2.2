import numpy as np
import pytest

from numcompute_stream.stats import (
    histogram,
    maximum,
    mean,
    median,
    minimum,
    percentile,
    quantile,
    std,
    StreamingHistogram,
    StreamingStats,
    update_stats,
)


def test_mean_of_a_simple_list() -> None:
    assert mean([1, 2, 3, 4]) == pytest.approx(2.5)


def test_mean_reduces_along_the_requested_axis() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])

    np.testing.assert_allclose(mean(data, axis=0), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(mean(data, axis=1), [2.0, 5.0])


def test_mean_propagates_nan_unless_skipna_is_set() -> None:
    assert np.isnan(mean([1, np.nan, 3]))
    assert mean([1, np.nan, 3], skipna=True) == pytest.approx(2.0)


def test_median_handles_odd_and_even_lengths() -> None:
    assert median([3, 1, 2]) == pytest.approx(2.0)
    assert median([4, 1, 3, 2]) == pytest.approx(2.5)


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_matches_numpy_for_each_ddof(ddof: int) -> None:
    data = [1, 2, 3, 4]

    assert std(data, ddof=ddof) == pytest.approx(np.std(data, ddof=ddof))


def test_minimum_and_maximum_of_a_simple_list() -> None:
    assert minimum([3, 1, 2]) == 1
    assert maximum([3, 1, 2]) == 3


def test_quantile_and_percentile_match_known_values() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    np.testing.assert_allclose(quantile([1, 2, 3, 4], [0.25, 0.75]), [1.75, 3.25])
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_histogram_returns_counts_and_bin_edges() -> None:
    counts, edges = histogram([1, 2, 2, 3, 3, 3], bins=[0, 2, 4])

    np.testing.assert_array_equal(counts, [1, 5])
    np.testing.assert_allclose(edges, [0, 2, 4])


def test_invalid_and_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])

    with pytest.raises(TypeError, match="real numeric values"):
        mean(["a", "b"])

    with pytest.raises(ValueError, match="ddof must be a non-negative integer"):
        std([1, 2, 3], ddof=-1)

    with pytest.raises(ValueError, match="q must be between 0 and 1"):
        quantile([1, 2, 3], 2)

    with pytest.raises(ValueError, match="infinite values"):
        histogram([1, np.inf])

# Streaming statistics


@pytest.fixture
def stream_data():
    rng = np.random.default_rng(7)
    return rng.normal(size=(1000, 3)) * [1.0, 10.0, 100.0] + [0.0, 5.0, -3.0]


@pytest.mark.parametrize("n_chunks", [1, 3, 17])
def test_streaming_moments_match_batch_for_any_chunking(stream_data, n_chunks):
    stats = StreamingStats()
    for chunk in np.array_split(stream_data, n_chunks):
        stats.update(chunk)
    np.testing.assert_allclose(stats["mean"], stream_data.mean(axis=0))
    np.testing.assert_allclose(stats["variance"], stream_data.var(axis=0))
    np.testing.assert_allclose(stats["std"], stream_data.std(axis=0))
    np.testing.assert_allclose(stats["min"], stream_data.min(axis=0))
    np.testing.assert_allclose(stats["max"], stream_data.max(axis=0))
    np.testing.assert_array_equal(stats["count"], [1000, 1000, 1000])


def test_streaming_sample_variance_uses_ddof():
    stats = StreamingStats(ddof=1).update([1.0, 2.0]).update([3.0, 4.0])
    assert stats["variance"] == pytest.approx(np.var([1, 2, 3, 4], ddof=1))


def test_streaming_quantiles_are_exact_until_reservoir_fills(stream_data):
    stats = StreamingStats(quantiles=[0.1, 0.5, 0.9], reservoir_size=1000)
    for chunk in np.array_split(stream_data, 9):
        stats.update(chunk)
    assert stats["quantiles_exact"] is True
    np.testing.assert_allclose(
        stats["quantiles"], np.quantile(stream_data, [0.1, 0.5, 0.9], axis=0)
    )


def test_streaming_quantiles_approximate_with_bounded_memory():
    values = np.random.default_rng(0).uniform(size=50_000)
    stats = StreamingStats(reservoir_size=2000, random_state=1)
    for chunk in np.array_split(values, 25):
        stats.update(chunk)
    assert stats["quantiles_exact"] is False
    assert stats._reservoir.shape == (2000, 1)
    np.testing.assert_allclose(stats["quantiles"], [0.25, 0.5, 0.75], atol=0.03)


def test_streaming_histogram_counts_are_exact_after_range_growth(stream_data):
    stats = StreamingStats(bins=8)
    for chunk in np.array_split(stream_data, 11):
        stats.update(chunk)
    for feature in range(3):
        expected, _ = np.histogram(stream_data[:, feature], bins=stats["bin_edges"][feature])
        np.testing.assert_array_equal(stats["histogram"][feature], expected)


@pytest.mark.parametrize("bins", [4, 5])
def test_histogram_doubling_keeps_counts_exact_for_odd_and_even_bins(bins):
    chunks = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 10.0]), np.array([-20.0, 3.0])]
    histogram = StreamingHistogram(bins=bins)
    for chunk in chunks:
        histogram.update(chunk)
    expected, _ = np.histogram(np.concatenate(chunks), bins=histogram.edges)
    np.testing.assert_array_equal(histogram.counts, expected)
    assert histogram.edges[0] <= -20 and histogram.edges[-1] >= 10


def test_histogram_first_chunk_matches_numpy():
    values = np.array([0.0, 1.0, 1.0, 2.5, 4.0])
    counts, edges = np.histogram(values, bins=4)
    histogram = StreamingHistogram(bins=4).update(values)
    np.testing.assert_array_equal(histogram.counts, counts)
    np.testing.assert_allclose(histogram.edges, edges)


def test_histogram_constant_first_chunk_gets_unit_range():
    histogram = StreamingHistogram(bins=2).update([3.0, 3.0])
    np.testing.assert_allclose(histogram.edges, [2.5, 3.0, 3.5])
    assert histogram.counts.sum() == 2


def test_fixed_range_histogram_counts_out_of_range_values():
    histogram = StreamingHistogram(bins=2, range=(0.0, 1.0)).update([0.1, 0.9, 5.0, -1.0])
    np.testing.assert_array_equal(histogram.counts, [1, 1])
    assert histogram.n_out_of_range_ == 2


def test_histogram_rejects_invalid_range():
    with pytest.raises(ValueError, match="range"):
        StreamingHistogram(range=(1.0, 1.0))


def test_streaming_stats_combine_chunks_and_exact_quantiles():
    state = update_stats([1, 2, 3])
    update_stats([4, 5], state)
    assert state["count"] == 5
    assert state["mean"] == pytest.approx(3.0)
    assert state["variance"] == pytest.approx(2.0)
    np.testing.assert_allclose(state["quantiles"], [2, 3, 4])
    assert int(np.sum(state["histogram"])) == 5


def test_streaming_stats_can_retain_a_sliding_window():
    state = update_stats([1, 2, 3], window_size=3)
    update_stats([4, 5], state)
    assert state["count"] == 3
    assert state["mean"] == pytest.approx(4.0)
    np.testing.assert_allclose(state["quantiles"], [3.5, 4.0, 4.5])
    assert state["histogram"].sum() == 3
    assert state["n_rows"] == 5


def test_streaming_stats_can_ignore_nan_values():
    state = update_stats([1.0, np.nan, 3.0], skipna=True)
    assert state["mean"] == pytest.approx(2.0)
    assert state["count"] == 2


def test_streaming_stats_handle_per_feature_nan_counts():
    stats = StreamingStats(skipna=True).update([[1.0, np.nan], [3.0, 4.0]])
    np.testing.assert_array_equal(stats["count"], [2, 1])
    np.testing.assert_allclose(stats["mean"], [2.0, 4.0])


def test_streaming_stats_reject_nan_by_default_and_infinity_always():
    with pytest.raises(ValueError, match="skipna"):
        update_stats([1.0, np.nan])
    with pytest.raises(ValueError, match="infinity"):
        update_stats([1.0, np.inf], skipna=True)


def test_streaming_stats_before_data_are_nan():
    result = StreamingStats().result()
    assert result["count"] == 0
    assert np.isnan(result["mean"]) and np.isnan(result["variance"])


def test_streaming_stats_reset_clears_state():
    stats = StreamingStats().update([1.0, 2.0])
    stats.reset()
    assert stats["count"] == 0


def test_streaming_stats_reject_feature_count_change():
    stats = StreamingStats().update([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expected 2"):
        stats.update([[1.0, 2.0, 3.0]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 0}, "bins"),
        ({"window_size": 0}, "window_size"),
        ({"quantiles": [1.5]}, "quantiles"),
        ({"reservoir_size": 0}, "reservoir_size"),
        ({"ddof": -1}, "ddof"),
    ],
)
def test_streaming_stats_reject_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StreamingStats(**kwargs)


def test_update_stats_rejects_non_numeric_chunks_and_bad_state():
    with pytest.raises(TypeError, match="numeric"):
        update_stats(["a"])
    with pytest.raises(TypeError, match="StreamingStats"):
        update_stats([1.0], state={})


@pytest.mark.parametrize("seed", range(40))
def test_histogram_stays_exact_on_random_streams(seed):
    """Random integer streams often land exactly on bin edges, so any drift would show."""
    rng = np.random.default_rng(seed)
    bins = int(rng.integers(1, 12))
    chunks = [
        rng.integers(-100, 100, size=rng.integers(1, 40)).astype(float)
        for _ in range(rng.integers(1, 15))
    ]
    histogram = StreamingHistogram(bins=bins)
    for chunk in chunks:
        histogram.update(chunk)
    expected, _ = np.histogram(np.concatenate(chunks), bins=histogram.edges)
    np.testing.assert_array_equal(histogram.counts, expected)
