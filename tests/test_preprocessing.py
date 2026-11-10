"""Check scaling, missing-value imputation, and categorical encoding."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    ColumnSelector,
    Imputer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


# StandardScaler


def test_standard_scaler_normal():
    """Standardise values so their mean is zero and standard deviation is one."""
    X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.mean(X_scaled), 0.0)
    assert np.isclose(np.std(X_scaled), 1.0)


def test_standard_scaler_constant():
    """A constant feature should become zeros without dividing by zero."""
    X = np.array([5.0, 5.0, 5.0, 5.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert scaler.scale_ == 1.0
    np.testing.assert_allclose(X_scaled, np.zeros_like(X))


def test_standard_scaler_separate_train_test():
    """Transform new values with statistics learned from the training data."""
    X_train = np.array([0.0, 10.0])  # mean = 5.0, std = 5.0
    X_test = np.array([15.0, 25.0])

    scaler = StandardScaler(epsilon=0.0)
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.testing.assert_allclose(X_test_scaled, np.array([2.0, 4.0]))


def test_standard_scaler_unfitted_error():
    """Reject transforming data before the scaler has been fitted."""
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted yet"):
        scaler.transform(np.array([1.0, 2.0, 3.0]))


# MinMaxScaler


def test_minmax_scaler_default_range():
    """Scale the observed minimum and maximum to the default range."""
    X = np.array([-10.0, 0.0, 10.0, 30.0])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.min(X_scaled), 0.0)
    assert np.isclose(np.max(X_scaled), 1.0)
    np.testing.assert_allclose(X_scaled, np.array([0.0, 0.25, 0.5, 1.0]))


def test_minmax_scaler_custom_range():
    """Allow scaling into a caller-selected range."""
    X = np.array([0.0, 5.0, 10.0])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled, np.array([-1.0, 0.0, 1.0]))


def test_minmax_scaler_constant():
    """A constant feature should map to the lower end of the selected range."""
    X = np.array([7.0, 7.0, 7.0])
    scaler = MinMaxScaler(feature_range=(2.0, 5.0))
    X_scaled = scaler.fit_transform(X)

    assert scaler.data_range_ == 1.0
    np.testing.assert_allclose(X_scaled, np.array([2.0, 2.0, 2.0]))


# Imputer


def test_imputer_mean_nan():
    """Replace missing values with the mean of observed values."""
    X = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    imputer = Imputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([1.0, 3.0, 3.0, 3.0, 5.0]))


def test_imputer_median_nan():
    """Use the median when outliers could skew the replacement value."""
    X = np.array([1.0, 2.0, 100.0, np.nan])
    imputer = Imputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Median of [1.0, 2.0, 100.0] is 2.0
    np.testing.assert_allclose(X_imputed, np.array([1.0, 2.0, 100.0, 2.0]))


def test_imputer_custom_missing_values_placeholder():
    """Recognise a caller-provided marker for missing numeric values."""
    X = np.array([10.0, -999.0, 20.0, -999.0])
    imputer = Imputer(strategy="mean", missing_values=-999.0)
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([10.0, 15.0, 20.0, 15.0]))


def test_imputer_recognizes_string_and_none_missing_values():
    values = np.array(["  NaN ", None, "kept"], dtype=object)
    imputer = Imputer(strategy="constant", fill_value="missing")

    np.testing.assert_array_equal(
        imputer.fit_transform(values), ["missing", "missing", "kept"]
    )


def test_imputer_constant_and_all_missing_fallback():
    """Fall back to fill_value when a column has no observed values."""
    X_all_nan = np.array([np.nan, np.nan, np.nan])
    imputer = Imputer(strategy="mean", fill_value=-1.0)
    X_imputed = imputer.fit_transform(X_all_nan)

    np.testing.assert_allclose(X_imputed, np.array([-1.0, -1.0, -1.0]))


def test_imputer_invalid_strategy():
    """Reject strategies that the imputer does not support."""
    with pytest.raises(ValueError, match="Strategy must be"):
        Imputer(strategy="most_frequent")


# One-hot encoding and input edge cases


def test_onehot_encoder_normal():
    """Turn integer categories into one binary indicator column per category."""
    X = np.array([0, 2, 1, 0])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert X_encoded.shape == (4, 3)
    np.testing.assert_array_equal(X_encoded, expected)


def test_onehot_encoder_supports_object_integer_categories():
    values = np.array([10, 1000000, 10], dtype=object)

    encoded = OneHotEncoder().fit_transform(values)

    assert encoded.shape == (3, 2)
    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1], [1, 0]])


def test_onehot_encoder_vectorizes_string_category_lookup():
    encoder = OneHotEncoder().fit(["red", "blue", "red"])

    encoded = encoder.transform(["blue", "red"])

    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(["green"])


def test_onehot_encoder_invalid_inputs():
    """Reject unsupported numeric categories and categories not seen during fitting."""
    encoder = OneHotEncoder()

    # Categories must be integers, not values that only happen to look numeric.
    with pytest.raises(ValueError, match="integer categorical data"):
        encoder.fit(np.array([0.0, 1.5, 2.0]))

    # Negative values are outside the encoder's supported category range.
    with pytest.raises(ValueError, match="non-negative integers"):
        encoder.fit(np.array([-1, 0, 1]))

    # Transform cannot assign a column to a category that was never learned.
    encoder.fit(np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(np.array([0, 1, 2]))

    encoder.fit(np.array(["red", "blue"]))
    with pytest.raises(ValueError, match="string categorical data"):
        encoder.transform(np.array([1, 2]))


def test_preprocessing_non_contiguous_and_invalid_dimensions():
    """Handle strided input while still rejecting arrays with multiple dimensions."""
    # This view skips every other item rather than copying into a new array.
    base = np.array([10.0, 99.0, 20.0, 99.0, 30.0, 99.0])
    X_strided = base[::2]  # [10.0, 20.0, 30.0]
    assert not X_strided.flags["C_CONTIGUOUS"]

    scaler = MinMaxScaler()
    np.testing.assert_allclose(
        scaler.fit_transform(X_strided), np.array([0.0, 0.5, 1.0])
    )

    # Feature matrices are accepted; higher-dimensional input is not.
    X_3d = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="1D or 2D"):
        scaler.fit(X_3d)


# Streaming (partial_fit) behaviour


@pytest.mark.parametrize("n_chunks", [1, 2, 5, 13])
def test_standard_scaler_partial_fit_equals_batch_fit(n_chunks):
    values = np.random.default_rng(n_chunks).normal(50, 7, size=200)
    streamed = StandardScaler()
    for chunk in np.array_split(values, n_chunks):
        streamed.partial_fit(chunk)
    batch = StandardScaler().fit(values)
    assert streamed.mean_ == pytest.approx(batch.mean_)
    assert streamed.var_ == pytest.approx(batch.var_)
    assert streamed.n_samples_seen_ == values.size


def test_standard_scaler_combines_chunks_and_zero_variance_correctly():
    scaler = StandardScaler(epsilon=0)
    scaler.partial_fit([5.0, 5.0])
    assert scaler.scale_ == 1.0  # constant so far: no division by zero
    scaler.partial_fit([1.0, 9.0])
    assert scaler.mean_ == pytest.approx(5.0)
    assert scaler.var_ == pytest.approx(8.0)
    assert scaler.scale_ == pytest.approx(np.sqrt(8.0))


def test_standard_scaler_partial_fit_is_stable_for_large_offsets():
    values = 1e9 + np.array([1.0, 2.0, 3.0, 4.0])
    scaler = StandardScaler()
    for value in values:
        scaler.partial_fit([value])
    assert scaler.var_ == pytest.approx(1.25)


def test_standard_scaler_ema_tracks_drift_faster_than_exact_mode():
    exact, ema = StandardScaler(), StandardScaler(ema_alpha=0.5)
    for scaler in (exact, ema):
        scaler.partial_fit(np.zeros(100))
        scaler.partial_fit(np.full(10, 10.0))
    assert exact.mean_ == pytest.approx(10 * 10 / 110)
    assert ema.mean_ == pytest.approx(5.0)
    assert ema.var_ == pytest.approx(25.0)  # mixture variance of 0 and 10


def test_standard_scaler_rejects_invalid_ema_alpha():
    with pytest.raises(ValueError, match="ema_alpha"):
        StandardScaler(ema_alpha=0.0)


def test_partial_fit_ignores_empty_chunks():
    scaler = StandardScaler().partial_fit([1.0, 3.0])
    scaler.partial_fit([])
    assert scaler.n_samples_seen_ == 2


def test_standard_scaler_rejects_nan_chunk():
    with pytest.raises(ValueError, match="NaN or infinity"):
        StandardScaler().partial_fit([1.0, np.nan])


def test_minmax_scaler_updates_chunk_extrema():
    scaler = MinMaxScaler().partial_fit([2, 4])
    scaler.partial_fit([-2, 10])
    assert (scaler.data_min_, scaler.data_max_) == (-2.0, 10.0)
    np.testing.assert_allclose(scaler.transform([-2, 10]), [0, 1])


def test_imputer_updates_running_mean():
    imputer = Imputer().partial_fit([1.0, np.nan, 3.0])
    imputer.partial_fit([5.0, np.nan])
    assert imputer.statistics_ == pytest.approx(3.0)


def test_imputer_median_partial_fit_equals_batch():
    imputer = Imputer(strategy="median").partial_fit([1.0, np.nan, 100.0])
    imputer.partial_fit([2.0, 3.0])
    assert imputer.statistics_ == pytest.approx(np.median([1.0, 100.0, 2.0, 3.0]))


def test_imputer_ignores_all_missing_followup_chunk():
    imputer = Imputer().partial_fit([2.0, 4.0])
    imputer.partial_fit([np.nan])
    assert imputer.statistics_ == pytest.approx(3.0)


def test_imputer_learns_once_observations_arrive():
    imputer = Imputer(fill_value=-1.0).partial_fit([np.nan, np.nan])
    assert imputer.statistics_ == -1.0
    imputer.partial_fit([4.0, np.nan])
    assert imputer.statistics_ == pytest.approx(4.0)


def test_mean_imputer_keeps_constant_size_state_over_many_chunks():
    imputer = Imputer(strategy="mean").partial_fit([1.0, 3.0])
    for _ in range(100):
        imputer.partial_fit([5.0])
    assert imputer.statistics_ == pytest.approx((4 + 500) / 102)
    assert imputer._observed_values_.size == 0
    assert imputer.n_samples_seen_ == 102


def test_mean_imputer_avoids_overflow_for_large_finite_values():
    imputer = Imputer(strategy="mean").partial_fit([1e308, 1e308])
    imputer.partial_fit([1e308])
    assert imputer.statistics_ == pytest.approx(1e308)


def test_imputer_rejects_infinite_observed_values():
    with pytest.raises(ValueError, match="infinite observed"):
        Imputer(strategy="mean").partial_fit([1.0, np.inf])


def test_imputer_reads_numbers_from_object_arrays():
    values = np.array([1.0, None, 3.0], dtype=object)
    np.testing.assert_allclose(Imputer().fit_transform(values).astype(float), [1, 2, 3])


def test_imputer_preserves_constant_fallback_for_string_chunks():
    imputer = Imputer(fill_value="missing").partial_fit(["red", ""])
    imputer.partial_fit(["blue", ""])
    np.testing.assert_array_equal(imputer.transform(["", "blue"]), ["missing", "blue"])


def test_onehot_encoder_appends_new_categories_without_changing_old_columns():
    encoder = OneHotEncoder().partial_fit(["blue", "red"])
    before = encoder.transform(["blue", "red"])
    encoder.partial_fit(["amber"])
    after = encoder.transform(["blue", "red"])
    np.testing.assert_array_equal(after[:, :2], before)
    assert after.shape == (2, 3)


def test_onehot_encoder_rejects_unknown_categories_until_partial_fit():
    encoder = OneHotEncoder().fit(["a", "b"])
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(["c"])
    encoder.partial_fit(["c"])
    assert encoder.transform(["c"]).shape == (1, 3)


def test_onehot_encoder_can_ignore_unknown_categories():
    encoder = OneHotEncoder(handle_unknown="ignore").fit(["a", "b"])
    np.testing.assert_array_equal(encoder.transform(["b", "zzz"]), [[0, 1], [0, 0]])


def test_onehot_encoder_rejects_invalid_handle_unknown():
    with pytest.raises(ValueError, match="handle_unknown"):
        OneHotEncoder(handle_unknown="drop")


def test_onehot_partial_fit_rejects_dtype_change():
    encoder = OneHotEncoder().partial_fit([0, 1])
    with pytest.raises(ValueError, match="integer"):
        encoder.partial_fit(["a"])


def test_column_selector_selects_and_casts_columns():
    table = np.array([[1, "a", "2.5"], [0, "b", "nan"]], dtype=object)
    selector = ColumnSelector([0, 2], dtype=float)
    selected = selector.fit_transform(table)
    assert selected.dtype == float
    np.testing.assert_allclose(selected, [[1, 2.5], [0, np.nan]])


@pytest.mark.parametrize("columns", [[], [[0]], ["a"]])
def test_column_selector_rejects_invalid_columns(columns):
    with pytest.raises(ValueError, match="columns"):
        ColumnSelector(columns)


def test_column_selector_rejects_out_of_range_column():
    with pytest.raises(ValueError, match="out of range"):
        ColumnSelector([3]).fit(np.zeros((2, 2)))


# Feature matrices (2D input) and numerical boundaries


@pytest.fixture
def matrix():
    return np.array([[1.0, 10.0, 5.0], [3.0, 20.0, 5.0], [5.0, 60.0, 5.0]])


def test_standard_scaler_scales_each_column_of_a_matrix(matrix):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    assert scaled.shape == matrix.shape
    np.testing.assert_allclose(scaled[:, :2].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled[:, :2].std(axis=0), 1.0)
    np.testing.assert_allclose(scaled[:, 2], 0.0)  # constant column
    np.testing.assert_allclose(scaler.mean_, [3.0, 30.0, 5.0])
    np.testing.assert_allclose(scaler.scale_[2], 1.0)


def test_standard_scaler_matrix_partial_fit_equals_batch(matrix):
    data = np.vstack([matrix, matrix * 2 + 1])
    streamed = StandardScaler().partial_fit(data[:2]).partial_fit(data[2:])
    batch = StandardScaler().fit(data)
    np.testing.assert_allclose(streamed.var_, batch.var_)
    np.testing.assert_allclose(streamed.transform(data), batch.transform(data))


def test_scaler_rejects_changed_feature_count(matrix):
    scaler = StandardScaler().fit(matrix)
    with pytest.raises(ValueError, match="fitted with 3"):
        scaler.transform(matrix[:, :2])
    with pytest.raises(ValueError, match="fitted with 3"):
        scaler.partial_fit(matrix[:, :2])


@pytest.mark.parametrize("spread", [1e-12, 1e-9, 1e-6])
def test_standard_scaler_does_not_underscale_tiny_features(spread):
    values = np.array([-1.0, 0.0, 1.0]) * spread
    scaled = StandardScaler().fit_transform(values)
    assert np.std(scaled) == pytest.approx(1.0)


def test_standard_scaler_treats_rounding_level_variation_as_constant():
    values = 1e8 + np.array([0.0, 1.5e-8, 0.0])  # below float64 resolution of 1e8
    scaler = StandardScaler().fit(values)
    assert scaler.scale_ == 1.0


def test_minmax_scaler_scales_matrix_columns_independently(matrix):
    scaled = MinMaxScaler().fit_transform(matrix)
    np.testing.assert_allclose(scaled[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(scaled[:, 1], [0.0, 0.2, 1.0])
    np.testing.assert_allclose(scaled[:, 2], 0.0)


@pytest.mark.parametrize("feature_range", [(1.0, 1.0), (2.0, 1.0), (0.0, np.inf), (0.0,)])
def test_minmax_scaler_rejects_invalid_feature_range(feature_range):
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=feature_range)


def test_minmax_scaler_can_clip_values_outside_seen_range():
    scaler = MinMaxScaler(clip=True).fit([0.0, 10.0])
    np.testing.assert_allclose(scaler.transform([-5.0, 5.0, 15.0]), [0.0, 0.5, 1.0])
    unclipped = MinMaxScaler().fit([0.0, 10.0]).transform([15.0])
    np.testing.assert_allclose(unclipped, [1.5])


def test_minmax_scaler_handles_range_beyond_float_limit():
    scaler = MinMaxScaler().fit([-1e308, 0.0, 1e308])
    scaled = scaler.transform([-1e308, 0.0, 1e308])
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])


def test_imputer_fills_each_matrix_column_with_its_own_statistic():
    X = np.array([[1.0, np.nan], [np.nan, 10.0], [3.0, 30.0]])
    imputer = Imputer().fit(X)
    np.testing.assert_allclose(imputer.statistics_, [2.0, 20.0])
    np.testing.assert_allclose(imputer.transform(X), [[1, 20], [2, 10], [3, 30]])


def test_imputer_matrix_partial_fit_updates_each_column():
    imputer = Imputer().partial_fit(np.array([[1.0, 2.0]]))
    imputer.partial_fit(np.array([[3.0, np.nan]]))
    np.testing.assert_allclose(imputer.statistics_, [2.0, 2.0])
    np.testing.assert_array_equal(imputer.n_samples_seen_, [2, 1])


def test_onehot_encoder_encodes_matrix_columns_with_stable_growth():
    encoder = OneHotEncoder().fit(np.array([["a", "x"], ["b", "y"]]))
    assert encoder.categories_ == 4
    encoder.partial_fit(np.array([["c", "x"]]))
    encoded = encoder.transform(np.array([["a", "x"], ["c", "y"]]))
    np.testing.assert_array_equal(encoded, [[1, 0, 1, 0, 0], [0, 0, 0, 1, 1]])


def test_matrix_fitted_encoder_rejects_1d_input():
    encoder = OneHotEncoder().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="2D matrix"):
        encoder.transform(["a"])


def test_imputer_detects_missing_text_values_without_python_loops():
    values = np.array([None, np.nan, "", "  NaN ", "nan", "None", "kept", 0, b""], dtype=object)
    mask = Imputer(strategy="constant", fill_value="?")._get_missing_mask(values)
    # "None" is ordinary text; only the None object itself is missing.
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 1, 0, 0, 0, 1])


def test_scaler_type_error_reports_received_dtype():
    with pytest.raises(TypeError, match="Received dtype <U1"):
        StandardScaler().fit(np.array(["a", "b"]))
