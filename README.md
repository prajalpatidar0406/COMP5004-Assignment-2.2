# NumCompute

NumCompute is a numerical computing toolkit implemented using **plain Python and NumPy**, with Matplotlib for reusable visualisations.

The project implements numerical data loading, preprocessing, sorting and searching, ranking, statistics, evaluation metrics, finite-difference differentiation, and pipeline functionality. **NumCompute-Stream** extends every component to data that arrives in chunks: incremental `partial_fit` preprocessing, a buffered decision tree, streaming statistics and metrics, a chunk-by-chunk `StreamTrainer`, and reusable plots.

External machine learning/deep learning libraries such as **scikit-learn, Pandas and PyTorch are not used**.

---

## Features

- Load CSV files whole or as a bounded-memory stream of chunks.
- Impute, scale and encode features, in batch or incrementally (`partial_fit`), for a single
  feature or a whole `(n_samples, n_features)` matrix.
- Train a depth-limited Gini/entropy **decision tree** with buffered streaming retraining.
- Drive training and evaluation chunk by chunk with **`StreamTrainer`** (test-then-train), which
  logs per-chunk and cumulative metrics, timings and memory.
- Maintain **streaming statistics**: mean, variance, quantiles and exact adaptive histograms,
  optionally over a sliding window.
- Accumulate **streaming metrics** (accuracy, precision, recall, F1, confusion matrix, AUC) and
  rolling-window metrics.
- Plot metrics over time, model comparisons and predictions vs ground truth with Matplotlib.
- Sort, search, rank and select values; estimate gradients and Jacobians.
- Benchmark representative toolkit operations against naive references.

---

## Project Structure

```text
.
├── NumCompute/
│   ├── numcompute/        original batch package, preserved from main
│   └── numcompute_stream/ separate streaming package
│       ├── io.py              CSV loading and chunked iteration
│       ├── preprocessing.py   StandardScaler, MinMaxScaler, Imputer, OneHotEncoder, ColumnSelector
│       ├── pipeline.py        Pipeline, Compose, FeatureUnion (batch + partial_fit)
│       ├── tree.py            DecisionTreeClassifier with buffered retraining
│       ├── stream.py          StreamTrainer, deep_sizeof
│       ├── stats.py           batch statistics + StreamingStats, StreamingHistogram, update_stats
│       ├── metrics.py         batch metrics + streaming metric classes
│       ├── visualise.py       reusable Matplotlib plots
│       ├── benchmarking.py    timing harness, buffer-size and toolkit benchmarks
│       └── sort_search.py, rank.py, optim.py, utils.py, histogram_display.py
├── tests/                 one test file per module, plus:
│   ├── test_reference_checks.py   independent checks against pure-Python references
│   └── test_integration.py        end-to-end CSV → pipeline → trainer → plots
├── demo/
│   ├── quickstart.ipynb   tour of the batch toolkit
│   ├── stream_demo.ipynb  streaming workflow on the Titanic data
│   └── data/titanic.csv
├── docs/DEBUGGING.md      concrete debugging case studies
├── README.md, CODING_STYLE.md, pyproject.toml, requirements*.txt
```

Installing this project makes both import packages available: `numcompute` is the
original batch API, while `numcompute_stream` contains the separate streaming
implementation. The quickstart notebook uses `numcompute`; the stream demo uses
`numcompute_stream`.

When behavior is shared, `numcompute_stream` re-exports the implementation from
`numcompute` rather than maintaining a duplicate. Streaming-specific classes and
algorithms remain in `numcompute_stream`.

---

## Requirements

* Python 3.10 or newer
* NumPy (installed from `requirements.txt`)
* Matplotlib (installed from `requirements.txt`)
* Plain Python standard library where required

The project does not uses any external ML/DL libraries such as:
* scikit-learn
* pandas
* PyTorch

---

## Installation

Clone the repository and move into its root directory:

```bash
git clone https://github.com/prajalpatidar0406/COMP5004-Assignment-2.2.git
cd COMP5004-Assignment-2.2
```

Create and activate a virtual environment.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the package and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install .
python -m pip install jupyter
```

The installed package is named `numcompute_stream` and can be imported from Python scripts or Jupyter notebooks.

---

## Modules

### `io.py` — Data loading

Supports:

* `load_csv(path, ...) -> (header, data)` for complete-table loading
* `iter_csv(path, chunksize=1000, ...)` for bounded-memory iteration
* Custom delimiters, requested or inferred dtypes, and fill/drop missing modes
* Numeric arrays for numeric tables and object arrays for mixed columns

`load_csv` returns the column names and a 2D NumPy array using
`numpy.loadtxt`. Use `iter_csv` to process data in bounded-memory chunks.


---

### `preprocessing.py` — Data preprocessing

| Transformer | Learns | Streaming update (`partial_fit`) | State memory |
|---|---|---|---|
| `StandardScaler(epsilon=0.0, ema_alpha=None)` | mean, variance | exact parallel Welford merge, or exponential moving average for drift | O(features) |
| `MinMaxScaler(feature_range=(0, 1), clip=False)` | min, max | running extrema | O(features) |
| `Imputer(strategy="mean" \| "median" \| "constant")` | replacement value | running mean / exact median | O(1) mean, O(n) median |
| `OneHotEncoder(handle_unknown="error" \| "ignore")` | category vocabulary | new categories appended as **new trailing columns** | O(categories) |
| `ColumnSelector(columns, dtype=None)` | — | stateless | — |

All accept a single feature `(n_samples,)` **or** a matrix `(n_samples, n_features)`; each column
keeps its own statistics. Scalers are vectorised across columns, and learned attributes are
floats for 1D input and per-feature arrays for 2D input.

Numerical boundaries are handled explicitly:

- **Constant features.** A feature counts as constant only if its variance is within
  floating-point rounding error of zero (a relative bound). A feature with a real but tiny spread,
  such as std `1e-12`, is still scaled exactly to unit variance; an additive epsilon would have
  under-scaled it.
- **`MinMaxScaler`.** Rejects invalid ranges (non-finite, or `min >= max`), can clip streamed
  values that fall outside the extrema seen so far, and stays finite when the data spans more
  than the float64 range (e.g. `±1e308`).

```python
scaler = StandardScaler()
for X_chunk in chunks:            # X_chunk: (n_samples, n_features)
    scaler.partial_fit(X_chunk)
X_scaled = scaler.transform(X_new)
```

---

### `tree.py` — Streaming decision tree

`DecisionTreeClassifier(max_depth=None, min_samples_split=2, max_features=None, buffer_size=1000,
criterion="gini", random_state=None)`

- `fit(X, y)` grows a CART tree from scratch. For each node, every candidate feature is sorted
  once, and **all thresholds are scored in one vectorised pass** using cumulative class counts:
  O(m · n log n) per node.
- Thresholds are midpoints between distinct values.
- The tree is grown with an explicit stack, so unlimited-depth trees cannot hit Python's
  recursion limit.
- `partial_fit(X_chunk, y_chunk)` uses **buffered retraining**: chunks are appended to a buffer,
  and once at least `buffer_size` samples are pending, a new tree is trained on the buffer and the
  buffer is cleared. Between retrains, `predict` uses the last trained tree.
- `predict`, `predict_proba` (class mix of each leaf) and `score` are available, plus
  `retrain_count_`, `depth_`, `n_leaves_`, `buffered_samples` and `is_fitted`.
- `max_features` accepts an int, a fraction, `"sqrt"` or `"log2"`, and `random_state` makes the
  sampling reproducible.

```python
tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
for X_chunk, y_chunk in stream:
    tree.partial_fit(X_chunk, y_chunk)
```

`DecisionTree` is an alias for `DecisionTreeClassifier`, so the form used in the
assignment brief, `DecisionTree(max_depth=5, buffer_size=500)`, also works.

---

### `pipeline.py` — Pipeline composition

`Pipeline`, `Compose` and `FeatureUnion` all support `fit`, `transform` and **`partial_fit`**.
`Pipeline.partial_fit` updates every transformer incrementally, transforms the chunk with the
updated state, and calls the final estimator's own `partial_fit`, so a buffered tree buffers.
When an encoder learns a new category mid-stream, its new column is appended at the end of the
output, so earlier columns never change meaning. Use `ColumnSelector` inside a `FeatureUnion` to
route numeric and categorical columns of one table through different preprocessing.

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
pipe.partial_fit(X_chunk, y_chunk)
pipe.predict(X_new); pipe.predict_proba(X_new)
```

---

### `stream.py` — Streaming training and logging

`StreamTrainer(model, pipeline=None, *, metrics=None, logger=None, track_memory=True)` manages a
model, optional preprocessing and logging.

- `fit_chunk(X, y)` updates the pipeline and model with one chunk.
- `score_chunk(X, y)` evaluates and appends a record containing per-chunk and cumulative accuracy,
  every extra metric (per chunk and as `cumulative_<name>`), sample counts, fit/predict time and
  the model's **memory footprint**.
- `run(stream)` performs **prequential (test-then-train)** evaluation, so no chunk is scored by a
  model that has already trained on it.
- `metric_history(name)` returns one logged quantity as an array, ready for plotting.
- Score-based metrics such as `AUC` automatically receive `predict_proba` output. `logger` may be
  a callable or a `logging.Logger`.

```python
trainer = StreamTrainer(pipe, metrics={"f1": F1(), "auc": AUC()})
trainer.run(stream)
trainer.metric_history("cumulative_accuracy")
```

---

### `stats.py` — Descriptive statistics

Batch functions: `mean`, `median`, `std`, `minimum`, `maximum`, `quantile`, `percentile`,
`histogram` (NaN-aware via `skipna`).

Streaming, through `StreamingStats` or the functional `update_stats(X_chunk, state)`. Each chunk
may be 1D or 2D (per-feature statistics):

| Statistic | Algorithm | Memory |
|---|---|---|
| mean, variance, std, min, max | parallel Welford/Chan merge, NaN-aware | O(features) |
| quantiles | uniform reservoir sample (vectorised Algorithm R): exact until `reservoir_size` rows, unbiased afterwards | O(reservoir) |
| histogram | `StreamingHistogram`: fixed bin count; when data leaves the range the bin width doubles and bin pairs merge, so **counts stay exact** without storing data | O(bins) |
| any of the above over a sliding window | `window_size=W`: exact over the last W rows | O(W) |

```python
state = update_stats(X_chunk)            # first chunk
state = update_stats(next_chunk, state)  # later chunks
state["mean"], state["quantiles"], state["histogram"]
```

---

### `metrics.py` — Model evaluation

Batch functions: `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`,
`roc_curve`, `auc` and `roc_auc_score`.

The streaming classes all implement `update(y_true_chunk, y_pred_chunk)`, `result()` and
`reset()`. After any sequence of chunks, `result()` equals the batch metric on the concatenated
data.

| Class | Notes |
|---|---|
| `Accuracy` | O(1) state |
| `Precision`, `Recall`, `F1` | binary (`positive_label`) or `average="micro" \| "macro" \| "weighted"` for multiclass |
| `ConfusionMatrix` | label set grows as new labels appear (existing counts are remapped), or is fixed with `labels=` |
| `AUC` | exact (stores scores); `num_thresholds=T` gives an O(T)-memory binned AUC; `NaN` until both classes are seen |
| `RollingWindowMetric(metric, window_size)` | any metric over the last *k* chunks |

---

### `optim.py` — Numerical differentiation

`grad(f, x, h=1e-5, method="central" | "forward")` and `jacobian(F, x, ...)`.

- The step is `max(h, sqrt(eps)·|x_i|)`: exactly `h` at ordinary magnitudes, and enlarged only
  when `h` would vanish in rounding (with a fixed step, `grad(x**2, 1e20)` used to return 0).
- The difference quotient divides by the step that was actually representable.
- A vanishing step, or a function value that is NaN/infinite, raises `ValueError`.

---

### `visualise.py` — Plots

| Function | Purpose |
|---|---|
| `plot_metric_over_time(metric_values, title, ylabel)` | one metric across chunks (NaN shown as a gap) |
| `compare_models(metric1, metric2, labels)` | two models on the same chunks, e.g. different `buffer_size` values |
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | latest chunk; marks mistakes, works with string labels |
| `plot_decision_tree(tree, feature_names, class_names)` | fitted tree structure with split thresholds and class counts |

All three accept `save_path=` (any Matplotlib format), `show=` (inline display) and `ax=` (draw
into an existing subplot), and return `(figure, axes)`.

---

### `utils.py` — Shared utilities

Distances (`euclidean_distance`, `manhattan_distance`, `cosine_similarity`) rescale their
operands first, so values near the float64 limits neither overflow (`1e200`) nor underflow
(`1e-200`). The module also provides activations, `topk_indices`, `batches`, and the shared
validators `check_positive_int` and `check_supervised_chunk`.

---

### `benchmarking.py` — Performance measurement

- `benchmark`, `compare` and `format_table` time any callable and record the environment.
- `benchmark_streaming_buffer_sizes` measures accuracy and retraining cost per `buffer_size`.
- `benchmark_toolkit` times representative NumCompute operations against the naive approach
  each one replaces. It checks that both give the same result before timing.

---

### `sort_search.py` - Sorting and searching algorithms

Includes:

* Stable sorting
* Multi-key sorting
* `topk(values, k, largest=True, return_indices=True)`
* Top-k selection using `np.argpartition`
* Quickselect
* Binary search

Binary search returns the insertion index and whether the value exists.

For streams, `StreamingTopK(k, largest=True)` keeps the k best values seen so far
in O(k) memory: call `update(chunk)` for each chunk and `result(return_indices=True)`
for the values (best first) and their positions in the stream. Ties go to the value
seen first.

---

### `rank.py` - Ranking and percentile

`rank(data, method="average")` supports the `average`, `dense`, and `ordinal` ranking methods and handles tied values.

```python
rank(data, method="average")
```

Percentile interpolation supports `linear`, `lower`, `higher`, and `midpoint` methods.

```python
percentile(data, q, interpolation="linear")
```

For streams, `StreamingRanker` merges each chunk into a sorted reference with
`partial_fit(chunk)` (or `update(chunk)`). It then answers `percentile(q, interpolation)`
and `percentile_rank(values)` over everything seen so far. Results are exact, so it
keeps all valid values; for bounded memory use `stats.StreamingStats` quantiles.


---

## Running Tests

```bash
python -m pytest
```

Tests are organised one file per module (`test_tree.py`, `test_stream.py`, `test_stats.py`,
`test_metrics.py`, `test_preprocessing.py`, `test_pipeline.py`, `test_visualise.py`,
`test_optim.py`, `test_utils.py`, `test_benchmarking.py`, ...). Two further files add
independent evidence:

- `test_reference_checks.py` checks results against implementations that share no code with
  NumCompute: exact rational arithmetic (`fractions`), the `statistics` and `math` modules, a
  brute-force split search and the pairwise Mann–Whitney definition of AUC.
- `test_integration.py` runs the real Titanic CSV through streaming statistics, a mixed-type
  pipeline, `StreamTrainer`, saved plots and the buffer benchmark.

Streaming components are also tested for **chunking invariance**: several different chunkings of
the same data must give the batch result.

See [docs/DEBUGGING.md](docs/DEBUGGING.md) for concrete debugging case studies (symptom → root
cause → fix → regression test).

---

# Demo

* `demo/quickstart.ipynb` — a tour of the batch toolkit: CSV loading, preprocessing, sorting and
  ranking, statistics, metrics, gradients, pipelines.
* `demo/stream_demo.ipynb` — the streaming workflow. It reads the Titanic CSV in chunks, keeps
  streaming statistics, trains a mixed-type pipeline with a buffered tree using test-then-train,
  logs metrics, plots them, and benchmarks buffer sizes.

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebooks locate the repository root themselves, so they run from either the root or `demo/`.

---

# Benchmark

Run `python -m numcompute_stream.benchmarking --suite toolkit` (or `--suite all`). Each NumCompute
operation is compared with the straightforward approach it replaces, after checking that both
give the same answer.

Measured locally (Python 3.13.9, NumPy 2.3.5, macOS 27.2.1 arm64; median of 3–7 repeats). Rerun on
your machine, because results depend on hardware and load.

| Workload | n | Naive reference | NumCompute | Speed-up |
|---|---:|---|---|---:|
| StandardScaler fit+transform (10 features) | 2,000 | pure-Python per-column loops | vectorised along axis 0 | ~23–35x |
| Streaming mean/variance (500 chunks) | 200,000 | keep all data, recompute each chunk | `StreamingStats` Welford merge | ~2x, with O(1) instead of O(N) memory |
| Confusion matrix (5 classes) | 20,000 | Python loop over pairs | lookup-table encoding + `bincount` | ~3x |
| Decision-tree best split (3 features) | 800 | re-partition for every threshold | sorted cumulative counts | ~110x |
| ROC AUC | 2,000 | all positive/negative pairs | sort + trapezoid | ~150–210x |
| Top-10 selection | 20,000 | full Python sort | `np.argpartition` | ~27–49x |

The original arithmetic micro-benchmark (`--suite arithmetic`) is still available. Example local
measurement (Python 3.13.7, NumPy 2.5.4, macOS 27.0.1, arm64; seeded 200,000-element input, one
warm-up and five timed repeats):

| Implementation | Median (s) | Minimum (s) |
|---|---:|---:|
| Python loop | 0.00737 | 0.00692 |
| NumPy vectorised | 0.0000735 | 0.0000306 |

---

# Project Team

| Team Member   | Main Contribution |
| -------- | ----------------- |
| Prajal Patidar | io.py (shared contribution), optim.py, benchmarking.py, utils.py, pipeline.py, and project integration |
| Manthan Lad | stats.py, metrics.py, display_histogram.py, and technical report |
| Mansi Khare | io.py (shared contribution), rank.py, and presentation |
| Sarthak Sharma | preprocessing.py, sort_search.py, and technical report |
