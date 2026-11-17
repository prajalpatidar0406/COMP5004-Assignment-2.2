# NumCompute

NumCompute is a small numerical-computing toolkit built with **Python and NumPy**. It also
includes reusable Matplotlib plots.

It offers two separate APIs:

- `numcompute` works with complete arrays and tables.
- `numcompute_stream` handles data that arrives in chunks, with incremental preprocessing,
  buffered decision-tree training, streaming statistics and metrics, and `StreamTrainer`.

The streaming package reuses batch implementations where the behavior is shared. The project
does not depend on machine-learning libraries such as **scikit-learn** or **PyTorch**, or on
**pandas**.

---

## Features

- Read a CSV all at once, or process it chunk by chunk to keep memory use bounded.
- Impute, scale and encode one-dimensional data or feature matrices, either in one pass or
  incrementally with `partial_fit`.
- Train a Gini or entropy **decision tree**, with an optional depth limit and buffered retraining.
- Evaluate chunks with **`StreamTrainer`** using a test-then-train workflow; inspect accuracy,
  other metrics, timing and memory in its logs.
- Track **streaming statistics** such as mean, variance, quantiles and adaptive histograms, with
  an optional sliding window.
- Calculate **streaming metrics** (accuracy, precision, recall, F1, confusion matrix and AUC),
  including rolling-window results.
- Plot metrics, compare models and inspect predictions against ground truth.
- Sort, search, rank and select values; estimate gradients and Jacobians.
- Compare representative operations with straightforward reference implementations.

---

## Project Structure

```text
.
├── NumCompute/
│   ├── numcompute/          Batch API
│   └── numcompute_stream/   Streaming API
├── tests/
├── demo/
├── README.md
├── CODING_STYLE.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

After installation, both packages are available: `numcompute` is the original batch API, and
`numcompute_stream` is the separate streaming API. The quickstart notebook uses `numcompute`;
the stream demo uses `numcompute_stream`.

Shared behavior is imported from `numcompute`, rather than copied into the streaming package.
Streaming-specific classes and algorithms live in `numcompute_stream`.

---

## Requirements

* Python 3.10 or newer
* NumPy and Matplotlib (installed with the package)
* Jupyter, if you want to run the notebooks
* Pytest, if you want to run the test suite

The project does not depend on external machine-learning libraries such as scikit-learn or
PyTorch, or on pandas.

---

## Installation

Clone the repository, then change into its root directory:

```bash
git clone https://github.com/prajalpatidar0406/COMP5004-Assignment-2.2.git
cd COMP5004-Assignment-2.2
```

Create and activate a virtual environment before installing the project.

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

Install the package and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install .
```

Jupyter is optional. Install it separately if you plan to run the notebooks:

```bash
python -m pip install jupyter
```

You can import `numcompute` or `numcompute_stream` from a Python script or notebook.

---

## Streaming modules (`numcompute_stream`)

These modules provide the streaming API. Shared batch implementations are imported from
`numcompute` where behavior is the same.

### `io.py` — Data loading

Supports:

* `load_csv(path, ...) -> (header, data)` for complete-table loading
* `iter_csv(path, chunksize=1000, ...)` for bounded-memory iteration
* Custom delimiters, requested or inferred dtypes, and fill/drop missing modes
* Numeric arrays for numeric tables and object arrays for mixed columns

`load_csv` returns the column names and a 2D NumPy array using `numpy.loadtxt`. For larger files,
use `iter_csv` to process one chunk at a time.

---

### `preprocessing.py` — Data preprocessing

| Transformer | Learns | Streaming update (`partial_fit`) | State memory |
|---|---|---|---|
| `StandardScaler(epsilon=0.0, ema_alpha=None)` | mean, variance | exact parallel Welford merge, or exponential moving average for drift | O(features) |
| `MinMaxScaler(feature_range=(0, 1), clip=False)` | min, max | running extrema | O(features) |
| `Imputer(strategy="mean" \| "median" \| "constant")` | replacement value | running mean / exact median | O(features) mean, O(n × features) median |
| `OneHotEncoder(handle_unknown="error" \| "ignore")` | category vocabulary | new categories appended as **new trailing columns** | O(categories) |
| `ColumnSelector(columns, dtype=None)` | — | stateless | — |

Scalers, `Imputer` and `OneHotEncoder` accept a single feature `(n_samples,)` or a matrix
`(n_samples, n_features)`; each column keeps its own state. `ColumnSelector` expects a matrix.
Scalers work across columns, and their learned attributes are floats for 1D input and per-feature
arrays for 2D input.

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
  once, and **all thresholds are scored in one vectorised pass** using cumulative class counts.
  For *m* candidate features, *n* samples and *k* classes, split search costs
  O(m · (n log n + nk)) per node.
- Thresholds use midpoints between distinct feature values when the midpoint is representable.
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
updated state, and updates the final estimator with `partial_fit` when available (falling back to
`fit` otherwise). A buffered tree therefore buffers the transformed chunks. `OneHotEncoder`
appends newly learned categories to the end of its output, preserving existing columns. In a
`FeatureUnion`, however, a growing branch can shift the columns of later branches, so place
category-encoding branches last. Use `ColumnSelector` to route table columns through different
preprocessing.

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])
pipe.partial_fit(X_chunk, y_chunk)
predictions = pipe.predict(X_new)
probabilities = pipe.predict_proba(X_new)
```

---

### `stream.py` — Streaming training and logging

`StreamTrainer(model, pipeline=None, *, metrics=None, logger=None, track_memory=True)` manages a
model, optional preprocessing and logging.

- `fit_chunk(X, y)` updates the pipeline and model with one chunk.
- `score_chunk(X, y)` evaluates and appends a record containing per-chunk and cumulative accuracy,
  every extra metric (per chunk and as `cumulative_<name>`), sample counts, fit/predict time and
  the **estimated memory use** of the model and optional pipeline. For a `RollingWindowMetric`,
  `cumulative_<name>` reports its current window, not the full stream.
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
| mean, variance, std, min, max | parallel Welford/Chan merge; ignores NaN when `skipna=True` | O(features) |
| quantiles | uniform reservoir sample (vectorised Algorithm R): exact up to `reservoir_size` rows, then estimated from the sample | O(features × reservoir) |
| histogram | `StreamingHistogram` in adaptive mode: fixed bin count; when data leaves the range the bin width doubles and bin pairs merge, so **counts stay exact** without storing data | O(features × bins) |
| any of the above over a sliding window | `window_size=W`: exact over the last W rows | O(features × W) |

```python
state = update_stats(X_chunk)            # first chunk
state = update_stats(next_chunk, state)  # later chunks
state["mean"], state["quantiles"], state["histogram"]
```

---

### `metrics.py` — Model evaluation

Batch functions: `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`,
`roc_curve`, `auc` and `roc_auc_score`.

The streaming classes provide `update(...)`, `result()` and `reset()`. Pass predicted labels to
classification metrics and positive-class scores to `AUC`. Exact, non-windowed metrics match the
batch result over the combined chunks. Binned AUC uses bounded memory and is approximate;
`RollingWindowMetric` reports only the most recent *k* chunks.

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
| `plot_predictions_vs_ground_truth(y_true, y_pred)` | one chunk; marks mistakes and supports string labels |
| `plot_decision_tree(tree, feature_names, class_names)` | fitted tree structure with split thresholds and class counts |

All plotting functions accept `save_path=` (any Matplotlib format), `show=` (display the plot)
and `ax=` (draw into an existing subplot). Each returns `(figure, axes)`.

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
- `benchmark_toolkit` compares representative NumCompute operations with the straightforward
  approaches they replace. It checks that both produce the same result before timing them.

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

`Ranker.rank(data, method="average")` supports the `average`, `dense`, and `ordinal` ranking
methods and handles tied values. `Ranker.percentile(data, q, interpolation="linear")` accepts
percentiles from 0 to 100 and supports `linear`, `lower`, `higher`, and `midpoint` interpolation.

```python
from numcompute.rank import Ranker

ranker = Ranker()
ranker.rank(data, method="average")
ranker.percentile(data, q=50, interpolation="linear")
```

For streams, `StreamingRanker` merges each chunk into a sorted reference with
`partial_fit(chunk)` (or `update(chunk)`). It then answers `percentile(q, interpolation)`
and `percentile_rank(values)` over everything seen so far. Results are exact, so it
keeps all valid values; for bounded memory use `stats.StreamingStats` quantiles.

---

## Running Tests

Run the suite from the repository root:

```bash
python -m pytest
```

Modules have focused test files, including `test_io.py`, `test_preprocessing.py`, `test_tree.py`,
`test_pipeline.py`, `test_stream.py`, `test_stats.py`, `test_metrics.py`, `test_sort_search.py`,
`test_rank.py`, `test_optim.py`, `test_utils.py`, `test_visualise.py` and `test_benchmarking.py`.
The suite also includes independent reference checks and an end-to-end integration test:

- `test_reference_checks.py` checks results against implementations that share no code with
  NumCompute: exact rational arithmetic (`fractions`), the `statistics` and `math` modules, a
  brute-force split search and the pairwise Mann–Whitney definition of AUC.
- `test_integration.py` runs the real Titanic CSV through streaming statistics, a mixed-type
  pipeline, `StreamTrainer`, saved plots and the buffer benchmark.

Chunking-invariance tests verify that streaming moments and accuracy, precision, recall and F1
match their batch results across different ways of splitting the same data.

---

## Demo

* `demo/quickstart.ipynb` introduces the batch toolkit, including CSV loading, preprocessing,
  sorting, ranking, statistics, metrics, gradients and pipelines.
* `demo/stream_demo.ipynb` walks through the streaming workflow: it reads the Titanic CSV in
  chunks, tracks statistics, trains a mixed-type pipeline with a buffered tree, logs and plots
  metrics, and compares buffer sizes.

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebooks locate the repository root themselves, so they run from either the root or `demo/`.

---

## Benchmark

Run `python -m numcompute_stream.benchmarking --suite toolkit` (or `--suite all`). Each benchmark
first checks that NumCompute and its reference produce the same answer, then compares their
runtime.

The table below shows local measurements (Python 3.13.9, NumPy 2.3.5, macOS 27.2.1 arm64; median
of 3–7 repeats). Your results will vary with hardware and system load, so run the benchmark on
your own machine for a meaningful comparison.

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

## Project Team

| Team Member   | Main Contribution |
| -------- | ----------------- |
| Prajal Patidar | io.py (shared contribution), optim.py, benchmarking.py, utils.py, pipeline.py, and project integration |
| Manthan Lad | stats.py, metrics.py, histogram_display.py, and technical report |
| Mansi Khare | io.py (shared contribution), rank.py, and presentation |
| Sarthak Sharma | preprocessing.py, sort_search.py, and technical report |
