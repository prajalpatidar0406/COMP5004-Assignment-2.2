"""Matplotlib plots for streaming experiments.

Every plotting function here:

* takes plain sequences or arrays, so it works with ``StreamTrainer``
  histories (``trainer.metric_history("accuracy")``), benchmark results or
  lists typed by hand;
* draws into an existing ``ax`` if one is given (useful for subplots),
  otherwise creates a new figure;
* saves to ``save_path`` if it is given, and shows the figure when
  ``show=True``;
* returns ``(figure, axes)`` so the caller can adjust the plot further.

All plots share one style. Series colours always come in the same order,
blue then orange, a pair that people with colour-vision deficiency can
still tell apart. Lines are 2px wide, the grid and spines are kept light,
and text uses neutral greys instead of the series colours.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .tree import DecisionTreeClassifier

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is a declared dependency
    plt = None

# The first series is always blue and the second always orange, whatever
# their values, so a colour means the same thing on every plot.
SERIES_COLORS = ("#2a78d6", "#eb6834")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
LINE_WIDTH = 2.0
MARKER_SIZE = 6.0  # points; about 8px at the default 100 dpi


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting; install it with pip.")


def _as_series(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Check that ``values`` is a non-empty 1D numeric series and return it as floats.

    NaN is allowed and is drawn as a gap.
    """
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence. Received shape {series.shape}.")
    if series.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return series


def _chunk_axis(n_points: int, x_values: npt.ArrayLike | None) -> npt.NDArray[Any]:
    """Return the x positions, defaulting to chunk numbers starting at 1."""
    if x_values is None:
        return np.arange(1, n_points + 1)
    xs = np.asarray(x_values)
    if xs.shape != (n_points,):
        raise ValueError("x_values must have one entry per metric value.")
    return xs


def _prepare_axes(ax: Any | None, figsize: tuple[float, float]) -> tuple[Any, Any]:
    """Return ``(figure, axes)``, creating them if ``ax`` is None."""
    _require_matplotlib()
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure
    return figure, ax


def _style_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared light grid and spines and the grey title and labels."""
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)


def _finish(figure: Any, save_path: str | None, show: bool) -> None:
    """Save the figure if a path is given, and show it if ``show`` is True."""
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def plot_metric_over_time(
    metric_values: Sequence[float] | npt.ArrayLike,
    title: str,
    ylabel: str,
    *,
    xlabel: str = "Chunk",
    x_values: npt.ArrayLike | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot one streaming metric (e.g. accuracy) across chunks.

    Parameters
    ----------
    metric_values : sequence of float
        One value per chunk. NaN values, such as an undefined AUC, show up
        as gaps instead of being dropped.
    title, ylabel : str
        Chart title and y-axis label.
    xlabel : str, default="Chunk"
        X-axis label.
    x_values : array-like, optional
        X positions, for example ``trainer.metric_history("chunk")``.
        Defaults to chunk numbers 1..n.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure is created if this is not given.
    save_path : str, optional
        File to save the figure to. The format comes from the extension.
    show : bool, default=True
        Whether to display the figure (inline in a notebook).

    Returns
    -------
    tuple
        ``(figure, axes)``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the values are not a non-empty 1D sequence, or ``x_values`` has
        the wrong length.
    """
    values = _as_series(metric_values, "metric_values")
    xs = _chunk_axis(values.size, x_values)
    figure, ax = _prepare_axes(ax, (7.0, 3.8))
    ax.plot(xs, values, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, marker="o",
            markersize=MARKER_SIZE)
    _style_axes(ax, title, xlabel, ylabel)
    _finish(figure, save_path, show)
    return figure, ax


def compare_models(
    metric1: Sequence[float] | npt.ArrayLike,
    metric2: Sequence[float] | npt.ArrayLike,
    labels: Sequence[str],
    *,
    title: str = "Model comparison",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    x_values: npt.ArrayLike | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Compare two models' streaming metrics on the same chunks.

    A typical use is comparing two trees with different ``buffer_size``
    values. Both series share one y-axis. Each model has its own colour and
    marker shape (circles for the first, squares for the second), and its
    name appears in the legend and as a label at the end of its line.

    Parameters
    ----------
    metric1, metric2 : sequence of float
        Per-chunk values for each model. Both must have the same length.
    labels : sequence of two str
        Model names, in the same order as ``metric1`` and ``metric2``.
    title, ylabel, xlabel, x_values, ax, save_path, show
        Same as in :func:`plot_metric_over_time`.

    Returns
    -------
    tuple
        ``(figure, axes)``.

    Raises
    ------
    ValueError
        If ``labels`` is not two non-empty names, or the two series have
        different lengths.
    """
    if len(labels) != 2 or any(not label for label in labels):
        raise ValueError("labels must contain two non-empty model names.")
    first = _as_series(metric1, "metric1")
    second = _as_series(metric2, "metric2")
    if first.size != second.size:
        raise ValueError(
            "model metrics must have the same number of chunks. "
            f"Received {first.size} and {second.size}."
        )
    xs = _chunk_axis(first.size, x_values)

    figure, ax = _prepare_axes(ax, (7.0, 3.8))
    for series, label, color, marker in zip(
        (first, second), labels, SERIES_COLORS, ("o", "s")
    ):
        ax.plot(xs, series, color=color, linewidth=LINE_WIDTH, marker=marker,
                markersize=MARKER_SIZE, label=label)
        finite = np.flatnonzero(np.isfinite(series))
        if finite.size:
            last = finite[-1]
            ax.annotate(f" {label}", (xs[last], series[last]), color=TEXT_SECONDARY,
                        fontsize=9, va="center", annotation_clip=False)
    _style_axes(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    _finish(figure, save_path, show)
    return figure, ax


def plot_predictions_vs_ground_truth(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    title: str = "Predictions vs ground truth",
    ax: Any | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot predicted against actual values for one chunk, usually the latest.

    Labels can be of any dtype. Non-numeric labels are placed on the y-axis
    in sorted order, with their names as tick labels. Ground truth is drawn
    as blue dots and predictions as orange crosses. Each wrong prediction is
    joined to its true value by a thin grey line, and the title shows the
    accuracy on the chunk.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Actual and predicted labels, or regression values.
    title, ax, save_path, show
        Same as in :func:`plot_metric_over_time`.

    Returns
    -------
    tuple
        ``(figure, axes)``.

    Raises
    ------
    ValueError
        If the arrays are not 1D, have different shapes, or are empty.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D arrays. Received shapes {true.shape} and {pred.shape}.")
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. Received {true.shape} and {pred.shape}.")
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    tick_labels = None
    if true.dtype.kind in "biuf" and pred.dtype.kind in "biuf":
        true_y, pred_y = true.astype(float), pred.astype(float)
    else:
        # Map each label to an integer position so it can be plotted; the
        # tick labels show the original names.
        tick_labels, codes = np.unique(np.concatenate((true, pred)).astype(str),
                                       return_inverse=True)
        true_y, pred_y = codes[: true.size], codes[true.size :]

    samples = np.arange(true.size)
    wrong = true != pred
    figure, ax = _prepare_axes(ax, (8.0, 3.6))
    ax.vlines(samples[wrong], true_y[wrong], pred_y[wrong], color=TEXT_SECONDARY,
              linewidth=1.0, alpha=0.6)
    ax.scatter(samples, true_y, s=MARKER_SIZE**2 * 1.4, color=SERIES_COLORS[0],
               label="Ground truth", zorder=3, edgecolors="white", linewidths=1.0)
    ax.scatter(samples, pred_y, s=MARKER_SIZE**2, color=SERIES_COLORS[1], marker="x",
               label="Prediction", zorder=4, linewidths=LINE_WIDTH)
    if tick_labels is not None:
        ax.set_yticks(np.arange(tick_labels.size), tick_labels)
    elif np.all(np.mod(np.concatenate((true_y, pred_y)), 1) == 0):
        ax.set_yticks(np.unique(np.concatenate((true_y, pred_y))))

    accuracy = 1.0 - np.mean(wrong)
    _style_axes(ax, f"{title} (accuracy {accuracy:.1%})", "Sample in chunk", "Label")
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    _finish(figure, save_path, show)
    return figure, ax


def plot_decision_tree(
    tree: DecisionTreeClassifier,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Decision tree",
    ax: Any | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Draw a fitted decision tree, colouring each node by its majority class.

    Internal nodes show their split feature and threshold; all nodes include
    the number of training samples and counts for each class.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        A fitted classifier.
    feature_names : sequence of str, optional
        One display name per feature. Defaults to ``feature 0``, ``feature 1``,
        and so on.
    class_names : sequence of str, optional
        Display names in the same order as ``tree.classes_``. Defaults to the
        string form of each class label.
    title : str, default="Decision tree"
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure is created if this is not given.
    save_path : str, optional
        File to save the figure to. The format comes from the extension.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    tuple
        ``(figure, axes)``.

    Raises
    ------
    ImportError
        If Matplotlib is not installed.
    ValueError
        If the tree is not fitted, or a provided name list does not match the
        number of features or classes.

    Complexity
    ----------
    Time: O(nodes). Space: O(nodes) for the node layout.
    """
    _require_matplotlib()
    from matplotlib import colormaps
    from matplotlib.patches import Patch

    if not tree.is_fitted:
        raise ValueError("tree must be fitted before it can be plotted.")
    root = tree._root
    if root is None:
        raise ValueError("tree must be fitted before it can be plotted.")

    n_features = tree.n_features_in_
    names = list(feature_names) if feature_names is not None else [
        f"feature {index}" for index in range(n_features)
    ]
    if len(names) != n_features or any(not name for name in names):
        raise ValueError("feature_names must contain one non-empty name per tree feature.")

    classes = list(tree.classes_)
    labels = list(class_names) if class_names is not None else [
        str(value) for value in classes
    ]
    if len(labels) != len(classes) or any(not label for label in labels):
        raise ValueError("class_names must contain one non-empty name per tree class.")

    nodes: list[tuple[float, int, Any]] = []
    positions: dict[int, float] = {}
    depths: dict[int, int] = {}
    next_leaf = 0
    stack = [(root, 0, False)]
    while stack:
        node, depth, visited = stack.pop()
        if node.is_leaf:
            position = float(next_leaf)
            next_leaf += 1
        elif not visited:
            if node.left is None or node.right is None:
                raise RuntimeError("tree has an internal node without two children.")
            stack.append((node, depth, True))
            stack.append((node.right, depth + 1, False))
            stack.append((node.left, depth + 1, False))
            continue
        else:
            if node.left is None or node.right is None:
                raise RuntimeError("tree has an internal node without two children.")
            position = (positions[id(node.left)] + positions[id(node.right)]) / 2

        positions[id(node)] = position
        depths[id(node)] = depth
        nodes.append((position, depth, node))

    figure, tree_ax = _prepare_axes(ax, (max(8.0, next_leaf * 1.55),
                                         max(3.8, (tree.depth_ + 1) * 1.6)))
    colors: list[str | tuple[float, float, float]]
    if len(classes) == 2:
        colors = ["#dceafa", "#fde4d8"]
    else:
        colors = []
        for index in range(len(classes)):
            red, green, blue, _ = colormaps["tab10"](index % 10)
            colors.append((
                0.55 + 0.45 * red,
                0.55 + 0.45 * green,
                0.55 + 0.45 * blue,
            ))

    for _, _, node in nodes:
        if node.is_leaf:
            continue
        assert node.left is not None and node.right is not None
        x0, y0 = positions[id(node)], -depths[id(node)]
        for child, branch in ((node.left, "yes"), (node.right, "no")):
            x1, y1 = positions[id(child)], -(depths[id(child)])
            tree_ax.plot([x0, x1], [y0 - 0.12, y1 + 0.38],
                         color=TEXT_SECONDARY, linewidth=1.1, zorder=1)
            tree_ax.text((x0 + x1) / 2, (y0 + y1) / 2, branch, fontsize=8,
                         color=TEXT_SECONDARY, ha="center", va="center",
                         backgroundcolor="white")

    for x, depth, node in nodes:
        counts = node.class_counts
        predicted_index = int(np.argmax(counts))
        class_counts = " / ".join(str(int(count)) for count in counts)
        if node.is_leaf:
            label = f"Predict: {labels[predicted_index]}\nn={counts.sum()} | counts: {class_counts}"
        else:
            assert node.feature is not None
            label = (
                f"{names[node.feature]} <= {node.threshold:.3g}\n"
                f"n={counts.sum()} | counts: {class_counts}"
            )
        tree_ax.text(
            x, -depth, label, ha="center", va="center", fontsize=8,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": colors[predicted_index],
                  "edgecolor": "#8a8a8a", "linewidth": 0.8},
            zorder=2,
        )

    tree_ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12, pad=18)
    tree_ax.set_xlim(-0.8, max(0.8, next_leaf - 0.2))
    tree_ax.set_ylim(-tree.depth_ - 0.65, 0.65)
    tree_ax.axis("off")
    tree_ax.legend(
        handles=[
            Patch(facecolor=colors[index], edgecolor="#8a8a8a",
                  label=f"Predicts {label}")
            for index, label in enumerate(labels)
        ],
        loc="upper right",
        frameon=False,
    )
    _finish(figure, save_path, show)
    return figure, tree_ax
