"""Reproducible paper figure for the SCM noise-misspecification examples.

Each example notebook saves raw estimated and true treatment effects with
``save_effect_results``.  Running this module after all four notebooks have
completed builds the shared 1 x 4 vector PDF without retraining any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import expit


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "paper_results"
DEFAULT_FIGURE_PATH = HERE / "scm_noise_misspecification_comparison.pdf"

RESULT_PATHS = {
    ("binary", "cocycles"): RESULTS_DIR / "binary_cocycles.npz",
    ("binary", "gaussian_base"): RESULTS_DIR / "binary_gaussian_base.npz",
    ("mixed_tails", "cocycles"): RESULTS_DIR / "mixed_tails_cocycles.npz",
    ("mixed_tails", "gaussian_base"): RESULTS_DIR / "mixed_tails_gaussian_base.npz",
}

METHOD_LABELS = {
    "cocycles": "Cocycles",
    "gaussian_base": "Gaussian base",
}

NOISE_LABELS = {
    "binary": "Binary noise",
    "mixed_tails": "Mixed-tail noise",
}

ESTIMATED_COLOR = "#0072B2"
TRUE_COLOR = "#D55E00"

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.4,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}


def _as_finite_vector(values: Any, *, name: str) -> np.ndarray:
    """Convert a tensor/array-like object to a finite float64 vector."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        count = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {count} non-finite values.")
    return array


def save_effect_results(
    output_path: str | Path,
    *,
    estimated_effect: Any,
    true_effect: Any,
    method: str,
    noise: str,
    seed: int,
    truth_seed: int,
    n_train_per_treatment: int,
    n_effect: int,
    epochs: int,
    learning_rate: float,
) -> Path:
    """Save everything needed to reconstruct one figure panel."""
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")
    if noise not in NOISE_LABELS:
        raise ValueError(f"Unknown noise type: {noise}")

    estimated = _as_finite_vector(estimated_effect, name="estimated_effect")
    truth = _as_finite_vector(true_effect, name="true_effect")
    if estimated.size != n_effect:
        raise ValueError(
            f"Expected {n_effect} estimated effects, received {estimated.size}."
        )
    if truth.size != n_effect:
        raise ValueError(f"Expected {n_effect} true effects, received {truth.size}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        estimated_effect=estimated.astype(np.float32),
        true_effect=truth.astype(np.float32),
        method=np.str_(method),
        noise=np.str_(noise),
        seed=np.int64(seed),
        truth_seed=np.int64(truth_seed),
        n_train_per_treatment=np.int64(n_train_per_treatment),
        n_effect=np.int64(n_effect),
        epochs=np.int64(epochs),
        learning_rate=np.float64(learning_rate),
    )
    return output_path


def load_effect_results(path: str | Path) -> dict[str, Any]:
    """Load a saved panel while keeping NumPy object loading disabled."""
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    result["method"] = str(result["method"].item())
    result["noise"] = str(result["noise"].item())
    return result


def _empirical_cdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)
    return np.searchsorted(ordered, grid, side="right") / ordered.size


def _draw_panel(
    ax: plt.Axes,
    result: dict[str, Any],
) -> tuple[Line2D, Line2D, float]:
    estimated = _as_finite_vector(
        result["estimated_effect"], name="estimated_effect"
    )
    truth = _as_finite_vector(result["true_effect"], name="true_effect")
    noise = result["noise"]

    if noise == "binary":
        grid = np.linspace(-1.0, 2.0, 800)
        estimated_curve = _empirical_cdf(estimated, grid)
        true_curve = _empirical_cdf(truth, grid)
        ax.set_xlim(-1.0, 2.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks([-1.0, 0.0, 1.0, 2.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel(r"$Y(1)-Y(0)$")
    elif noise == "mixed_tails":
        grid = np.linspace(0.0, 1.0, 800)
        # The bounded monotone transform makes the extreme asymmetric tail
        # visible without clipping raw effects. ECDFs handle both smooth and
        # near-degenerate fitted distributions without arbitrary jitter or
        # bandwidth choices.
        # Clipping only prevents floating-point under/overflow at the exact
        # endpoints; values beyond this range are visually indistinguishable
        # on the [0, 1] axis but should not become an artificial atom at zero.
        estimated_transformed = expit(np.clip(estimated, -40.0, 40.0))
        true_transformed = expit(np.clip(truth, -40.0, 40.0))
        estimated_curve = _empirical_cdf(estimated_transformed, grid)
        true_curve = _empirical_cdf(true_transformed, grid)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlabel(r"$\sigma\,\left(Y(1)-Y(0)\right)$")
    else:
        raise ValueError(f"Unknown noise type in result: {noise}")

    estimated_line = ax.plot(
        grid,
        estimated_curve,
        color=ESTIMATED_COLOR,
        label="Estimated",
        zorder=3,
    )[0]
    ax.fill_between(
        grid,
        0.0,
        estimated_curve,
        color=ESTIMATED_COLOR,
        alpha=0.18,
        linewidth=0.0,
        zorder=1,
    )
    true_line = ax.plot(
        grid,
        true_curve,
        color=TRUE_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=2.6,
        label="True",
        zorder=4,
    )[0]

    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    return estimated_line, true_line, float(
        max(np.max(estimated_curve), np.max(true_curve))
    )


def plot_result_preview(path: str | Path) -> tuple[plt.Figure, plt.Axes]:
    """Display one notebook result using exactly the shared panel styling."""
    result = load_effect_results(path)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
        estimated_line, true_line, _ = _draw_panel(ax, result)
        ax.set_ylabel("CDF")
        ax.set_title(
            f"{METHOD_LABELS[result['method']]} — {NOISE_LABELS[result['noise']]}"
        )
        ax.legend(handles=[estimated_line, true_line], frameon=False)
    return fig, ax


def make_comparison_figure(
    output_path: str | Path = DEFAULT_FIGURE_PATH,
) -> Path:
    """Build the paper-ready 1 x 4 comparison from all saved results."""
    missing = [path for path in RESULT_PATHS.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Run all four example notebooks before building the figure. "
            f"Missing:\n{missing_text}"
        )

    results = {
        key: load_effect_results(path) for key, path in RESULT_PATHS.items()
    }
    methods = ("gaussian_base", "cocycles")
    noises = ("binary", "mixed_tails")
    panel_letters = ("a", "b", "c", "d")

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(7.2, 2.65), constrained_layout=True)
        subfigures = fig.subfigures(1, 2, wspace=0.05)
        legend_handles: tuple[Line2D, Line2D] | None = None
        panel = 0

        axes: list[plt.Axes] = []
        for subfigure, noise in zip(subfigures, noises):
            subfigure.suptitle(NOISE_LABELS[noise])
            pair_axes = subfigure.subplots(1, 2, sharey=True)
            for method, ax in zip(methods, pair_axes):
                axes.append(ax)
                estimated_line, true_line, _ = _draw_panel(
                    ax, results[(noise, method)]
                )
                if legend_handles is None:
                    legend_handles = (estimated_line, true_line)
                ax.set_title(METHOD_LABELS[method], pad=7)
                ax.text(
                    -0.18,
                    1.04,
                    f"({panel_letters[panel]})",
                    transform=ax.transAxes,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )
                if panel > 0:
                    ax.tick_params(labelleft=False)
                panel += 1

        axes[0].set_ylabel("CDF")

        assert legend_handles is not None
        fig.legend(
            handles=legend_handles,
            labels=("Estimated", "True"),
            loc="outside lower center",
            ncol=2,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.8,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            bbox_inches="tight",
            metadata={
                "Title": "SCM noise misspecification comparison",
                "Creator": "Matplotlib",
            },
        )
        plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Destination PDF path.",
    )
    args = parser.parse_args()
    output_path = make_comparison_figure(args.output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
