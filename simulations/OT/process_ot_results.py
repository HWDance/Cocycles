#!/usr/bin/env python3
"""Build trimmed summary tables for the two OT simulation designs.

This script reproduces the processing used in
``results_viewer_sparse-checkpoint.ipynb``:

* retain Seq-OT/KR, squared-Euclidean OT, and Cocycles;
* align Cocycles' ``ATE21``/``RMSE21`` fields with the direct-effect fields;
* assign zero counterfactual-path inconsistency to Cocycles, whose learned
  cocycle is compositionally consistent by construction; and
* within each method/rho cell, sequentially remove observations outside the
  1% and 99% quantiles of every displayed metric before aggregation.

The saved ``RMSE`` quantities are the mean per-observation Euclidean error,
matching the historical notebook and runner naming.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RHOS = (0.1, 0.3, 0.5, 0.7, 0.9)
EXPECTED_REPLICATES = 20
METHOD_ORDER = (
    "Seq-OT",
    "Seq-OT (wrong order)",
    "Cocycles",
    "Cocycles (wrong order)",
    "OT",
)
DESIGN_TITLES = {
    "design_i": "Design I: additive, multivariate Laplace noise",
    "design_ii": "Design II: non-additive, independent Laplace noise",
}
METRIC_LABELS = {
    "ate_error": "ATE error",
    "cf_rmse": "CF RMSE",
    "cf_inconsistency": "CF path inconsistency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.01,
        help="Fraction trimmed from each tail per metric (default: 0.01).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Tail probability for reported quantile intervals (default: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for generated CSV, Markdown, and LaTeX files.",
    )
    parser.add_argument(
        "--design-i-base",
        type=Path,
        default=RESULTS_DIR
        / "OT_affine_results_trials=20_n=500_m=500_additive=True_multivariate=True.pt",
        help="Historical design-I file containing OT baselines.",
    )
    parser.add_argument(
        "--design-ii-base",
        type=Path,
        default=RESULTS_DIR
        / "OT_affine_results_trials=20_n=500_m=500_additive=False_multivariate=False_dist=laplace.pt",
        help="Historical design-II file containing correct-order Seq-OT and OT.",
    )
    parser.add_argument(
        "--design-i-seqot",
        type=Path,
        default=SCRIPT_DIR / "seqot_results_chain.pt",
    )
    parser.add_argument(
        "--design-ii-seqot-wrong",
        type=Path,
        default=SCRIPT_DIR / "seqot_results_design_ii_wrong_order.pt",
    )
    parser.add_argument(
        "--design-i-cocycles",
        type=Path,
        default=SCRIPT_DIR / "cocycle_results_chain.pt",
    )
    parser.add_argument(
        "--design-ii-cocycles",
        type=Path,
        default=SCRIPT_DIR / "cocycle_results.pt",
    )
    return parser.parse_args()


def load_pt(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    # These are trusted, repository-owned simulation payloads. Explicitly
    # disable weights-only loading because they contain NumPy scalar objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def method_label(method: str, wrong_order: bool) -> str:
    base = {"seqot": "Seq-OT", "cocycles": "Cocycles", "ot": "OT"}[method]
    return f"{base} (wrong order)" if wrong_order else base


def normalize_record(
    result: dict[str, Any],
    *,
    design: str,
    method: str,
    wrong_order: bool,
    source: Path,
) -> dict[str, Any]:
    if method == "cocycles":
        ate_error = result["ATE21"]
        cf_rmse = result["RMSE21"]
        # This is the same convention used by the old evaluation notebook.
        cf_inconsistency = 0.0
    else:
        ate_error = result["ATE21direct"]
        cf_rmse = result["RMSE21direct"]
        cf_inconsistency = result["RMSEinconsistency"]

    return {
        "design": design,
        "method": method,
        "method_label": method_label(method, wrong_order),
        "wrong_order": bool(wrong_order),
        "rho": float(result["corr"]),
        "seed": int(result["seed"]),
        "ate_error": float(ate_error),
        "cf_rmse": float(cf_rmse),
        "cf_inconsistency": float(cf_inconsistency),
        "source": str(source.relative_to(SCRIPT_DIR)),
    }


def normalize_tuple_payload(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_pt(path):
        if not isinstance(item, tuple) or len(item) != 6:
            raise ValueError(f"Unexpected tuple payload in {path}: {item!r}")
        method_raw, design, wrong_order, rho, seed, result = item
        method = "cocycles" if method_raw == "cocycle" else str(method_raw)
        row = normalize_record(
            result,
            design=str(design),
            method=method,
            wrong_order=bool(wrong_order),
            source=path,
        )
        if not math.isclose(row["rho"], float(rho)) or row["seed"] != int(seed):
            raise ValueError(f"Outer and inner metadata disagree in {path}: {item[:5]}")
        rows.append(row)
    return rows


def normalize_historical_base(
    path: Path, *, design: str, include_correct_seqot: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in load_pt(path):
        name = str(result["name"])
        if name == "OT_dist=sqeuclidean":
            method = "ot"
            wrong_order = False
        elif name == "KReps0" and include_correct_seqot:
            method = "seqot"
            wrong_order = False
        else:
            continue
        rows.append(
            normalize_record(
                result,
                design=design,
                method=method,
                wrong_order=wrong_order,
                source=path,
            )
        )
    return rows


def collect_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Design I uses the new matched Seq-OT run; the historical file supplies OT.
    rows.extend(
        normalize_historical_base(
            args.design_i_base, design="design_i", include_correct_seqot=False
        )
    )
    rows.extend(normalize_tuple_payload(args.design_i_seqot))
    rows.extend(normalize_tuple_payload(args.design_i_cocycles))

    # Design II's historical file supplies correct-order Seq-OT and OT; the new
    # targeted run supplies only the formerly missing wrong-order Seq-OT cells.
    rows.extend(
        normalize_historical_base(
            args.design_ii_base, design="design_ii", include_correct_seqot=True
        )
    )
    rows.extend(normalize_tuple_payload(args.design_ii_seqot_wrong))
    rows.extend(normalize_tuple_payload(args.design_ii_cocycles))

    frame = pd.DataFrame(rows)
    validate_coverage(frame)
    method_rank = {name: rank for rank, name in enumerate(METHOD_ORDER)}
    frame["_method_rank"] = frame["method_label"].map(method_rank)
    frame = frame.sort_values(
        ["design", "_method_rank", "rho", "seed"], ignore_index=True
    ).drop(columns="_method_rank")
    return frame


def validate_coverage(frame: pd.DataFrame) -> None:
    required_columns = {
        "design",
        "method",
        "method_label",
        "wrong_order",
        "rho",
        "seed",
        "ate_error",
        "cf_rmse",
        "cf_inconsistency",
        "source",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Normalized results lack columns: {sorted(missing)}")

    duplicate = frame.duplicated(
        ["design", "method", "wrong_order", "rho", "seed"], keep=False
    )
    if duplicate.any():
        raise ValueError(
            "Duplicate design/method/order/rho/seed rows:\n"
            + frame.loc[duplicate].to_string(index=False)
        )

    expected_cells = {
        (design, method, rho)
        for design in DESIGN_TITLES
        for method in METHOD_ORDER
        for rho in RHOS
    }
    counts = frame.groupby(["design", "method_label", "rho"]).size()
    actual_cells = set(counts.index)
    if actual_cells != expected_cells:
        missing_cells = sorted(expected_cells - actual_cells)
        extra_cells = sorted(actual_cells - expected_cells)
        raise ValueError(
            f"Unexpected result coverage; missing={missing_cells}, extra={extra_cells}"
        )
    bad_counts = counts[counts != EXPECTED_REPLICATES]
    if not bad_counts.empty:
        raise ValueError(
            f"Expected {EXPECTED_REPLICATES} replicates per cell:\n{bad_counts}"
        )

    metrics = ["ate_error", "cf_rmse", "cf_inconsistency"]
    finite = frame[metrics].map(lambda value: math.isfinite(float(value)))
    if not finite.all().all():
        raise ValueError("At least one requested metric is non-finite.")


def trim_group(
    group: pd.DataFrame, metric_columns: Iterable[str], trim_frac: float
) -> pd.DataFrame:
    """Match the notebook's sequential per-metric quantile filtering."""
    trimmed = group.copy()
    for column in metric_columns:
        lower = trimmed[column].quantile(trim_frac)
        upper = trimmed[column].quantile(1.0 - trim_frac)
        trimmed = trimmed[
            (trimmed[column] >= lower) & (trimmed[column] <= upper)
        ]
    return trimmed


def summarize(frame: pd.DataFrame, trim_frac: float, alpha: float) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["design", "method", "method_label", "wrong_order", "rho"]
    all_metrics = ["ate_error", "cf_rmse", "cf_inconsistency"]

    for keys, group in frame.groupby(group_columns, sort=False):
        design, method, label, wrong_order, rho = keys
        displayed_metrics = ["ate_error", "cf_rmse"]
        if design == "design_ii":
            displayed_metrics.append("cf_inconsistency")
        trimmed = trim_group(group, displayed_metrics, trim_frac)
        if trimmed.empty:
            raise ValueError(f"Trimming removed every row for {keys}")

        row: dict[str, Any] = {
            "design": design,
            "method": method,
            "method_label": label,
            "wrong_order": bool(wrong_order),
            "rho": float(rho),
            "n_raw": len(group),
            "n_trimmed": len(trimmed),
        }
        for metric in all_metrics:
            if metric not in displayed_metrics:
                for stat in ("mean", "sd", "median", "q05", "q95"):
                    row[f"{metric}_{stat}"] = math.nan
                continue
            values = trimmed[metric]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std()
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q05"] = values.quantile(alpha)
            row[f"{metric}_q95"] = values.quantile(1.0 - alpha)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    method_rank = {name: rank for rank, name in enumerate(METHOD_ORDER)}
    summary["_method_rank"] = summary["method_label"].map(method_rank)
    return summary.sort_values(
        ["design", "_method_rank", "rho"], ignore_index=True
    ).drop(columns="_method_rank")


def format_value(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def markdown_table(summary: pd.DataFrame, design: str, metric: str) -> str:
    subset = summary[summary["design"] == design]
    mean_column = f"{metric}_mean"
    sd_column = f"{metric}_sd"
    lines = [
        f"### {METRIC_LABELS[metric]}",
        "",
        "| Method | " + " | ".join(f"ρ={rho:.1f}" for rho in RHOS) + " |",
        "|---|" + "---:|" * len(RHOS),
    ]
    for method in METHOD_ORDER:
        method_rows = subset[subset["method_label"] == method].set_index("rho")
        cells = [
            format_value(
                float(method_rows.loc[rho, mean_column]),
                float(method_rows.loc[rho, sd_column]),
            )
            for rho in RHOS
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(summary: pd.DataFrame, trim_frac: float, alpha: float) -> str:
    lines = [
        "# OT simulation results",
        "",
        (
            "Entries are trimmed mean ± sample SD. Within each method–ρ cell, "
            f"the script sequentially retains the [{trim_frac:.2f}, "
            f"{1.0 - trim_frac:.2f}] quantile range for every displayed metric, "
            "matching the old results-viewer notebook. The accompanying CSV also "
            f"contains medians, [{alpha:.2f}, {1.0 - alpha:.2f}] quantiles, and "
            "raw/retained sample counts."
        ),
        "",
        (
            "Cocycles path inconsistency is set to 0, following the old notebook: "
            "a single learned cocycle gives compositionally consistent paths by construction."
        ),
        "",
    ]
    for design in ("design_i", "design_ii"):
        lines.extend([f"## {DESIGN_TITLES[design]}", ""])
        metrics = ["ate_error", "cf_rmse"]
        if design == "design_ii":
            metrics.append("cf_inconsistency")
        for metric in metrics:
            lines.extend([markdown_table(summary, design, metric), ""])
    return "\n".join(lines).rstrip() + "\n"


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%")


def render_latex(summary: pd.DataFrame, trim_frac: float) -> str:
    blocks = [
        "% Generated by simulations/OT/process_ot_results.py",
        f"% Sequential two-sided quantile trim: {100 * trim_frac:.1f} percent.",
        "% Entries are trimmed mean $\\pm$ sample SD.",
        "",
    ]
    for design in ("design_i", "design_ii"):
        metrics = ["ate_error", "cf_rmse"]
        if design == "design_ii":
            metrics.append("cf_inconsistency")
        for metric in metrics:
            subset = summary[summary["design"] == design]
            mean_column = f"{metric}_mean"
            sd_column = f"{metric}_sd"
            blocks.extend(
                [
                    r"\begin{table}[t]",
                    r"\centering",
                    (
                        r"\caption{" + latex_escape(DESIGN_TITLES[design]) + ": "
                        + latex_escape(METRIC_LABELS[metric])
                        + r" (trimmed mean $\pm$ SD).}"
                    ),
                    r"\begin{tabular}{l" + "r" * len(RHOS) + "}",
                    r"\toprule",
                    "Method & "
                    + " & ".join(rf"$\rho={rho:.1f}$" for rho in RHOS)
                    + " " + chr(92) * 2,
                    r"\midrule",
                ]
            )
            for method in METHOD_ORDER:
                method_rows = subset[subset["method_label"] == method].set_index("rho")
                cells = [
                    f"{float(method_rows.loc[rho, mean_column]):.4f} "
                    rf"$\pm$ {float(method_rows.loc[rho, sd_column]):.4f}"
                    for rho in RHOS
                ]
                blocks.append(latex_escape(method) + " & " + " & ".join(cells) + " " + chr(92) * 2)
            blocks.extend(
                [
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table}",
                    "",
                ]
            )
    return "\n".join(blocks)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.trim_frac < 0.5:
        raise ValueError("--trim-frac must be in [0, 0.5).")
    if not 0.0 < args.alpha < 0.5:
        raise ValueError("--alpha must be in (0, 0.5).")

    per_run = collect_rows(args)
    summary = summarize(per_run, args.trim_frac, args.alpha)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / "ot_results_per_run.csv"
    summary_path = output_dir / "ot_results_summary.csv"
    markdown_path = output_dir / "ot_results_tables.md"
    latex_path = output_dir / "ot_results_tables.tex"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(
        render_markdown(summary, args.trim_frac, args.alpha), encoding="utf-8"
    )
    latex_path.write_text(render_latex(summary, args.trim_frac), encoding="utf-8")

    print(f"Validated {len(per_run)} per-run records across {len(summary)} cells.")
    print(f"Saved {per_run_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {markdown_path}")
    print(f"Saved {latex_path}")
    print()
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
