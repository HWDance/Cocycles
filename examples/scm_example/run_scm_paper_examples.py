"""Execute all SCM paper examples and rebuild their comparison figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep numerical libraries from creating large CPU thread pools inside a
# scheduled GPU job. These defaults may still be overridden by the caller.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient

from scm_misspecification_figure import make_comparison_figure


HERE = Path(__file__).resolve().parent
NOTEBOOKS = (
    "cocycles_binary_example.ipynb",
    "gaussian_flow_binary_example.ipynb",
    "cocycles_mixedtails_example.ipynb",
    "gaussian_flow_mixedtails_example.ipynb",
)


def execute_notebook(path: Path, *, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute(cwd=str(HERE))
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=21_600,
        help="Maximum seconds allowed for any one notebook cell.",
    )
    parser.add_argument(
        "--only",
        choices=NOTEBOOKS,
        action="append",
        help="Execute only the named notebook; may be supplied more than once.",
    )
    args = parser.parse_args()

    selected = tuple(args.only) if args.only else NOTEBOOKS
    for filename in selected:
        path = HERE / filename
        print(f"Executing {path}", flush=True)
        execute_notebook(path, timeout=args.timeout)
        print(f"Completed {path}", flush=True)

    if set(selected) == set(NOTEBOOKS):
        output_path = make_comparison_figure()
        print(f"Saved {output_path}", flush=True)
    else:
        print("Skipped combined figure because only a subset was executed.", flush=True)


if __name__ == "__main__":
    main()
