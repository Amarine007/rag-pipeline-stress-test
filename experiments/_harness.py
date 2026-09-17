"""Shared entry-point plumbing for the sweep-based experiments.

Experiments 1 and 2 differ only in which config they load and which variable
that config sweeps -- the run logic is identical. This holds the common CLI and
result-writing so each `run.py` stays a statement of intent rather than a copy
of the same sixty lines.

Experiment 3 does not use this: its conditions are categorical (retrieval vs.
three stuffed positions) rather than a scalar sweep, so it has its own runner.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.sweep import run_sweep, summarize


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_value(raw: str):
    """Sweep values arrive as strings; chunk sizes are ints, ratios floats."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def run_sweep_experiment(default_config: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--values", nargs="*", default=None, help="Override the swept values."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-condition progress output."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sweep_key = next(iter(config.sweep))
    values = [_parse_value(v) for v in args.values] if args.values else None

    log_path = PROJECT_ROOT / "results" / "logs" / f"{config.name}.jsonl"
    csv_path = PROJECT_ROOT / "results" / f"{config.name}.csv"

    print(f"Experiment: {config.name}")
    print(f"Varying:    {sweep_key} = {values or config.sweep[sweep_key]}")
    print(
        f"Held fixed: seed={config.corpus.seed}, k={config.retrieval.k}, "
        f"n_relevant_docs={config.corpus.n_relevant_docs}, "
        f"chunking={config.chunking.strategy}/{config.chunking.chunk_size}, "
        f"distractor_ratio={config.corpus.distractor_ratio}"
    )
    print(f"Model:      {config.generation.model} (judge: {config.judge.model})")

    results = run_sweep(config, log_path, values=values, show_progress=not args.quiet)

    rows = summarize(results)
    write_csv(rows, csv_path)
    print(f"\nWrote {len(rows)} conditions to {csv_path}")
    print(f"Per-question detail in {log_path}")
    return 0


def bootstrap_path() -> None:
    """Let `python experiments/<name>/run.py` work without installing the package."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
