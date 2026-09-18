"""Build the paper's figures and markdown tables from experiment result CSVs.

Reads `results/*.csv` (written by the experiment runners) and emits PNGs to
`results/figures/` plus markdown tables to stdout for pasting into the paper.

Two conventions are load-bearing rather than cosmetic:

- **Retrieval and answer quality never share an axis.** Every figure is a pair
  of panels -- retrieval on the left, answers on the right. Putting them on one
  plot with two y-scales would imply a relationship between the scales that does
  not exist, and would hide the case this project cares about most: retrieval
  holding steady while answer quality falls.
- **Every series carries a marker shape and a direct label**, not just a hue.
  One of the palette colors sits below 3:1 contrast on white, so color alone is
  not allowed to carry identity.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"

# Validated categorical slots (light surface). Markers and dashes provide the
# secondary encoding that the contrast warning on slot 3 requires.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

SERIES_STYLE = {
    0: {"color": BLUE, "marker": "o", "linestyle": "-"},
    1: {"color": ORANGE, "marker": "s", "linestyle": "--"},
    2: {"color": AQUA, "marker": "^", "linestyle": ":"},
}


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _spread(values: list[float], minimum_gap: float = 0.055) -> dict[int, float]:
    """Vertical offsets, in points, that stop near-equal end labels overprinting.

    Series that converge -- hit rate and MRR both reaching ~1.0 -- would
    otherwise stack their labels in the same place and render as unreadable
    overlap. Labels are nudged apart only when they actually collide, so a
    well-separated panel keeps every label on its own point.
    """
    order = sorted((v, i) for i, v in enumerate(values) if v is not None)
    offsets: dict[int, float] = {}
    previous = None
    for value, index in order:
        if previous is not None and value - previous < minimum_gap:
            offsets[index] = offsets.get(previous_index, 0.0) + 9.0
        else:
            offsets[index] = 0.0
        previous, previous_index = value, index
    return offsets


def plot_series(ax, x, rows, keys_labels, label_last: bool = True) -> None:
    """Draw one panel's series, direct-labeling the final point of each."""
    finals: list[float | None] = []
    for i, (key, label) in enumerate(keys_labels):
        values = [float(r[key]) if r[key] not in ("", None) else None for r in rows]
        finals.append(values[-1] if values else None)
        style = SERIES_STYLE[i % len(SERIES_STYLE)]
        ax.plot(
            x, values, label=label, linewidth=2, markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, **style,
        )

    if label_last:
        offsets = _spread([v for v in finals])
        for i, value in enumerate(finals):
            if value is None:
                continue
            ax.annotate(
                f"{value:.2f}",
                xy=(x[-1], value),
                xytext=(6, offsets.get(i, 0.0)),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
                va="center",
            )
    # Room on the right so the direct labels never collide with the frame.
    ax.margins(x=0.12)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="best")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sweep_figure(csv_name: str, xlabel: str, title: str, out_name: str, log_x: bool = False):
    """Two-panel figure: retrieval quality left, answer quality right."""
    path = RESULTS / csv_name
    if not path.exists():
        print(f"  skipped {csv_name} (not found -- run the experiment first)")
        return None

    rows = read_csv(path)
    x = [float(r["value"]) for r in rows]
    n = rows[0]["n"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (left, right):
        ax.set_facecolor("#fcfcfb")

    plot_series(
        left, x, rows,
        [("hit_rate_at_k", "hit rate@k"), ("mrr", "MRR"), ("precision_at_k", "precision@k")],
    )
    style_axes(left, xlabel, "score", "Retrieval quality")

    plot_series(
        right, x, rows,
        [
            ("accuracy", "answer accuracy"),
            ("hallucination_rate", "hallucination rate"),
            ("abstention_rate", "abstention rate"),
        ],
    )
    style_axes(right, xlabel, "rate", "Answer quality")

    # Tick only at the values actually swept, on either scale: an auto-chosen
    # tick at, say, 450 would invite reading the curve between conditions that
    # were never run.
    for ax in (left, right):
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off()

    fig.suptitle(title, fontsize=13, color=INK, x=0.06, ha="left", y=0.99)
    fig.text(0.06, 0.005, f"n = {n} questions per condition", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / out_name
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return rows


def strategy_figure():
    """Grouped bars for experiment 1b, whose conditions are categorical.

    `sweep_figure` cannot serve this one: its x values are strategy names, not
    numbers, and a line joining `fixed` to `recursive` would imply a continuum
    between two discrete chunkers. Retrieval and answer metrics stay in separate
    panels here for the same reason they do everywhere else.
    """
    path = RESULTS / "chunk_strategy_sensitivity.csv"
    if not path.exists():
        print("  skipped chunk_strategy_sensitivity.csv (not found -- run the experiment first)")
        return None

    rows = read_csv(path)
    labels = [r["value"] for r in rows]
    positions = range(len(labels))

    panels = (
        ("Retrieval quality", [("hit_rate_at_k", "hit rate@k"), ("mrr", "MRR"),
                               ("precision_at_k", "precision@k")]),
        ("Answer quality", [("accuracy", "accuracy"), ("abstention_rate", "abstention"),
                            ("hallucination_rate", "hallucination")]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("#fcfcfb")

    width = 0.26
    for ax, (title, keys_labels) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        for i, (key, label) in enumerate(keys_labels):
            offset = (i - 1) * width
            values = [float(r[key]) for r in rows]
            ax.bar(
                [p + offset for p in positions], values, width=width, label=label,
                color=SERIES_STYLE[i]["color"], zorder=3,
            )
            for p, value in zip(positions, values):
                ax.annotate(
                    f"{value:.2f}", xy=(p + offset, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=MUTED,
                )
        style_axes(ax, "chunking strategy", "score", title)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")

    n = rows[0]["n"]
    size = rows[0].get("mean_chunk_chars", "")
    fig.suptitle(
        "Chunking strategy at a fixed chunk size", fontsize=13, color=INK,
        x=0.06, ha="left", y=0.99,
    )
    fig.text(
        0.06, 0.005,
        f"n = {n} questions per condition, chunk size held at 128 characters"
        + (f" (mean chunk: {float(size):.0f} chars)" if size else ""),
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "chunk_strategy_sensitivity.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return rows


def long_context_figure():
    """Bar chart across the four conditions, with the position probe grouped."""
    path = RESULTS / "long_context_vs_retrieval.csv"
    if not path.exists():
        print("  skipped long_context_vs_retrieval.csv (not found -- run the experiment first)")
        return None

    rows = read_csv(path)
    labels = [r["condition"] for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]
    # The RAG condition is a different kind of thing from the three stuffed
    # positions, so it is colored apart rather than reading as a fourth position.
    colors = [BLUE if label == "rag" else ORANGE for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.bar(labels, accuracy, color=colors, width=0.6, zorder=3)
    for bar, value in zip(bars, accuracy):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )

    style_axes(ax, "condition", "answer accuracy", "Retrieval vs. full-context stuffing")
    ax.grid(axis="x", visible=False)
    n = rows[0]["n"]
    fig.text(
        0.02, 0.005,
        f"n = {n} questions per condition. Blue: retrieval. Orange: whole corpus stuffed, "
        "gold document at the named position.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "long_context_vs_retrieval.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return rows


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render rows as a markdown table -- the required table view for accessibility."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if value in ("", None):
                cells.append("--")
            else:
                try:
                    cells.append(f"{float(value):.3f}" if "." in str(value) else str(value))
                except (TypeError, ValueError):
                    cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SWEEP_COLUMNS = [
    ("value", "condition"),
    ("n", "n"),
    ("hit_rate_at_k", "hit rate@k"),
    ("precision_at_k", "precision@k"),
    ("mrr", "MRR"),
    # Below 1.0, some questions have no answer-bearing chunk anywhere in the
    # corpus, which caps hit rate independently of the retriever. Reported
    # beside hit rate so the two are never read apart.
    ("mean_relevant_chunks_in_corpus", "gold chunks avail."),
    ("accuracy", "accuracy"),
    ("abstention_rate", "abstention"),
    ("hallucination_rate", "hallucination"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="Print markdown tables too.")
    args = parser.parse_args()

    print("Building figures...")
    chunk = sweep_figure(
        "chunk_size_sensitivity.csv",
        "chunk size (characters)",
        "Chunk-size sensitivity",
        "chunk_size_sensitivity.png",
        # Linear, not log. The swept sizes are near-evenly spaced above 200, and
        # a log axis compresses the top end until the 700 and 800 tick labels
        # overprint -- exactly where the interesting recovery happens.
        log_x=False,
    )
    strategy = strategy_figure()
    distractor_k1 = sweep_figure(
        "distractor_sensitivity_k1.csv",
        "distractor ratio (distractors per relevant document)",
        "Distractor sensitivity at k = 1",
        "distractor_sensitivity_k1.png",
    )
    distractor = sweep_figure(
        "distractor_sensitivity.csv",
        "distractor ratio (distractors per relevant document)",
        "Distractor sensitivity",
        "distractor_sensitivity.png",
    )
    long_context = long_context_figure()

    if args.tables:
        for name, rows, columns in (
            ("Chunk-size sensitivity", chunk, SWEEP_COLUMNS),
            ("Chunking strategy", strategy, SWEEP_COLUMNS),
            ("Distractor sensitivity", distractor, SWEEP_COLUMNS),
            ("Distractor sensitivity at k=1", distractor_k1, SWEEP_COLUMNS),
            (
                "Long-context vs. retrieval",
                long_context,
                [
                    ("condition", "condition"),
                    ("n", "n"),
                    ("accuracy", "accuracy"),
                    ("abstention_rate", "abstention"),
                    ("hallucination_rate", "hallucination"),
                    ("mean_input_tokens", "mean input tokens"),
                ],
            ),
        ):
            if rows:
                print(f"\n### {name}\n")
                print(markdown_table(rows, columns))

    return 0


if __name__ == "__main__":
    sys.exit(main())
