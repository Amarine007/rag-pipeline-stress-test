"""Sweep execution: run one config across the values of its single variable.

Generic machinery lives here so each experiment's `run.py` stays a thin
statement of *what* it varies rather than a reimplementation of *how* to vary
it.

Two details this handles that are easy to get wrong by hand:

- **The eval question set is held fixed across conditions.** When a sweep
  changes the corpus (the distractor experiment), the questions are taken from
  the baseline corpus and reused, so every condition answers the identical
  questions about the identical 40 relevant documents. Regenerating questions
  per condition would silently change the test between conditions and make them
  incomparable.
- **The embedder is shared across conditions.** Loading the model once avoids
  paying the torch import per condition, and guarantees every condition embeds
  with the same weights.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig, apply_override
from src.embeddings.embedder import Embedder
from src.generation.claude_client import ClaudeClient
from src.ingestion.documents import Document, EvalQuestion
from src.ingestion.synthetic import generate_corpus, subsample_questions
from src.pipeline import RAGPipeline, RunResult


def _corpus_for(
    config: ExperimentConfig,
    baseline_questions: list[EvalQuestion] | None,
) -> tuple[list[Document], list[EvalQuestion]]:
    """Build this condition's corpus, keeping the eval questions fixed.

    Distractors are appended after the relevant documents and the generator is
    seeded identically, so the relevant documents and their planted facts are
    byte-identical across every distractor ratio. Only the surrounding noise
    changes -- which is the whole point of the controlled comparison.
    """
    documents, questions = generate_corpus(config.corpus)
    if baseline_questions is not None:
        _assert_questions_still_valid(documents, baseline_questions)
        questions = baseline_questions
    return documents, questions


def _assert_questions_still_valid(
    documents: list[Document], questions: list[EvalQuestion]
) -> None:
    """Fail loudly if a condition's corpus no longer contains the fixed questions' answers."""
    by_id = {d.doc_id: d for d in documents}
    for question in questions:
        document = by_id.get(question.gold_doc_id)
        if document is None or document.fact(question.gold_fact_id) is None:
            raise RuntimeError(
                f"Question {question.question_id} references {question.gold_fact_id}, which is "
                "missing from this condition's corpus. The eval set must be identical across "
                "conditions or the comparison is meaningless."
            )


def run_sweep(
    config: ExperimentConfig,
    output_path: Path,
    values: Iterable[Any] | None = None,
    show_progress: bool = True,
    max_questions: int | None = None,
) -> list[RunResult]:
    """Run every condition of `config`'s sweep, appending results to `output_path`.

    `max_questions` truncates the eval set for a reduced pilot run. It shrinks
    the questions only, never the corpus, and applies identically to every
    condition -- so conditions stay comparable with each other, though a pilot's
    numbers are not comparable with a full run's.

    Returns the results in sweep order.
    """
    if not config.sweep:
        raise ValueError(f"Config {config.name!r} defines no sweep.")

    sweep_key, default_values = next(iter(config.sweep.items()))
    sweep_values = list(values) if values is not None else list(default_values)

    # Fresh log per invocation, so a rerun never interleaves with stale results.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # Shared across conditions: one model load, one cache.
    embedder = Embedder(config.embedding)
    client = ClaudeClient(
        use_cache=config.generation.use_cache,
        use_batch=config.generation.use_batch,
    )

    # The eval set comes from the baseline corpus and never changes afterwards.
    _, baseline_questions = generate_corpus(config.corpus)
    baseline_questions = subsample_questions(baseline_questions, max_questions)

    results: list[RunResult] = []
    for i, value in enumerate(sweep_values, start=1):
        if show_progress:
            print(f"\n[{i}/{len(sweep_values)}] {sweep_key} = {value}", flush=True)

        condition_config = apply_override(config, sweep_key, value)
        documents, questions = _corpus_for(condition_config, baseline_questions)

        pipeline = RAGPipeline(condition_config, client=client, embedder=embedder)
        pipeline.set_corpus(documents, questions)
        pipeline.build_index()

        result = pipeline.evaluate(
            condition={
                "experiment": config.name,
                "variable": sweep_key,
                "value": value,
                # Recorded so a pilot log can never be mistaken for a full run.
                "pilot_max_questions": max_questions,
            }
        )
        results.append(result)

        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        if show_progress:
            _print_condition_summary(result)

    return results


def _print_condition_summary(result: RunResult) -> None:
    retrieval, answers = result.retrieval_metrics, result.answer_metrics
    print(
        f"    retrieval: hit@k={retrieval['hit_rate_at_k']:.3f} "
        f"mrr={retrieval['mrr']:.3f} p@k={retrieval['precision_at_k']:.3f}  |  "
        f"answers: acc={answers['accuracy']:.3f} "
        f"abstain={answers['abstention_rate']:.3f} "
        f"halluc={answers['hallucination_rate']:.3f}  (n={answers['n']})",
        flush=True,
    )


def summarize(results: list[RunResult]) -> list[dict]:
    """Flatten results into rows for a table or figure, keeping the two metric families apart."""
    rows = []
    for result in results:
        rows.append(
            {
                "variable": result.condition.get("variable"),
                "value": result.condition.get("value"),
                "n": result.answer_metrics.get("n", 0),
                # Retrieval quality
                "hit_rate_at_k": result.retrieval_metrics.get("hit_rate_at_k"),
                "precision_at_k": result.retrieval_metrics.get("precision_at_k"),
                "recall_at_k": result.retrieval_metrics.get("recall_at_k"),
                "mrr": result.retrieval_metrics.get("mrr"),
                "mean_relevant_chunks_in_corpus": result.retrieval_metrics.get(
                    "mean_relevant_chunks_in_corpus"
                ),
                # Answer quality
                "accuracy": result.answer_metrics.get("accuracy"),
                "abstention_rate": result.answer_metrics.get("abstention_rate"),
                "hallucination_rate": result.answer_metrics.get("hallucination_rate"),
                # Context
                "n_chunks": result.corpus_stats.get("n_chunks"),
                "n_documents": result.corpus_stats.get("n_documents"),
                "mean_chunk_chars": result.corpus_stats.get("mean_chunk_chars"),
            }
        )
    return rows
