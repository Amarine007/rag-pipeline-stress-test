"""Experiment 3 -- Long-context stuffing vs. retrieved context.

Four conditions over one fixed corpus and one fixed eval set:

    rag          retrieval, top-k chunks in the prompt
    full_start   whole corpus stuffed, gold document near the beginning
    full_middle  whole corpus stuffed, gold document near the middle
    full_end     whole corpus stuffed, gold document near the end

The three stuffed conditions are the lost-in-the-middle probe. They contain the
identical set of documents and very nearly the identical token count, differing
only in *where* the answer-bearing document sits. If accuracy dips for
`full_middle` relative to `full_start` and `full_end`, position is what moved
it, because nothing else did.

Retrieval metrics are undefined for the stuffed conditions -- there is no
retrieval step to score -- and are recorded as null rather than zero. A zero
would read as "retrieval failed" when the truth is "retrieval did not happen".

    python experiments/long_context_vs_retrieval/run.py
    python experiments/long_context_vs_retrieval/run.py --conditions rag full_middle
    python experiments/long_context_vs_retrieval/run.py --max-questions 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._harness import write_csv  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.embeddings.embedder import Embedder  # noqa: E402
from src.evaluation.answer_scorer import aggregate_answers  # noqa: E402
from src.evaluation.metrics import aggregate_retrieval, score_retrieval  # noqa: E402
from src.generation.claude_client import ClaudeClient  # noqa: E402
from src.generation.prompts import format_chunks, format_documents  # noqa: E402
from src.ingestion.documents import Document, EvalQuestion  # noqa: E402
from src.ingestion.synthetic import subsample_questions  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "long_context_vs_retrieval.yaml"

POSITIONS = {"full_start": 0.0, "full_middle": 0.5, "full_end": 1.0}
ALL_CONDITIONS = ["rag", *POSITIONS]


def order_documents(
    documents: list[Document], gold_doc_id: str, position: float
) -> list[Document]:
    """Return all documents with the gold one placed at a fractional position.

    The non-gold documents keep their corpus order in every condition, so the
    only difference between `full_start`, `full_middle`, and `full_end` is the
    index of a single document.
    """
    others = [d for d in documents if d.doc_id != gold_doc_id]
    gold = next(d for d in documents if d.doc_id == gold_doc_id)
    index = min(len(others), max(0, round(position * len(others))))
    return [*others[:index], gold, *others[index:]]


def run_condition(
    condition: str,
    pipeline: RAGPipeline,
    questions: list[EvalQuestion],
    show_progress: bool = True,
) -> dict:
    """Run every eval question under one condition and score it."""
    documents_by_id = {d.doc_id: d for d in pipeline.documents}
    started = time.time()

    retrieval_results = []
    answer_scores = []
    per_question = []

    for i, question in enumerate(questions, start=1):
        if show_progress:
            print(f"  [{i}/{len(questions)}] {question.question_id}", end="\r", flush=True)

        gold_position = None
        if condition == "rag":
            hits = pipeline.retrieve(question.question)
            retrieval = score_retrieval(question, hits, pipeline.chunks, documents_by_id)
            retrieval_results.append(retrieval)
            context = format_chunks([h.chunk for h in hits])
            retrieved_hit = retrieval.hit
        else:
            ordered = order_documents(
                pipeline.documents, question.gold_doc_id, POSITIONS[condition]
            )
            gold_position = next(
                i for i, d in enumerate(ordered) if d.doc_id == question.gold_doc_id
            )
            context = format_documents(ordered)
            # The gold document is present by construction in every stuffed
            # condition; there is no retrieval step that could have missed it.
            retrieved_hit = None

        response = pipeline.generate_response(question.question, context)
        score = pipeline.judge.score(
            question.question_id, question.question, response.text, question.answer
        )
        answer_scores.append(score)

        per_question.append(
            {
                "condition": condition,
                "question_id": question.question_id,
                "reference": question.answer,
                "prediction": response.text,
                "retrieved_hit": retrieved_hit,
                "gold_doc_position": gold_position,
                "n_documents_in_context": None if condition == "rag" else len(pipeline.documents),
                "input_tokens": response.input_tokens,
                "answer_correct": score.correct,
                "answer_abstained": score.abstained,
                "answer_hallucinated": score.hallucinated,
            }
        )

    answers = aggregate_answers(answer_scores)
    # Null, not zero: retrieval did not happen in the stuffed conditions.
    retrieval = aggregate_retrieval(retrieval_results) if condition == "rag" else {
        "n": len(questions),
        "hit_rate_at_k": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "mrr": None,
        "mean_relevant_chunks_in_corpus": None,
    }

    mean_input_tokens = (
        round(sum(r["input_tokens"] for r in per_question) / len(per_question), 1)
        if per_question
        else 0.0
    )

    if show_progress:
        print(
            f"    {condition:<12} acc={answers['accuracy']:.3f} "
            f"abstain={answers['abstention_rate']:.3f} "
            f"halluc={answers['hallucination_rate']:.3f} "
            f"mean_input_tokens={mean_input_tokens:.0f}  (n={answers['n']})",
            flush=True,
        )

    return {
        "condition": {"experiment": "long_context_vs_retrieval", "value": condition},
        "retrieval_metrics": retrieval,
        "answer_metrics": answers,
        "mean_input_tokens": mean_input_tokens,
        "per_question": per_question,
        "corpus_stats": pipeline.corpus_stats(),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conditions", nargs="*", default=ALL_CONDITIONS)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N questions, for a reduced pilot run. "
            "The corpus is unchanged; only the eval set shrinks."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    # A pilot writes to its own files so it cannot overwrite a full run's results.
    suffix = f"-pilot{args.max_questions}" if args.max_questions else ""
    log_path = PROJECT_ROOT / "results" / "logs" / f"{config.name}{suffix}.jsonl"
    csv_path = PROJECT_ROOT / "results" / f"{config.name}{suffix}.csv"

    client = ClaudeClient(use_cache=config.generation.use_cache)
    pipeline = RAGPipeline(config, client=client, embedder=Embedder(config.embedding))
    pipeline.build_corpus()
    pipeline.build_index()

    questions = subsample_questions(pipeline.questions, args.max_questions)

    stats = pipeline.corpus_stats()
    print(f"Experiment: {config.name}")
    if args.max_questions:
        print(f"PILOT RUN:  first {args.max_questions} questions only -- not a full result")
    print(f"Corpus:     {stats['n_documents']} documents "
          f"({stats['n_relevant_documents']} relevant, "
          f"{stats['n_distractor_documents']} distractors), {stats['n_chunks']} chunks")
    print(f"Questions:  {len(questions)}")
    print(f"Model:      {config.generation.model}")
    print(f"Conditions: {', '.join(args.conditions)}\n")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    rows = []
    for condition in args.conditions:
        if condition not in ALL_CONDITIONS:
            raise SystemExit(f"Unknown condition {condition!r}. Choose from {ALL_CONDITIONS}.")
        result = run_condition(condition, pipeline, questions)

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

        rows.append(
            {
                "condition": condition,
                "n": result["answer_metrics"]["n"],
                "accuracy": result["answer_metrics"]["accuracy"],
                "abstention_rate": result["answer_metrics"]["abstention_rate"],
                "hallucination_rate": result["answer_metrics"]["hallucination_rate"],
                "hit_rate_at_k": result["retrieval_metrics"]["hit_rate_at_k"],
                "mrr": result["retrieval_metrics"]["mrr"],
                "mean_input_tokens": result["mean_input_tokens"],
            }
        )

    write_csv(rows, csv_path)
    print(f"\nWrote {len(rows)} conditions to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
