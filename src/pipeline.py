"""End-to-end pipeline orchestration.

Wires the hand-written stages together: corpus -> chunks -> embeddings -> FAISS
index -> retrieval -> generation -> evaluation. Experiments drive this class;
they do not reimplement any of it.

Retrieval and answer quality are evaluated in the same pass but recorded as
separate metric blocks, and `RunResult` never exposes a combined score.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.chunking.base import Chunk, chunk_corpus, verify_offsets
from src.chunking.strategies import build_chunker
from src.config import ExperimentConfig
from src.embeddings.embedder import Embedder
from src.evaluation.answer_scorer import AnswerScore, LLMJudge, aggregate_answers
from src.evaluation.metrics import (
    QuestionRetrievalResult,
    aggregate_retrieval,
    score_retrieval,
)
from src.generation.claude_client import ClaudeClient
from src.generation.prompts import (
    ANSWER_SYSTEM,
    build_answer_prompt,
    format_chunks,
    format_documents,
)
from src.ingestion.documents import Document, EvalQuestion
from src.ingestion.synthetic import generate_corpus
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import SearchHit


@dataclass
class RunResult:
    """Everything one condition produced. Retrieval and answers stay separate."""

    condition: dict
    retrieval_metrics: dict = field(default_factory=dict)
    answer_metrics: dict = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)
    corpus_stats: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """A configured pipeline over one corpus."""

    def __init__(
        self,
        config: ExperimentConfig,
        client: ClaudeClient | None = None,
        embedder: Embedder | None = None,
    ):
        self.config = config
        self.client = client or ClaudeClient(use_cache=config.generation.use_cache)
        self.embedder = embedder or Embedder(config.embedding)
        self.judge = LLMJudge(self.client, config.judge)

        self.documents: list[Document] = []
        self.questions: list[EvalQuestion] = []
        self.chunks: list[Chunk] = []
        self.retriever: Retriever | None = None

    # -- construction ----------------------------------------------------

    def build_corpus(self) -> tuple[list[Document], list[EvalQuestion]]:
        """Generate the seeded synthetic corpus and its eval questions."""
        self.documents, self.questions = generate_corpus(self.config.corpus)
        return self.documents, self.questions

    def set_corpus(self, documents: list[Document], questions: list[EvalQuestion]) -> None:
        """Use a corpus built elsewhere -- e.g. one shared across conditions."""
        self.documents, self.questions = documents, questions

    def build_index(self, show_progress: bool = False) -> None:
        """Chunk the corpus, embed it, and load it into FAISS."""
        if not self.documents:
            raise RuntimeError("No corpus. Call build_corpus() or set_corpus() first.")

        chunker = build_chunker(
            self.config.chunking.strategy,
            self.config.chunking.chunk_size,
            self.config.chunking.chunk_overlap,
        )
        self.chunks = chunk_corpus(chunker, self.documents)
        # Offsets underpin every retrieval metric, so they are checked on every
        # run rather than trusted.
        verify_offsets(self.chunks, self.documents)

        self.retriever = Retriever(self.embedder, self.config.retrieval)
        self.retriever.index_chunks(self.chunks, show_progress=show_progress)

    # -- inference -------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> list[SearchHit]:
        if self.retriever is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self.retriever.retrieve(question, k)

    def generate(self, question: str, context: str) -> str:
        response = self.client.complete(
            prompt=build_answer_prompt(question, context),
            system=ANSWER_SYSTEM,
            model=self.config.generation.model,
            max_tokens=self.config.generation.max_tokens,
            effort=self.config.generation.effort,
            temperature=self.config.generation.temperature,
        )
        return response.text

    def answer_with_retrieval(self, question: str, k: int | None = None) -> tuple[str, list[SearchHit]]:
        hits = self.retrieve(question, k)
        return self.generate(question, format_chunks([h.chunk for h in hits])), hits

    def answer_with_full_context(self, question: str, documents: list[Document]) -> str:
        """The stuffed condition: no retrieval, the whole corpus in the prompt."""
        return self.generate(question, format_documents(documents))

    # -- evaluation ------------------------------------------------------

    def evaluate(
        self,
        condition: dict | None = None,
        questions: list[EvalQuestion] | None = None,
        show_progress: bool = False,
    ) -> RunResult:
        """Run the full eval set through retrieval + generation and score both."""
        if self.retriever is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        eval_questions = questions if questions is not None else self.questions
        documents_by_id = {d.doc_id: d for d in self.documents}

        started = time.time()
        retrieval_results: list[QuestionRetrievalResult] = []
        answer_scores: list[AnswerScore] = []
        per_question: list[dict] = []

        for i, question in enumerate(eval_questions, start=1):
            if show_progress:
                print(f"  [{i}/{len(eval_questions)}] {question.question_id}", flush=True)

            hits = self.retrieve(question.question)
            retrieval = score_retrieval(question, hits, self.chunks, documents_by_id)
            retrieval_results.append(retrieval)

            prediction = self.generate(question.question, format_chunks([h.chunk for h in hits]))
            score = self.judge.score(
                question.question_id, question.question, prediction, question.answer
            )
            answer_scores.append(score)

            per_question.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "reference": question.answer,
                    "prediction": prediction,
                    # Retrieval outcome
                    "retrieved_hit": retrieval.hit,
                    "first_relevant_rank": retrieval.first_relevant_rank,
                    "precision_at_k": retrieval.precision_at_k,
                    "recall_at_k": retrieval.recall_at_k,
                    "reciprocal_rank": retrieval.reciprocal_rank,
                    "retrieved_chunk_ids": [h.chunk.chunk_id for h in hits],
                    # Answer outcome
                    "answer_correct": score.correct,
                    "answer_abstained": score.abstained,
                    "answer_hallucinated": score.hallucinated,
                }
            )

        return RunResult(
            condition=condition or {},
            retrieval_metrics=aggregate_retrieval(retrieval_results),
            answer_metrics=aggregate_answers(answer_scores),
            per_question=per_question,
            corpus_stats=self.corpus_stats(),
            elapsed_seconds=round(time.time() - started, 2),
        )

    def corpus_stats(self) -> dict:
        n_distractors = sum(1 for d in self.documents if d.is_distractor)
        chunk_lengths = [len(c) for c in self.chunks]
        return {
            "n_documents": len(self.documents),
            "n_relevant_documents": len(self.documents) - n_distractors,
            "n_distractor_documents": n_distractors,
            "n_chunks": len(self.chunks),
            "mean_chunk_chars": (
                round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0.0
            ),
        }


def write_result(result: RunResult, path: Path) -> None:
    """Append one condition's result to a JSONL run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
