"""Prompt templates for answering and for judging.

All three experiments share these templates so that a difference in results is
attributable to the variable under test rather than to prompt wording.

The answer prompt requires the model to emit `NOT IN CONTEXT` when the context
does not contain the answer. That turns abstention into a measurable outcome and
separates two very different failure modes: a model that correctly declines when
retrieval missed, versus one that confidently invents a value from a distractor.
"""

from __future__ import annotations

from src.chunking.base import Chunk
from src.ingestion.documents import Document

ABSTAIN_TOKEN = "NOT IN CONTEXT"

ANSWER_SYSTEM = (
    "You answer questions strictly from the provided context.\n"
    "Rules:\n"
    f"1. If the context does not contain the answer, reply with exactly: {ABSTAIN_TOKEN}\n"
    "2. Never use outside knowledge, and never guess.\n"
    "3. The context may describe several similarly named systems. Answer only about "
    "the system named in the question.\n"
    "4. Answer in one short sentence."
)

JUDGE_SYSTEM = (
    "You grade whether a predicted answer matches a reference answer.\n"
    "Reply with exactly one word: CORRECT or INCORRECT.\n"
    "Grade on semantic equivalence of the key fact, not on wording, punctuation, "
    "or sentence structure. '36 hours' and 'every 36 hours' are the same answer.\n"
    "A prediction that states a different value, names a different system, or "
    "declines to answer is INCORRECT."
)


def format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a numbered context block."""
    return "\n\n".join(
        f"[{i}] (from {chunk.doc_id})\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )


def format_documents(documents: list[Document]) -> str:
    """Render whole documents as a numbered context block, for the stuffed condition."""
    return "\n\n".join(
        f"[{i}] (document {doc.doc_id})\n{doc.text}" for i, doc in enumerate(documents, start=1)
    )


def build_answer_prompt(question: str, context: str) -> str:
    """The single user-message shape used by both the RAG and stuffed conditions.

    Keeping them identical is what makes the long-context comparison fair: only
    the contents of `context` differ, never the framing around it.
    """
    return f"Context:\n{context}\n\nQuestion: {question}"


def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Predicted answer: {prediction}\n\n"
        "Is the predicted answer correct? Reply CORRECT or INCORRECT."
    )
