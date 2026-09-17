"""Document and fact-span data model.

The central design decision of this project lives here.

Retrieval ground truth cannot be stored as "chunk 7 is the right answer",
because the chunk-size experiment re-chunks the corpus on every condition and
chunk 7 means something different each time. Instead each document records the
exact **character span** of any planted fact. A chunk is then gold for a
question if it comes from the right document and its character range overlaps
that fact's span.

That makes gold labels invariant to chunking strategy, which is the only way
recall@k stays comparable across chunk sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FactSpan:
    """The character range of a planted, question-answering fact within a document."""

    fact_id: str
    start: int
    end: int
    text: str

    def overlaps(self, start: int, end: int) -> bool:
        """True if [start, end) intersects this span at all."""
        return start < self.end and end > self.start

    def overlap_chars(self, start: int, end: int) -> int:
        """Number of characters shared with [start, end)."""
        return max(0, min(end, self.end) - max(start, self.start))


@dataclass(frozen=True)
class Document:
    """A corpus document, possibly carrying planted facts."""

    doc_id: str
    text: str
    topic: str = ""
    is_distractor: bool = False
    fact_spans: tuple[FactSpan, ...] = field(default_factory=tuple)
    metadata: dict = field(default_factory=dict)

    def fact(self, fact_id: str) -> FactSpan | None:
        for span in self.fact_spans:
            if span.fact_id == fact_id:
                return span
        return None

    def validate_spans(self) -> None:
        """Assert every recorded span actually points at its own text.

        Cheap insurance against an off-by-one in corpus generation silently
        corrupting every retrieval metric downstream.
        """
        for span in self.fact_spans:
            actual = self.text[span.start : span.end]
            if actual != span.text:
                raise ValueError(
                    f"Fact span {span.fact_id} in {self.doc_id} is misaligned.\n"
                    f"  expected: {span.text!r}\n"
                    f"  actual:   {actual!r}"
                )


@dataclass(frozen=True)
class EvalQuestion:
    """One question in the fixed eval set.

    `gold_doc_id` + `gold_fact_id` together identify the span that must be
    retrieved; `answer` is what a correct response must convey.
    """

    question_id: str
    question: str
    answer: str
    gold_doc_id: str
    gold_fact_id: str
    topic: str = ""
