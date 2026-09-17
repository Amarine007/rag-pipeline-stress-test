"""Tests for the lost-in-the-middle positioning logic.

If `order_documents` did not hold the context constant apart from the gold
document's index, the three stuffed conditions would differ in content as well
as position and the experiment would measure nothing in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.long_context_vs_retrieval.run import order_documents  # noqa: E402
from src.ingestion.documents import Document  # noqa: E402

DOCS = [Document(doc_id=f"doc-{i:04d}", text=f"Document {i}.") for i in range(10)]
GOLD = "doc-0004"


@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
def test_every_document_appears_exactly_once(position):
    ordered = order_documents(DOCS, GOLD, position)
    assert len(ordered) == len(DOCS)
    assert {d.doc_id for d in ordered} == {d.doc_id for d in DOCS}


def test_gold_lands_at_the_requested_position():
    assert order_documents(DOCS, GOLD, 0.0)[0].doc_id == GOLD
    assert order_documents(DOCS, GOLD, 1.0)[-1].doc_id == GOLD

    middle = order_documents(DOCS, GOLD, 0.5)
    index = next(i for i, d in enumerate(middle) if d.doc_id == GOLD)
    # Not at either edge -- that is the whole content of the "middle" condition.
    assert 0 < index < len(middle) - 1


def test_non_gold_order_is_identical_across_positions():
    """The only thing that may change between conditions is the gold document's index."""
    orders = [
        [d.doc_id for d in order_documents(DOCS, GOLD, p) if d.doc_id != GOLD]
        for p in (0.0, 0.5, 1.0)
    ]
    assert orders[0] == orders[1] == orders[2]


def test_context_content_is_identical_across_positions():
    """Same documents, same text, same token count -- only the ordering differs."""
    texts = [
        sorted(d.text for d in order_documents(DOCS, GOLD, p)) for p in (0.0, 0.5, 1.0)
    ]
    assert texts[0] == texts[1] == texts[2]


def test_single_document_corpus_is_handled():
    single = [DOCS[0]]
    for position in (0.0, 0.5, 1.0):
        assert [d.doc_id for d in order_documents(single, "doc-0000", position)] == ["doc-0000"]
