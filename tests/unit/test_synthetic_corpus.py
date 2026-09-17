"""Corpus generation tests.

Two properties matter most: the corpus is reproducible from its seed, and every
question has exactly one correct answer span anywhere in the corpus. If a
distractor accidentally answered a question, the distractor experiment would be
measuring nothing.
"""

from __future__ import annotations

import pytest

from src.config import CorpusConfig
from src.ingestion.synthetic import generate_corpus


def test_same_seed_produces_identical_corpus():
    config = CorpusConfig(n_relevant_docs=8, distractor_ratio=1.0, seed=42)
    first_docs, first_qs = generate_corpus(config)
    second_docs, second_qs = generate_corpus(config)
    assert [d.text for d in first_docs] == [d.text for d in second_docs]
    assert [q.question for q in first_qs] == [q.question for q in second_qs]
    assert [q.answer for q in first_qs] == [q.answer for q in second_qs]


def test_different_seeds_produce_different_corpora():
    a, _ = generate_corpus(CorpusConfig(n_relevant_docs=8, seed=1))
    b, _ = generate_corpus(CorpusConfig(n_relevant_docs=8, seed=2))
    assert [d.text for d in a] != [d.text for d in b]


@pytest.mark.parametrize("ratio,expected_distractors", [(0.0, 0), (0.5, 4), (1.0, 8), (2.0, 16)])
def test_distractor_ratio_controls_corpus_composition(ratio, expected_distractors):
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=8, distractor_ratio=ratio, seed=7)
    )
    assert sum(1 for d in docs if d.is_distractor) == expected_distractors
    assert sum(1 for d in docs if not d.is_distractor) == 8
    # The eval set must not grow with the distractor ratio; it is the fixed
    # control across every condition of that experiment.
    assert len(questions) == 8


def test_every_question_has_exactly_one_gold_span():
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=16, distractor_ratio=2.0, seed=11)
    )
    by_id = {d.doc_id: d for d in docs}
    for question in questions:
        doc = by_id[question.gold_doc_id]
        assert doc.fact(question.gold_fact_id) is not None
        assert len(doc.fact_spans) == 1


def test_answer_appears_in_its_gold_span():
    docs, questions = generate_corpus(CorpusConfig(n_relevant_docs=16, seed=13))
    by_id = {d.doc_id: d for d in docs}
    for question in questions:
        span = by_id[question.gold_doc_id].fact(question.gold_fact_id)
        assert question.answer in span.text


def test_no_distractor_contains_an_answer_to_any_question():
    """The core integrity property of the distractor experiment."""
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=16, distractor_ratio=2.0, seed=17)
    )
    distractors = [d for d in docs if d.is_distractor]
    assert distractors

    for distractor in distractors:
        assert distractor.fact_spans == ()
        for question in questions:
            # A distractor describes a different system, so the question's
            # subject should never appear in it.
            assert question.topic not in distractor.text


def test_spans_are_byte_aligned():
    docs, _ = generate_corpus(CorpusConfig(n_relevant_docs=16, distractor_ratio=1.0, seed=19))
    for doc in docs:
        doc.validate_spans()  # raises on misalignment


def test_fact_is_not_always_first_sentence():
    """Position within the document must not be a hidden confound."""
    docs, questions = generate_corpus(CorpusConfig(n_relevant_docs=24, seed=23))
    by_id = {d.doc_id: d for d in docs}
    starts = [by_id[q.gold_doc_id].fact(q.gold_fact_id).start for q in questions]
    assert len(set(starts)) > 1
    assert any(s > 0 for s in starts)


def test_questions_are_balanced_across_attribute_types():
    _, questions = generate_corpus(CorpusConfig(n_relevant_docs=32, seed=29))
    attributes = {q.gold_fact_id.split(":")[1] for q in questions}
    assert len(attributes) >= 6


def test_exhausting_the_name_pool_is_a_clear_error():
    with pytest.raises(ValueError, match="unique system names"):
        generate_corpus(CorpusConfig(n_relevant_docs=500, distractor_ratio=1.0))
