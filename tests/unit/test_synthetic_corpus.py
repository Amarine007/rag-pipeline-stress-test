"""Corpus generation tests.

Two properties matter most: the corpus is reproducible from its seed, and every
question has exactly one correct answer span anywhere in the corpus. If a
distractor accidentally answered a question, the distractor experiment would be
measuring nothing.
"""

from __future__ import annotations

import pytest

from src.config import CorpusConfig
from src.ingestion.synthetic import (
    _ATTRIBUTES,
    _all_names,
    _validate_value_pools,
    generate_corpus,
    subsample_questions,
)


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
    # Sized from the pool rather than hardcoded, so growing the pools does not
    # silently turn this into a test that exercises nothing.
    too_many = len(_all_names()) + 1
    with pytest.raises(ValueError, match="unique system names"):
        generate_corpus(CorpusConfig(n_relevant_docs=too_many, distractor_ratio=0.0))


def test_name_pool_covers_the_widest_planned_experiment():
    """The distractor sweep at ratio 4.0 needs five unique names per question.

    This is the constraint that couples sample size to the distractor
    experiment: raising n_relevant_docs without checking it fails the run at the
    widest ratio only, several conditions into a sweep that has already spent money.
    """
    widest_ratio = 4.0
    questions = 120
    assert len(_all_names()) >= questions * (1 + widest_ratio)


# -- Multi-sentence facts ------------------------------------------------
#
# The point of a multi-sentence fact is that no single sentence is sufficient:
# one sentence binds the system name, another states the value. These tests pin
# that property, because a template edit that let the value sentence repeat the
# system name would silently restore the single-sentence behaviour and make the
# chunk-size results overstate the effect again.


@pytest.mark.parametrize("n_sentences", [2, 3])
def test_multi_sentence_fact_spans_several_sentences(n_sentences):
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=8, fact_sentences=n_sentences, seed=11)
    )
    by_id = {d.doc_id: d for d in docs}
    for question in questions:
        span = by_id[question.gold_doc_id].fact(question.gold_fact_id)
        assert span.text.count(". ") == n_sentences - 1
        assert span.has_required_span


def test_value_sentence_carries_the_answer_but_not_the_system_name():
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=8, fact_sentences=3, seed=13)
    )
    by_id = {d.doc_id: d for d in docs}
    for question in questions:
        document = by_id[question.gold_doc_id]
        span = document.fact(question.gold_fact_id)
        # The value lives in the required sentence...
        assert question.answer in span.required_text
        # ...and the system name does not, so a chunk holding only that
        # sentence cannot say which system the value belongs to.
        assert question.topic not in span.required_text
        # The name is bound elsewhere in the passage.
        assert question.topic in span.text


def test_single_sentence_facts_record_no_required_subrange():
    """The extra criterion must be a no-op for the one-sentence corpus."""
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=8, fact_sentences=1, seed=17)
    )
    by_id = {d.doc_id: d for d in docs}
    for question in questions:
        assert not by_id[question.gold_doc_id].fact(question.gold_fact_id).has_required_span


def test_distractors_match_relevant_documents_in_shape():
    """A distractor must not be identifiable by being shorter or flatter."""
    docs, _ = generate_corpus(
        CorpusConfig(n_relevant_docs=12, distractor_ratio=1.0, fact_sentences=3, seed=19)
    )
    relevant = [d for d in docs if not d.is_distractor]
    distractors = [d for d in docs if d.is_distractor]
    assert distractors and relevant
    # Same sentence budget: the decoy passage takes the planted fact's place.
    assert {d.text.count(".") for d in distractors} == {d.text.count(".") for d in relevant}
    assert all(d.fact_spans == () for d in distractors)


def test_invalid_fact_sentence_count_is_rejected():
    with pytest.raises(ValueError, match="fact_sentences must be 1, 2, or 3"):
        generate_corpus(CorpusConfig(n_relevant_docs=4, fact_sentences=4))


# -- Pilot subsampling ---------------------------------------------------


def test_subsample_takes_a_prefix_and_leaves_the_corpus_alone():
    docs, questions = generate_corpus(CorpusConfig(n_relevant_docs=32, seed=31))
    pilot = subsample_questions(questions, 10)
    assert len(pilot) == 10
    assert pilot == questions[:10]
    # The corpus is untouched: a pilot must pose the same retrieval problem.
    assert len(docs) == 32


def test_subsample_is_near_balanced_across_attribute_types():
    """A pilot that only asked one kind of question would not be a useful pilot."""
    _, questions = generate_corpus(CorpusConfig(n_relevant_docs=32, seed=37))
    attributes = {q.gold_fact_id.split(":")[1] for q in subsample_questions(questions, 8)}
    assert len(attributes) == 8


def test_subsample_with_no_limit_is_a_passthrough():
    _, questions = generate_corpus(CorpusConfig(n_relevant_docs=8, seed=41))
    assert subsample_questions(questions, None) is questions


def test_subsample_rejects_a_nonsense_limit():
    _, questions = generate_corpus(CorpusConfig(n_relevant_docs=8, seed=43))
    with pytest.raises(ValueError, match="at least 1"):
        subsample_questions(questions, 0)


# -- answer uniqueness ----------------------------------------------------
#
# The distractor experiment asks whether a model reports a distractor's value
# under pressure. That measurement only works if the gold answer appears in
# exactly one document: otherwise a model reading the right number off the
# wrong document scores correct, and "graceful degradation" and "got lucky"
# become indistinguishable. Before the gold/filler split this failed for 100%
# of questions.


@pytest.mark.parametrize("ratio", [0.0, 1.0, 2.0, 4.0])
def test_answer_appears_in_exactly_one_document(ratio):
    docs, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=32, distractor_ratio=ratio, fact_sentences=3, seed=101)
    )
    for question in questions:
        carriers = [d.doc_id for d in docs if question.answer in d.text]
        assert carriers == [question.gold_doc_id], (
            f"{question.question_id}: answer {question.answer!r} appears in {carriers}"
        )


def test_gold_and_filler_value_pools_are_disjoint():
    for attr in _ATTRIBUTES:
        assert set(attr.gold_values).isdisjoint(attr.filler_values)
        assert attr.gold_values and attr.filler_values


def test_no_value_is_a_substring_of_another():
    """'20 milliseconds' inside '120 milliseconds' would fake a match."""
    _validate_value_pools()  # raises on violation
    for attr in _ATTRIBUTES:
        for value in attr.values:
            assert not any(value in other for other in attr.values if other != value)


def test_gold_values_are_dealt_without_replacement():
    _, questions = generate_corpus(
        CorpusConfig(n_relevant_docs=40, fact_sentences=3, seed=103)
    )
    by_attribute = {}
    for q in questions:
        by_attribute.setdefault(q.gold_fact_id.split(":")[1], []).append(q.answer)
    for attribute, answers in by_attribute.items():
        assert len(set(answers)) == len(answers), f"{attribute} reused a gold value"


def test_exhausting_gold_values_is_a_clear_error():
    """Better a loud failure than a silently confounded corpus."""
    smallest = min(len(a.gold_values) for a in _ATTRIBUTES)
    with pytest.raises(ValueError, match="Ran out of unique gold values"):
        generate_corpus(CorpusConfig(n_relevant_docs=smallest * len(_ATTRIBUTES) + 8))
