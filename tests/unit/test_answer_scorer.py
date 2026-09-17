"""Answer-scoring tests.

The INCORRECT-before-CORRECT parsing test guards a genuinely dangerous bug:
"INCORRECT" contains "CORRECT" as a substring, so a naive check would grade
every wrong answer as right and silently inflate every accuracy number reported
in the paper.
"""

from __future__ import annotations

import pytest

from src.evaluation.answer_scorer import (
    AnswerScore,
    LLMJudge,
    aggregate_answers,
    exact_match,
    is_abstention,
)
from src.generation.claude_client import LLMResponse
from src.generation.prompts import ABSTAIN_TOKEN


class FakeClient:
    """Stands in for ClaudeClient, returning a scripted verdict."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.calls = 0

    def complete(self, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.verdict, model="fake", cached=False, stop_reason="end_turn")


def judge_with(verdict: str) -> tuple[LLMJudge, FakeClient]:
    client = FakeClient(verdict)
    return LLMJudge(client), client


def test_incorrect_is_not_parsed_as_correct():
    judge, _ = judge_with("INCORRECT")
    assert not judge.score("q", "question?", "every 12 hours", "every 36 hours").correct


def test_correct_verdict_parses():
    judge, _ = judge_with("CORRECT")
    assert judge.score("q", "question?", "every 36 hours", "every 36 hours").correct


@pytest.mark.parametrize("verdict", ["correct", "  CORRECT  ", "CORRECT."])
def test_verdict_parsing_tolerates_formatting(verdict):
    judge, _ = judge_with(verdict)
    assert judge.score("q", "question?", "36 hours", "36 hours").correct


def test_unparseable_verdict_scores_incorrect():
    """A confused judge must never inflate accuracy."""
    judge, _ = judge_with("I'm not sure how to grade this.")
    assert not judge.score("q", "question?", "36 hours", "36 hours").correct


def test_abstention_skips_the_judge_call():
    judge, client = judge_with("CORRECT")
    score = judge.score("q", "question?", ABSTAIN_TOKEN, "every 36 hours")
    assert score.abstained
    assert not score.correct
    assert not score.hallucinated
    assert client.calls == 0, "spent a judge call on an answer that declined to answer"


def test_abstention_detection():
    assert is_abstention(ABSTAIN_TOKEN)
    assert is_abstention(f"  {ABSTAIN_TOKEN.lower()}  ")
    assert not is_abstention("The shard rebuilds every 36 hours.")


def test_exact_match_ignores_wording():
    score = exact_match("q", "It rebuilds every 36 hours.", "every 36 hours")
    assert score.correct


def test_exact_match_rejects_a_different_value():
    assert not exact_match("q", "It rebuilds every 12 hours.", "every 36 hours").correct


def test_hallucination_is_wrong_without_abstaining():
    wrong = AnswerScore("q", correct=False, abstained=False, prediction="12 hours", reference="36")
    declined = AnswerScore("q", correct=False, abstained=True, prediction="?", reference="36")
    assert wrong.hallucinated
    assert not declined.hallucinated


def test_aggregate_separates_the_three_outcomes():
    scores = [
        AnswerScore("a", correct=True, abstained=False, prediction="x", reference="x"),
        AnswerScore("b", correct=False, abstained=True, prediction="?", reference="y"),
        AnswerScore("c", correct=False, abstained=False, prediction="z", reference="w"),
        AnswerScore("d", correct=True, abstained=False, prediction="v", reference="v"),
    ]
    agg = aggregate_answers(scores)
    assert agg["n"] == 4
    assert agg["accuracy"] == pytest.approx(0.5)
    assert agg["abstention_rate"] == pytest.approx(0.25)
    assert agg["hallucination_rate"] == pytest.approx(0.25)


def test_aggregate_of_nothing_is_zero_not_a_crash():
    assert aggregate_answers([])["n"] == 0
