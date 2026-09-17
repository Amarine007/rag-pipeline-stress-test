"""Answer-correctness scoring.

Two scorers, deliberately kept separate from the retrieval metrics in
`metrics.py`. A pipeline can retrieve perfectly and still answer wrong, or
retrieve badly and answer right by luck; collapsing the two into one number
hides the only interesting signal this project is trying to surface.

`exact_match` is a cheap deterministic check used in tests and as a sanity
baseline. `LLMJudge` is the reported scorer, because the generated answers are
full sentences and string matching would punish correct paraphrases.

Outcomes are three-way, not binary: **correct**, **incorrect**, and
**abstained**. An abstention (the model saying it cannot find the answer) is a
different and much less harmful failure than confidently reporting a distractor's
value, and the distractor experiment is largely about which of the two a model
does under pressure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import JudgeConfig
from src.generation.claude_client import ClaudeClient
from src.generation.prompts import ABSTAIN_TOKEN, JUDGE_SYSTEM, build_judge_prompt


@dataclass(frozen=True)
class AnswerScore:
    """One graded answer."""

    question_id: str
    correct: bool
    abstained: bool
    prediction: str
    reference: str
    judge_raw: str = ""

    @property
    def hallucinated(self) -> bool:
        """Wrong, and did not have the decency to say so."""
        return not self.correct and not self.abstained


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_abstention(prediction: str) -> bool:
    return ABSTAIN_TOKEN.lower() in prediction.lower()


def exact_match(question_id: str, prediction: str, reference: str) -> AnswerScore:
    """Substring containment after normalization. Deterministic, no API call."""
    abstained = is_abstention(prediction)
    correct = (not abstained) and _normalize(reference) in _normalize(prediction)
    return AnswerScore(
        question_id=question_id,
        correct=correct,
        abstained=abstained,
        prediction=prediction,
        reference=reference,
        judge_raw="exact_match",
    )


class LLMJudge:
    """Grades semantic equivalence with a Claude call, cached like any other."""

    def __init__(self, client: ClaudeClient, config: JudgeConfig | None = None):
        self.client = client
        self.config = config or JudgeConfig()

    def score(
        self, question_id: str, question: str, prediction: str, reference: str
    ) -> AnswerScore:
        # An abstention is graded without spending a judge call: the model has
        # already told us it found nothing, and there is nothing to compare.
        if is_abstention(prediction):
            return AnswerScore(
                question_id=question_id,
                correct=False,
                abstained=True,
                prediction=prediction,
                reference=reference,
                judge_raw="abstained",
            )

        response = self.client.complete(
            prompt=build_judge_prompt(question, reference, prediction),
            system=JUDGE_SYSTEM,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            effort=self.config.effort,
        )

        verdict = response.text.strip().upper()
        # Check INCORRECT first: "INCORRECT" contains "CORRECT" as a substring,
        # so testing for CORRECT first would grade every wrong answer as right.
        if verdict.startswith("INCORRECT") or "INCORRECT" in verdict:
            correct = False
        elif "CORRECT" in verdict:
            correct = True
        else:
            # An unparseable verdict is scored wrong rather than dropped, so a
            # judge failure can never quietly inflate the reported accuracy.
            correct = False

        return AnswerScore(
            question_id=question_id,
            correct=correct,
            abstained=False,
            prediction=prediction,
            reference=reference,
            judge_raw=response.text.strip(),
        )


def aggregate_answers(scores: list[AnswerScore]) -> dict[str, float]:
    """Mean answer metrics, carrying `n` so results are read against their sample size."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "abstention_rate": 0.0, "hallucination_rate": 0.0}
    return {
        "n": n,
        "accuracy": sum(s.correct for s in scores) / n,
        "abstention_rate": sum(s.abstained for s in scores) / n,
        "hallucination_rate": sum(s.hallucinated for s in scores) / n,
    }
