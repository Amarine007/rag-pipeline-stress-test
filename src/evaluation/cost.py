"""Cost accounting from measured token usage.

A pilot exists to answer two questions before the full grid is paid for: does
the judge parse, and what will this cost? The second one should be measured
rather than guessed, so every condition records the tokens it actually spent and
this module prices them.

Prices are per million tokens, as published for the first-party API. They are
data, not constants baked into logic, because they change and because the
project may run the grid on more than one model.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per million tokens, (input, output).
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# The Batches API bills at half the standard rate.
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class CostEstimate:
    """What a run cost, and what a larger one would."""

    input_tokens: int
    output_tokens: int
    api_calls: int
    cached_calls: int
    usd: float

    def scaled(self, factor: float) -> "CostEstimate":
        """This run's cost scaled by `factor` -- e.g. a pilot to a full grid.

        Linear in the number of questions, which holds here because every
        question is an independent call of roughly the same shape. It does not
        hold across conditions whose context size differs, so scale within an
        experiment rather than across the whole project.
        """
        return CostEstimate(
            input_tokens=round(self.input_tokens * factor),
            output_tokens=round(self.output_tokens * factor),
            api_calls=round(self.api_calls * factor),
            cached_calls=self.cached_calls,
            usd=self.usd * factor,
        )


def price_usage(usage: dict, model: str, batched: bool = False) -> CostEstimate:
    """Price one usage block. Unknown models raise rather than silently costing zero."""
    if model not in PRICES_PER_MTOK:
        raise ValueError(
            f"No price on file for {model!r}. Add it to PRICES_PER_MTOK -- "
            "defaulting to zero would report a free run."
        )
    input_price, output_price = PRICES_PER_MTOK[model]
    discount = BATCH_DISCOUNT if batched else 1.0

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    usd = discount * (
        input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000
    )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=usage.get("api_calls", 0),
        cached_calls=usage.get("cached_calls", 0),
        usd=usd,
    )


def total_usage(usages: list[dict]) -> dict:
    """Sum usage blocks across conditions."""
    keys = ("input_tokens", "output_tokens", "api_calls", "cached_calls")
    return {key: sum(u.get(key, 0) for u in usages) for key in keys}


def format_cost_report(
    usages: list[dict],
    model: str,
    batched: bool,
    questions_run: int,
    questions_full: int,
) -> str:
    """A short report pricing what ran and extrapolating to the full eval set."""
    usage = total_usage(usages)
    measured = price_usage(usage, model, batched)

    # Two independent counters, not a subset: `api_calls` counts requests that
    # reached the API and were paid for, `cached_calls` counts requests answered
    # from disk that never became API calls at all. Phrasing them as "N calls, M
    # of them cached" would imply the second is part of the first.
    lines = [
        f"  API calls:     {measured.api_calls} paid, "
        f"plus {measured.cached_calls} replayed from cache at no cost",
        f"  Tokens:        {measured.input_tokens:,} in / {measured.output_tokens:,} out",
        f"  Cost:          ${measured.usd:.2f}"
        + ("  (batched, 50% rate)" if batched else "  (sequential, full rate)"),
    ]

    if measured.api_calls == 0:
        lines.append("  Everything replayed from cache -- this run spent nothing.")
        return "\n".join(lines)

    if questions_run and questions_full > questions_run:
        factor = questions_full / questions_run
        projected = measured.scaled(factor)
        lines.append(
            f"  Projected:     ${projected.usd:.2f} for the full "
            f"{questions_full}-question eval set ({factor:.1f}x this run)"
        )
        if not batched:
            lines.append(
                f"                 ${projected.usd * BATCH_DISCOUNT:.2f} if run with --batch"
            )
        lines.append(
            "                 (upper bound: conditions that retrieve identical "
            "context share cache entries)"
        )
    return "\n".join(lines)
