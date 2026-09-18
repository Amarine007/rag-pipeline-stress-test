"""Direct Anthropic SDK wrapper with a content-addressed response cache.

Two things here are load-bearing for the project's reproducibility claim.

**Temperature is gated, not assumed.** Current Claude models removed the
sampling parameters: sending `temperature` to `claude-opus-5` returns a 400. So
the client sends it only for models that still accept it (Haiku 4.5 and older),
and silently omitting it elsewhere is correct behavior, not an oversight.

**The cache is what makes runs reproducible.** Every call is keyed by a SHA-256
over the exact request -- model, system prompt, user prompt, and every parameter
that affects output. A re-run of an experiment replays byte-identical responses
from disk and costs nothing; only genuinely new (prompt, config) pairs reach the
API. This is a weaker guarantee than temperature=0 would give across *cold*
runs, and the paper says so explicitly rather than implying determinism it does
not have.

**Batch transport is a cost decision, not a semantic one.** `complete_batch`
sends the same requests through the Message Batches API at half price, which
suits this project exactly: an experiment grid is a few hundred independent
calls and nothing about it is latency-sensitive. The cache key is computed from
the request alone, so a batched run and a sequential run share cache entries
and produce identical results -- the transport never enters the hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT

# Model families that removed `temperature`/`top_p`/`top_k`. Sending a sampling
# parameter to any of these returns a 400, so it must be withheld.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-mythos-5",
)

DEFAULT_CACHE_DIR = PROJECT_ROOT / ".llm_cache"


def model_accepts_sampling(model: str) -> bool:
    """True if `temperature` may be sent to this model without a 400."""
    return not model.startswith(_NO_SAMPLING_PREFIXES)


@dataclass(frozen=True)
class BatchRequest:
    """One call to send through the Batches API.

    `custom_id` identifies the result on the way back. Batch results arrive in
    arbitrary order, so it is the only thing tying a response to its question.
    """

    custom_id: str
    prompt: str
    system: str
    model: str = "claude-opus-5"
    max_tokens: int = 1024
    effort: str = "low"
    temperature: float | None = None


@dataclass
class LLMResponse:
    """A completion plus the bookkeeping the experiments need."""

    text: str
    model: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class ClaudeClient:
    """Calls the Messages API, memoizing responses on disk."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        use_batch: bool = False,
        poll_seconds: int = 20,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.use_batch = use_batch
        self.poll_seconds = poll_seconds
        self._client = None
        # Running token totals, so a pilot can price the full grid from measured
        # usage instead of a guess. Cached calls are counted separately because
        # they cost nothing -- conflating them would understate a cold run.
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "api_calls": 0,
            "cached_calls": 0,
        }
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            from dotenv import load_dotenv

            load_dotenv()
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                    "your key, or export it in the environment."
                )
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _cache_key(payload: dict) -> str:
        # sort_keys makes the serialization canonical, so logically identical
        # requests hash identically regardless of dict construction order.
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A truncated cache entry (interrupted write, full disk) should cost
            # one API call, not crash an eight-hour experiment sweep.
            return None

    def _write_cache(self, key: str, record: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)  # atomic, so a crash never leaves a half-written entry

    @staticmethod
    def _build_request(
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        effort: str,
        temperature: float | None,
    ) -> dict:
        """Assemble the API request. Shared by the single and batch paths so the
        two can never drift, and so their cache keys are identical."""
        request: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": prompt}],
        }

        if temperature is not None:
            if model_accepts_sampling(model):
                request["temperature"] = temperature
            else:
                raise ValueError(
                    f"Model {model!r} does not accept `temperature` -- the sampling "
                    "parameters were removed on this model family and sending one "
                    "returns a 400. Leave temperature unset and rely on the response "
                    "cache for reproducibility, or switch to claude-haiku-4-5."
                )
        return request

    def _count(self, record: dict | None) -> None:
        """Add one call to the running usage totals. `None` means a cache hit."""
        if record is None:
            self.usage["cached_calls"] += 1
            return
        self.usage["api_calls"] += 1
        self.usage["input_tokens"] += record.get("input_tokens", 0)
        self.usage["output_tokens"] += record.get("output_tokens", 0)

    def usage_snapshot(self) -> dict:
        return dict(self.usage)

    @staticmethod
    def usage_delta(before: dict, after: dict) -> dict:
        """Usage accrued between two snapshots -- one condition's share of a run."""
        return {key: after[key] - before[key] for key in after}

    @staticmethod
    def _record_from_message(message) -> dict:
        """Flatten an API message into the dict the cache and LLMResponse share."""
        return {
            "text": "".join(b.text for b in message.content if b.type == "text").strip(),
            "model": message.model,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "stop_reason": message.stop_reason or "",
        }

    def complete(
        self,
        prompt: str,
        system: str,
        model: str = "claude-opus-5",
        max_tokens: int = 1024,
        effort: str = "low",
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send one message and return the text response."""
        request = self._build_request(prompt, system, model, max_tokens, effort, temperature)
        key = self._cache_key(request)
        if self.use_cache:
            hit = self._read_cache(key)
            if hit is not None:
                self._count(None)
                return LLMResponse(
                    text=hit["text"],
                    model=hit.get("model", model),
                    cached=True,
                    input_tokens=hit.get("input_tokens", 0),
                    output_tokens=hit.get("output_tokens", 0),
                    stop_reason=hit.get("stop_reason", ""),
                )

        response = self.client.messages.create(**request)
        record = self._record_from_message(response)
        self._count(record)

        # A refusal is a legitimate outcome to record and report, not to cache as
        # if it were an answer -- caching it would freeze a transient safety
        # decision into every future replay of the experiment.
        if self.use_cache and record["stop_reason"] != "refusal":
            self._write_cache(key, record)

        return LLMResponse(cached=False, **record)

    def complete_batch(
        self,
        requests: list[BatchRequest],
        show_progress: bool = True,
    ) -> dict[str, LLMResponse]:
        """Send many requests through the Batches API at half price.

        Returns a dict keyed by `custom_id`. Cached requests never reach the
        API, so a re-run of an experiment submits nothing and returns instantly;
        only genuine misses are batched.

        Batches are not latency-optimized -- most finish within an hour, and the
        ceiling is 24 -- which is the right trade for an experiment grid and the
        wrong one for anything interactive.
        """
        results: dict[str, LLMResponse] = {}
        pending: list[tuple[str, dict, str]] = []  # (custom_id, request, cache_key)

        for item in requests:
            request = self._build_request(
                item.prompt, item.system, item.model, item.max_tokens,
                item.effort, item.temperature,
            )
            key = self._cache_key(request)
            hit = self._read_cache(key) if self.use_cache else None
            if hit is not None:
                self._count(None)
                results[item.custom_id] = LLMResponse(
                    text=hit["text"],
                    model=hit.get("model", item.model),
                    cached=True,
                    input_tokens=hit.get("input_tokens", 0),
                    output_tokens=hit.get("output_tokens", 0),
                    stop_reason=hit.get("stop_reason", ""),
                )
            else:
                pending.append((item.custom_id, request, key))

        if not pending:
            if show_progress and requests:
                print(f"    batch: all {len(requests)} responses served from cache")
            return results

        results.update(self._run_batch(pending, len(requests), show_progress))
        return results

    def _run_batch(
        self,
        pending: list[tuple[str, dict, str]],
        total: int,
        show_progress: bool,
    ) -> dict[str, LLMResponse]:
        """Submit the cache misses, wait for the batch, and collect its results."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        cache_keys = {custom_id: key for custom_id, _, key in pending}

        if show_progress:
            print(
                f"    batch: {len(pending)} new request(s), "
                f"{total - len(pending)} from cache -- submitting",
                flush=True,
            )

        batch = self.client.messages.batches.create(
            requests=[
                Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**request))
                for custom_id, request, _ in pending
            ]
        )

        while True:
            status = self.client.messages.batches.retrieve(batch.id)
            if status.processing_status == "ended":
                break
            if show_progress:
                counts = status.request_counts
                print(
                    f"    batch {batch.id}: {status.processing_status} "
                    f"(processing={counts.processing}, succeeded={counts.succeeded}, "
                    f"errored={counts.errored})",
                    flush=True,
                )
            time.sleep(self.poll_seconds)

        results: dict[str, LLMResponse] = {}
        failures: list[str] = []
        for entry in self.client.messages.batches.results(batch.id):
            if entry.result.type != "succeeded":
                # Recorded and raised rather than silently dropped: a missing
                # answer would shrink n for one condition only, which would
                # quietly bias a comparison across conditions.
                failures.append(f"{entry.custom_id}: {entry.result.type}")
                continue

            record = self._record_from_message(entry.result.message)
            self._count(record)
            if self.use_cache and record["stop_reason"] != "refusal":
                self._write_cache(cache_keys[entry.custom_id], record)
            results[entry.custom_id] = LLMResponse(cached=False, **record)

        missing = set(cache_keys) - set(results)
        if failures or missing:
            raise RuntimeError(
                f"Batch {batch.id} did not return every response. "
                f"Failed: {failures or 'none'}. Missing: {sorted(missing) or 'none'}. "
                "Re-run to retry -- successful responses are already cached, so a "
                "retry only re-sends what failed."
            )

        if show_progress:
            print(f"    batch {batch.id}: {len(results)} response(s) collected", flush=True)
        return results

    def cache_stats(self) -> dict:
        """Count and total size of cached responses, for run logs."""
        if not self.cache_dir.exists():
            return {"entries": 0, "bytes": 0}
        files = list(self.cache_dir.glob("*.json"))
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}
