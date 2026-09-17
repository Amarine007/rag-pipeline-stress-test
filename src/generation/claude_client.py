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
"""

from __future__ import annotations

import hashlib
import json
import os
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

    def __init__(self, cache_dir: Path | None = None, use_cache: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self._client = None
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

        key = self._cache_key(request)
        if self.use_cache:
            hit = self._read_cache(key)
            if hit is not None:
                return LLMResponse(
                    text=hit["text"],
                    model=hit.get("model", model),
                    cached=True,
                    input_tokens=hit.get("input_tokens", 0),
                    output_tokens=hit.get("output_tokens", 0),
                    stop_reason=hit.get("stop_reason", ""),
                )

        response = self.client.messages.create(**request)

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        record = {
            "text": text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason or "",
        }

        # A refusal is a legitimate outcome to record and report, not to cache as
        # if it were an answer -- caching it would freeze a transient safety
        # decision into every future replay of the experiment.
        if self.use_cache and response.stop_reason != "refusal":
            self._write_cache(key, record)

        return LLMResponse(cached=False, **record)

    def cache_stats(self) -> dict:
        """Count and total size of cached responses, for run logs."""
        if not self.cache_dir.exists():
            return {"entries": 0, "bytes": 0}
        files = list(self.cache_dir.glob("*.json"))
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}
