"""Config-loading and Claude-client tests.

The temperature-gating tests are the important ones: sending `temperature` to a
current model is a 400, and a regression there would break every experiment at
runtime rather than at import.
"""

from __future__ import annotations

import json

import pytest

from src.config import ExperimentConfig, apply_override, load_config
from src.generation.claude_client import ClaudeClient, model_accepts_sampling

# -- config ---------------------------------------------------------------


def write_yaml(tmp_path, body: str):
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_reads_sections(tmp_path):
    path = write_yaml(
        tmp_path,
        """
name: chunk_sweep
seed: 99
chunking:
  strategy: recursive
  chunk_size: 256
retrieval:
  k: 3
sweep:
  chunking.chunk_size: [128, 256, 512]
""",
    )
    config = load_config(path)
    assert config.name == "chunk_sweep"
    assert config.seed == 99
    assert config.chunking.strategy == "recursive"
    assert config.chunking.chunk_size == 256
    assert config.retrieval.k == 3
    assert config.sweep == {"chunking.chunk_size": [128, 256, 512]}


def test_unknown_key_is_rejected(tmp_path):
    """A typo must fail loudly, not silently run with a default."""
    path = write_yaml(tmp_path, "chunking:\n  chunk_sizes: [100]\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(path)


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = write_yaml(tmp_path, "chunk_size: 400\n")
    with pytest.raises(ValueError, match="Unknown top-level key"):
        load_config(path)


def test_multi_variable_sweep_is_rejected(tmp_path):
    """'One variable at a time' is enforced, not merely documented."""
    path = write_yaml(
        tmp_path,
        "sweep:\n  chunking.chunk_size: [100, 200]\n  retrieval.k: [1, 5]\n",
    )
    with pytest.raises(ValueError, match="one variable at a time"):
        load_config(path)


def test_apply_override_is_non_mutating():
    base = ExperimentConfig()
    modified = apply_override(base, "chunking.chunk_size", 999)
    assert modified.chunking.chunk_size == 999
    assert base.chunking.chunk_size != 999, "override leaked into the source config"


def test_apply_override_rejects_unknown_field():
    with pytest.raises(ValueError, match="has no field"):
        apply_override(ExperimentConfig(), "chunking.nonsense", 1)


def test_apply_override_rejects_unknown_section():
    with pytest.raises(ValueError, match="Unknown config section"):
        apply_override(ExperimentConfig(), "nonsense.field", 1)


# -- temperature gating ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5-1"],
)
def test_current_models_reject_sampling(model):
    assert not model_accepts_sampling(model)


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5"])
def test_older_models_accept_sampling(model):
    assert model_accepts_sampling(model)


def test_temperature_on_current_model_raises_before_the_api_call(tmp_path):
    """Fail fast locally rather than spending a request to earn a 400."""
    client = ClaudeClient(cache_dir=tmp_path, use_cache=True)
    with pytest.raises(ValueError, match="does not accept `temperature`"):
        client.complete(prompt="hi", system="sys", model="claude-opus-5", temperature=0.0)


# -- response cache -------------------------------------------------------


def test_cache_key_is_order_independent():
    a = ClaudeClient._cache_key({"model": "m", "max_tokens": 5})
    b = ClaudeClient._cache_key({"max_tokens": 5, "model": "m"})
    assert a == b


def test_cache_key_changes_with_any_parameter():
    base = {"model": "claude-opus-5", "system": "s", "max_tokens": 10}
    key = ClaudeClient._cache_key(base)
    assert key != ClaudeClient._cache_key({**base, "max_tokens": 11})
    assert key != ClaudeClient._cache_key({**base, "system": "s2"})
    assert key != ClaudeClient._cache_key({**base, "model": "claude-sonnet-5"})


def test_cache_hit_returns_without_a_client(tmp_path):
    """A cached call must not touch the network -- there is no API key here."""
    client = ClaudeClient(cache_dir=tmp_path, use_cache=True)
    request = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "system": "sys",
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "prompt"}],
    }
    key = ClaudeClient._cache_key(request)
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"text": "cached answer", "model": "claude-opus-5", "stop_reason": "end_turn"}),
        encoding="utf-8",
    )

    response = client.complete(prompt="prompt", system="sys")
    assert response.cached
    assert response.text == "cached answer"


def test_corrupt_cache_entry_is_ignored_not_fatal(tmp_path):
    client = ClaudeClient(cache_dir=tmp_path, use_cache=True)
    key = "deadbeef"
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert client._read_cache(key) is None


def test_cache_stats_counts_entries(tmp_path):
    client = ClaudeClient(cache_dir=tmp_path, use_cache=True)
    client._write_cache("a", {"text": "x"})
    client._write_cache("b", {"text": "y"})
    assert client.cache_stats()["entries"] == 2
