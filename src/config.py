"""Experiment configuration.

Every knob the experiments turn -- chunk size, k, distractor ratio, context
length, model IDs, seeds -- is declared here and loaded from YAML in `configs/`.
Nothing in `src/` is allowed to hardcode one of these values; an experiment that
cannot be re-run from its config file is not reproducible, which defeats the
purpose of the project.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CorpusConfig:
    """Controls synthetic corpus generation."""

    n_relevant_docs: int = 40
    """Documents that contain a planted, question-answering fact."""

    distractor_ratio: float = 0.0
    """Distractor documents per relevant document. 2.0 means 80 distractors for 40 relevant."""

    filler_sentences_per_doc: int = 6
    """Sentences of on-topic filler surrounding the planted fact."""

    fact_sentences: int = 1
    """How many consecutive sentences the planted fact spans (1, 2, or 3).

    At 1 the fact is a single self-contained sentence, which makes the
    chunk-boundary mechanism unrealistically crisp: a chunk either has the whole
    answer or none of it. At 2 or 3 the system name, the value, and its
    elaboration sit in different sentences, so a chunk boundary can sever the
    value from the entity it belongs to -- the ordinary case in real corpora.

    Defaults to 1 so that constructing a `CorpusConfig()` reproduces the original
    corpus; every config in `configs/` sets this explicitly.
    """

    seed: int = 20260917


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "fixed"
    """One of: 'fixed', 'recursive'."""

    chunk_size: int = 400
    """Target chunk size in characters."""

    chunk_overlap: int = 50
    """Characters of overlap between consecutive chunks."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
    """L2-normalize vectors so inner product is cosine similarity."""


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 5
    index_type: str = "flat"
    """'flat' is exact brute-force search. Approximate indexes would add a
    confounding variable to experiments that are about chunking and distractors,
    so exact search is the default."""


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "claude-opus-5"
    max_tokens: int = 1024
    effort: str = "low"
    """Answering from retrieved passages is not a reasoning task, and the
    experiment grid makes hundreds of these calls."""

    temperature: float | None = None
    """Only sent for models that still accept sampling parameters. On
    claude-opus-5 and other current models `temperature` was removed and sending
    it returns a 400 -- see `src/generation/claude_client.py`."""

    use_cache: bool = True
    """Content-addressed disk cache. This, not temperature, is what makes runs
    reproducible."""


@dataclass(frozen=True)
class JudgeConfig:
    """The LLM-judge that scores answer correctness."""

    model: str = "claude-opus-5"
    max_tokens: int = 512
    effort: str = "low"
    use_cache: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete, re-runnable experiment specification."""

    name: str = "unnamed"
    description: str = ""
    seed: int = 20260917

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    sweep: dict[str, list[Any]] = field(default_factory=dict)
    """The one variable an experiment varies, e.g. {'chunking.chunk_size': [200, 400, 800]}.
    Enforcing a single key here is what keeps 'change one variable at a time' honest."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTION_TYPES = {
    "corpus": CorpusConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "retrieval": RetrievalConfig,
    "generation": GenerationConfig,
    "judge": JudgeConfig,
}


def _build_section(name: str, raw: dict[str, Any] | None) -> Any:
    cls = _SECTION_TYPES[name]
    raw = raw or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in config section '{name}'. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, rejecting unknown keys.

    Silently ignoring a typo'd key would mean an experiment quietly runs with a
    default instead of the intended value, so unknown keys are a hard error.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    sections = {name: _build_section(name, raw.pop(name, None)) for name in _SECTION_TYPES}

    top_level = {"name", "description", "seed", "sweep"}
    unknown = set(raw) - top_level
    if unknown:
        raise ValueError(
            f"Unknown top-level key(s) {sorted(unknown)} in {path}. "
            f"Valid keys: {sorted(top_level | set(_SECTION_TYPES))}"
        )

    sweep = raw.get("sweep", {}) or {}
    if len(sweep) > 1:
        raise ValueError(
            f"Config {path} sweeps {len(sweep)} variables ({sorted(sweep)}). "
            "Experiments vary exactly one variable at a time -- split this into "
            "separate configs."
        )

    return ExperimentConfig(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        seed=raw.get("seed", 20260917),
        sweep=sweep,
        **sections,
    )


def apply_override(config: ExperimentConfig, dotted_key: str, value: Any) -> ExperimentConfig:
    """Return a copy of `config` with one dotted key replaced, e.g. 'chunking.chunk_size'.

    Used to expand a sweep into its individual conditions. Returns a new object
    rather than mutating, so a sweep cannot leak state between conditions.
    """
    if "." not in dotted_key:
        return dataclasses.replace(config, **{dotted_key: value})

    section_name, attr = dotted_key.split(".", 1)
    if section_name not in _SECTION_TYPES:
        raise ValueError(f"Unknown config section '{section_name}' in override '{dotted_key}'")

    section = getattr(config, section_name)
    if not hasattr(section, attr):
        raise ValueError(f"Section '{section_name}' has no field '{attr}'")

    return dataclasses.replace(config, **{section_name: dataclasses.replace(section, **{attr: value})})
