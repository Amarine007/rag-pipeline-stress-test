"""Seeded synthetic corpus generation.

Why synthetic rather than a public QA dataset:

1. **Exact ground truth.** We know the character span of every answer, so
   recall@k and MRR are computed against a precise label rather than a fuzzy
   string match.
2. **Controllable distractors.** Distractor documents are generated from the
   same templates as relevant ones, differing only in entity and values. They
   are topically and lexically near-identical but carry no correct answer --
   which is exactly the failure mode the distractor experiment probes. Sampling
   "irrelevant" documents from an unrelated corpus would make the test trivially
   easy and prove nothing.
3. **Fictional entities.** No model can answer from pretraining, so a correct
   answer is evidence that retrieval supplied the fact.
4. **Tunable fact shape.** `CorpusConfig.fact_sentences` controls whether a
   planted fact is one self-contained sentence or a passage whose system name,
   value, and elaboration sit in different sentences. The single-sentence form
   makes chunk boundaries unrealistically decisive; the multi-sentence form is
   the ordinary case and is what the experiments run on.

Everything derives from `CorpusConfig.seed`; the same seed yields a
byte-identical corpus.
"""

from __future__ import annotations

import random

from src.config import CorpusConfig
from src.ingestion.documents import Document, EvalQuestion, FactSpan

# Fictional system names, built as prefix x suffix so the pool is large and the
# names are deliberately confusable -- "Halcyon Index" vs "Halcyon Vault" is a
# realistic retrieval hazard.
_NAME_PREFIXES = [
    "Halcyon", "Ashford", "Meridian", "Borealis", "Calyx", "Tessera", "Lodestar",
    "Vantage", "Quillon", "Sable", "Thornwood", "Ironvale", "Pellucid", "Marlow",
    "Cinderbrook", "Alderon", "Wraythe", "Solstice", "Kestrel", "Fenmark",
    "Orrery", "Blackthorn", "Glimmer", "Hollowmere", "Verdant", "Nightjar",
    "Cavalier", "Tidewater", "Umbral", "Larkspur", "Ravenspur", "Silvermoor",
]
_NAME_SUFFIXES = ["Index", "Vault", "Ledger", "Array", "Registry", "Cascade", "Store", "Atlas"]

_TEAMS = [
    "Meridian Data Group", "Northfield Platform", "Blue Harbor Infrastructure",
    "Camden Reliability", "Sutro Storage", "Ravenna Search", "Lindholm Systems",
    "Pike Street Data",
]
_CITIES = [
    "Trondheim", "Rotterdam", "Kyoto", "Valparaiso", "Gdansk", "Wellington",
    "Reykjavik", "Montevideo",
]


class _Attribute:
    """One queryable property: how to state it, and how to ask about it.

    Each attribute can be stated two ways. `statement` is the compact
    single-sentence form, which doubles as filler. The multi-sentence form
    deliberately splits answerability across sentences:

        setup        names the system and introduces the property, no value
        value        states the value, referring back anaphorically -- the
                     system name never appears here
        elaboration  a consequence, also anaphoric, no value

    Only the value sentence carries the answer, and only the setup sentence
    binds it to a system, so neither is sufficient alone. That is what makes a
    chunk boundary falling between them a real failure rather than an artifact
    of every fact being one tidy sentence.
    """

    def __init__(
        self,
        key: str,
        values: list[str],
        statement: str,
        question: str,
        setup: str,
        value_statement: str,
        elaboration: str,
    ):
        self.key = key
        self.values = values
        self.statement = statement  # formatted with {name} and {value}
        self.question = question  # formatted with {name}
        self.setup = setup  # formatted with {name}
        self.value_statement = value_statement  # formatted with {value}
        self.elaboration = elaboration  # no placeholders

    def sentence(self, name: str, value: str) -> str:
        return self.statement.format(name=name, value=value)

    def sentences(self, name: str, value: str, n_sentences: int) -> tuple[list[str], int]:
        """Render the fact as `n_sentences` sentences.

        Returns the sentences plus the index of the one stating the value, so
        the caller can record its character range separately.
        """
        if n_sentences == 1:
            return [self.sentence(name, value)], 0
        if n_sentences == 2:
            return [self.setup.format(name=name), self.value_statement.format(value=value)], 1
        if n_sentences == 3:
            return (
                [
                    self.setup.format(name=name),
                    self.value_statement.format(value=value),
                    self.elaboration,
                ],
                1,
            )
        raise ValueError(f"fact_sentences must be 1, 2, or 3; got {n_sentences}.")

    def ask(self, name: str) -> str:
        return self.question.format(name=name)


_ATTRIBUTES: list[_Attribute] = [
    _Attribute(
        "rebuild_cadence",
        [f"every {n} hours" for n in (12, 18, 24, 36, 48, 72)],
        "{name} rebuilds its primary shard {value}.",
        "How often does {name} rebuild its primary shard?",
        setup="{name} maintains a primary shard that is rebuilt on a fixed schedule.",
        value_statement="That rebuild runs {value}.",
        elaboration="Operators are paged automatically if two consecutive rebuilds are missed.",
    ),
    _Attribute(
        "latency_budget",
        [f"{n} milliseconds" for n in (15, 25, 45, 60, 90, 120)],
        "Query latency on {name} is budgeted at {value} for the 99th percentile.",
        "What is the 99th-percentile query latency budget for {name}?",
        setup="{name} publishes a latency budget covering its 99th-percentile query path.",
        value_statement="That budget is set at {value}.",
        elaboration="Sustained breaches of it open an automatic reliability review.",
    ),
    _Attribute(
        "vector_dim",
        [f"{n} dimensions" for n in (128, 256, 384, 768, 1024, 1536)],
        "{name} stores its vectors in {value}.",
        "How many dimensions does {name} use to store its vectors?",
        setup="{name} keeps one dense vector for every indexed record.",
        value_statement="Each of those vectors is stored in {value}.",
        elaboration="The width was fixed at the last index migration and has not changed since.",
    ),
    _Attribute(
        "centroids",
        [f"{n} centroids" for n in (512, 1024, 2048, 4096, 8192)],
        "Coarse quantization on {name} uses an inverted file index with {value}.",
        "How many centroids does {name} use for coarse quantization?",
        setup="Coarse quantization on {name} runs through an inverted file index.",
        value_statement="That index is partitioned across {value}.",
        elaboration="Probing more partitions per query trades latency for recall.",
    ),
    _Attribute(
        "retention",
        [f"{n} days" for n in (7, 30, 90, 180, 365)],
        "Deleted records on {name} are retained for {value} before purging.",
        "How long does {name} retain deleted records before purging them?",
        setup="{name} does not purge deleted records immediately.",
        value_statement="They are retained for {value} before the purge runs.",
        elaboration="The delay exists so that an accidental deletion can still be reversed.",
    ),
    _Attribute(
        "storage",
        [f"{n} terabytes" for n in (3, 8, 14, 22, 40, 64)],
        "A full rebuild of {name} consumes roughly {value} of scratch disk.",
        "How much scratch disk does a full rebuild of {name} consume?",
        setup="A full rebuild of {name} stages its intermediate output on scratch disk.",
        value_statement="That staging consumes roughly {value}.",
        elaboration="The scratch volume is provisioned with headroom above this figure.",
    ),
    _Attribute(
        "auth",
        [
            "hardware token authentication",
            "mutual TLS certificates",
            "single sign-on through the corporate identity provider",
            "short-lived signed URLs",
        ],
        "Access to the {name} administrative console requires {value}.",
        "What does access to the {name} administrative console require?",
        setup="The {name} administrative console sits behind its own access control.",
        value_statement="Reaching it requires {value}.",
        elaboration="Read-only dashboards are exempt from that requirement.",
    ),
    _Attribute(
        "maintainer",
        [],  # filled per-document from _TEAMS x _CITIES
        "{name} is maintained by {value}.",
        "Who maintains {name}?",
        setup="{name} has a single owning team rather than shared stewardship.",
        value_statement="It is maintained by {value}.",
        elaboration="Escalations outside working hours route to that team's on-call rotation.",
    ),
]

_ATTRS_BY_KEY = {a.key: a for a in _ATTRIBUTES}


def _all_names() -> list[str]:
    return [f"{p} {s}" for p in _NAME_PREFIXES for s in _NAME_SUFFIXES]


def _value_for(attr: _Attribute, rng: random.Random) -> str:
    if attr.key == "maintainer":
        return f"the {rng.choice(_TEAMS)} in {rng.choice(_CITIES)}"
    return rng.choice(attr.values)


def _build_document(
    doc_id: str,
    name: str,
    queried_key: str | None,
    n_filler: int,
    rng: random.Random,
    is_distractor: bool,
    fact_sentences: int = 1,
) -> tuple[Document, str | None]:
    """Assemble one document; return it plus the answer value of its planted fact.

    Exactly one attribute is "queried" (the planted fact). The rest become filler
    describing the same system. Because each system appears in exactly one
    document and only its queried attribute is ever asked about, there is
    precisely one correct span per question anywhere in the corpus.

    Distractors carry no queried attribute, but they do get a decoy rendered in
    the same multi-sentence shape, so they stay the same length and read the same
    way as a relevant document. Only the absence of a `FactSpan` distinguishes
    them, and nothing in the corpus text gives that away.
    """
    keys = [a.key for a in _ATTRIBUTES]
    filler_keys = [k for k in keys if k != queried_key]
    rng.shuffle(filler_keys)

    # The decoy consumes one filler slot, so relevant and distractor documents
    # end up with the same sentence budget.
    decoy_key = filler_keys.pop(0) if queried_key is None else None
    filler_keys = filler_keys[:n_filler]

    # A block is one contiguous run of sentences. Filler blocks are a single
    # sentence; a fact block may be several, and records where its value sits.
    Block = tuple[str, str | None, tuple[int, int] | None]
    blocks: list[Block] = [
        (_ATTRS_BY_KEY[k].sentence(name, _value_for(_ATTRS_BY_KEY[k], rng)), None, None)
        for k in filler_keys
    ]

    def build_fact_block(key: str, record_as: str | None) -> tuple[Block, str]:
        attr = _ATTRS_BY_KEY[key]
        value = _value_for(attr, rng)
        sentences, value_index = attr.sentences(name, value, fact_sentences)
        text = " ".join(sentences)
        offset = sum(len(s) + 1 for s in sentences[:value_index])
        required = (offset, offset + len(sentences[value_index]))
        return (text, record_as, required), value

    answer_value: str | None = None
    fact_block: Block | None = None
    if queried_key is not None:
        fact_block, answer_value = build_fact_block(queried_key, queried_key)
    elif decoy_key is not None:
        fact_block, _ = build_fact_block(decoy_key, None)

    # Place the fact somewhere in the middle of the document rather than always
    # first, so position within the document is not a hidden confound.
    if fact_block is not None:
        insert_at = rng.randrange(len(blocks) + 1) if blocks else 0
        blocks.insert(insert_at, fact_block)

    # Assemble text while recording the exact offsets of the planted fact.
    parts: list[str] = []
    spans: list[FactSpan] = []
    cursor = 0
    for text, key, required in blocks:
        if parts:
            parts.append(" ")
            cursor += 1
        if key is not None:
            required_start, required_end = required
            spans.append(
                FactSpan(
                    fact_id=f"{doc_id}:{key}",
                    start=cursor,
                    end=cursor + len(text),
                    text=text,
                    # Single-sentence facts leave this unset: the whole span is
                    # the value, so the extra criterion would be a no-op anyway.
                    required_start=cursor + required_start if fact_sentences > 1 else None,
                    required_end=cursor + required_end if fact_sentences > 1 else None,
                    required_text=text[required_start:required_end] if fact_sentences > 1 else "",
                )
            )
        parts.append(text)
        cursor += len(text)

    doc = Document(
        doc_id=doc_id,
        text="".join(parts),
        topic=name,
        is_distractor=is_distractor,
        fact_spans=tuple(spans),
        metadata={"system_name": name, "queried_attribute": queried_key},
    )
    doc.validate_spans()
    return doc, answer_value


def generate_corpus(config: CorpusConfig) -> tuple[list[Document], list[EvalQuestion]]:
    """Generate a corpus and its matching eval question set.

    Returns (documents, questions). Questions are generated only for
    non-distractor documents; distractors carry no planted fact and can never be
    a correct retrieval.
    """
    if config.fact_sentences not in (1, 2, 3):
        raise ValueError(
            f"corpus.fact_sentences must be 1, 2, or 3; got {config.fact_sentences}."
        )

    rng = random.Random(config.seed)

    names = _all_names()
    rng.shuffle(names)

    n_distractors = int(round(config.n_relevant_docs * config.distractor_ratio))
    needed = config.n_relevant_docs + n_distractors
    if needed > len(names):
        raise ValueError(
            f"Need {needed} unique system names but the pool holds {len(names)}. "
            "Reduce n_relevant_docs or distractor_ratio, or extend the name pools."
        )

    documents: list[Document] = []
    questions: list[EvalQuestion] = []

    # Cycle attributes so the eval set is balanced across question types rather
    # than accidentally over-weighting one.
    for i in range(config.n_relevant_docs):
        name = names[i]
        queried_key = _ATTRIBUTES[i % len(_ATTRIBUTES)].key
        doc_id = f"doc-{i:04d}"
        doc, answer = _build_document(
            doc_id,
            name,
            queried_key,
            config.filler_sentences_per_doc,
            rng,
            is_distractor=False,
            fact_sentences=config.fact_sentences,
        )
        documents.append(doc)
        questions.append(
            EvalQuestion(
                question_id=f"q-{i:04d}",
                question=_ATTRS_BY_KEY[queried_key].ask(name),
                answer=answer or "",
                gold_doc_id=doc_id,
                gold_fact_id=f"{doc_id}:{queried_key}",
                topic=name,
            )
        )

    for j in range(n_distractors):
        name = names[config.n_relevant_docs + j]
        doc_id = f"distractor-{j:04d}"
        # Distractors get a full complement of sentences drawn from the same
        # templates, so they are statistically indistinguishable from relevant
        # documents until you check whether they answer the question.
        doc, _ = _build_document(
            doc_id,
            name,
            None,
            config.filler_sentences_per_doc,
            rng,
            is_distractor=True,
            fact_sentences=config.fact_sentences,
        )
        documents.append(doc)

    return documents, questions


def subsample_questions(
    questions: list[EvalQuestion], max_questions: int | None
) -> list[EvalQuestion]:
    """Take the first `max_questions` questions, for reduced pilot runs.

    The corpus is deliberately left untouched -- only the eval set shrinks -- so
    a pilot exercises the same retrieval problem as the full run at a fraction of
    the API cost.

    First-N rather than a random sample: `generate_corpus` cycles through the
    attribute list when assigning queried attributes, so the first N questions
    are already near-balanced across question types (exactly balanced when N is a
    multiple of the attribute count). A random subset would be no more
    representative and would add a second seed to reason about.

    A pilot's numbers are still a pilot's numbers. `n` travels with every metric
    block so a reduced run can never be mistaken for a full one.
    """
    if max_questions is None:
        return questions
    if max_questions < 1:
        raise ValueError(f"max_questions must be at least 1; got {max_questions}.")
    return questions[:max_questions]
