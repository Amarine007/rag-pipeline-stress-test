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
    """One queryable property: how to state it, and how to ask about it."""

    def __init__(self, key: str, values: list[str], statement: str, question: str):
        self.key = key
        self.values = values
        self.statement = statement  # formatted with {name} and {value}
        self.question = question  # formatted with {name}

    def sentence(self, name: str, value: str) -> str:
        return self.statement.format(name=name, value=value)

    def ask(self, name: str) -> str:
        return self.question.format(name=name)


_ATTRIBUTES: list[_Attribute] = [
    _Attribute(
        "rebuild_cadence",
        [f"every {n} hours" for n in (12, 18, 24, 36, 48, 72)],
        "{name} rebuilds its primary shard {value}.",
        "How often does {name} rebuild its primary shard?",
    ),
    _Attribute(
        "latency_budget",
        [f"{n} milliseconds" for n in (15, 25, 45, 60, 90, 120)],
        "Query latency on {name} is budgeted at {value} for the 99th percentile.",
        "What is the 99th-percentile query latency budget for {name}?",
    ),
    _Attribute(
        "vector_dim",
        [f"{n} dimensions" for n in (128, 256, 384, 768, 1024, 1536)],
        "{name} stores its vectors in {value}.",
        "How many dimensions does {name} use to store its vectors?",
    ),
    _Attribute(
        "centroids",
        [f"{n} centroids" for n in (512, 1024, 2048, 4096, 8192)],
        "Coarse quantization on {name} uses an inverted file index with {value}.",
        "How many centroids does {name} use for coarse quantization?",
    ),
    _Attribute(
        "retention",
        [f"{n} days" for n in (7, 30, 90, 180, 365)],
        "Deleted records on {name} are retained for {value} before purging.",
        "How long does {name} retain deleted records before purging them?",
    ),
    _Attribute(
        "storage",
        [f"{n} terabytes" for n in (3, 8, 14, 22, 40, 64)],
        "A full rebuild of {name} consumes roughly {value} of scratch disk.",
        "How much scratch disk does a full rebuild of {name} consume?",
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
    ),
    _Attribute(
        "maintainer",
        [],  # filled per-document from _TEAMS x _CITIES
        "{name} is maintained by {value}.",
        "Who maintains {name}?",
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
) -> tuple[Document, str | None]:
    """Assemble one document; return it plus the answer value of its planted fact.

    Exactly one attribute is "queried" (the planted fact). The rest become filler
    describing the same system. Because each system appears in exactly one
    document and only its queried attribute is ever asked about, there is
    precisely one correct span per question anywhere in the corpus.
    """
    keys = [a.key for a in _ATTRIBUTES]
    filler_keys = [k for k in keys if k != queried_key]
    rng.shuffle(filler_keys)
    filler_keys = filler_keys[:n_filler]

    # Place the planted fact somewhere in the middle of the document rather than
    # always first, so position within the document is not a hidden confound.
    sentences: list[tuple[str, str | None]] = [
        (_ATTRS_BY_KEY[k].sentence(name, _value_for(_ATTRS_BY_KEY[k], rng)), None)
        for k in filler_keys
    ]

    answer_value: str | None = None
    if queried_key is not None:
        attr = _ATTRS_BY_KEY[queried_key]
        answer_value = _value_for(attr, rng)
        insert_at = rng.randrange(len(sentences) + 1) if sentences else 0
        sentences.insert(insert_at, (attr.sentence(name, answer_value), queried_key))

    # Assemble text while recording the exact offsets of the planted sentence.
    parts: list[str] = []
    spans: list[FactSpan] = []
    cursor = 0
    for sentence, key in sentences:
        if parts:
            parts.append(" ")
            cursor += 1
        if key is not None:
            spans.append(
                FactSpan(
                    fact_id=f"{doc_id}:{key}",
                    start=cursor,
                    end=cursor + len(sentence),
                    text=sentence,
                )
            )
        parts.append(sentence)
        cursor += len(sentence)

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
            doc_id, name, queried_key, config.filler_sentences_per_doc, rng, is_distractor=False
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
            doc_id, name, None, config.filler_sentences_per_doc + 1, rng, is_distractor=True
        )
        documents.append(doc)

    return documents, questions
