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
#
# The pool is 56 x 20 = 1120 names. Size is a real constraint rather than
# decoration: the corpus needs n_relevant_docs * (1 + distractor_ratio) unique
# names, so the distractor experiment at ratio 4.0 needs five names per
# question. At 1120 the eval set can reach 224 questions at the widest ratio,
# which is what keeps the sample size and the distractor sweep from trading off
# against each other.
_NAME_PREFIXES = [
    "Halcyon", "Ashford", "Meridian", "Borealis", "Calyx", "Tessera", "Lodestar",
    "Vantage", "Quillon", "Sable", "Thornwood", "Ironvale", "Pellucid", "Marlow",
    "Cinderbrook", "Alderon", "Wraythe", "Solstice", "Kestrel", "Fenmark",
    "Orrery", "Blackthorn", "Glimmer", "Hollowmere", "Verdant", "Nightjar",
    "Cavalier", "Tidewater", "Umbral", "Larkspur", "Ravenspur", "Silvermoor",
    "Windermere", "Carrowmore", "Ellesmere", "Thistledown", "Brackwater",
    "Ferncliff", "Aldergrove", "Stonehaven", "Mirefield", "Coldharbour",
    "Westmarch", "Gravenhurst", "Linderoth", "Ashgrove", "Duskwood", "Highmoor",
    "Fairwater", "Oakhollow", "Brightwater", "Northwick", "Sedgemoor",
    "Wrenfield", "Amberlyn", "Castellan",
]
_NAME_SUFFIXES = [
    "Index", "Vault", "Ledger", "Array", "Registry", "Cascade", "Store", "Atlas",
    "Repository", "Catalog", "Archive", "Lattice", "Manifold", "Corpus", "Table",
    "Graph", "Shard", "Depot", "Directory", "Chronicle",
]

_TEAMS = [
    "Meridian Data Group", "Northfield Platform", "Blue Harbor Infrastructure",
    "Camden Reliability", "Sutro Storage", "Ravenna Search", "Lindholm Systems",
    "Pike Street Data",
]
_CITIES = [
    "Trondheim", "Rotterdam", "Kyoto", "Valparaiso", "Gdansk", "Wellington",
    "Reykjavik", "Montevideo",
]


# Value pools. Two invariants hold across every attribute, and both are checked
# at import by `_validate_value_pools`:
#
# 1. **No value is a substring of another within an attribute.** Numeric values
#    therefore all carry the same digit count -- "20 milliseconds" inside
#    "120 milliseconds" would make an answer look present in a document that
#    never stated it.
# 2. **Pools are large enough to assign gold values without replacement.** Half
#    of each pool is reserved for planted answers, and 40 values means 20 gold
#    values per attribute, which supports up to 160 relevant documents.
_HOURS = tuple(range(12, 92, 2))
_MILLISECONDS = tuple(range(11, 91, 2))
_DAYS = tuple(range(10, 90, 2))
_TERABYTES = tuple(range(13, 93, 2))
_DIMENSIONS = tuple(range(1024, 1024 + 40 * 76, 76))
_CENTROIDS = tuple(range(1031, 1031 + 40 * 223, 223))

_AUTH_METHODS = [
    "hardware token authentication", "mutual TLS certificates",
    "single sign-on through the corporate identity provider",
    "short-lived signed URLs", "an approved bastion host",
    "a break-glass ticket with two approvers", "a hardware-backed passkey",
    "an attested device certificate",
]
_AUTH_SCOPES = [
    "for every session", "for all write operations",
    "when connecting from outside the corporate network",
    "after ninety days of inactivity",
    "for anyone outside the platform team",
]
_AUTH_VALUES = [f"{m} {s}" for m in _AUTH_METHODS for s in _AUTH_SCOPES]
_MAINTAINER_VALUES = [f"the {t} in {c}" for t in _TEAMS for c in _CITIES]


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

    Values are partitioned into two disjoint halves: one that only ever appears
    as a planted answer, one that only ever appears as filler. Without that
    split the pools are small enough (and every document mentions nearly every
    attribute) that a question's answer string reliably turns up in some other
    document -- measured at 100% of questions, and present in the retrieved
    context for 58% of them at the widest distractor ratio. A model reading the
    right number off the wrong document would then score correct, which would
    quietly destroy the distractor experiment's ability to tell a real retrieval
    from a lucky one.

    The split alternates rather than cutting the list in half, so gold and
    filler values interleave and the two sets have no systematic difference in
    magnitude for the retriever to key on.
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
        self.filler_values = values[0::2]
        self.gold_values = values[1::2]
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
        [f"every {n} hours" for n in _HOURS],
        "{name} rebuilds its primary shard {value}.",
        "How often does {name} rebuild its primary shard?",
        setup="{name} maintains a primary shard that is rebuilt on a fixed schedule.",
        value_statement="That rebuild runs {value}.",
        elaboration="Operators are paged automatically if two consecutive rebuilds are missed.",
    ),
    _Attribute(
        "latency_budget",
        [f"{n} milliseconds" for n in _MILLISECONDS],
        "Query latency on {name} is budgeted at {value} for the 99th percentile.",
        "What is the 99th-percentile query latency budget for {name}?",
        setup="{name} publishes a latency budget covering its 99th-percentile query path.",
        value_statement="That budget is set at {value}.",
        elaboration="Sustained breaches of it open an automatic reliability review.",
    ),
    _Attribute(
        "vector_dim",
        [f"{n} dimensions" for n in _DIMENSIONS],
        "{name} stores its vectors in {value}.",
        "How many dimensions does {name} use to store its vectors?",
        setup="{name} keeps one dense vector for every indexed record.",
        value_statement="Each of those vectors is stored in {value}.",
        elaboration="The width was fixed at the last index migration and has not changed since.",
    ),
    _Attribute(
        "centroids",
        [f"{n} centroids" for n in _CENTROIDS],
        "Coarse quantization on {name} uses an inverted file index with {value}.",
        "How many centroids does {name} use for coarse quantization?",
        setup="Coarse quantization on {name} runs through an inverted file index.",
        value_statement="That index is partitioned across {value}.",
        elaboration="Probing more partitions per query trades latency for recall.",
    ),
    _Attribute(
        "retention",
        [f"{n} days" for n in _DAYS],
        "Deleted records on {name} are retained for {value} before purging.",
        "How long does {name} retain deleted records before purging them?",
        setup="{name} does not purge deleted records immediately.",
        value_statement="They are retained for {value} before the purge runs.",
        elaboration="The delay exists so that an accidental deletion can still be reversed.",
    ),
    _Attribute(
        "storage",
        [f"{n} terabytes" for n in _TERABYTES],
        "A full rebuild of {name} consumes roughly {value} of scratch disk.",
        "How much scratch disk does a full rebuild of {name} consume?",
        setup="A full rebuild of {name} stages its intermediate output on scratch disk.",
        value_statement="That staging consumes roughly {value}.",
        elaboration="The scratch volume is provisioned with headroom above this figure.",
    ),
    _Attribute(
        "auth",
        _AUTH_VALUES,
        "Access to the {name} administrative console requires {value}.",
        "What does access to the {name} administrative console require?",
        setup="The {name} administrative console sits behind its own access control.",
        value_statement="Reaching it requires {value}.",
        elaboration="Read-only dashboards are exempt from that requirement.",
    ),
    _Attribute(
        "maintainer",
        _MAINTAINER_VALUES,
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


def _validate_value_pools() -> None:
    """Check the two pool invariants at import rather than trusting them.

    Both are easy to break by editing a value list and impossible to notice from
    the corpus by eye, and either one silently corrupts the distractor results.
    """
    for attr in _ATTRIBUTES:
        for value in attr.values:
            others = [v for v in attr.values if v != value]
            if any(value in other for other in others):
                raise ValueError(
                    f"Value {value!r} of attribute {attr.key!r} is a substring of another "
                    "value in the same pool. An answer would then appear present in a "
                    "document that never stated it. Numeric pools must share a digit count."
                )
        if len(set(attr.values)) != len(attr.values):
            raise ValueError(f"Attribute {attr.key!r} has duplicate values.")


def _value_for(attr: _Attribute, rng: random.Random) -> str:
    """Pick a filler value. Filler never draws from the gold half."""
    return rng.choice(attr.filler_values)


def _build_document(
    doc_id: str,
    name: str,
    queried_key: str | None,
    n_filler: int,
    rng: random.Random,
    is_distractor: bool,
    fact_sentences: int = 1,
    gold_value: str | None = None,
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
        # A real planted fact uses the gold value the caller reserved for it.
        # A distractor's decoy is filler wearing a fact's shape, so it draws
        # from the filler half and can never carry a correct answer.
        value = gold_value if record_as is not None else _value_for(attr, rng)
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

    # Gold values are dealt without replacement per attribute, so no two
    # planted answers for the same attribute are ever the same string. Combined
    # with the gold/filler split, this makes a question's answer appear in
    # exactly one document in the whole corpus -- which is what lets a correct
    # answer be read as evidence of retrieval rather than of a lucky match.
    gold_pools: dict[str, list[str]] = {}
    for attr in _ATTRIBUTES:
        pool = list(attr.gold_values)
        rng.shuffle(pool)
        gold_pools[attr.key] = pool

    documents: list[Document] = []
    questions: list[EvalQuestion] = []

    # Cycle attributes so the eval set is balanced across question types rather
    # than accidentally over-weighting one.
    for i in range(config.n_relevant_docs):
        name = names[i]
        queried_key = _ATTRIBUTES[i % len(_ATTRIBUTES)].key
        doc_id = f"doc-{i:04d}"
        pool = gold_pools[queried_key]
        if not pool:
            raise ValueError(
                f"Ran out of unique gold values for attribute {queried_key!r} at "
                f"n_relevant_docs={config.n_relevant_docs}. Each attribute holds "
                f"{len(_ATTRS_BY_KEY[queried_key].gold_values)} gold values and they are "
                "dealt without replacement; extend that attribute's value pool to go "
                "higher. Reusing one would let a correct answer come from the wrong "
                "document."
            )
        doc, answer = _build_document(
            doc_id,
            name,
            queried_key,
            config.filler_sentences_per_doc,
            rng,
            is_distractor=False,
            fact_sentences=config.fact_sentences,
            gold_value=pool.pop(),
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


_validate_value_pools()
