# Where Retrieval-Augmented Generation Breaks: Three Controlled Stress Tests

**Status: draft. Method and limitations are complete; results sections are empty
pending experiment runs.** No numbers appear anywhere in this document until the
experiments have actually been run. Placeholders are marked `[PENDING]` rather
than filled with plausible estimates.

---

## 1. Motivation

Retrieval-augmented generation is usually demonstrated rather than tested. A
typical demo shows a pipeline answering a question correctly and stops there,
which establishes that the technique can work but says nothing about when it
stops working, or which of its stages fails first.

That gap matters in practice, because a RAG pipeline has two failure points that
look identical from the outside. A wrong answer can mean the retriever never
surfaced the relevant passage, or it can mean the retriever surfaced it and the
generator ignored it. These call for opposite fixes — better chunking or
embeddings in the first case, better prompting or a different model in the
second — and a single end-to-end accuracy score cannot distinguish them.

This paper characterizes three known failure modes under controlled conditions,
measuring retrieval quality and answer quality as separate quantities
throughout:

1. **Chunk-size sensitivity.** How retrieval and answer quality vary with the
   chunk size and chunking strategy used at index time.
2. **Distractor sensitivity.** What happens as the corpus fills with documents
   that are topically and lexically similar to the answer-bearing one but
   contain no answer.
3. **Long-context vs. retrieved-context.** Whether stuffing an entire corpus
   into a long context window outperforms retrieval, and whether the position of
   the answer within that context affects whether the model finds it — the
   "lost in the middle" effect.

### 1.1 Relation to existing work

The three phenomena studied here are not novel discoveries; each has been
reported in the literature. The contribution of this work is a single controlled
harness that measures all three against one fixed corpus and one fixed question
set, with retrieval and answer quality separated, and with exact ground truth
rather than approximate string matching.

- **Lost in the middle.** Liu et al. (2023) showed that language models retrieve
  information most reliably from the start and end of a long context, with a
  measurable dip in the middle. Experiment 3 replicates this probe on a current
  model and contrasts it with retrieval on the same corpus.
- **Distractor sensitivity and retrieval noise.** Work on noisy retrieval (e.g.
  Cuconasu et al., 2024, on the role of irrelevant and distracting passages)
  reports that irrelevant context can degrade generation quality, and that
  *near-miss* distractors are more damaging than random ones. Experiment 2 uses
  distractors generated from the same templates as the gold documents, which is
  the near-miss case by construction.
- **Chunking.** Chunk size is widely treated as a tuning parameter in practice
  but is less often measured as an independent variable with retrieval and
  answer quality reported apart. Experiment 1 does so.

*(Citations to be completed with full references before submission.)*

---

## 2. Method

### 2.1 Pipeline

Every stage is implemented from scratch: no orchestration framework is used
anywhere. The stages are ingestion, chunking, embedding, retrieval, generation,
and evaluation.

| Stage | Implementation |
| --- | --- |
| Chunking | Hand-written fixed-width and recursive character splitters |
| Embedding | `sentence-transformers` `all-MiniLM-L6-v2`, run locally, L2-normalized |
| Vector store | FAISS `IndexFlatIP` used directly — exact search, no approximation |
| Generation | Claude (`claude-opus-5`) via the `anthropic` SDK |
| Scoring | Retrieval metrics computed locally; answer correctness by LLM judge |

Exact (flat) search is deliberate. An approximate index introduces recall loss
of its own, which would confound experiments whose subject *is* retrieval recall.

### 2.2 Corpus

The corpus is synthetic and seeded. Each document describes a fictional software
system — names like "Halcyon Index" — using a fixed set of sentence templates
covering eight attributes (rebuild cadence, latency budget, vector
dimensionality, maintainer, and so on). Each relevant document contains exactly
one *planted fact*: the sentence stating the attribute that its question asks
about. The remaining sentences describe other attributes of the same system and
are never queried.

Three properties motivate this design over a public QA dataset:

1. **Exact ground truth.** The character span of every answer is known, so
   retrieval metrics are computed against a precise label rather than a fuzzy
   string match.
2. **Controlled distractors.** Distractor documents are generated from the same
   templates, differing only in system name and attribute values. They are
   lexically and topically near-identical to gold documents while containing no
   correct answer — the hard case for a dense retriever. Sampling "irrelevant"
   documents from an unrelated corpus would make the task trivially easy.
3. **No pretraining leakage.** The entities are fictional, so a correct answer
   is evidence that retrieval supplied the fact rather than that the model
   already knew it.

A test asserts that no distractor document contains an answer to any question,
so the distractor condition cannot accidentally become answerable.

### 2.3 Ground truth that survives re-chunking

Retrieval labels are stored as **character spans**, not chunk identifiers. Chunk
boundaries move whenever chunk size changes, so a label of the form "chunk 7 is
correct" means something different in every condition of Experiment 1.

A chunk is counted as relevant to a question if it comes from that question's
gold document and overlaps the planted fact's span by at least 50% of the fact's
length. The threshold is a parameter; 50% is the reported setting, chosen so
that a chunk holding a bare sliver of the fact does not count as answer-bearing
while a chunk holding most of it does.

### 2.4 Metrics

Retrieval and answer quality are reported separately and never combined into a
single score.

**Retrieval.**

| Metric | Definition |
| --- | --- |
| hit rate@k | Fraction of questions with ≥1 relevant chunk in the top *k* |
| precision@k | Relevant chunks retrieved / *k* |
| recall@k | Relevant chunks retrieved / relevant chunks in corpus |
| MRR | Mean of 1/(rank of first relevant chunk) |

A note on which to read. Textbook recall@k divides by the *total* number of
relevant chunks in the corpus — a denominator that itself changes with chunk
size, since a small chunking may split one fact across several chunks while a
large one keeps it whole. That makes recall@k incomparable across precisely the
conditions Experiment 1 varies. **hit rate@k has a denominator of one per
question by construction**, is comparable across all conditions, and is the
quantity that actually predicts whether generation can succeed. It is the
headline retrieval metric here; recall@k is reported alongside for completeness.

**Answer quality.** Outcomes are three-way rather than binary:

| Outcome | Meaning |
| --- | --- |
| Correct | The judge rules the prediction semantically equivalent to the reference |
| Abstained | The model replied `NOT IN CONTEXT` |
| Hallucinated | Wrong, and did not abstain |

The distinction between abstention and hallucination is central to Experiment 2.
A model that says "I cannot find this" when distractors crowd out the gold
passage has failed far less harmfully than one that confidently reports a
different system's numbers, and a binary correct/incorrect score treats them
identically.

Answers are graded by an LLM judge, because generated answers are full sentences
and exact string matching would penalize correct paraphrases. The judge is
prompted to rule on semantic equivalence of the key fact. An unparseable verdict
is scored *incorrect*, so a judge failure can never inflate reported accuracy.

### 2.5 Experimental controls

- Exactly one variable changes per experiment; the config loader rejects any
  configuration that sweeps more than one.
- The eval question set is fixed across all conditions within an experiment, and
  each condition's corpus is verified to still contain every gold fact before
  scoring.
- Corpus generation, distractor sampling, and fact placement are all seeded, so
  the relevant documents are byte-identical across every distractor ratio; only
  the surrounding noise changes.
- The planted fact is placed at a random position within its document rather
  than always first, so within-document position is not a hidden confound.

### 2.6 Reproducibility, and its limit

Everything except the language model is deterministic under a fixed seed.

The model is not, and cannot be made so by the usual means: current Claude
models have **removed the sampling parameters**, so `temperature=0` is not
available — sending `temperature` to `claude-opus-5` returns an API error.
Reproducibility is instead achieved with a content-addressed response cache:
every call is keyed by a SHA-256 of the model, system prompt, user prompt, and
parameters, and cached to disk. Re-running an experiment replays cached
responses exactly.

This makes a *re-run* of a completed experiment exactly reproducible. It does
**not** make a cold run on a fresh cache reproducible, and the residual sampling
variance across cold runs is unquantified in this work. See Limitations.

---

## 3. Experiment 1 — Chunk-size sensitivity

**Varied:** chunk size (64–1200 characters), and separately, chunking strategy
(fixed-width vs. recursive) at a fixed size.
**Held fixed:** corpus, *k*, eval question set, embedding model, generation model.
**Overlap:** zero, so that a fact severed by a boundary stays severed.

**Hypothesis.** As chunk size falls below the length of a planted fact, the fact
is split across a boundary and no single chunk remains answer-bearing, so
retrieval collapses. At the other extreme, large chunks retrieve the fact
reliably but dilute it with unrelated filler, which should show up as a
generation problem rather than a retrieval one — retrieval metrics holding while
answer accuracy softens.

*(A preliminary check in the integration test suite confirms the low end of this
effect is real in this corpus: hit rate at 32-character chunks is measurably
below hit rate at 400-character chunks.)*

### 3.1 Results

`[PENDING — experiment not yet run]`

---

## 4. Experiment 2 — Distractor sensitivity

**Varied:** ratio of distractor documents to relevant documents (0× to 4×).
**Held fixed:** the 40 relevant documents, chunking, *k*, eval question set.

**Hypothesis.** Two failure modes should appear and should be separable:

- The gold chunk is pushed out of the top *k* by near-identical distractors — a
  **retrieval** failure, visible as falling hit rate@k and MRR.
- The gold chunk survives in the top *k*, but the generator answers with a
  distractor's value anyway — a **generation** failure, visible as answer
  accuracy falling faster than hit rate@k.

The abstention/hallucination split indicates which way the model fails as noise
rises.

### 4.1 Results

`[PENDING — experiment not yet run]`

---

## 5. Experiment 3 — Long-context vs. retrieved-context

**Conditions:** `rag` (top-*k* retrieval), and three full-context conditions in
which the entire corpus is placed in the prompt with the gold document near the
start, the middle, or the end.
**Held fixed:** model, corpus, eval question set.

The three stuffed conditions contain the identical set of documents in the
identical relative order, differing only in the index of one document. If
accuracy dips for the middle position relative to start and end, position is
what moved it, because nothing else did.

Retrieval metrics are undefined for the stuffed conditions — there is no
retrieval step — and are reported as blank rather than zero. A zero would read
as "retrieval failed" when the truth is that retrieval did not happen.

### 5.1 Results

`[PENDING — experiment not yet run]`

---

## 6. Limitations

These are stated plainly because the results are only worth as much as their
caveats.

1. **Small N.** The eval set is 40 questions (30 for Experiment 3). Differences
   of a few percentage points between adjacent conditions are within noise for
   samples this size and should not be read as trends. Only large, monotonic
   movements support any claim.

2. **Synthetic corpus.** The corpus buys exact ground truth and controlled
   distractors at the cost of realism. Its documents are short, uniformly
   structured, and template-generated. Real corpora have irregular structure,
   redundancy, and facts that are stated across several sentences rather than
   one. The *mechanisms* demonstrated here should generalize; the specific
   thresholds — the chunk size at which retrieval collapses, the distractor
   ratio at which accuracy falls — almost certainly do not.

3. **Single-sentence facts.** Every planted fact is one self-contained sentence.
   This makes the chunk-boundary mechanism unusually clean and probably
   overstates how sharply chunk size matters relative to a corpus where answers
   require combining information across sentences. No multi-hop questions are
   tested.

4. **One embedding model, one generation model.** All results are for
   `all-MiniLM-L6-v2` and `claude-opus-5`. The 384-dimensional MiniLM embedder is
   small by current standards, and a stronger embedder would likely shift the
   distractor results in particular. Nothing here should be read as a claim
   about RAG in general as opposed to this configuration.

5. **LLM-as-judge.** Answer correctness is graded by a model, which introduces
   its own error. The judge is not validated against human labels in this work.
   Its failure mode is partly mitigated — unparseable verdicts count as
   incorrect, so judge confusion deflates rather than inflates accuracy — but
   its agreement rate with a human grader is unmeasured.

6. **Cold-run non-determinism.** As described in §2.6, the response cache makes
   re-runs exact but does not eliminate sampling variance on a cold cache.
   Reported numbers come from a single cold run per condition; no variance
   across repeated cold runs is estimated. A repeated-run variance estimate would
   materially strengthen the results and is the most valuable single addition to
   this work.

7. **Position probe granularity.** Experiment 3 tests three positions
   (start/middle/end), not a fine-grained sweep, so it can detect the presence of
   a position effect but not its shape.

---

## 7. Reproducing this work

```bash
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY

python experiments/chunk_size_sensitivity/run.py
python experiments/distractor_sensitivity/run.py
python experiments/long_context_vs_retrieval/run.py
python scripts/make_figures.py --tables
```

Per-condition aggregates land in `results/*.csv`; per-question detail, including
every prediction and every judge verdict, lands in `results/logs/*.jsonl`.
