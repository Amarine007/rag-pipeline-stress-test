# RAG Pipeline Stress Test

A retrieval-augmented generation pipeline written from scratch, and a short empirical study of
three ways it breaks.

Most RAG demos are built to show the technique working. This one is built to find the edges:
where retrieval succeeds but generation fails, where adding more context makes answers worse, and
where a chunking parameter nobody thinks about quietly costs you most of your recall.

## What's hand-written here

Every stage. Ingestion, chunking, embedding, retrieval, generation, and evaluation are all
implemented directly — there is no LangChain, LlamaIndex, or comparable orchestration layer
anywhere in this repository. FAISS is used as a bare index. Claude is called through the
`anthropic` SDK directly.

This is deliberate: the project is about understanding RAG internals, and a framework that hides
the retrieval loop hides exactly the thing being studied.

## The three stress tests

| Experiment | Varied | Held fixed | Retrieval metrics | Answer metrics |
| --- | --- | --- | --- | --- |
| Chunk-size sensitivity | Chunk size and chunking strategy | Corpus, `k`, eval set | recall@k, precision@k, MRR | Answer correctness |
| Distractor sensitivity | Ratio of topically similar but irrelevant documents | Corpus size, chunk size, `k`, eval set | recall@k, precision@k | Answer correctness |
| Long-context vs. retrieved-context | Full-context stuffing vs. retrieval; position of the fact in context | Model, corpus, eval set | — (no retrieval in the stuffed condition) | Answer correctness by fact position |

Retrieval quality and answer quality are always reported separately. A pipeline can fail at
either stage independently, and collapsing them into a single number throws away the only
interesting signal.

## Status

**All three stress tests have run and are written up** in [`docs/paper_draft.md`](docs/paper_draft.md).
One produced a strong positive result; the other two produced a negative and a null result, and
are reported as such.

### Experiment 1 — chunk-size sensitivity

9 chunk sizes, n=80 questions, `claude-opus-5`. Documents in this corpus run 611–762 characters.

| chunk size (chars) | 64 | 128 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hit rate@5 | 0.000 | 0.263 | 0.700 | 0.850 | 0.938 | 0.875 | 0.875 | 1.000 | 1.000 |
| answer accuracy | 0.025 | 0.425 | 0.825 | 0.912 | 0.988 | 0.950 | 0.875 | 1.000 | 1.000 |
| hallucination rate | 0.013 | 0.025 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.000 | 0.000 |

Three findings:

- **The worst chunk size is not the smallest.** Accuracy peaks at 400, dips to a trough at 600,
  then recovers completely at 700. The trough's interval does not overlap the recovery's, so the
  curve is genuinely non-monotonic. A chunk size *slightly below typical document length* is a
  trap: it cuts each document into one large chunk and one useless fragment. This is a claim
  about the relationship between chunk size and document length, not about any absolute size.
- **Answer accuracy exceeds hit rate@5 wherever retrieval is imperfect** — by 16 points at chunk
  size 128 — because the generator reassembles facts from fragments the retrieval metric scores
  as misses. The gap is the measurement of how much the generator rescues a mediocre retriever.
- **Degradation runs through abstention, not fabrication.** Hallucination never exceeds 2
  questions in 80, and is exactly zero in five of nine conditions.

A companion run varies only the chunking *strategy* at a fixed 128-character size: boundary-aware
recursive splitting more than doubles hit rate@5 (0.263 → 0.625) and takes hallucination to zero,
without changing the chunk size at all.

### Experiment 2 — distractor sensitivity

6 ratios (0× to 4×, up to 400 documents), n=80 questions. **A negative result, reported as one:**
neither hypothesized failure mode appeared. Hallucination stayed at exactly 0.000 in every
condition, and accuracy tracked hit rate@5 to three decimals throughout — the signature of a
generator that answers when the retriever supplies the fact and abstains when it does not.

| distractor ratio | 0× | 0.5× | 1× | 2× | 3× | 4× |
| --- | --- | --- | --- | --- | --- | --- |
| hit rate@5 | 1.000 | 1.000 | 1.000 | 0.988 | 0.938 | 0.938 |
| MRR | 0.933 | 0.917 | 0.899 | 0.872 | 0.846 | 0.846 |
| answer accuracy | 1.000 | 1.000 | 1.000 | 0.988 | 0.938 | 0.938 |
| hallucination rate | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

The signal is in **MRR, not hit rate**. MRR falls steadily (0.933 → 0.846) while hit rate@5 holds
at 1.000 through 1×: distractors are pushing the gold chunk down the ranking, but k=5 has enough
headroom to absorb it. A pipeline reporting only hit rate@k at a generous k would read as robust
here while the same pipeline at k=1 was already losing answers.

Failures are also a property of the question rather than of the noise — the set of failing
questions is strictly nested as distractors are added, and the same five fail at both 3× and 4×.

A follow-up sweep at **k=1** removes the four spare retrieval slots that were absorbing the
ranking pressure. Degradation roughly doubles (11.2 points vs 6.2 from 0× to 4×) and becomes
monotonic at every step — confirming that what MRR was measuring at k=5 was real, and that the
headroom, not the retriever's robustness, was keeping it away from the generator. Hallucination
is still exactly 0.000 in all six conditions: across both sweeps, **960 questions without a single
fabricated answer.**

### Reproducibility — how much is sampling noise?

`temperature` no longer exists on current Claude models, so cold runs can't be pinned. The two
conditions the chunk-size claim depends on (400, the peak; 600, the trough) were re-run three more
times with the response cache **disabled**, 310 fresh API calls each:

| draw | chunk 400 | chunk 600 |
| --- | --- | --- |
| original | 0.9875 | 0.8750 |
| cold run 1 | 0.9875 | 0.8750 |
| cold run 2 | 0.9875 | 0.8750 |
| cold run 3 | 0.9875 | 0.8750 |

**Aggregate variance across four independent draws is exactly zero**, and all 960 question-level
verdict comparisons agree — the same questions fail, by identity, every time. The model is *not*
deterministic (10–20% of answer texts differ between draws), but the variance lives entirely in
phrasing, not in which fact gets reported. Two conditions were sampled, both lopsided; the
near-coin-flip condition at 128 was not, and that caveat is stated in the paper.

### Experiment 3 — long-context vs. retrieved-context

4 conditions, n=50 questions, 100-document corpus. **A null result.** All four conditions answered
50 of 50 correctly, so no lost-in-the-middle effect could be detected.

| condition | rag | full_start | full_middle | full_end |
| --- | --- | --- | --- | --- |
| mean input tokens | 872 | 26,491 | 26,491 | 26,491 |
| answer accuracy | 1.000 | 1.000 | 1.000 | 1.000 |

The manipulation itself was correct — the gold document sits at index 0, 50 and 99 of 100, and
input tokens are identical to the digit across the three stuffed conditions. The task was simply
too easy to register a position effect: 26k tokens is a small fraction of the model's window, and
each question names the specific system it asks about, making the task closer to lookup than
synthesis. At 50/50 the 95% interval is [0.929, 1.000], so **an effect smaller than ~7 points
could not have been seen.** This rules out a large position effect under these conditions and
nothing more; redesigning it with a far longer context and synthesis-style questions is the
highest-value follow-up in the project.

The one usable comparison is cost. The stuffed conditions match retrieval's accuracy while sending
**30× more input tokens per question** (26,491 vs 872) — roughly $0.13 against $0.004 per question
— and that gap widens linearly with corpus size. Where both approaches are equally accurate,
retrieval is the same answer at a thirtieth of the price.

---

Full per-condition results are in `results/`, per-question detail in `results/logs/`. Read every
number against its N: n=80 puts the 95% Wilson interval on a proportion near 0.5 at roughly ±11
points, so only large movements support a claim.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use and runs locally
thereafter — no embedding API calls, no per-run cost, and identical vectors on every run.

## Reproducibility

Everything that can be seeded, is: corpus generation, distractor sampling, fact placement, and
any shuffling all derive from a seed set in the experiment config.

The LLM is the exception. On `claude-opus-5`, `temperature` and the other sampling parameters
have been removed from the API, so generation cannot be pinned to a fixed temperature. Instead
every LLM call is cached on disk, keyed by a hash of the model, system prompt, user prompt, and
parameters. Re-running an experiment replays those cached responses exactly and costs nothing;
only genuinely new prompt/config combinations reach the API. Sampling variance across cold runs
is a real limitation and is reported as one.

## Layout

```
src/            # library code — the pipeline itself
  ingestion/    #   document loading and cleaning
  chunking/     #   chunking strategies
  embeddings/   #   embedding generation
  retrieval/    #   FAISS index wrapper and retriever
  generation/   #   Claude call wrapper and prompt templates
  evaluation/   #   recall@k, precision@k, MRR, answer-correctness scoring
  pipeline.py   #   end-to-end orchestration
experiments/    # one-off run and analysis code, one directory per stress test
configs/        # experiment configs — chunk size, k, ratios, seeds, model IDs
data/           # raw corpus, processed chunks, eval question sets
results/        # logs and figures
docs/           # paper draft
tests/          # unit and integration tests
scripts/        # setup and run entry points
```

## License

MIT
