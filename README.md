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

🚧 **In progress.** The pipeline is complete and tested. One of the three stress tests has been
run; the other two and the paper's results sections are outstanding.

**Experiment 1 — chunk-size sensitivity** (8 chunk sizes, n=80 questions, `claude-opus-5`):

| chunk size (chars) | 64 | 128 | 200 | 300 | 400 | 600 | 800 | 1200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hit rate@5 | 0.000 | 0.263 | 0.700 | 0.850 | 0.938 | 0.875 | 1.000 | 1.000 |
| answer accuracy | 0.025 | 0.425 | 0.825 | 0.912 | 0.988 | 0.875 | 1.000 | 1.000 |
| hallucination rate | 0.013 | 0.025 | 0.000 | 0.000 | 0.000 | 0.013 | 0.000 | 0.000 |

Retrieval and answer quality are reported separately and never combined. Two observations, both
with 95% confidence intervals computed and neither yet written up: answer accuracy exceeds hit
rate@5 wherever retrieval is imperfect — by 16 points at chunk size 128 — because the generator
reassembles facts from fragments that the retrieval metric scores as misses; and degradation runs
almost entirely through abstention rather than fabrication.

Full per-condition results are in `results/`. Read every number against its N: n=80 puts the 95%
interval on a proportion at roughly ±11 points, so only large movements support a claim.

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
