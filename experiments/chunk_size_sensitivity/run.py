"""Experiment 1 -- Chunk-size sensitivity.

Sweeps chunk size against a fixed corpus, fixed k, and a fixed eval question
set, measuring retrieval quality and answer quality separately.

The mechanism under test: as chunk size falls below the length of a planted
fact, the fact is severed across a chunk boundary and no single chunk remains
answer-bearing, so retrieval collapses. At the other end, large chunks retrieve
the fact reliably but bury it in filler -- a generation problem, not a retrieval
one. Reporting both metric families is what separates the two.

    python experiments/chunk_size_sensitivity/run.py
    python experiments/chunk_size_sensitivity/run.py --config configs/chunk_strategy_sensitivity.yaml
    python experiments/chunk_size_sensitivity/run.py --values 128 400 800
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._harness import run_sweep_experiment  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "chunk_size_sensitivity.yaml"

if __name__ == "__main__":
    sys.exit(run_sweep_experiment(DEFAULT_CONFIG))
