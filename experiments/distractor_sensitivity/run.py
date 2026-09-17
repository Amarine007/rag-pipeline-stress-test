"""Experiment 2 -- Distractor sensitivity.

Injects increasing ratios of topically-similar-but-irrelevant documents while
holding the relevant documents, chunking, k, and the eval question set fixed.

Distractors are generated from the same sentence templates as relevant
documents, differing only in system name and attribute values. They are
therefore near-identical in topic and vocabulary to the gold document while
containing no correct answer -- the hard case for a dense retriever, not the
easy one.

Two distinct failure modes are expected, and the point of measuring retrieval
and answers separately is to tell them apart:

  - the gold chunk is pushed out of the top k          -> retrieval failure
  - the gold chunk survives but the model answers with
    a distractor's value anyway                        -> generation failure

The abstention vs. hallucination split in the answer metrics further separates
a model that degrades gracefully (says it cannot find the answer) from one that
degrades dangerously (confidently reports the wrong system's numbers).

    python experiments/distractor_sensitivity/run.py
    python experiments/distractor_sensitivity/run.py --values 0.0 1.0 4.0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._harness import run_sweep_experiment  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "distractor_sensitivity.yaml"

if __name__ == "__main__":
    sys.exit(run_sweep_experiment(DEFAULT_CONFIG))
