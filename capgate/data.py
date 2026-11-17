"""The replay table: one row per question, as delivered by the UQ repo
(columns fixed by its exporter). `fake` makes stand-in rows with the same columns."""

import numpy as np
import pandas as pd

# USD per 1M tokens (input, output) for the expensive tier: claude-sonnet-5-5.
EXP_PRICE_IN, EXP_PRICE_OUT = 2.00, 10.00

NEEDED = ["query_id", "family", "split", "score", "cheap_correct", "exp_correct",
          "exp_tokens_in", "exp_tokens_out"]


def load(path):
    df = pd.read_parquet(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"replay table is missing columns: {missing}")
    return df


def fake(n=2000, seed=0, cheap_acc=0.6, exp_acc=0.9):
    """Stand-in rows. The score is informative but imperfect (AUROC about 0.75)."""
    rng = np.random.default_rng(seed)
    cheap = rng.random(n) < cheap_acc
    logit = np.log(cheap_acc / (1 - cheap_acc)) + np.where(cheap, 0.7, -0.7) + rng.normal(0, 1, n)
    return pd.DataFrame({
        "query_id": [f"fake-{i}" for i in range(n)],
        "family": rng.choice(["math", "finance", "medical", "open_qa", "mc_knowledge", "reading_comprehension"], n),
        "split": rng.choice(["dev", "calib"], n),
        "score": 1 / (1 + np.exp(-logit)),
        "cheap_correct": cheap.astype(int),
        "exp_correct": (rng.random(n) < exp_acc).astype(int),
        "exp_tokens_in": rng.lognormal(5.6, 0.8, n).astype(int) + 50,
        "exp_tokens_out": rng.lognormal(4.5, 0.7, n).astype(int) + 5,
    })
