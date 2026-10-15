"""Chronological train / validation / test split."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> pd.DataFrame:
    """Assign `split` by time: the first `train_frac` of transactions to train, the
    next `val_frac` to val, the rest to test.

    Cut points snap to a timestamp so one minute never spans two splits. Any feature
    later computed for a val/test row must use only data up to that row's timestamp.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    df = df.sort_values(["timestamp", "tx_id"], kind="stable").reset_index(drop=True)
    n = len(df)
    t1 = df["timestamp"].iloc[int(n * train_frac)]
    t2 = df["timestamp"].iloc[int(n * (train_frac + val_frac))]
    labels = np.where(df["timestamp"] < t1, "train", np.where(df["timestamp"] < t2, "val", "test"))
    df["split"] = pd.Categorical(labels, categories=SPLITS)
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split", observed=True)
    out = pd.DataFrame({
        "rows": g.size(),
        "laundering": g["is_laundering"].sum(),
        "laundering_rate": g["is_laundering"].mean(),
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
    })
    return out.reindex(SPLITS)
