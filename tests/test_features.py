"""Tests for step 2 features: correctness against a brute-force version, and no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.features import build_features, entity_table


def make_clean_tx(n: int = 600, n_nodes: int = 25, seed: int = 1) -> pd.DataFrame:
    """Minimal frame in the step 1 output schema."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n)
    dst = np.where(rng.random(n) < 0.1, src, rng.integers(0, n_nodes, n))   # ~10% self-loops
    minutes = np.sort(rng.integers(0, 60 * 24 * 4, n))
    amount = rng.lognormal(6, 1.5, n)
    df = pd.DataFrame({
        "tx_id": np.arange(n),
        "timestamp": pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m"),
        "from_idx": src, "to_idx": dst,
        "from_node": [f"1_{i}" for i in src], "to_node": [f"1_{i}" for i in dst],
        "amount_paid_usd": amount, "amount_received_usd": amount,
        "payment_format_code": rng.integers(0, 3, n).astype("int8"),
        "is_cross_bank": np.zeros(n, "int8"),
        "is_self_loop": (src == dst).astype("int8"),
        "is_laundering": (rng.random(n) < 0.05).astype("int8"),
    })
    df["split"] = pd.Categorical(np.where(df.index < n * 0.6, "train", "test"))
    return df


def brute_force(tx: pd.DataFrame, i: int) -> dict:
    """Slow, obviously-correct reference for row i."""
    row = tx.iloc[i]
    past = tx[(tx["timestamp"] < row["timestamp"]) & (tx["is_self_loop"] == 0)]
    s, d, t = row["from_idx"], row["to_idx"], row["timestamp"]
    out_s = past[past["from_idx"] == s]
    in_d = past[past["to_idx"] == d]
    out_24h = out_s[out_s["timestamp"] > t - pd.Timedelta(hours=24)]
    return {
        "snd_out_cnt_all": len(out_s),
        "snd_out_cnt_24h": len(out_24h),
        "snd_out_sum_usd_24h": out_24h["amount_paid_usd"].sum(),
        "snd_out_distinct_cp_all": out_s["to_idx"].nunique(),
        "snd_out_max_usd_all": out_s["amount_paid_usd"].max() if len(out_s) else np.nan,
        "rcv_in_cnt_all": len(in_d),
        "rcv_in_distinct_cp_all": in_d["from_idx"].nunique(),
        "pair_prior_cnt": int(((past["from_idx"] == s) & (past["to_idx"] == d)).sum()),
        "pair_reverse_prior_cnt": int(((past["from_idx"] == d) & (past["to_idx"] == s)).sum()),
        "snd_self_cnt_all": int(((tx["timestamp"] < t) & (tx["from_idx"] == s) & (tx["is_self_loop"] == 1)).sum()),
    }


def test_matches_brute_force():
    tx = make_clean_tx()
    feats = build_features(tx)
    for i in np.random.default_rng(0).choice(len(tx), 80, replace=False):
        for col, expected in brute_force(tx, i).items():
            got = feats.iloc[i][col]
            if pd.isna(expected):
                assert pd.isna(got), (i, col)
            else:
                assert got == pytest.approx(expected, rel=1e-4), (i, col, got, expected)


def test_no_future_leakage():
    """Deleting every later transaction must not change features of earlier ones."""
    tx = make_clean_tx()
    full = build_features(tx)
    cutoff = tx["timestamp"].iloc[350]
    early = build_features(tx[tx["timestamp"] <= cutoff])
    cols = [c for c in early.columns if c not in {"timestamp", "split"}]
    pd.testing.assert_frame_equal(full.loc[early.index, cols], early[cols])


def test_same_minute_transactions_do_not_see_each_other():
    tx = make_clean_tx(n=4, n_nodes=3)
    tx["timestamp"] = pd.Timestamp("2022-09-01 10:00")
    tx["from_idx"], tx["to_idx"] = [0, 0, 0, 0], [1, 1, 2, 2]
    tx["from_node"], tx["to_node"] = "1_0", ["1_1", "1_1", "1_2", "1_2"]
    tx["is_self_loop"] = 0
    feats = build_features(tx)
    assert (feats["snd_out_cnt_all"] == 0).all()
    assert (feats["pair_prior_cnt"] == 0).all()


def test_labels_are_not_features():
    feats = build_features(make_clean_tx())
    from aml.features import feature_columns
    assert "is_laundering" not in feature_columns(feats)


def test_entity_features():
    tx = make_clean_tx(n=50, n_nodes=4)
    accounts = pd.DataFrame({
        "bank": ["01", "1", "1", "1"],                # '01' must normalise to '1'
        "account": ["0", "1", "2", "3"],
        "entity_id": ["E1", "E1", "E2", "E3"],
        "entity_name": ["Corporation #1", "Corporation #1", "Partnership #7", "Individual #9"],
    })
    assert entity_table(accounts)["entity_n_accounts"].tolist() == [2, 2, 1, 1]
    feats = build_features(tx, accounts)
    pair01 = (tx["from_idx"] == 0) & (tx["to_idx"] == 1)
    assert (feats.loc[pair01, "same_entity"] == 1).all()
    assert (feats["snd_entity_type"] >= 0).all()
