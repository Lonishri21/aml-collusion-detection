"""Unit tests for step 1. Run with: pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.cleaning import add_flags, add_node_ids, add_usd_amounts, estimate_fx_rates, normalize_bank, trim_after
from aml.io import load_patterns
from aml.splitting import temporal_split

RATES = {"US Dollar": 1.0, "Euro": 1.17, "Rupee": 0.012}


def make_transactions(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame in the post-load (renamed) schema."""
    rng = np.random.default_rng(seed)
    currencies = list(RATES)
    pay = rng.choice(currencies, n)
    recv = np.where(rng.random(n) < 0.5, pay, rng.choice(currencies, n))
    paid = rng.uniform(10, 10_000, n).round(2)
    received = paid * np.array([RATES[c] for c in pay]) / np.array([RATES[c] for c in recv])
    return pd.DataFrame({
        "tx_id": np.arange(n),
        "timestamp": pd.Timestamp("2022-09-01") + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 5, n)), unit="m"),
        "from_bank": rng.choice(["010", "10", "3208"], n),
        "from_account": rng.choice(["A1", "A2", "A3"], n),
        "to_bank": rng.choice(["010", "3208"], n),
        "to_account": rng.choice(["A1", "B1"], n),
        "amount_paid": paid,
        "pay_currency": pd.Categorical(pay),
        "amount_received": received,
        "recv_currency": pd.Categorical(recv),
        "payment_format": pd.Categorical(rng.choice(["ACH", "Cheque", "Wire"], n)),
        "is_laundering": (rng.random(n) < 0.05).astype("int8"),
    })


def test_normalize_bank_strips_leading_zeros():
    out = normalize_bank(pd.Series(["010", "10", " 0003208 ", "ABC"]))
    assert out.tolist() == ["10", "10", "3208", "ABC"]


def test_same_account_number_at_different_banks_is_two_nodes():
    df = make_transactions()
    df, nodes = add_node_ids(df)
    assert {"10_A1", "3208_A1"} <= set(nodes["node_id"])
    assert "010_A1" not in set(nodes["node_id"])           # leading zero normalised away
    assert (nodes["node_id"].to_numpy()[df["from_idx"]] == df["from_node"].to_numpy()).all()


def test_fx_rates_recovered_from_data():
    df = make_transactions()
    rates, diag = estimate_fx_rates(df)
    for cur, rate in RATES.items():
        assert rates[cur] == pytest.approx(rate, rel=1e-3)
    assert (diag["coef_of_variation"].fillna(0) < 1e-3).all()


def test_usd_amounts_agree_on_cross_currency_rows():
    df = make_transactions()
    rates, _ = estimate_fx_rates(df)
    df = add_usd_amounts(df, rates)
    gap = (df["amount_paid_usd"] - df["amount_received_usd"]).abs() / df["amount_paid_usd"]
    assert gap.max() < 1e-3


def test_missing_usd_pair_raises():
    df = make_transactions()
    df["pay_currency"] = df["pay_currency"].cat.add_categories("Yen")
    df.loc[0, ["pay_currency", "recv_currency"]] = ["Yen", "Euro"]
    with pytest.raises(ValueError, match="Yen"):
        estimate_fx_rates(df)


def test_flags():
    df, _ = add_node_ids(make_transactions())
    df, codes = add_flags(df)
    assert set(codes) == {"ACH", "Cheque", "Wire"}
    self_loop = df["from_node"] == df["to_node"]
    assert (df["is_self_loop"] == self_loop.astype("int8")).all()


def test_trim_after():
    df = make_transactions()
    out = trim_after(df, "2022-09-03")
    assert out["timestamp"].max() < pd.Timestamp("2022-09-03")
    assert trim_after(df, None) is df


def test_temporal_split_is_ordered_and_does_not_share_minutes():
    df = temporal_split(make_transactions(1_000))
    bounds = df.groupby("split", observed=True)["timestamp"].agg(["min", "max"]).loc[["train", "val", "test"]]
    assert (bounds["max"].iloc[:-1].to_numpy() < bounds["min"].iloc[1:].to_numpy()).all()
    assert df["split"].value_counts(normalize=True)["train"] == pytest.approx(0.6, abs=0.05)


def test_temporal_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        temporal_split(make_transactions(), train_frac=0.8, val_frac=0.3)


def test_load_patterns_handles_truncated_file(tmp_path):
    row = "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1"
    text = "\r\n".join([
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out", row, row,
        "END LAUNDERING ATTEMPT - FAN-OUT", "",
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops", row,
        "2022/09/04 04:45,0223",                              # cut-off line
    ])
    path = tmp_path / "patterns.txt"
    path.write_text(text)
    p = load_patterns(path)
    assert len(p) == 3
    assert p.groupby("pattern_type")["attempt_complete"].first().to_dict() == {"CYCLE": False, "FAN-OUT": True}
    assert p["amount_paid"].dtype == "float64"
