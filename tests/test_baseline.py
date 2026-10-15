"""Tests for step 3: evaluation helpers, train sampling, and a small end-to-end baseline run."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aml.baseline import evaluate, make_splits, metrics_record, per_pattern, train_search
from aml.config import BaselineConfig
from aml.metrics import at_threshold, threshold_at_fpr

ACH = 0


def make_feature_table(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Minimal frame in the step 2 output schema, with one informative feature."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype("int8")
    split = np.select([np.arange(n) < n * 0.6, np.arange(n) < n * 0.8], ["train", "val"], "test")
    return pd.DataFrame({
        "tx_id": np.arange(n),
        "timestamp": pd.Timestamp("2022-09-01") + pd.to_timedelta(np.arange(n), unit="m"),
        "split": pd.Categorical(split),
        "is_laundering": y,
        "payment_format_code": np.where(rng.random(n) < 0.5, ACH, 1).astype("int8"),
        "pair_prior_cnt": rng.integers(0, 5, n),
        "pair_reverse_prior_cnt": rng.integers(0, 2, n),
        "snd_out_cnt_all": rng.integers(0, 50, n),
        "log_amount_usd": rng.normal(7, 1, n),
        "signal": y * 2.0 + rng.normal(0, 1, n),
        "noise": rng.normal(0, 1, n),
    })


def test_threshold_at_fpr_flags_the_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(100_000) < 0.01).astype(int)
    score = rng.random(len(y)) + y
    thr = threshold_at_fpr(y, score, 0.05)
    assert at_threshold(y, score, thr)["fpr"] == pytest.approx(0.05, abs=0.002)


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = at_threshold(y, score, 0.5)
    assert r == {"recall": 0.5, "precision": 0.5, "fpr": 1 / 3, "alerts": 2, "caught": 1}


def test_make_splits_samples_train_only():
    data = make_feature_table()
    splits = make_splits(data, ["signal", "noise"], BaselineConfig(neg_sample_rate=0.2))
    train = data[data["split"] == "train"]
    # every laundering transaction kept, about 20% of normal ones
    assert splits.y_train.sum() == train["is_laundering"].sum()
    kept_normal = (splits.y_train == 0).sum() / (train["is_laundering"] == 0).sum()
    assert kept_normal == pytest.approx(0.2, abs=0.03)
    # validation and test are complete and aligned with their metadata
    for name, X, y in [("val", splits.X_val, splits.y_val), ("test", splits.X_test, splits.y_test)]:
        part = data[data["split"] == name]
        assert len(X) == len(part) == len(splits.meta[name])
        np.testing.assert_array_equal(y, part["is_laundering"].to_numpy())
        np.testing.assert_allclose(X["signal"], part["signal"].astype("float32"))


def test_make_splits_is_deterministic():
    data = make_feature_table()
    a = make_splits(data, ["signal"], BaselineConfig(seed=7))
    b = make_splits(data, ["signal"], BaselineConfig(seed=7))
    pd.testing.assert_frame_equal(a.X_train, b.X_train)


def test_per_pattern():
    patterns = pd.DataFrame({"tx_id": [1, 2, 3, 99, None], "attempt_id": [10, 10, 11, 12, 13],
                             "pattern_type": ["CYCLE", "CYCLE", "CYCLE", "FAN-OUT", "FAN-OUT"]})
    out = per_pattern(patterns, pd.Series([1, 2, 3, 4]), np.array([True, False, False, True]))
    # tx 99 and the missing tx_id are outside the scored set
    assert list(out.index) == ["CYCLE"]
    assert out.loc["CYCLE", "transactions"] == 3
    assert out.loc["CYCLE", "recall_at_1pct"] == pytest.approx(1 / 3)
    assert out.loc["CYCLE", "attempts_with_any_alert"] == pytest.approx(0.5)


def test_baseline_end_to_end():
    pytest.importorskip("xgboost")
    data = make_feature_table()
    cfg = BaselineConfig(neg_sample_rate=0.5)
    splits = make_splits(data, ["signal", "noise"], cfg)
    model, params, search = train_search(splits, search=[{"max_depth": 2, "scale_pos_weight": 1.0}])
    result = evaluate(model, params, search, splits, ACH, config=cfg)

    assert result.test["pr_auc"] > 5 * result.test["random_pr_auc"]
    assert result.gain_share.index[0] == "signal"
    assert set(result.by_format.index) == {"ACH only", "all other formats"}
    # thresholds come from validation: budgets are met there exactly
    assert result.budget_val.loc[0.05, "fpr"] == pytest.approx(0.05, abs=0.01)

    record = metrics_record(result, splits, cfg)
    json.dumps(record)   # serialisable without a fallback
    assert record["params"]["max_depth"] == 2


def test_paired_bootstrap():
    from aml.metrics import paired_bootstrap
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    weak = rng.random(len(y)) + 0.3 * y
    strong = rng.random(len(y)) + 2.0 * y
    blocks = np.arange(len(y)) // 100
    same = paired_bootstrap(y, weak, weak, blocks, 0.9, 0.9, n_boot=50)
    assert (same.abs().drop(columns="share_of_draws_above_0") == 0).all().all()
    better = paired_bootstrap(y, weak, strong, blocks, 0.9, 0.9, n_boot=50)
    assert better.loc["pr_auc", "ci_low"] > 0
    assert better.loc["pr_auc", "share_of_draws_above_0"] == 1.0


def test_load_splits_adds_aligned_extra_features(tmp_path):
    from aml.baseline import load_splits
    data = make_feature_table(n=500)
    data.drop(columns=["noise"]).to_parquet(tmp_path / "f.parquet")
    data[["tx_id", "noise"]].to_parquet(tmp_path / "g.parquet")
    splits = load_splits(tmp_path / "f.parquet", ["signal"], extra=[(tmp_path / "g.parquet", ["noise"])])
    assert splits.features == ["signal", "noise"]
    np.testing.assert_allclose(splits.X_val["noise"], data.loc[data["split"] == "val", "noise"].astype("float32"))
    data[["tx_id", "noise"]].iloc[::-1].to_parquet(tmp_path / "g.parquet")
    with pytest.raises(ValueError):
        load_splits(tmp_path / "f.parquet", ["signal"], extra=[(tmp_path / "g.parquet", ["noise"])])
