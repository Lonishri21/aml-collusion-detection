"""Tests for step 7: the feature store reproduces batch features, scoring, drift monitor, API."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.features import build_features, feature_columns
from aml.graph_features import build_graph_features
from aml.metrics import psi
from aml.serving import DriftMonitor, FeatureStore, Scorer
from test_features import make_clean_tx

FORMATS = {"ACH": 0, "Cash": 1, "Cheque": 2}
HUBS = np.array([2])


@pytest.fixture(scope="module")
def history():
    tx = make_clean_tx(n=1500, n_nodes=40, seed=11)
    tx["tx_id"] = np.arange(len(tx))
    batch = build_features(tx)
    graph = build_graph_features(tx, HUBS)
    batch = pd.concat([batch, graph[feature_columns(graph)]], axis=1)
    nodes = pd.DataFrame({"node_id": [f"1_{i}" for i in range(40)], "node_idx": np.arange(40)})
    store = FeatureStore.from_frames(tx, nodes, HUBS, feature_columns(batch), {"US Dollar": 1.0}, FORMATS)
    return tx, batch, store


def as_raw(tx: pd.DataFrame) -> pd.DataFrame:
    names = {v: k for k, v in FORMATS.items()}
    return pd.DataFrame({
        "timestamp": tx["timestamp"].to_numpy(),
        "from_bank": "001", "from_account": tx["from_node"].str.split("_").str[1].to_numpy(),
        "to_bank": "1", "to_account": tx["to_node"].str.split("_").str[1].to_numpy(),
        "amount_paid": tx["amount_paid_usd"].to_numpy(), "payment_currency": "US Dollar",
        "amount_received": tx["amount_received_usd"].to_numpy(), "receiving_currency": "US Dollar",
        "payment_format": tx["payment_format_code"].map(names).to_numpy(),
    })


def test_store_reproduces_batch_features(history):
    tx, batch, store = history
    rows = np.random.default_rng(0).choice(len(tx), 300, replace=False)
    raw = as_raw(tx.iloc[rows])
    raw["to_bank"] = raw["from_bank"]                     # same bank -> is_cross_bank as in the fixture
    X = store.transform(store.prepare(raw))
    expected = batch.iloc[rows][store.features].reset_index(drop=True)
    pd.testing.assert_frame_equal(X.reset_index(drop=True), expected, check_dtype=False)


def test_unknown_accounts_have_no_history(history):
    tx, _, store = history
    raw = as_raw(tx.iloc[[-1]])
    raw["from_account"], raw["to_account"] = "NEW1", "NEW2"
    raw["timestamp"] = tx["timestamp"].max() + pd.Timedelta(hours=1)
    clean = store.prepare(raw)
    assert not clean["known_sender"].iloc[0] and not clean["known_receiver"].iloc[0]
    X = store.transform(clean)
    assert X["snd_out_cnt_all"].iloc[0] == 0 and X["pair_prior_cnt"].iloc[0] == 0
    assert X["is_self_loop"].iloc[0] == 0                 # two different new accounts


def test_prepare_rejects_bad_input(history):
    tx, _, store = history
    raw = as_raw(tx.iloc[[0]])
    with pytest.raises(ValueError, match="currency"):
        store.prepare(raw.assign(payment_currency="Doubloon"))
    with pytest.raises(ValueError, match="payment_format"):
        store.prepare(raw.assign(payment_format="Carrier pigeon"))
    with pytest.raises(ValueError, match="rebuilt"):
        store.prepare(raw.assign(timestamp=store.horizon_end + pd.Timedelta(days=1)))


def test_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.01
    assert psi(ref, rng.normal(1.0, 1, size=20_000)) > 0.2
    assert psi(ref, np.full(100, np.nan)) > 0.2           # everything missing is a shift


def test_drift_monitor():
    rng = np.random.default_rng(1)
    ref = rng.random(50_000)
    m = DriftMonitor(ref, threshold=0.99, budget=0.01, min_count=1_000)
    assert m.report()["status"] == "warming_up"
    m.update(rng.random(5_000))
    assert m.report()["status"] == "ok"
    m.update(rng.random(5_000) ** 0.2)                    # scores shift upwards
    r = m.report()
    assert r["status"] == "drift" and r["alert_rate_ratio"] > 1.5


@pytest.fixture(scope="module")
def client(history):
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aml.api import create_app

    tx, batch, store = history
    X = batch[store.features].astype("float32")
    booster = xgb.train({"max_depth": 2, "objective": "binary:logistic"},
                        xgb.DMatrix(X, label=batch["is_laundering"], feature_names=list(X.columns)), 10)
    scorer = Scorer(store, booster, threshold=0.5, budget=0.01)
    with TestClient(create_app(scorer, reference=np.random.default_rng(0).random(5_000))) as c:
        yield c, tx


def payload(tx: pd.DataFrame) -> dict:
    raw = as_raw(tx)
    raw["timestamp"] = raw["timestamp"].astype(str)
    return {"payments": raw.to_dict(orient="records")}


def test_api_health_and_score(client):
    c, tx = client
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/score", json=payload(tx.tail(5)))
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 5 and all(0 <= x["score"] <= 1 and len(x["top_features"]) == 3 for x in results)
    assert c.get("/monitoring").json()["window"] >= 5


def test_api_rejects_bad_payments(client):
    c, tx = client
    body = payload(tx.tail(1))
    body["payments"][0]["payment_currency"] = "Doubloon"
    assert c.post("/score", json=body).status_code == 422
    body = payload(tx.tail(1))
    body["payments"][0]["amount_paid"] = -5
    assert c.post("/score", json=body).status_code == 422
    assert c.post("/score", json={"payments": []}).status_code == 422
