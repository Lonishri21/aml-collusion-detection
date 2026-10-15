"""Tests for step 5: snapshots use only earlier days, scaling, and a small end-to-end training run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml.config import GNNConfig  # noqa: E402
from aml.gnn import (CATEGORICAL, EdgeSet, TabularScaler, build_snapshots, combine_scores,  # noqa: E402
                     predict, train_gnn)
from aml.features import day_index  # noqa: E402
from test_features import make_clean_tx  # noqa: E402


@pytest.fixture(scope="module")
def tx():
    return make_clean_tx(n=1500, n_nodes=60, seed=5)


def test_snapshots_only_use_earlier_days(tx):
    hubs = np.array([], dtype=np.int64)
    snaps = build_snapshots(tx, hubs, n_days=4)
    day0 = tx["timestamp"].min().normalize()
    # an edge in snapshot k must come from a payment before midnight starting day k
    for k, snap in enumerate(snaps):
        past = tx[(tx["timestamp"] < day0 + pd.Timedelta(days=k)) & (tx["is_self_loop"] == 0)]
        expected = set(zip(past["from_idx"], past["to_idx"]))
        got = set(map(tuple, snap.edge_index.T.tolist()))
        assert got == expected
    assert snaps[0].edge_index.shape[1] == 0


def test_snapshots_ignore_future_rows(tx):
    hubs = np.array([], dtype=np.int64)
    full = build_snapshots(tx, hubs, n_days=4)
    day0 = tx["timestamp"].min().normalize()
    early = tx[tx["timestamp"] < day0 + pd.Timedelta(days=2)]
    part = build_snapshots(early, hubs, n_days=4)
    for k in (0, 1, 2):
        assert torch.equal(full[k].edge_index, part[k].edge_index)
        # same raw facts; features are standardised with different statistics, so compare ranks
        a, b = full[k].x.numpy(), part[k].x.numpy()
        for j in range(a.shape[1]):
            assert np.array_equal(np.argsort(a[:, j], kind="stable"), np.argsort(b[:, j], kind="stable"))


def test_tabular_scaler():
    X = pd.DataFrame({"a": [0.0, 10.0, np.nan, -5.0], "payment_format_code": [0, 1, 1, 2],
                      "snd_entity_type": [0, 0, 1, -1], "rcv_entity_type": [1, 1, 1, 1]})
    s = TabularScaler().fit(X)
    Z = s.transform(X)
    assert Z.shape == (4, 1 + 3 + 3 + 1)
    assert Z[2, 0] == 0.0                       # missing -> the mean
    assert set(s.categories) == set(CATEGORICAL)
    np.testing.assert_array_equal(Z[:, 1:4].sum(1), 1)


def test_combine_scores_is_monotone():
    p = np.array([0.01, 0.1, 0.5, 0.9])
    c = combine_scores(p, p)
    np.testing.assert_allclose(c, p, rtol=1e-6)
    assert (np.diff(combine_scores(p, np.full(4, 0.3))) > 0).all()


def test_train_and_predict_small(tx):
    hubs = np.array([], dtype=np.int64)
    snaps = build_snapshots(tx, hubs, n_days=4)
    day0 = tx["timestamp"].min().normalize()
    rng = np.random.default_rng(0)
    signal = rng.normal(size=len(tx)).astype(np.float32)
    y = (signal + rng.normal(0, 0.5, len(tx)) > 1.6).astype(np.int8)

    def edges(rows):
        return EdgeSet(src=torch.from_numpy(tx["from_idx"].to_numpy(np.int64)[rows]),
                       dst=torch.from_numpy(tx["to_idx"].to_numpy(np.int64)[rows]),
                       tab=torch.from_numpy(signal[rows, None]),
                       day=day_index(tx["timestamp"], day0)[rows], y=y[rows])

    rows = np.arange(len(tx))
    train, val = edges(rows[:1000]), edges(rows[1000:])
    cfg = GNNConfig(hidden=8, chunk_rows=256, max_epochs=8, patience=8, lr=1e-2)
    model, history = train_gnn(snaps, train, val, cfg)
    assert len(history) == 8
    p = predict(model, snaps, val)
    assert p.shape == (len(val.y),) and np.isfinite(p).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(val.y, p) > 0.8          # learns the tabular signal


def test_neighbourhood_embeddings_match_full_graph(tx):
    from aml.gnn import EdgeGNN
    snaps = build_snapshots(tx, np.array([3]), n_days=4)
    snap = snaps[3]
    assert not (snap.edge_index == 3).any()          # the hub passes no messages
    torch.manual_seed(0)
    model = EdgeGNN(snap.x.shape[1], 1, hidden=8).eval()
    nodes = torch.tensor([0, 5, 17, 3, 5])
    full = model.embed(snap)[nodes]
    h, pos = model.embed_nodes(snap, nodes)
    torch.testing.assert_close(h[pos], full)
