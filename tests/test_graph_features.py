"""Tests for step 4 graph features: correctness against brute force, and no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.graph_features import build_graph_features, find_hubs, pagerank
from test_features import make_clean_tx


def dense_tx(n: int = 900, n_nodes: int = 12, seed: int = 3) -> pd.DataFrame:
    """Few nodes and many payments, so cycles and multi-hop paths are common."""
    return make_clean_tx(n=n, n_nodes=n_nodes, seed=seed)


def brute_force(tx: pd.DataFrame, i: int, hubs: set[int]) -> dict:
    row = tx.iloc[i]
    s, d, t = row["from_idx"], row["to_idx"], row["timestamp"]
    past = tx[(tx["is_self_loop"] == 0) & (tx["timestamp"] < t)]
    first = past.groupby(["from_idx", "to_idx"])["timestamp"].min().to_dict()
    out = {}
    succ: dict[int, list[int]] = {}
    for a, b in first:
        succ.setdefault(a, []).append(b)

    def paths(a, b, hops):
        """Formation times of simple paths a -> ... -> b with `hops` edges avoiding hub intermediaries."""
        found = []

        def walk(node, visited, latest, left):
            for nxt in succ.get(node, []):
                tt = max(latest, first[(node, nxt)])
                if left == 1:
                    if nxt == b:
                        found.append(tt)
                elif nxt not in visited and nxt not in hubs and nxt != b:
                    walk(nxt, visited | {nxt}, tt, left - 1)
        walk(a, {a}, pd.Timestamp.min, hops)
        return found

    for name, (a, b, hops) in {"cyc3": (d, s, 2), "cyc4": (d, s, 3), "fwd2": (s, d, 2), "fwd3": (s, d, 3)}.items():
        p = paths(a, b, hops) if s != d else []        # self-transfers have no paths
        out[f"{name}_cnt"] = len(p)
        if name != "fwd3":
            out[f"{name}_mins_since_newest"] = (t - max(p)) / pd.Timedelta(minutes=1) if p else np.nan

    def new_cp(node, col, cp_col, minutes):
        firsts = past.groupby([col, cp_col])["timestamp"].min().reset_index()
        firsts = firsts[firsts[col] == node]
        return int((firsts["timestamp"] > t - pd.Timedelta(minutes=minutes)).sum())

    out["snd_out_new_cp_1h"] = new_cp(s, "from_idx", "to_idx", 60)
    out["snd_out_new_cp_24h"] = new_cp(s, "from_idx", "to_idx", 1440)
    out["rcv_in_new_cp_24h"] = new_cp(d, "to_idx", "from_idx", 1440)
    out["snd_in_new_cp_24h"] = new_cp(s, "to_idx", "from_idx", 1440)
    return out


@pytest.fixture(scope="module")
def built():
    tx = dense_tx()
    hubs = np.array([0])                      # treat node 0 as a hub
    return tx, hubs, build_graph_features(tx, hubs)


def test_matches_brute_force(built):
    tx, hubs, g = built
    rng = np.random.default_rng(0)
    rows = rng.choice(len(tx), 60, replace=False)
    assert g["cyc3_cnt"].gt(0).any() and g["cyc4_cnt"].gt(0).any(), "fixture should contain cycles"
    for i in rows:
        expected = brute_force(tx, i, set(hubs.tolist()))
        for k, v in expected.items():
            got = g.iloc[i][k]
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(got), (i, k)
            else:
                assert got == pytest.approx(v), (i, k, got, v)


def test_same_component_uses_previous_days_only(built):
    tx, hubs, g = built
    day = tx["timestamp"].dt.normalize()
    i = int(np.flatnonzero((day > day.min()).to_numpy())[5])
    row = tx.iloc[i]
    past = tx[(tx["is_self_loop"] == 0) & (tx["timestamp"] < day.iloc[i])
              & ~tx["from_idx"].isin(hubs) & ~tx["to_idx"].isin(hubs)]
    # union-find over edges seen before midnight
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b in zip(past["from_idx"], past["to_idx"]):
        parent[find(a)] = find(b)
    s, d = row["from_idx"], row["to_idx"]
    expected = s in parent and d in parent and find(s) == find(d)
    assert g.iloc[i]["same_component"] == int(expected)


def test_no_leakage_from_future_rows(built):
    tx, hubs, g = built
    cut = tx["timestamp"].iloc[len(tx) // 2]
    early = tx[tx["timestamp"] < cut]
    g_early = build_graph_features(early, hubs)
    cols = [c for c in g.columns if c not in ("timestamp", "split")]
    pd.testing.assert_frame_equal(g_early[cols], g.loc[early.index, cols])


def test_find_hubs_uses_train_only():
    tx = make_clean_tx(n=600, n_nodes=200, seed=4)
    train = tx[tx["split"] == "train"]
    # a fan-out to 50 accounts in the test split must not make node 0 a hub
    later = tx[tx["split"] == "test"].head(50).assign(from_idx=0, to_idx=np.arange(300, 350), is_self_loop=0)
    hubs = find_hubs(pd.concat([tx, later]), min_degree=10)
    assert 0 not in set(find_hubs(train, min_degree=10)) or 0 in hubs
    np.testing.assert_array_equal(hubs, find_hubs(train, min_degree=10))


def test_pagerank_symmetric_cycle():
    r = pagerank(np.array([0, 1, 2]), np.array([1, 2, 0]), 3)
    np.testing.assert_allclose(r, 1 / 3)
    star = pagerank(np.array([1, 2, 3]), np.array([0, 0, 0]), 4)
    assert star[0] > star[1] == pytest.approx(star[2])
