"""Tests for step 6: label-free scores, communities from earlier days only, and group recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("igraph")

from aml.unsupervised import (combine, community_scores, community_table, group_recovery,  # noqa: E402
                              label_free_threshold, leiden_partition, percentile_of, signed_log)
from test_features import make_clean_tx  # noqa: E402


def two_cliques():
    """Two dense groups of 5 accounts joined by a single payment."""
    pairs = [(a, b) for g in (range(5), range(5, 10)) for a in g for b in g if a != b]
    a, b = np.array(pairs + [(4, 5)]).T
    return a, b


def test_leiden_finds_the_two_groups():
    a, b = two_cliques()
    member = leiden_partition(a, b, n_nodes=12)
    assert len(set(member[:5])) == 1 and len(set(member[5:10])) == 1
    assert member[0] != member[5]
    assert (member[10:] == -1).all()                  # accounts with no payment


def test_community_table():
    a, b = two_cliques()
    member = leiden_partition(a, b, n_nodes=10)
    minute = np.arange(len(a))
    t = community_table(member, a, b, minute, first_seen=np.zeros(10, np.int64), cut=len(a))
    assert len(t) == 2
    np.testing.assert_allclose(t["density"], 1.0)    # complete directed graph inside each group
    np.testing.assert_allclose(t["reciprocity"], 1.0)
    assert (t["internal_share"] < 1).all()           # the bridge payment is external


def test_community_scores_use_only_earlier_days():
    tx = make_clean_tx(n=1200, n_nodes=80, seed=7)
    full = community_scores(tx, hubs=np.array([], dtype=np.int64))
    day0 = tx["timestamp"].min().normalize()
    early = tx[tx["timestamp"] < day0 + pd.Timedelta(days=2)]
    part = community_scores(early, hubs=np.array([], dtype=np.int64))
    rows = early.index
    np.testing.assert_array_equal(part.cross_community, full.cross_community[rows])
    assert (full.cross_community[(tx["timestamp"] < day0 + pd.Timedelta(days=1)).to_numpy()] == 1).all()


def test_percentiles_and_combination():
    ref = np.arange(100.0)
    np.testing.assert_allclose(percentile_of(ref, np.array([-1, 0, 50, 1000])), [0, 0, 0.5, 1])
    s = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([30.0, 20.0, 10.0])}
    np.testing.assert_allclose(combine(s, s), [1 / 3, 1 / 3, 1 / 3])
    assert label_free_threshold(ref, 0.1) == pytest.approx(89.1)


def test_signed_log_puts_missing_below_observed():
    X = pd.DataFrame({"x": [0.0, 10.0, np.nan, -3.0], "empty": [np.nan] * 4})
    v = signed_log(X)
    assert v[2, 0] < v[:, 0][~np.isnan(X["x"].to_numpy())].min()
    assert (v[:, 1] == 0).all()


def test_group_recovery():
    a, b = two_cliques()
    member = leiden_partition(a, b, n_nodes=10)
    ts = pd.Timestamp("2022-09-01 10:00")
    attempts = pd.DataFrame({
        "attempt_id": [1, 1, 2, 2], "pattern_type": ["CYCLE", "CYCLE", "FAN-OUT", "FAN-OUT"],
        "timestamp": [ts] * 4, "from_idx": [0, 1, 0, 0], "to_idx": [1, 2, 5, 6],
    })
    out = group_recovery(attempts, [member], member, ts.normalize())
    assert out.loc["CYCLE", "share_recovered"] == 1.0      # 1 and 2 are in 0's group
    assert out.loc["FAN-OUT", "share_recovered"] == 0.0    # 5 and 6 are in the other group
    assert out.loc["all", "attempts"] == 2
