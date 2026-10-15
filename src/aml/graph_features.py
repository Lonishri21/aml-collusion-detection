"""Step 4: point-in-time graph features for every transaction.

Same rule as step 2: a feature for a transaction at minute t only uses transactions strictly
before minute t, and labels are never used. Step 2 described each account on its own and the
direct relationship between sender and receiver; these features describe the network around them.

The graph has one edge per distinct (sender, receiver) pair, which exists from the minute of
their first payment. Self-transfers are not edges.

- **Paths** between the two accounts of a transaction, counted exactly. For a payment S -> R:
  `cyc3` counts routes R -> w -> S, so this payment closes a 3-cycle; `cyc4` counts
  R -> w1 -> w2 -> S; `fwd2` / `fwd3` count routes S -> w -> R and S -> w1 -> w2 -> R, so money
  from S already reached R through intermediaries (scatter-gather). A path exists from the
  minute its last edge appears. Hubs (accounts with very many counterparties in the train
  period) are not used as intermediaries: almost any two accounts connect through a hub, so
  such paths carry no information.
- **Fan**: new counterparties in the last hour / 24 hours (fan-out of the sender, fan-in of the
  receiver), which step 2 only had as all-time totals.
- **Daily snapshots** of the graph as of midnight before the transaction's day: PageRank,
  connected-component size and whether sender and receiver were already connected. A
  same-day edge is not visible to these three, which only makes them more conservative.

The graph is sparse (HI-Small: ~650k distinct pairs, ~1M 2-hop and ~1.5M 3-hop paths without
hubs), so every path is enumerated once and each transaction's counts are binary searches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .features import EventHistory, _minutes, day_index, first_midnight
from .logging_utils import get_logger

log = get_logger(__name__)

INPUT_COLUMNS = ["tx_id", "timestamp", "from_idx", "to_idx", "amount_paid_usd", "amount_received_usd",
                 "is_self_loop", "is_laundering", "split"]

FAN_WINDOWS = {"1h": 60, "24h": 24 * 60}


def find_hubs(tx: pd.DataFrame, min_degree: int, split: str = "train") -> np.ndarray:
    """Accounts with at least `min_degree` distinct counterparties (in or out) within `split`."""
    e = tx[(tx["split"] == split) & (tx["is_self_loop"] == 0)][["from_idx", "to_idx"]].drop_duplicates()
    out_deg = e.groupby("from_idx").size()
    in_deg = e.groupby("to_idx").size()
    hubs = np.union1d(out_deg.index[out_deg >= min_degree], in_deg.index[in_deg >= min_degree])
    return hubs.astype(np.int64)


def first_edges(src: np.ndarray, dst: np.ndarray, minute: np.ndarray) -> pd.DataFrame:
    """One row per distinct directed pair with the minute of its first payment (`f`)."""
    e = pd.DataFrame({"a": src, "b": dst, "f": minute})
    return e.groupby(["a", "b"], sort=False)["f"].min().reset_index()


def enumerate_paths(edges: pd.DataFrame, hubs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All simple 2-hop and 3-hop paths that avoid hubs as intermediaries.

    Returns frames with endpoints `a`, `b` and `t`, the minute the path exists from (its
    latest edge). One row per distinct route, so counts are numbers of distinct intermediaries.
    """
    via = edges[~edges["a"].isin(hubs)]                          # edges leaving a usable intermediary
    p2 = edges.rename(columns={"b": "w", "f": "f1"}).merge(
        via.rename(columns={"a": "w", "f": "f2"}), on="w")
    p2 = p2[p2["a"] != p2["b"]]
    p2 = p2.assign(t=np.maximum(p2["f1"], p2["f2"]))[["a", "w", "b", "t"]]

    p3 = p2[~p2["b"].isin(hubs)].rename(columns={"b": "x", "t": "t12"}).merge(
        edges.rename(columns={"a": "x", "f": "f3"}), on="x")
    p3 = p3[(p3["b"] != p3["a"]) & (p3["b"] != p3["w"])]
    p3 = p3.assign(t=np.maximum(p3["t12"], p3["f3"]))[["a", "b", "t"]]
    return p2[["a", "b", "t"]], p3


@dataclass
class PathIndex:
    """Paths sorted by (a, b, minute) to count those between two accounts before a minute."""

    keys: np.ndarray
    t: np.ndarray
    n_nodes: int
    T: int

    @classmethod
    def build(cls, paths: pd.DataFrame, n_nodes: int, T: int) -> "PathIndex":
        if float(n_nodes) ** 2 * T >= 2 ** 62:
            raise OverflowError("Path key would overflow int64; too many nodes for this encoding")
        keys = (paths["a"].to_numpy(np.int64) * n_nodes + paths["b"].to_numpy(np.int64)) * T \
            + paths["t"].to_numpy(np.int64)
        keys = np.sort(keys)
        return cls(keys, keys % T, n_nodes, T)

    def query(self, a: np.ndarray, b: np.ndarray, minute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(number of a -> b paths existing before `minute`, minutes since the newest one formed)."""
        base = (a.astype(np.int64) * self.n_nodes + b) * self.T
        lo = np.searchsorted(self.keys, base, "left")
        hi = np.searchsorted(self.keys, base + minute, "left")
        cnt = hi - lo
        if self.keys.size == 0:
            return cnt, np.full(len(a), np.nan)
        newest = self.t[np.clip(hi - 1, 0, None)]
        return cnt, np.where(cnt > 0, minute - newest, np.nan)


def pagerank(rows: np.ndarray, cols: np.ndarray, n: int, damping: float = 0.85,
             tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """PageRank of an unweighted directed graph on nodes 0..n-1 (dangling mass spread uniformly)."""
    out_deg = np.bincount(rows, minlength=n).astype(np.float64)
    # Column-stochastic transition: M[j, i] = 1 / out_deg(i) for each edge i -> j.
    M = sparse.csr_matrix((1.0 / out_deg[rows], (cols, rows)), shape=(n, n))
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = damping * (M @ r + r[dangling].sum() / n) + (1 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def snapshot_features(edges: pd.DataFrame, hubs: np.ndarray, cut: int, n_nodes: int) -> dict[str, np.ndarray]:
    """Node-level features of the graph made of edges first seen before minute `cut`.

    PageRank is scaled so the average active account scores 1. Components ignore hubs, which
    would otherwise join nearly every account into one component. Accounts with no edge yet
    get NaN.
    """
    e = edges[edges["f"] < cut]
    a, b = e["a"].to_numpy(np.int64), e["b"].to_numpy(np.int64)
    active = np.zeros(n_nodes, bool)
    active[a] = active[b] = True
    idx = np.flatnonzero(active)
    local = np.full(n_nodes, -1, np.int64)
    local[idx] = np.arange(len(idx))

    pr = np.full(n_nodes, np.nan)
    if len(idx):
        pr[idx] = pagerank(local[a], local[b], len(idx)) * len(idx)

    keep = ~(np.isin(a, hubs) | np.isin(b, hubs))
    g = sparse.coo_matrix((np.ones(keep.sum()), (local[a[keep]], local[b[keep]])), shape=(len(idx), len(idx)))
    _, labels = connected_components(g, directed=True, connection="weak")
    size = np.bincount(labels)
    comp = np.full(n_nodes, -1, np.int64)
    comp[idx] = labels
    comp_size = np.full(n_nodes, np.nan)
    comp_size[idx] = size[labels]
    hub_mask = np.zeros(n_nodes, bool)
    hub_mask[hubs] = True
    comp[hub_mask] = -1                    # hubs are left out of components
    comp_size[hub_mask] = np.nan
    return {"pagerank": pr, "component": comp, "component_size": comp_size}


@dataclass
class GraphHistories:
    """Everything step 4 needs to describe a payment from the graph so far: built once, queried
    for any (sender, receiver, minute). Used by `build_graph_features` and by the scoring service.

    Daily snapshot features are computed on first use for each day and cached.
    """

    edges: pd.DataFrame
    paths2: PathIndex
    paths3: PathIndex
    hubs: np.ndarray
    hub_mask: np.ndarray
    n_nodes: int
    day0_minute: int                       # minute (from the history origin) of day 0's midnight
    _snapshots: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, src: np.ndarray, dst: np.ndarray, minute: np.ndarray, self_loop: np.ndarray,
              hubs: np.ndarray, n_nodes: int, T: int, day0_minute: int) -> "GraphHistories":
        e = ~self_loop
        edges = first_edges(src[e], dst[e], minute[e])
        p2, p3 = enumerate_paths(edges, hubs)
        log.info("%s distinct edges, %s 2-hop and %s 3-hop paths (%d hubs excluded as intermediaries)",
                 f"{len(edges):,}", f"{len(p2):,}", f"{len(p3):,}", len(hubs))
        hub_mask = np.zeros(n_nodes, bool)
        hub_mask[hubs] = True
        return cls(edges, PathIndex.build(p2, n_nodes, T), PathIndex.build(p3, n_nodes, T),
                   np.asarray(hubs), hub_mask, n_nodes, int(day0_minute))

    def snapshot(self, day: int) -> dict[str, np.ndarray]:
        """Node features of the graph as of midnight starting `day` (0 = first day of history)."""
        if day not in self._snapshots:
            self._snapshots[day] = snapshot_features(self.edges, self.hubs, self.day0_minute + day * 24 * 60,
                                                     self.n_nodes)
        return self._snapshots[day]

    def query(self, s: np.ndarray, d: np.ndarray, t: np.ndarray, day: np.ndarray,
              out_hist: EventHistory, in_hist: EventHistory) -> pd.DataFrame:
        """Step 4 features of payments s -> d at minutes t on calendar days `day`.

        `out_hist` / `in_hist` are step 2's payment histories (outgoing by sender, incoming by
        receiver), shared rather than rebuilt.
        """
        f: dict[str, np.ndarray] = {}
        f["cyc3_cnt"], f["cyc3_mins_since_newest"] = self.paths2.query(d, s, t)
        f["cyc4_cnt"], f["cyc4_mins_since_newest"] = self.paths3.query(d, s, t)
        f["fwd2_cnt"], f["fwd2_mins_since_newest"] = self.paths2.query(s, d, t)
        f["fwd3_cnt"], _ = self.paths3.query(s, d, t)
        f["snd_is_hub"], f["rcv_is_hub"] = self.hub_mask[s], self.hub_mask[d]

        for name, w in FAN_WINDOWS.items():
            f[f"snd_out_new_cp_{name}"] = out_hist.new_counterparties(s, t, w)   # fan-out
            f[f"rcv_in_new_cp_{name}"] = in_hist.new_counterparties(d, t, w)     # fan-in
        f["snd_in_new_cp_24h"] = in_hist.new_counterparties(s, t, FAN_WINDOWS["24h"])
        f["rcv_out_new_cp_24h"] = out_hist.new_counterparties(d, t, FAN_WINDOWS["24h"])

        snap = {k: np.full(len(s), np.nan) for k in
                ["snd_pagerank", "rcv_pagerank", "snd_component_size", "rcv_component_size", "same_component"]}
        for k in np.unique(day):
            rows = day == k
            g = self.snapshot(int(k))
            snap["snd_pagerank"][rows], snap["rcv_pagerank"][rows] = g["pagerank"][s[rows]], g["pagerank"][d[rows]]
            snap["snd_component_size"][rows] = g["component_size"][s[rows]]
            snap["rcv_component_size"][rows] = g["component_size"][d[rows]]
            cs, cd = g["component"][s[rows]], g["component"][d[rows]]
            snap["same_component"][rows] = (cs == cd) & (cs >= 0)
        f |= snap

        for k, v in f.items():
            if k.endswith("_cnt") or "_new_cp_" in k:
                f[k] = v.astype(np.int32)
            elif k.startswith(("snd_is", "rcv_is", "same_")):
                f[k] = v.astype(np.int8)
            else:
                f[k] = v.astype(np.float32)
        return pd.DataFrame(f)


def build_graph_features(tx: pd.DataFrame, hubs: np.ndarray) -> pd.DataFrame:
    """Return one row per transaction: `tx_id`, `split`, `is_laundering` and all graph features.

    `tx` is the cleaned output of step 1; `hubs` comes from `find_hubs` on the train split, so
    it never depends on validation or test data. Rows keep the input order.
    """
    origin = tx["timestamp"].min()
    minute = _minutes(tx["timestamp"], origin)
    T = int(minute.max()) + 1
    src = tx["from_idx"].to_numpy(np.int64)
    dst = tx["to_idx"].to_numpy(np.int64)
    n_nodes = int(max(src.max(), dst.max())) + 1
    self_loop = tx["is_self_loop"].to_numpy(bool)
    e = ~self_loop
    day = day_index(tx["timestamp"], origin)
    graph = GraphHistories.build(src, dst, minute, self_loop, hubs, n_nodes, T, first_midnight(origin))

    paid = tx["amount_paid_usd"].to_numpy(np.float64)
    received = tx["amount_received_usd"].to_numpy(np.float64)
    out_hist = EventHistory.build(src[e], minute[e], dst[e], paid[e], T)
    in_hist = EventHistory.build(dst[e], minute[e], src[e], received[e], T)
    out = graph.query(src, dst, minute, day, out_hist, in_hist)
    out.index = tx.index

    meta = tx[["tx_id", "timestamp", "split", "is_laundering"]]
    log.info("Built %d graph features for %s transactions", out.shape[1], f"{len(out):,}")
    return pd.concat([meta, out], axis=1)
