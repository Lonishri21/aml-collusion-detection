"""Step 5: a graph neural network (GraphSAGE) that scores transactions.

**No leakage, same rule as before.** A transaction on calendar day k is scored with a graph made
only of payments before midnight starting day k (a daily snapshot), plus the step 2 and step 4
features of the transaction itself, which already cover the same day up to the minute before it.
Labels are only used as training targets, never as inputs.

**Model.** Each account starts from a few facts about its history (counts, amounts, PageRank,
component size, ...). Two GraphSAGE layers pass messages along payments in **both directions**
(from payers and from payees, with separate weights, since direction matters in money flows), so
an account's embedding summarises its 2-hop neighbourhood. A small network then scores each
payment from the sender's and receiver's embeddings and the transaction's tabular features.
Hub accounts (step 4) keep their own features but pass no messages: an average over a hub's
thousands of counterparties carries no information and would make every neighbourhood huge.

**Ablation.** The same model with message passing switched off (`message_passing=False`) sees
the same inputs except the neighbourhood, so the gap between the two is what the GNN itself adds.

**Training in mini-batches.** Each step takes a batch of one day's transactions and runs message
passing only on the accounts within 2 hops of their senders and receivers. That gives exactly the
same embeddings for them as the full graph at a fraction of the cost (GraphSAGE's mini-batch
scheme, without sampling). Prediction embeds the whole snapshot once per day.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .config import GNNConfig
from .features import _minutes, day_index, first_midnight
from .graph_features import first_edges, snapshot_features
from .logging_utils import get_logger
from .metrics import ranking_metrics

log = get_logger(__name__)

NODE_FEATURES = ["log_out_cnt", "log_in_cnt", "log_out_distinct", "log_in_distinct", "log_out_usd",
                 "log_in_usd", "log_self_cnt", "log_mins_since_active", "log_pagerank",
                 "log_component_size", "is_hub", "is_active"]
# Tabular features treated as categories (one-hot) rather than numbers.
CATEGORICAL = ["payment_format_code", "snd_entity_type", "rcv_entity_type"]


@dataclass
class Snapshot:
    x: torch.Tensor            # [n_nodes, len(NODE_FEATURES)]
    edge_index: torch.Tensor   # [2, n_edges], sender -> receiver, edges first seen before the cut


def build_snapshots(tx: pd.DataFrame, hubs: np.ndarray, n_days: int) -> list[Snapshot]:
    """One graph per calendar day k, made of payments strictly before midnight starting day k.

    `tx` needs timestamp, from_idx, to_idx, amount_paid_usd and is_self_loop (all splits; only
    rows before each cut are used). Node features are standardised with statistics of the last
    snapshot, so all snapshots share one scale.
    """
    origin = tx["timestamp"].min()
    minute = _minutes(tx["timestamp"], origin)
    offset = first_midnight(origin)
    src, dst = tx["from_idx"].to_numpy(np.int64), tx["to_idx"].to_numpy(np.int64)
    n = int(max(src.max(), dst.max())) + 1
    usd = tx["amount_paid_usd"].to_numpy(np.float64)
    self_loop = tx["is_self_loop"].to_numpy(bool)
    e = ~self_loop
    edges = first_edges(src[e], dst[e], minute[e])
    hub = np.zeros(n, bool)
    hub[hubs] = True

    raw = []
    for k in range(n_days):
        cut = int(offset + k * 24 * 60)
        past = minute < cut
        pe, ps = past & e, past & self_loop
        ed = edges[edges["f"] < cut]
        snap = snapshot_features(edges, hubs, cut, n)
        both = pd.Series(np.concatenate([minute[past], minute[past]]))
        latest = both.groupby(np.concatenate([src[past], dst[past]])).max()
        last = np.full(n, -np.inf)
        last[latest.index.to_numpy()] = latest.to_numpy()
        active = np.isfinite(last)
        f = np.column_stack([
            np.log1p(np.bincount(src[pe], minlength=n)),
            np.log1p(np.bincount(dst[pe], minlength=n)),
            np.log1p(np.bincount(ed["a"], minlength=n)),
            np.log1p(np.bincount(ed["b"], minlength=n)),
            np.log1p(np.bincount(src[pe], weights=usd[pe], minlength=n)),
            np.log1p(np.bincount(dst[pe], weights=usd[pe], minlength=n)),
            np.log1p(np.bincount(src[ps], minlength=n)),
            np.where(active, np.log1p(np.clip(cut - last, 0, None)), 0.0),
            np.log(np.nan_to_num(snap["pagerank"], nan=1.0)),
            np.log(np.nan_to_num(snap["component_size"], nan=1.0)),
            hub.astype(float),
            active.astype(float),
        ])
        ed = ed[~(hub[ed["a"].to_numpy()] | hub[ed["b"].to_numpy()])]     # hubs pass no messages
        ei = torch.from_numpy(np.vstack([ed["a"].to_numpy(np.int64), ed["b"].to_numpy(np.int64)]))
        raw.append((f, ei))
        log.info("snapshot day %d: %s edges, %s active accounts", k, f"{ei.shape[1]:,}", f"{active.sum():,}")

    ref = raw[-1][0][raw[-1][0][:, -1] == 1]       # active accounts of the last snapshot
    mean, std = ref.mean(0), ref.std(0) + 1e-6
    mean[-2:], std[-2:] = 0, 1                      # leave the two flags as 0/1
    return [Snapshot(torch.from_numpy(((f - mean) / std).astype(np.float32)), ei) for f, ei in raw]


class TabularScaler:
    """Signed log, standardise (train statistics), missing -> 0 (the mean), one-hot categories."""

    def fit(self, X: pd.DataFrame) -> "TabularScaler":
        self.numeric = [c for c in X.columns if c not in CATEGORICAL]
        v = self._log(X[self.numeric].to_numpy(np.float64))
        self.mean, self.std = np.nanmean(v, 0), np.nanstd(v, 0) + 1e-6
        self.categories = {c: np.unique(X[c].dropna().to_numpy()) for c in CATEGORICAL if c in X}
        return self

    @staticmethod
    def _log(v: np.ndarray) -> np.ndarray:
        return np.sign(v) * np.log1p(np.abs(v))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        z = (self._log(X[self.numeric].to_numpy(np.float64)) - self.mean) / self.std
        parts = [np.clip(np.nan_to_num(z, nan=0.0), -10, 10)]
        for c, cats in self.categories.items():
            parts.append((X[c].to_numpy()[:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(parts).astype(np.float32)


class EdgeGNN(nn.Module):
    def __init__(self, node_dim: int, tab_dim: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, message_passing: bool = True):
        super().__init__()
        self.encode = nn.Linear(node_dim, hidden)
        n_conv = layers if message_passing else 0
        self.from_payers = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(n_conv)])
        self.from_payees = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(n_conv)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_conv)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + tab_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def embed(self, snap: Snapshot) -> torch.Tensor:
        h = torch.relu(self.encode(snap.x))
        reverse = snap.edge_index.flip(0)
        for a, b, norm in zip(self.from_payers, self.from_payees, self.norms):
            h = norm(h + self.dropout(torch.relu(a(h, snap.edge_index) + b(h, reverse))))
        return h

    def embed_nodes(self, snap: Snapshot, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeddings of `nodes` computed on their neighbourhood only (equal to `embed(snap)[nodes]`).

        Returns (embeddings of the subgraph, position of each of `nodes` in it).
        """
        sub, pos = neighbourhood(snap, nodes, len(self.from_payers))
        return self.embed(sub), pos

    def score(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([h[src], h[dst], tab], dim=1)).squeeze(1)


def neighbourhood(snap: Snapshot, nodes: torch.Tensor, hops: int) -> tuple[Snapshot, torch.Tensor]:
    """Subgraph of all accounts within `hops` of `nodes` (either direction), relabelled 0..m-1.

    Keeping every edge between those accounts makes the embeddings of `nodes` after `hops`
    message-passing layers identical to the full-graph ones.
    """
    n = snap.x.shape[0]
    keep = torch.zeros(n, dtype=torch.bool)
    keep[nodes] = True
    a, b = snap.edge_index
    for _ in range(hops):
        touch = keep[a] | keep[b]
        keep[a[touch]] = True
        keep[b[touch]] = True
    subset = torch.nonzero(keep).squeeze(1)
    local = torch.full((n,), -1, dtype=torch.long)
    local[subset] = torch.arange(len(subset))
    inside = keep[a] & keep[b] if hops else torch.zeros(len(a), dtype=torch.bool)
    return Snapshot(snap.x[subset], local[snap.edge_index[:, inside]]), local[nodes]


@dataclass
class EdgeSet:
    """Transactions of one split as tensors, with the snapshot (day) each one is scored on."""
    src: torch.Tensor
    dst: torch.Tensor
    tab: torch.Tensor
    day: np.ndarray
    y: np.ndarray


def make_edge_set(X: pd.DataFrame, meta: pd.DataFrame, nodes: pd.DataFrame, scaler: TabularScaler,
                  day0: pd.Timestamp, y: np.ndarray) -> EdgeSet:
    """`nodes` maps tx_id -> from_idx, to_idx (indexed by tx_id)."""
    idx = nodes.loc[meta["tx_id"].to_numpy()]
    return EdgeSet(src=torch.from_numpy(idx["from_idx"].to_numpy(np.int64, copy=True)),
                   dst=torch.from_numpy(idx["to_idx"].to_numpy(np.int64, copy=True)),
                   tab=torch.from_numpy(scaler.transform(X)),
                   day=day_index(meta["timestamp"], day0), y=y)


@torch.no_grad()
def predict(model: EdgeGNN, snaps: list[Snapshot], edges: EdgeSet, chunk: int = 200_000) -> np.ndarray:
    """Laundering probability for every transaction, each scored on its own day's snapshot."""
    model.eval()
    out = np.empty(len(edges.y), np.float32)
    for k in np.unique(edges.day):
        h = model.embed(snaps[k])
        rows = np.flatnonzero(edges.day == k)
        for i in range(0, len(rows), chunk):
            r = torch.from_numpy(rows[i:i + chunk])
            out[rows[i:i + chunk]] = torch.sigmoid(
                model.score(h, edges.src[r], edges.dst[r], edges.tab[r])).numpy()
    return out


def train_gnn(snaps: list[Snapshot], train: EdgeSet, val: EdgeSet, config: GNNConfig = GNNConfig(),
              message_passing: bool = True) -> tuple[EdgeGNN, pd.DataFrame]:
    """Train with early stopping on validation PR-AUC; returns the best model and the epoch log.

    Each step scores a chunk of one training day's sampled transactions, embedding only their
    accounts' neighbourhood in that day's snapshot.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = EdgeGNN(snaps[0].x.shape[1], train.tab.shape[1], config.hidden, config.layers,
                    config.dropout, message_passing)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pos = float(train.y.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt((len(train.y) - pos) / pos)))
    y_train = torch.from_numpy(train.y.astype(np.float32))

    history, best, best_state, waited = [], -np.inf, None, 0
    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()
        model.train()
        batches = []
        for k in np.unique(train.day):
            rows = rng.permutation(np.flatnonzero(train.day == k))
            batches += [(k, rows[i:i + config.chunk_rows]) for i in range(0, len(rows), config.chunk_rows)]
        losses = []
        for j in rng.permutation(len(batches)):
            k, rows = batches[j]
            r = torch.from_numpy(rows)
            h, pos = model.embed_nodes(snaps[k], torch.cat([train.src[r], train.dst[r]]))
            loss = loss_fn(model.score(h, pos[:len(r)], pos[len(r):], train.tab[r]), y_train[r])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_metrics = ranking_metrics(val.y, predict(model, snaps, val))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
                        "minutes": (time.time() - t0) / 60})
        log.info("epoch %d: loss %.4f, val PR-AUC %.4f (%.1f min)", epoch, history[-1]["train_loss"],
                 val_metrics["pr_auc"], history[-1]["minutes"])
        if val_metrics["pr_auc"] > best:
            best, best_state, waited = val_metrics["pr_auc"], copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= config.patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history).set_index("epoch")


def combine_scores(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Average of two models' log-odds: a simple ensemble whose threshold is still set on validation."""
    eps = 1e-7

    def logit(p):
        p = np.clip(p.astype(np.float64), eps, 1 - eps)
        return np.log(p / (1 - p))

    return 1 / (1 + np.exp(-(logit(p_a) + logit(p_b)) / 2))
