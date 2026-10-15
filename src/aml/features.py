"""Step 2: point-in-time account features for every transaction.

Rule: a feature for a transaction at minute t may only use transactions that happened
strictly before minute t. Transactions in the same minute are treated as simultaneous and
are not visible to each other. Labels are never used.

Because every feature looks only backwards, features can be computed once over the whole
cleaned dataset: a test-set transaction sees train/val history, exactly as a production
model would see all history up to the moment it scores a payment.

Implementation: each account's history is stored as one sorted array of keys
(node * T + minute). A window query is then two binary searches plus differences of
cumulative sums, so all ~5M transactions are processed with vectorised NumPy instead of
per-account Python loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cleaning import node_key
from .logging_utils import get_logger

log = get_logger(__name__)

# Window name -> length in minutes (None = all history before the transaction).
WINDOWS: dict[str, int | None] = {"1h": 60, "24h": 24 * 60, "all": None}

# Rows queried at a time; keeps peak memory low on a laptop.
CHUNK_ROWS = 500_000

# Columns of transactions.parquet that build_features needs (read only these).
INPUT_COLUMNS = [
    "tx_id", "timestamp", "from_idx", "to_idx", "from_node", "to_node",
    "amount_paid_usd", "amount_received_usd", "payment_format_code",
    "is_cross_bank", "is_self_loop", "is_laundering", "split",
]


def _minutes(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return ((ts - origin) // pd.Timedelta(minutes=1)).to_numpy(dtype=np.int64)


def day_index(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Calendar day of each timestamp, counted from the day of `origin` (0, 1, ...)."""
    return ((ts.dt.normalize() - origin.normalize()) // pd.Timedelta(days=1)).to_numpy()


def first_midnight(origin: pd.Timestamp) -> int:
    """Minute, relative to `origin`, of the midnight starting origin's day (0 or negative)."""
    return int(_minutes(pd.Series([origin.normalize()]), origin)[0])


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _before(cs: np.ndarray, idx: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Per-node running total of everything before position `idx` (0 if none)."""
    if cs.size == 0:
        return np.zeros(len(idx))
    return np.where(idx > start, cs[np.clip(idx - 1, 0, None)], 0.0)


@dataclass
class EventHistory:
    """All events of one direction (e.g. outgoing payments), sorted by (node, minute).

    Running totals restart for every node, so an account's features depend only on that
    account's own rows: exactly reproducible, and unaffected by unrelated data.
    """

    keys: np.ndarray        # node * T + minute, ascending
    minute: np.ndarray
    cs_amount: np.ndarray   # per-node inclusive running sums
    cs_log: np.ndarray
    cs_log2: np.ndarray
    cs_first: np.ndarray    # per-node running count of first-seen counterparties
    cummax: np.ndarray      # per-node running max amount
    T: int

    @classmethod
    def build(cls, node: np.ndarray, minute: np.ndarray, counterparty: np.ndarray,
              amount: np.ndarray, T: int) -> "EventHistory":
        order = np.lexsort((minute, node))
        node, minute = node[order], minute[order]
        counterparty, amount = counterparty[order], amount[order]
        log_amt = np.log1p(np.clip(amount, 0, None))
        first = (~pd.DataFrame({"n": node, "c": counterparty}).duplicated()).astype(np.float64)
        g = pd.DataFrame({"amount": amount, "log": log_amt, "log2": log_amt ** 2,
                          "first": first.to_numpy()}).groupby(node)
        cs = g.cumsum()
        return cls(
            keys=node.astype(np.int64) * T + minute,
            minute=minute,
            cs_amount=cs["amount"].to_numpy(),
            cs_log=cs["log"].to_numpy(),
            cs_log2=cs["log2"].to_numpy(),
            cs_first=cs["first"].to_numpy(),
            cummax=g["amount"].cummax().to_numpy(),
            T=T,
        )

    def query(self, node: np.ndarray, minute: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        """Features of each `node` from its events strictly before `minute`."""
        base = node.astype(np.int64) * self.T
        start = np.searchsorted(self.keys, base, side="left")           # node's first event
        hi = np.searchsorted(self.keys, base + minute, side="left")     # first event at/after t
        has = hi > start
        out: dict[str, np.ndarray] = {}

        for name, w in WINDOWS.items():
            if w is None:
                lo = start
            else:   # window (t - w, t): minutes >= t - w + 1
                lo = np.maximum(start, np.searchsorted(self.keys, base + minute - w + 1, side="left"))
            cnt = hi - lo
            total = _before(self.cs_amount, hi, start) - _before(self.cs_amount, lo, start)
            out[f"{prefix}_cnt_{name}"] = cnt
            out[f"{prefix}_sum_usd_{name}"] = total
            out[f"{prefix}_mean_usd_{name}"] = _safe_div(total, cnt)

        cnt_all = hi - start
        mean_log = _safe_div(_before(self.cs_log, hi, start), cnt_all)
        mean_log2 = _safe_div(_before(self.cs_log2, hi, start), cnt_all)
        var = np.clip(mean_log2 - mean_log ** 2, 0, None)
        out[f"{prefix}_std_log_usd_all"] = np.where(cnt_all >= 2, np.sqrt(var), np.nan)
        last = np.clip(hi - 1, 0, None)
        empty = len(self.minute) == 0
        out[f"{prefix}_max_usd_all"] = np.full(len(node), np.nan) if empty else np.where(has, self.cummax[last], np.nan)
        distinct = _before(self.cs_first, hi, start)
        out[f"{prefix}_distinct_cp_all"] = distinct
        out[f"{prefix}_cp_ratio_all"] = _safe_div(distinct, cnt_all)   # 1 = every payment to someone new
        if empty:
            out[f"{prefix}_mins_since_last"] = np.full(len(node), np.nan)
            out[f"{prefix}_mins_since_first"] = np.full(len(node), np.nan)
        else:
            first_pos = np.clip(start, 0, len(self.minute) - 1)
            out[f"{prefix}_mins_since_last"] = np.where(has, minute - self.minute[last], np.nan)
            out[f"{prefix}_mins_since_first"] = np.where(has, minute - self.minute[first_pos], np.nan)
        return out

    def new_counterparties(self, node: np.ndarray, minute: np.ndarray, window: int) -> np.ndarray:
        """Number of first-ever counterparties of each `node` in the `window` minutes before `minute`."""
        base = node.astype(np.int64) * self.T
        start = np.searchsorted(self.keys, base, side="left")
        hi = np.searchsorted(self.keys, base + minute, side="left")
        lo = np.maximum(start, np.searchsorted(self.keys, base + minute - window + 1, side="left"))
        return _before(self.cs_first, hi, start) - _before(self.cs_first, lo, start)


@dataclass
class PairHistory:
    """Directed (sender, receiver) pairs, sorted, to count earlier payments between two accounts."""

    keys: np.ndarray
    n_nodes: int
    T: int

    @classmethod
    def build(cls, src: np.ndarray, dst: np.ndarray, minute: np.ndarray, n_nodes: int, T: int) -> "PairHistory":
        if float(n_nodes) ** 2 * T >= 2 ** 62:
            raise OverflowError("Pair key would overflow int64; too many nodes for this encoding")
        keys = (src.astype(np.int64) * n_nodes + dst) * T + minute
        return cls(np.sort(keys), n_nodes, T)

    def count_before(self, src: np.ndarray, dst: np.ndarray, minute: np.ndarray) -> np.ndarray:
        base = (src.astype(np.int64) * self.n_nodes + dst) * self.T
        return np.searchsorted(self.keys, base + minute, "left") - np.searchsorted(self.keys, base, "left")


def entity_table(accounts: pd.DataFrame) -> pd.DataFrame:
    """node_id -> entity_id, entity type (Corporation, Partnership, ...) and accounts per entity."""
    t = pd.DataFrame({
        "node_id": node_key(accounts["bank"], accounts["account"]),
        "entity_id": accounts["entity_id"],
        "entity_type": accounts["entity_name"].str.split(" #").str[0],
    })
    t["entity_n_accounts"] = t.groupby("entity_id")["entity_id"].transform("size")
    return t.drop_duplicates("node_id")


@dataclass
class EntityLookup:
    """Owner facts per account index: entity type code, accounts per entity, entity code (-1 = unknown).

    Arrays have one slot per account index, so lookups are vectorised; indices without an entry in
    the accounts file are unknown (-1 / NaN).
    """

    etype: np.ndarray
    n_accounts: np.ndarray
    entity_code: np.ndarray
    types: list[str]

    @classmethod
    def build(cls, idx: np.ndarray, node_ids: np.ndarray, accounts: pd.DataFrame, size: int) -> "EntityLookup":
        """`idx[i]` is the account index of node id `node_ids[i]`; `size` is the number of slots."""
        ent = entity_table(accounts).set_index("node_id").reindex(node_ids)
        types = sorted(ent["entity_type"].dropna().unique())
        etype = np.full(size, -1, np.int8)
        etype[idx] = pd.Categorical(ent["entity_type"], categories=types).codes
        n_acc = np.full(size, np.nan, np.float32)
        n_acc[idx] = ent["entity_n_accounts"].to_numpy(dtype=np.float32)
        code = np.full(size, -1, np.int64)
        code[idx] = pd.factorize(ent["entity_id"], use_na_sentinel=True)[0]
        return cls(etype, n_acc, code, types)

    def features(self, src: np.ndarray, dst: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "snd_entity_type": self.etype[src], "rcv_entity_type": self.etype[dst],
            "snd_entity_n_accounts": self.n_accounts[src], "rcv_entity_n_accounts": self.n_accounts[dst],
            "same_entity": ((self.entity_code[src] == self.entity_code[dst])
                            & (self.entity_code[src] >= 0)).astype(np.int8),
        }


@dataclass
class AccountHistories:
    """Everything step 2 needs to describe a payment from history: built once, queried for any
    (sender, receiver, minute). Used by `build_features` and by the scoring service."""

    out_hist: EventHistory
    in_hist: EventHistory
    self_keys: np.ndarray
    pairs: PairHistory
    T: int

    @classmethod
    def build(cls, src: np.ndarray, dst: np.ndarray, minute: np.ndarray, self_loop: np.ndarray,
              paid: np.ndarray, received: np.ndarray, T: int, n_nodes: int) -> "AccountHistories":
        # Self-transfers are kept out of the payment histories (they are not edges between
        # accounts) and summarised by their own count instead.
        e = ~self_loop
        log.info("Building histories from %s non-self transactions", f"{e.sum():,}")
        return cls(
            out_hist=EventHistory.build(src[e], minute[e], dst[e], paid[e], T),
            in_hist=EventHistory.build(dst[e], minute[e], src[e], received[e], T),
            self_keys=np.sort(src[self_loop].astype(np.int64) * T + minute[self_loop]),
            pairs=PairHistory.build(src[e], dst[e], minute[e], n_nodes, T),
            T=T,
        )

    def query(self, s: np.ndarray, d: np.ndarray, t: np.ndarray, paid: np.ndarray, received: np.ndarray,
              fmt: np.ndarray, cross_bank: np.ndarray, self_loop: np.ndarray, hour: np.ndarray) -> pd.DataFrame:
        """Step 2 features (without entity info) of payments s -> d at minutes t."""
        f: dict[str, np.ndarray] = {}
        f |= self.out_hist.query(s, t, "snd_out")
        f |= self.in_hist.query(s, t, "snd_in")
        f |= self.out_hist.query(d, t, "rcv_out")
        f |= self.in_hist.query(d, t, "rcv_in")
        base = s.astype(np.int64) * self.T
        f["snd_self_cnt_all"] = np.searchsorted(self.self_keys, base + t) - np.searchsorted(self.self_keys, base)
        f["pair_prior_cnt"] = self.pairs.count_before(s, d, t)            # sender paid receiver before
        f["pair_reverse_prior_cnt"] = self.pairs.count_before(d, s, t)    # receiver paid sender before
        # The current transaction itself
        f["log_amount_usd"] = np.log1p(np.clip(paid, 0, None))
        f["amount_vs_snd_mean"] = _safe_div(paid, f["snd_out_mean_usd_all"])
        f["amount_vs_rcv_in_mean"] = _safe_div(received, f["rcv_in_mean_usd_all"])
        f["payment_format_code"] = fmt
        f["is_cross_bank"] = cross_bank
        f["is_self_loop"] = self_loop.astype(np.int8)
        f["hour"] = hour.astype(np.int8)
        # Compact types: counts as int32, everything else continuous as float32.
        for k, v in f.items():
            if k.endswith(("_cnt_1h", "_cnt_24h", "_cnt_all", "prior_cnt")):
                f[k] = v.astype(np.int32)
            elif v.dtype.kind == "f" or k.endswith("distinct_cp_all"):
                f[k] = v.astype(np.float32)
        return pd.DataFrame(f)


def build_features(tx: pd.DataFrame, accounts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per transaction: `tx_id`, `split`, `is_laundering` and all features.

    `tx` is the cleaned output of step 1 (needs timestamp, from_idx, to_idx, USD amounts,
    flags, payment_format_code). Rows keep the input order.
    """
    origin = tx["timestamp"].min()
    minute = _minutes(tx["timestamp"], origin)
    T = int(minute.max()) + 1
    src = tx["from_idx"].to_numpy(np.int64)
    dst = tx["to_idx"].to_numpy(np.int64)
    n_nodes = int(max(src.max(), dst.max())) + 1
    self_loop = tx["is_self_loop"].to_numpy(bool)
    paid = tx["amount_paid_usd"].to_numpy(np.float64)
    received = tx["amount_received_usd"].to_numpy(np.float64)
    hist = AccountHistories.build(src, dst, minute, self_loop, paid, received, T, n_nodes)

    fmt = tx["payment_format_code"].to_numpy()
    cross_bank = tx["is_cross_bank"].to_numpy()
    hour = tx["timestamp"].dt.hour.to_numpy()

    log.info("Querying sender and receiver histories in chunks of %s rows", f"{CHUNK_ROWS:,}")
    parts = [hist.query(src[sl], dst[sl], minute[sl], paid[sl], received[sl], fmt[sl], cross_bank[sl],
                        self_loop[sl], hour[sl])
             for sl in (slice(i, i + CHUNK_ROWS) for i in range(0, len(tx), CHUNK_ROWS))]
    out = pd.concat(parts, ignore_index=True)
    del parts
    out.index = tx.index

    if accounts is not None:
        nodes = pd.concat([
            pd.DataFrame({"idx": src, "node_id": tx["from_node"].astype(str).to_numpy()}),
            pd.DataFrame({"idx": dst, "node_id": tx["to_node"].astype(str).to_numpy()}),
        ]).drop_duplicates("idx")
        lookup = EntityLookup.build(nodes["idx"].to_numpy(), nodes["node_id"].to_numpy(), accounts, n_nodes)
        for k, v in lookup.features(src, dst).items():
            out[k] = v
        log.info("Entity info joined (%s); %.2f%% of senders not in accounts file",
                 ", ".join(f"{i}={t}" for i, t in enumerate(lookup.types)),
                 100 * (out["snd_entity_type"] < 0).mean())

    meta = tx[["tx_id", "timestamp", "split", "is_laundering"]]
    log.info("Built %d features for %s transactions", out.shape[1], f"{len(out):,}")
    return pd.concat([meta, out], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers, split and label."""
    return [c for c in df.columns if c not in {"tx_id", "timestamp", "split", "is_laundering"}]
