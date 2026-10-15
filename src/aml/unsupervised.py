"""Step 6: detecting laundering without labels.

A bank facing a new scheme has no labels for it. These methods never see labels: labels are only
used afterwards to measure how much laundering they find. The no-leakage rule still holds: a
transaction on calendar day k is scored with communities found in payments before midnight
starting day k (daily snapshots, as in step 5) and with its own step 2 / step 4 features, which
only look backwards.

Three scores per transaction:

- **Isolation Forest** on the transaction's step 2 + step 4 features: how unusual the payment is.
  Fitted on a uniform random sample of training transactions (labels ignored).
- **Crossing communities**: Leiden community detection (modularity) on the account graph, without
  hubs. A payment between two different communities, or to/from an account with no history,
  leaves the sender's usual circle.
- **Unusual communities**: an Isolation Forest over statistics of each day's communities (size,
  density, reciprocity, account age, share of new relationships, ...); a payment scores as the
  more unusual of its sender's and receiver's communities.

The combined score averages the three, each turned into a percentile of its validation scores,
with equal weights chosen in advance (choosing weights with labels would defeat the purpose).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import _minutes, day_index, first_midnight
from .logging_utils import get_logger

log = get_logger(__name__)

def signed_log(X: pd.DataFrame) -> np.ndarray:
    """Signed log1p, with missing values below every observed value (like step 2's screening)."""
    v = X.to_numpy(np.float64)
    v = np.sign(v) * np.log1p(np.abs(v))
    with warnings.catch_warnings():            # all-missing columns in a chunk are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanmin(v, axis=0) - 1
    return np.where(np.isnan(v), np.where(np.isnan(low), 0, low), v)


def fit_transaction_forest(X_sample: pd.DataFrame, seed: int = 42, n_estimators: int = 200) -> IsolationForest:
    """Isolation Forest on a uniform sample of transactions (no labels)."""
    return IsolationForest(n_estimators=n_estimators, max_samples=512, random_state=seed,
                           n_jobs=-1).fit(signed_log(X_sample))


def transaction_anomaly(forest: IsolationForest, X: pd.DataFrame, chunk: int = 500_000) -> np.ndarray:
    """Higher = more unusual."""
    return np.concatenate([-forest.score_samples(signed_log(X.iloc[i:i + chunk]))
                           for i in range(0, len(X), chunk)])


def leiden_partition(a: np.ndarray, b: np.ndarray, n_nodes: int, seed: int = 42) -> np.ndarray:
    """Community of each account (Leiden, modularity, edges weighted by number of payments).

    `a`, `b` are payments as account pairs; direction is ignored. Accounts with no payment get -1.
    """
    import random

    import igraph as ig

    lo, hi = np.minimum(a, b), np.maximum(a, b)
    w = pd.DataFrame({"lo": lo, "hi": hi}).value_counts().reset_index(name="w")
    g = ig.Graph(n=n_nodes, edges=np.column_stack([w["lo"], w["hi"]]).tolist())
    g.es["weight"] = w["w"].to_numpy(np.float64)
    ig.set_random_number_generator(random.Random(seed))
    part = g.community_leiden(objective_function="modularity", weights="weight", n_iterations=-1)
    member = np.asarray(part.membership)
    seen = np.zeros(n_nodes, bool)
    seen[a] = seen[b] = True
    return np.where(seen, member, -1)


def community_table(member: np.ndarray, a: np.ndarray, b: np.ndarray, minute: np.ndarray,
                    first_seen: np.ndarray, cut: int) -> pd.DataFrame:
    """Statistics of each community (size >= 3) from payments before `cut`, indexed by community id.

    `a`, `b`, `minute` are the payments before the cut; `first_seen` is each account's first
    activity minute.
    """
    ca, cb = member[a], member[b]
    internal = ca == cb
    pairs = pd.DataFrame({"c": ca[internal], "a": a[internal], "b": b[internal], "m": minute[internal]})
    per_pair = pairs.groupby(["c", "a", "b"])["m"].agg(["size", "min"]).reset_index()
    rev = per_pair.rename(columns={"a": "b", "b": "a"})[["c", "a", "b"]].assign(back=1)
    per_pair = per_pair.merge(rev, on=["c", "a", "b"], how="left")

    nodes = pd.DataFrame({"c": member, "age": cut - first_seen})
    nodes = nodes[nodes["c"] >= 0]
    size = nodes.groupby("c").size()
    t = pd.DataFrame({
        "size": size,
        "edges": per_pair.groupby("c").size(),
        "payments": per_pair.groupby("c")["size"].sum(),
        "reciprocal": per_pair.groupby("c")["back"].sum(),
        "new_24h": (per_pair["min"] >= cut - 24 * 60).groupby(per_pair["c"]).sum(),
        "external": pd.Series(np.concatenate([ca[~internal], cb[~internal]])).value_counts(),
        "mean_age_min": nodes.groupby("c")["age"].mean(),
    }).fillna(0)
    t = t[t["size"] >= 3]
    edges = t["edges"].clip(lower=1)
    return pd.DataFrame({
        "log_size": np.log(t["size"]),
        "density": t["edges"] / (t["size"] * (t["size"] - 1)),
        "reciprocity": t["reciprocal"] / edges,
        "log_internal_payments_per_edge": np.log1p(t["payments"] / edges),
        "internal_share": t["payments"] / (t["payments"] + t["external"]).clip(lower=1),
        "log_mean_account_age_h": np.log1p(t["mean_age_min"] / 60),
        "new_edge_share_24h": t["new_24h"] / edges,
    }, index=t.index)


@dataclass
class CommunityScores:
    cross_community: np.ndarray       # 1 if sender and receiver are in different communities (or unseen)
    community_anomaly: np.ndarray     # max anomaly of sender's and receiver's communities (0 if none)
    member_by_day: list[np.ndarray]


def community_scores(tx: pd.DataFrame, hubs: np.ndarray, seed: int = 42) -> CommunityScores:
    """Score every transaction with the communities of its day's snapshot (payments before midnight).

    `tx` needs timestamp, from_idx, to_idx, is_self_loop for all transactions.
    """
    origin = tx["timestamp"].min()
    minute = _minutes(tx["timestamp"], origin)
    offset = first_midnight(origin)
    day = day_index(tx["timestamp"], origin)
    src, dst = tx["from_idx"].to_numpy(np.int64), tx["to_idx"].to_numpy(np.int64)
    n = int(max(src.max(), dst.max())) + 1
    hub = np.zeros(n, bool)
    hub[hubs] = True
    usable = (tx["is_self_loop"].to_numpy() == 0) & ~hub[src] & ~hub[dst]
    first_seen = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(first_seen, src, minute)
    np.minimum.at(first_seen, dst, minute)

    cross = np.ones(len(tx), np.int8)
    anomaly = np.zeros(len(tx))
    members = []
    for k in np.unique(day):
        cut = int(offset + k * 24 * 60)
        past = usable & (minute < cut)
        rows = day == k
        if past.sum() == 0:
            members.append(np.full(n, -1))
            continue
        member = leiden_partition(src[past], dst[past], n, seed)
        member[hub] = -1
        members.append(member)
        table = community_table(member, src[past], dst[past], minute[past], first_seen, cut)
        forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(table.to_numpy())
        comm_score = pd.Series(-forest.score_samples(table.to_numpy()), index=table.index)
        lookup = np.zeros(int(member.max()) + 2)
        lookup[comm_score.index.to_numpy()] = comm_score.to_numpy() - comm_score.min()
        cs, cd = member[src[rows]], member[dst[rows]]
        cross[rows] = ~((cs >= 0) & (cs == cd))
        anomaly[rows] = np.maximum(np.where(cs >= 0, lookup[cs], 0), np.where(cd >= 0, lookup[cd], 0))
        log.info("day %d: %s communities (%d with 3+ accounts), %.1f%% of the day's payments cross communities",
                 k, f"{member.max() + 1:,}", len(table), 100 * cross[rows].mean())
    return CommunityScores(cross, anomaly, members)


def group_recovery(attempts: pd.DataFrame, member_by_day: list[np.ndarray], full_member: np.ndarray,
                   day0: pd.Timestamp) -> pd.DataFrame:
    """After one alert, how much of a laundering attempt's group sits in the alerted account's community?

    For each attempt in the Patterns file (`attempts`: attempt_id, pattern_type, timestamp,
    from_idx, to_idx), take the sender of its first transaction as the alerted account and the
    communities found on payments up to the end of the attempt's last day: this is an
    investigation after the fact, not a prediction. Reports the share of the attempt's other
    accounts found in that community, the community's size (how many accounts an investigator
    would review), and the share expected if the community were a random set of that size.
    """
    rows = []
    n_active = int((full_member >= 0).sum())
    for (ptype, _), g in attempts.sort_values("timestamp").groupby(["pattern_type", "attempt_id"]):
        last_day = int(day_index(g["timestamp"], day0).max())
        member = member_by_day[last_day + 1] if last_day + 1 < len(member_by_day) else full_member
        seed = int(g["from_idx"].iloc[0])
        others = np.setdiff1d(np.union1d(g["from_idx"], g["to_idx"]), [seed])
        if member[seed] < 0 or len(others) == 0:
            continue
        size = int((member == member[seed]).sum())
        rows.append({"pattern_type": ptype, "share_recovered": float((member[others] == member[seed]).mean()),
                     "community_size": size, "random_share": (size - 1) / max(n_active - 1, 1)})
    r = pd.DataFrame(rows)
    out = r.groupby("pattern_type").agg(
        attempts=("share_recovered", "size"), share_recovered=("share_recovered", "mean"),
        all_recovered=("share_recovered", lambda s: (s == 1).mean()),
        median_community_size=("community_size", "median"), random_share=("random_share", "mean"))
    out.loc["all"] = [len(r), r["share_recovered"].mean(), (r["share_recovered"] == 1).mean(),
                      r["community_size"].median(), r["random_share"].mean()]
    return out


def percentile_of(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Share of `reference` values below each x (0..1); maps any score onto a common scale."""
    ref = np.sort(reference)
    return np.searchsorted(ref, x, side="left") / len(ref)


def combine(scores_val: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight average of each score's percentile within its validation distribution."""
    return np.mean([percentile_of(scores_val[k], scores[k]) for k in scores], axis=0)


def label_free_threshold(score_val: np.ndarray, share: float) -> float:
    """Cut-off that flags `share` of all validation transactions (no labels needed)."""
    return float(np.quantile(score_val, 1 - share))
