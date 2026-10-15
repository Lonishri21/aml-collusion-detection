"""Step 7: scoring new payments in production, and watching for drift.

**Feature store.** Steps 2 and 4 describe a payment with histories (each account's payments, pairs
of accounts, paths in the graph, daily graph snapshots). The store holds exactly those structures,
built once from the transaction history with the same code as the batch pipeline, so a payment
scored here gets the same features it would get in `features.parquet` / `graph_features.parquet`
(a test checks this). Every query only counts history strictly before the payment's minute.

A new payment sees the whole stored history. Payments scored by the service are not added to the
store: the store is rebuilt from the updated history on a schedule (e.g. nightly), the usual
batch-refresh design. Accounts never seen before get the features of an account with no history.

**Scorer.** The step 4 XGBoost (best single model, no GPU or graph library needed at serving
time) with the alert threshold fixed on validation for the 1% budget. Each alert comes with the
features that pushed its score up most (XGBoost's per-feature contributions), for investigators.

**Drift monitor.** Compares recent scores with the validation scores the threshold was set on:
the alert rate against the budget, and the population stability index (PSI) of the score.
Step 3 found the threshold drifting from 1% on validation to 1.9% of normal payments on test;
this is the check that would have raised it.
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import node_key, normalize_bank
from .config import Config
from .features import AccountHistories, EntityLookup, _minutes, day_index, first_midnight
from .graph_features import GraphHistories
from .logging_utils import get_logger
from .metrics import psi

log = get_logger(__name__)

# Fields of one incoming payment, in the raw dataset's terms.
RAW_FIELDS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid",
              "payment_currency", "amount_received", "receiving_currency", "payment_format"]


@dataclass
class FeatureStore:
    origin: pd.Timestamp            # minute 0 of the history
    T: int                          # minutes the store can answer for (history + horizon)
    history_end: pd.Timestamp
    node_index: pd.Index            # account id ("bank_account") -> account index
    accounts: AccountHistories
    graph: GraphHistories
    entities: EntityLookup | None
    fx_rates: dict[str, float]
    format_codes: dict[str, int]
    features: list[str]             # model inputs, in model order

    @property
    def unknown(self) -> int:
        """Account index used for accounts not in the history (it has no events)."""
        return len(self.node_index)

    @classmethod
    def build(cls, config: Config = Config(), horizon_days: int = 30) -> "FeatureStore":
        """Build from the step 1 outputs (all cleaned transactions) and the step 2/4 settings."""
        from .io import load_accounts

        out = config.paths.processed_dir
        tx = pd.read_parquet(out / "transactions.parquet", columns=[
            "timestamp", "from_idx", "to_idx", "amount_paid_usd", "amount_received_usd", "is_self_loop"])
        graph_info = json.loads((out / "graph_feature_list.json").read_text())
        return cls.from_frames(
            tx, pd.read_parquet(out / "nodes.parquet"), np.array(graph_info["hubs"]),
            features=json.loads((out / "feature_list.json").read_text())["features"] + graph_info["features"],
            fx_rates=json.loads((out / "fx_rates.json").read_text()),
            format_codes=json.loads((out / "payment_formats.json").read_text()),
            accounts=load_accounts(config.paths.accounts_csv) if config.paths.accounts_csv.exists() else None,
            horizon_days=horizon_days)

    @classmethod
    def from_frames(cls, tx: pd.DataFrame, nodes: pd.DataFrame, hubs: np.ndarray, features: list[str],
                    fx_rates: dict[str, float], format_codes: dict[str, int],
                    accounts: pd.DataFrame | None = None, horizon_days: int = 30) -> "FeatureStore":
        """Build from in-memory step 1 outputs (`tx`: cleaned transactions, `nodes`: node_id, node_idx)."""
        origin = tx["timestamp"].min()
        minute = _minutes(tx["timestamp"], origin)
        T = int(minute.max()) + 1 + horizon_days * 24 * 60
        n = len(nodes) + 1                                   # + one slot for unknown accounts
        src, dst = tx["from_idx"].to_numpy(np.int64), tx["to_idx"].to_numpy(np.int64)
        self_loop = tx["is_self_loop"].to_numpy(bool)
        paid = tx["amount_paid_usd"].to_numpy(np.float64)
        received = tx["amount_received_usd"].to_numpy(np.float64)

        histories = AccountHistories.build(src, dst, minute, self_loop, paid, received, T, n)
        graph = GraphHistories.build(src, dst, minute, self_loop, np.asarray(hubs), n, T, first_midnight(origin))
        entities = None
        if accounts is not None:
            entities = EntityLookup.build(nodes["node_idx"].to_numpy(), nodes["node_id"].astype(str).to_numpy(),
                                          accounts, n)
        store = cls(origin=origin, T=T, history_end=tx["timestamp"].max(),
                    node_index=pd.Index(nodes.sort_values("node_idx")["node_id"].astype(str)),
                    accounts=histories, graph=graph, entities=entities,
                    fx_rates=fx_rates, format_codes=format_codes, features=features)
        log.info("Feature store: %s accounts, history %s to %s, answers up to %s", f"{len(nodes):,}",
                 origin, store.history_end, store.horizon_end)
        return store

    @property
    def horizon_end(self) -> pd.Timestamp:
        return self.origin + pd.Timedelta(minutes=self.T - 1)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "FeatureStore":
        with open(path, "rb") as f:
            return pickle.load(f)

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Clean incoming payments exactly like step 1: account ids, USD amounts, codes and flags."""
        missing = [c for c in RAW_FIELDS if c not in raw]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        df = pd.DataFrame({"timestamp": pd.to_datetime(raw["timestamp"])})
        if (df["timestamp"] > self.horizon_end).any():
            raise ValueError(f"timestamps after {self.horizon_end}: the feature store must be rebuilt")
        for col, values in [("payment_currency", raw["payment_currency"]), ("receiving_currency", raw["receiving_currency"])]:
            unknown = sorted(set(values.astype(str)) - set(self.fx_rates))
            if unknown:
                raise ValueError(f"unknown {col}: {unknown}")
        unknown = sorted(set(raw["payment_format"].astype(str)) - set(self.format_codes))
        if unknown:
            raise ValueError(f"unknown payment_format: {unknown}")

        from_bank, to_bank = normalize_bank(raw["from_bank"]), normalize_bank(raw["to_bank"])
        from_node, to_node = node_key(from_bank, raw["from_account"]), node_key(to_bank, raw["to_account"])
        idx_from, idx_to = self.node_index.get_indexer(from_node), self.node_index.get_indexer(to_node)
        df["from_idx"] = np.where(idx_from >= 0, idx_from, self.unknown)
        df["to_idx"] = np.where(idx_to >= 0, idx_to, self.unknown)
        df["known_sender"], df["known_receiver"] = idx_from >= 0, idx_to >= 0
        df["amount_paid_usd"] = raw["amount_paid"].astype(float).to_numpy() * \
            raw["payment_currency"].astype(str).map(self.fx_rates).to_numpy()
        df["amount_received_usd"] = raw["amount_received"].astype(float).to_numpy() * \
            raw["receiving_currency"].astype(str).map(self.fx_rates).to_numpy()
        df["payment_format_code"] = raw["payment_format"].astype(str).map(self.format_codes).astype("int8").to_numpy()
        df["is_cross_bank"] = (from_bank.to_numpy() != to_bank.to_numpy()).astype("int8")
        df["is_self_loop"] = (from_node.to_numpy() == to_node.to_numpy()).astype("int8")
        return df

    def transform(self, clean: pd.DataFrame) -> pd.DataFrame:
        """Step 2 + step 4 features of cleaned payments, in model order."""
        minute = _minutes(clean["timestamp"], self.origin)
        day = day_index(clean["timestamp"], self.origin)
        s, d = clean["from_idx"].to_numpy(np.int64), clean["to_idx"].to_numpy(np.int64)
        f = self.accounts.query(s, d, minute, clean["amount_paid_usd"].to_numpy(np.float64),
                                clean["amount_received_usd"].to_numpy(np.float64),
                                clean["payment_format_code"].to_numpy(), clean["is_cross_bank"].to_numpy(),
                                clean["is_self_loop"].to_numpy(bool), clean["timestamp"].dt.hour.to_numpy())
        if self.entities is not None:
            for k, v in self.entities.features(s, d).items():
                f[k] = v
        g = self.graph.query(s, d, minute, day, self.accounts.out_hist, self.accounts.in_hist)
        return pd.concat([f, g], axis=1)[self.features]


@dataclass
class Scorer:
    store: FeatureStore
    booster: object                 # xgboost.Booster
    threshold: float
    budget: float

    @classmethod
    def load(cls, store_path: Path, config: Config = Config()) -> "Scorer":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(config.paths.models_dir / "xgb_graph.json")
        metrics = json.loads((config.paths.reports_dir / "metrics_step4_graph.json").read_text())
        budget = config.baseline.headline_fpr
        return cls(FeatureStore.load(store_path), booster, metrics["thresholds_from_val"][str(budget)], budget)

    def score(self, raw: pd.DataFrame, top: int = 3) -> pd.DataFrame:
        """Laundering score, alert flag and the features that raised the score most."""
        import xgboost as xgb

        clean = self.store.prepare(raw)
        X = self.store.transform(clean)
        dm = xgb.DMatrix(X.astype("float32"), feature_names=list(X.columns))
        best = self.booster.attr("best_iteration")        # set when trained with early stopping
        rounds = (0, int(best) + 1) if best is not None else (0, 0)
        score = self.booster.predict(dm, iteration_range=rounds)
        contrib = self.booster.predict(dm, pred_contribs=True, iteration_range=rounds)[:, :-1]
        names = np.array(X.columns)
        order = np.argsort(-contrib, axis=1)[:, :top]
        return pd.DataFrame({
            "score": score,
            "alert": score >= self.threshold,
            "top_features": [list(names[o]) for o in order],
            "known_sender": clean["known_sender"].to_numpy(),
            "known_receiver": clean["known_receiver"].to_numpy(),
        })


class DriftMonitor:
    """Recent scores vs the validation scores the alert threshold was set on.

    Status is "drift" when the recent alert rate is more than `max_ratio` times the budget (or
    below 1 / max_ratio of it), or the score PSI exceeds `max_psi` (0.2 is the usual "significant
    shift" rule of thumb); "warming_up" until `min_count` payments have been seen.
    """

    def __init__(self, reference: np.ndarray, threshold: float, budget: float, window: int = 50_000,
                 min_count: int = 1_000, max_ratio: float = 1.5, max_psi: float = 0.2):
        self.reference = np.asarray(reference)
        self.threshold, self.budget = threshold, budget
        self.recent: deque = deque(maxlen=window)
        self.min_count, self.max_ratio, self.max_psi = min_count, max_ratio, max_psi
        self.expected_rate = float((self.reference >= threshold).mean())

    def update(self, scores: np.ndarray) -> None:
        self.recent.extend(np.asarray(scores, dtype=float).tolist())

    def report(self) -> dict:
        n = len(self.recent)
        out = {"window": n, "budget": self.budget, "expected_alert_rate": self.expected_rate}
        if n < self.min_count:
            return {**out, "status": "warming_up"}
        cur = np.fromiter(self.recent, float, n)
        rate = float((cur >= self.threshold).mean())
        score_psi = psi(self.reference, cur)
        ratio = rate / self.expected_rate if self.expected_rate > 0 else float("inf")
        drift = ratio > self.max_ratio or ratio < 1 / self.max_ratio or score_psi > self.max_psi
        return {**out, "alert_rate": rate, "alert_rate_ratio": ratio, "score_psi": score_psi,
                "status": "drift" if drift else "ok"}
