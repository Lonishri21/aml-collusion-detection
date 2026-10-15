"""Step 3: XGBoost baseline on the step 2 point-in-time features.

Train fits the model, validation picks the settings and the alert thresholds, and test is scored
once at the end. Graph features (step 4) and GNNs (step 5) count as progress only if they beat
these numbers on the same test set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BaselineConfig
from .logging_utils import get_logger
from .metrics import at_threshold, ranking_metrics, threshold_at_fpr

log = get_logger(__name__)

# Kept next to the model inputs for reference rules, slicing results and error analysis.
META_COLUMNS = ["tx_id", "timestamp", "is_laundering", "payment_format_code", "pair_prior_cnt",
                "pair_reverse_prior_cnt", "snd_out_cnt_all", "log_amount_usd"]

BASE_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=100,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    n_jobs=-1,
)


def search_space(class_ratio: float) -> list[dict]:
    """Settings tried on validation. `class_ratio` is normal / laundering in the sampled train set;
    its square root is a moderate up-weight for the rare class."""
    balanced = float(np.sqrt(class_ratio))
    return [
        {"max_depth": 4, "scale_pos_weight": 1.0},
        {"max_depth": 6, "scale_pos_weight": 1.0},
        {"max_depth": 6, "scale_pos_weight": balanced},
        {"max_depth": 8, "scale_pos_weight": balanced},
    ]


@dataclass
class Splits:
    features: list[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    meta: dict[str, pd.DataFrame]   # META_COLUMNS for "train" (sampled), "val" and "test", aligned with X_*
    overview: pd.DataFrame          # size, laundering and period of each (unsampled) split


def make_splits(data: pd.DataFrame, features: list[str], config: BaselineConfig = BaselineConfig()) -> Splits:
    """Model matrices from the step 2 feature table.

    Train keeps every laundering transaction and a random `neg_sample_rate` of normal ones, which
    keeps memory laptop-sized. Validation and test are never sampled.
    """
    split = data["split"].astype(str).to_numpy()
    label = data["is_laundering"].to_numpy()

    overview = pd.DataFrame({
        name: {"transactions": int((split == name).sum()), "laundering": int(label[split == name].sum()),
               "laundering_rate": float(label[split == name].mean()),
               "start": data.loc[split == name, "timestamp"].min(),
               "end": data.loc[split == name, "timestamp"].max()}
        for name in ["train", "val", "test"]
    }).T

    rng = np.random.default_rng(config.seed)
    train_rows = (split == "train") & ((label == 1) | (rng.random(len(data)) < config.neg_sample_rate))
    meta_cols = [c for c in META_COLUMNS if c in data.columns]

    def matrix(rows: np.ndarray) -> pd.DataFrame:
        return data.loc[rows, features].astype("float32").reset_index(drop=True)

    return Splits(
        features=features,
        X_train=matrix(train_rows), y_train=label[train_rows],
        X_val=matrix(split == "val"), y_val=label[split == "val"],
        X_test=matrix(split == "test"), y_test=label[split == "test"],
        meta={"train": data.loc[train_rows, meta_cols].reset_index(drop=True),
              **{name: data.loc[split == name, meta_cols].reset_index(drop=True) for name in ["val", "test"]}},
        overview=overview,
    )


def load_splits(features_path: Path, features: list[str], config: BaselineConfig = BaselineConfig(),
                extra: list[tuple[Path, list[str]]] = ()) -> Splits:
    """Read only the needed columns of features.parquet and build the splits.

    `extra` adds feature tables from later steps as (path, columns); they must have the same
    rows in the same order as features.parquet, which is checked on `tx_id`.
    """
    cols = list(dict.fromkeys(["split"] + META_COLUMNS + features))
    data = pd.read_parquet(features_path, columns=cols)
    for path, extra_cols in extra:
        more = pd.read_parquet(path, columns=["tx_id"] + extra_cols)
        if not np.array_equal(more["tx_id"].to_numpy(), data["tx_id"].to_numpy()):
            raise ValueError(f"{path} rows do not line up with {features_path}; rebuild both")
        for c in extra_cols:
            data[c] = more[c].to_numpy()
        features = features + extra_cols
        del more
    splits = make_splits(data, features, config)
    log.info("%d features | train (sampled) %s | val %s | test %s", len(features),
             splits.X_train.shape, splits.X_val.shape, splits.X_test.shape)
    return splits


def train_search(splits: Splits, search: list[dict] | None = None, seed: int = 42):
    """Fit each setting with early stopping on validation PR-AUC and keep the best on validation.

    Returns (best model, its search params, table of all runs sorted by validation PR-AUC).
    """
    import xgboost as xgb

    y_train = splits.y_train
    if search is None:
        search = search_space((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    runs, best_model, best_params, best_score = [], None, None, -np.inf
    for i, params in enumerate(search):
        t0 = time.time()
        model = xgb.XGBClassifier(**BASE_PARAMS, **params, random_state=seed)
        model.fit(splits.X_train, y_train, eval_set=[(splits.X_val, splits.y_val)], verbose=False)
        score_val = model.predict_proba(splits.X_val)[:, 1]
        thr = threshold_at_fpr(splits.y_val, score_val, 0.01)
        metrics = ranking_metrics(splits.y_val, score_val)
        runs.append({
            "run": i, **params,
            "trees": model.best_iteration + 1,
            "val_pr_auc": metrics["pr_auc"],
            "val_roc_auc": metrics["roc_auc"],
            "val_recall_at_1pct_fpr": at_threshold(splits.y_val, score_val, thr)["recall"],
            "minutes": (time.time() - t0) / 60,
        })
        log.info("run %d: %s -> val PR-AUC %.4f (%.1f min)", i, params, metrics["pr_auc"], runs[-1]["minutes"])
        if metrics["pr_auc"] > best_score:
            best_model, best_params, best_score = model, params, metrics["pr_auc"]

    table = pd.DataFrame(runs).set_index("run").sort_values("val_pr_auc", ascending=False)
    return best_model, best_params, table


def per_format(y: np.ndarray, score: np.ndarray, is_ach: np.ndarray, threshold: float) -> pd.DataFrame:
    """Results within ACH and within all other formats, so the model can't look good only by
    learning "ACH = suspicious"."""
    rows = {}
    for name, mask in [("ACH only", is_ach), ("all other formats", ~is_ach)]:
        rows[name] = {**ranking_metrics(y[mask], score[mask]),
                      "random_pr_auc": float(y[mask].mean()),
                      "laundering": int(y[mask].sum()),
                      "recall_at_1pct": at_threshold(y[mask], score[mask], threshold)["recall"]}
    out = pd.DataFrame(rows).T
    out["lift_over_random"] = out["pr_auc"] / out["random_pr_auc"]
    return out


def per_pattern(patterns: pd.DataFrame, tx_id: pd.Series, flagged: np.ndarray) -> pd.DataFrame:
    """Recall per laundering scheme for pattern transactions in the scored set.

    `attempts_with_any_alert` is the share of attempts with at least one alert: one alert is
    often enough for investigators to open a case and find the rest.
    """
    hit = pd.Series(flagged, index=tx_id.to_numpy())
    pat = patterns.dropna(subset=["tx_id"])[["tx_id", "attempt_id", "pattern_type"]].copy()
    pat["tx_id"] = pat["tx_id"].astype("int64")
    pat = pat[pat["tx_id"].isin(hit.index)]
    pat["caught"] = hit.reindex(pat["tx_id"]).to_numpy()
    out = pat.groupby("pattern_type").agg(
        attempts=("attempt_id", "nunique"),
        transactions=("tx_id", "size"),
        recall_at_1pct=("caught", "mean"),
    ).sort_values("recall_at_1pct", ascending=False)
    out["attempts_with_any_alert"] = (pat.groupby(["pattern_type", "attempt_id"])["caught"].any()
                                      .groupby("pattern_type").mean())
    return out


def gain_share(model, features: list[str]) -> pd.Series:
    """Each feature's share of total split gain (0 for unused features)."""
    gain = pd.Series(model.get_booster().get_score(importance_type="gain"), name="gain")
    gain = gain.reindex(features).fillna(0).sort_values(ascending=False)
    return gain / gain.sum()


@dataclass
class BaselineResult:
    model: object
    params: dict
    search: pd.DataFrame
    thresholds: dict[float, float]
    score_val: np.ndarray
    score_test: np.ndarray
    budget_val: pd.DataFrame
    budget_test: pd.DataFrame
    validation: dict
    test: dict
    by_format: pd.DataFrame
    by_pattern: pd.DataFrame | None
    gain_share: pd.Series | None    # XGBoost only


def _days(ts: pd.Series) -> float:
    return (ts.max() - ts.min()).total_seconds() / 86400


def evaluate(model, params: dict, search: pd.DataFrame, splits: Splits, ach_code: int,
             patterns: pd.DataFrame | None = None,
             config: BaselineConfig = BaselineConfig()) -> BaselineResult:
    """Fix thresholds on validation, then score test once with everything frozen."""
    return evaluate_scores(model.predict_proba(splits.X_val)[:, 1], model.predict_proba(splits.X_test)[:, 1],
                           model, params, search, splits, ach_code, patterns, config,
                           gain=gain_share(model, splits.features))


def evaluate_scores(score_val: np.ndarray, score_test: np.ndarray, model, params: dict, search: pd.DataFrame,
                    splits: Splits, ach_code: int, patterns: pd.DataFrame | None = None,
                    config: BaselineConfig = BaselineConfig(), gain: pd.Series | None = None,
                    thresholds: dict[float, float] | None = None) -> BaselineResult:
    """`evaluate` for any model's validation and test scores (rows in `splits` order).

    `thresholds` (budget -> cut-off) overrides the default, which flags each budget's share of
    normal validation payments; unsupervised methods pass cut-offs chosen without labels.
    """
    y_val, y_test = splits.y_val, splits.y_test
    if thresholds is None:
        thresholds = {f: threshold_at_fpr(y_val, score_val, f) for f in config.fpr_budgets}
    headline_thr = thresholds[config.headline_fpr]

    def budget_table(y, score, ts):
        table = pd.DataFrame({f: at_threshold(y, score, t) for f, t in thresholds.items()}).T
        table.index.name = "fpr_budget"
        table["alerts_per_day"] = table["alerts"] / _days(ts)
        return table

    test_meta = splits.meta["test"]
    flagged = score_test >= headline_thr
    return BaselineResult(
        model=model, params=params, search=search, thresholds=thresholds,
        score_val=score_val, score_test=score_test,
        budget_val=budget_table(y_val, score_val, splits.meta["val"]["timestamp"]),
        budget_test=budget_table(y_test, score_test, test_meta["timestamp"]),
        validation=ranking_metrics(y_val, score_val),
        test={**ranking_metrics(y_test, score_test),
              "random_pr_auc": float(y_test.mean()),
              **{f"{k}_at_1pct": v for k, v in at_threshold(y_test, score_test, headline_thr).items()}},
        by_format=per_format(y_test, score_test, (test_meta["payment_format_code"] == ach_code).to_numpy(),
                             headline_thr),
        by_pattern=None if patterns is None else per_pattern(patterns, test_meta["tx_id"], flagged),
        gain_share=gain,
    )


def metrics_record(result: BaselineResult, splits: Splits, config: BaselineConfig = BaselineConfig(),
                   name: str = "XGBoost baseline (step 2 features)") -> dict:
    """JSON-ready summary of a run (reports/metrics_step3_baseline.json and later steps)."""
    def plain(d: dict) -> dict:
        return {k: v.item() if isinstance(v, np.generic) else v for k, v in d.items()}

    model = result.model
    if hasattr(model, "best_iteration"):        # XGBoost: fixed settings + the searched ones
        params = {**{k: v for k, v in BASE_PARAMS.items() if k != "n_jobs"}, "random_state": config.seed,
                  **result.params, "best_iteration": int(model.best_iteration)}
    else:
        params = result.params
    return {
        "model": name,
        "params": params,
        "n_features": len(splits.features),
        "train_negative_sample_rate": config.neg_sample_rate,
        "search": [plain(r) for r in result.search.reset_index().to_dict(orient="records")],
        "thresholds_from_val": {str(k): v for k, v in result.thresholds.items()},
        "validation": plain(result.validation),
        "test": plain(result.test),
        "test_by_budget": {str(k): plain(v) for k, v in result.budget_test.to_dict(orient="index").items()},
        "test_by_format": {k: plain(v) for k, v in result.by_format.to_dict(orient="index").items()},
        "test_per_pattern": None if result.by_pattern is None else
        {k: plain(v) for k, v in result.by_pattern.to_dict(orient="index").items()},
        "top_features_by_gain": None if result.gain_share is None else
        result.gain_share.head(15).round(4).to_dict(),
    }
