"""Each pipeline step end to end: read inputs, run the step, write its outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import add_flags, add_node_ids, add_usd_amounts, estimate_fx_rates, trim_after
from .config import Config
from .io import load_patterns, load_transactions, write_json
from .logging_utils import get_logger
from .patterns import match_patterns, pattern_coverage
from .quality import daily_profile, laundering_rate_by, quality_report
from .schema import CLEAN_COLUMNS
from .splitting import split_summary, temporal_split

log = get_logger(__name__)

# Output of each step that later steps read, and the command that writes it.
PRODUCED_BY = {
    "transactions.parquet": "clean",
    "pattern_transactions.parquet": "clean",
    "features.parquet": "features",
    "baseline_scores.parquet": "baseline",
    "graph_features.parquet": "graph-features",
    "graph_scores.parquet": "graph-model",
}


def _require(out: Path, *names: str) -> None:
    for name in names:
        if not (out / name).exists():
            raise FileNotFoundError(f"{out / name} not found; run `python main.py {PRODUCED_BY[name]}` first")


def _read_json(path: Path):
    return json.loads(path.read_text())


def _load_patterns(out: Path) -> pd.DataFrame | None:
    path = out / "pattern_transactions.parquet"
    return pd.read_parquet(path, columns=["tx_id", "attempt_id", "pattern_type"]) if path.exists() else None


def _write_scores(splits, path: Path, **scores: tuple[np.ndarray, np.ndarray]) -> None:
    """Validation and test scores per transaction, so later steps can compare models row by row.

    Each keyword is an output column holding (validation scores, test scores).
    """
    pd.concat([
        splits.meta[name][["tx_id", "is_laundering"]].assign(split=name, **{c: v[i] for c, v in scores.items()})
        for i, name in enumerate(("val", "test"))
    ]).to_parquet(path, index=False)


@dataclass
class CleaningResult:
    transactions: pd.DataFrame
    nodes: pd.DataFrame
    fx_rates: dict[str, float]
    fx_diagnostics: pd.DataFrame
    format_codes: dict[str, int]
    summary: dict


def clean(raw: pd.DataFrame, config: Config = Config()) -> CleaningResult:
    """Run every cleaning step on already-loaded raw transactions (no file I/O).

    The notebook calls this directly; `run_cleaning` wraps it with reading and writing.
    """
    cfg = config.cleaning
    summary: dict = {"quality_raw": quality_report(raw)}

    df = trim_after(raw, cfg.max_date)
    summary["rows_removed_by_max_date"] = int(len(raw) - len(df))

    df, nodes = add_node_ids(df)
    rates, fx_diag = estimate_fx_rates(df, base=cfg.base_currency)
    df = add_usd_amounts(df, rates)
    df, format_codes = add_flags(df)
    df = temporal_split(df, cfg.train_frac, cfg.val_frac)

    cc = df[df["is_cross_currency"] == 1]
    gap = ((cc["amount_paid_usd"] - cc["amount_received_usd"]).abs() / cc["amount_paid_usd"]).median()
    summary.update({
        "config": {"max_date": cfg.max_date, "train_frac": cfg.train_frac, "val_frac": cfg.val_frac},
        "rows_clean": int(len(df)),
        "n_nodes": int(len(nodes)),
        "self_loop_share": float(df["is_self_loop"].mean()),
        "cross_bank_share": float(df["is_cross_bank"].mean()),
        "cross_currency_share": float(df["is_cross_currency"].mean()),
        "fx_check_median_relative_gap": None if cc.empty else float(gap),
        "split": split_summary(df).reset_index().to_dict(orient="records"),
    })
    return CleaningResult(df[CLEAN_COLUMNS], nodes, rates, fx_diag, format_codes, summary)


def run_cleaning(config: Config = Config()) -> CleaningResult:
    """Read raw files, clean, and write all step 1 artifacts to data/processed."""
    paths = config.paths
    out = paths.processed_dir
    out.mkdir(parents=True, exist_ok=True)

    result = clean(load_transactions(paths.transactions_csv), config)
    df = result.transactions

    if paths.patterns_txt.exists():
        matched = match_patterns(df, load_patterns(paths.patterns_txt))
        matched.to_parquet(out / "pattern_transactions.parquet", index=False)
        result.summary["patterns"] = pattern_coverage(df, matched)
    else:
        log.warning("No patterns file at %s; skipping pattern matching", paths.patterns_txt)

    df.to_parquet(out / "transactions.parquet", index=False)
    result.nodes.to_parquet(out / "nodes.parquet", index=False)
    daily_profile(df).to_csv(out / "daily_profile.csv")
    for col in ("payment_format", "is_cross_currency", "is_self_loop"):
        laundering_rate_by(df, col).to_csv(out / f"laundering_by_{col}.csv")
    result.fx_diagnostics.to_csv(out / "fx_diagnostics.csv")
    write_json(result.fx_rates, out / "fx_rates.json")
    write_json(result.format_codes, out / "payment_formats.json")
    write_json(result.summary, out / "cleaning_summary.json")

    log.info("Wrote step 1 outputs to %s", out)
    log.info("\n%s", split_summary(df).to_string())
    return result


def run_features(config: Config = Config()) -> pd.DataFrame:
    """Step 2: read cleaned transactions, build point-in-time features, screen them on train."""
    from .features import INPUT_COLUMNS, build_features, feature_columns
    from .io import load_accounts
    from .metrics import univariate_screen

    paths = config.paths
    out = paths.processed_dir
    _require(out, "transactions.parquet")

    tx = pd.read_parquet(out / "transactions.parquet", columns=INPUT_COLUMNS)
    accounts = load_accounts(paths.accounts_csv) if paths.accounts_csv.exists() else None
    if accounts is None:
        log.warning("No accounts file at %s; entity features skipped", paths.accounts_csv)

    feats = build_features(tx, accounts)
    cols = feature_columns(feats)
    feats.to_parquet(out / "features.parquet", index=False)
    write_json({"features": cols}, out / "feature_list.json")

    train = feats[feats["split"] == "train"]
    screen = univariate_screen(train, cols)
    screen.to_csv(out / "feature_screening_train.csv")
    log.info("Wrote %d features to %s (train base rate %.4f%%)", len(cols), out / "features.parquet",
             100 * screen.attrs["base_rate"])
    log.info("Top features on train by PR-AUC:\n%s", screen.head(10).to_string())
    return feats


def run_baseline(config: Config = Config()):
    """Step 3: train the XGBoost baseline on step 2 features and write model, metrics and scores."""
    from .baseline import evaluate, load_splits, metrics_record, train_search

    paths, cfg = config.paths, config.baseline
    out = paths.processed_dir
    _require(out, "features.parquet")

    features = _read_json(out / "feature_list.json")["features"]
    ach = _read_json(out / "payment_formats.json")["ACH"]
    splits = load_splits(out / "features.parquet", features, cfg)
    model, params, search = train_search(splits, seed=cfg.seed)
    result = evaluate(model, params, search, splits, ach, _load_patterns(out), cfg)

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(paths.models_dir / "xgb_baseline.json")
    write_json(metrics_record(result, splits, cfg), paths.reports_dir / "metrics_step3_baseline.json")
    _write_scores(splits, out / "baseline_scores.parquet", score=(result.score_val, result.score_test))

    t = result.test
    log.info("Selected %s (%d trees)", params, model.best_iteration + 1)
    log.info("Test PR-AUC %.4f (random %.4f), ROC-AUC %.4f; at the 1%% budget: recall %.1f%%, "
             "realised FPR %.2f%%", t["pr_auc"], t["random_pr_auc"], t["roc_auc"],
             100 * t["recall_at_1pct"], 100 * t["fpr_at_1pct"])
    log.info("Wrote %s, %s and %s", paths.models_dir / "xgb_baseline.json",
             paths.reports_dir / "metrics_step3_baseline.json", out / "baseline_scores.parquet")
    return result


def run_graph_features(config: Config = Config()) -> pd.DataFrame:
    """Step 4a: point-in-time graph features, screened on train like step 2."""
    from .features import feature_columns
    from .graph_features import INPUT_COLUMNS, build_graph_features, find_hubs
    from .metrics import univariate_screen

    out = config.paths.processed_dir
    _require(out, "transactions.parquet")

    tx = pd.read_parquet(out / "transactions.parquet", columns=INPUT_COLUMNS)
    hubs = find_hubs(tx, config.graph.hub_min_degree)
    feats = build_graph_features(tx, hubs)
    cols = feature_columns(feats)
    feats.to_parquet(out / "graph_features.parquet", index=False)
    write_json({"features": cols, "hubs": hubs.tolist(), "hub_min_degree": config.graph.hub_min_degree},
               out / "graph_feature_list.json")

    screen = univariate_screen(feats[feats["split"] == "train"], cols)
    screen.to_csv(out / "graph_feature_screening_train.csv")
    log.info("Wrote %d graph features to %s", len(cols), out / "graph_features.parquet")
    log.info("Top graph features on train by PR-AUC:\n%s", screen.head(10).to_string())
    return feats


# Earlier models a new one can be compared with: (scores file in data/processed, metrics file in reports).
REFERENCES = {
    "baseline": ("baseline_scores.parquet", "metrics_step3_baseline.json"),
    "graph": ("graph_scores.parquet", "metrics_step4_graph.json"),
}


def compare_with_baseline(test: pd.DataFrame, y_test: np.ndarray, score: np.ndarray, threshold: float,
                          config: Config = Config(), reference: str = "baseline") -> pd.DataFrame:
    """Paired block-bootstrap comparison with an earlier model on the same test rows.

    `test` holds `tx_id` and `timestamp` of the test split in model order; one-hour blocks are
    resampled. `reference` is a key of REFERENCES (the step 3 baseline by default).
    """
    from .metrics import paired_bootstrap

    paths, cfg = config.paths, config.baseline
    scores_file, metrics_file = REFERENCES[reference]
    base = pd.read_parquet(paths.processed_dir / scores_file)
    base = base[base["split"] == "test"]
    if not np.array_equal(base["tx_id"].to_numpy(), test["tx_id"].to_numpy()):
        raise ValueError(f"{scores_file} does not match the test split; rerun that step")
    base_thr = _read_json(paths.reports_dir / metrics_file)["thresholds_from_val"][str(cfg.headline_fpr)]
    hour = test["timestamp"].dt.floor("h").to_numpy()
    return paired_bootstrap(y_test, base["score"].to_numpy(), score, hour, base_thr, threshold,
                            n_boot=config.graph.n_bootstrap, seed=cfg.seed, fpr=cfg.headline_fpr)


def run_graph_model(config: Config = Config()):
    """Step 4b: the step 3 model recipe on step 2 + graph features, compared with the baseline."""
    from .baseline import evaluate, load_splits, metrics_record, train_search

    paths, cfg = config.paths, config.baseline
    out = paths.processed_dir
    _require(out, "features.parquet", "graph_features.parquet", "baseline_scores.parquet")

    features = _read_json(out / "feature_list.json")["features"]
    graph_cols = _read_json(out / "graph_feature_list.json")["features"]
    ach = _read_json(out / "payment_formats.json")["ACH"]
    splits = load_splits(out / "features.parquet", features, cfg, extra=[(out / "graph_features.parquet", graph_cols)])
    model, params, search = train_search(splits, seed=cfg.seed)
    result = evaluate(model, params, search, splits, ach, _load_patterns(out), cfg)

    comparison = compare_with_baseline(splits.meta["test"], splits.y_test, result.score_test,
                                       result.thresholds[cfg.headline_fpr], config)
    base_metrics = _read_json(paths.reports_dir / "metrics_step3_baseline.json")

    record = metrics_record(result, splits, cfg, name="XGBoost, step 2 + graph features")
    record["n_graph_features"] = len(graph_cols)
    record["gain_over_baseline_test"] = comparison.to_dict(orient="index")
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(paths.models_dir / "xgb_graph.json")
    write_json(record, paths.reports_dir / "metrics_step4_graph.json")
    _write_scores(splits, out / "graph_scores.parquet", score=(result.score_val, result.score_test))

    t = result.test
    log.info("Selected %s (%d trees)", params, model.best_iteration + 1)
    log.info("Test PR-AUC %.4f (baseline %.4f), recall at the 1%% budget %.1f%% (baseline %.1f%%)",
             t["pr_auc"], base_metrics["test"]["pr_auc"], 100 * t["recall_at_1pct"],
             100 * base_metrics["test"]["recall_at_1pct"])
    log.info("Gain over baseline on test (95%% block-bootstrap interval):\n%s", comparison.to_string())
    return result, comparison


def run_gnn(config: Config = Config(), max_epochs: int | None = None):
    """Step 5: GraphSAGE on daily snapshots + the step 2/4 features, an ablation without message
    passing, and an ensemble with the step 4 model; all compared with steps 3 and 4."""
    from dataclasses import replace

    import torch

    from .baseline import evaluate_scores, load_splits, metrics_record
    from .gnn import (NODE_FEATURES, TabularScaler, build_snapshots, combine_scores, make_edge_set,
                      predict, train_gnn)

    paths, cfg = config.paths, config.baseline
    gcfg = config.gnn if max_epochs is None else replace(config.gnn, max_epochs=max_epochs)
    out = paths.processed_dir
    _require(out, "features.parquet", "graph_features.parquet", "baseline_scores.parquet", "graph_scores.parquet")

    features = _read_json(out / "feature_list.json")["features"]
    graph_info = _read_json(out / "graph_feature_list.json")
    ach = _read_json(out / "payment_formats.json")["ACH"]
    patterns = _load_patterns(out)

    tx = pd.read_parquet(out / "transactions.parquet",
                         columns=["tx_id", "timestamp", "from_idx", "to_idx", "amount_paid_usd", "is_self_loop"])
    day0 = tx["timestamp"].min().normalize()
    n_days = int((tx["timestamp"].max().normalize() - day0) / pd.Timedelta(days=1)) + 1
    snaps = build_snapshots(tx, np.array(graph_info["hubs"]), n_days)
    nodes = tx.set_index("tx_id")[["from_idx", "to_idx"]]
    del tx

    # Same rows, sample and tabular inputs as the step 4 model.
    splits = load_splits(out / "features.parquet", features, cfg,
                         extra=[(out / "graph_features.parquet", graph_info["features"])])
    scaler = TabularScaler().fit(splits.X_train)
    sets = {name: make_edge_set(X, splits.meta[name], nodes, scaler, day0, y) for name, X, y in [
        ("train", splits.X_train, splits.y_train), ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test)]}
    splits.X_train = splits.X_val = splits.X_test = None
    del nodes

    graph_scores = pd.read_parquet(out / "graph_scores.parquet")
    p_graph = {s: graph_scores.loc[graph_scores["split"] == s, "score"].to_numpy() for s in ("val", "test")}

    models, scores, histories = {}, {}, {}
    for name, mp in [("gnn", True), ("no_message_passing", False)]:
        log.info("Training %s", name)
        models[name], histories[name] = train_gnn(snaps, sets["train"], sets["val"], gcfg, message_passing=mp)
        scores[name] = {s: predict(models[name], snaps, sets[s]) for s in ("val", "test")}
    scores["ensemble"] = {s: combine_scores(scores["gnn"][s], p_graph[s]) for s in ("val", "test")}

    labels = {"gnn": "GraphSAGE (daily snapshots) + step 2/4 features",
              "no_message_passing": "Same network without message passing (ablation)",
              "ensemble": "Average log-odds of GraphSAGE and the step 4 XGBoost"}
    report, test_meta = {}, splits.meta["test"]
    for name, sc in scores.items():
        if name == "ensemble":
            params = {"members": ["gnn", "graph"]}
        else:
            params = {**vars(gcfg), "message_passing": name == "gnn", "node_features": NODE_FEATURES}
        epochs = histories.get(name, pd.DataFrame())
        result = evaluate_scores(sc["val"], sc["test"], models.get(name), params, epochs, splits, ach, patterns, cfg)
        rec = metrics_record(result, splits, cfg, name=labels[name])
        rec["search"] = [{"epoch": int(i), **{k: float(v) for k, v in r.items()}} for i, r in epochs.iterrows()]
        thr = result.thresholds[cfg.headline_fpr]
        for key, ref in [("vs_baseline_test", "baseline"), ("vs_graph_model_test", "graph")]:
            rec[key] = compare_with_baseline(test_meta, splits.y_test, sc["test"], thr, config,
                                             ref).to_dict(orient="index")
        report[name] = rec
        log.info("%s: test PR-AUC %.4f, recall at 1%% budget %.1f%%, PR-AUC vs step 4 %+.4f",
                 name, rec["test"]["pr_auc"], 100 * rec["test"]["recall_at_1pct"],
                 rec["vs_graph_model_test"]["pr_auc"]["difference"])

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": models["gnn"].state_dict(), "config": vars(gcfg),
                "scaler": {"numeric": scaler.numeric, "mean": scaler.mean, "std": scaler.std,
                           "categories": scaler.categories}}, paths.models_dir / "gnn.pt")
    write_json(report, paths.reports_dir / "metrics_step5_gnn.json")
    _write_scores(splits, out / "gnn_scores.parquet",
                  **{f"score_{name}": (sc["val"], sc["test"]) for name, sc in scores.items()})
    log.info("Wrote %s, %s and %s", paths.models_dir / "gnn.pt", paths.reports_dir / "metrics_step5_gnn.json",
             out / "gnn_scores.parquet")
    return report


def run_unsupervised(config: Config = Config()):
    """Step 6: label-free scores (Isolation Forest, communities), measured against labels and patterns."""
    from .baseline import Splits, evaluate_scores, metrics_record
    from .metrics import ranking_metrics, recall_at_fpr
    from .unsupervised import (combine, community_scores, fit_transaction_forest, group_recovery,
                               label_free_threshold, leiden_partition, transaction_anomaly)

    paths, cfg = config.paths, config.baseline
    out = paths.processed_dir
    _require(out, "features.parquet", "graph_features.parquet", "pattern_transactions.parquet")
    features = _read_json(out / "feature_list.json")["features"]
    graph_info = _read_json(out / "graph_feature_list.json")
    hubs = np.array(graph_info["hubs"])
    ach = _read_json(out / "payment_formats.json")["ACH"]
    patterns = _load_patterns(out)

    tx = pd.read_parquet(out / "transactions.parquet",
                         columns=["tx_id", "timestamp", "from_idx", "to_idx", "is_self_loop", "is_laundering", "split"])
    comm = community_scores(tx, hubs, cfg.seed)

    # Isolation Forest on step 2 + step 4 features, fitted on a uniform sample of train (labels ignored).
    cols = features + graph_info["features"]
    X = pd.read_parquet(out / "features.parquet", columns=features)
    graph = pd.read_parquet(out / "graph_features.parquet", columns=["tx_id"] + graph_info["features"])
    if not np.array_equal(graph["tx_id"].to_numpy(), tx["tx_id"].to_numpy()):
        raise ValueError("graph_features.parquet rows do not line up with transactions.parquet")
    for c in graph_info["features"]:
        X[c] = graph[c].to_numpy()
    del graph
    split = tx["split"].astype(str).to_numpy()
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(np.flatnonzero(split == "train"), 200_000, replace=False)
    forest = fit_transaction_forest(X.iloc[sample][cols], cfg.seed)
    iso = transaction_anomaly(forest, X[cols])
    fmt = X["payment_format_code"].to_numpy()
    del X

    raw = {"isolation_forest": iso, "cross_community": comm.cross_community.astype(np.float64),
           "community_anomaly": comm.community_anomaly}
    val = split == "val"
    scores = {**raw, "combined": combine({k: v[val] for k, v in raw.items()}, raw)}

    # Same evaluation as steps 3-5, but alert thresholds flag a share of ALL validation payments.
    y = tx["is_laundering"].to_numpy()
    meta_cols = ["tx_id", "timestamp", "is_laundering"]
    meta = {s: tx.loc[split == s, meta_cols].reset_index(drop=True).assign(payment_format_code=fmt[split == s])
            for s in ("val", "test")}
    splits = Splits(features=cols, X_train=None, y_train=None, X_val=None, y_val=y[split == "val"],
                    X_test=None, y_test=y[split == "test"], meta=meta, overview=None)
    report = {}
    for name in ("isolation_forest", "community_anomaly", "combined"):
        sv, st = scores[name][split == "val"], scores[name][split == "test"]
        thresholds = {f: label_free_threshold(sv, f) for f in cfg.fpr_budgets}
        result = evaluate_scores(sv, st, None, {"label_free": True}, pd.DataFrame(), splits, ach, patterns, cfg,
                                 thresholds=thresholds)
        rec = metrics_record(result, splits, cfg, name=name)
        rec["test"]["recall_at_equal_fpr_1pct"] = recall_at_fpr(splits.y_test, st, 0.01)
        report[name] = rec
        log.info("%s: test PR-AUC %.4f, ROC-AUC %.3f, recall flagging 1%% of val volume %.1f%%",
                 name, rec["test"]["pr_auc"], rec["test"]["roc_auc"], 100 * rec["test"]["recall_at_1pct"])

    # Crossing communities is yes/no: report how much it concentrates laundering.
    cross = comm.cross_community == 1
    report["cross_community"] = {
        s: {"share_of_payments_crossing": float(cross[split == s].mean()),
            "laundering_rate_crossing": float(y[(split == s) & cross].mean()),
            "laundering_rate_within": float(y[(split == s) & ~cross].mean()),
            "share_of_laundering_crossing": float(cross[(split == s) & (y == 1)].mean()),
            **ranking_metrics(y[split == s], cross[split == s].astype(float))}
        for s in ("train", "val", "test")}

    # Every day can be evaluated: nothing here was fitted on labels.
    thr = label_free_threshold(scores["combined"][val], cfg.headline_fpr)
    flagged = pd.Series(scores["combined"] >= thr, index=tx["tx_id"].to_numpy())
    pat = patterns.dropna(subset=["tx_id"]).copy()
    pat["tx_id"] = pat["tx_id"].astype("int64")
    pat["flagged"] = flagged.reindex(pat["tx_id"]).to_numpy()
    by_attempt = pat.groupby(["pattern_type", "attempt_id"])["flagged"].agg(["any", "mean"])
    report["patterns_all_days"] = by_attempt.groupby("pattern_type").agg(
        attempts=("any", "size"), attempts_with_any_alert=("any", "mean"),
        transaction_recall=("mean", "mean")).to_dict(orient="index")

    # Collusion groups: after one alert, do communities recover the rest of the attempt?
    src, dst = tx["from_idx"].to_numpy(np.int64), tx["to_idx"].to_numpy(np.int64)
    usable = (tx["is_self_loop"].to_numpy() == 0) & ~np.isin(src, hubs) & ~np.isin(dst, hubs)
    full_member = leiden_partition(src[usable], dst[usable], int(max(src.max(), dst.max())) + 1, cfg.seed)
    full_member[hubs] = -1
    attempts = pat.drop(columns="flagged").merge(
        tx[["tx_id", "timestamp", "from_idx", "to_idx"]], on="tx_id")
    recovery = group_recovery(attempts, comm.member_by_day, full_member, tx["timestamp"].min().normalize())
    report["group_recovery"] = recovery.to_dict(orient="index")

    write_json(report, paths.reports_dir / "metrics_step6_unsupervised.json")
    tx[["tx_id", "split", "is_laundering"]].assign(**{f"score_{k}": v for k, v in scores.items()}).to_parquet(
        out / "unsupervised_scores.parquet", index=False)
    log.info("Group recovery by pattern type:\n%s", recovery.to_string())
    log.info("Wrote %s and %s", paths.reports_dir / "metrics_step6_unsupervised.json",
             out / "unsupervised_scores.parquet")
    return report


def run_build_store(config: Config = Config()):
    """Step 7a: build the serving feature store from the full history and save it."""
    from .serving import FeatureStore

    path = config.paths.processed_dir / "feature_store.pkl"
    store = FeatureStore.build(config)
    store.save(path)
    log.info("Wrote %s (%.0f MB)", path, path.stat().st_size / 1e6)
    return store


def run_drift_report(config: Config = Config(), top_features: int = 10):
    """Step 7b: what the drift monitor would have reported, day by day, on validation and test.

    For the served model (step 4): each day's alert rate at the validation threshold (overall
    and on normal payments), the PSI of its scores against validation, and the PSI of the most
    important input features against the training period.
    """
    from .metrics import psi

    paths, cfg = config.paths, config.baseline
    out = paths.processed_dir
    metrics = _read_json(paths.reports_dir / "metrics_step4_graph.json")
    threshold = metrics["thresholds_from_val"][str(cfg.headline_fpr)]
    top = list(metrics["top_features_by_gain"])[:top_features]

    scores = pd.read_parquet(out / "graph_scores.parquet")
    graph_cols = _read_json(out / "graph_feature_list.json")["features"]
    feats = pd.read_parquet(out / "features.parquet",
                            columns=["tx_id", "timestamp", "split"] + [c for c in top if c not in graph_cols])
    for c in [c for c in top if c in graph_cols]:
        feats[c] = pd.read_parquet(out / "graph_features.parquet", columns=[c])[c].to_numpy()
    train = feats[feats["split"] == "train"].sample(500_000, random_state=cfg.seed)
    scored = feats.merge(scores[["tx_id", "score", "is_laundering"]], on="tx_id")
    reference = scored.loc[scored["split"] == "val", "score"].to_numpy()

    rows = []
    for day, g in scored.groupby(scored["timestamp"].dt.date):
        normal = g["is_laundering"] == 0
        flagged = g["score"] >= threshold
        rows.append({
            "day": str(day), "split": "/".join(sorted(g["split"].astype(str).unique())), "payments": len(g),
            "alert_rate": float(flagged.mean()), "fpr": float(flagged[normal].mean()),
            "score_psi_vs_val": psi(reference, g["score"].to_numpy()),
            **{f"psi_{c}": psi(train[c].to_numpy(dtype=float), g[c].to_numpy(dtype=float)) for c in top},
        })
    report = pd.DataFrame(rows).set_index("day")
    psi_cols = [f"psi_{c}" for c in top]
    report["max_feature_psi"] = report[psi_cols].max(axis=1)
    report["most_shifted_feature"] = report[psi_cols].idxmax(axis=1).str.removeprefix("psi_")
    report["expected_alert_rate"] = float((reference >= threshold).mean())
    report["status"] = np.where((report["alert_rate"] > 1.5 * report["expected_alert_rate"])
                                | (report["score_psi_vs_val"] > 0.2), "drift", "ok")
    report.to_csv(paths.reports_dir / "drift_report.csv")
    log.info("Daily drift report (served model, threshold %.4f):\n%s", threshold,
             report[["split", "payments", "alert_rate", "fpr", "score_psi_vs_val", "max_feature_psi",
                     "most_shifted_feature", "status"]].to_string())
    return report
