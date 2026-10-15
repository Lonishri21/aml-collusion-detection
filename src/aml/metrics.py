"""Evaluation helpers shared by feature screening (step 2) and models (step 3+)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, max_fpr: float = 0.01) -> float:
    """Share of laundering caught while flagging at most `max_fpr` of normal transactions."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(tpr[fpr <= max_fpr].max())


def ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR-AUC (average precision) and ROC-AUC of a score."""
    return {"pr_auc": float(average_precision_score(y_true, score)),
            "roc_auc": float(roc_auc_score(y_true, score))}


def threshold_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Score cut-off that flags a share `fpr` of the negatives.

    Fixed on validation and applied unchanged to test, as a team would before deploying.
    """
    return float(np.quantile(score[y_true == 0], 1 - fpr))


def at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Recall, precision, realised FPR and alert counts when flagging `score >= threshold`."""
    flagged = score >= threshold
    pos = y_true == 1
    tp = int((flagged & pos).sum())
    fp = int((flagged & ~pos).sum())
    return {
        "recall": tp / max(int(pos.sum()), 1),
        "precision": tp / max(tp + fp, 1),
        "fpr": fp / max(int((~pos).sum()), 1),
        "alerts": tp + fp,
        "caught": tp,
    }


def paired_bootstrap(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, blocks: np.ndarray,
                     thr_a: float, thr_b: float, n_boot: int = 200, seed: int = 0,
                     fpr: float = 0.01) -> pd.DataFrame:
    """Difference B - A in PR-AUC, recall at equal FPR, and recall and alerts at fixed thresholds.

    Both models are scored on the same resampled rows. `blocks` groups rows (e.g. by hour)
    and whole blocks are resampled, because laundering arrives in bursts and rows close in time
    are not independent; this gives wider, more honest intervals than resampling rows.

    `recall_at_equal_fpr` compares both models at the same share of normal payments flagged, the
    fair ranking comparison. `recall_at_threshold` and `alerts_at_threshold` use each model's
    threshold from validation, i.e. what would happen in deployment, where the two models may
    flag different volumes.
    """
    def stats(idx):
        y = y_true[idx]
        at_a, at_b = at_threshold(y, score_a[idx], thr_a), at_threshold(y, score_b[idx], thr_b)
        return (average_precision_score(y, score_b[idx]) - average_precision_score(y, score_a[idx]),
                recall_at_fpr(y, score_b[idx], fpr) - recall_at_fpr(y, score_a[idx], fpr),
                at_b["recall"] - at_a["recall"],
                at_b["alerts"] - at_a["alerts"])

    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    draws = np.array([stats(np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]))
                      for _ in range(n_boot)])
    point = stats(np.arange(len(y_true)))
    return pd.DataFrame({
        "difference": point,
        "ci_low": np.quantile(draws, 0.025, axis=0),
        "ci_high": np.quantile(draws, 0.975, axis=0),
        "share_of_draws_above_0": (draws > 0).mean(axis=0),
    }, index=["pr_auc", "recall_at_equal_fpr", "recall_at_threshold", "alerts_at_threshold"])


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index of `current` against `reference`.

    Bins are quantiles of the reference, so each holds ~1/bins of it; PSI sums
    (cur% - ref%) * ln(cur% / ref%) over bins. Rule of thumb: < 0.1 stable, 0.1-0.2 moderate
    shift, > 0.2 significant shift. Missing values form their own bin.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    ref_ok, cur_ok = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1])) if len(ref_ok) else np.array([])

    def shares(all_values, ok):
        counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1)
        counts = np.append(counts, len(all_values) - len(ok))
        return counts / max(len(all_values), 1)

    eps = 1e-4
    ref, cur = shares(reference, ref_ok) + eps, shares(current, cur_ok) + eps
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def univariate_screen(df: pd.DataFrame, features: list[str], label: str = "is_laundering") -> pd.DataFrame:
    """How well each feature separates the classes on its own.

    `roc_auc` is direction-free (0.5 = useless, 1 = perfect) and `direction` says whether
    higher values mean more risk. `pr_auc` uses the better direction; compare it with the
    base rate (share of laundering), which is what a useless feature would score.
    Missing values are scored as the lowest value.
    """
    y = df[label].to_numpy()
    rows = []
    for col in features:
        x = df[col].astype("float64")
        x = x.fillna(x.min() - 1 if x.notna().any() else 0).to_numpy()
        if np.unique(x).size < 2:
            rows.append({"feature": col, "roc_auc": 0.5, "direction": "n/a", "pr_auc": y.mean()})
            continue
        auc = roc_auc_score(y, x)
        sign = 1 if auc >= 0.5 else -1
        rows.append({
            "feature": col,
            "roc_auc": max(auc, 1 - auc),
            "direction": "higher = riskier" if sign == 1 else "lower = riskier",
            "pr_auc": average_precision_score(y, sign * x),
        })
    out = pd.DataFrame(rows).set_index("feature").sort_values("pr_auc", ascending=False)
    out.attrs["base_rate"] = float(y.mean())
    return out
