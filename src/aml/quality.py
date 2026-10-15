"""Data-quality checks and descriptive tables used by the notebook and the pipeline."""
from __future__ import annotations

import pandas as pd


def quality_report(df: pd.DataFrame) -> dict:
    """Headline checks on raw (renamed) transactions."""
    raw = df.drop(columns=["tx_id"], errors="ignore")
    return {
        "rows": int(len(df)),
        "nulls_per_column": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "nonpositive_amount_paid": int((df["amount_paid"] <= 0).sum()),
        "nonpositive_amount_received": int((df["amount_received"] <= 0).sum()),
        "time_min": str(df["timestamp"].min()),
        "time_max": str(df["timestamp"].max()),
        "laundering_rows": int(df["is_laundering"].sum()),
        "laundering_rate": float(df["is_laundering"].mean()),
    }


def daily_profile(df: pd.DataFrame, amount_col: str = "amount_paid_usd") -> pd.DataFrame:
    agg = {"tx": ("is_laundering", "size"), "laundering": ("is_laundering", "sum")}
    if amount_col in df.columns:
        agg["usd_volume"] = (amount_col, "sum")
    d = df.groupby(df["timestamp"].dt.date).agg(**agg)
    d.index.name = "date"
    d["laundering_rate"] = d["laundering"] / d["tx"]
    return d


def laundering_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["is_laundering"]
    out = pd.DataFrame({"tx": g.size(), "laundering": g.sum(), "laundering_rate": g.mean()})
    out["share_of_laundering"] = out["laundering"] / out["laundering"].sum()
    return out.sort_values("laundering_rate", ascending=False)
