"""Linking the Patterns file to cleaned transactions."""
from __future__ import annotations

import pandas as pd

from .cleaning import node_key
from .logging_utils import get_logger

log = get_logger(__name__)

MATCH_KEYS = ["timestamp", "from_node", "to_node", "amount_paid", "pay_currency", "payment_format"]


def match_patterns(transactions: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Attach `tx_id` from the cleaned transactions to each pattern row.

    Matching uses timestamp, both node keys, amount, currency and payment format.
    Rows with no match (e.g. outside the trimmed date range) get `tx_id = NA`.
    When several transactions share the same key, the first is used.
    """
    p = patterns.copy()
    p["from_node"] = node_key(p["from_bank"], p["from_account"])
    p["to_node"] = node_key(p["to_bank"], p["to_account"])

    t = transactions[MATCH_KEYS + ["tx_id"]].copy()
    # Align dtypes on both sides (categories/strings would otherwise refuse to merge).
    for col in ("from_node", "to_node", "pay_currency", "payment_format"):
        p[col] = p[col].astype(str)
        t[col] = t[col].astype(str)
    t = t.drop_duplicates(subset=MATCH_KEYS, keep="first")

    out = p.merge(t, on=MATCH_KEYS, how="left")
    out["tx_id"] = out["tx_id"].astype("Int64")
    log.info("Matched %d of %d pattern rows to transactions", out["tx_id"].notna().sum(), len(out))
    return out


def pattern_coverage(transactions: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """How much of the labelled laundering the Patterns file explains."""
    in_pattern = transactions["tx_id"].isin(matched["tx_id"].dropna())
    laundering = transactions["is_laundering"] == 1
    return {
        "laundering_tx": int(laundering.sum()),
        "laundering_tx_in_patterns": int((laundering & in_pattern).sum()),
        "laundering_tx_without_pattern": int((laundering & ~in_pattern).sum()),
        "pattern_tx_not_labelled_laundering": int((~laundering & in_pattern).sum()),
    }
