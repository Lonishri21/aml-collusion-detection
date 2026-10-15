"""Cleaning steps: node identity, currency conversion, flags and encodings.

Every function takes a DataFrame and returns a new or updated one, so the same steps
run identically in the notebook and in the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ dates
def trim_after(df: pd.DataFrame, max_date: str | None) -> pd.DataFrame:
    """Drop transactions on or after `max_date` (YYYY-MM-DD). No-op when None."""
    if not max_date:
        return df
    keep = df["timestamp"] < pd.Timestamp(max_date)
    log.info("Trimming %s transactions on/after %s", f"{(~keep).sum():,}", max_date)
    return df.loc[keep].copy()


# ------------------------------------------------------------------ node identity
def normalize_bank(s: pd.Series) -> pd.Series:
    """Canonical bank code: numeric codes lose leading zeros ('010' -> '10').

    The transactions file writes bank codes with leading zeros, the accounts file
    without, so both must be normalized before joining or building node IDs.
    """
    s = s.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    out = s.copy()
    ok = num.notna()
    out[ok] = num[ok].astype("int64").astype(str)
    return out


def node_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Account identifier used as a graph node. Account numbers repeat across banks,
    so the node is the (bank, account) pair."""
    return normalize_bank(bank) + "_" + account.astype(str).str.strip()


def add_node_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `from_node`/`to_node` keys and integer `from_idx`/`to_idx`.

    Returns the updated transactions and a node table (`node_id`, `node_idx`) whose
    index order is what graph libraries such as PyTorch Geometric will use.
    """
    df = df.copy()
    df["from_bank"] = normalize_bank(df["from_bank"])
    df["to_bank"] = normalize_bank(df["to_bank"])
    df["from_node"] = node_key(df["from_bank"], df["from_account"])
    df["to_node"] = node_key(df["to_bank"], df["to_account"])

    index = pd.Index(pd.unique(np.concatenate([df["from_node"].to_numpy(), df["to_node"].to_numpy()])))
    nodes = pd.DataFrame({"node_id": index, "node_idx": np.arange(len(index), dtype=np.int64)})
    df["from_idx"] = index.get_indexer(df["from_node"])
    df["to_idx"] = index.get_indexer(df["to_node"])
    df["from_node"] = df["from_node"].astype("string")
    df["to_node"] = df["to_node"].astype("string")
    log.info("Built %s nodes from bank+account pairs", f"{len(nodes):,}")
    return df, nodes


# ------------------------------------------------------------------ currency
def estimate_fx_rates(df: pd.DataFrame, base: str = "US Dollar") -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate '1 unit of X = r base' from the data itself.

    The dataset ships no FX table, but each cross-currency row implies a rate
    (amount_received / amount_paid). Rows with the base currency on one side give a
    direct rate; the median is used per currency. The diagnostics table reports how
    many rows back each rate and how much it varies (coefficient of variation).
    """
    pay = df["pay_currency"].astype(str)
    recv = df["recv_currency"].astype(str)
    cross = pay != recv

    to_base = cross & (recv == base)       # pay X, receive base
    from_base = cross & (pay == base)      # pay base, receive X
    implied = pd.concat([
        pd.DataFrame({"currency": pay[to_base],
                      "rate": df.loc[to_base, "amount_received"] / df.loc[to_base, "amount_paid"]}),
        pd.DataFrame({"currency": recv[from_base],
                      "rate": df.loc[from_base, "amount_paid"] / df.loc[from_base, "amount_received"]}),
    ])
    implied = implied[np.isfinite(implied["rate"]) & (implied["rate"] > 0)]

    diag = implied.groupby("currency")["rate"].agg(rate="median", std="std", n_rows="count")
    diag["coef_of_variation"] = diag["std"] / diag["rate"]
    diag = diag.drop(columns="std").sort_index()

    rates = {base: 1.0, **diag["rate"].to_dict()}
    missing = sorted((set(pay) | set(recv)) - set(rates))
    if missing:
        raise ValueError(f"No direct {base} pairs to estimate a rate for: {missing}")
    return rates, diag


def add_usd_amounts(df: pd.DataFrame, rates: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["amount_paid_usd"] = df["amount_paid"] * df["pay_currency"].astype(str).map(rates).astype("float64")
    df["amount_received_usd"] = df["amount_received"] * df["recv_currency"].astype(str).map(rates).astype("float64")
    return df


# ------------------------------------------------------------------ flags / encodings
def add_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Binary flags plus an integer code for payment format (mapping returned)."""
    df = df.copy()
    formats = sorted(df["payment_format"].astype(str).unique())
    format_codes = {f: i for i, f in enumerate(formats)}
    df["payment_format_code"] = df["payment_format"].astype(str).map(format_codes).astype("int8")
    df["is_cross_currency"] = (df["pay_currency"].astype(str) != df["recv_currency"].astype(str)).astype("int8")
    df["is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype("int8")
    df["is_self_loop"] = (df["from_node"] == df["to_node"]).astype("int8")
    return df, format_codes
