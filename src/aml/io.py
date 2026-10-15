"""Reading the three raw files and writing processed artifacts."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .schema import (
    ACCOUNT_RENAME,
    RAW_DTYPES,
    RAW_TRANSACTION_COLUMNS,
    TIMESTAMP_FORMAT,
    TRANSACTION_RENAME,
)

log = get_logger(__name__)

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<type>[A-Z\-]+)(?::\s*(?P<detail>.*))?$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - (?P<type>[A-Z\-]+)")


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the raw transactions CSV with canonical column names and a stable `tx_id`.

    `tx_id` is the 0-based row position in the raw file, so any later table can be
    joined back to the original data.
    """
    path = Path(path)
    log.info("Reading %s", path.name)
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    if list(df.columns) != RAW_TRANSACTION_COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}: {list(df.columns)}")
    df = df.rename(columns=TRANSACTION_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
    log.info("Loaded %s transactions", f"{len(df):,}")
    return df


def load_accounts(path: str | Path) -> pd.DataFrame:
    """Load the accounts file (bank, account, owning entity)."""
    df = pd.read_csv(path, dtype=str).rename(columns=ACCOUNT_RENAME)
    missing = set(ACCOUNT_RENAME.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Accounts file is missing columns: {sorted(missing)}")
    return df


def load_patterns(path: str | Path) -> pd.DataFrame:
    """Parse the Patterns file into one row per laundering transaction.

    The file is a sequence of blocks:
        BEGIN LAUNDERING ATTEMPT - <TYPE>: <detail>
        <transaction rows in the same CSV layout as the transactions file>
        END LAUNDERING ATTEMPT - <TYPE>

    Returns the transaction columns (canonical names) plus `attempt_id`,
    `pattern_type`, `pattern_detail` and `attempt_complete` (False when the file
    ends before the block's END line). Malformed rows are skipped and counted.
    """
    rows: list[list[str]] = []
    meta: list[tuple[int, str, str]] = []
    completed: set[int] = set()
    attempt_id, ptype, detail = -1, None, ""
    malformed = 0

    with open(path, encoding="utf-8", newline="") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if m := _BEGIN.match(line):
                attempt_id += 1
                ptype, detail = m["type"], (m["detail"] or "").strip()
                continue
            if _END.match(line):
                completed.add(attempt_id)
                ptype = None
                continue
            fields = next(csv.reader([line]))
            if ptype is None or len(fields) != len(RAW_TRANSACTION_COLUMNS):
                malformed += 1
                continue
            rows.append(fields)
            meta.append((attempt_id, ptype, detail))

    if malformed:
        log.warning("Patterns file: skipped %d malformed or orphan line(s)", malformed)
    if attempt_id + 1 > len(completed):
        log.warning("Patterns file: %d attempt(s) have no END line (file may be truncated)",
                    attempt_id + 1 - len(completed))

    df = pd.DataFrame(rows, columns=RAW_TRANSACTION_COLUMNS).rename(columns=TRANSACTION_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    for col in ("amount_paid", "amount_received"):
        df[col] = df[col].astype("float64")
    df["is_laundering"] = df["is_laundering"].astype("int8")
    m = pd.DataFrame(meta, columns=["attempt_id", "pattern_type", "pattern_detail"])
    df = pd.concat([m, df], axis=1)
    df["attempt_complete"] = df["attempt_id"].isin(completed)
    log.info("Parsed %d attempts, %d transactions from %s",
             df["attempt_id"].nunique(), len(df), Path(path).name)
    return df


def write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
