"""Raw file schemas and the canonical column names used everywhere else."""

# The raw CSV has two columns literally named "Account"; pandas reads the second as "Account.1".
RAW_TRANSACTION_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]

TRANSACTION_RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "recv_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "pay_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

RAW_DTYPES = {
    "From Bank": str, "Account": str, "To Bank": str, "Account.1": str,
    "Receiving Currency": "category", "Payment Currency": "category",
    "Payment Format": "category", "Is Laundering": "int8",
    "Amount Received": "float64", "Amount Paid": "float64",
}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

ACCOUNT_RENAME = {
    "Bank Name": "bank_name",
    "Bank ID": "bank",
    "Account Number": "account",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
}

# Columns written to data/processed/transactions.parquet, in order.
CLEAN_COLUMNS = [
    "tx_id", "timestamp",
    "from_bank", "from_account", "to_bank", "to_account",
    "from_node", "to_node", "from_idx", "to_idx",
    "amount_paid", "pay_currency", "amount_received", "recv_currency",
    "amount_paid_usd", "amount_received_usd",
    "payment_format", "payment_format_code",
    "is_cross_currency", "is_cross_bank", "is_self_loop",
    "is_laundering", "split",
]
