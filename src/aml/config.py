"""Project paths and pipeline settings in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/aml/config.py -> project root is two levels above the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    # AML_PROCESSED_DIR lets the scoring container read a mounted data folder.
    processed_dir: Path = Path(os.environ.get("AML_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"
    dataset: str = "HI-Small"

    @property
    def transactions_csv(self) -> Path:
        return self.raw_dir / f"{self.dataset}_Trans.csv"

    @property
    def accounts_csv(self) -> Path:
        return self.raw_dir / f"{self.dataset}_accounts.csv"

    @property
    def patterns_txt(self) -> Path:
        return self.raw_dir / f"{self.dataset}_Patterns.txt"


@dataclass(frozen=True)
class CleaningConfig:
    # HI-Small: 11-18 Sep hold ~1.1k transactions, ~60% laundering (patterns finishing
    # after normal traffic stops). Keeping them would put them all in the test set.
    max_date: str | None = "2022-09-11"
    base_currency: str = "US Dollar"
    train_frac: float = 0.6
    val_frac: float = 0.2


@dataclass(frozen=True)
class BaselineConfig:
    # Share of normal TRAINING transactions kept (all laundering is kept). Val and test are never sampled.
    neg_sample_rate: float = 0.2
    seed: int = 42
    # Alert budgets: share of normal payments flagged, thresholds fixed on validation.
    fpr_budgets: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)
    headline_fpr: float = 0.01


@dataclass(frozen=True)
class GraphConfig:
    # Accounts with this many distinct counterparties in the train split are hubs: not used as
    # path intermediaries or in components. HI-Small has 11, sending ~9% of payments.
    hub_min_degree: int = 1000
    # Hour blocks resampled to put a confidence interval on the gain over the baseline.
    n_bootstrap: int = 200


@dataclass(frozen=True)
class GNNConfig:
    hidden: int = 64
    layers: int = 2               # message-passing layers = hops of neighbourhood seen
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    chunk_rows: int = 2048        # sampled training transactions per step
    max_epochs: int = 30
    patience: int = 4             # epochs without validation PR-AUC improvement before stopping
    seed: int = 42


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
