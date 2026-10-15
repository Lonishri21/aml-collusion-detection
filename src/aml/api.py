"""Step 7: HTTP API for scoring payments and checking drift.

Run with `python main.py serve` (or `uvicorn aml.api:app`). The feature store is read from
`AML_FEATURE_STORE` (default: data/processed/feature_store.pkl, written by `python main.py build-store`).

    GET  /health       store and model in use
    POST /score        score up to 10,000 payments: score, alert, top features
    GET  /monitoring   recent alert rate and score drift against validation
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config
from .serving import DriftMonitor, Scorer


class Payment(BaseModel):
    timestamp: datetime
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_paid: float = Field(ge=0)
    payment_currency: str
    amount_received: float = Field(ge=0)
    receiving_currency: str
    payment_format: str


class ScoreRequest(BaseModel):
    payments: list[Payment] = Field(min_length=1, max_length=10_000)


class ScoredPayment(BaseModel):
    score: float
    alert: bool
    top_features: list[str]
    known_sender: bool
    known_receiver: bool


class ScoreResponse(BaseModel):
    threshold: float
    results: list[ScoredPayment]


def load_reference(config: Config = Config()) -> np.ndarray:
    """Validation scores of the served model: what the alert threshold was set on."""
    s = pd.read_parquet(config.paths.processed_dir / "graph_scores.parquet", columns=["split", "score"])
    return s.loc[s["split"] == "val", "score"].to_numpy()


def create_app(scorer: Scorer | None = None, reference: np.ndarray | None = None) -> FastAPI:
    """App factory; tests pass a small scorer, production loads everything at startup."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = Config()
        store_path = Path(os.environ.get("AML_FEATURE_STORE", config.paths.processed_dir / "feature_store.pkl"))
        state["scorer"] = scorer or Scorer.load(store_path, config)
        ref = reference if reference is not None else load_reference(config)
        state["monitor"] = DriftMonitor(ref, state["scorer"].threshold, state["scorer"].budget)
        yield
        state.clear()

    app = FastAPI(title="AML collusion detection", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        s = state["scorer"]
        return {"status": "ok", "model": "XGBoost, step 2 + graph features", "threshold": s.threshold,
                "alert_budget": s.budget, "history_end": str(s.store.history_end),
                "scores_until": str(s.store.horizon_end), "accounts": len(s.store.node_index)}

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        s = state["scorer"]
        raw = pd.DataFrame([p.model_dump() for p in request.payments])
        try:
            out = s.score(raw)
        except ValueError as e:                 # unknown currency/format, beyond the store's horizon
            raise HTTPException(status_code=422, detail=str(e)) from e
        state["monitor"].update(out["score"].to_numpy())
        return ScoreResponse(threshold=s.threshold, results=[
            ScoredPayment(score=float(r.score), alert=bool(r.alert), top_features=list(r.top_features),
                          known_sender=bool(r.known_sender), known_receiver=bool(r.known_receiver))
            for r in out.itertuples()])

    @app.get("/monitoring")
    def monitoring() -> dict:
        return state["monitor"].report()

    return app


app = create_app()
