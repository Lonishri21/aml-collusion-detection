
from __future__ import annotations

import argparse
import time
from dataclasses import replace

from aml.config import BaselineConfig, CleaningConfig, Config, GraphConfig, Paths
from aml.logging_utils import get_logger
from aml.pipeline import (run_baseline, run_build_store, run_cleaning, run_drift_report, run_features, run_gnn,
                          run_graph_features, run_graph_model, run_unsupervised)

log = get_logger("main")


STEPS = {
    "clean": run_cleaning,
    "features": run_features,
    "baseline": run_baseline,
    "graph-features": run_graph_features,
    "graph-model": run_graph_model,
    "gnn": run_gnn,
    "unsupervised": run_unsupervised,
    "build-store": run_build_store,
    "monitor": run_drift_report,
}


def run_all(config: Config, start: str = "clean", skip_gnn: bool = False) -> None:
    """Run the steps from `start` to the end with default settings, stopping at the first failure."""
    names = list(STEPS)[list(STEPS).index(start):]
    if skip_gnn and "gnn" in names:
        names.remove("gnn")
    durations = {}
    for i, name in enumerate(names, 1):
        log.info("=== step %d/%d: %s ===", i, len(names), name)
        t0 = time.time()
        try:
            STEPS[name](config)
        except Exception:
            log.error("Step %r failed. Fix the error above, then resume with: python main.py all --from %s",
                      name, name)
            raise
        durations[name] = (time.time() - t0) / 60
    log.info("All steps done:\n%s", "\n".join(f"  {n:<15} {m:6.1f} min" for n, m in durations.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AML collusion detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="Step 1: load, validate, clean and split transactions")
    clean.add_argument("--dataset", default="HI-Small", help="File prefix in data/raw (default: HI-Small)")
    clean.add_argument("--max-date", default=CleaningConfig.max_date,
                       help="Drop transactions on/after this date (YYYY-MM-DD), or 'none'")
    clean.add_argument("--train-frac", type=float, default=CleaningConfig.train_frac)
    clean.add_argument("--val-frac", type=float, default=CleaningConfig.val_frac)

    features = sub.add_parser("features", help="Step 2: point-in-time account features")
    features.add_argument("--dataset", default="HI-Small", help="File prefix in data/raw (default: HI-Small)")

    baseline = sub.add_parser("baseline", help="Step 3: XGBoost baseline on the step 2 features")
    baseline.add_argument("--neg-sample-rate", type=float, default=BaselineConfig.neg_sample_rate,
                          help="Share of normal training transactions kept (default: 0.2)")

    graph = sub.add_parser("graph-features", help="Step 4a: point-in-time graph features")
    graph.add_argument("--hub-min-degree", type=int, default=GraphConfig.hub_min_degree,
                       help="Distinct counterparties in train that make an account a hub (default: 1000)")
    sub.add_parser("graph-model", help="Step 4b: XGBoost on step 2 + graph features vs the baseline")

    gnn = sub.add_parser("gnn", help="Step 5: GraphSAGE vs steps 3-4 (needs torch, torch_geometric)")
    gnn.add_argument("--max-epochs", type=int, default=None, help="Override the epoch limit (default: 30)")

    sub.add_parser("unsupervised", help="Step 6: label-free detection (Isolation Forest, communities)")

    sub.add_parser("build-store", help="Step 7: build the serving feature store from the full history")
    sub.add_parser("monitor", help="Step 7: daily drift report of the served model on validation and test")
    serve = sub.add_parser("serve", help="Step 7: run the scoring API (needs build-store first)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    everything = sub.add_parser("all", help="Run every step in order with default settings (~2 hours)")
    everything.add_argument("--dataset", default="HI-Small", help="File prefix in data/raw (default: HI-Small)")
    everything.add_argument("--from", dest="start", choices=list(STEPS), default="clean",
                            help="Start from this step, e.g. to resume after a failure (default: clean)")
    everything.add_argument("--skip-gnn", action="store_true", help="Skip step 5, the longest (needs torch)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "clean":
        max_date = None if str(args.max_date).lower() == "none" else args.max_date
        config = Config(
            paths=replace(Paths(), dataset=args.dataset),
            cleaning=CleaningConfig(max_date=max_date, train_frac=args.train_frac, val_frac=args.val_frac),
        )
        run_cleaning(config)
    elif args.command == "features":
        run_features(Config(paths=replace(Paths(), dataset=args.dataset)))
    elif args.command == "baseline":
        run_baseline(Config(baseline=BaselineConfig(neg_sample_rate=args.neg_sample_rate)))
    elif args.command == "graph-features":
        run_graph_features(Config(graph=GraphConfig(hub_min_degree=args.hub_min_degree)))
    elif args.command == "graph-model":
        run_graph_model()
    elif args.command == "gnn":
        run_gnn(max_epochs=args.max_epochs)
    elif args.command == "unsupervised":
        run_unsupervised()
    elif args.command == "build-store":
        run_build_store()
    elif args.command == "monitor":
        run_drift_report()
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("aml.api:app", host=args.host, port=args.port)
    elif args.command == "all":
        run_all(Config(paths=replace(Paths(), dataset=args.dataset)), args.start, args.skip_gnn)


if __name__ == "__main__":
    main()
