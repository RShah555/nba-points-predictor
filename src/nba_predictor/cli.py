"""Command line entry point.

nba-predictor train [--trials 20] [--no-mlflow]   tune, evaluate and save the model
nba-predictor refresh                             pull the latest games, schedule and injuries
nba-predictor lines                               save today's real prop lines and grade old ones
nba-predictor serve [--port 8000]                 run the app (API + dashboard)
"""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nba-predictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="download data, tune, evaluate and save the model")
    train.add_argument("--trials", type=int, default=None, help="Optuna trials per stat (default from settings)")
    train.add_argument("--no-mlflow", action="store_true", help="skip MLflow experiment tracking")
    sub.add_parser("refresh", help="pull the latest games, schedule and injury report")
    sub.add_parser("lines", help="snapshot today's real prop lines and grade finished ones (needs NBA_ODDS_API_KEY)")
    serve = sub.add_parser("serve", help="run the app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "train":
        from .train import main as train_main

        train_main(trials=args.trials, use_mlflow=not args.no_mlflow)
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("nba_predictor.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
        return

    from .config import get_settings
    from .store import LeagueStore

    settings = get_settings()
    store = LeagueStore(settings)
    store.load()
    if args.command == "refresh":
        print(f"{len(store.raw):,} player games for {', '.join(store.seasons)}; {len(store.injuries)} on injury report")
        return

    from . import lines
    from .modeling import ModelBundle
    from .predictor import Predictor

    if not settings.odds_api_key:
        raise SystemExit("Set NBA_ODDS_API_KEY first (free key at https://the-odds-api.com).")
    predictor = Predictor(ModelBundle.load(settings.bundle_path), store, settings)
    props = lines.fetch_props(settings.odds_api_key, settings.odds_markets)
    ledger = lines.grade(lines.snapshot(predictor, props), store.norm)
    ledger.to_csv(settings.ledger_path, index=False)
    print(f"Saved {len(props)} props. Record so far: {lines.summary(ledger)}")


if __name__ == "__main__":
    main()
