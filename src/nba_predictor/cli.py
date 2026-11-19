"""Command line entry point.

nba-predictor train [--trials 20] [--no-mlflow]
nba-predictor refresh
nba-predictor serve [--port 8000] [--reload]
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
    sub.add_parser("refresh", help="download the latest games and schedule for the season in progress")
    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "train":
        from .train import main as train_main

        train_main(trials=args.trials, use_mlflow=not args.no_mlflow)
    elif args.command == "refresh":
        from .config import get_settings
        from .store import LeagueStore

        settings = get_settings()
        settings.live_refresh_hours = 0  # force a download
        store = LeagueStore(settings)
        store.load()
        print(f"Loaded {len(store.raw):,} player games for {', '.join(store.seasons)}; {len(store.players):,} players")
    else:
        import uvicorn

        uvicorn.run("nba_predictor.api:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
