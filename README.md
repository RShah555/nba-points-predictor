# NBA Predictor

Projects every NBA player's next game: points, rebounds, assists and more, each as a full range of
outcomes rather than one number. On top of that it gives fantasy points (ESPN, Yahoo, Sleeper),
start/sit odds, betting line value, and a live record against real sportsbooks.

**Stack:** LightGBM, Optuna, SciPy and FastAPI in Python, plus a Next.js and TypeScript dashboard,
all served as one app. Data comes from NBA.com (`nba_api` and NBA's CDN) and ESPN's injury report.

## Results

<!-- RESULTS:START -->
Held out test: 23,805 player games from 2025-26, trained on 47,154 games from 2023-24 and 2024-25.

**Points**

| Method | MAE (points) | RMSE (points) |
| --- | --- | --- |
| Season average so far | 4.69 | 6.17 |
| Last 5 game average | 4.81 | 6.33 |
| v1 gradient boosting | 4.62 | 6.00 |
| v2 LightGBM + Optuna | 4.51 | 5.86 |

The v2 model's average miss is 4.0% smaller than the best baseline and 2.3% smaller than v1.

**Every stat** (MAE, and how often the actual result landed inside the 80% interval)

| Stat | Model | Season avg | Last 5 | 80% interval coverage |
| --- | --- | --- | --- | --- |
| PTS | 4.51 | 4.69 | 4.81 | 84.9% |
| REB | 1.87 | 1.92 | 1.99 | 88.8% |
| AST | 1.34 | 1.37 | 1.41 | 90.7% |
| FG3M | 0.90 | 0.90 | 0.94 | 93.2% |
| STL | 0.73 | 0.73 | 0.76 | 94.9% |
| BLK | 0.52 | 0.51 | 0.52 | 95.1% |
| TOV | 0.88 | 0.91 | 0.94 | 93.4% |
| FGM | 1.74 | 1.79 | 1.85 | 88.7% |
| FGA | 2.68 | 2.88 | 2.87 | 85.7% |
| FTM | 1.35 | 1.35 | 1.39 | 92.4% |
| FTA | 1.63 | 1.65 | 1.69 | 91.6% |

**Fantasy points** (expected fantasy points vs actual)

| Scoring | Model MAE | Season avg MAE |
| --- | --- | --- |
| ESPN | 9.63 | 10.04 |
| Yahoo | 7.50 | 7.95 |

**Calibration (points).** Log likelihood per game -2.9993 (recalibrated), -3.0235 (raw negative binomial), -3.6203 (plain Poisson). Largest PIT bin deviation from a perfect 10%: 0.9% (raw: 2.4%). Over/under Brier score 0.2364 vs 0.25 for a coin flip.

**Prop backtest (points, synthetic lines).** 14,110 games of players averaging 20+ minutes, priced by a baseline book that knows only the season average (calibrated on its own past errors, 4.5% margin). The model bet 11,452 of them, quarter Kelly, at most 2% per bet and 20% per day, no compounding: hit rate 61.0%, ROI on stakes 17.3%, max drawdown 5.5%. This shows the model's probabilities beat a season average line, not that it beats sportsbooks, which also price in injuries, lineups and market money.
<!-- RESULTS:END -->

## How it works

1. **Data.** Every player game from the last three seasons (about 24,000 a season). Live, the season
   in progress updates from NBA's CDN, which also works from cloud servers, and the schedule supplies
   each player's next opponent, home or away, and rest days. Games still in progress are ignored until
   they're final.
2. **Features (158).** Recent form over 3 to 20 games, minutes and usage, shooting, rest and
   schedule, last season, the opponent's defense, and **which regular teammates are out**. Every
   feature uses only games played *before* the one being predicted. Live predictions run the same
   code as training, and tests check both properties.
3. **Injuries.** In history, a teammate counts as "out" when a regular (15+ minutes a game) didn't
   play. Live, the "out" list comes from ESPN's injury report, and the player page lets you toggle any
   teammate to see a what-if.
4. **Models.** One LightGBM model per stat, tuned with Optuna on a later season than it trains on.
   Each prediction gets a SHAP explanation in plain units, like "Teammates out +2.1 points".
5. **Ranges, not just averages.** Each stat gets a negative binomial distribution whose spread
   depends on the player. The distribution is then recalibrated so its probabilities hold up on games
   it never saw. A Gaussian copula links the stats, so fantasy points and combined props are simulated
   realistically.
6. **Evaluation.** Walk-forward: train on 2023-24 and tune on 2024-25, then train on both and test on
   all of 2025-26. The baselines are the player's season average and his last 5 game average.

**Betting tools.** Prop pricing removes the bookmaker's margin, then computes edge, expected value
and a fractional Kelly stake. The backtest bets against a fake bookmaker built from the season
average, so it shows real signal but isn't proof the model beats real books. For that, the
**real-lines tracker** saves each day's actual sportsbook lines with the model's picks and grades
them after the games. It's off until you add a free key from
[the-odds-api.com](https://the-odds-api.com) as `NBA_ODDS_API_KEY`.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cd web && npm install && npm run build && cd ..      # builds the dashboard into web/out

nba-predictor train      # downloads 3 seasons once, tunes, evaluates (about 7 min)
nba-predictor serve      # the whole app at http://127.0.0.1:8000
```

| Command | What it does |
| --- | --- |
| `nba-predictor train --trials 0` | Trains without tuning (about 2 min) |
| `nba-predictor refresh` | Pulls the latest games, schedule and injury report |
| `nba-predictor lines` | Saves today's real prop lines and grades finished ones (needs the key) |
| `pytest`, `ruff check .`, `mypy` | Tests, lint, type check |
| `cd web && npm run dev` | Dashboard with hot reload on :3000, proxying `/api` to :8000 |

**Docker:** `docker build -t nba-predictor . && docker run -p 7860:7860 nba-predictor` runs the same
single app the hosted version uses.

## Code map

```
src/nba_predictor/
  data.py, cdn.py, injuries.py   where data comes from (stats.nba.com, NBA CDN, ESPN)
  features.py                    leakage safe features, shared by training and serving
  modeling.py, distributions.py  LightGBM + Optuna, calibrated distributions, copula, SHAP
  train.py, backtest.py          walk-forward training, evaluation, backtest, model card
  store.py, predictor.py         live data and everything the app computes
  fantasy.py, quant.py, lines.py fantasy scoring, betting math, real-lines tracker
  api/                           FastAPI (also serves the dashboard)
web/                             Next.js dashboard, exported as a static site
tests/                           pytest suite (no network: every source is mocked)
```

## Limitations

* The injury report is only as good as ESPN's feed, and late scratches after it updates are missed.
* Players need 5 games in a season before they count in training.
* Each stat has its own model, so simulated points aren't forced to equal 2·FGM + 3PM + FTM.
* Everything here is informational. The model's record against real books starts only once the
  tracker is on.
