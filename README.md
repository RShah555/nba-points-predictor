# NBA Predictor

Probabilistic next game projections for every NBA player: the full distribution of points, rebounds,
assists, threes, steals, blocks, turnovers and shooting volume. These feed fantasy projections for ESPN,
Yahoo and Sleeper scoring, start/sit odds, and a prop pricing tool with expected value and Kelly sizing.

**Stack:** Python, LightGBM, Optuna, SciPy, MLflow and FastAPI on the backend; Next.js, TypeScript,
Tailwind and Recharts on the frontend. All data is NBA.com data, pulled with `nba_api`.

> v1 (a single points model with a Streamlit app) is tagged [`v1.0.0`](../../tree/v1.0.0). v2 keeps its design rules:
> no leakage, the same train and test seasons, the same baselines, and the season worked out from today's date.

## Results

<!-- RESULTS:START -->
Held out test: 23,805 player games from 2025-26, trained on 47,154 games from 2023-24 and 2024-25.

**Points**

| Method | MAE (points) | RMSE (points) |
| --- | --- | --- |
| Season average so far | 4.69 | 6.17 |
| Last 5 game average | 4.81 | 6.33 |
| v1 gradient boosting | 4.62 | 6.00 |
| v2 LightGBM + Optuna | 4.58 | 5.95 |

The v2 model's average miss is 2.5% smaller than the best baseline and 0.9% smaller than v1.

**Every stat** (MAE, and how often the actual result landed inside the 80% interval)

| Stat | Model | Season avg | Last 5 | 80% interval coverage |
| --- | --- | --- | --- | --- |
| PTS | 4.58 | 4.69 | 4.81 | 85.1% |
| REB | 1.89 | 1.92 | 1.99 | 89.0% |
| AST | 1.35 | 1.37 | 1.41 | 91.0% |
| FG3M | 0.90 | 0.90 | 0.94 | 93.4% |
| STL | 0.73 | 0.73 | 0.76 | 95.0% |
| BLK | 0.52 | 0.51 | 0.52 | 95.2% |
| TOV | 0.89 | 0.91 | 0.94 | 93.5% |
| FGM | 1.75 | 1.79 | 1.85 | 88.9% |
| FGA | 2.75 | 2.88 | 2.87 | 85.7% |
| FTM | 1.35 | 1.35 | 1.39 | 92.5% |
| FTA | 1.64 | 1.65 | 1.69 | 91.7% |

**Fantasy points** (expected fantasy points vs actual)

| Scoring | Model MAE | Season avg MAE |
| --- | --- | --- |
| ESPN | 9.75 | 10.04 |
| Yahoo | 7.66 | 7.95 |

**Calibration (points).** Log likelihood per game -3.0178 (recalibrated), -3.0396 (raw negative binomial), -3.6864 (plain Poisson). Largest PIT bin deviation from a perfect 10%: 0.9% (raw: 1.9%). Over/under Brier score 0.2399 vs 0.25 for a coin flip.

**Prop backtest (points, synthetic lines).** 14,110 games of players averaging 20+ minutes, priced by a baseline book that knows only the season average (calibrated on its own past errors, 4.5% margin). The model bet 10,514 of them, quarter Kelly, at most 2% per bet and 20% per day, no compounding: hit rate 59.8%, ROI on stakes 15.5%, max drawdown 6.4%. This shows the model's probabilities beat a season average line, not that it beats sportsbooks, which also price in injuries, lineups and market money.
<!-- RESULTS:END -->

## How it works

```
stats.nba.com ──► data.py ──► features.py ──► modeling.py ──► distributions.py ──► predictor.py ──► FastAPI ──► Next.js
 LeagueGameLog    cache +      152 leakage    LightGBM per     negative binomial    projections,     /api/*       dashboard
 + schedule       retries      safe features  stat, Optuna     + recalibration      props, slate
                                              tuned            + Gaussian copula
```

**Data.** A single `LeagueGameLog` call per season returns every player game, about 24,000 per season.
Finished seasons are cached forever. The season in progress refreshes every 6 hours in a background thread,
and the API keeps serving the cached copy while it downloads. The schedule (`ScheduleLeagueV2`) fills in
each player's real next opponent, home or away, and days of rest. Every call to stats.nba.com retries
3 times, 30 seconds apart.

**Features (152).** Every feature for a game uses only games played before it: each stat is shifted one
game before any window. The features are:

* 3, 5, 10 and 20 game rolling means and an exponentially weighted mean for 17 box score stats
* per 36 minute rates, a usage proxy, true shooting and 3 point percentage, and a minutes trend
* volatility (10 game standard deviation)
* season to date averages and last season's averages
* rest days, back to backs, games in the last 7 days, and days into the season
* the opponent's season to date and last 10 game numbers allowed for each stat, plus possessions allowed

**Train/serve parity.** To predict a game that hasn't been played, the predictor appends a placeholder
row for it and runs the same feature pipeline used in training. There is no second implementation that
could drift from the first. `tests/test_features.py` checks that a live feature row matches the training
row for the same game exactly.

**Models.** There is one LightGBM model per stat, using a Poisson objective: counts with a log link, so
predictions are always positive. Optuna's TPE sampler tunes each model on a later season than it trains
on. Explanations come from LightGBM's built in TreeSHAP. Because the model works on a log scale, SHAP
values multiply rather than add. The API groups them into feature families and converts them to the
stat's own units, so "Opponent defense +1.4 points" adds up exactly to the projection.

**Distributions.** A predicted mean isn't enough for props or fantasy, so the model builds a full
distribution in three steps:

1. **Shape.** A negative binomial whose spread grows with the mean: `size = exp(c0 + c1 log mu)`, fit by
   maximum likelihood on out of sample predictions.
2. **Recalibration.** Points come in 2s and 3s, so a negative binomial's shape is slightly off for them.
   A monotone map G learned from out of sample PIT values corrects the CDF to G(F(y))
   (Kuleshov et al., 2018). For stats whose shape was already right, G is close to the identity.
3. **Dependence.** A Gaussian copula fit on normal scores of the PIT values ties the 11 stats together,
   so simulations capture that a hot shooting night lifts PTS, FGM and 3PM at once. That matters for
   fantasy points, PRA props and double-double odds.

**Evaluation.** Evaluation runs walk forward over seasons:

| Fold | Train | Test | Role |
| --- | --- | --- | --- |
| 1 | 2023-24 | 2024-25 | validation: hyperparameters are tuned here |
| 2 | 2023-24, 2024-25 | 2025-26 | held out test: the headline numbers |
| serve | all three | 2026-27 live | the model the API uses |

Each fold's distribution parameters are fit on the previous fold's predictions, so the test season never
informs its own intervals. The baselines are the two numbers a fantasy player would actually use: the
player's season average so far and his last 5 game average. The v1 model is retrained on the same split
as a third reference.

**Quant.** The quant tools are:

* **Prop pricing.** Converts American odds, removes the vig proportionally, and computes edge against
  the fair price, expected value per unit, and growth optimal Kelly sizing with pushes handled
  (`f = (bp − q) / (b(p + q))`). The stake is fractional Kelly, capped.
* **Combo markets.** PRA, PR, PA, RA and steals plus blocks are priced from copula simulations.
* **Backtest.** Historical sportsbook lines aren't freely available, so the backtest bets against a
  baseline book. The book knows only the player's season average, with its spread calibrated on its
  own past errors, hangs the half point line nearest its median, and prices both sides at its fair
  probability plus a 4.5% margin. A model with no extra information finds zero bets against it; a
  unit test checks that. Bets are limited to players averaging 20+ minutes, at most 2% of the bankroll
  per bet and 20% per day, with no compounding. The backtest reports the profit curve, drawdown, hit
  rate by edge bucket, Brier score and a reliability diagram.

  Why a 2.5% MAE gain still gives a real edge: a single game's points have an SD of about 6, so MAE is
  mostly noise that no model can remove. Moving the mean a point or two barely changes MAE, but it
  moves P(over) by about 10 points. Real books also price in injuries, lineups and market money,
  which this baseline doesn't, so treat ROI as an upper bound. Calibration is the honest number.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,mlops]"

nba-predictor train          # downloads 3 seasons once, tunes, evaluates, writes models/ and this README's results
nba-predictor serve          # API on http://127.0.0.1:8000  (docs at /docs)

cd web && npm install && npm run dev                # dashboard on http://localhost:3000
```

Or use `docker compose up --build`, which runs the API on port 8000 and the web app on port 3000.

Other commands:

```bash
pytest                       # leakage, parity, distribution, betting math, fantasy and API tests
ruff check . && mypy         # lint and type check
nba-predictor refresh        # force download today's games and schedule
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse training runs
```

`nba-predictor train --trials 0` skips tuning and finishes in about 90 seconds.

## The app

| Page | What it does |
| --- | --- |
| **Slate** | Today's games (or the next game day) and every rotation player's projected fantasy points, sortable by ESPN, Yahoo or Sleeper scoring |
| **Player** | The projection for the next game: an interval and distribution chart for each stat, "why this number" SHAP explanations, milestone odds (double-double, 30+ points), fantasy points on each platform, and the recent game log |
| **Props** | Enter a line and odds for any market and get model probability, fair price, edge, EV and a Kelly stake |
| **Start/Sit** | Two to six players head to head: each one's fantasy distribution, P(A outscores B) and P(best) |
| **Model** | Backtest results, the calibration (PIT) histograms before and after recalibration, a reliability diagram, the bankroll curve, feature importance and the stat correlation matrix |

## API

| Method | Path | |
| --- | --- | --- |
| GET | `/api/health` | liveness, model and data status |
| GET | `/api/meta` | season, teams, markets, scoring systems, headline metrics |
| GET | `/api/players?q=` | player search, ranked by name match and minutes |
| GET | `/api/players/{id}` | player card and last 25 games |
| POST | `/api/projections` | full projection: distributions, combos, fantasy, milestones, explanations |
| POST | `/api/props` | prop pricing for one market |
| POST | `/api/compare` | start/sit simulation |
| GET | `/api/slate` | the next game day's projections for every rotation player |
| GET | `/api/model/report` | everything from the last training run |

If you leave out `opponent`, the request uses the player's next scheduled game. Interactive docs are at
`/docs`.

## Project layout

```
src/nba_predictor/
  config.py          settings (NBA_* environment variables or .env)
  seasons.py         season arithmetic from today's date
  data.py            LeagueGameLog download, retries, cache, schema checks
  features.py        leakage safe features shared by training and serving
  modeling.py        LightGBM + Optuna, model bundle, SHAP explanations
  distributions.py   negative binomial, recalibration, Gaussian copula
  fantasy.py         ESPN / Yahoo / Sleeper scoring
  quant.py           odds, vig, EV, Kelly
  backtest.py        accuracy, calibration, prop betting backtest
  train.py           walk forward training pipeline, report, model card, MLflow
  store.py           live league data and schedule for serving
  predictor.py       projections, simulations, props, start/sit, slate
  api/               FastAPI app and request schemas
web/                 Next.js dashboard
tests/               pytest suite
models/              trained bundle, report.json, metrics.json, model_card.md
```

## Limitations

* The model has no injury or lineup news. If a star teammate is ruled out an hour before tip, it
  doesn't know.
* Players need 5 games in a season before they count in training and evaluation.
* Each stat has its own model, so simulated points aren't forced to equal 2·FGM + 3PM + FTM. The copula
  keeps them strongly correlated instead.
* Opponent strength means what a defense has allowed overall, not position by position matchups.
* Projections and prop prices are informational. The model has never been tested against real closing
  lines.
