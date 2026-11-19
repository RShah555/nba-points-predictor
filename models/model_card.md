# Model card: NBA player projections v2.0.0

Trained 2026-11-19T03:19:18+00:00.

## What it does
Predicts the full probability distribution of a player's PTS, REB, AST, FG3M, STL, BLK, TOV, FGM, FGA, FTM, FTA in his next game.
Each stat has a LightGBM model (Poisson objective) for its mean and a negative binomial distribution whose
spread depends on the mean. A Gaussian copula ties the stats together for fantasy points and combo props.

## Data
NBA.com regular season player game logs via `nba_api` (one LeagueGameLog call per season).
Held out evaluation: trained on 2023-24, 2024-25 (47,154 games),
tested on 2025-26 (23,805 games). The serving model is refit on
2023-24, 2024-25, 2025-26. 152 features, all computed only from games before
the one being predicted.

## Results on 2025-26
```
Method                        MAE    RMSE
Season average so far        4.69    6.17
Last 5 game average          4.81    6.33
v1 gradient boosting         4.62    6.00
v2 LightGBM + Optuna         4.58    5.95

Stat   Model MAE  Season avg   Last 5  80% cover
PTS        4.575       4.693    4.813      85.1%
REB        1.890       1.921    1.993      89.0%
AST        1.349       1.368    1.410      91.0%
FG3M       0.897       0.900    0.936      93.4%
STL        0.729       0.731    0.759      95.0%
BLK        0.522       0.508    0.523      95.2%
TOV        0.891       0.910    0.944      93.5%
FGM        1.754       1.794    1.847      88.9%
FGA        2.755       2.876    2.874      85.7%
FTM        1.354       1.355    1.391      92.5%
FTA        1.639       1.645    1.689      91.7%
```

## Intended use
Fantasy start/sit decisions, projections and learning. The prop pricing tools are informational.
The betting backtest uses synthetic lines (season average so far), which are much easier to beat than
real sportsbook lines.

## Limitations
* No injury, lineup or minutes news: if a teammate is ruled out an hour before tip, the model does not know.
* Players with fewer than 5 games in a season are excluded from training and evaluation.
* Stat models are trained separately, so simulated PTS is not forced to equal 2*FGM + FG3M + FTM.
* Opponent strength is measured by what a defense allowed, not by matchup or position.
