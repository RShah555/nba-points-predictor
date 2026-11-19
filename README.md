# NBA Points Predictor

Projects how many points an NBA player will score in his next game, using his recent form, the opponent's defense, home court and days of rest. Built with Python, scikit-learn and Streamlit on top of NBA.com data from `nba_api`.

## How it works

* **Data.** Every player game log for three seasons, pulled with one `LeagueGameLog` call per season (roughly 25,000 player games per season).
* **Features.** 3, 5 and 10 game rolling averages for points, minutes, shot attempts, free throw attempts, assists, rebounds and turnovers, plus season averages, home or away, days of rest, back to backs, and how many points the opponent has been allowing.
* **No leakage.** Every feature is shifted so it only uses games played *before* the one being predicted. `tests/test_features.py` checks this.
* **Model.** scikit-learn `HistGradientBoostingRegressor`.
* **Evaluation.** Trained on two seasons and tested on a full season the model never saw, against the two baselines a fantasy player would actually use: the player's season average so far and his last 5 game average.

## Results

<!-- RESULTS:START -->
Tested on 23,805 player games from the 2025-26 season, trained on 47,154 games from 2023-24 and 2024-25.

| Method | MAE (points) | RMSE (points) |
| --- | --- | --- |
| Season average so far | 4.69 | 6.18 |
| Last 5 game average | 4.81 | 6.33 |
| Gradient boosting model | 4.62 | 6.0 |

The model's average miss is 1.5% smaller than the best baseline.
<!-- RESULTS:END -->

## Run it

```bash
pip install -r requirements.txt
python train_model.py        # downloads the data once, trains, prints results, saves models/points_model.joblib
pytest                       # checks the features never look ahead
streamlit run app.py
```

In the app, search for a player, pick his next opponent, home or away and days of rest, and it returns the projection next to his recent scoring trend.
