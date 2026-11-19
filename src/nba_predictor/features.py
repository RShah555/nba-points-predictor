"""Leakage safe feature engineering, shared by training and live inference.

Rule: every feature for a game is computed only from games played before it.
Player features shift each stat by one game before any rolling, expanding or
exponentially weighted window. Opponent features shift the opponent's games the
same way.

Train/serve parity: to predict a future game we append a placeholder row for it
(stats unknown) and run the exact same pipeline, then read off that last row.
There is no second implementation of the features that could drift from the
one the model was trained on. tests/test_features.py checks this.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS: tuple[str, ...] = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA")
ROLL_STATS: tuple[str, ...] = (
    "PTS",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
)
NUMERIC = (*ROLL_STATS,)
WINDOWS: tuple[int, ...] = (3, 5, 10, 20)
EWM_HALFLIFE = 5
RATE_STATS: tuple[str, ...] = ("PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FGA", "FTA")
VOL_STATS: tuple[str, ...] = ("PTS", "MIN", "REB", "AST")
SEASON_STATS: tuple[str, ...] = (*TARGETS, "MIN")
PREV_SEASON_STATS: tuple[str, ...] = ("PTS", "MIN", "REB", "AST", "FG3M")
DEFENSE_STATS: tuple[str, ...] = (*TARGETS, "POSS")
MIN_PRIOR_GAMES = 5  # skip a player's first few games of a season, there is little history yet
NEXT_GAME_ID = "NEXT"


def _player_feature_names() -> list[str]:
    names = [f"{s}_roll_{w}" for s in ROLL_STATS for w in WINDOWS]
    names += [f"{s}_ewm" for s in ROLL_STATS]
    names += [f"{s}_std_10" for s in VOL_STATS]
    names += [f"{s}_per36_10" for s in RATE_STATS]
    names += ["USAGE_per36_10", "FG_PCT_20", "FG3_PCT_20", "FT_PCT_20", "TS_PCT_20", "MIN_TREND"]
    names += [f"SEASON_AVG_{s}" for s in SEASON_STATS]
    names += [f"PREV_SEASON_{s}" for s in PREV_SEASON_STATS] + ["PREV_SEASON_GP"]
    names += ["GAMES_PLAYED", "HOME", "REST_DAYS", "BACK_TO_BACK", "GAMES_LAST_7D", "DAYS_INTO_SEASON"]
    return names


DEFENSE_FEATURES: list[str] = [f"OPP_{s}_ALLOWED" for s in DEFENSE_STATS] + [
    f"OPP_{s}_ALLOWED_L10" for s in DEFENSE_STATS
]

# Teammates out: regulars (15+ minutes over their last 10 games, played for the team in the last
# 21 days) who are not playing. In history that is whoever did not appear in the box score; live it
# comes from the injury report. Their recent minutes, points, assists, rebounds and usage are summed.
ABSENCE_STATS: tuple[str, ...] = ("MIN", "PTS", "AST", "REB", "USG")
ABSENCE_FEATURES: list[str] = [f"OUT_{s}" for s in ABSENCE_STATS] + ["OUT_COUNT"]
REGULAR_MINUTES = 15.0
REGULAR_DAYS = 21

FEATURES: list[str] = _player_feature_names() + ABSENCE_FEATURES + DEFENSE_FEATURES

# Families used to turn per-feature SHAP values into explanations a person can read.
FEATURE_GROUPS: dict[str, str] = {}
for _f in FEATURES:
    if _f.startswith("OUT_"):
        FEATURE_GROUPS[_f] = "Teammates out"
    elif _f.startswith("OPP_"):
        FEATURE_GROUPS[_f] = "Opponent defense"
    elif _f.startswith(("PREV_SEASON",)):
        FEATURE_GROUPS[_f] = "Last season"
    elif _f.startswith("SEASON_AVG") or _f == "GAMES_PLAYED":
        FEATURE_GROUPS[_f] = "Season to date"
    elif _f in ("REST_DAYS", "BACK_TO_BACK", "GAMES_LAST_7D", "DAYS_INTO_SEASON"):
        FEATURE_GROUPS[_f] = "Rest and schedule"
    elif _f == "HOME":
        FEATURE_GROUPS[_f] = "Home court"
    elif _f.startswith("MIN") or _f == "MIN_TREND":
        FEATURE_GROUPS[_f] = "Minutes and role"
    elif "PCT" in _f or "USAGE" in _f or "per36" in _f:
        FEATURE_GROUPS[_f] = "Usage and efficiency"
    elif "_std_" in _f:
        FEATURE_GROUPS[_f] = "Volatility"
    else:
        FEATURE_GROUPS[_f] = "Recent form"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Make PlayerGameLog and LeagueGameLog frames (cached or fresh) look the same."""
    df = df.rename(columns={"Player_ID": "PLAYER_ID", "Game_ID": "GAME_ID"}).copy()
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.lstrip("0")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    parts = df["MATCHUP"].str.split(" ")
    df["TEAM"] = parts.str[0]
    df["OPP"] = parts.str[-1]
    return df.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort").reset_index(drop=True)


def _games_in_last(dates: pd.Series, days: int) -> np.ndarray:
    """For each game, how many of the player's earlier games fell in the previous `days` days."""
    d = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    first_in_window = np.searchsorted(d, d - days, side="left")
    return np.arange(len(d)) - first_in_window


def _grouped(frame: pd.DataFrame, keys: list[pd.Series]):
    return frame.groupby(keys, sort=False)


def _drop_group_levels(result: pd.DataFrame, n_levels: int) -> pd.DataFrame:
    return result.reset_index(level=list(range(n_levels)), drop=True).sort_index()


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every player level feature. `df` must come from normalize() (sorted by player and date)."""
    stats = list(ROLL_STATS)
    pid = df["PLAYER_ID"]
    prev = df.groupby("PLAYER_ID", sort=False)[stats].shift(1)
    by_player = _grouped(prev, [pid])
    out: dict[str, pd.Series] = {}

    sums_10 = _drop_group_levels(by_player.rolling(10, min_periods=1).sum(), 1)
    sums_20 = _drop_group_levels(by_player.rolling(20, min_periods=1).sum(), 1)
    for w in WINDOWS:
        means = _drop_group_levels(by_player.rolling(w, min_periods=1).mean(), 1)
        for s in stats:
            out[f"{s}_roll_{w}"] = means[s]
    ewm = _drop_group_levels(by_player.ewm(halflife=EWM_HALFLIFE).mean(), 1)
    for s in stats:
        out[f"{s}_ewm"] = ewm[s]
    std = _drop_group_levels(_grouped(prev[list(VOL_STATS)], [pid]).rolling(10, min_periods=3).std(), 1)
    for s in VOL_STATS:
        out[f"{s}_std_10"] = std[s]

    minutes = sums_10["MIN"].where(sums_10["MIN"] > 0)
    for s in RATE_STATS:
        out[f"{s}_per36_10"] = 36 * sums_10[s] / minutes
    out["USAGE_per36_10"] = 36 * (sums_10["FGA"] + 0.44 * sums_10["FTA"] + sums_10["TOV"]) / minutes

    def pct(made: str, att: str) -> pd.Series:
        return sums_20[made] / sums_20[att].where(sums_20[att] > 0)

    out["FG_PCT_20"] = pct("FGM", "FGA")
    out["FG3_PCT_20"] = pct("FG3M", "FG3A")
    out["FT_PCT_20"] = pct("FTM", "FTA")
    shots = 2 * (sums_20["FGA"] + 0.44 * sums_20["FTA"])
    out["TS_PCT_20"] = sums_20["PTS"] / shots.where(shots > 0)
    out["MIN_TREND"] = out["MIN_roll_3"] - out["MIN_roll_10"]

    season_keys = [pid, df["SEASON_ID"]]
    prev_season = df.groupby(["PLAYER_ID", "SEASON_ID"], sort=False)[list(SEASON_STATS)].shift(1)
    season_avg = _drop_group_levels(_grouped(prev_season, season_keys).expanding().mean(), 2)
    for s in SEASON_STATS:
        out[f"SEASON_AVG_{s}"] = season_avg[s]
    out["GAMES_PLAYED"] = df.groupby(["PLAYER_ID", "SEASON_ID"], sort=False).cumcount().astype(float)

    rest = df.groupby("PLAYER_ID", sort=False)["GAME_DATE"].diff().dt.days
    out["REST_DAYS"] = rest.clip(upper=7).fillna(3)
    out["BACK_TO_BACK"] = (out["REST_DAYS"] == 1).astype(float)
    out["GAMES_LAST_7D"] = (
        df.groupby("PLAYER_ID", sort=False)["GAME_DATE"]
        .transform(lambda s: pd.Series(_games_in_last(s, 7), index=s.index))
        .astype(float)
    )
    season_start = pd.to_datetime(df["SEASON_ID"].str[1:] + "-10-01")
    out["DAYS_INTO_SEASON"] = (df["GAME_DATE"] - season_start).dt.days.astype(float)

    features = pd.DataFrame(out, index=df.index)
    result = pd.concat([df.drop(columns=[c for c in features.columns if c in df.columns]), features], axis=1)
    return _add_previous_season(result)


def _add_previous_season(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's averages from the season before (a whole season that ended before this one began)."""
    played = df[df["GAME_ID"] != NEXT_GAME_ID]
    table = played.groupby(["PLAYER_ID", "SEASON_ID"])[list(PREV_SEASON_STATS)].mean()
    table["GP"] = played.groupby(["PLAYER_ID", "SEASON_ID"]).size()
    table = table.add_prefix("PREV_SEASON_").reset_index()
    table["SEASON_ID"] = (table["SEASON_ID"].astype(int) + 1).astype(str)
    df = df.drop(columns=[c for c in table.columns if c.startswith("PREV_SEASON_") and c in df.columns])
    return df.merge(table, on=["PLAYER_ID", "SEASON_ID"], how="left")


def team_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: what each team produced, summed from its players."""
    cols = [*TARGETS, "OREB"]
    played = df[df["GAME_ID"] != NEXT_GAME_ID]
    totals = played.groupby(["SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM", "OPP"], as_index=False)[cols].sum()
    totals["POSS"] = totals["FGA"] - totals["OREB"] + totals["TOV"] + 0.44 * totals["FTA"]
    return totals


def defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """What each defense allowed per game before a given game: season to date and last 10 games."""
    allowed = (
        team_games(df)
        .drop(columns="TEAM")
        .rename(columns={"OPP": "DEF_TEAM"})
        .sort_values(["DEF_TEAM", "GAME_DATE"], kind="mergesort")
        .reset_index(drop=True)
    )
    stats = list(DEFENSE_STATS)
    keys = [allowed["DEF_TEAM"], allowed["SEASON_ID"]]
    prev = allowed.groupby(["DEF_TEAM", "SEASON_ID"], sort=False)[stats].shift(1)
    season = _drop_group_levels(_grouped(prev, keys).expanding().mean(), 2)
    last10 = _drop_group_levels(_grouped(prev, keys).rolling(10, min_periods=1).mean(), 2)
    out = allowed[["DEF_TEAM", "GAME_ID"]].copy()
    for s in stats:
        out[f"OPP_{s}_ALLOWED"] = season[s]
        out[f"OPP_{s}_ALLOWED_L10"] = last10[s]
    return out


def defense_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Each defense's numbers going into its next game (includes every game played so far).

    Uses the latest season in the data, so early in a new season it reflects the new season.
    Indexed by team abbreviation, columns are DEFENSE_FEATURES.
    """
    allowed = team_games(df).rename(columns={"OPP": "DEF_TEAM"})
    allowed = allowed[allowed["SEASON_ID"] == allowed["SEASON_ID"].max()].sort_values("GAME_DATE")
    stats = list(DEFENSE_STATS)
    season = allowed.groupby("DEF_TEAM")[stats].mean().add_prefix("OPP_").add_suffix("_ALLOWED")
    last10 = (
        allowed.groupby("DEF_TEAM")
        .tail(10)
        .groupby("DEF_TEAM")[stats]
        .mean()
        .add_prefix("OPP_")
        .add_suffix("_ALLOWED_L10")
    )
    return pd.concat([season, last10], axis=1)[DEFENSE_FEATURES]


def player_form(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's last 10 game averages *after* every game (what a teammate brings to the next one)."""
    played = df[df["GAME_ID"] != NEXT_GAME_ID]
    stats = played[["MIN", "PTS", "AST", "REB"]].assign(USG=played["FGA"] + 0.44 * played["FTA"] + played["TOV"])
    form = _drop_group_levels(_grouped(stats, [played["PLAYER_ID"]]).rolling(10, min_periods=1).mean(), 1)
    return pd.concat([played[["PLAYER_ID", "TEAM", "GAME_ID", "GAME_DATE"]], form], axis=1)


def _absence_row(regulars: pd.DataFrame) -> dict[str, float]:
    row = {f"OUT_{s}": float(regulars[s].sum()) for s in ABSENCE_STATS}
    row["OUT_COUNT"] = float(len(regulars))
    return row


def _regulars(roster: pd.DataFrame, team: str, on: pd.Timestamp) -> pd.DataFrame:
    recent = roster["GAME_DATE"] >= on - pd.Timedelta(days=REGULAR_DAYS)
    return roster[(roster["TEAM"] == team) & recent & (roster["MIN"] >= REGULAR_MINUTES)]


def team_absences(df: pd.DataFrame) -> pd.DataFrame:
    """For every team game, the regulars who did not play, summed (one row per TEAM, GAME_ID).

    Walks the season day by day: a player's roster entry is his latest game (team and form) from
    *before* the current day, so a traded player counts for his new team only.
    """
    form = player_form(df).sort_values("GAME_DATE", kind="mergesort")
    roster = form.iloc[0:0]
    out = []
    for day, games in form.groupby("GAME_DATE", sort=True):
        for (team, game_id), box in games.groupby(["TEAM", "GAME_ID"]):
            regs = _regulars(roster, str(team), pd.Timestamp(str(day)))
            out.append(
                {"TEAM": team, "GAME_ID": game_id, **_absence_row(regs[~regs["PLAYER_ID"].isin(box["PLAYER_ID"])])}
            )
        roster = pd.concat([roster[~roster["PLAYER_ID"].isin(games["PLAYER_ID"])], games], ignore_index=True)
    return pd.DataFrame(out, columns=["TEAM", "GAME_ID", *ABSENCE_FEATURES])


def roster_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Every player's latest team and form: who counts as a regular going into the next game."""
    return player_form(df).sort_values("GAME_DATE", kind="mergesort").groupby("PLAYER_ID").tail(1)


def build_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature frame for every player game with enough history, plus the latest defense snapshot."""
    df = add_player_features(normalize(raw))
    df = df.merge(defense_features(df), left_on=["OPP", "GAME_ID"], right_on=["DEF_TEAM", "GAME_ID"], how="left").drop(
        columns=["DEF_TEAM"]
    )
    df = df.merge(team_absences(df), on=["TEAM", "GAME_ID"], how="left")
    df = df[df["GAMES_PLAYED"] >= MIN_PRIOR_GAMES]
    return df.reset_index(drop=True), defense_snapshot(normalize(raw))


@dataclass(frozen=True)
class GameContext:
    """A game that has not been played yet."""

    player_id: int
    opponent: str
    home: bool
    game_date: dt.date
    season_id: str
    team: str | None = None
    teammates_out: tuple[int, ...] = ()


def next_game_features(
    history: pd.DataFrame,
    contexts: Sequence[GameContext],
    snapshot: pd.DataFrame,
    roster: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows for upcoming games, one per context, in the same order.

    `history` holds raw game logs (any players). Each context's player must appear in it.
    `roster` (from roster_snapshot) says who the regulars are; each context lists which are out.
    """
    ids = [c.player_id for c in contexts]
    if len(set(ids)) < len(ids):  # same player twice: each scenario must not see the other as a past game
        return pd.concat([next_game_features(history, [c], snapshot, roster) for c in contexts], ignore_index=True)
    hist = normalize(history)
    hist = hist[hist["PLAYER_ID"].isin(set(ids))]
    placeholders = []
    for i, c in enumerate(contexts):
        last = hist[hist["PLAYER_ID"] == c.player_id]
        if last.empty:
            raise KeyError(f"No games found for player {c.player_id}")
        placeholders.append(
            {
                "PLAYER_ID": c.player_id,
                "SEASON_ID": c.season_id,
                "GAME_ID": NEXT_GAME_ID,
                "GAME_DATE": pd.Timestamp(c.game_date),
                "HOME": int(c.home),
                "TEAM": c.team or last["TEAM"].iloc[-1],
                "OPP": c.opponent,
                "_ORDER": i,
            }
        )
    future = pd.DataFrame(placeholders)
    combined = pd.concat([hist, future], ignore_index=True)
    combined = combined.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort").reset_index(drop=True)
    feats = add_player_features(combined)
    rows = feats[feats["GAME_ID"] == NEXT_GAME_ID].sort_values("_ORDER").reset_index(drop=True)
    for col in DEFENSE_FEATURES:
        rows[col] = rows["OPP"].map(snapshot[col]) if col in snapshot else np.nan
    absences = []
    for c, team in zip(contexts, rows["TEAM"], strict=True):
        if roster is None:
            absences.append(_absence_row(pd.DataFrame(columns=list(ABSENCE_STATS))))
            continue
        regs = _regulars(roster[roster["PLAYER_ID"] != c.player_id], str(team), pd.Timestamp(c.game_date))
        absences.append(_absence_row(regs[regs["PLAYER_ID"].isin(c.teammates_out)]))
    for col in ABSENCE_FEATURES:
        rows[col] = [a[col] for a in absences]
    return rows
