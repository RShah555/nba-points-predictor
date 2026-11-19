export type Stat = "PTS" | "REB" | "AST" | "FG3M" | "STL" | "BLK" | "TOV" | "FGM" | "FGA" | "FTM" | "FTA";
export type ScoringKey = "espn" | "yahoo" | "sleeper";

export interface Player {
  id: number;
  name: string;
  team: string;
  team_id: number | null;
  games: number;
  pts: number | null;
  reb: number | null;
  ast: number | null;
  min: number | null;
  headshot: string;
  injury: Injury | null;
}

export interface Injury {
  status: string;
  note: string;
  return_date: string | null;
  out: boolean;
}

export interface Teammate {
  id: number;
  name: string;
  min: number;
  pts: number;
  injury: Injury | null;
  out: boolean;
}

export interface Game {
  opponent: string;
  home: boolean;
  date: string;
  rest_days: number;
  back_to_back: boolean;
  source: "schedule" | "manual";
  teammates_out: number;
  teammates?: Teammate[];
}

export interface TrackerRow {
  game_date: string;
  player_id: number;
  player: string;
  stat: Stat;
  line: number;
  over_odds: number;
  under_odds: number;
  books: number;
  p_over: number;
  pick: "over" | "under" | "pass";
  pick_odds: number | null;
  edge: number | null;
  actual: number | null;
  result: "over" | "under" | "push" | null;
  profit: number | null;
}

export interface Tracker {
  enabled: boolean;
  summary: {
    props_tracked: number;
    graded: number;
    picks: number;
    wins: number;
    losses: number;
    pushes: number;
    units: number;
    roi: number | null;
    brier_model: number | null;
    pending: number;
  };
  rows: TrackerRow[];
}

export interface Quantiles {
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

export interface StatLine {
  stat: Stat;
  label: string;
  mean: number;
  median: number;
  quantiles: Quantiles;
  sd: number;
  season_avg: number | null;
  last_5: number | null;
  last_10: number | null;
  distribution: { value: number; p: number }[];
}

export interface FantasySummary {
  key: ScoringKey;
  name: string;
  mean: number;
  quantiles: Quantiles;
  histogram: { x: number; p: number }[];
}

export interface Explanation {
  base: number;
  prediction: number;
  steps: { group: string; delta: number; multiplier: number }[];
  top_features: { feature: string; value: number | null; effect_pct: number; group: string }[];
}

export interface Projection {
  player: Player;
  game: Game;
  stats: Record<Stat, StatLine>;
  combos: Record<string, { mean: number } & Quantiles>;
  fantasy: Record<ScoringKey, FantasySummary>;
  milestones: Record<string, number>;
  explanations: Record<Stat, Explanation>;
}

export interface GameRow {
  GAME_DATE: string;
  MATCHUP: string;
  WL: string;
  MIN: number;
  PTS: number;
  REB: number;
  AST: number;
  FG3M: number;
  STL: number;
  BLK: number;
  TOV: number;
  FGM: number;
  FGA: number;
  PLUS_MINUS: number;
}

export interface SideQuote {
  side: "over" | "under";
  odds: number;
  model_prob: number;
  implied_prob: number;
  fair_prob: number;
  edge: number;
  expected_value: number;
  kelly_full: number;
  kelly_stake: number;
}

export interface PropQuote {
  player: Player;
  game: Game;
  market: string;
  label: string;
  line: number;
  method: string;
  projection: { mean: number } & Quantiles;
  p_over: number;
  p_under: number;
  p_push: number;
  hold: number;
  over: SideQuote;
  under: SideQuote;
  recommendation: "over" | "under" | "pass";
}

export interface Compare {
  scoring: string;
  players: { player: Player; game: Game; fantasy: FantasySummary; p_best: number }[];
  p_beats: (number | null)[][];
  note: string;
}

export interface Slate {
  date: string | null;
  games: { id: string; home: string; away: string; status: string; tipoff_utc: string | null }[];
  players: { player: Player; game: Game; mean: Record<Stat, number>; fantasy: Record<ScoringKey, number> }[];
}

export interface Meta {
  version: string;
  model_trained_at: string;
  season: string;
  data_seasons: string[];
  data_loaded_at: number;
  today: string;
  targets: Stat[];
  markets: { key: string; label: string }[];
  scoring: { key: ScoringKey; name: string; weights: Record<string, number> }[];
  teams: string[];
  headline: Record<string, { MAE: number; RMSE: number }> | null;
}

export interface Metrics {
  MAE: number;
  RMSE: number;
}

export interface TargetReport extends Metrics {
  season_avg: Metrics;
  last_5: Metrics;
  coverage_80: number;
  coverage_50: number;
  pit_histogram: number[];
  pit_histogram_raw: number[];
  pit_max_deviation: number;
  log_likelihood: number;
  nb_log_likelihood: number;
  poisson_log_likelihood: number;
}

export interface Backtest {
  stat: Stat;
  games: number;
  bets: number;
  hit_rate: number | null;
  roi: number | null;
  start_bankroll: number;
  final_bankroll: number;
  max_drawdown: number;
  brier_model: number;
  brier_coin_flip: number;
  over_rate: number;
  reliability: { bin: string; predicted: number; observed: number; n: number }[];
  edge_buckets: { edge: string; bets: number; hit_rate: number; flat_roi: number }[];
  equity_curve: { date: string; bankroll: number }[];
}

export interface Report {
  version: string;
  trained_at: string;
  train_seasons: string[];
  test_season: string;
  train_rows: number;
  test_rows: number;
  n_features: number;
  optuna_trials: number;
  headline: Record<string, Metrics>;
  improvement_pct: number;
  improvement_vs_v1_pct: number;
  targets: Record<Stat, TargetReport>;
  folds: { train: string[]; test: string; role: string; rows: number; targets: Record<Stat, TargetReport> }[];
  fantasy: Record<string, { model: Metrics; season_avg: Metrics }>;
  backtests: Record<string, Backtest>;
  importance: Record<string, number>;
  copula: { stats: Stat[]; correlation: number[][] };
  duration_seconds: number;
}
