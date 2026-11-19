export const STAT_LABEL: Record<string, string> = {
  PTS: "Points",
  REB: "Rebounds",
  AST: "Assists",
  FG3M: "3PM",
  STL: "Steals",
  BLK: "Blocks",
  TOV: "Turnovers",
  FGM: "FGM",
  FGA: "FGA",
  FTM: "FTM",
  FTA: "FTA",
};

export const HEADLINE_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"] as const;

export const SCORING = [
  { key: "espn", name: "ESPN" },
  { key: "yahoo", name: "Yahoo" },
  { key: "sleeper", name: "Sleeper" },
] as const;

export const num = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;

export const signed = (v: number, digits = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

export const americanOdds = (p: number) => {
  if (p <= 0 || p >= 1) return "–";
  const o = p >= 0.5 ? (-100 * p) / (1 - p) : (100 * (1 - p)) / p;
  return o > 0 ? `+${Math.round(o)}` : `${Math.round(o)}`;
};

export const gameLabel = (g: { opponent: string; home: boolean }) => `${g.home ? "vs" : "@"} ${g.opponent}`;

export const shortDate = (iso: string) =>
  new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
