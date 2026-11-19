import type { Compare, GameRow, Meta, Player, Projection, PropQuote, Report, ScoringKey, Slate, Tracker } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export interface GameInput {
  player_id: number;
  opponent?: string | null;
  home?: boolean | null;
  rest_days?: number | null;
  teammates_out?: number[] | null;
}

export interface PropInput extends GameInput {
  market: string;
  line: number;
  over_odds: number;
  under_odds: number;
  bankroll: number;
  kelly_multiplier: number;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const playerHref = (id: number) => `/player/?id=${id}`;

export const api = {
  meta: () => request<Meta>("/meta"),
  report: () => request<Report>("/model/report"),
  search: (q: string, limit = 8) => request<Player[]>(`/players?q=${encodeURIComponent(q)}&limit=${limit}`),
  player: (id: number) => request<{ player: Player; games: GameRow[] }>(`/players/${id}`),
  projection: (input: GameInput) => request<Projection>("/projections", post(input)),
  prop: (input: PropInput) => request<PropQuote>("/props", post(input)),
  compare: (players: GameInput[], scoring: ScoringKey) => request<Compare>("/compare", post({ players, scoring })),
  slate: (minMinutes = 18) => request<Slate>(`/slate?min_minutes=${minMinutes}`),
  tracker: () => request<Tracker>("/tracker"),
};
