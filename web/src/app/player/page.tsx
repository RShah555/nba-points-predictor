"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { CalendarDays, RotateCcw, Scale } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { DistributionChart, Histogram, TrendChart, Waterfall } from "@/components/charts";
import {
  Advanced,
  Avatar,
  Card,
  ErrorState,
  Field,
  InjuryBadge,
  Segmented,
  Skeleton,
  StatTile,
  inputClass,
} from "@/components/ui";
import { api, type GameInput } from "@/lib/api";
import { HEADLINE_STATS, SCORING, STAT_LABEL, gameLabel, num, pct, shortDate } from "@/lib/format";
import type { Stat } from "@/lib/types";

type Scenario = Omit<GameInput, "player_id">;

function PlayerView({ id }: { id: number }) {
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [stat, setStat] = useState<Stat>("PTS");

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const detail = useQuery({ queryKey: ["player", id], queryFn: () => api.player(id) });
  const proj = useQuery({
    queryKey: ["projection", id, scenario],
    queryFn: () => api.projection({ player_id: id, ...scenario }),
    placeholderData: (prev) => prev,
    retry: false,
  });

  const p = proj.data;
  const game = p?.game;
  const player = p?.player ?? detail.data?.player;
  const line = p?.stats[stat];
  const label = STAT_LABEL[stat]?.toLowerCase() ?? "";

  const change = (patch: Scenario) =>
    setScenario({
      opponent: scenario?.opponent ?? game?.opponent,
      home: scenario?.home ?? game?.home,
      rest_days: scenario?.rest_days ?? game?.rest_days,
      teammates_out: scenario?.teammates_out ?? game?.teammates?.filter((t) => t.out).map((t) => t.id),
      ...patch,
    });
  const toggleTeammate = (tid: number) => {
    const out = new Set(game?.teammates?.filter((t) => t.out).map((t) => t.id));
    if (out.has(tid)) out.delete(tid);
    else out.add(tid);
    change({ teammates_out: [...out] });
  };

  return (
    <div className="flex flex-col gap-4">
      {/* who and when */}
      <div className="card flex flex-wrap items-center gap-4 p-4 sm:p-5">
        {player ? <Avatar src={player.headshot} name={player.name} size={64} /> : <Skeleton className="size-16 rounded-full" />}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-2xl font-semibold text-ink">{player?.name ?? "Loading…"}</h1>
            <InjuryBadge injury={player?.injury} />
          </div>
          <div className="tabular text-[14px] text-ink-3">
            {player && `${player.team} · ${num(player.pts)} pts, ${num(player.reb)} reb, ${num(player.ast)} ast this season`}
          </div>
          {player?.injury && <div className="mt-1 text-[13px] text-ink-2">{player.injury.note}</div>}
        </div>
        {game && (
          <div className="text-right">
            <div className="flex items-center justify-end gap-1.5 text-[15px] font-semibold text-ink">
              <CalendarDays className="size-4 text-ink-3" aria-hidden />
              {shortDate(game.date)} {gameLabel(game)}
            </div>
            <div className="text-[13px] text-ink-3">
              {game.back_to_back ? "Second night of a back to back" : `${game.rest_days} days of rest`}
            </div>
          </div>
        )}
      </div>

      {player?.injury?.out && (
        <div className="card border-bad! px-4 py-3 text-[14px] text-ink-2">
          <b className="text-bad">{player.name} is listed {player.injury.status.toLowerCase()}.</b> The numbers below show what
          to expect if he does play.
        </div>
      )}

      {proj.error && <ErrorState error={proj.error} hint="Pick an opponent under “Change the matchup”." />}

      {/* the answer */}
      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[13px] text-ink-3">Projected {label}</div>
              {line ? (
                <>
                  <div className="tabular text-5xl font-semibold tracking-tight text-ink">{num(line.mean)}</div>
                  <p className="mt-1 text-[14px] text-ink-2">
                    Usually between <b className="text-ink">{line.quantiles.p10}</b> and{" "}
                    <b className="text-ink">{line.quantiles.p90}</b>. Season average {num(line.season_avg)}, last 5 games{" "}
                    {num(line.last_5)}.
                  </p>
                </>
              ) : (
                <Skeleton className="mt-1 h-16 w-48" />
              )}
            </div>
            <Segmented
              size="sm"
              value={stat}
              options={(["PTS", "REB", "AST", "FG3M"] as Stat[]).map((s) => ({ key: s, name: STAT_LABEL[s] ?? s }))}
              onChange={setStat}
            />
          </div>
          {line ? (
            <DistributionChart data={line.distribution} quantiles={line.quantiles} mean={line.mean} unit={label} height={200} />
          ) : (
            <Skeleton className="h-52" />
          )}
        </Card>

        <Card className="lg:col-span-2" title="What's driving it" subtitle={`How the projection moves away from a typical player's ${label}`}>
          {p ? <Waterfall explanation={p.explanations[stat]} unit={label} /> : <Skeleton className="h-56" />}
        </Card>
      </div>

      {/* who's playing */}
      <Card
        title="Who's playing"
        subtitle="Teammates marked out come from the injury report. Tap anyone to see the projection if they sit or play."
        action={
          scenario && (
            <button onClick={() => setScenario(null)} className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] text-ink-2 hover:bg-surface-2">
              <RotateCcw className="size-3.5" /> Reset
            </button>
          )
        }
      >
        {game?.teammates ? (
          <div className="flex flex-wrap gap-2">
            {game.teammates.map((t) => (
              <button
                key={t.id}
                onClick={() => toggleTeammate(t.id)}
                aria-pressed={t.out}
                title={t.injury?.note}
                className={clsx(
                  "flex items-center gap-2 rounded-full border px-3 py-1.5 text-[13px] transition-colors",
                  t.out ? "border-transparent bg-surface-2 text-ink-3 line-through" : "border-line text-ink hover:bg-surface-2",
                )}
              >
                {t.name}
                <span className="tabular text-[11px] text-ink-3 no-underline">{Math.round(t.min)} min</span>
                {t.injury && !t.out && <span className="text-[11px] text-ink-3">GTD</span>}
              </button>
            ))}
          </div>
        ) : (
          <Skeleton className="h-10" />
        )}
        <details className="mt-4">
          <summary className="cursor-pointer text-[13px] text-ink-3 hover:text-ink-2">Change the matchup</summary>
          <div className="mt-3 flex flex-wrap items-end gap-3">
            <Field label="Opponent">
              <select className={inputClass} value={game?.opponent ?? ""} onChange={(e) => change({ opponent: e.target.value })}>
                {meta.data?.teams.map((t) => <option key={t}>{t}</option>)}
              </select>
            </Field>
            <Field label="Location">
              <Segmented
                size="sm"
                value={game?.home ? "home" : "away"}
                options={[{ key: "home", name: "Home" }, { key: "away", name: "Away" }]}
                onChange={(v) => change({ home: v === "home" })}
              />
            </Field>
            <Field label={`Days of rest: ${game?.rest_days ?? "–"}`}>
              <input
                type="range"
                min={1}
                max={7}
                value={Math.min(game?.rest_days ?? 2, 7)}
                onChange={(e) => change({ rest_days: Number(e.target.value) })}
                className="h-9 w-40 accent-[var(--accent)]"
              />
            </Field>
          </div>
        </details>
      </Card>

      {/* the rest, compact */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        {HEADLINE_STATS.map((s) =>
          p ? (
            <button key={s} onClick={() => setStat(s)} className={clsx("rounded-xl text-left ring-accent", stat === s && "ring-2")}>
              <StatTile label={STAT_LABEL[s] ?? s} value={num(p.stats[s].mean)} sub={`${p.stats[s].quantiles.p10}–${p.stats[s].quantiles.p90}`} />
            </button>
          ) : (
            <Skeleton key={s} className="h-[84px]" />
          ),
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Fantasy points">
          {p ? (
            <div className="grid grid-cols-3 gap-2">
              {SCORING.map((s) => (
                <StatTile key={s.key} label={s.name} value={num(p.fantasy[s.key].mean)} sub={`${p.fantasy[s.key].quantiles.p10}–${p.fantasy[s.key].quantiles.p90}`} />
              ))}
            </div>
          ) : (
            <Skeleton className="h-24" />
          )}
        </Card>
        <Card title="Chances">
          {p ? (
            <div className="grid grid-cols-3 gap-2">
              <StatTile label="Double-double" value={pct(p.milestones.double_double, 0)} />
              <StatTile label="20+ points" value={pct(p.milestones.pts_20_plus, 0)} />
              <StatTile label="30+ points" value={pct(p.milestones.pts_30_plus, 0)} />
            </div>
          ) : (
            <Skeleton className="h-24" />
          )}
        </Card>
      </div>

      <Card
        title={`Recent ${label}`}
        action={
          <Link href={`/props/?player=${id}`} className="flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-white">
            <Scale className="size-3.5" /> Check a betting line
          </Link>
        }
      >
        {detail.data ? <TrendChart games={detail.data.games} stat={stat as "PTS"} projection={line?.mean} /> : <Skeleton className="h-52" />}
      </Card>

      <Advanced title="Advanced: full distribution, model inputs, game log">
        {p && line && (
          <>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {(["p10", "p25", "p50", "p75", "p90"] as const).map((q) => (
                <StatTile key={q} label={q === "p50" ? "Median" : `${q.slice(1)}th percentile`} value={line.quantiles[q]} />
              ))}
            </div>
            <div>
              <h3 className="mb-2 text-[13px] font-medium text-ink-2">Biggest individual inputs ({label})</h3>
              <ul className="flex flex-col gap-1 text-[13px]">
                {p.explanations[stat].top_features.map((f) => (
                  <li key={f.feature} className="tabular flex justify-between gap-2">
                    <span className="truncate text-ink-2">
                      {f.feature} <span className="text-ink-3">= {num(f.value, 2)}</span>
                    </span>
                    <span>{f.effect_pct >= 0 ? "+" : ""}{f.effect_pct.toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              {SCORING.map((s, i) => (
                <div key={s.key}>
                  <h3 className="mb-1 text-[13px] font-medium text-ink-2">{s.name} fantasy distribution</h3>
                  <Histogram data={p.fantasy[s.key].histogram} unit="fantasy points" color={`var(--series-${i + 1})`} marker={p.fantasy[s.key].mean} height={120} />
                </div>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
              {Object.entries(p.combos).map(([k, c]) => (
                <StatTile key={k} label={k === "STOCKS" ? "Stl + Blk" : k} value={num(c.mean)} sub={`${c.p10}–${c.p90}`} />
              ))}
            </div>
          </>
        )}
        {detail.data && (
          <div className="-mx-2 overflow-x-auto">
            <table className="tabular w-full min-w-[640px] text-[13px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  {["Date", "Matchup", "", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV"].map((h, i) => (
                    <th key={i} className={clsx("px-2 py-1.5 font-medium", i > 2 && "text-right")}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detail.data.games.map((g) => (
                  <tr key={g.GAME_DATE + g.MATCHUP} className="border-t border-line">
                    <td className="px-2 py-1.5 text-ink-3">{g.GAME_DATE}</td>
                    <td className="px-2 py-1.5 text-ink-2">{g.MATCHUP}</td>
                    <td className={clsx("px-2 py-1.5", g.WL === "W" ? "text-good" : "text-bad")}>{g.WL}</td>
                    {[g.MIN, g.PTS, g.REB, g.AST, g.FG3M, g.STL, g.BLK, g.TOV].map((v, i) => (
                      <td key={i} className={clsx("px-2 py-1.5 text-right", i === 1 ? "font-semibold text-ink" : "text-ink-2")}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Advanced>
    </div>
  );
}

function PlayerPage() {
  const id = Number(useSearchParams().get("id"));
  if (!id) return <ErrorState error="No player selected" hint="Search for a player at the top of the page." />;
  return <PlayerView key={id} id={id} />;
}

export default function Page() {
  return (
    <Suspense>
      <PlayerPage />
    </Suspense>
  );
}
