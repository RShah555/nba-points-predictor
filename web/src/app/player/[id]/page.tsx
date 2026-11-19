"use client";

import { useQuery } from "@tanstack/react-query";
import { CalendarDays, RotateCcw, Scale } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { DistributionChart, Histogram, TrendChart, Waterfall } from "@/components/charts";
import { Avatar, Badge, Card, ErrorState, Field, Segmented, Skeleton, StatTile, inputClass } from "@/components/ui";
import { api, type GameInput } from "@/lib/api";
import { HEADLINE_STATS, SCORING, STAT_LABEL, gameLabel, num, pct, shortDate } from "@/lib/format";
import type { Stat } from "@/lib/types";

const STAT_TABS = HEADLINE_STATS.map((s) => ({ key: s as Stat, name: STAT_LABEL[s] ?? s }));

export default function PlayerPage() {
  const id = Number(useParams<{ id: string }>().id);
  const [override, setOverride] = useState<Omit<GameInput, "player_id"> | null>(null);
  const [stat, setStat] = useState<Stat>("PTS");

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const detail = useQuery({ queryKey: ["player", id], queryFn: () => api.player(id) });
  const proj = useQuery({
    queryKey: ["projection", id, override],
    queryFn: () => api.projection({ player_id: id, ...override }),
    placeholderData: (prev) => prev,
    retry: false,
  });

  const p = proj.data;
  const game = p?.game;
  const player = p?.player ?? detail.data?.player;
  const line = p?.stats[stat];

  const setContext = (patch: Partial<Omit<GameInput, "player_id">>) =>
    setOverride({
      opponent: override?.opponent ?? game?.opponent,
      home: override?.home ?? game?.home,
      rest_days: override?.rest_days ?? game?.rest_days,
      ...patch,
    });

  return (
    <div className="flex flex-col gap-4">
      {/* header */}
      <div className="card flex flex-wrap items-center gap-4 p-4 sm:p-5">
        {player ? <Avatar src={player.headshot} name={player.name} size={72} /> : <Skeleton className="size-[72px] rounded-full" />}
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-2xl font-semibold text-ink">{player?.name ?? "Loading…"}</h1>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-3">
            <span className="font-medium text-ink-2">{player?.team}</span>
            {player && (
              <span className="tabular">
                {num(player.pts)} pts · {num(player.reb)} reb · {num(player.ast)} ast · {num(player.min)} min · {player.games} GP
              </span>
            )}
          </div>
        </div>
        {game && (
          <div className="flex flex-col items-start gap-1 sm:items-end">
            <div className="flex items-center gap-2 text-[15px] font-semibold text-ink">
              <CalendarDays className="size-4 text-ink-3" aria-hidden />
              {shortDate(game.date)} {gameLabel(game)}
            </div>
            <div className="flex gap-1.5">
              <Badge tone={game.source === "schedule" ? "accent" : "neutral"}>
                {game.source === "schedule" ? "From schedule" : "Custom scenario"}
              </Badge>
              <Badge>{game.back_to_back ? "Back to back" : `${game.rest_days} days rest`}</Badge>
            </div>
          </div>
        )}
      </div>

      {/* scenario controls */}
      <div className="card flex flex-wrap items-end gap-3 p-3">
        <Field label="Opponent">
          <select
            className={inputClass}
            value={game?.opponent ?? ""}
            onChange={(e) => setContext({ opponent: e.target.value })}
          >
            {meta.data?.teams.map((t) => <option key={t}>{t}</option>)}
          </select>
        </Field>
        <Field label="Location">
          <Segmented
            size="sm"
            value={game?.home ? "home" : "away"}
            options={[{ key: "home", name: "Home" }, { key: "away", name: "Away" }]}
            onChange={(v) => setContext({ home: v === "home" })}
          />
        </Field>
        <Field label={`Days of rest: ${game?.rest_days ?? "–"}`}>
          <input
            type="range"
            min={1}
            max={7}
            value={Math.min(game?.rest_days ?? 2, 7)}
            onChange={(e) => setContext({ rest_days: Number(e.target.value) })}
            className="h-9 w-40 accent-[var(--accent)]"
          />
        </Field>
        {override && (
          <button onClick={() => setOverride(null)} className="flex h-9 items-center gap-1.5 rounded-lg px-3 text-[13px] text-ink-2 hover:bg-surface-2">
            <RotateCcw className="size-3.5" /> Next scheduled game
          </button>
        )}
        <Link href={`/props?player=${id}`} className="ml-auto flex h-9 items-center gap-1.5 rounded-lg bg-accent px-3 text-[13px] font-medium text-white">
          <Scale className="size-3.5" /> Price a prop
        </Link>
      </div>

      {proj.error && <ErrorState error={proj.error} hint="Pick an opponent above to project a custom game." />}

      {/* headline stat tiles */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
        {HEADLINE_STATS.map((s) =>
          p ? (
            <button key={s} onClick={() => setStat(s)} className={`rounded-xl text-left ring-accent ${stat === s ? "ring-2" : ""}`}>
              <StatTile
                label={STAT_LABEL[s] ?? s}
                value={num(p.stats[s].mean)}
                sub={`${p.stats[s].quantiles.p10}–${p.stats[s].quantiles.p90} · avg ${num(p.stats[s].season_avg)}`}
              />
            </button>
          ) : (
            <Skeleton key={s} className="h-[84px]" />
          ),
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card
          className="lg:col-span-3"
          title={`${STAT_LABEL[stat]} distribution`}
          subtitle={line ? `Median ${line.median}, 80% of outcomes between ${line.quantiles.p10} and ${line.quantiles.p90}` : undefined}
          action={<Segmented size="sm" value={stat} options={STAT_TABS.slice(0, 4)} onChange={setStat} />}
        >
          {line ? (
            <>
              <DistributionChart data={line.distribution} quantiles={line.quantiles} mean={line.mean} unit={STAT_LABEL[stat]?.toLowerCase() ?? ""} height={220} />
              <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-5">
                {(["p10", "p25", "p50", "p75", "p90"] as const).map((q) => (
                  <div key={q} className="rounded-lg bg-surface-2 px-2.5 py-1.5">
                    <div className="text-[11px] text-ink-3">{q === "p50" ? "Median" : `${q.slice(1)}th pct`}</div>
                    <div className="tabular text-[15px] font-semibold text-ink">{line.quantiles[q]}</div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <Skeleton className="h-64" />
          )}
        </Card>

        <Card className="lg:col-span-2" title="Why this projection" subtitle="SHAP contributions by feature family, in the stat's own units">
          {p ? (
            <>
              <Waterfall explanation={p.explanations[stat]} unit={STAT_LABEL[stat]?.toLowerCase() ?? ""} />
              <details className="mt-3 text-[13px]">
                <summary className="cursor-pointer text-ink-3 hover:text-ink-2">Top individual features</summary>
                <ul className="mt-2 flex flex-col gap-1">
                  {p.explanations[stat].top_features.map((f) => (
                    <li key={f.feature} className="tabular flex justify-between gap-2">
                      <span className="truncate text-ink-2">{f.feature} <span className="text-ink-3">= {num(f.value, 2)}</span></span>
                      <span className={f.effect_pct >= 0 ? "text-ink" : "text-ink-2"}>{f.effect_pct >= 0 ? "+" : ""}{f.effect_pct.toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </details>
            </>
          ) : (
            <Skeleton className="h-64" />
          )}
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {SCORING.map((s, i) => {
          const f = p?.fantasy[s.key];
          return (
            <Card key={s.key} title={`${s.name} fantasy`} subtitle={f ? `80% range ${f.quantiles.p10}–${f.quantiles.p90}` : undefined}
              action={f && <span className="tabular text-2xl font-semibold text-ink">{num(f.mean)}</span>}>
              {f ? <Histogram data={f.histogram} unit="fantasy points" color={`var(--series-${i + 1})`} marker={f.mean} height={120} /> : <Skeleton className="h-28" />}
            </Card>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Milestones and combos" subtitle="From 10,000 joint simulations (stats are correlated through a Gaussian copula)">
          {p ? (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <StatTile label="Double-double" value={pct(p.milestones.double_double)} />
              <StatTile label="Triple-double" value={pct(p.milestones.triple_double)} />
              <StatTile label="20+ points" value={pct(p.milestones.pts_20_plus)} />
              <StatTile label="30+ points" value={pct(p.milestones.pts_30_plus)} />
              <StatTile label="3+ threes" value={pct(p.milestones.fg3m_3_plus)} />
              {Object.entries(p.combos).slice(0, 4).map(([k, c]) => (
                <StatTile key={k} label={k === "STOCKS" ? "Stl + Blk" : k} value={num(c.mean)} sub={`${c.p10}–${c.p90}`} />
              ))}
            </div>
          ) : (
            <Skeleton className="h-48" />
          )}
        </Card>

        <Card title={`${STAT_LABEL[stat]} by game`} subtitle="Last 25 games">
          {detail.data ? (
            <TrendChart games={detail.data.games} stat={stat as "PTS"} projection={line?.mean} />
          ) : (
            <Skeleton className="h-52" />
          )}
        </Card>
      </div>

      <Card title="Game log">
        {detail.data ? (
          <div className="-mx-2 overflow-x-auto">
            <table className="tabular w-full min-w-[720px] text-[13px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  {["Date", "Matchup", "", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV", "FG", "+/-"].map((h, i) => (
                    <th key={i} className={`px-2 py-1.5 font-medium ${i > 2 ? "text-right" : ""}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detail.data.games.map((g) => (
                  <tr key={g.GAME_DATE + g.MATCHUP} className="border-t border-line">
                    <td className="px-2 py-1.5 text-ink-3">{g.GAME_DATE}</td>
                    <td className="px-2 py-1.5 text-ink-2">{g.MATCHUP}</td>
                    <td className={`px-2 py-1.5 ${g.WL === "W" ? "text-good" : "text-bad"}`}>{g.WL}</td>
                    {[g.MIN, g.PTS, g.REB, g.AST, g.FG3M, g.STL, g.BLK, g.TOV].map((v, i) => (
                      <td key={i} className={`px-2 py-1.5 text-right ${i === 1 ? "font-semibold text-ink" : "text-ink-2"}`}>{v}</td>
                    ))}
                    <td className="px-2 py-1.5 text-right text-ink-2">{g.FGM}-{g.FGA}</td>
                    <td className="px-2 py-1.5 text-right text-ink-2">{g.PLUS_MINUS > 0 ? `+${g.PLUS_MINUS}` : g.PLUS_MINUS}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : detail.error ? (
          <ErrorState error={detail.error} />
        ) : (
          <Skeleton className="h-64" />
        )}
      </Card>
    </div>
  );
}
