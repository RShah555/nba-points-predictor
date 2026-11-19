"use client";

import { useQuery } from "@tanstack/react-query";
import { Trophy, X } from "lucide-react";
import { useState } from "react";

import { OverlayChart, SERIES } from "@/components/charts";
import { PlayerSearch } from "@/components/player-search";
import { Avatar, Card, ErrorState, Segmented, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { SCORING, gameLabel, num, pct, shortDate } from "@/lib/format";
import type { Player, ScoringKey } from "@/lib/types";

export default function ComparePage() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [scoring, setScoring] = useState<ScoringKey>("espn");
  const ids = players.map((p) => p.id);
  const result = useQuery({
    queryKey: ["compare", ids, scoring],
    queryFn: () => api.compare(ids.map((id) => ({ player_id: id })), scoring),
    enabled: ids.length >= 2,
    placeholderData: (prev) => prev,
    retry: false,
  });
  const data = ids.length >= 2 ? result.data : undefined;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-ink">Start / sit</h1>
          <p className="text-[14px] text-ink-3">Simulate each player&apos;s next game and see who is most likely to score the most fantasy points.</p>
        </div>
        <Segmented value={scoring} options={SCORING} onChange={setScoring} />
      </div>

      <Card>
        <div className="flex flex-wrap items-center gap-2">
          {players.map((p, i) => (
            <span key={p.id} className="flex items-center gap-2 rounded-full bg-surface-2 py-1 pl-1 pr-2 text-[13px] text-ink">
              <span className="size-2.5 rounded-full" style={{ background: SERIES[i] }} aria-hidden />
              <Avatar src={p.headshot} name={p.name} size={24} />
              {p.name}
              <button aria-label={`Remove ${p.name}`} onClick={() => setPlayers(players.filter((x) => x.id !== p.id))} className="text-ink-3 hover:text-ink">
                <X className="size-3.5" />
              </button>
            </span>
          ))}
          {players.length < 6 && (
            <PlayerSearch
              className="w-full sm:w-72"
              placeholder={players.length < 2 ? "Add at least two players" : "Add another player"}
              onSelect={(p) => !ids.includes(p.id) && setPlayers([...players, p])}
            />
          )}
        </div>
      </Card>

      {result.error && ids.length >= 2 && <ErrorState error={result.error} />}
      {ids.length >= 2 && !data && <Skeleton className="h-72" />}

      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[...data.players].map((row, i) => ({ row, i })).sort((a, b) => b.row.p_best - a.row.p_best).map(({ row, i }, rank) => (
              <div key={row.player.id} className={`card p-4 ${rank === 0 ? "ring-1 ring-accent" : ""}`}>
                <div className="flex items-center gap-3">
                  <Avatar src={row.player.headshot} name={row.player.name} size={44} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 truncate font-medium text-ink">
                      <span className="size-2.5 shrink-0 rounded-full" style={{ background: SERIES[i] }} aria-hidden />
                      {row.player.name}
                      {rank === 0 && <Trophy className="size-3.5 text-accent" aria-label="Most likely top scorer" />}
                    </div>
                    <div className="text-[12px] text-ink-3">{shortDate(row.game.date)} {gameLabel(row.game)}</div>
                  </div>
                </div>
                <div className="tabular mt-3 grid grid-cols-3 gap-2 text-center">
                  <div><div className="text-[11px] text-ink-3">Projected</div><div className="text-lg font-semibold text-ink">{num(row.fantasy.mean)}</div></div>
                  <div><div className="text-[11px] text-ink-3">80% range</div><div className="text-[14px] text-ink-2">{row.fantasy.quantiles.p10}–{row.fantasy.quantiles.p90}</div></div>
                  <div><div className="text-[11px] text-ink-3">P(best)</div><div className="text-lg font-semibold text-ink">{pct(row.p_best, 0)}</div></div>
                </div>
              </div>
            ))}
          </div>

          <div className="grid gap-4 lg:grid-cols-5">
            <Card className="lg:col-span-3" title={`${data.scoring} fantasy point distributions`}>
              <OverlayChart series={data.players.map((p) => ({ name: p.player.name, data: p.fantasy.histogram }))} />
            </Card>
            <Card className="lg:col-span-2" title="Head to head" subtitle="Chance the row player outscores the column player">
              <div className="overflow-x-auto">
                <table className="tabular w-full text-[13px]">
                  <thead>
                    <tr>
                      <th />
                      {data.players.map((p) => (
                        <th key={p.player.id} className="px-1.5 py-1 text-right font-medium text-ink-3">{p.player.name.split(" ").slice(-1)[0]}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.p_beats.map((row, i) => (
                      <tr key={i} className="border-t border-line">
                        <th className="py-1.5 pr-2 text-left font-medium text-ink-2">{data.players[i]?.player.name.split(" ").slice(-1)[0]}</th>
                        {row.map((v, j) => (
                          <td key={j} className={`px-1.5 py-1.5 text-right ${v != null && v > 0.5 ? "font-semibold text-ink" : "text-ink-3"}`}>{v == null ? "–" : pct(v, 0)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-[12px] text-ink-3">{data.note}</p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
