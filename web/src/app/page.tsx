"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { Avatar, Card, ErrorState, Segmented, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { SCORING, gameLabel, num, shortDate } from "@/lib/format";
import type { ScoringKey } from "@/lib/types";

const tipoff = (utc: string | null) =>
  utc ? new Date(utc).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : "";

export default function SlatePage() {
  const [scoring, setScoring] = useState<ScoringKey>("espn");
  const [game, setGame] = useState<string | null>(null);
  const slate = useQuery({ queryKey: ["slate"], queryFn: () => api.slate() });

  const players = useMemo(() => {
    const all = slate.data?.players ?? [];
    const g = slate.data?.games.find((x) => x.id === game);
    const filtered = g ? all.filter((p) => p.player.team === g.home || p.player.team === g.away) : all;
    return [...filtered].sort((a, b) => b.fantasy[scoring] - a.fantasy[scoring]);
  }, [slate.data, game, scoring]);

  if (slate.error) return <ErrorState error={slate.error} hint="Is the API running? Start it with `nba-predictor serve`." />;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-ink">Slate</h1>
          <p className="text-[14px] text-ink-3">
            {slate.data?.date ? `${shortDate(slate.data.date)} · ${slate.data.games.length} games` : "Loading the next game day…"}
          </p>
        </div>
        <Segmented value={scoring} options={SCORING} onChange={setScoring} />
      </div>

      <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1">
        {slate.isLoading && Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-16 w-36 shrink-0" />)}
        {slate.data?.games.map((g) => (
          <button
            key={g.id}
            onClick={() => setGame(game === g.id ? null : g.id)}
            aria-pressed={game === g.id}
            className={clsx(
              "card shrink-0 px-3.5 py-2.5 text-left transition-colors",
              game === g.id ? "border-accent! bg-accent-wash!" : "hover:bg-surface-2",
            )}
          >
            <div className="text-[14px] font-semibold text-ink">
              {g.away} <span className="text-ink-3">@</span> {g.home}
            </div>
            <div className="text-[12px] text-ink-3">{g.status === "final" ? "Final" : g.status === "live" ? "Live" : tipoff(g.tipoff_utc)}</div>
          </button>
        ))}
      </div>

      <Card
        title="Projected fantasy leaders"
        subtitle={`Expected ${SCORING.find((s) => s.key === scoring)?.name} points, rotation players (18+ minutes recently). Pick a player for the full distribution.`}
      >
        {slate.isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-12" />)}
          </div>
        ) : players.length === 0 ? (
          <p className="text-[14px] text-ink-3">No games scheduled.</p>
        ) : (
          <div className="-mx-2 overflow-x-auto">
            <table className="tabular w-full min-w-[640px] text-[14px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  <th className="w-10 px-2 py-2 font-medium">#</th>
                  <th className="px-2 py-2 font-medium">Player</th>
                  <th className="px-2 py-2 font-medium">Game</th>
                  {["PTS", "REB", "AST", "3PM"].map((h) => (
                    <th key={h} className="px-2 py-2 text-right font-medium">{h}</th>
                  ))}
                  <th className="px-2 py-2 text-right font-medium">FPTS</th>
                  <th className="w-6" />
                </tr>
              </thead>
              <tbody>
                {players.map((p, i) => (
                  <tr key={p.player.id} className="group border-t border-line hover:bg-surface-2">
                    <td className="px-2 py-2 text-ink-3">{i + 1}</td>
                    <td className="px-2 py-2">
                      <Link href={`/player/${p.player.id}`} className="flex items-center gap-2.5">
                        <Avatar src={p.player.headshot} name={p.player.name} size={34} />
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-ink">{p.player.name}</span>
                          <span className="block text-[12px] text-ink-3">{p.player.team}</span>
                        </span>
                      </Link>
                    </td>
                    <td className="px-2 py-2 text-ink-2">
                      {gameLabel(p.game)}
                      {p.game.back_to_back && <span className="ml-1.5 text-[11px] text-ink-3">B2B</span>}
                    </td>
                    <td className="px-2 py-2 text-right text-ink">{num(p.mean.PTS)}</td>
                    <td className="px-2 py-2 text-right text-ink-2">{num(p.mean.REB)}</td>
                    <td className="px-2 py-2 text-right text-ink-2">{num(p.mean.AST)}</td>
                    <td className="px-2 py-2 text-right text-ink-2">{num(p.mean.FG3M)}</td>
                    <td className="px-2 py-2 text-right font-semibold text-ink">{num(p.fantasy[scoring])}</td>
                    <td className="pr-2">
                      <Link href={`/player/${p.player.id}`} aria-label={`Open ${p.player.name}`}>
                        <ChevronRight className="size-4 text-ink-3 group-hover:text-ink" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
