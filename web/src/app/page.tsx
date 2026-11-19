"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { Avatar, ErrorState, InjuryBadge, Segmented, Skeleton } from "@/components/ui";
import { api, playerHref } from "@/lib/api";
import { SCORING, gameLabel, num, shortDate } from "@/lib/format";
import type { ScoringKey } from "@/lib/types";

const tipoff = (utc: string | null) =>
  utc ? new Date(utc).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : "";

export default function TonightPage() {
  const [scoring, setScoring] = useState<ScoringKey>("espn");
  const [game, setGame] = useState<string | null>(null);
  const slate = useQuery({ queryKey: ["slate"], queryFn: () => api.slate() });

  const players = useMemo(() => {
    const all = slate.data?.players ?? [];
    const g = slate.data?.games.find((x) => x.id === game);
    const filtered = g ? all.filter((p) => p.player.team === g.home || p.player.team === g.away) : all;
    return [...filtered].sort((a, b) => b.fantasy[scoring] - a.fantasy[scoring]);
  }, [slate.data, game, scoring]);

  if (slate.error) return <ErrorState error={slate.error} hint="The app may still be starting up. Try again in a minute." />;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-ink">
            {slate.data?.date ? `Games ${shortDate(slate.data.date)}` : "Tonight"}
          </h1>
          <p className="text-[14px] text-ink-3">
            Projected fantasy points for everyone playing. Click a player to see the full breakdown.
          </p>
        </div>
        <Segmented value={scoring} options={SCORING} onChange={setScoring} />
      </div>

      {slate.isLoading && slate.failureCount > 0 && (
        <div className="card px-4 py-3 text-[14px] text-ink-2">
          Waking up the server and loading the latest games. This takes about a minute after a quiet spell.
        </div>
      )}

      <div className="-mx-4 flex gap-2 overflow-x-auto px-4 pb-1">
        {slate.isLoading && Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-14 w-32 shrink-0" />)}
        {slate.data?.games.map((g) => (
          <button
            key={g.id}
            onClick={() => setGame(game === g.id ? null : g.id)}
            aria-pressed={game === g.id}
            className={clsx(
              "card shrink-0 px-3.5 py-2 text-left transition-colors",
              game === g.id ? "border-accent! bg-accent-wash!" : "hover:bg-surface-2",
            )}
          >
            <div className="text-[14px] font-semibold text-ink">
              {g.away} <span className="text-ink-3">@</span> {g.home}
            </div>
            <div className="text-[12px] text-ink-3">
              {g.status === "final" ? "Final" : g.status === "live" ? "Live" : tipoff(g.tipoff_utc)}
            </div>
          </button>
        ))}
      </div>

      <section className="card overflow-hidden">
        {slate.isLoading ? (
          <div className="flex flex-col gap-2 p-4">
            {Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-12" />)}
          </div>
        ) : players.length === 0 ? (
          <p className="p-5 text-[14px] text-ink-3">No upcoming games.</p>
        ) : (
          <ul>
            {players.map((p, i) => (
              <li key={p.player.id} className="border-t border-line first:border-t-0">
                <Link
                  href={playerHref(p.player.id)}
                  className="group flex items-center gap-3 px-4 py-2.5 hover:bg-surface-2 sm:gap-4"
                >
                  <span className="tabular w-6 text-right text-[13px] text-ink-3">{i + 1}</span>
                  <Avatar src={p.player.headshot} name={p.player.name} size={38} />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="truncate font-medium text-ink">{p.player.name}</span>
                      <InjuryBadge injury={p.player.injury} />
                    </span>
                    <span className="block text-[12px] text-ink-3">
                      {p.player.team} {gameLabel(p.game)}
                      {p.game.teammates_out > 0 && ` · ${p.game.teammates_out} teammate${p.game.teammates_out > 1 ? "s" : ""} out`}
                    </span>
                  </span>
                  <span className="tabular hidden text-[13px] text-ink-2 sm:block">
                    {num(p.mean.PTS)} pts · {num(p.mean.REB)} reb · {num(p.mean.AST)} ast
                  </span>
                  <span className="tabular w-14 text-right">
                    <span className="block text-[16px] font-semibold text-ink">{num(p.fantasy[scoring])}</span>
                    <span className="block text-[11px] text-ink-3">fantasy</span>
                  </span>
                  <ChevronRight className="size-4 text-ink-3 group-hover:text-ink" aria-hidden />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
