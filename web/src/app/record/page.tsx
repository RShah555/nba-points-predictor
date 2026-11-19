"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import Link from "next/link";

import { Card, ErrorState, Skeleton, StatTile } from "@/components/ui";
import { api, playerHref } from "@/lib/api";
import { STAT_LABEL, pct } from "@/lib/format";

export default function RecordPage() {
  const tracker = useQuery({ queryKey: ["tracker"], queryFn: api.tracker });
  if (tracker.error) return <ErrorState error={tracker.error} />;
  const t = tracker.data;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-ink">Record vs real sportsbooks</h1>
        <p className="text-[14px] text-ink-3">
          Every day the model&apos;s picks are saved against real prop lines before tip off, then graded after the games.
          This is the honest test: real lines, no hindsight.
        </p>
      </div>

      {!t ? (
        <Skeleton className="h-40" />
      ) : !t.enabled && t.summary.props_tracked === 0 ? (
        <Card title="Not switched on yet">
          <ol className="list-decimal space-y-1.5 pl-5 text-[14px] text-ink-2">
            <li>
              Get a free key at <a className="text-accent underline" href="https://the-odds-api.com" target="_blank" rel="noreferrer">the-odds-api.com</a>{" "}
              (500 requests a month covers points props for every game).
            </li>
            <li>
              Set it as <code className="rounded bg-surface-2 px-1">NBA_ODDS_API_KEY</code> where the app runs (a <code className="rounded bg-surface-2 px-1">.env</code> file locally, or a secret on the host).
            </li>
            <li>
              Once a day before games, run <code className="rounded bg-surface-2 px-1">nba-predictor lines</code>. Results fill in here automatically.
            </li>
          </ol>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <StatTile label="Record" value={`${t.summary.wins}-${t.summary.losses}${t.summary.pushes ? `-${t.summary.pushes}` : ""}`} sub={`${t.summary.picks} picks`} />
            <StatTile
              label="Profit"
              value={`${t.summary.units >= 0 ? "+" : ""}${t.summary.units.toFixed(2)}u`}
              sub="1 unit per pick"
              tone={t.summary.units >= 0 ? "good" : "bad"}
            />
            <StatTile label="Return per bet" value={pct(t.summary.roi)} />
            <StatTile label="Waiting on games" value={t.summary.pending} sub={`${t.summary.props_tracked} lines tracked`} />
          </div>
          <Card title="Latest lines">
            <div className="-mx-2 overflow-x-auto">
              <table className="tabular w-full min-w-[640px] text-[13px]">
                <thead>
                  <tr className="text-left text-[12px] text-ink-3">
                    {["Date", "Player", "Line", "Model over %", "Pick", "Result", "Profit"].map((h) => (
                      <th key={h} className="px-2 py-1.5 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {t.rows.map((r) => (
                    <tr key={`${r.game_date}-${r.player_id}-${r.stat}`} className="border-t border-line">
                      <td className="px-2 py-1.5 text-ink-3">{r.game_date}</td>
                      <td className="px-2 py-1.5"><Link className="text-ink hover:underline" href={playerHref(r.player_id)}>{r.player}</Link></td>
                      <td className="px-2 py-1.5 text-ink-2">{STAT_LABEL[r.stat]} {r.line}</td>
                      <td className="px-2 py-1.5 text-ink-2">{pct(r.p_over, 0)}</td>
                      <td className="px-2 py-1.5 capitalize text-ink">{r.pick}</td>
                      <td className="px-2 py-1.5 text-ink-2">{r.result ? `${r.result} (${r.actual})` : "pending"}</td>
                      <td className={clsx("px-2 py-1.5", r.profit == null ? "text-ink-3" : r.profit >= 0 ? "text-good" : "text-bad")}>
                        {r.profit == null ? "–" : `${r.profit >= 0 ? "+" : ""}${r.profit.toFixed(2)}`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
