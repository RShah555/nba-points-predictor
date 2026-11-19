"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Calculator, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { DistributionChart } from "@/components/charts";
import { PlayerSearch } from "@/components/player-search";
import { Avatar, Badge, Card, ErrorState, Field, StatTile, inputClass } from "@/components/ui";
import { api } from "@/lib/api";
import { americanOdds, gameLabel, num, pct, shortDate, signed } from "@/lib/format";
import type { SideQuote, Stat } from "@/lib/types";

function SideCard({ q, recommended }: { q: SideQuote; recommended: boolean }) {
  return (
    <div className={`rounded-xl p-3.5 ${recommended ? "bg-accent-wash ring-1 ring-accent" : "bg-surface-2"}`}>
      <div className="flex items-center justify-between">
        <span className="text-[15px] font-semibold capitalize text-ink">{q.side}</span>
        <span className="tabular text-[13px] text-ink-2">{q.odds > 0 ? `+${q.odds}` : q.odds}</span>
      </div>
      <dl className="tabular mt-2 grid grid-cols-2 gap-y-1 text-[13px]">
        <dt className="text-ink-3">Model probability</dt><dd className="text-right font-medium text-ink">{pct(q.model_prob)}</dd>
        <dt className="text-ink-3">Fair price (model)</dt><dd className="text-right text-ink-2">{americanOdds(q.model_prob)}</dd>
        <dt className="text-ink-3">Book, no vig</dt><dd className="text-right text-ink-2">{pct(q.fair_prob)}</dd>
        <dt className="text-ink-3">Edge</dt><dd className={`text-right font-medium ${q.edge > 0 ? "text-good" : "text-bad"}`}>{signed(q.edge * 100)} pts</dd>
        <dt className="text-ink-3">EV per $100</dt><dd className={`text-right ${q.expected_value > 0 ? "text-good" : "text-ink-2"}`}>{signed(q.expected_value * 100, 2)}</dd>
        <dt className="text-ink-3">Kelly (full)</dt><dd className="text-right text-ink-2">{pct(q.kelly_full)}</dd>
        <dt className="text-ink-3">Suggested stake</dt><dd className="text-right font-semibold text-ink">${q.kelly_stake.toFixed(2)}</dd>
      </dl>
    </div>
  );
}

function PropsLab() {
  const params = useSearchParams();
  const router = useRouter();
  const playerId = params.get("player") ? Number(params.get("player")) : null;
  const [form, setForm] = useState({ market: "PTS", line: 24.5, over_odds: -110, under_odds: -110, bankroll: 1000, kelly_multiplier: 0.25 });
  const set = (patch: Partial<typeof form>) => setForm((f) => ({ ...f, ...patch }));

  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const player = useQuery({ queryKey: ["player", playerId], queryFn: () => api.player(playerId!), enabled: playerId != null });
  const proj = useQuery({ queryKey: ["projection", playerId, null], queryFn: () => api.projection({ player_id: playerId! }), enabled: playerId != null, retry: false });
  const quote = useMutation({ mutationFn: () => api.prop({ player_id: playerId!, ...form }) });

  const single = proj.data?.stats[form.market as Stat];
  const q = quote.data;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-ink">Prop lab</h1>
        <p className="text-[14px] text-ink-3">Model probability vs the book: vig removed, edge, expected value and fractional Kelly sizing.</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-2" title="Bet">
          <div className="flex flex-col gap-3">
            {playerId && player.data ? (
              <div className="flex items-center gap-3 rounded-xl bg-surface-2 p-2.5">
                <Avatar src={player.data.player.headshot} name={player.data.player.name} size={40} />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium text-ink">{player.data.player.name}</div>
                  <div className="text-[12px] text-ink-3">
                    {proj.data ? `${shortDate(proj.data.game.date)} ${gameLabel(proj.data.game)}` : player.data.player.team}
                  </div>
                </div>
                <button aria-label="Change player" onClick={() => router.replace("/props")} className="rounded-md p-1 text-ink-3 hover:bg-surface-3 hover:text-ink">
                  <X className="size-4" />
                </button>
              </div>
            ) : (
              <PlayerSearch autoFocus placeholder="Choose a player" onSelect={(p) => router.replace(`/props?player=${p.id}`)} />
            )}
            <div className="grid grid-cols-2 gap-3">
              <Field label="Market">
                <select className={inputClass} value={form.market} onChange={(e) => set({ market: e.target.value })}>
                  {meta.data?.markets.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
                </select>
              </Field>
              <Field label="Line">
                <input className={inputClass} type="number" step={0.5} min={0} value={form.line} onChange={(e) => set({ line: Number(e.target.value) })} />
              </Field>
              <Field label="Over odds">
                <input className={inputClass} type="number" step={5} value={form.over_odds} onChange={(e) => set({ over_odds: Number(e.target.value) })} />
              </Field>
              <Field label="Under odds">
                <input className={inputClass} type="number" step={5} value={form.under_odds} onChange={(e) => set({ under_odds: Number(e.target.value) })} />
              </Field>
              <Field label="Bankroll ($)">
                <input className={inputClass} type="number" min={1} value={form.bankroll} onChange={(e) => set({ bankroll: Number(e.target.value) })} />
              </Field>
              <Field label={`Kelly fraction: ${Math.round(form.kelly_multiplier * 100)}%`}>
                <input type="range" min={0.05} max={1} step={0.05} value={form.kelly_multiplier} onChange={(e) => set({ kelly_multiplier: Number(e.target.value) })} className="h-9 accent-[var(--accent)]" />
              </Field>
            </div>
            <button
              disabled={!playerId || quote.isPending}
              onClick={() => quote.mutate()}
              className="flex h-10 items-center justify-center gap-2 rounded-lg bg-accent font-medium text-white disabled:opacity-50"
            >
              <Calculator className="size-4" /> {quote.isPending ? "Pricing…" : "Price it"}
            </button>
            {quote.error && <ErrorState error={quote.error} />}
          </div>
        </Card>

        <div className="flex flex-col gap-4 lg:col-span-3">
          {q ? (
            <Card
              title={`${q.label} ${q.line}`}
              subtitle={`${q.method}. Projection ${num(q.projection.mean)} (80%: ${q.projection.p10}–${q.projection.p90}). Book hold ${pct(q.hold)}.`}
              action={<Badge tone={q.recommendation === "pass" ? "neutral" : "accent"}>{q.recommendation === "pass" ? "Pass: no edge" : `Value on the ${q.recommendation}`}</Badge>}
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <SideCard q={q.over} recommended={q.recommendation === "over"} />
                <SideCard q={q.under} recommended={q.recommendation === "under"} />
              </div>
              {q.p_push > 0 && <p className="mt-2 text-[12px] text-ink-3">Push (exactly {q.line}): {pct(q.p_push)}</p>}
            </Card>
          ) : (
            <Card title="How it works">
              <ol className="list-decimal space-y-1.5 pl-5 text-[14px] text-ink-2">
                <li>The model builds a calibrated distribution for the stat (combos and fantasy points come from 10,000 correlated simulations).</li>
                <li>The book&apos;s odds are converted to probabilities and the vig is removed proportionally to get a fair price.</li>
                <li>Edge is the model&apos;s probability minus the fair probability; expected value uses the actual odds you would be paid.</li>
                <li>Stake is fractional Kelly (25% by default) capped at 5% of bankroll, because model error makes full Kelly far too aggressive.</li>
              </ol>
            </Card>
          )}
          {single && (
            <Card title="Distribution vs the line" subtitle="Bars in blue are the model's middle 80%">
              <DistributionChart data={single.distribution} quantiles={single.quantiles} mean={single.mean} line={form.line} unit={single.label.toLowerCase()} height={200} />
            </Card>
          )}
          {!single && q && (
            <div className="grid grid-cols-3 gap-2">
              <StatTile label="P(over)" value={pct(q.p_over)} />
              <StatTile label="P(under)" value={pct(q.p_under)} />
              <StatTile label="Median" value={num(q.projection.p50)} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PropsPage() {
  return (
    <Suspense>
      <PropsLab />
    </Suspense>
  );
}
