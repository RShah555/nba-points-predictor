"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { EquityChart, HBars, Heatmap, PitChart, ReliabilityChart } from "@/components/charts";
import { Card, ErrorState, Segmented, Skeleton, StatTile } from "@/components/ui";
import { api } from "@/lib/api";
import { STAT_LABEL, num, pct } from "@/lib/format";
import type { Stat } from "@/lib/types";

const PIT_STATS = ["PTS", "REB", "AST", "FG3M"] as const;
const tabs = PIT_STATS.map((s) => ({ key: s as Stat, name: STAT_LABEL[s] ?? s }));

export default function ModelPage() {
  const report = useQuery({ queryKey: ["report"], queryFn: api.report });
  const [pitStat, setPitStat] = useState<Stat>("PTS");
  const [btStat, setBtStat] = useState<Stat>("PTS");

  if (report.error) return <ErrorState error={report.error} hint="Train the model with `nba-predictor train`." />;
  const r = report.data;
  if (!r) return <Skeleton className="h-[600px]" />;

  const best = r.headline["v2 LightGBM + Optuna"]!;
  const bt = r.backtests[btStat]!;
  const t = r.targets[pitStat];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-ink">Model</h1>
        <p className="text-[14px] text-ink-3">
          v{r.version}, trained {new Date(r.trained_at).toLocaleString()}. Held out test: {r.test_rows.toLocaleString()} player games from {r.test_season},
          trained on {r.train_rows.toLocaleString()} from {r.train_seasons.join(" and ")}. {r.n_features} features, {r.optuna_trials} Optuna trials per stat.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <StatTile label="Points MAE (test season)" value={num(best.MAE, 2)} sub={`${r.improvement_pct}% better than the best baseline`} tone="good" />
        <StatTile label="vs v1 model" value={`${r.improvement_vs_v1_pct}%`} sub={`v1 MAE ${num(r.headline["v1 gradient boosting"]?.MAE, 2)}`} />
        <StatTile label="Points 80% interval coverage" value={pct(r.targets.PTS.coverage_80)} sub="whole numbers push this above 80%" />
        <StatTile label="Over/under Brier (points)" value={r.backtests.PTS?.brier_model.toFixed(4)} sub="0.25 = coin flip" tone="good" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Points: model vs baselines" subtitle="Average miss on the held out season (lower is better)">
          <table className="tabular w-full text-[14px]">
            <thead>
              <tr className="text-left text-[12px] text-ink-3">
                <th className="py-1.5 font-medium">Method</th>
                <th className="py-1.5 text-right font-medium">MAE</th>
                <th className="py-1.5 text-right font-medium">RMSE</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(r.headline).map(([name, m]) => (
                <tr key={name} className="border-t border-line">
                  <td className={`py-2 ${name.startsWith("v2") ? "font-semibold text-ink" : "text-ink-2"}`}>{name}</td>
                  <td className="py-2 text-right text-ink">{m.MAE.toFixed(2)}</td>
                  <td className="py-2 text-right text-ink-2">{m.RMSE.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3 className="mb-1 mt-4 text-[13px] font-medium text-ink-2">Fantasy points (expected vs actual)</h3>
          <table className="tabular w-full text-[14px]">
            <tbody>
              {Object.entries(r.fantasy).map(([name, f]) => (
                <tr key={name} className="border-t border-line">
                  <td className="py-2 text-ink-2">{name}</td>
                  <td className="py-2 text-right text-ink">MAE {f.model.MAE.toFixed(2)}</td>
                  <td className="py-2 text-right text-ink-3">season avg {f.season_avg.MAE.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Every stat" subtitle="MAE on the held out season, and how often the result landed inside the 80% interval">
          <div className="overflow-x-auto">
            <table className="tabular w-full text-[13px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  <th className="py-1.5 font-medium">Stat</th>
                  <th className="py-1.5 text-right font-medium">Model</th>
                  <th className="py-1.5 text-right font-medium">Season avg</th>
                  <th className="py-1.5 text-right font-medium">Last 5</th>
                  <th className="py-1.5 text-right font-medium">Better by</th>
                  <th className="py-1.5 text-right font-medium">80% cover</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(r.targets).map(([s, m]) => {
                  const base = Math.min(m.season_avg.MAE, m.last_5.MAE);
                  const gain = (base - m.MAE) / base;
                  return (
                    <tr key={s} className="border-t border-line">
                      <td className="py-1.5 text-ink-2">{STAT_LABEL[s] ?? s}</td>
                      <td className="py-1.5 text-right font-medium text-ink">{m.MAE.toFixed(2)}</td>
                      <td className="py-1.5 text-right text-ink-2">{m.season_avg.MAE.toFixed(2)}</td>
                      <td className="py-1.5 text-right text-ink-2">{m.last_5.MAE.toFixed(2)}</td>
                      <td className={`py-1.5 text-right ${gain >= 0 ? "text-good" : "text-bad"}`}>{pct(gain)}</td>
                      <td className="py-1.5 text-right text-ink-2">{pct(m.coverage_80)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Calibration (PIT histogram)"
          subtitle="If the distributions are honest, every bin holds 10% of games"
          action={<Segmented size="sm" value={pitStat} options={tabs} onChange={setPitStat} />}
        >
          <PitChart raw={t.pit_histogram_raw} calibrated={t.pit_histogram} />
          <p className="tabular mt-2 text-[12px] text-ink-3">
            Log likelihood per game: {t.log_likelihood} recalibrated · {t.nb_log_likelihood} raw negative binomial · {t.poisson_log_likelihood} Poisson
          </p>
        </Card>
        <Card
          title="Over/under reliability"
          subtitle="Predicted chance of going over the baseline line vs how often it happened"
          action={<Segmented size="sm" value={btStat} options={tabs} onChange={setBtStat} />}
        >
          <ReliabilityChart bins={bt.reliability} />
        </Card>
      </div>

      <Card
        title={`${STAT_LABEL[btStat]} prop backtest (synthetic lines)`}
        subtitle="Against a baseline book that knows only the season average (calibrated on its own past errors, 4.5% margin). Fixed stakes, no compounding. Real books also price injuries, lineups and market money, so read the ROI as an upper bound."
        action={<Segmented size="sm" value={btStat} options={tabs} onChange={setBtStat} />}
      >
        <div className="mb-4 grid grid-cols-2 gap-2 md:grid-cols-5">
          <StatTile label="Bets" value={bt.bets.toLocaleString()} sub={`of ${bt.games.toLocaleString()} games`} />
          <StatTile label="Hit rate" value={pct(bt.hit_rate)} />
          <StatTile label="ROI on stakes" value={pct(bt.roi)} tone={bt.roi != null && bt.roi > 0 ? "good" : "bad"} sub="quarter Kelly, 2%/bet cap" />
          <StatTile label="Bankroll" value={`$${Math.round(bt.final_bankroll).toLocaleString()}`} sub={`from $${bt.start_bankroll.toLocaleString()}`} />
          <StatTile label="Max drawdown" value={pct(bt.max_drawdown)} tone="bad" />
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <EquityChart curve={bt.equity_curve} start={bt.start_bankroll} />
          </div>
          <div>
            <h3 className="mb-2 text-[13px] font-medium text-ink-2">By model edge</h3>
            <table className="tabular w-full text-[13px]">
              <thead>
                <tr className="text-left text-[12px] text-ink-3">
                  <th className="py-1 font-medium">Edge</th>
                  <th className="py-1 text-right font-medium">Bets</th>
                  <th className="py-1 text-right font-medium">Hit</th>
                  <th className="py-1 text-right font-medium">Flat ROI</th>
                </tr>
              </thead>
              <tbody>
                {bt.edge_buckets.map((b) => (
                  <tr key={b.edge} className="border-t border-line">
                    <td className="py-1.5 text-ink-2">{b.edge}</td>
                    <td className="py-1.5 text-right text-ink-2">{b.bets.toLocaleString()}</td>
                    <td className="py-1.5 text-right text-ink">{pct(b.hit_rate)}</td>
                    <td className={`py-1.5 text-right ${b.flat_roi >= 0 ? "text-good" : "text-bad"}`}>{pct(b.flat_roi)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-[12px] text-ink-3">A model with real signal wins more often as its edge grows.</p>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="lg:col-span-2" title="What drives the points model" subtitle="Share of LightGBM split gain by feature family">
          <HBars rows={Object.entries(r.importance).map(([label, value]) => ({ label, value }))} />
        </Card>
        <Card className="lg:col-span-3" title="How stats move together" subtitle="Copula correlation of held out residuals: the dependence used in every joint simulation">
          <Heatmap labels={r.copula.stats} matrix={r.copula.correlation} />
        </Card>
      </div>

      <Card title="Walk forward folds">
        <div className="overflow-x-auto">
          <table className="tabular w-full text-[13px]">
            <thead>
              <tr className="text-left text-[12px] text-ink-3">
                <th className="py-1.5 font-medium">Train</th>
                <th className="py-1.5 font-medium">Test</th>
                <th className="py-1.5 font-medium">Role</th>
                <th className="py-1.5 text-right font-medium">Games</th>
                <th className="py-1.5 text-right font-medium">PTS MAE</th>
                <th className="py-1.5 text-right font-medium">Season avg</th>
                <th className="py-1.5 text-right font-medium">PTS 80% cover</th>
              </tr>
            </thead>
            <tbody>
              {r.folds.map((f) => (
                <tr key={f.test} className="border-t border-line">
                  <td className="py-1.5 text-ink-2">{f.train.join(", ")}</td>
                  <td className="py-1.5 text-ink">{f.test}</td>
                  <td className="py-1.5 text-ink-3">{f.role}</td>
                  <td className="py-1.5 text-right text-ink-2">{f.rows.toLocaleString()}</td>
                  <td className="py-1.5 text-right font-medium text-ink">{f.targets.PTS.MAE.toFixed(2)}</td>
                  <td className="py-1.5 text-right text-ink-2">{f.targets.PTS.season_avg.MAE.toFixed(2)}</td>
                  <td className="py-1.5 text-right text-ink-2">{pct(f.targets.PTS.coverage_80)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
