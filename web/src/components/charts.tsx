"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { num, pct, signed } from "@/lib/format";
import type { Explanation, GameRow, Quantiles } from "@/lib/types";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)"];
const MUTED_FILL = "var(--surface-3)";
const axis = { tickLine: false, axisLine: { stroke: "var(--axis)" }, tick: { fontSize: 11 } } as const;

function TipBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="card tabular px-2.5 py-1.5 text-[12px] text-ink shadow-xl" style={{ borderRadius: 10 }}>
      {children}
    </div>
  );
}

function Key({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
      <span
        className="inline-block h-0.5 w-3.5 rounded"
        style={dashed ? { borderTop: `2px dashed ${color}` } : { background: color, height: 2 }}
      />
      {label}
    </span>
  );
}

export function LegendRow({ items }: { items: { color: string; label: string; dashed?: boolean; swatch?: boolean }[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
      {items.map((i) =>
        i.swatch ? (
          <span key={i.label} className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
            <span className="inline-block size-2.5 rounded-sm" style={{ background: i.color }} />
            {i.label}
          </span>
        ) : (
          <Key key={i.label} {...i} />
        ),
      )}
    </div>
  );
}

/** Probability of each outcome; the middle 80% is highlighted, the tails recede. */
export function DistributionChart({ data, quantiles, mean, line, unit, height = 180 }: {
  data: { value: number; p: number }[];
  quantiles: Quantiles;
  mean: number;
  line?: number | null;
  unit: string;
  height?: number;
}) {
  return (
    <div>
      <LegendRow
        items={[
          { color: "var(--series-1)", label: "80% range", swatch: true },
          { color: MUTED_FILL, label: "Tails", swatch: true },
          { color: "var(--text-secondary)", label: `Mean ${num(mean)}`, dashed: true },
          ...(line != null ? [{ color: "var(--series-2)", label: `Line ${line}` }] : []),
        ]}
      />
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }} barCategoryGap={1}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="value" {...axis} interval="preserveStartEnd" minTickGap={16} />
          <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} />
          <Tooltip
            cursor={{ fill: "var(--accent-wash)" }}
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <TipBox>
                  <b>{payload[0].payload.value}</b> {unit}: {pct(payload[0].payload.p)}
                </TipBox>
              ) : null
            }
          />
          <Bar dataKey="p" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.value} fill={d.value >= quantiles.p10 && d.value <= quantiles.p90 ? "var(--series-1)" : MUTED_FILL} />
            ))}
          </Bar>
          <ReferenceLine x={Math.round(mean)} stroke="var(--text-secondary)" strokeDasharray="4 3" strokeWidth={1.5} />
          {line != null && <ReferenceLine x={Math.floor(line)} stroke="var(--series-2)" strokeWidth={2} />}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Points per game and the 5 game average, oldest to newest. */
export function TrendChart({ games, stat = "PTS", projection }: { games: GameRow[]; stat?: keyof GameRow; projection?: number }) {
  const rows = [...games].reverse().map((g, i, all) => {
    const window = all.slice(Math.max(0, i - 4), i + 1);
    const avg = window.reduce((s, x) => s + Number(x[stat]), 0) / window.length;
    return { date: g.GAME_DATE.slice(5), matchup: g.MATCHUP, value: Number(g[stat]), avg: Math.round(avg * 10) / 10 };
  });
  return (
    <div>
      <LegendRow
        items={[
          { color: "var(--series-1)", label: "Game" },
          { color: "var(--series-2)", label: "5 game average" },
          ...(projection != null ? [{ color: "var(--text-secondary)", label: `Projection ${num(projection)}`, dashed: true }] : []),
        ]}
      />
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: -22 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="date" {...axis} minTickGap={24} />
          <YAxis {...axis} axisLine={false} width={40} allowDecimals={false} />
          <Tooltip
            cursor={{ stroke: "var(--axis)" }}
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <TipBox>
                  <div className="text-ink-3">{payload[0].payload.matchup}</div>
                  <div>Game: <b>{payload[0].payload.value}</b></div>
                  <div>5 game avg: <b>{payload[0].payload.avg}</b></div>
                </TipBox>
              ) : null
            }
          />
          {projection != null && <ReferenceLine y={projection} stroke="var(--text-secondary)" strokeDasharray="4 3" />}
          <Line dataKey="value" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: "var(--series-1)" }} activeDot={{ r: 5, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
          <Line dataKey="avg" stroke="var(--series-2)" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** SHAP waterfall: start at the league baseline, each feature family pushes the projection up or down. */
export function Waterfall({ explanation, unit }: { explanation: Explanation; unit: string }) {
  const steps = explanation.steps.filter((s) => Math.abs(s.delta) >= 0.02);
  const max = Math.max(...steps.map((s) => Math.abs(s.delta)), 0.01);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between text-[12px] text-ink-3">
        <span>Typical player baseline</span>
        <span className="tabular text-ink-2">{num(explanation.base)}</span>
      </div>
      {steps.map((s) => {
        const width = `${(Math.abs(s.delta) / max) * 50}%`;
        const up = s.delta >= 0;
        return (
          <div
            key={s.group}
            className="group grid grid-cols-[minmax(0,9rem)_1fr_3.5rem] items-center gap-2 rounded-md px-1 py-0.5 hover:bg-surface-2"
            title={`${s.group}: ${signed(s.delta, 2)} ${unit} (×${s.multiplier.toFixed(3)})`}
          >
            <span className="truncate text-[13px] text-ink-2">{s.group}</span>
            <div className="relative h-3">
              <div className="absolute inset-y-0 left-1/2 w-px bg-[var(--axis)]" />
              <div
                className="absolute inset-y-0 rounded"
                style={{
                  width,
                  left: up ? "50%" : undefined,
                  right: up ? undefined : "50%",
                  background: up ? "var(--div-pos)" : "var(--div-neg)",
                }}
              />
            </div>
            <span className={`tabular text-right text-[13px] ${up ? "text-ink" : "text-ink-2"}`}>{signed(s.delta)}</span>
          </div>
        );
      })}
      <div className="flex justify-between border-t border-line pt-1.5 text-[13px] font-medium text-ink">
        <span>Projection</span>
        <span className="tabular">{num(explanation.prediction)}</span>
      </div>
    </div>
  );
}

export function Histogram({ data, color = "var(--series-1)", unit, height = 140, marker }: {
  data: { x: number; p: number }[];
  color?: string;
  unit: string;
  height?: number;
  marker?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }} barCategoryGap={1}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="x" {...axis} tickFormatter={(v: number) => String(Math.round(v))} minTickGap={20} />
        <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} />
        <Tooltip
          cursor={{ fill: "var(--accent-wash)" }}
          content={({ active, payload }) =>
            active && payload?.[0] ? (
              <TipBox>
                ~{num(payload[0].payload.x)} {unit}: {pct(payload[0].payload.p)}
              </TipBox>
            ) : null
          }
        />
        <Bar dataKey="p" fill={color} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
        {marker != null && <ReferenceLine x={marker} stroke="var(--text-secondary)" strokeDasharray="4 3" />}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Several distributions on one axis (start/sit). */
export function OverlayChart({ series }: { series: { name: string; data: { x: number; p: number }[] }[] }) {
  return (
    <div>
      <LegendRow items={series.map((s, i) => ({ color: SERIES[i] ?? SERIES[0]!, label: s.name }))} />
      <ResponsiveContainer width="100%" height={220}>
        <LineChart margin={{ top: 6, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} {...axis} tickFormatter={(v: number) => String(Math.round(v))} />
          <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} />
          <Tooltip
            cursor={{ stroke: "var(--axis)" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TipBox>
                  <div className="text-ink-3">~{num(Number(label))} fantasy points</div>
                  {payload.map((p) => (
                    <div key={String(p.name)}>
                      {p.name}: <b>{pct(Number(p.value))}</b>
                    </div>
                  ))}
                </TipBox>
              ) : null
            }
          />
          {series.map((s, i) => (
            <Line key={s.name} name={s.name} data={s.data} dataKey="p" type="monotone" stroke={SERIES[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** PIT histogram before and after recalibration; a calibrated model sits on the 10% line. */
export function PitChart({ raw, calibrated }: { raw: number[]; calibrated: number[] }) {
  const data = raw.map((r, i) => ({ bin: `${i * 10}–${(i + 1) * 10}%`, raw: r, calibrated: calibrated[i] }));
  return (
    <div>
      <LegendRow
        items={[
          { color: "var(--series-2)", label: "Raw negative binomial", swatch: true },
          { color: "var(--series-1)", label: "Recalibrated", swatch: true },
          { color: "var(--text-secondary)", label: "Perfect (10%)", dashed: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -12 }} barGap={2}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="bin" {...axis} interval={1} />
          <YAxis {...axis} axisLine={false} domain={[0, "auto"]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} />
          <Tooltip
            cursor={{ fill: "var(--accent-wash)" }}
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <TipBox>
                  <div className="text-ink-3">PIT {payload[0].payload.bin}</div>
                  <div>Raw: <b>{pct(payload[0].payload.raw)}</b></div>
                  <div>Recalibrated: <b>{pct(payload[0].payload.calibrated)}</b></div>
                </TipBox>
              ) : null
            }
          />
          <ReferenceLine y={0.1} stroke="var(--text-secondary)" strokeDasharray="4 3" />
          <Bar dataKey="raw" fill="var(--series-2)" radius={[4, 4, 0, 0]} maxBarSize={14} isAnimationActive={false} />
          <Bar dataKey="calibrated" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={14} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Predicted P(over) vs how often the over actually hit. On the diagonal = honest probabilities. */
export function ReliabilityChart({ bins }: { bins: { bin: string; predicted: number; observed: number; n: number }[] }) {
  return (
    <div>
      <LegendRow
        items={[
          { color: "var(--series-1)", label: "Model", swatch: true },
          { color: "var(--text-secondary)", label: "Perfect calibration", dashed: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
          <CartesianGrid />
          <XAxis type="number" dataKey="predicted" domain={[0, 1]} {...axis} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} name="Predicted" />
          <YAxis type="number" dataKey="observed" domain={[0, 1]} {...axis} axisLine={false} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} width={44} name="Observed" />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-secondary)" strokeDasharray="4 3" />
          <Tooltip
            cursor={false}
            content={({ active, payload }) =>
              active && payload?.[0] ? (
                <TipBox>
                  <div>Predicted over: <b>{pct(payload[0].payload.predicted)}</b></div>
                  <div>Actual over: <b>{pct(payload[0].payload.observed)}</b></div>
                  <div className="text-ink-3">{payload[0].payload.n.toLocaleString()} games</div>
                </TipBox>
              ) : null
            }
          />
          <Scatter data={bins} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} line={{ stroke: "var(--series-1)", strokeWidth: 2 }} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function EquityChart({ curve, start }: { curve: { date: string; bankroll: number }[]; start: number }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={curve} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="date" {...axis} tickFormatter={(d: string) => d.slice(5)} minTickGap={36} />
        <YAxis {...axis} axisLine={false} width={56} tickFormatter={(v: number) => (v >= 10000 ? `${Math.round(v / 1000)}K` : v.toLocaleString())} />
        <ReferenceLine y={start} stroke="var(--axis)" />
        <Tooltip
          cursor={{ stroke: "var(--axis)" }}
          content={({ active, payload }) =>
            active && payload?.[0] ? (
              <TipBox>
                <div className="text-ink-3">{payload[0].payload.date}</div>
                <b>${Number(payload[0].payload.bankroll).toLocaleString()}</b>
              </TipBox>
            ) : null
          }
        />
        <Line dataKey="bankroll" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function HBars({ rows, format = (v: number) => pct(v) }: { rows: { label: string; value: number }[]; format?: (v: number) => string }) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((r) => (
        <div key={r.label} className="grid grid-cols-[minmax(0,9rem)_1fr_3.5rem] items-center gap-2 rounded-md px-1 py-0.5 hover:bg-surface-2" title={`${r.label}: ${format(r.value)}`}>
          <span className="truncate text-[13px] text-ink-2">{r.label}</span>
          <div className="h-3">
            <div className="h-full rounded-r" style={{ width: `${(r.value / max) * 100}%`, background: "var(--series-1)", minWidth: 2 }} />
          </div>
          <span className="tabular text-right text-[13px] text-ink">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Correlation matrix: diverging blue (positive) / red (negative) through a neutral gray at 0. */
export function Heatmap({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const color = (v: number) => {
    const pole = v >= 0 ? "var(--div-pos)" : "var(--div-neg)";
    return `color-mix(in oklab, ${pole} ${Math.round(Math.min(Math.abs(v), 1) * 100)}%, var(--div-mid))`;
  };
  return (
    <div className="overflow-x-auto">
      <table className="tabular border-separate text-[11px]" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} className="px-1 font-medium text-ink-3">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="pr-1 text-right font-medium text-ink-3">{labels[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  title={`${labels[i]} × ${labels[j]}: ${v.toFixed(2)}`}
                  className="h-8 w-10 rounded text-center"
                  style={{ background: color(v), color: Math.abs(v) > 0.45 ? "#ffffff" : "var(--text-secondary)" }}
                >
                  {i === j ? "" : v.toFixed(2).replace("0.", ".")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-ink-3">
        <span>−1</span>
        <span className="h-2 w-28 rounded" style={{ background: "linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))" }} />
        <span>+1</span>
      </div>
    </div>
  );
}
