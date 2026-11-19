"use client";

import clsx from "clsx";
import { AlertTriangle } from "lucide-react";
import { useState } from "react";

export function Card({ title, subtitle, action, className, children }: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={clsx("card p-4 sm:p-5", className)}>
      {(title || action) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] text-ink-3">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({ label, value, sub, tone }: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "good" | "bad";
}) {
  return (
    <div className="rounded-xl bg-surface-2 px-3.5 py-3">
      <div className="text-[12px] text-ink-3">{label}</div>
      <div className="tabular mt-1 text-2xl font-semibold text-ink">{value}</div>
      {sub && (
        <div className={clsx("tabular mt-0.5 text-[12px]", tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-ink-2")}>
          {sub}
        </div>
      )}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "md" }: {
  value: T;
  options: readonly { key: T; name: string }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div role="tablist" className="inline-flex rounded-lg bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.key}
          role="tab"
          aria-selected={value === o.key}
          onClick={() => onChange(o.key)}
          className={clsx(
            "rounded-md font-medium transition-colors",
            size === "sm" ? "px-2.5 py-1 text-[12px]" : "px-3 py-1.5 text-[13px]",
            value === o.key ? "bg-surface-3 text-ink shadow-sm" : "text-ink-3 hover:text-ink-2",
          )}
        >
          {o.name}
        </button>
      ))}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-lg bg-surface-2", className)} />;
}

export function ErrorState({ error, hint }: { error: unknown; hint?: string }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="card flex items-start gap-3 p-4 text-[14px]">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" aria-hidden />
      <div>
        <div className="font-medium text-ink">Something went wrong</div>
        <div className="text-ink-2">{message}</div>
        {hint && <div className="mt-1 text-ink-3">{hint}</div>}
      </div>
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "accent" | "good" | "bad" }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        tone === "accent" && "bg-accent-wash text-accent",
        tone === "good" && "bg-surface-2 text-good",
        tone === "bad" && "bg-surface-2 text-bad",
        tone === "neutral" && "bg-surface-2 text-ink-2",
      )}
    >
      {children}
    </span>
  );
}

export function Avatar({ src, name, size = 44 }: { src: string; name: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  const initials = name.split(" ").map((p) => p[0]).slice(0, 2).join("");
  return (
    <div
      className="relative shrink-0 overflow-hidden rounded-full bg-surface-3"
      style={{ width: size, height: size }}
      aria-hidden
    >
      {failed ? (
        <div className="flex size-full items-center justify-center text-[12px] font-semibold text-ink-2">{initials}</div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt=""
          loading="lazy"
          onError={() => setFailed(true)}
          className="absolute left-1/2 top-[8%] h-[110%] w-auto max-w-none -translate-x-1/2 object-cover"
        />
      )}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-[12px] text-ink-3">
      {label}
      {children}
    </label>
  );
}

export const inputClass =
  "h-9 rounded-lg border border-line bg-surface-2 px-2.5 text-[14px] text-ink outline-none focus:border-accent tabular";
