"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Search } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";

import { api } from "@/lib/api";
import { num } from "@/lib/format";
import type { Player } from "@/lib/types";

import { Avatar } from "./ui";

function useDebounced<T>(value: T, ms = 150): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

/** Combobox player search. Press "/" or Ctrl/Cmd+K anywhere to focus it when `hotkey` is set. */
export function PlayerSearch({ onSelect, placeholder = "Search players", hotkey = false, autoFocus = false, className }: {
  onSelect: (p: Player) => void;
  placeholder?: string;
  hotkey?: boolean;
  autoFocus?: boolean;
  className?: string;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const listId = useId();
  const query = useDebounced(q.trim());
  const { data = [], isFetching } = useQuery({
    queryKey: ["search", query],
    queryFn: () => api.search(query),
    enabled: open,
    placeholderData: (prev) => prev,
  });

  useEffect(() => {
    if (!hotkey) return;
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest("input, textarea, select");
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hotkey]);

  useEffect(() => setActive(0), [query]);

  const choose = (p: Player) => {
    onSelect(p);
    setQ("");
    setOpen(false);
    input.current?.blur();
  };

  return (
    <div className={clsx("relative", className)}>
      <div className="flex h-9 items-center gap-2 rounded-lg border border-line bg-surface-2 px-2.5 focus-within:border-accent">
        <Search className="size-4 shrink-0 text-ink-3" aria-hidden />
        <input
          ref={input}
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          autoFocus={autoFocus}
          value={q}
          placeholder={placeholder}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, data.length - 1));
            else if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
            else if (e.key === "Enter" && data[active]) choose(data[active]);
            else if (e.key === "Escape") input.current?.blur();
            else return;
            e.preventDefault();
          }}
          className="min-w-0 flex-1 bg-transparent text-[14px] text-ink outline-none placeholder:text-ink-3"
        />
        {hotkey && (
          <kbd className="hidden rounded border border-line px-1.5 text-[11px] text-ink-3 sm:block">Ctrl K</kbd>
        )}
      </div>
      {open && (
        <ul
          id={listId}
          role="listbox"
          className="card absolute left-0 right-0 top-11 z-50 max-h-96 overflow-y-auto p-1 shadow-2xl"
        >
          {data.length === 0 && (
            <li className="px-3 py-2.5 text-[13px] text-ink-3">{isFetching ? "Searching…" : "No players found"}</li>
          )}
          {data.map((p, i) => (
            <li
              key={p.id}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => choose(p)}
              onMouseEnter={() => setActive(i)}
              className={clsx(
                "flex cursor-pointer items-center gap-3 rounded-lg px-2 py-1.5",
                i === active && "bg-surface-2",
              )}
            >
              <Avatar src={p.headshot} name={p.name} size={32} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[14px] text-ink">{p.name}</div>
                <div className="text-[12px] text-ink-3">{p.team}</div>
              </div>
              <div className="tabular text-right text-[12px] text-ink-2">
                {num(p.pts)} <span className="text-ink-3">pts</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
