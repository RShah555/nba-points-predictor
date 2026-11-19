"use client";

import clsx from "clsx";
import { Activity, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { playerHref } from "@/lib/api";

import { PlayerSearch } from "./player-search";

const LINKS = [
  { href: "/", label: "Tonight" },
  { href: "/compare/", label: "Start/Sit" },
  { href: "/props/", label: "Props" },
  { href: "/record/", label: "Record" },
  { href: "/model/", label: "How it works" },
];

function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light" | null>(null);
  useEffect(() => {
    const saved = (() => {
      try {
        return localStorage.getItem("theme");
      } catch {
        return null;
      }
    })();
    const current = saved === "light" || saved === "dark" ? saved
      : window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    setTheme(current);
  }, []);
  const toggle = () => {
    const next = theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* private mode */
    }
    setTheme(next);
  };
  return (
    <button
      onClick={toggle}
      aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
      className="flex size-9 items-center justify-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink"
    >
      {theme === "light" ? <Moon className="size-4" /> : <Sun className="size-4" />}
    </button>
  );
}

export function Nav() {
  const path = usePathname();
  const router = useRouter();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-page/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <Link href="/" className="flex shrink-0 items-center gap-2 font-semibold text-ink">
          <span className="flex size-7 items-center justify-center rounded-lg bg-accent text-white">
            <Activity className="size-4" aria-hidden />
          </span>
          <span className="hidden sm:inline">Courtside</span>
        </Link>
        <nav className="flex min-w-0 items-center gap-0.5 overflow-x-auto">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href.replace(/\/$/, ""));
            return (
              <Link
                key={l.href}
                href={l.href}
                className={clsx(
                  "whitespace-nowrap rounded-lg px-2.5 py-1.5 text-[13px] font-medium",
                  active ? "bg-surface-2 text-ink" : "text-ink-3 hover:text-ink-2",
                )}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <PlayerSearch
          hotkey
          className="ml-auto hidden w-72 md:block"
          onSelect={(p) => router.push(playerHref(p.id))}
        />
        <ThemeToggle />
      </div>
    </header>
  );
}
