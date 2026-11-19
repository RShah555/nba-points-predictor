import type { Metadata, Viewport } from "next";

import { Nav } from "@/components/nav";
import { Providers } from "@/components/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "NBA Predictor",
  description: "Probabilistic NBA player projections, fantasy points and prop pricing.",
};

export const viewport: Viewport = { themeColor: "#0d0d0d", width: "device-width", initialScale: 1 };

// Apply a saved theme before first paint so there is no flash.
const themeScript = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen">
        <Providers>
          <Nav />
          <main className="mx-auto max-w-7xl px-4 pb-16 pt-5">{children}</main>
          <footer className="mx-auto max-w-7xl px-4 pb-8 text-[12px] text-ink-3">
            Data from NBA.com via nba_api. Projections and prop prices are for information only.
          </footer>
        </Providers>
      </body>
    </html>
  );
}
