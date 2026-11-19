import type { NextConfig } from "next";

const dev = process.env.NODE_ENV === "development";
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

// Production builds are a static site (web/out) that the FastAPI app serves, so the whole
// product is one process at one URL. In development, /api is proxied to a separately run API.
const config: NextConfig = {
  output: dev ? undefined : "export",
  trailingSlash: true, // pages export as folder/index.html, which FastAPI's static files serve directly
  skipTrailingSlashRedirect: true, // but never add a slash to /api calls
  images: { unoptimized: true },
  ...(dev && {
    async rewrites() {
      return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
    },
  }),
};

export default config;
