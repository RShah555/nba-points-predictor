import type { NextConfig } from "next";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  output: "standalone",
  // The browser talks to /api on the same origin; Next proxies it to FastAPI, so no CORS is needed.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  images: {
    remotePatterns: [{ protocol: "https", hostname: "cdn.nba.com" }],
  },
};

export default config;
