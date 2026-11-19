"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError } from "@/lib/api";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60_000,
            refetchOnWindowFocus: false,
            // 503 means the server is still starting up (a hosted app waking from sleep): keep waiting.
            retry: (count, error) => (error instanceof ApiError && error.status === 503 ? count < 40 : count < 1),
            retryDelay: (count, error) => (error instanceof ApiError && error.status === 503 ? 3000 : 1000 * 2 ** count),
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
