// Server-side Supabase client for React Server Components.
// Public dashboard for now — uses the anon key, no cookie/session handling.
// When auth lands, swap for `@supabase/ssr` createServerClient with cookies.

import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";

export function getServerClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) {
    throw new Error(
      "Supabase env vars missing. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in frontend/.env.local."
    );
  }
  return createClient<Database>(url, key, {
    auth: { persistSession: false },
  });
}
