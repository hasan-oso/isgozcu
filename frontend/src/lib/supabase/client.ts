// Browser-side Supabase client.
// Uses the public anon key — bundled into the client JS by Next.js. Safe.
"use client";

import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

let _browser: ReturnType<typeof createClient<Database>> | null = null;

export function getBrowserClient() {
  if (_browser) return _browser;
  _browser = createClient<Database>(url, key, {
    auth: { persistSession: false },
  });
  return _browser;
}
