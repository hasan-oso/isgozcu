import type { NextConfig } from "next";
import path from "node:path";

// Read the Supabase URL hostname so next/image can serve snapshot thumbnails.
function supabaseHostname(): string | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  if (!url) return null;
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

const sbHost = supabaseHostname();

const nextConfig: NextConfig = {
  // Silences the "multiple lockfiles" warning by anchoring Turbopack to this app.
  turbopack: {
    root: path.resolve(__dirname),
  },
  images: {
    remotePatterns: sbHost
      ? [
          {
            protocol: "https",
            hostname: sbHost,
            pathname: "/storage/v1/object/public/**",
          },
        ]
      : [],
  },
};

export default nextConfig;
