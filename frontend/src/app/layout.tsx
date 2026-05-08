import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Header } from "@/components/layout/header";
import { Sidebar } from "@/components/layout/sidebar";
import { getServerClient } from "@/lib/supabase/server";
import type { Site } from "@/lib/supabase/types";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "İşGözcü - Şantiye Güvenlik Zekası",
  description:
    "Yapay zekâ destekli inşaat şantiyesi iş güvenliği izleme ve raporlama sistemi.",
};

async function getDefaultSiteContext(): Promise<{
  name: string;
  location: string | null;
}> {
  try {
    const sb = getServerClient();
    const { data } = await sb
      .from("sites")
      .select("*")
      .order("created_at", { ascending: true })
      .limit(1);
    const rows = (data ?? []) as Site[];
    if (rows.length > 0) {
      return { name: rows[0].name, location: rows[0].location ?? null };
    }
  } catch {
    // Falls through to placeholder so the layout still renders if Supabase
    // is misconfigured.
  }
  return { name: "Demo Şantiye", location: "Ankara, Türkiye" };
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const site = await getDefaultSiteContext();
  return (
    <html
      lang="tr"
      className={`${inter.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-bg-page text-text-primary">
        <Header siteName={site.name} location={site.location} />
        <Sidebar />
        <main className="pt-14 pl-16 min-h-screen">{children}</main>
      </body>
    </html>
  );
}
