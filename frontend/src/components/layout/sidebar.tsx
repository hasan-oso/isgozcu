"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  AlertTriangle,
  BarChart3,
  Radio,
  Settings,
  Users,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
}

const NAV: readonly NavItem[] = [
  { href: "/", label: "Canlı", icon: Radio },
  { href: "/ihlaller", label: "İhlaller", icon: AlertTriangle },
  { href: "/raporlar", label: "Raporlar", icon: BarChart3 },
  { href: "/isciler", label: "İşçiler", icon: Users },
  { href: "/ayarlar", label: "Ayarlar", icon: Settings },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 bottom-0 z-30 w-16 group",
        "hover:w-60 transition-[width] duration-200 ease-out",
        "bg-bg-panel border-r border-[color:var(--border-subtle)]",
        "pt-14 flex flex-col"
      )}
    >
      <nav className="flex flex-col gap-1 px-2 py-3 flex-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors whitespace-nowrap overflow-hidden",
                active
                  ? "bg-accent text-white"
                  : "text-text-muted hover:bg-[color:var(--bg-panel-2)] hover:text-text-primary"
              )}
            >
              <Icon size={18} className="flex-shrink-0" />
              <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                {label}
              </span>
            </Link>
          );
        })}
      </nav>
      <div className="px-2 pb-3">
        <div className="px-3 py-2 text-[10px] uppercase tracking-wider text-text-muted opacity-0 group-hover:opacity-100 transition-opacity">
          v0.3.1
        </div>
      </div>
    </aside>
  );
}
