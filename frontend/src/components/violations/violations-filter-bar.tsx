"use client";

import { Filter, X } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useMemo, useState, useTransition } from "react";
import type { Camera } from "@/lib/supabase/types";
import { cn } from "@/lib/utils";

interface FiltersBarProps {
  cameras: Camera[];
}

const TYPE_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "Tüm türler" },
  { value: "no_hardhat", label: "Baret yok" },
  { value: "no_vest", label: "Yelek yok" },
  { value: "no_mask", label: "Maske yok" },
];

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "Tümü" },
  { value: "active", label: "Aktif" },
  { value: "resolved", label: "Çözüldü" },
];

export function ViolationsFilterBar({ cameras }: FiltersBarProps) {
  const router = useRouter();
  const params = useSearchParams();
  const [pending, startTransition] = useTransition();
  const [open, setOpen] = useState(true);

  const current = useMemo(
    () => ({
      from: params.get("from") ?? defaultFromIso(),
      to: params.get("to") ?? defaultToIso(),
      type: params.get("type") ?? "",
      camera: params.get("camera") ?? "",
      status: params.get("status") ?? "",
    }),
    [params]
  );

  const update = useCallback(
    (patch: Partial<typeof current>) => {
      const next = new URLSearchParams(params.toString());
      const merged = { ...current, ...patch };
      for (const [k, v] of Object.entries(merged)) {
        if (v) next.set(k, v);
        else next.delete(k);
      }
      next.delete("page");
      startTransition(() => {
        router.push(`/ihlaller?${next.toString()}`);
      });
    },
    [params, current, router]
  );

  const reset = () => {
    startTransition(() => {
      router.push("/ihlaller");
    });
  };

  const inputClass =
    "h-9 rounded border border-[color:var(--border-subtle)] bg-bg-panel-2 px-2 text-sm text-text-primary outline-none focus:border-[color:var(--accent)] focus:ring-1 focus:ring-[color:var(--accent)]";

  return (
    <section className="rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel">
      <header className="flex items-center justify-between border-b border-[color:var(--border-subtle)] px-4 py-2.5">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 text-sm text-text-primary"
          aria-expanded={open}
          aria-controls="filter-body"
        >
          <Filter size={14} />
          Filtreler
          {pending ? (
            <span className="text-[10px] uppercase tracking-wider text-text-muted">
              yükleniyor…
            </span>
          ) : null}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-1 text-xs text-text-muted hover:text-text-primary"
        >
          <X size={12} />
          Sıfırla
        </button>
      </header>
      {open ? (
        <div
          id="filter-body"
          className="grid grid-cols-1 gap-3 px-4 py-3 sm:grid-cols-2 lg:grid-cols-5"
        >
          <Field label="Başlangıç">
            <input
              type="date"
              className={inputClass}
              value={current.from}
              onChange={(e) => update({ from: e.target.value })}
            />
          </Field>
          <Field label="Bitiş">
            <input
              type="date"
              className={inputClass}
              value={current.to}
              onChange={(e) => update({ to: e.target.value })}
            />
          </Field>
          <Field label="Tür">
            <select
              className={inputClass}
              value={current.type}
              onChange={(e) => update({ type: e.target.value })}
            >
              {TYPE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Kamera">
            <select
              className={inputClass}
              value={current.camera}
              onChange={(e) => update({ camera: e.target.value })}
            >
              <option value="">Tüm kameralar</option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Durum">
            <select
              className={inputClass}
              value={current.status}
              onChange={(e) => update({ status: e.target.value })}
            >
              {STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </Field>
        </div>
      ) : null}
    </section>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className={cn("flex flex-col gap-1")}>
      <span className="text-[10px] uppercase tracking-wider text-text-muted">
        {label}
      </span>
      {children}
    </label>
  );
}

function defaultFromIso(): string {
  const d = new Date();
  d.setDate(d.getDate() - 7);
  return d.toISOString().slice(0, 10);
}

function defaultToIso(): string {
  // Default upper bound is the day AFTER today so today's events are included.
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}
