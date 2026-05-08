import Link from "next/link";
import { Compass } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-6 py-10">
      <div className="flex max-w-md flex-col items-center gap-3 rounded-lg border border-[color:var(--border-subtle)] bg-bg-panel px-8 py-10 text-center">
        <Compass size={42} className="text-[color:var(--accent)]" />
        <h1 className="text-2xl font-semibold text-text-primary">
          Sayfa bulunamadı
        </h1>
        <p className="text-sm text-text-muted">
          Aradığınız sayfa mevcut değil ya da taşınmış olabilir.
        </p>
        <Link
          href="/"
          className="mt-2 inline-flex items-center gap-1 rounded bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
        >
          Ana sayfaya dön
        </Link>
      </div>
    </div>
  );
}
