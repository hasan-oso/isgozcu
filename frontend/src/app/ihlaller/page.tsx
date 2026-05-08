import { ViolationsFilterBar } from "@/components/violations/violations-filter-bar";
import { ViolationsTable } from "@/components/violations/violations-table";
import { getCameras, getEvents } from "@/lib/supabase/queries";

export const dynamic = "force-dynamic";

interface PageProps {
  searchParams: Promise<{
    from?: string;
    to?: string;
    type?: string;
    camera?: string;
    status?: string;
    page?: string;
  }>;
}

const PAGE_SIZE = 20;

function startOfDayIso(d: string | undefined): string | undefined {
  if (!d) return undefined;
  const dt = new Date(`${d}T00:00:00`);
  return Number.isNaN(dt.valueOf()) ? undefined : dt.toISOString();
}

function endOfDayIso(d: string | undefined): string | undefined {
  if (!d) return undefined;
  const dt = new Date(`${d}T23:59:59.999`);
  return Number.isNaN(dt.valueOf()) ? undefined : dt.toISOString();
}

export default async function ViolationsPage({ searchParams }: PageProps) {
  const sp = await searchParams;
  const page = Math.max(1, Number(sp.page ?? "1") || 1);
  const status =
    sp.status === "active" || sp.status === "resolved" ? sp.status : undefined;

  const [data, cameras] = await Promise.all([
    getEvents({
      from: startOfDayIso(sp.from),
      to: endOfDayIso(sp.to),
      type: sp.type || undefined,
      cameraId: sp.camera || undefined,
      status,
      limit: PAGE_SIZE,
      offset: (page - 1) * PAGE_SIZE,
    }),
    getCameras(),
  ]);

  return (
    <div className="px-6 py-5">
      <header className="mb-4 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold text-text-primary">İhlaller</h1>
        <span className="text-sm text-text-muted">
          {data.total} kayıt
        </span>
      </header>

      <ViolationsFilterBar cameras={cameras} />

      <div className="mt-4">
        <ViolationsTable
          rows={data.rows}
          total={data.total}
          page={page}
          pageSize={PAGE_SIZE}
        />
      </div>
    </div>
  );
}
