import { WorkersGrid } from "@/components/workers/workers-grid";
import { getWorkersWithLatestEvent } from "@/lib/supabase/queries";

export const dynamic = "force-dynamic";

export default async function WorkersPage() {
  const workers = await getWorkersWithLatestEvent();
  return (
    <div className="px-6 py-5">
      <header className="mb-4 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold text-text-primary">İşçiler</h1>
        <span className="text-sm text-text-muted">
          {workers.length} kayıtlı işçi
        </span>
      </header>
      <WorkersGrid workers={workers} />
    </div>
  );
}
