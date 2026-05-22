import type { AgentEvent } from "@/lib/ws";

export default function InterventionTimeline({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) {
    return <div className="text-zinc-500 text-sm">No interventions yet</div>;
  }
  return (
    <ol className="space-y-1 text-xs font-mono overflow-y-auto max-h-[calc(100vh-200px)]">
      {events.slice(-30).map((e, i) => (
        <li key={i} className="flex gap-2 items-start">
          <span className="text-zinc-500 w-12 text-right">t={i}</span>
          <span className="flex-1">
            {e.type === "steering_applied" && e.edits?.length > 0
              ? e.edits.map((edit: any) => `${edit.label} ${edit.delta > 0 ? "+" : ""}${edit.delta}`).join(", ")
              : "(no edits)"}
          </span>
        </li>
      ))}
    </ol>
  );
}
