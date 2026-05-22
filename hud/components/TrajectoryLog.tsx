import type { AgentEvent } from "@/lib/ws";

export default function TrajectoryLog({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) {
    return <div className="text-zinc-500 text-sm">Waiting for agent actions...</div>;
  }
  return (
    <ol className="space-y-1 text-xs font-mono overflow-y-auto max-h-[calc(100vh-200px)]">
      {events.slice(-30).map((e, i) => (
        <li key={i} className="flex gap-2 items-start">
          <span className="text-zinc-500 w-10 text-right">{i + 1}.</span>
          <span className="flex-1">
            {e.action?.action === "click" && `click → ${e.action.target}`}
            {e.action?.action === "type" && `type "${e.action.text}" in ${e.action.target}`}
            {e.action?.action === "done" && `done: ${e.action.reason || ""}`}
            {e.action?.action === "invalid" && `[invalid] ${e.action.raw?.slice(0, 60) || ""}`}
          </span>
          {e.success && <span className="text-green-400">✓</span>}
        </li>
      ))}
    </ol>
  );
}
