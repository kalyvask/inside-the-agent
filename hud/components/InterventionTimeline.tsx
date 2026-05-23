"use client";

import { useEffect, useRef } from "react";
import type { AgentEvent } from "@/lib/ws";

const SOURCE_COLORS: Record<string, string> = {
  targeted: "bg-emerald-700 text-emerald-100",
  static:   "bg-blue-700 text-blue-100",
  dynamic:  "bg-cyan-700 text-cyan-100",
  random:   "bg-orange-700 text-orange-100",
  "wrong-sign": "bg-purple-700 text-purple-100",
  hud:      "bg-yellow-600 text-yellow-50",
};

function fmtTime(ts?: number): string {
  if (!ts) return "--:--:--";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false });
}

export default function InterventionTimeline({ events }: { events: AgentEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="text-zinc-500 text-sm space-y-2">
        <div>No interventions yet</div>
        <div className="text-[10px] text-zinc-600">
          Audit log: every steering edit gets logged with timestamp, source, feature ID,
          and delta. HUD-initiated commands appear in yellow.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 overflow-hidden">
      <div className="text-[10px] text-zinc-500 flex justify-between">
        <span>{events.length} edit{events.length !== 1 ? "s" : ""} applied</span>
        <span>auto-scroll</span>
      </div>
      <ol className="space-y-1 text-xs font-mono overflow-y-auto max-h-[calc(100vh-260px)] pr-1">
        {events.slice(-50).map((e, i) => {
          const edits = (e.edits || []) as any[];
          if (!edits.length) return null;
          return (
            <li key={i} className="border-l-2 border-zinc-700 pl-2 py-1">
              <div className="flex items-center justify-between text-[10px] text-zinc-500">
                <span>{fmtTime(e.timestamp)}</span>
                <span>step {e.step ?? "?"}</span>
              </div>
              <ul className="space-y-0.5 mt-0.5">
                {edits.map((edit, j) => {
                  const src = edit.source || "static";
                  const cls = SOURCE_COLORS[src] || "bg-zinc-700 text-zinc-200";
                  return (
                    <li key={j} className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider ${cls}`}>
                        {src}
                      </span>
                      <span className="text-zinc-300 truncate flex-1">
                        {edit.label || `f${edit.feature_id}`}
                      </span>
                      <span className={`tabular-nums ${edit.delta < 0 ? "text-red-300" : "text-emerald-300"}`}>
                        {edit.delta > 0 ? `+${edit.delta}` : edit.delta}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </li>
          );
        })}
        <div ref={bottomRef} />
      </ol>
    </div>
  );
}
