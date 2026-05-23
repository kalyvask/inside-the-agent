"use client";

/**
 * CommandQueue — pending HUD-issued steering commands awaiting the next agent
 * step. v0.7-D: visual receipt that what the user just clicked is queued and
 * will arrive at the next decision. Without this panel, clicking a preset
 * felt like nothing was happening until the agent's next features_read.
 *
 * The queue is owned by HUD client state, not the server. When the agent
 * drains pending commands and emits a `steering_applied` event with
 * source="hud", the parent page clears the matching entries.
 */

import type { PendingCommand } from "@/lib/ws";

type Props = {
  pending: PendingCommand[];
  onCancel?: (idx: number) => void;
};

export default function CommandQueue({ pending, onCancel }: Props) {
  if (!pending?.length) {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        Queue empty — click a preset in Steering controls to queue an edit.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {pending.map((c, i) => {
        const isAmplify = c.delta > 0;
        const ageMs = Date.now() - c.queued_at;
        return (
          <div
            key={`${c.feature_id}-${c.queued_at}`}
            className="flex items-center justify-between gap-2 text-xs px-2 py-1.5 bg-zinc-800 rounded border border-yellow-500/30"
          >
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="font-mono text-yellow-300 shrink-0">→</span>
              <span className="font-mono text-zinc-200 truncate">
                {c.label || `f${c.feature_id}`}
              </span>
              <span
                className={`tabular-nums font-mono shrink-0 ${
                  isAmplify ? "text-emerald-300" : "text-red-300"
                }`}
              >
                {isAmplify ? "+" : ""}
                {c.delta.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-[10px] text-zinc-500 tabular-nums">
                {Math.round(ageMs / 100) / 10}s
              </span>
              {onCancel && (
                <button
                  onClick={() => onCancel(i)}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 hover:bg-red-800 text-zinc-300 hover:text-white"
                  title="Cancel this queued edit"
                >
                  ×
                </button>
              )}
            </div>
          </div>
        );
      })}
      <div className="text-[10px] text-zinc-500 mt-1 font-mono">
        Will apply at the next agent step
      </div>
    </div>
  );
}
