"use client";

/**
 * CommandQueue — HUD-issued steering edits, lifecycle-aware.
 *
 * v0.16 (reviewer P0 fix): previously this panel labeled everything
 * "queued for next step" even after the edit had landed, which made
 * the steering feel never-applied. Now each entry has an explicit
 * state — queued / applied / expired — with distinct color, badge,
 * and age-string.
 *
 *   queued   yellow  "← waiting for next step"
 *   applied  emerald "● landed at step N"
 *   expired  zinc    "✓ consumed"  (auto-removes 5s after expiry)
 */

import type { PendingCommand } from "@/lib/ws";

type Props = {
  pending: PendingCommand[];
  onCancel?: (idx: number) => void;
};

const STATE_STYLES = {
  queued: {
    container: "bg-zinc-800 border-yellow-500/40",
    badge: "bg-yellow-500/20 text-yellow-300",
    label: "QUEUED",
    hint: "waiting for next step",
  },
  applied: {
    container: "bg-emerald-900/30 border-emerald-500/60",
    badge: "bg-emerald-500/30 text-emerald-200",
    label: "APPLIED",
    hint: "landed this step",
  },
  expired: {
    container: "bg-zinc-900 border-zinc-700",
    badge: "bg-zinc-700 text-zinc-400",
    label: "CONSUMED",
    hint: "one-shot done",
  },
} as const;

function ageString(start: number): string {
  const age = (Date.now() - start) / 1000;
  if (age < 1) return "now";
  if (age < 60) return `${age.toFixed(0)}s ago`;
  return `${(age / 60).toFixed(0)}m ago`;
}

export default function CommandQueue({ pending, onCancel }: Props) {
  if (!pending?.length) {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        Click a preset in Steering controls to queue an edit. Each entry
        will progress: <span className="text-yellow-300">QUEUED</span> →{" "}
        <span className="text-emerald-300">APPLIED</span> →{" "}
        <span className="text-zinc-400">CONSUMED</span>.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {pending.map((c, i) => {
        const styles = STATE_STYLES[c.state] || STATE_STYLES.queued;
        const isAmplify = c.delta > 0;
        // Pick the most-relevant timestamp for the age string.
        const relevantTs =
          c.state === "expired" && c.expired_at ? c.expired_at :
          c.state === "applied" && c.applied_at ? c.applied_at :
          c.queued_at;
        return (
          <div
            key={`${c.feature_id}-${c.queued_at}`}
            className={`flex items-center justify-between gap-2 text-xs px-2 py-1.5 rounded border ${styles.container}`}
          >
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span
                className={`shrink-0 text-[9px] px-1.5 py-0.5 rounded font-mono uppercase ${styles.badge}`}
              >
                {styles.label}
              </span>
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
              <span className="text-[10px] text-zinc-500 tabular-nums" title={styles.hint}>
                {ageString(relevantTs)}
              </span>
              {onCancel && c.state === "queued" && (
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
    </div>
  );
}
