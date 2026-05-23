"use client";

/**
 * CurrentAction — single-line strip that ALWAYS shows what the agent
 * just did and whether the browser executed it.
 *
 * v0.16 (reviewer P0 fix): "The HUD shows browser state and intervention
 * state, but the viewer has to infer what the agent just did. Add a
 * prominent strip: Step N action: click 'Cable Length' / executed: yes/no.
 * For real-web demos, executed is crucial."
 *
 * Placement: directly under the BrowserViewport so the audience reads
 * (1) what the page looks like, (2) what the agent decided to do about
 * it, in vertical order without scanning.
 */

import type { AgentEvent } from "@/lib/ws";

type Props = {
  /** The last action_chosen event in the current trajectory. */
  lastAction?: AgentEvent;
  /** Most recent step number observed. */
  step?: number;
  /** Whether the env actually dispatched the action (from v0.8 executed flag). */
  executed?: boolean | null;
};

function formatAction(action: any): string {
  if (!action) return "—";
  const kind = action.action;
  if (kind === "type") {
    const text = action.text ? `"${String(action.text).slice(0, 40)}"` : "";
    return `type into ${action.target || "?"} ${text}`;
  }
  if (kind === "click") {
    return `click ${action.target || "?"}`;
  }
  if (kind === "navigate") {
    return `navigate to ${String(action.url || "").slice(0, 50)}`;
  }
  if (kind === "scroll") {
    return `scroll ${action.direction || "down"}`;
  }
  if (kind === "submit") {
    return `submit ${action.target || ""}`;
  }
  if (kind === "done") {
    return `done${action.reason ? ` (${action.reason.slice(0, 40)})` : ""}`;
  }
  if (kind === "invalid") {
    return `INVALID — model emitted unparseable JSON`;
  }
  return JSON.stringify(action).slice(0, 70);
}

export default function CurrentAction({ lastAction, step, executed }: Props) {
  if (!lastAction || !lastAction.action) {
    return (
      <div className="flex items-center gap-3 px-3 py-1.5 bg-zinc-800/60 rounded text-xs font-mono border border-zinc-800">
        <span className="text-zinc-500">waiting for first action…</span>
      </div>
    );
  }
  const action = lastAction.action;
  const stepNum = typeof step === "number" ? step : lastAction.step;
  const actionText = formatAction(action);

  // executed: explicit True (DOM dispatched), explicit False (Playwright miss),
  // or null/undefined (older trajectories before v0.8).
  let execBadge: { bg: string; text: string; symbol: string };
  if (executed === true) {
    execBadge = { bg: "bg-emerald-900/60 border-emerald-700", text: "text-emerald-200", symbol: "✓ executed on DOM" };
  } else if (executed === false) {
    execBadge = { bg: "bg-red-900/60 border-red-700", text: "text-red-200", symbol: "✗ selector miss" };
  } else {
    execBadge = { bg: "bg-zinc-800 border-zinc-700", text: "text-zinc-400", symbol: "exec=?" };
  }

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 bg-zinc-800/60 rounded text-xs font-mono border border-zinc-800">
      <span className="shrink-0 text-zinc-500 uppercase tracking-wider">step</span>
      <span className="shrink-0 tabular-nums text-zinc-100">{stepNum !== undefined ? stepNum : "?"}</span>
      <span className="shrink-0 text-zinc-600">→</span>
      <span className="flex-1 min-w-0 truncate text-zinc-100" title={actionText}>
        {actionText}
      </span>
      <span
        className={`shrink-0 px-2 py-0.5 rounded border text-[10px] ${execBadge.bg} ${execBadge.text}`}
        title={
          executed === true
            ? "Playwright successfully dispatched this action against a real DOM element."
            : executed === false
            ? "Action parsed as valid JSON but Playwright couldn't find a matching element on the page."
            : "executed flag unavailable (pre-v0.8 trajectory)."
        }
      >
        {execBadge.symbol}
      </span>
    </div>
  );
}
