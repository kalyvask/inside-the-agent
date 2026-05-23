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
  /** v0.18: what the model would have done at this step on the IDENTICAL
   * prompt WITHOUT the steering edits. Emitted by the agent via a twin
   * brain call (edits={}) whenever the policy intervened. Renders below
   * the actual action as a counterfactual line so the audience sees the
   * causal answer to "is the steering doing anything?". Absent when no
   * intervention happened this step. */
  counterfactualAction?: any;
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

function actionsDiffer(a: any, b: any): boolean {
  if (!a || !b) return Boolean(a || b);
  if (a.action !== b.action) return true;
  if ((a.target || "") !== (b.target || "")) return true;
  if ((a.text || "") !== (b.text || "")) return true;
  return false;
}

export default function CurrentAction({
  lastAction,
  step,
  executed,
  counterfactualAction,
}: Props) {
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

  // v0.18: counterfactual side-by-side. If we have an un-steered
  // prediction for the SAME step, show it as a second row tagged
  // "WITHOUT EDIT" so the audience sees the causal contrast at the
  // exact decision moment.
  const showCounterfactual = !!counterfactualAction;
  const cfText = showCounterfactual ? formatAction(counterfactualAction) : "";
  const diverged = showCounterfactual && actionsDiffer(action, counterfactualAction);

  return (
    <div className="flex flex-col gap-1">
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

      {showCounterfactual && (
        <div
          className={`flex items-center gap-3 px-3 py-1 rounded text-xs font-mono border ${
            diverged
              ? "bg-amber-950/40 border-amber-700/40 text-amber-200/90"
              : "bg-zinc-900/40 border-zinc-800 text-zinc-500"
          }`}
          title={
            diverged
              ? "Twin brain call with edits={} produced a DIFFERENT action on the IDENTICAL prompt — the steering caused this divergence."
              : "Twin brain call with edits={} produced the SAME action — the steering didn't change behavior at this step."
          }
        >
          <span className="shrink-0 text-[9px] uppercase tracking-wider opacity-70">
            without edit
          </span>
          <span className="shrink-0 text-zinc-600">→</span>
          <span className="flex-1 min-w-0 truncate" title={cfText}>
            {cfText}
          </span>
          <span
            className={`shrink-0 px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wider ${
              diverged
                ? "bg-amber-900/50 text-amber-200"
                : "bg-zinc-800 text-zinc-500"
            }`}
          >
            {diverged ? "diverged" : "same"}
          </span>
        </div>
      )}
    </div>
  );
}
