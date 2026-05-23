"use client";

/**
 * BeforeAfterDiff — side-by-side comparison of the baseline action vs the
 * current run's action for the same step.
 *
 * Reviewer-requested HUD addition (v0.7-D). The money-shot answer to "what
 * did the intervention actually do to the agent's behavior?". Baseline is
 * loaded from data/baselines/<task_id>.jsonl (written by the runner at the
 * end of any baseline run) and emitted to the HUD as `baseline_action`
 * events at the start of each non-baseline run.
 *
 * Renders one row per step, with both actions side-by-side. The current
 * step's row is visually emphasized. If the actions differ on `action`
 * type or `target`, the row is flagged as "diverged" (yellow border).
 * If they match, "same" (zinc).
 */

import type { AgentEvent } from "@/lib/ws";

type Props = {
  baselineByStep: Map<number, any>;  // baseline action keyed by step
  currentTrajectory: AgentEvent[];    // current run's action_chosen events
  currentStep?: number;               // highlight this step
  currentPolicy?: string;             // skip diff if this IS the baseline
};

function summarizeAction(action: any): string {
  if (!action) return "—";
  const kind = action.action;
  if (kind === "type") {
    const t = action.text ? `"${String(action.text).slice(0, 30)}"` : "";
    return `type ${action.target || "?"} ${t}`;
  }
  if (kind === "click") {
    return `click ${action.target || "?"}`;
  }
  if (kind === "navigate") {
    return `navigate ${String(action.url || "").slice(0, 40)}`;
  }
  if (kind === "scroll") {
    return `scroll ${action.direction || "down"}`;
  }
  if (kind === "done") {
    return `done (${action.reason || ""})`;
  }
  if (kind === "submit") {
    return `submit ${action.target || ""}`;
  }
  if (kind === "invalid") {
    return `INVALID`;
  }
  return JSON.stringify(action).slice(0, 50);
}

function actionsDiffer(a: any, b: any): boolean {
  if (!a || !b) return Boolean(a || b);
  if (a.action !== b.action) return true;
  if ((a.target || "") !== (b.target || "")) return true;
  if ((a.text || "") !== (b.text || "")) return true;
  return false;
}

export default function BeforeAfterDiff({
  baselineByStep,
  currentTrajectory,
  currentStep,
  currentPolicy,
}: Props) {
  if (currentPolicy === "baseline") {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        Diff disabled: this run IS the baseline. Run a steered policy to compare.
      </div>
    );
  }
  if (baselineByStep.size === 0) {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        No baseline cache for this task. Run <span className="font-mono">--policy baseline</span> first.
      </div>
    );
  }

  // Build row list from union of step indices in either source, sorted ascending.
  const currentByStep = new Map<number, any>();
  currentTrajectory.forEach((ev) => {
    if (typeof ev.step === "number" && ev.action) {
      currentByStep.set(ev.step, ev.action);
    }
  });
  const allSteps = Array.from(
    new Set([...baselineByStep.keys(), ...currentByStep.keys()])
  ).sort((a, b) => a - b);

  if (!allSteps.length) {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        Waiting for current run's first action…
      </div>
    );
  }

  let divergedCount = 0;
  allSteps.forEach((s) => {
    if (actionsDiffer(baselineByStep.get(s), currentByStep.get(s))) divergedCount++;
  });

  return (
    <div className="flex flex-col gap-1 overflow-y-auto max-h-full">
      <div className="grid grid-cols-[2rem_1fr_1fr] gap-2 text-[10px] font-mono uppercase text-zinc-500 px-1">
        <span>step</span>
        <span>baseline</span>
        <span>{currentPolicy || "current"}</span>
      </div>
      {allSteps.map((s) => {
        const base = baselineByStep.get(s);
        const cur = currentByStep.get(s);
        const diverged = actionsDiffer(base, cur);
        const isCurrent = s === currentStep;
        return (
          <div
            key={s}
            className={`grid grid-cols-[2rem_1fr_1fr] gap-2 text-xs px-1 py-0.5 rounded ${
              isCurrent ? "bg-zinc-800 ring-1 ring-emerald-500/40" : ""
            } ${
              diverged ? "border-l-2 border-yellow-500" : "border-l-2 border-transparent"
            }`}
          >
            <span className="tabular-nums text-zinc-400 font-mono">{s}</span>
            <span className="font-mono text-zinc-300 truncate" title={summarizeAction(base)}>
              {summarizeAction(base)}
            </span>
            <span
              className={`font-mono truncate ${
                diverged ? "text-emerald-300" : "text-zinc-400"
              }`}
              title={summarizeAction(cur)}
            >
              {summarizeAction(cur)}
            </span>
          </div>
        );
      })}
      <div className="text-[10px] text-zinc-500 mt-2 font-mono">
        {divergedCount} / {allSteps.length} steps diverged from baseline
      </div>
    </div>
  );
}
