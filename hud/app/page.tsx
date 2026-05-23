"use client";

import { useEffect, useMemo, useState } from "react";
import BrowserViewport from "@/components/BrowserViewport";
import FeatureBars from "@/components/FeatureBars";
import SteeringControls from "@/components/SteeringControls";
import InterventionTimeline from "@/components/InterventionTimeline";
import Verdict from "@/components/Verdict";
import SteeringFlash from "@/components/SteeringFlash";
import DemoBanner from "@/components/DemoBanner";
import EffectSizeStrip from "@/components/EffectSizeStrip";
import CommandQueue from "@/components/CommandQueue";
import BeforeAfterDiff from "@/components/BeforeAfterDiff";
import {
  connectWS,
  type AgentEvent,
  type SteeringEdit,
  type PendingCommand,
} from "@/lib/ws";

export default function Page() {
  const [features, setFeatures] = useState<any[]>([]);
  const [screenshot, setScreenshot] = useState<string>("");
  const [trajectory, setTrajectory] = useState<AgentEvent[]>([]);
  const [interventions, setInterventions] = useState<AgentEvent[]>([]);

  // Demo metadata
  const [taskId, setTaskId] = useState<string | undefined>();
  const [policy, setPolicy] = useState<string | undefined>();
  const [step, setStep] = useState<number | undefined>();
  const [totalSteps, setTotalSteps] = useState<number | undefined>();

  // v0.7-D cockpit metadata
  const [positionMode, setPositionMode] = useState<AgentEvent["position_mode"]>();
  const [seed, setSeed] = useState<number | undefined>();
  const [steeringEndpoint, setSteeringEndpoint] = useState<AgentEvent["steering_endpoint"]>();
  const [currentEdits, setCurrentEdits] = useState<SteeringEdit[]>([]);
  const [baselineByStep, setBaselineByStep] = useState<Map<number, any>>(
    () => new Map()
  );
  const [pending, setPending] = useState<PendingCommand[]>([]);

  // Verdict + flash
  const [verdictVisible, setVerdictVisible] = useState(false);
  const [verdictSuccess, setVerdictSuccess] = useState<boolean | null>(null);
  const [flashEdits, setFlashEdits] = useState<SteeringEdit[]>([]);
  const [highlightedIds, setHighlightedIds] = useState<number[]>([]);

  // v0.10: intervention pulse — when steering fires, the viewport gets a
  // visible ring (color-coded by source: emerald for targeted/static,
  // sky for noise, yellow for hud, purple for wrong-sign, blue for prompt-only,
  // amber for dynamic). Auto-clears after 3s so consecutive interventions are
  // visually distinct from each other rather than merging into one glow.
  const [interventionPulse, setInterventionPulse] = useState<{
    source: string;
    edits: SteeringEdit[];
  } | null>(null);

  useEffect(() => {
    const dispose = connectWS((ev) => {
      switch (ev.type) {
        case "demo_banner":
          setTaskId(ev.task_id);
          if (ev.policy) setPolicy(ev.policy);
          setTotalSteps(ev.total_steps);
          setStep(undefined);
          setVerdictVisible(false);
          setVerdictSuccess(null);
          setTrajectory([]);
          setInterventions([]);
          setHighlightedIds([]);
          setCurrentEdits([]);
          setInterventionPulse(null);
          // Reset the baseline cache only if the new run is for a different
          // task — otherwise we want to keep replaying baseline_action events
          // we already buffered.
          break;
        case "policy_meta":
          if (ev.policy) setPolicy(ev.policy);
          setPositionMode(ev.position_mode);
          setSeed(ev.seed);
          setSteeringEndpoint(ev.steering_endpoint);
          if (ev.max_steps !== undefined) setTotalSteps(ev.max_steps);
          break;
        case "baseline_action":
          if (typeof ev.step === "number" && ev.action) {
            // Functional Map update so React picks up the change.
            setBaselineByStep((prev) => {
              const next = new Map(prev);
              next.set(ev.step!, ev.action);
              return next;
            });
          }
          break;
        case "step_started":
          setStep(ev.step);
          // New step: any edits that didn't get cleared by a matching
          // steering_applied stay in pending — they'll apply this step.
          break;
        case "features_read":
          setFeatures(ev.features || []);
          break;
        case "steering_applied":
          if (ev.edits) {
            setFlashEdits(ev.edits);
            setHighlightedIds(ev.edits.map((e) => e.feature_id));
            setInterventions((prev) => [...prev, ev]);
            setCurrentEdits(ev.edits);
            // v0.10: fire the viewport pulse using the first edit's source
            // (dynamic / targeted / hud / etc.) so the audience sees the
            // intervention land visually, not just in the side panels.
            const primarySource = ev.edits[0]?.source || "targeted";
            setInterventionPulse({ source: primarySource, edits: ev.edits });
            setTimeout(() => setInterventionPulse(null), 3000);
            // Clear matching pending HUD commands — they've now landed.
            const hudIds = new Set(
              ev.edits
                .filter((e) => e.source === "hud")
                .map((e) => e.feature_id)
            );
            if (hudIds.size > 0) {
              setPending((prev) => prev.filter((c) => !hudIds.has(c.feature_id)));
            }
            setTimeout(() => setHighlightedIds([]), 2500);
          }
          break;
        case "action_chosen":
          setTrajectory((prev) => [...prev, ev]);
          break;
        case "env_updated":
          if (ev.screenshot_path) setScreenshot(ev.screenshot_path);
          break;
        case "task_done":
          setVerdictSuccess(ev.success ?? false);
          setVerdictVisible(true);
          break;
      }
    });
    return dispose;
  }, []);

  // When user clicks a steering preset, register it in the local queue. The
  // queue clears when a matching steering_applied event arrives.
  const onSteeringQueued = (edits: SteeringEdit[]) => {
    if (!edits?.length) {
      // Reset case — clear pending too.
      setPending([]);
      return;
    }
    const now = Date.now();
    setPending((prev) => [
      ...prev,
      ...edits.map((e) => ({
        feature_id: e.feature_id,
        delta: e.delta,
        label: e.label,
        queued_at: now,
      })),
    ]);
  };

  const cancelPending = (idx: number) => {
    setPending((prev) => prev.filter((_, i) => i !== idx));
  };

  const baselineByStepMemo = useMemo(() => baselineByStep, [baselineByStep]);

  return (
    <main className="h-screen flex flex-col overflow-hidden">
      <DemoBanner
        taskId={taskId}
        policy={policy}
        step={step}
        totalSteps={totalSteps}
        positionMode={positionMode}
        seed={seed}
        steeringEndpoint={steeringEndpoint}
      />

      {/* v0.10 demo-fit layout: viewport widened to col-span 8 and Row 1
          stretched to 1.35fr so the live page screenshot dominates the
          frame the audience actually looks at. The intervention pulse
          (ring + colored glow) makes it visually obvious WHEN a steering
          edit lands — previously you only saw it in the side panels. */}
      <div
        className="grid gap-2 flex-1 p-2 overflow-hidden min-h-0"
        style={{
          gridTemplateColumns: "repeat(12, minmax(0, 1fr))",
          gridTemplateRows: "minmax(0, 1.35fr) minmax(0, 0.65fr)",
        }}
      >
        {/* Row 1 left: viewport. Pulse ring fires for 3s on steering_applied.
            Color encodes the intervention source so dynamic-policy vs targeted
            vs hud-issued edits are visually distinguishable. */}
        <section
          className={
            "col-span-8 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col min-h-0 " +
            "transition-all duration-300 " +
            (interventionPulse
              ? {
                  targeted: "ring-4 ring-emerald-400 shadow-[0_0_45px_rgba(52,211,153,0.55)]",
                  static: "ring-4 ring-emerald-400 shadow-[0_0_45px_rgba(52,211,153,0.55)]",
                  dynamic: "ring-4 ring-amber-400 shadow-[0_0_45px_rgba(251,191,36,0.55)]",
                  hud: "ring-4 ring-yellow-300 shadow-[0_0_45px_rgba(253,224,71,0.55)]",
                  noise: "ring-4 ring-sky-400 shadow-[0_0_45px_rgba(56,189,248,0.55)]",
                  "wrong-sign": "ring-4 ring-purple-400 shadow-[0_0_45px_rgba(192,132,252,0.55)]",
                  random: "ring-4 ring-orange-400 shadow-[0_0_45px_rgba(251,146,60,0.55)]",
                  "prompt-only": "ring-4 ring-blue-400 shadow-[0_0_45px_rgba(96,165,250,0.55)]",
                  "failure_mining": "ring-4 ring-rose-400 shadow-[0_0_45px_rgba(251,113,133,0.55)]",
                }[interventionPulse.source] || "ring-4 ring-zinc-400"
              : "")
          }
        >
          <div className="flex items-center justify-between mb-2 shrink-0">
            <h2 className="text-sm font-mono uppercase text-zinc-400">Browser viewport</h2>
            {interventionPulse && (
              <span
                className={
                  "text-[10px] font-mono uppercase px-2 py-0.5 rounded animate-pulse " +
                  {
                    targeted: "bg-emerald-900 text-emerald-200",
                    static: "bg-emerald-900 text-emerald-200",
                    dynamic: "bg-amber-900 text-amber-200",
                    hud: "bg-yellow-700 text-yellow-100",
                    noise: "bg-sky-900 text-sky-200",
                    "wrong-sign": "bg-purple-900 text-purple-200",
                    random: "bg-orange-900 text-orange-200",
                    "prompt-only": "bg-blue-900 text-blue-200",
                    failure_mining: "bg-rose-900 text-rose-200",
                  }[interventionPulse.source] || "bg-zinc-800 text-zinc-200"
                }
              >
                ⚡ INTERVENTION · {interventionPulse.source}
                {" · "}
                {interventionPulse.edits.length}{" "}
                {interventionPulse.edits.length === 1 ? "edit" : "edits"}
              </span>
            )}
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            <BrowserViewport screenshotPath={screenshot} />
          </div>
        </section>

        {/* Row 1 right: features (narrower) */}
        <section className="col-span-4 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col min-h-0">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400 shrink-0">Active features</h2>
          <div className="flex-1 min-h-0 overflow-y-auto">
            <FeatureBars features={features} highlightedIds={highlightedIds} />
          </div>
        </section>

        {/* Row 2 left: steering controls + queue stacked (widened 4 -> 5) */}
        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col gap-2 min-h-0">
          <div className="shrink-0">
            <h2 className="text-sm font-mono uppercase mb-1 text-zinc-400">Steering controls</h2>
            <SteeringControls onApply={onSteeringQueued} />
          </div>
          <div className="border-t border-zinc-800 pt-2 flex-1 min-h-0 overflow-hidden flex flex-col">
            <h2 className="text-sm font-mono uppercase mb-1 text-zinc-400 shrink-0 flex items-center gap-2">
              <span>Command queue</span>
              {pending.length > 0 && (
                <span className="px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-300 text-[10px]">
                  {pending.length}
                </span>
              )}
              {/* v0.12: agent-connection hint — if no step_started in the
                  last 30s and pending > 0, the queue is dormant. */}
              {pending.length > 0 && step === undefined && (
                <span className="text-[10px] text-amber-400 normal-case">
                  ⚠ no agent listening — start one via `python record_demo.py`
                </span>
              )}
            </h2>
            <div className="flex-1 min-h-0 overflow-y-auto">
              <CommandQueue pending={pending} onCancel={cancelPending} />
            </div>
          </div>
        </section>

        {/* Row 2 middle: effect size + intervention log stacked (widened 3 -> 4) */}
        <section className="col-span-4 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col gap-2 min-h-0">
          <div className="shrink-0">
            <h2 className="text-sm font-mono uppercase mb-1 text-zinc-400">
              Effect size
              {currentEdits.length > 0 && (
                <span className="ml-2 text-[10px] text-zinc-500">
                  step {step !== undefined ? step + 1 : "—"}
                </span>
              )}
            </h2>
            <EffectSizeStrip edits={currentEdits} />
          </div>
          <div className="border-t border-zinc-800 pt-2 flex-1 min-h-0 overflow-hidden flex flex-col">
            <h2 className="text-sm font-mono uppercase mb-1 text-zinc-400 shrink-0">Intervention log</h2>
            <div className="flex-1 min-h-0 overflow-y-auto">
              <InterventionTimeline events={interventions} />
            </div>
          </div>
        </section>

        {/* Row 2 right: before/after diff (narrowed 5 -> 3) */}
        <section className="col-span-3 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col min-h-0">
          <h2 className="text-sm font-mono uppercase mb-1 text-zinc-400 shrink-0">
            Before / after diff
            {baselineByStep.size > 0 && (
              <span className="ml-2 text-[10px] text-zinc-500">
                {baselineByStep.size} cached
              </span>
            )}
          </h2>
          <div className="flex-1 min-h-0 overflow-y-auto">
            <BeforeAfterDiff
              baselineByStep={baselineByStepMemo}
              currentTrajectory={trajectory}
              currentStep={step}
              currentPolicy={policy}
            />
          </div>
        </section>
      </div>

      <SteeringFlash edits={flashEdits} />
      <Verdict
        visible={verdictVisible}
        success={verdictSuccess}
        taskId={taskId}
        totalSteps={totalSteps}
        policy={policy}
      />
    </main>
  );
}
