"use client";

import { useEffect, useMemo, useState } from "react";
import BrowserViewport from "@/components/BrowserViewport";
import FeatureBars from "@/components/FeatureBars";
import SteeringControls from "@/components/SteeringControls";
import InterventionTimeline from "@/components/InterventionTimeline";
import TrajectoryLog from "@/components/TrajectoryLog";
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
    <main className="h-screen flex flex-col">
      <DemoBanner
        taskId={taskId}
        policy={policy}
        step={step}
        totalSteps={totalSteps}
        positionMode={positionMode}
        seed={seed}
        steeringEndpoint={steeringEndpoint}
      />

      <div className="grid grid-cols-12 gap-2 flex-1 p-2 overflow-hidden auto-rows-min">
        {/* Row 1 left: viewport spans 2 rows */}
        <section className="col-span-7 row-span-2 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Browser viewport</h2>
          <BrowserViewport screenshotPath={screenshot} />
        </section>

        {/* Row 1 right: features */}
        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Active features</h2>
          <FeatureBars features={features} highlightedIds={highlightedIds} />
        </section>

        {/* Row 2 right: steering controls + queue (stacked) */}
        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden flex flex-col gap-3">
          <div>
            <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Steering controls</h2>
            <SteeringControls onApply={onSteeringQueued} />
          </div>
          <div className="border-t border-zinc-800 pt-2">
            <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">
              Command queue
              {pending.length > 0 && (
                <span className="ml-2 px-1.5 py-0.5 rounded bg-yellow-500/20 text-yellow-300 text-[10px]">
                  {pending.length}
                </span>
              )}
            </h2>
            <CommandQueue pending={pending} onCancel={cancelPending} />
          </div>
        </section>

        {/* Row 3: effect size + intervention timeline + trajectory + diff */}
        <section className="col-span-4 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">
            Effect size
            {currentEdits.length > 0 && (
              <span className="ml-2 text-[10px] text-zinc-500">
                step {step !== undefined ? step + 1 : "—"}
              </span>
            )}
          </h2>
          <EffectSizeStrip edits={currentEdits} />
        </section>

        <section className="col-span-3 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Intervention log</h2>
          <InterventionTimeline events={interventions} />
        </section>

        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">
            Before / after diff
            {baselineByStep.size > 0 && (
              <span className="ml-2 text-[10px] text-zinc-500">
                vs baseline ({baselineByStep.size} cached steps)
              </span>
            )}
          </h2>
          <BeforeAfterDiff
            baselineByStep={baselineByStepMemo}
            currentTrajectory={trajectory}
            currentStep={step}
            currentPolicy={policy}
          />
        </section>

        {/* Row 4: trajectory full-width */}
        <section className="col-span-12 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Trajectory</h2>
          <TrajectoryLog events={trajectory} />
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
