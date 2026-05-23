"use client";

import { useEffect, useState } from "react";
import BrowserViewport from "@/components/BrowserViewport";
import FeatureBars from "@/components/FeatureBars";
import SteeringControls from "@/components/SteeringControls";
import InterventionTimeline from "@/components/InterventionTimeline";
import TrajectoryLog from "@/components/TrajectoryLog";
import Verdict from "@/components/Verdict";
import SteeringFlash from "@/components/SteeringFlash";
import DemoBanner from "@/components/DemoBanner";
import { connectWS, type AgentEvent, type SteeringEdit } from "@/lib/ws";

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
          setPolicy(ev.policy);
          setTotalSteps(ev.total_steps);
          setStep(undefined);
          setVerdictVisible(false);
          setVerdictSuccess(null);
          setTrajectory([]);
          setInterventions([]);
          setHighlightedIds([]);
          break;
        case "step_started":
          setStep(ev.step);
          break;
        case "features_read":
          setFeatures(ev.features || []);
          break;
        case "steering_applied":
          if (ev.edits) {
            setFlashEdits(ev.edits);
            setHighlightedIds(ev.edits.map((e) => e.feature_id));
            setInterventions((prev) => [...prev, ev]);
            // Clear highlights after 2.5s
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

  return (
    <main className="h-screen flex flex-col">
      <DemoBanner taskId={taskId} policy={policy} step={step} totalSteps={totalSteps} />

      <div className="grid grid-cols-12 gap-2 flex-1 p-2 overflow-hidden">
        <section className="col-span-7 row-span-2 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Browser viewport</h2>
          <BrowserViewport screenshotPath={screenshot} />
        </section>

        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Active features</h2>
          <FeatureBars features={features} highlightedIds={highlightedIds} />
        </section>

        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Steering controls</h2>
          <SteeringControls onApply={() => {}} />
        </section>

        <section className="col-span-7 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Trajectory</h2>
          <TrajectoryLog events={trajectory} />
        </section>

        <section className="col-span-5 bg-zinc-900 rounded p-2 overflow-hidden">
          <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Intervention timeline</h2>
          <InterventionTimeline events={interventions} />
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
