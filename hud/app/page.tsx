"use client";

import { useEffect, useState } from "react";
import BrowserViewport from "@/components/BrowserViewport";
import FeatureBars from "@/components/FeatureBars";
import SteeringControls from "@/components/SteeringControls";
import InterventionTimeline from "@/components/InterventionTimeline";
import TrajectoryLog from "@/components/TrajectoryLog";
import { connectWS, type AgentEvent } from "@/lib/ws";

export default function Page() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [features, setFeatures] = useState<{ id: number; label: string; activation: number }[]>([]);
  const [screenshot, setScreenshot] = useState<string>("");
  const [trajectory, setTrajectory] = useState<AgentEvent[]>([]);
  const [interventions, setInterventions] = useState<AgentEvent[]>([]);

  useEffect(() => {
    const dispose = connectWS((ev) => {
      setEvents((prev) => [...prev.slice(-200), ev]);
      if (ev.type === "features_read") setFeatures(ev.features);
      if (ev.type === "env_updated") setScreenshot(ev.screenshot_path);
      if (ev.type === "action_chosen") setTrajectory((prev) => [...prev, ev]);
      if (ev.type === "steering_applied") setInterventions((prev) => [...prev, ev]);
    });
    return dispose;
  }, []);

  return (
    <main className="grid grid-cols-12 gap-2 h-screen p-2">
      <section className="col-span-7 row-span-2 bg-zinc-900 rounded p-2">
        <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Browser viewport</h2>
        <BrowserViewport screenshotPath={screenshot} />
      </section>
      <section className="col-span-5 bg-zinc-900 rounded p-2">
        <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Active features</h2>
        <FeatureBars features={features} onSuppress={(id) => console.log("suppress", id)} />
      </section>
      <section className="col-span-5 bg-zinc-900 rounded p-2">
        <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Steering controls</h2>
        <SteeringControls onApply={(edits) => console.log("apply", edits)} />
      </section>
      <section className="col-span-7 bg-zinc-900 rounded p-2">
        <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Trajectory</h2>
        <TrajectoryLog events={trajectory} />
      </section>
      <section className="col-span-5 bg-zinc-900 rounded p-2">
        <h2 className="text-sm font-mono uppercase mb-2 text-zinc-400">Intervention timeline</h2>
        <InterventionTimeline events={interventions} />
      </section>
    </main>
  );
}
