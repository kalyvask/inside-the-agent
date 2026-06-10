"use client";

import { useState } from "react";

type Edit = { feature_id: number; delta: number; label: string };

// In dev, the WS URL is ws://localhost:8765/feed. We POST to endpoints on the
// same host:port (just swap the scheme + the path).
function controlUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return wsUrl.replace(/^ws/, "http").replace(/\/feed$/, "/control");
}

function startRunUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return wsUrl.replace(/^ws/, "http").replace(/\/feed$/, "/start_run");
}

function clearUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return wsUrl.replace(/^ws/, "http").replace(/\/feed$/, "/clear");
}

async function postEdit(edit: Edit) {
  try {
    await fetch(controlUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feature_id: edit.feature_id,
        delta: edit.delta,
        label: edit.label,
        source: "hud",
        one_shot: true,
      }),
    });
  } catch (e) {
    console.warn("Failed to POST steering command:", e);
  }
}

async function postReset() {
  // A "reset" is a delta=0 marker which the agent's drain treats as a clear.
  // It also drains any pending queued commands by emptying them server-side.
  try {
    const drainUrl = controlUrl().replace(/\/control$/, "/control/pending");
    await fetch(drainUrl, { method: "GET" });
  } catch (e) {
    console.warn("Failed to drain pending commands:", e);
  }
}

export default function SteeringControls({
  onApply,
  agentLive,
}: {
  onApply?: (edits: Edit[]) => void;
  /** Set by the parent page when a recent step_started has been observed.
   * Used to gray out preset buttons + show the "Start run" cue when no
   * agent is listening, so the user doesn't click into the void. */
  agentLive?: boolean;
}) {
  const [active, setActive] = useState<Edit[]>([]);
  const [busy, setBusy] = useState(false);
  const [startBusy, setStartBusy] = useState(false);
  const [startedAt, setStartedAt] = useState<number | null>(null);

  // v0.19: both run buttons fire the SAME eBay task (real_ebay.json) —
  // user explicitly asked the demo to live on a real site. The only
  // difference is the policy: targeted (the headline, 2 SAE edits at
  // step 0) vs baseline (no steering, the comparison condition).
  async function startAgentRun(policyName: "targeted" | "baseline" = "targeted") {
    setStartBusy(true);
    try {
      // Clear stale buffer first so the new run's demo_banner is what the
      // HUD sees, not buffered events from a previous run.
      await fetch(clearUrl(), { method: "POST" });
      const r = await fetch(startRunUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          policy: policyName,
          task: "shopgym/tasks/real_ebay.json",
          pause: 6.0,           // slow enough for HUD clicks to land
          position_mode: "all",
          limit: 1,
          trials: 1,
          output_suffix: `hud_${policyName}`,
        }),
      });
      if (r.ok) {
        setStartedAt(Date.now());
        // Auto-clear the "just started" marker after 2 min — plenty for
        // any single demo run to complete.
        setTimeout(() => setStartedAt(null), 120000);
      } else {
        console.warn("start_run failed:", await r.text());
      }
    } catch (e) {
      console.warn("start_run threw:", e);
    } finally {
      setTimeout(() => setStartBusy(false), 800);
    }
  }

  const presets: { name: string; edits: Edit[]; isReset?: boolean }[] = [
    {
      name: "Suppress f26737 (UI-selection vocab)",
      edits: [{ feature_id: 26737, delta: -6, label: "f26737_ui_selection_vocab" }],
    },
    {
      name: "Amplify f23803 (distraction-avoidance)",
      edits: [{ feature_id: 23803, delta: +6, label: "f23803_distraction_avoidance_vocab" }],
    },
    {
      name: "Targeted combo (both)",
      edits: [
        { feature_id: 26737, delta: -6, label: "f26737_ui_selection_vocab" },
        { feature_id: 23803, delta: +6, label: "f23803_distraction_avoidance_vocab" },
      ],
    },
    {
      name: "Clear / Reset",
      edits: [],
      isReset: true,
    },
  ];

  async function applyPreset(p: { name: string; edits: Edit[]; isReset?: boolean }) {
    setBusy(true);
    if (p.isReset) {
      // Reset: clear local state AND drain any pending commands queued on server.
      setActive([]);
      await postReset();
    } else {
      setActive(p.edits);
      for (const edit of p.edits) {
        await postEdit(edit);
      }
    }
    onApply?.(p.edits);
    setTimeout(() => setBusy(false), 400);
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      {/* v0.14: kick a live agent run from inside the HUD so steering clicks
          actually drive behavior. Default config = targeted on real_ebay.json
          @ pause=6.0 (long enough for a human to click presets between
          steps). */}
      <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-zinc-800">
        {/* v0.26: both buttons are DISABLED while a run is live. Two concurrent
            runners write the same trajectory file and corrupt it; ws_server
            also rejects /start_run with 409 while one is alive (belt+braces). */}
        <button
          disabled={startBusy || agentLive}
          onClick={() => startAgentRun("targeted")}
          className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-colors ${
            startBusy
              ? "bg-zinc-700 text-zinc-500 cursor-wait"
              : agentLive
                ? "bg-zinc-700 text-zinc-500 cursor-not-allowed"
                : "bg-emerald-600 hover:bg-emerald-500 text-white animate-pulse"
          }`}
          title={
            agentLive
              ? "A run is in progress — wait for it to finish (concurrent runs corrupt the trajectory file)."
              : "Live targeted run on real eBay: f26737 -6 + f23803 +6 at step 0, then unsteered."
          }
        >
          {startBusy
            ? "starting…"
            : agentLive
              ? "● run in progress"
              : "▶ targeted (eBay)"}
        </button>
        <button
          disabled={startBusy || agentLive}
          onClick={() => startAgentRun("baseline")}
          className={`px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-colors ${
            startBusy || agentLive
              ? "bg-zinc-700 text-zinc-500 cursor-not-allowed"
              : "bg-zinc-700 hover:bg-zinc-600 text-zinc-100"
          }`}
          title={
            agentLive
              ? "A run is in progress — wait for it to finish (concurrent runs corrupt the trajectory file)."
              : "Live baseline run on real eBay — NO steering. Use this for side-by-side comparison against the targeted run."
          }
        >
          ▷ baseline (no steering)
        </button>
        {agentLive ? (
          <span className="text-[10px] text-emerald-400 font-mono">
            ● agent live — click presets to inject HUD edits
          </span>
        ) : (
          <span className="text-[10px] text-zinc-500 font-mono">
            both buttons run real eBay · pick one
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <button
            key={p.name}
            disabled={busy}
            onClick={() => applyPreset(p)}
            title={
              agentLive
                ? "Click — will apply at the next agent step"
                : "Will queue, but no agent is listening yet. Click START AGENT RUN above first."
            }
            className={`px-3 py-1 rounded text-xs transition-colors ${
              busy
                ? "bg-zinc-700 text-zinc-500 cursor-wait"
                : agentLive
                  ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-100"
                  : "bg-zinc-800/60 hover:bg-zinc-700 text-zinc-400 opacity-70"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="border-t border-zinc-800 pt-2">
        <div className="text-xs text-zinc-400 mb-2">
          Active edits (queued for next agent step)
        </div>
        {active.length === 0 ? (
          <div className="text-zinc-600 text-xs">No edits applied</div>
        ) : (
          <ul className="space-y-1 text-xs font-mono">
            {active.map((e, i) => (
              <li key={i} className="flex justify-between bg-zinc-800/50 px-2 py-1 rounded">
                <span>{e.label}</span>
                <span className={e.delta < 0 ? "text-red-400" : "text-emerald-400"}>
                  {e.delta > 0 ? `+${e.delta}` : e.delta}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="text-[10px] text-zinc-500 mt-1">
        POSTs to <code className="text-zinc-400">/control</code> on ws_server. Agent drains
        before its next decision.
      </div>
    </div>
  );
}
