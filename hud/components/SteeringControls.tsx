"use client";

import { useState } from "react";

type Edit = { feature_id: number; delta: number; label: string };

// In dev, the WS URL is ws://localhost:8765/feed. We POST to /control on the
// same host:port (just swap the scheme).
function controlUrl(): string {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  // ws://host:port/feed -> http://host:port/control
  return wsUrl.replace(/^ws/, "http").replace(/\/feed$/, "/control");
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
}: {
  onApply?: (edits: Edit[]) => void;
}) {
  const [active, setActive] = useState<Edit[]>([]);
  const [busy, setBusy] = useState(false);

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
      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <button
            key={p.name}
            disabled={busy}
            onClick={() => applyPreset(p)}
            className={`px-3 py-1 rounded text-xs transition-colors ${
              busy
                ? "bg-zinc-700 text-zinc-500 cursor-wait"
                : "bg-zinc-800 hover:bg-zinc-700 text-zinc-100"
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
