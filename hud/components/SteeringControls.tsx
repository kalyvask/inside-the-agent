"use client";

import { useState } from "react";

type Edit = { feature_id: number; delta: number; label: string };

export default function SteeringControls({
  onApply,
}: {
  onApply: (edits: Edit[]) => void;
}) {
  const [edits, setEdits] = useState<Edit[]>([]);

  const presets: { name: string; edits: Edit[] }[] = [
    { name: "Suppress promo bias", edits: [{ feature_id: 9012, delta: -3, label: "promotional_bias" }] },
    { name: "Amplify planning", edits: [{ feature_id: 1234, delta: +5, label: "planning" }] },
    { name: "Suppress hallucination", edits: [{ feature_id: 5678, delta: -3, label: "hallucination" }] },
    { name: "Reset", edits: [] },
  ];

  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <button
            key={p.name}
            onClick={() => {
              setEdits(p.edits);
              onApply(p.edits);
            }}
            className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 rounded text-xs"
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="border-t border-zinc-800 pt-2">
        <div className="text-xs text-zinc-400 mb-2">Active edits</div>
        {edits.length === 0 ? (
          <div className="text-zinc-600 text-xs">No edits applied</div>
        ) : (
          <ul className="space-y-1 text-xs font-mono">
            {edits.map((e, i) => (
              <li key={i} className="flex justify-between bg-zinc-800/50 px-2 py-1 rounded">
                <span>{e.label}</span>
                <span className={e.delta < 0 ? "text-red-400" : "text-green-400"}>
                  {e.delta > 0 ? `+${e.delta}` : e.delta}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
