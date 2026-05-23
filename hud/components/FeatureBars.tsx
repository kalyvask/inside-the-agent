"use client";

import { useEffect, useState } from "react";
import type { FeaturePoint } from "@/lib/ws";

type Category = "risk" | "behavioral" | "epistemic" | "task" | "other";

const CATEGORY_COLORS: Record<Category, string> = {
  risk:       "bg-red-500",
  behavioral: "bg-blue-500",
  epistemic:  "bg-amber-500",
  task:       "bg-emerald-500",
  other:      "bg-zinc-600",
};

const CATEGORY_LABEL_COLORS: Record<Category, string> = {
  risk:       "text-red-300",
  behavioral: "text-blue-300",
  epistemic:  "text-amber-300",
  task:       "text-emerald-300",
  other:      "text-zinc-400",
};

function inferCategory(label: string): Category {
  const l = (label || "").toLowerCase();
  if (/(promot|impuls|distract)/.test(l)) return "risk";
  if (/(halluc|confab|uncertain)/.test(l)) return "epistemic";
  if (/(goal|task)/.test(l)) return "task";
  if (/(plan|deliber)/.test(l)) return "behavioral";
  return "other";
}

export default function FeatureBars({
  features,
  highlightedIds = [],
  onSuppress,
}: {
  features: FeaturePoint[];
  highlightedIds?: number[];
  onSuppress?: (id: number) => void;
}) {
  // Smooth bar animation: keep previous activations and interpolate via CSS transition.
  const [animatedFeatures, setAnimatedFeatures] = useState<FeaturePoint[]>([]);

  useEffect(() => {
    if (features && features.length > 0) {
      setAnimatedFeatures(features);
    }
  }, [features]);

  if (!animatedFeatures?.length) {
    return <div className="text-zinc-500 text-sm">No features yet</div>;
  }

  return (
    <div className="flex flex-col gap-2 overflow-y-auto max-h-[calc(100vh-200px)]">
      {animatedFeatures.slice(0, 12).map((f) => {
        const cat = (f.category as Category) || inferCategory(f.label);
        const isHighlighted = highlightedIds.includes(f.id);
        return (
          <div
            key={f.id}
            className={`flex items-center gap-2 text-xs transition-all duration-200 ${
              isHighlighted ? "ring-2 ring-yellow-400 rounded px-1 py-0.5 bg-zinc-800/40" : ""
            }`}
          >
            <span
              className={`w-44 truncate font-mono ${CATEGORY_LABEL_COLORS[cat]}`}
              title={`${f.label || `feature ${f.id}`}  (${cat})`}
            >
              {f.label || `feature ${f.id}`}
            </span>
            <div className="flex-1 bg-zinc-800 rounded h-5 overflow-hidden">
              <div
                className={`h-full ${CATEGORY_COLORS[cat]} transition-all duration-700 ease-out`}
                style={{ width: `${Math.min(100, Math.max(0, f.activation * 50))}%` }}
              />
            </div>
            <span className="w-12 text-right tabular-nums text-zinc-400">{f.activation.toFixed(2)}</span>
            {onSuppress && (
              <button
                onClick={() => onSuppress(f.id)}
                className="text-xs px-2 py-1 bg-red-900 hover:bg-red-800 rounded text-white"
              >
                −
              </button>
            )}
          </div>
        );
      })}

      {/* Legend */}
      <div className="mt-3 pt-3 border-t border-zinc-800 flex gap-3 text-xs">
        {(["risk", "behavioral", "epistemic", "task"] as Category[]).map((c) => (
          <div key={c} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded ${CATEGORY_COLORS[c]}`} />
            <span className={CATEGORY_LABEL_COLORS[c]}>{c}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
