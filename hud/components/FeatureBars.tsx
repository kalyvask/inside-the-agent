"use client";

import { useEffect, useState } from "react";
import type { FeaturePoint } from "@/lib/ws";

// v0.16 (reviewer P0 fix): only categorize features we have evidence
// for; everything else gets an explicit `[unlabelled]` tag rather than
// being silently lumped into "other" with a gray bar that looks
// indistinguishable from a tagged "other".

type Category = "risk" | "behavioral" | "epistemic" | "task" | "unlabelled";

const CATEGORY_COLORS: Record<Category, string> = {
  risk:       "bg-red-500",
  behavioral: "bg-blue-500",
  epistemic:  "bg-amber-500",
  task:       "bg-emerald-500",
  unlabelled: "bg-zinc-700",
};

const CATEGORY_LABEL_COLORS: Record<Category, string> = {
  risk:       "text-red-300",
  behavioral: "text-blue-300",
  epistemic:  "text-amber-300",
  task:       "text-emerald-300",
  unlabelled: "text-zinc-500",
};

// Display-name shortener: f26737_ui_selection_vocab -> "f26737 selection vocab"
// so labels fit without truncating. Same idea as effect-size labels.
function shortName(label: string, id: number): string {
  if (!label) return `feature ${id}`;
  // strip "fN_" prefix and any "_vocab" suffix the catalog uses
  const stripped = label
    .replace(/^f\d+_/, "")
    .replace(/_vocab$/, "")
    .replace(/_/g, " ");
  return `f${id} ${stripped}`;
}

function inferCategory(f: FeaturePoint): Category {
  if (f.category) return (f.category as Category);
  const l = (f.label || "").toLowerCase();
  // Only assign a category when there's clear textual evidence.
  if (!l) return "unlabelled";
  if (/(promot|impuls|distract|fail_mode)/.test(l)) return "risk";
  if (/(halluc|confab|uncertain|ui_selection|selection_vocab)/.test(l)) return "epistemic";
  if (/(goal|task|distraction_avoidance)/.test(l)) return "task";
  if (/(plan|deliber)/.test(l)) return "behavioral";
  return "unlabelled";
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

  // v0.17b: cap default rows to 8 so the right rail stops feeling like
  // an oversized box. Reviewer: "cap to 6-8 rows."
  //
  // v0.17c: ALSO use h-full + flex-col so the panel uses its full height —
  // features at the top, legend pinned to the bottom via mt-auto, the
  // empty space between them becomes deliberate breathing room instead
  // of a dark gap below the legend.
  const visibleFeatures = animatedFeatures.slice(0, 8);
  const hiddenCount = animatedFeatures.length - visibleFeatures.length;

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-col gap-2">
        {visibleFeatures.map((f) => {
        const cat = inferCategory(f);
        const isHighlighted = highlightedIds.includes(f.id);
        const isLabelled = cat !== "unlabelled";
        const display = isLabelled ? shortName(f.label, f.id) : `feature ${f.id}`;
        return (
          <div
            key={f.id}
            className={`flex items-center gap-2 text-xs transition-all duration-200 ${
              isHighlighted ? "ring-2 ring-yellow-400 rounded px-1 py-0.5 bg-zinc-800/40" : ""
            }`}
            title={
              `id=${f.id}\n` +
              `label=${f.label || "(none)"}\n` +
              `category=${cat}\n` +
              `activation=${f.activation.toFixed(3)}`
            }
          >
            <span className={`w-44 truncate font-mono ${CATEGORY_LABEL_COLORS[cat]}`}>
              {display}
              {!isLabelled && (
                <span className="ml-1 text-[9px] text-zinc-600 uppercase">[unknown]</span>
              )}
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
        {hiddenCount > 0 && (
          <div className="text-[10px] text-zinc-500 font-mono pl-1 pt-1">
            +{hiddenCount} more features below threshold
          </div>
        )}
      </div>

      {/* Legend — pinned to the bottom of the panel via mt-auto so the
          empty space between features and legend becomes deliberate
          breathing room rather than a dark gap below the legend. */}
      <div className="mt-auto pt-3 border-t border-zinc-800 flex flex-wrap gap-x-3 gap-y-1 text-xs">
        {(["risk", "behavioral", "epistemic", "task", "unlabelled"] as Category[]).map((c) => (
          <div key={c} className="flex items-center gap-1">
            <div className={`w-2.5 h-2.5 rounded-sm ${CATEGORY_COLORS[c]}`} />
            <span className={`text-[10px] ${CATEGORY_LABEL_COLORS[c]}`}>{c}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
