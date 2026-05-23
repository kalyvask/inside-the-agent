"use client";

/**
 * EffectSizeStrip — visualizes the magnitude and direction of every steering
 * edit currently in effect for this step.
 *
 * Reviewer-requested HUD addition (v0.7-D). Live answer to "what's the
 * intervention doing right now?": each edit is rendered as a bar centered on
 * zero, growing left (suppress / negative delta) or right (amplify / positive
 * delta). Source-coded by color: targeted = emerald, random = orange,
 * wrong-sign = purple, noise = sky, hud (user-issued) = yellow.
 */

import type { SteeringEdit } from "@/lib/ws";

type Props = {
  edits: SteeringEdit[];
  // Max absolute delta to scale bars against (so bars stay readable across
  // policies). 8.0 covers our targeted ±6 with headroom for HUD overrides.
  maxAbs?: number;
};

const SOURCE_COLORS: Record<string, string> = {
  targeted:    "bg-emerald-500",
  random:      "bg-orange-500",
  "wrong-sign": "bg-purple-500",
  noise:       "bg-sky-500",
  hud:         "bg-yellow-400",
  "prompt-only": "bg-blue-500",
};

const SOURCE_LABEL_COLORS: Record<string, string> = {
  targeted:    "text-emerald-300",
  random:      "text-orange-300",
  "wrong-sign": "text-purple-300",
  noise:       "text-sky-300",
  hud:         "text-yellow-300",
  "prompt-only": "text-blue-300",
};

export default function EffectSizeStrip({ edits, maxAbs = 8.0 }: Props) {
  if (!edits?.length) {
    return (
      <div className="text-xs text-zinc-500 italic py-2">
        No active steering this step
      </div>
    );
  }

  // v0.16: reviewer flagged `f26737_ui_selection_voc...` truncation as
  // unreadable. Shorten label to the form `f<id> <key word>`.
  function shortLabel(label: string, fid: number): string {
    if (!label) return `f${fid}`;
    // strip "fN_" prefix, "_vocab" suffix, replace underscores with spaces,
    // then keep the first two semantic words for compactness.
    const stripped = label
      .replace(/^f\d+_/, "")
      .replace(/_vocab$/, "")
      .replace(/_/g, " ");
    const words = stripped.split(/\s+/).slice(0, 2).join(" ");
    return `f${fid} ${words}`;
  }

  return (
    <div className="flex flex-col gap-1.5">
      {edits.map((e, i) => {
        const src = e.source || "targeted";
        const color = SOURCE_COLORS[src] || "bg-zinc-500";
        const labelColor = SOURCE_LABEL_COLORS[src] || "text-zinc-300";
        const widthPct = Math.min(50, (Math.abs(e.delta) / maxAbs) * 50);
        const isNegative = e.delta < 0;
        return (
          <div
            key={`${e.feature_id}-${i}`}
            className="grid grid-cols-[10rem_1fr_3rem] items-center gap-2 text-xs"
            title={`${e.label || `feature ${e.feature_id}`}  source=${src}  Δ=${e.delta.toFixed(2)}`}
          >
            <span className={`truncate font-mono ${labelColor}`}>
              {shortLabel(e.label || "", e.feature_id)}
            </span>
            {/* Bipolar bar: centered on zero, grows left or right */}
            <div className="relative h-3 bg-zinc-800 rounded overflow-hidden">
              {/* Center line */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-zinc-600" />
              {/* Bar */}
              <div
                className={`absolute top-0 bottom-0 ${color} transition-all duration-500`}
                style={{
                  width: `${widthPct}%`,
                  ...(isNegative
                    ? { right: "50%" }
                    : { left: "50%" }),
                }}
              />
            </div>
            <span
              className={`text-right tabular-nums font-mono ${
                isNegative ? "text-red-300" : "text-emerald-300"
              }`}
            >
              {e.delta > 0 ? "+" : ""}
              {e.delta.toFixed(1)}
            </span>
          </div>
        );
      })}
      <div className="text-[10px] text-zinc-500 mt-1 font-mono">
        bar = |Δ| / {maxAbs.toFixed(1)} · left = suppress · right = amplify
      </div>
    </div>
  );
}
