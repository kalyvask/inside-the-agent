"use client";

type Props = {
  taskId?: string;
  policy?: string;
  step?: number;
  totalSteps?: number;
  // v0.7-D cockpit metadata
  positionMode?: "all" | "all_prompt" | "last_prompt_only";
  seed?: number;
  steeringEndpoint?: "steer_act" | "steer_act_with_noise";
};

const POLICY_STYLES: Record<string, { bg: string; label: string }> = {
  baseline:     { bg: "bg-zinc-700",    label: "BASELINE — no steering" },
  random:       { bg: "bg-orange-700",  label: "RANDOM — control" },
  "wrong-sign": { bg: "bg-purple-700",  label: "WRONG-SIGN — ablation" },
  targeted:     { bg: "bg-emerald-700", label: "TARGETED — SAE-steered" },
  noise:        { bg: "bg-sky-700",     label: "NOISE — matched-norm" },
  "prompt-only":{ bg: "bg-blue-700",    label: "PROMPT-ONLY — control" },
  static:       { bg: "bg-blue-700",    label: "STATIC — fixed catalog" },
  dynamic:      { bg: "bg-cyan-700",    label: "DYNAMIC — rule-based" },
};

const POSITION_MODE_BG: Record<string, string> = {
  all:                "bg-amber-700",      // broad, less surgical
  all_prompt:         "bg-amber-600",
  last_prompt_only:   "bg-teal-700",       // surgical, defensible default
};

const POSITION_MODE_NOTE: Record<string, string> = {
  all:               "broad — every position",
  all_prompt:        "prompt-only",
  last_prompt_only:  "surgical — last prefill token",
};

export default function DemoBanner({
  taskId,
  policy,
  step,
  totalSteps,
  positionMode,
  seed,
  steeringEndpoint,
}: Props) {
  if (!taskId && !policy) {
    return (
      <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-2 text-xs font-mono text-zinc-500">
        Inside the Agent — waiting for trajectory...
      </div>
    );
  }
  const ps = POLICY_STYLES[policy || ""] || { bg: "bg-zinc-700", label: (policy || "?").toUpperCase() };
  const progress = totalSteps && step !== undefined ? (step + 1) / totalSteps : 0;
  const pmBg = POSITION_MODE_BG[positionMode || ""] || "bg-zinc-700";
  const pmNote = POSITION_MODE_NOTE[positionMode || ""];

  return (
    <div className="border-b border-zinc-800 bg-zinc-900">
      <div className="flex items-center justify-between px-4 py-2 text-xs font-mono">
        <div className="flex items-center gap-3 min-w-0">
          <span
            className={`${ps.bg} px-2 py-1 rounded text-white font-bold uppercase tracking-wider shrink-0`}
            title={`policy = ${policy}`}
          >
            {ps.label}
          </span>
          {positionMode && (
            <span
              className={`${pmBg} px-2 py-1 rounded text-white font-bold uppercase tracking-wider shrink-0`}
              title={`steering scope = ${positionMode}${pmNote ? `  (${pmNote})` : ""}`}
            >
              SCOPE: {positionMode.replace(/_/g, " ")}
            </span>
          )}
          {steeringEndpoint === "steer_act_with_noise" && (
            <span
              className="bg-sky-900 px-2 py-1 rounded text-sky-200 uppercase tracking-wider shrink-0"
              title="noise endpoint — matched-norm random residual perturbation, not feature edits"
            >
              endpoint: noise
            </span>
          )}
          <span className="text-zinc-500 shrink-0">·</span>
          <span className="text-zinc-400 shrink-0">task=</span>
          <span className="text-zinc-200 truncate">{taskId}</span>
          {typeof seed === "number" && (
            <>
              <span className="text-zinc-500 shrink-0">·</span>
              <span className="text-zinc-400 shrink-0">seed=</span>
              <span className="text-zinc-200 tabular-nums shrink-0">{seed}</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2 text-zinc-400 shrink-0">
          {step !== undefined && (
            <>
              <span>step</span>
              <span className="text-zinc-100 tabular-nums">{step + 1}</span>
              {totalSteps && <span>/ {totalSteps}</span>}
            </>
          )}
        </div>
      </div>
      {/* Progress bar */}
      <div className="h-0.5 bg-zinc-800">
        <div
          className={`h-full ${ps.bg} transition-all duration-500`}
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}
