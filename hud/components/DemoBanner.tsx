"use client";

type Props = {
  taskId?: string;
  policy?: string;
  step?: number;
  totalSteps?: number;
};

const POLICY_STYLES: Record<string, { bg: string; label: string }> = {
  baseline:    { bg: "bg-zinc-700",    label: "BASELINE — no steering" },
  random:      { bg: "bg-orange-700",  label: "RANDOM — control" },
  "wrong-sign":{ bg: "bg-purple-700",  label: "WRONG-SIGN — ablation" },
  targeted:    { bg: "bg-emerald-700", label: "TARGETED — SAE-steered" },
  static:      { bg: "bg-blue-700",    label: "STATIC — fixed catalog" },
  dynamic:     { bg: "bg-cyan-700",    label: "DYNAMIC — rule-based" },
};

export default function DemoBanner({ taskId, policy, step, totalSteps }: Props) {
  if (!taskId && !policy) {
    return (
      <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-2 text-xs font-mono text-zinc-500">
        Inside the Agent — waiting for trajectory...
      </div>
    );
  }
  const ps = POLICY_STYLES[policy || ""] || { bg: "bg-zinc-700", label: (policy || "?").toUpperCase() };
  const progress = totalSteps && step !== undefined ? (step + 1) / totalSteps : 0;

  return (
    <div className="border-b border-zinc-800 bg-zinc-900">
      <div className="flex items-center justify-between px-4 py-2 text-xs font-mono">
        <div className="flex items-center gap-3">
          <span className={`${ps.bg} px-2 py-1 rounded text-white font-bold uppercase tracking-wider`}>
            {ps.label}
          </span>
          <span className="text-zinc-400">task=</span>
          <span className="text-zinc-200">{taskId}</span>
        </div>
        <div className="flex items-center gap-2 text-zinc-400">
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
