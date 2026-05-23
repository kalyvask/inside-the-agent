"use client";

import { useEffect, useState } from "react";

type Props = {
  visible: boolean;
  success: boolean | null;
  taskId?: string;
  totalSteps?: number;
  policy?: string;
};

export default function Verdict({ visible, success, taskId, totalSteps, policy }: Props) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (visible) {
      setShow(true);
      // Auto-fade after 8s so a stale verdict doesn't block the next run.
      const t = setTimeout(() => setShow(false), 8000);
      return () => clearTimeout(t);
    } else {
      setShow(false);
    }
  }, [visible]);

  if (!show) return null;

  // v0.19: success === null means the task was qualitative (no verifier
  // evaluated). For eBay / AliExpress demos this is always the case —
  // reward is structurally 0 because there's nothing to score against.
  // Show a neutral "task ended" instead of a misleading FAILURE label.
  const isNeutral = success === null;
  const palette = isNeutral
    ? "bg-zinc-900/85 border-zinc-700/60 text-zinc-300"
    : success
    ? "bg-emerald-950/85 border-emerald-700/60 text-emerald-200"
    : "bg-red-950/85 border-red-800/60 text-red-200";
  const symbol = isNeutral ? "●" : success ? "✓" : "✗";
  const label = isNeutral ? "task ended" : success ? "success" : "failure";

  return (
    <div
      className={`fixed bottom-3 right-3 z-40 px-3 py-2 rounded-md border text-xs font-mono backdrop-blur-sm pointer-events-none ${palette}`}
      style={{ animation: "verdictSlide 0.35s ease-out" }}
    >
      <style jsx>{`
        @keyframes verdictSlide {
          0%   { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}</style>
      <div className="flex items-center gap-2">
        <span className="text-lg leading-none">{symbol}</span>
        <div className="leading-tight">
          <div className="font-bold uppercase tracking-wider">{label}</div>
          <div className="text-[10px] opacity-70">
            {policy} · {taskId} · {totalSteps ?? "?"} steps
            {isNeutral && " · qualitative (no verifier)"}
          </div>
        </div>
      </div>
    </div>
  );
}
