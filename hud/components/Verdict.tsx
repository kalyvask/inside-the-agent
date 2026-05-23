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

  if (!show || success === null) return null;

  // v0.17b: compact bottom-right toast instead of a full-screen modal.
  // Reviewer asked the verdict to annotate the browser, not cover the
  // product grid that caused success/failure.
  return (
    <div
      className={`fixed bottom-3 right-3 z-40 px-3 py-2 rounded-md border text-xs font-mono backdrop-blur-sm pointer-events-none ${
        success
          ? "bg-emerald-950/85 border-emerald-700/60 text-emerald-200"
          : "bg-red-950/85 border-red-800/60 text-red-200"
      }`}
      style={{ animation: "verdictSlide 0.35s ease-out" }}
    >
      <style jsx>{`
        @keyframes verdictSlide {
          0%   { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}</style>
      <div className="flex items-center gap-2">
        <span className="text-lg leading-none">{success ? "✓" : "✗"}</span>
        <div className="leading-tight">
          <div className="font-bold uppercase tracking-wider">
            {success ? "success" : "failure"}
          </div>
          <div className="text-[10px] opacity-70">
            {policy} · {taskId} · {totalSteps ?? "?"} steps
          </div>
        </div>
      </div>
    </div>
  );
}
