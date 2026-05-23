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

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none"
      style={{ animation: "verdictIn 0.6s ease-out" }}
    >
      <style jsx>{`
        @keyframes verdictIn {
          0%   { opacity: 0; transform: scale(0.6); }
          70%  { opacity: 1; transform: scale(1.05); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes verdictPulse {
          0%, 100% { box-shadow: 0 0 60px currentColor; }
          50%      { box-shadow: 0 0 120px currentColor; }
        }
      `}</style>
      <div
        className={`text-center px-12 py-10 rounded-xl border-4 backdrop-blur-md ${
          success
            ? "bg-emerald-900/50 border-emerald-400 text-emerald-300"
            : "bg-red-900/50 border-red-400 text-red-300"
        }`}
        style={{ animation: "verdictPulse 2s ease-in-out infinite" }}
      >
        <div className="text-8xl mb-3 font-bold">
          {success ? "✓" : "✗"}
        </div>
        <div className="text-4xl font-bold mb-2 tracking-wider">
          {success ? "SUCCESS" : "FAILURE"}
        </div>
        <div className="text-sm uppercase tracking-widest text-zinc-300">
          {taskId} · {policy} · {totalSteps} steps
        </div>
      </div>
    </div>
  );
}
