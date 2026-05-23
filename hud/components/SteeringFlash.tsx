"use client";

import { useEffect, useState } from "react";
import type { SteeringEdit } from "@/lib/ws";

const CATEGORY_COLORS: Record<string, string> = {
  risk:       "from-red-700 to-red-900",
  behavioral: "from-blue-700 to-blue-900",
  epistemic:  "from-amber-700 to-amber-900",
  task:       "from-emerald-700 to-emerald-900",
};

export default function SteeringFlash({ edits }: { edits: SteeringEdit[] }) {
  const [visible, setVisible] = useState(false);
  const [currentEdits, setCurrentEdits] = useState<SteeringEdit[]>([]);

  useEffect(() => {
    if (edits && edits.length > 0) {
      setCurrentEdits(edits);
      setVisible(true);
      const t = setTimeout(() => setVisible(false), 2200);
      return () => clearTimeout(t);
    }
  }, [edits]);

  if (!visible || currentEdits.length === 0) return null;

  return (
    <div
      className="fixed top-14 left-1/2 z-40 -translate-x-1/2 pointer-events-none"
      style={{ animation: "flashIn 0.35s ease-out" }}
    >
      <style jsx>{`
        @keyframes flashIn {
          0%   { opacity: 0; transform: translate(-50%, -20px); }
          100% { opacity: 1; transform: translate(-50%, 0); }
        }
      `}</style>
      <div className="bg-gradient-to-r from-yellow-600 to-amber-600 rounded-lg px-6 py-3 shadow-2xl border border-yellow-300/50">
        <div className="text-xs uppercase tracking-widest text-yellow-100 font-bold">
          ⚡ STEERING APPLIED
        </div>
        <div className="flex gap-4 mt-1 text-sm font-mono">
          {currentEdits.map((e, i) => (
            <div key={i} className="flex items-center gap-1">
              <span className="text-white">{e.label}</span>
              <span className={`font-bold ${e.delta < 0 ? "text-red-200" : "text-emerald-200"}`}>
                {e.delta > 0 ? `+${e.delta}` : e.delta}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
