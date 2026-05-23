"use client";

/**
 * TrajectoryBrowser — sidebar/dropdown that lists all saved
 * data/trajectories/*.jsonl files. Click one to spawn a replay through
 * the same cockpit. Zero Modal cost, deterministic playback.
 *
 * v0.21 (P1 reviewer ask): make the HUD usable post-demo for browsing
 * any past run. Also: lets you set up a side-by-side demo flow without
 * needing terminals — pick "google_shopping baseline" then click replay,
 * then pick "google_shopping targeted" and click again.
 */

import { useEffect, useState } from "react";

type Trajectory = {
  path: string;
  name: string;
  task_id: string;
  policy: string;
  n_steps: number;
  mtime: number;
  size: number;
};

function listUrl(): string {
  const ws = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return ws.replace(/^ws/, "http").replace(/\/feed$/, "/trajectories");
}

function replayUrl(): string {
  const ws = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return ws.replace(/^ws/, "http").replace(/\/feed$/, "/replay");
}

function clearUrl(): string {
  const ws = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/feed";
  return ws.replace(/^ws/, "http").replace(/\/feed$/, "/clear");
}

const REAL_WEB_PREFIXES = [
  "google_", "ebay_", "aliexpress_", "walmart_", "target_", "bestbuy_",
];

export default function TrajectoryBrowser() {
  const [trajectories, setTrajectories] = useState<Trajectory[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stepDelay, setStepDelay] = useState(2.5);

  async function refresh() {
    try {
      const r = await fetch(listUrl());
      const data = await r.json();
      setTrajectories(data.trajectories || []);
    } catch (e) {
      console.warn("trajectories list failed:", e);
    }
  }

  useEffect(() => {
    if (open) refresh();
  }, [open]);

  async function replay(t: Trajectory) {
    setBusy(true);
    try {
      await fetch(clearUrl(), { method: "POST" });
      const isQualitative = REAL_WEB_PREFIXES.some((p) => t.task_id.startsWith(p));
      await fetch(replayUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trajectory_path: t.path,
          step_delay: stepDelay,
          qualitative: isQualitative,
        }),
      });
      setOpen(false);
    } catch (e) {
      console.warn("replay failed:", e);
    } finally {
      setTimeout(() => setBusy(false), 600);
    }
  }

  // Compact button when closed; drawer when open.
  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700"
        title="Browse saved trajectories and replay them through this HUD without Modal calls."
      >
        ⏵ replay saved
      </button>
    );
  }

  return (
    <div className="fixed top-12 left-3 right-3 z-50 max-h-[70vh] bg-zinc-950 border border-zinc-700 rounded-lg shadow-xl overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800 bg-zinc-900">
        <h2 className="text-xs font-mono uppercase tracking-wider text-zinc-300">
          Replay saved trajectory
        </h2>
        <div className="flex items-center gap-3 text-xs">
          <label className="font-mono text-zinc-400">
            speed:
            <select
              value={stepDelay}
              onChange={(e) => setStepDelay(Number(e.target.value))}
              className="ml-1 bg-zinc-800 text-zinc-200 border border-zinc-700 rounded px-1 py-0.5 text-xs"
            >
              <option value={0.6}>fast (0.6s)</option>
              <option value={1.5}>normal (1.5s)</option>
              <option value={2.5}>slow (2.5s)</option>
              <option value={4.0}>demo (4.0s)</option>
            </select>
          </label>
          <button
            onClick={refresh}
            className="text-zinc-500 hover:text-zinc-200 font-mono"
          >
            ↻ refresh
          </button>
          <button
            onClick={() => setOpen(false)}
            className="text-zinc-500 hover:text-zinc-200 font-mono"
          >
            close
          </button>
        </div>
      </div>
      <div className="overflow-y-auto flex-1">
        {trajectories.length === 0 ? (
          <div className="p-4 text-xs text-zinc-500 italic">
            No trajectories yet — run an agent first, then come back.
          </div>
        ) : (
          <table className="w-full text-xs font-mono">
            <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
              <tr className="text-zinc-500 uppercase tracking-wider">
                <th className="text-left px-3 py-1.5">task</th>
                <th className="text-left px-3 py-1.5">policy</th>
                <th className="text-right px-3 py-1.5">steps</th>
                <th className="text-left px-3 py-1.5">file</th>
                <th className="px-3 py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {trajectories.map((t) => {
                const isQual = REAL_WEB_PREFIXES.some((p) =>
                  t.task_id.startsWith(p)
                );
                const policyColor =
                  t.policy === "targeted"
                    ? "text-emerald-300"
                    : t.policy === "baseline"
                    ? "text-zinc-400"
                    : t.policy === "wrong-sign"
                    ? "text-purple-300"
                    : t.policy === "noise"
                    ? "text-sky-300"
                    : "text-zinc-300";
                return (
                  <tr
                    key={t.path}
                    className="border-b border-zinc-900 hover:bg-zinc-900/60"
                  >
                    <td className="px-3 py-1.5 text-zinc-200">
                      {t.task_id}
                      {isQual && (
                        <span className="ml-2 text-[9px] uppercase text-zinc-600">
                          (real-web)
                        </span>
                      )}
                    </td>
                    <td className={`px-3 py-1.5 ${policyColor}`}>{t.policy}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-zinc-400">
                      {t.n_steps}
                    </td>
                    <td className="px-3 py-1.5 text-zinc-600 text-[10px] truncate max-w-[20rem]">
                      {t.name}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      <button
                        disabled={busy}
                        onClick={() => replay(t)}
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                          busy
                            ? "bg-zinc-700 text-zinc-500 cursor-wait"
                            : "bg-emerald-700 hover:bg-emerald-600 text-white"
                        }`}
                      >
                        ▶ replay
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
      <div className="px-3 py-1.5 border-t border-zinc-800 text-[10px] text-zinc-500 font-mono">
        Replays publish to this same ws_server. Same HUD, same components, no Modal calls.
      </div>
    </div>
  );
}
