type Feature = { id: number; label: string; activation: number };

const riskColor = (label: string) => {
  if (/promotional|impuls|halluc|distract/i.test(label)) return "bg-risk-high";
  if (/uncertain|confab/i.test(label)) return "bg-risk-med";
  if (/planning|goal/i.test(label)) return "bg-risk-low";
  return "bg-zinc-600";
};

export default function FeatureBars({
  features,
  onSuppress,
}: {
  features: Feature[];
  onSuppress: (id: number) => void;
}) {
  if (!features?.length) {
    return <div className="text-zinc-500 text-sm">No features yet</div>;
  }
  return (
    <div className="flex flex-col gap-2 overflow-y-auto max-h-[calc(100vh-200px)]">
      {features.slice(0, 12).map((f) => (
        <div key={f.id} className="flex items-center gap-2 text-xs">
          <span className="w-44 truncate font-mono text-zinc-300" title={f.label || `feature ${f.id}`}>
            {f.label || `feature ${f.id}`}
          </span>
          <div className="flex-1 bg-zinc-800 rounded h-5 overflow-hidden">
            <div
              className={`h-full ${riskColor(f.label)}`}
              style={{ width: `${Math.min(100, f.activation * 100)}%` }}
            />
          </div>
          <span className="w-10 text-right tabular-nums text-zinc-400">{f.activation.toFixed(2)}</span>
          <button
            onClick={() => onSuppress(f.id)}
            className="text-xs px-2 py-1 bg-red-900 hover:bg-red-800 rounded text-white"
          >
            −
          </button>
        </div>
      ))}
    </div>
  );
}
