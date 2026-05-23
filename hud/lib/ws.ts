// WebSocket client that subscribes to the browser-worker's event stream.
// Day 5 work: connect this to the actual agent runner. For now, it mocks events
// if NEXT_PUBLIC_WS_URL is unset (so the HUD renders something during dev).

export type FeaturePoint = {
  id: number;
  label: string;
  activation: number;
  category?: "risk" | "behavioral" | "epistemic" | "task" | "other";
  confidence?: number;
};

export type SteeringEdit = {
  feature_id: number;
  label: string;
  delta: number;
  source?: string;
  category?: "risk" | "behavioral" | "epistemic" | "task" | "other";
};

export type AgentEvent = {
  type:
    | "step_started"
    | "features_read"
    | "action_chosen"
    | "steering_applied"
    | "env_updated"
    | "task_done"
    | "demo_banner"
    | "ping";
  task_id?: string;
  step?: number;
  total_steps?: number;
  policy?: string;
  run_id?: string;
  features?: FeaturePoint[];
  action?: any;
  edits?: SteeringEdit[];
  screenshot_path?: string;
  success?: boolean;
  expected_success?: boolean;
  timestamp?: number;
};

export function connectWS(handler: (ev: AgentEvent) => void): () => void {
  const url = process.env.NEXT_PUBLIC_WS_URL;

  if (!url) {
    // Mock event stream for local dev without a backend.
    let mockStep = 0;
    const id = setInterval(() => {
      mockStep++;
      const mockFeatures: FeaturePoint[] = [
        { id: 9012, label: "promotional_bias", activation: 0.3 + Math.random() * 0.5, category: "risk" },
        { id: 1234, label: "planning", activation: 0.2 + Math.random() * 0.4, category: "behavioral" },
        { id: 5678, label: "goal_tracking", activation: 0.3 + Math.random() * 0.4, category: "task" },
        { id: 4444, label: "uncertainty", activation: Math.random() * 0.3, category: "epistemic" },
        { id: 7777, label: "impulsive_action", activation: Math.random() * 0.5, category: "risk" },
      ];
      handler({
        type: "features_read",
        step: mockStep,
        features: mockFeatures,
        timestamp: Date.now(),
      });
    }, 1200);
    return () => clearInterval(id);
  }

  const ws = new WebSocket(url);
  ws.onmessage = (msg) => {
    try {
      handler(JSON.parse(msg.data));
    } catch (e) {
      console.error("Bad WS message:", e);
    }
  };
  return () => ws.close();
}
