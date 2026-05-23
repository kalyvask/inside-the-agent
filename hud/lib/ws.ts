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
    | "policy_meta"
    | "baseline_action"
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
  // v0.7-D: cockpit metadata
  position_mode?: "all" | "all_prompt" | "last_prompt_only";
  seed?: number;
  max_steps?: number;
  max_new_tokens?: number;
  temperature?: number;
  steering_endpoint?: "steer_act" | "steer_act_with_noise";
};

// v0.7-D / v0.16: HUD-issued steering command tracked through a
// 3-state lifecycle so the user can see exactly where each click landed.
// Reviewer P0 fix: the v0.7-D queue conflated "queued" (waiting for
// agent) with "applied" (already in effect) with "expired" (drained
// and gone). Now each entry carries its own state and lives long
// enough to be readable.
//
//   queued  : HUD POSTed to /control, agent has not pulled it yet.
//   applied : agent drained it AND emitted steering_applied with
//             source="hud". This entry lives 2 more seconds visually
//             before transitioning to expired.
//   expired : the one-shot has been consumed. Render in gray and
//             auto-remove after 5 more seconds.
export type CommandLifecycleState = "queued" | "applied" | "expired";

export type PendingCommand = {
  feature_id: number;
  delta: number;
  label: string;
  queued_at: number;  // epoch ms
  state: CommandLifecycleState;
  applied_at?: number;
  expired_at?: number;
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

  // v0.7-D: auto-reconnect. When the agent run finishes, the ws_server
  // subprocess gets torn down by the runner's context manager; the HUD's
  // socket closes. Without retry the HUD would go dark forever. With
  // retry, the next agent run respawns ws_server and the HUD picks up
  // the new connection within ~1.5s.
  let ws: WebSocket | null = null;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function open() {
    if (closed) return;
    ws = new WebSocket(url!);
    ws.onmessage = (msg) => {
      try {
        handler(JSON.parse(msg.data));
      } catch (e) {
        console.error("Bad WS message:", e);
      }
    };
    ws.onclose = () => {
      ws = null;
      if (!closed) {
        reconnectTimer = setTimeout(open, 1500);
      }
    };
    ws.onerror = () => {
      // Don't spam the console; onclose will fire and trigger the retry.
    };
  }

  open();
  return () => {
    closed = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    ws?.close();
  };
}
