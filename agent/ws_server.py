"""
WebSocket server: bridges the agent loop and the HUD.

The agent loop publishes step-level events (features read, action chosen,
steering applied, env updated, task done). The HUD subscribes and renders.

Run as a sidecar to the agent runner. Default port 8765.

Usage:
  # Terminal 1: start the server
  python -m agent.ws_server

  # Terminal 2: run the agent (publishes via http POST)
  python -m bench.runner --policy targeted ...

  # Terminal 3: open the HUD
  cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev

Endpoints:
  WS   /feed          subscribe to event stream
  POST /publish       agent posts a new event (body = JSON event)
  GET  /health        liveness check
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Set

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="inside-the-agent ws-server", version="0.1")

# Serve saved Playwright screenshots so the HUD can render them in its
# BROWSER VIEWPORT panel. Path is relative to where ws_server was started.
_SCREENSHOTS_DIR = Path("data/screenshots")
_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=str(_SCREENSHOTS_DIR)), name="screenshots")

# Allow the HUD to connect from any localhost port during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State:
    clients: Set[WebSocket] = set()
    history: list[dict] = []
    HISTORY_LIMIT = 500
    # v0.3: command channel. HUD POSTs commands here; agent polls and drains.
    pending_commands: list[dict] = []


# ---------------------------------------------------------------------------
# Event schema (matches what the HUD expects)
# ---------------------------------------------------------------------------


class AgentEvent(BaseModel):
    type: str  # step_started | features_read | action_chosen | steering_applied | env_updated | task_done
    task_id: str | None = None
    step: int | None = None
    features: list[dict] | None = None
    action: dict | None = None
    edits: list[dict] | None = None
    screenshot_path: str | None = None
    success: bool | None = None
    timestamp: float | None = None


class SteeringCommand(BaseModel):
    """v0.3 HUD-to-runner command. The agent picks these up between steps
    and merges them onto the policy's plan for the next decision."""
    feature_id: int
    delta: float
    label: str = ""
    source: str = "hud"
    one_shot: bool = True  # if true, applied only on the next step


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------


async def broadcast(event: dict):
    if not State.clients:
        return
    payload = json.dumps(event)
    # Concurrent send; drop closed connections silently.
    coros = []
    for ws in list(State.clients):
        coros.append(_safe_send(ws, payload))
    await asyncio.gather(*coros, return_exceptions=True)


async def _safe_send(ws: WebSocket, payload: str):
    try:
        await ws.send_text(payload)
    except Exception:
        State.clients.discard(ws)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "clients": len(State.clients),
        "events_buffered": len(State.history),
        "pending_commands": len(State.pending_commands),
    }


@app.post("/clear")
async def clear():
    """v0.10: wipe the replay buffer + pending command queue.

    Use case: previous demo's events are buffered (50-event replay on
    connect) and the HUD shows stale content from an older run when
    the user refreshes. After /clear, the next HUD connection sees a
    blank slate until the next agent run publishes a demo_banner."""
    cleared = {
        "events": len(State.history),
        "pending": len(State.pending_commands),
    }
    State.history.clear()
    State.pending_commands.clear()
    # Tell currently-connected clients to reset their state too.
    await broadcast({"type": "demo_banner", "task_id": None, "policy": None,
                     "total_steps": None, "_reset": True,
                     "timestamp": time.time()})
    return {"cleared": cleared}


class StartRunRequest(BaseModel):
    """v0.14: HUD-triggered agent run launch.

    Used by the HUD's "Start agent run" button so a user watching the
    cockpit can kick a live agent demo without a separate terminal —
    crucial for steering-controls being meaningful (you can only inject
    HUD edits into a live agent, not into a static screenshot)."""
    policy: str = "targeted"
    task: str = "shopgym/tasks/real_ebay.json"
    pause: float = 6.0
    position_mode: str = "all"
    limit: int = 1
    trials: int = 1
    output_suffix: str = "hud"


class ReplayRequest(BaseModel):
    """v0.21 HUD-triggered trajectory replay."""
    trajectory_path: str
    step_delay: float = 1.5
    qualitative: bool = False


@app.get("/trajectories")
async def list_trajectories():
    """v0.21: list all trajectory files under data/trajectories/.

    The HUD's trajectory-browser sidebar calls this on mount to populate
    a clickable list. Each entry includes the file path + parsed metadata
    (task_id, policy, n_steps, mtime) so the user can pick a recording
    to replay without leaving the cockpit.
    """
    from pathlib import Path
    import json as _json
    traj_dir = Path("data/trajectories")
    if not traj_dir.exists():
        return {"trajectories": []}
    entries = []
    for path in sorted(traj_dir.glob("*.jsonl"), key=lambda p: -p.stat().st_mtime):
        try:
            first_line = ""
            n = 0
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        first_line = line
                    n += 1
            if not first_line:
                continue
            d = _json.loads(first_line)
            entries.append({
                "path": str(path),
                "name": path.name,
                "task_id": d.get("task_id", "?"),
                "policy": d.get("policy", "?"),
                "n_steps": n,
                "mtime": path.stat().st_mtime,
                "size": path.stat().st_size,
            })
        except Exception:
            continue
    return {"trajectories": entries[:200]}  # cap to avoid huge payloads


@app.post("/replay")
async def replay_trajectory(req: ReplayRequest):
    """v0.21: spawn `verify.replay_trajectory` as a subprocess that
    publishes events back to this same ws_server. Returns immediately
    with the spawned PID; events stream via /publish.

    Use case: HUD's trajectory browser lets the user click a past
    recording → server fires the replayer → cockpit visualizes the
    saved run with no Modal cost."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if not Path(req.trajectory_path).exists():
        return {"ok": False, "error": f"trajectory file not found: {req.trajectory_path}"}

    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "HUD_PUBLISH": "1",
    }
    cmd = [
        sys.executable, "-u", "-m", "verify.replay_trajectory",
        req.trajectory_path,
        "--step-delay", str(req.step_delay),
    ]
    # Default: don't spawn ws_server (we're already inside one), DO publish.
    # Both are the typer defaults so no flags needed.
    if req.qualitative:
        cmd.append("--qualitative")
    log_dir = Path("data/hud_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"replay_{int(time.time())}.log"
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    return {"ok": True, "pid": proc.pid, "log_path": str(log_path)}


@app.post("/start_run")
async def start_run(req: StartRunRequest):
    """Spawn a bench.runner subprocess that publishes events back to this
    same ws_server. Returns immediately with the spawned PID; the run
    streams events naturally via the existing /publish path."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if not Path(req.task).exists():
        return {"ok": False, "error": f"task file not found: {req.task}"}

    # Spawn detached so the agent run survives this HTTP request.
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "HUD_PUBLISH": "1",
        "OUTPUT_SUFFIX": req.output_suffix,
    }
    cmd = [
        sys.executable, "-u", "-m", "bench.runner",
        "--policy", req.policy,
        "--tasks", req.task,
        "--trials", str(req.trials),
        "--limit", str(req.limit),
        "--pause", str(req.pause),
        "--position-mode", req.position_mode,
    ]
    log_dir = Path("data/hud_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hud_run_{int(time.time())}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return {
        "ok": True,
        "pid": proc.pid,
        "log_path": str(log_path),
        "cmd": cmd,
    }


@app.post("/control")
async def control_post(cmd: SteeringCommand):
    """HUD posts a steering command. Stored until the agent drains it."""
    State.pending_commands.append({**cmd.model_dump(), "ts": time.time()})
    # Mirror to the event stream so the HUD intervention timeline picks it up.
    await broadcast({
        "type": "steering_applied",
        "edits": [{
            "feature_id": cmd.feature_id,
            "label": cmd.label or f"feature {cmd.feature_id}",
            "delta": cmd.delta,
            "source": "hud",
        }],
        "timestamp": time.time(),
    })
    return {"queued": True, "pending": len(State.pending_commands)}


@app.get("/control/pending")
async def control_drain():
    """Agent calls this between steps. Returns + clears pending commands."""
    cmds = list(State.pending_commands)
    State.pending_commands.clear()
    return {"commands": cmds}


@app.post("/publish")
async def publish(event: AgentEvent):
    payload = event.model_dump()
    State.history.append(payload)
    if len(State.history) > State.HISTORY_LIMIT:
        State.history.pop(0)
    await broadcast(payload)
    return {"ok": True, "subscribers": len(State.clients)}


@app.websocket("/feed")
async def feed(ws: WebSocket):
    await ws.accept()
    State.clients.add(ws)
    try:
        # Replay buffered events so the HUD doesn't start blank.
        for ev in State.history[-50:]:
            await ws.send_text(json.dumps(ev))
        # Keep connection open until client disconnects.
        while True:
            # Receive (with timeout) so dead clients get culled.
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                # Send a ping and continue.
                try:
                    await ws.send_text(json.dumps({"type": "ping", "timestamp": None}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        State.clients.discard(ws)


# ---------------------------------------------------------------------------
# Convenience: a Python publisher the agent loop can import.
# ---------------------------------------------------------------------------


def make_publisher(ws_server_url: str = "http://localhost:8765"):
    """
    Returns a callable: publish(event_dict) -> None.

    The agent loop can call this synchronously after each step instead of
    speaking HTTP itself.
    """
    import requests

    def publish(event: dict):
        try:
            requests.post(
                f"{ws_server_url}/publish",
                json=event,
                timeout=0.5,
            )
        except Exception:
            # Never let a broken HUD kill the benchmark.
            pass

    return publish


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 8765):
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
