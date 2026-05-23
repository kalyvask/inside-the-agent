"""
Replay a recorded trajectory through the HUD WebSocket.

Reads a JSONL trajectory written by agent.trajectory.TrajectoryLogger and
publishes the same event sequence to ws_server, with controllable timing.

This is what powers the demo video: deterministic, no Modal calls, no
network surprises.

Usage:
    # Replay a baseline failure to a running HUD
    python -m verify.replay_trajectory \\
        data/trajectories/promo_held_001_seed_0_baseline.jsonl --hud

    # Side-by-side recordings — run each in its own terminal
    python -m verify.replay_trajectory data/.../baseline.jsonl --hud --slow
    # ... record screen, then ...
    python -m verify.replay_trajectory data/.../targeted.jsonl --hud --slow

    # Just print events (debug)
    python -m verify.replay_trajectory data/.../baseline.jsonl --no-publish
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False)


def _category_for(label: str) -> str:
    """Infer steering category from a feature label."""
    label = (label or "").lower()
    if any(k in label for k in ("promot", "impuls", "distract")):
        return "risk"
    if any(k in label for k in ("halluc", "confab", "uncertain")):
        return "epistemic"
    if any(k in label for k in ("goal", "task")):
        return "task"
    if any(k in label for k in ("plan", "deliber")):
        return "behavioral"
    return "other"


@contextmanager
def _maybe_start_ws_server(enable: bool):
    if not enable:
        yield None
        return
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    console.print("[cyan]Starting ws_server on http://localhost:8765 ...[/cyan]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.ws_server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.5)
    try:
        yield proc
    finally:
        console.print("[cyan]Stopping ws_server...[/cyan]")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _publish(event: dict, ws_url: str, publish: bool):
    if not publish:
        console.print(f"[dim]{event['type']}[/dim] " + json.dumps(event)[:100])
        return
    import requests
    try:
        requests.post(f"{ws_url}/publish", json=event, timeout=1.0)
    except Exception as e:
        console.print(f"[red]Publish failed: {e}[/red]")


@app.command()
def main(
    trajectory: str = typer.Argument(..., help="Path to a trajectory JSONL"),
    hud: bool = typer.Option(False, "--hud", help="Spawn ws_server first"),
    publish: bool = typer.Option(True, "--publish/--no-publish", help="Send to ws_server"),
    ws_url: str = typer.Option("http://localhost:8765", help="ws_server URL"),
    step_delay: float = typer.Option(1.5, help="Seconds between steps (1.5 reads well on video)"),
    slow: bool = typer.Option(False, "--slow", help="Slow down to 2.5s/step for emphasis"),
    fast: bool = typer.Option(False, "--fast", help="Speed up to 0.6s/step"),
    show_verdict: bool = typer.Option(True, help="Show success/failure card at end"),
):
    """Replay a saved trajectory event-by-event to the HUD."""
    path = Path(trajectory)
    if not path.exists():
        console.print(f"[red]Not found: {path}[/red]")
        raise typer.Exit(1)

    if slow:
        step_delay = 2.5
    if fast:
        step_delay = 0.6

    steps = []
    with path.open() as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    if not steps:
        console.print("[red]Empty trajectory[/red]")
        raise typer.Exit(1)

    run_id = steps[0].get("run_id", path.stem)
    task_id = steps[0].get("task_id", "?")
    policy = steps[0].get("policy", "?")
    console.print(
        f"[bold]Replaying[/bold] {len(steps)} steps from [cyan]{path.name}[/cyan]\n"
        f"  task={task_id} policy={policy} delay={step_delay}s\n"
    )

    final_success = bool(steps[-1].get("result", {}).get("reward", 0) > 0)

    with _maybe_start_ws_server(hud):
        if hud:
            console.print(
                "[green]Open HUD in your browser: "
                "cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev[/green]\n"
                "[dim]Sleeping 3s so you can switch windows...[/dim]"
            )
            time.sleep(3)

        # Publish demo banner first
        _publish(
            {
                "type": "demo_banner",
                "task_id": task_id,
                "policy": policy,
                "run_id": run_id,
                "total_steps": len(steps),
                "expected_success": final_success,
            },
            ws_url,
            publish,
        )

        for step in steps:
            step_idx = step["step"]
            console.print(f"[bold yellow]Step {step_idx}[/bold yellow]")

            _publish({"type": "step_started", "task_id": task_id, "step": step_idx, "run_id": run_id}, ws_url, publish)

            # Features — enrich with category if missing.
            features = []
            for f in step.get("features", [])[:15]:
                category = f.get("category") or _category_for(f.get("label", ""))
                features.append(
                    {
                        "id": f["id"],
                        "label": f.get("label") or f"feature {f['id']}",
                        "activation": f["activation"],
                        "category": category,
                        "confidence": f.get("confidence", 0.0),
                    }
                )
            _publish({"type": "features_read", "step": step_idx, "features": features}, ws_url, publish)
            time.sleep(step_delay * 0.4)

            # Steering edits — convert to dict format expected by HUD
            edits = []
            for e in step.get("steering", []):
                edits.append(
                    {
                        "feature_id": e["feature_id"],
                        "label": e.get("label", f"feature {e['feature_id']}"),
                        "delta": e["delta"],
                        "source": e.get("source", "replay"),
                        "category": _category_for(e.get("label", "")),
                    }
                )
            if edits:
                _publish({"type": "steering_applied", "step": step_idx, "edits": edits}, ws_url, publish)
                time.sleep(step_delay * 0.2)

            # Action
            action = step.get("model", {}).get("parsed_action", {})
            _publish({"type": "action_chosen", "step": step_idx, "action": action}, ws_url, publish)

            # Env update — convert local file path to ws_server's /screenshots URL
            screenshot = step.get("observation", {}).get("screenshot_path", "")
            if screenshot:
                # Local path: data/screenshots/<filename>.png → /screenshots/<filename>.png
                filename = Path(screenshot).name
                screenshot_url = f"{ws_url}/screenshots/{filename}"
            else:
                screenshot_url = ""
            _publish({"type": "env_updated", "step": step_idx, "screenshot_path": screenshot_url}, ws_url, publish)

            time.sleep(step_delay * 0.4)

        # Verdict
        if show_verdict:
            _publish(
                {
                    "type": "task_done",
                    "task_id": task_id,
                    "success": final_success,
                    "run_id": run_id,
                    "total_steps": len(steps),
                    "policy": policy,
                },
                ws_url,
                publish,
            )
            console.print(
                f"\n[bold green]✓ SUCCESS[/bold green]" if final_success
                else f"\n[bold red]✗ FAILURE[/bold red]"
            )

    console.print("[dim]Replay complete.[/dim]")


if __name__ == "__main__":
    app()
