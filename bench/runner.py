"""
Benchmark runner. Runs N trials of a policy on a task suite.

Day 5/6 work: this is the entrypoint for the held-out benchmark.

Usage:
  python -m bench.runner --policy baseline --tasks bench/tasks/held_out.json --trials 3
  python -m bench.runner --policy targeted --tasks bench/tasks/held_out.json --trials 3
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
import yaml
from rich.console import Console
from rich.progress import Progress

from agent.hud_publisher import HudPublisher
from agent.llm_agent import AgentConfig, SAEAgent
from policies import POLICY_REGISTRY

console = Console()
app = typer.Typer(add_completion=False)


def _make_brain_call():
    """Returns a callable wrapping the Modal brain-server's steer_act endpoint."""
    import modal

    BrainServer = modal.Cls.from_name("inside-the-agent", "BrainServer")
    server = BrainServer()

    def call(prompt: str, edits: dict | None = None, mode: str = "act", **kwargs):
        if mode == "read":
            return server.read_features.remote(prompt, top_k=kwargs.get("top_k", 20))
        return server.steer_act.remote(
            prompt=prompt,
            edits=edits or {},
            max_new_tokens=kwargs.get("max_new_tokens", 96),
            temperature=kwargs.get("temperature", 0.2),
            top_k=kwargs.get("top_k", 20),
        )

    return call


def _load_catalog(path: str = "sae/features.yaml") -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    catalog = {}
    for cat, entries in (raw or {}).items():
        if not entries:
            continue
        for entry in entries:
            catalog[entry["id"]] = {**entry, "category": cat}
    return catalog


def _load_tasks(path: str, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        tasks = []
        for f in sorted(p.glob("*.json")):
            data = json.loads(f.read_text())
            if isinstance(data, list):
                tasks.extend(data)
            else:
                tasks.append(data)
    else:
        data = json.loads(p.read_text())
        tasks = data if isinstance(data, list) else [data]
    if limit:
        tasks = tasks[:limit]
    return tasks


def _make_env(headless: bool = True):
    """Construct the ShopGym browser environment."""
    from shopgym import ShopGymEnv
    return ShopGymEnv(headless=headless)


@contextmanager
def _maybe_start_ws_server(enable: bool):
    """If enable, start ws_server as a subprocess and tear it down on exit."""
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
    # Give uvicorn a moment to bind.
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


@app.command()
def main(
    policy: str = typer.Option("baseline", help="Policy name from POLICY_REGISTRY"),
    tasks: str = typer.Option("shopgym/tasks", help="Path to task JSON or directory"),
    trials: int = typer.Option(3, help="Trials per task"),
    limit: int = typer.Option(None, help="Run at most N tasks (for smoke tests)"),
    output: str = typer.Option("data/results", help="Output directory"),
    catalog_path: str = typer.Option("sae/features.yaml", help="Feature catalog YAML"),
    hud: bool = typer.Option(False, help="Spin up ws_server + publish events to HUD"),
    headed: bool = typer.Option(False, "--headed", help="Show the Playwright browser window (default headless)"),
    pause: float = typer.Option(0.0, help="Seconds to pause between agent steps (visible run = 1.5)"),
):
    if policy not in POLICY_REGISTRY:
        console.print(f"[red]Unknown policy: {policy}. Available: {list(POLICY_REGISTRY)}[/red]")
        raise typer.Exit(1)

    catalog = _load_catalog(catalog_path)
    if not catalog:
        console.print("[yellow]features.yaml is empty — running with baseline-equivalent behavior[/yellow]")

    task_list = _load_tasks(tasks, limit=limit)
    console.print(f"Loaded {len(task_list)} tasks, policy={policy}, trials={trials}")

    brain_call = _make_brain_call()
    env = _make_env(headless=not headed)
    policy_fn = POLICY_REGISTRY[policy]
    publisher = HudPublisher(enabled=hud)

    agent = SAEAgent(
        brain_call=brain_call,
        env=env,
        policy=policy_fn,
        feature_catalog=catalog,
        config=AgentConfig(step_pause=pause),
        hud=publisher,
    )

    results = []
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with _maybe_start_ws_server(hud):
        if hud:
            console.print(
                "[green]HUD: open hud/ in a second terminal "
                "(NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run dev)[/green]"
            )
        with Progress(console=console) as progress:
            bar = progress.add_task(f"Running {policy}", total=len(task_list) * trials)
            for task in task_list:
                for seed in range(trials):
                    r = agent.run(task=task, seed=seed, policy_name=policy)
                    results.append(r)
                    progress.advance(bar)

    out_file = out_dir / f"{policy}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    console.print(f"Wrote {len(results)} results to {out_file}")

    success_rate = sum(r["success"] for r in results) / len(results)
    console.print(f"[bold]Success rate ({policy}):[/bold] {success_rate:.2%}")


if __name__ == "__main__":
    app()
