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
    """Returns a callable wrapping the Modal brain-server's steer_act endpoint.

    Respects BRAIN_APP_NAME env var so the same runner can target either the
    Llama (inside-the-agent) or the Gemma fallback (inside-the-agent-gemma).
    """
    import os
    import modal

    app_name = os.environ.get("BRAIN_APP_NAME", "inside-the-agent")
    BrainServer = modal.Cls.from_name(app_name, "BrainServer")
    server = BrainServer()

    def call(prompt: str, edits: dict | None = None, mode: str = "act", **kwargs):
        if mode == "read":
            return server.read_features.remote(prompt, top_k=kwargs.get("top_k", 20))
        if mode == "noise":
            # v0.7-B: route the noise_control policy through the matched-norm
            # noise endpoint. The targeted residual delta has |Δ| ≈ 6, so the
            # noise vector is drawn isotropically and rescaled to that L2 norm
            # at the same position the targeted policy modifies.
            return server.steer_act_with_noise.remote(
                prompt=prompt,
                noise_seed=int(kwargs.get("noise_seed", 0)),
                noise_norm=float(kwargs.get("noise_norm", 6.0)),
                max_new_tokens=kwargs.get("max_new_tokens", 96),
                temperature=kwargs.get("temperature", 0.2),
                top_k=kwargs.get("top_k", 20),
                position_mode=kwargs.get("position_mode", "all"),
            )
        return server.steer_act.remote(
            prompt=prompt,
            edits=edits or {},
            max_new_tokens=kwargs.get("max_new_tokens", 96),
            temperature=kwargs.get("temperature", 0.2),
            top_k=kwargs.get("top_k", 20),
            position_mode=kwargs.get("position_mode", "all"),
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


BASELINE_CACHE_DIR = Path("data/baselines")


def _load_baseline_cache(task_id: str) -> list[dict]:
    """v0.7-D: read cached baseline actions for a task, keyed by step.

    Cache format: one JSON line per step with {"step": int, "action": dict}.
    Used by the HUD's before/after-diff panel — when a non-baseline policy
    runs this task, the HUD compares the live action against the cached
    baseline action for the same step.

    Returns [] if the cache doesn't exist (e.g., baseline hasn't been run
    yet for this task)."""
    cache_path = BASELINE_CACHE_DIR / f"{task_id}.jsonl"
    if not cache_path.exists():
        return []
    out = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_baseline_cache(task_id: str, trajectory_path: Path) -> None:
    """v0.7-D: after a baseline run, extract per-step actions from the
    trajectory JSONL and write the cache. Idempotent — last baseline trial
    wins, which is fine for the diff panel."""
    if not trajectory_path.exists():
        return
    BASELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            step_log = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({
            "step": step_log.get("step"),
            "action": step_log.get("model", {}).get("parsed_action"),
        })
    cache_path = BASELINE_CACHE_DIR / f"{task_id}.jsonl"
    cache_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_env(headless: bool = True, env_type: str = "shopgym"):
    """Construct the browser environment.

    env_type:
      - "shopgym" (default) — deterministic templated storefront
      - "web" — real public website via shopgym.web_env.WebEnv
    """
    if env_type == "web":
        from shopgym import WebEnv
        return WebEnv(headless=headless)
    from shopgym import ShopGymEnv
    return ShopGymEnv(headless=headless)


def _infer_env_type(tasks: list[dict]) -> str:
    """Tasks may declare env_type='web'; pick first non-default."""
    for t in tasks:
        if t.get("env_type") == "web":
            return "web"
    return "shopgym"


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
    position_mode: str = typer.Option(
        "all",
        help="Where the steering hook applies the residual delta. "
             "'all' = every position (v0.1 default, what produces the 83% headline). "
             "'last_prompt_only' = only the last token of prefill (v0.5 surgical default; gives 0% in our tests). "
             "'all_prompt' = all prompt positions, no generated tokens.",
    ),
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
    env_type = _infer_env_type(task_list)
    env = _make_env(headless=not headed, env_type=env_type)
    policy_fn = POLICY_REGISTRY[policy]
    # v0.7-D: --hud (foreground) implies BOTH spawn-server AND publishing.
    # HUD_PUBLISH=1 env (background reruns connecting to a separately-started
    # ws_server) implies publishing only. Either path enables HudPublisher;
    # explicit HUD_PUBLISH=0 disables both regardless of --hud.
    env_pub = os.environ.get("HUD_PUBLISH")
    if env_pub == "0":
        publish_enabled = False
    elif env_pub == "1":
        publish_enabled = True
    else:
        publish_enabled = hud
    publisher = HudPublisher(enabled=publish_enabled)

    # Wrap brain_call to also pass the position_mode argument by default.
    _raw_brain = brain_call
    def brain_call_with_mode(prompt, edits=None, mode="act", **kw):
        kw.setdefault("position_mode", position_mode)
        return _raw_brain(prompt=prompt, edits=edits, mode=mode, **kw)

    agent = SAEAgent(
        brain_call=brain_call_with_mode,
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
                # v0.7-D: pre-load baseline cache for this task if it exists,
                # so the HUD's before/after diff can render. Cache is written
                # automatically when policy=baseline (below).
                baseline_cache = _load_baseline_cache(task["id"])
                for seed in range(trials):
                    # v0.7-D: emit policy_meta so HUD shows position_mode +
                    # decoding params alongside the policy badge.
                    publisher.policy_meta(
                        policy=policy,
                        position_mode=position_mode,
                        seed=seed,
                        max_steps=task.get("max_steps"),
                        max_new_tokens=agent.cfg.max_new_tokens,
                        temperature=agent.cfg.temperature,
                        steering_endpoint=("steer_act_with_noise"
                                           if policy == "noise"
                                           else "steer_act"),
                    )
                    # Emit cached baseline actions for each step in advance so
                    # the HUD can render the diff alongside the live action.
                    for cached in baseline_cache:
                        publisher.baseline_action(step=cached["step"], action=cached["action"])

                    r = agent.run(task=task, seed=seed, policy_name=policy)
                    results.append(r)
                    # v0.7-D: if this was a baseline run, refresh the baseline
                    # cache so subsequent non-baseline runs of this task can
                    # show a before/after diff in the HUD.
                    if policy == "baseline" and seed == 0:
                        _write_baseline_cache(task["id"], Path(r["log_path"]))
                    progress.advance(bar)

    out_file = out_dir / f"{policy}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    console.print(f"Wrote {len(results)} results to {out_file}")

    # v0.7-C: separate quantitative from qualitative results before reporting.
    # WebEnv tasks have verifier="manual_observation" / "qualitative" and always
    # return reward=0.0 — they're meant for live exploratory demos, not for
    # evidence. Reporting a 0.00% success rate on those is misleading because
    # the 0% reflects "no verifier wired", not "agent failed".
    QUALITATIVE_VERIFIERS = {"manual", "manual_observation", "qualitative"}
    task_by_id = {t["id"]: t for t in task_list}
    quantitative = [
        r for r in results
        if task_by_id.get(r["task_id"], {}).get("verifier", "") not in QUALITATIVE_VERIFIERS
    ]
    qualitative_count = len(results) - len(quantitative)
    if quantitative:
        success_rate = sum(r["success"] for r in quantitative) / len(quantitative)
        console.print(
            f"[bold]Success rate ({policy}, quantitative n={len(quantitative)}):[/bold] "
            f"{success_rate:.2%}"
        )
    else:
        console.print(
            f"[yellow]No quantitative tasks in this run "
            f"(all {len(results)} were qualitative / manual_observation). "
            f"Inspect trajectories under data/trajectories/ instead.[/yellow]"
        )
    if qualitative_count:
        console.print(
            f"[dim]Skipped {qualitative_count} qualitative-only result(s) "
            f"from the success-rate aggregate.[/dim]"
        )


if __name__ == "__main__":
    app()
