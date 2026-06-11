.PHONY: help install ws hud replay deploy verify bench clean

help:
	@echo "Inside the Agent - Make targets"
	@echo ""
	@echo "  install        Install Python deps + Playwright + HUD deps"
	@echo ""
	@echo "  Offline demo (no API keys; run ws + hud in two terminals, then replay):"
	@echo "  ws             Start the WebSocket bridge on :8765 (seeds bundled demo run)"
	@echo "  hud            Build + serve the HUD on http://localhost:3000"
	@echo "  replay         Replay the bundled real-eBay demo run into the HUD"
	@echo ""
	@echo "  Live stack (needs Modal + HF token):"
	@echo "  deploy         Deploy brain-server to Modal"
	@echo "  verify         Day 1 verification (5 tests)"
	@echo "  bench          Run benchmark (set POLICY=baseline|random|wrong-sign|targeted)"
	@echo "  clean          Remove caches and build artifacts"

install:
	pip install -r requirements.txt
	playwright install chromium
	cd hud && npm install

# --- Offline demo --------------------------------------------------------
# ws_server seeds data/ from the tracked demo_assets/ fixture on startup, so a
# fresh clone can replay a real captured eBay run with zero API keys.
ws:
	python -m agent.ws_server

# Production build + start: `next dev`'s file watcher wedges under synced
# folders (OneDrive); build+start is reliable everywhere. The WS URL is baked
# at BUILD time.
hud:
	cd hud && NEXT_PUBLIC_WS_URL=ws://localhost:8765/feed npm run build && npm run start

replay:
	python -m verify.replay_trajectory data/trajectories/ebay_demo_targeted.jsonl

# --- Live stack ----------------------------------------------------------
deploy:
	modal deploy modal_deploy/app.py

verify:
	python -m verify.sae_smoke

POLICY ?= baseline
TASKS ?= shopgym/tasks/held_out.json
TRIALS ?= 3

bench:
	python -m bench.runner --policy $(POLICY) --tasks $(TASKS) --trials $(TRIALS)

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
