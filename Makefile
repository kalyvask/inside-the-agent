.PHONY: help install deploy verify bench demo clean

help:
	@echo "Inside the Agent - Make targets"
	@echo ""
	@echo "  install        Install Python deps + Playwright + HUD deps"
	@echo "  deploy         Deploy brain-server to Modal"
	@echo "  verify         Day 1 verification (5 tests)"
	@echo "  bench          Run benchmark (set POLICY=baseline|random|wrong-sign|targeted)"
	@echo "  demo           Start local HUD on http://localhost:3000"
	@echo "  clean          Remove caches and build artifacts"

install:
	pip install -e ".[dev]"
	playwright install chromium
	cd hud && npm install

deploy:
	modal deploy modal_deploy/app.py

verify:
	python -m verify.sae_smoke

POLICY ?= baseline
TASKS ?= shopgym/tasks/held_out.json
TRIALS ?= 3

bench:
	python -m bench.runner --policy $(POLICY) --tasks $(TASKS) --trials $(TRIALS)

demo:
	cd hud && npm run dev

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
