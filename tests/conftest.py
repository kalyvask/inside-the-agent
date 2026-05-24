"""
tests/conftest.py — stub heavy modules so test collection works in CI.

CI installs only what the unit tests need (pytest, ruff, pyyaml, pydantic,
typer, rich, jsonlines, requests). It does NOT install torch, transformers,
modal, or playwright because those add 1+ GB to the CI image and the unit
tests don't actually exercise them; they only need the modules to be
importable so production-code import chains don't crash at collection time.

This conftest pre-populates sys.modules with MagicMock stubs for each heavy
module before pytest collects any test. Any test that genuinely needs the
real implementation (e.g. an integration smoke against a real Modal
container) lives in the verify/ scripts or under bench.runner CLI usage,
not in tests/, so the stubbing here is safe for the unit-test surface.

If a future unit test does need the real torch/playwright (e.g. a
property-based test on a real model output), mark it with
@pytest.mark.skipif(_HEAVY_MODULES_ARE_STUBS, ...) and have CI install
the real package instead.
"""

import sys
from unittest.mock import MagicMock


_HEAVY_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "transformers",
    "modal",
    "playwright",
    "playwright.sync_api",
    "playwright.async_api",
    "huggingface_hub",
    "safetensors",
    "safetensors.torch",
]

_HEAVY_MODULES_ARE_STUBS = False

for _name in _HEAVY_MODULES:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()
        _HEAVY_MODULES_ARE_STUBS = True
