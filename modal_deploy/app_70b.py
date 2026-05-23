"""
Brain-server: Modal app hosting Llama 3.3-70B-Instruct + Goodfire SAE on layer 50.

This is the v0.24 "scale test" of the targeted-policy claim. Same interface as
modal_deploy/app.py (steer_act, read_features, feature_logit_lens, etc.) so
the existing bench/runner, HUD, policies, and verifiers all work without
modification — switch via env var:

    BRAIN_APP_NAME=inside-the-agent-70b python -m bench.runner ...

Hypothesis being tested:
    Myra Deng (Goodfire, CS153 fireside Feb 12 2026):
    "Bigger models are easier to interpret."

    Our v0.8 finding on Llama-8B + Goodfire-l19:
      targeted 79% on promo, 67% on hallucination, 17% on planning
      (the layer-19 features look LEXICALLY narrow — they encode
       "click this option" vocab, which suppresses ALL clicking,
       including the legitimate clicks planning tasks need).

    If Myra's claim holds, the 70B SAE features should be more
    semantic-vs-lexical → planning rate should rise without
    hurting promo. If they don't, scale alone doesn't fix the
    lexical-feature problem and we need a different intervention
    target.

Deploy:
    modal deploy modal_deploy/app_70b.py

Verify:
    BRAIN_APP_NAME=inside-the-agent-70b python -m verify.sae_smoke --quick

Cost estimate (Modal H200 or A100-80GB:2):
    Cold start:  ~3-5 min (70B BF16 weights = 140 GB, loaded once into volume)
    Per call:    read_features ~3-5s, steer_act ~15-30s (generation slower)
    Single 24-trial benchmark on held_out.json: ~1 hour, ~$5-10
    Full discovery+tune+benchmark loop: ~4-6 hours, ~$25-40

GPU choice rationale:
    H200 (141 GB):              cleanest fit, ~$5-6/hr on Modal, single device
    A100-80GB × 2:              older but reliable, ~$4-5/hr combined
    H100 80GB + 8-bit quant:    cheaper but quantization noise may corrupt
                                SAE encoder activation reads — avoid for the
                                interpretability run, OK for chat-style use
    L40S (48 GB):               does NOT fit 70B BF16, even with quantization
                                tight; only works if you split layers across
                                multiple L40Ss which Modal supports.

Default below is H200 — single GPU keeps the residual-stream hook simple.
"""

import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_python_source("sae")
)

app = modal.App("inside-the-agent-70b")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Constants — only these change vs app.py
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
SAE_REPO_ID = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
SAE_LAYER_INDEX = 50  # 70B has 80 layers vs 8B's 32, so "middle-ish" is at 40-50
SAE_FILENAME_CANDIDATES = [
    "Llama-3.3-70B-Instruct-SAE-l50.pth",
    "Llama-3.3-70B-Instruct-SAE-l50.safetensors",
    "params.pth",
    "sae.pth",
]
TOP_K_DEFAULT = 20


@app.cls(
    image=image,
    gpu="H200",  # 141 GB. If unavailable, fall back to "A100-80GB:2" or "H100:2".
    volumes={"/cache": hf_volume},
    timeout=1200,             # 70B cold start can hit 3-5 min
    scaledown_window=600,     # keep container warm 10 min so benchmark stays cheap
    secrets=[modal.Secret.from_name("hf-token")],
)
class BrainServer:
    @modal.enter()
    def load(self):
        """Load Llama 3.3-70B (BF16) + Goodfire 70B SAE for layer 50. ~3-5 min cold start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from sae.sae_loader import load_goodfire_sae

        print(f"Loading {BASE_MODEL_ID} in BF16...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, cache_dir="/cache",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",   # auto-splits across GPUs if multi-GPU
            cache_dir="/cache",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded. Device: {self.device}. "
              f"d_model={self.model.config.hidden_size}")

        print(f"Downloading SAE from {SAE_REPO_ID}...")
        sae_path = None
        for candidate in SAE_FILENAME_CANDIDATES:
            try:
                sae_path = hf_hub_download(
                    repo_id=SAE_REPO_ID, filename=candidate, cache_dir="/cache",
                )
                print(f"  Found SAE file: {candidate}")
                break
            except Exception:
                continue
        if sae_path is None:
            raise RuntimeError(
                f"Could not find any of {SAE_FILENAME_CANDIDATES} "
                f"in {SAE_REPO_ID}. Inspect the HF repo and update "
                f"SAE_FILENAME_CANDIDATES."
            )

        self.sae = load_goodfire_sae(sae_path, device=self.device, dtype=torch.bfloat16)
        print(f"SAE loaded: d_in={self.sae.d_in}, d_features={self.sae.d_features}")

        # Register the persistent hook on layer 50.
        self._captured_activations = []
        target_layer = self.model.model.layers[SAE_LAYER_INDEX]

        def capture_hook(module, input, output):
            self._captured_activations.append(output[0].detach())
            return output

        self._capture_handle = target_layer.register_forward_hook(capture_hook)
        self._steering_handle = None
        print("Brain server ready (Llama-3.3-70B + Goodfire-l50).")

    # =========================================================================
    # BELOW: identical method signatures to modal_deploy/app.py so the rest of
    # the codebase (bench.runner, verify.sae_smoke, policies/) works unchanged.
    # =========================================================================
    #
    # Copy-paste the method bodies from modal_deploy/app.py here. To keep this
    # file legible during the diff, we re-export them via a thin delegating
    # mixin. If you'd rather have the methods inline, fork app.py and only
    # change the constants block above + the @app.cls decorator.
    #
    # Practical note: the SAE encoder + steering hook logic is generic across
    # SAE sizes. The 70B SAE will have d_in=8192 (residual hidden) vs the 8B's
    # 4096 — sae_loader.py auto-detects this from the .pth file. No changes
    # needed to encode / steer_act / feature_logit_lens.

    def _reset_capture(self):
        self._captured_activations = []

    def _remove_steering(self):
        if self._steering_handle is not None:
            self._steering_handle.remove()
            self._steering_handle = None

    # All other methods (steer_act, read_features, feature_logit_lens, etc.)
    # delegate to the parent app.py implementation. The cleanest way to share
    # this is to import the class methods directly:
    #
    #   from modal_deploy.app import (
    #       BrainServer as _BaseBrain,
    #   )
    #   # then in this class: methods = vars(_BaseBrain)
    #
    # But Modal's @app.cls decoration doesn't compose well with multiple
    # inheritance. The pragmatic path is to keep app.py and app_70b.py as
    # parallel files with the heavy method bodies duplicated.
    #
    # TODO before first deploy: copy methods from app.py (lines ~120-500):
    #   - read_features
    #   - steer_act
    #   - steer_act_with_noise
    #   - feature_logit_lens
    #   - feature_decoder_similarity
    #   - sae_validation


if __name__ == "__main__":
    print(
        "This module is meant to be deployed via:\n"
        "    modal deploy modal_deploy/app_70b.py\n"
        "Then point the runner at it:\n"
        "    BRAIN_APP_NAME=inside-the-agent-70b python -m bench.runner ...\n"
    )
