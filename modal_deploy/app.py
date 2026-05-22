"""
Brain-server: Modal app hosting Llama 3.1-8B + Goodfire SAE on layer 19.

Endpoints:
  GET  /health           -> {"status": "ok", "model_loaded": bool}
  POST /read_features    -> {"top_features": [...], "logits": [...]}
  POST /steer_act        -> {"action": "...", "features_during": [...]}
  POST /act              -> {"action": "...", "features": [...]}    (alias to steer_act with no edits)

Deploy:
  modal deploy modal_deploy/app.py

Verify:
  python -m verify.sae_smoke
"""

import modal

# ---------------------------------------------------------------------------
# Image: pinned for reproducibility. BF16 model + SAE; no quantization.
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    # Mount our local sae/ module so the brain-server can import load_goodfire_sae
    .add_local_python_source("sae")
)

app = modal.App("inside-the-agent")

# Persistent volume so we don't re-download model weights every cold start.
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAE_REPO_ID = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_LAYER_INDEX = 19
TOP_K_DEFAULT = 20

# ---------------------------------------------------------------------------
# Brain-server class. One persistent container; model loaded once on enter.
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="L40S",  # 48GB VRAM, ~30% cheaper than A100-80GB. Llama-8B BF16 fits easily.
    volumes={"/cache": hf_volume},
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],  # set: modal secret create hf-token HF_TOKEN=...
)
class BrainServer:
    @modal.enter()
    def load(self):
        """Load Llama 3.1-8B (BF16) + Goodfire SAE for layer 19. ~2 min cold start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download

        # Import the SAE loader (lives in sae/ in the repo; copy in via mount).
        from sae.sae_loader import load_goodfire_sae

        print(f"Loading {BASE_MODEL_ID} in BF16...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            cache_dir="/cache",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/cache",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        print(f"Downloading SAE from {SAE_REPO_ID}...")
        # Verified filename from HuggingFace repo listing (2.15 GB).
        # SAE was trained on LMSYS-Chat-1M; L0 sparsity = 91 active features/token.
        sae_path = hf_hub_download(
            repo_id=SAE_REPO_ID,
            filename="Llama-3.1-8B-Instruct-SAE-l19.pth",
            cache_dir="/cache",
        )
        self.sae = load_goodfire_sae(sae_path, device=self.device, dtype=torch.bfloat16)
        print(f"SAE loaded: d_in={self.sae.d_in}, d_features={self.sae.d_features}")

        # Register the persistent hook (always captures L19 residual stream).
        self._captured_activations = []
        target_layer = self.model.model.layers[SAE_LAYER_INDEX]

        def capture_hook(module, input, output):
            # output[0] is the hidden state of shape (batch, seq, d_in)
            self._captured_activations.append(output[0].detach())
            return output

        self._capture_handle = target_layer.register_forward_hook(capture_hook)

        # The steering hook is added/removed per-request (so it's idempotent).
        self._steering_handle = None
        print("Brain-server ready.")

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _reset_capture(self):
        self._captured_activations = []

    def _remove_steering(self):
        if self._steering_handle is not None:
            self._steering_handle.remove()
            self._steering_handle = None

    def _install_steering(self, edits: dict, clamp_min: float = -100.0, clamp_max: float = 100.0):
        """
        Install a forward hook on layer 19 that applies feature-level deltas to
        the residual stream. Edits = {feature_id: delta}.

        The clamp is a safety rail against runaway activations only; do not set
        it tighter than Llama's typical residual range (±50-100). Tighter clamps
        will silently destroy the model's internal state and produce garbled output.
        """
        import torch

        self._remove_steering()
        target_layer = self.model.model.layers[SAE_LAYER_INDEX]

        # Precompute the activation-space delta: sum of (delta_i * W_dec[i]).
        # This avoids per-step encode/decode roundtripping.
        edit_features = list(edits.keys())
        edit_deltas = torch.tensor(
            [edits[f] for f in edit_features],
            dtype=torch.bfloat16,
            device=self.device,
        )
        # W_dec shape: (d_features, d_in). Select rows for edited features.
        W_dec_rows = self.sae.W_dec[edit_features]  # (n_edits, d_in)
        delta_activation = (edit_deltas.unsqueeze(-1) * W_dec_rows).sum(dim=0)  # (d_in,)

        def steering_hook(module, input, output):
            hidden = output[0]  # (batch, seq, d_in)
            new_hidden = hidden + delta_activation
            # Safety clamp wide enough to NOT alter normal residual values.
            new_hidden = torch.clamp(new_hidden, clamp_min, clamp_max)
            # output is a tuple; replace first element
            return (new_hidden,) + output[1:]

        self._steering_handle = target_layer.register_forward_hook(steering_hook)

    def _top_k_features(self, top_k: int = TOP_K_DEFAULT):
        """Given captured L19 activations, return top-k SAE features."""
        import torch

        if not self._captured_activations:
            return []
        # Use the LAST token's activation (the one that produced the new token).
        last_act = self._captured_activations[-1][0, -1, :]  # (d_in,)
        with torch.no_grad():
            features = self.sae.encode(last_act.unsqueeze(0)).squeeze(0)  # (d_features,)
        values, indices = torch.topk(features, top_k)
        return [
            {"id": int(idx), "activation": float(val)}
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @modal.method()
    def health(self):
        return {
            "status": "ok",
            "model": BASE_MODEL_ID,
            "sae": SAE_REPO_ID,
            "layer": SAE_LAYER_INDEX,
            "d_features": int(self.sae.d_features),
        }

    @modal.method()
    def read_features(self, prompt: str, top_k: int = TOP_K_DEFAULT):
        """Run a forward pass on the prompt, return top-k SAE features."""
        import torch

        self._reset_capture()
        self._remove_steering()  # no steering for plain reads

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        return {
            "top_features": self._top_k_features(top_k),
            "prompt": prompt,
        }

    @modal.method()
    def steer_act(
        self,
        prompt: str,
        edits: dict = None,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = TOP_K_DEFAULT,
        clamp_min: float = -100.0,
        clamp_max: float = 100.0,
    ):
        """
        Generate a response with optional steering.

        edits = {"1234": 5.0, "5678": -2.0}  # feature_id -> delta
        """
        import torch

        self._reset_capture()
        self._remove_steering()

        if edits:
            # Normalize keys to ints
            edits = {int(k): float(v) for k, v in edits.items()}
            self._install_steering(edits, clamp_min, clamp_max)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        result = {
            "response": response,
            "top_features": self._top_k_features(top_k),
            "edits_applied": edits or {},
        }
        self._remove_steering()
        return result


# ---------------------------------------------------------------------------
# Local entrypoints for quick testing
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke():
    """Quick smoke test from local machine. Run: modal run modal_deploy/app.py::smoke"""
    server = BrainServer()
    print("Health:", server.health.remote())
    print()
    print("Reading features for 'plan a browser action'...")
    r = server.read_features.remote("plan a browser action")
    for f in r["top_features"][:10]:
        print(f"  feature {f['id']:>6d}: {f['activation']:.3f}")
    print()
    print("Steered generation with feature edit {1234: 5.0}...")
    r = server.steer_act.remote(
        prompt="You are a browser agent. The page shows a 'Buy Now' button for earbuds. The goal is to buy a USB-C cable. What do you do?",
        edits={1234: 5.0},
        max_new_tokens=80,
    )
    print("Response:", r["response"])
    print("Top features after generation:")
    for f in r["top_features"][:5]:
        print(f"  feature {f['id']:>6d}: {f['activation']:.3f}")
