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

    def _install_steering(
        self,
        edits: dict,
        clamp_min: float = -100.0,
        clamp_max: float = 100.0,
        position_mode: str = "last_prompt_only",
    ):
        """
        Install a forward hook on layer 19 that applies feature-level deltas
        to the residual stream. Edits = {feature_id: delta}.

        position_mode controls WHERE the delta is added:
          - "last_prompt_only" (default, defensible):
              Modify only the LAST token of the PREFILL forward pass — i.e.,
              the residual whose hidden state predicts the first generated
              token. Subsequent generation steps (KV-cached, seq=1 each) are
              NOT modified. This is what "Step-0 steering at the decision
              moment" actually means.
          - "all_prompt":
              Modify all positions during prefill; do not modify during
              generation. Less surgical than "last_prompt_only" but still
              respects the prompt/generation boundary.
          - "all":
              Legacy v0.1 behavior. Modify every position in every forward
              pass — including all generated tokens. Documented but no longer
              the default because it overstates the surface of the
              intervention.

        Returns nothing. The handle is stored on self._steering_handle.
        """
        import torch

        self._remove_steering()
        target_layer = self.model.model.layers[SAE_LAYER_INDEX]

        edit_features = list(edits.keys())
        edit_deltas = torch.tensor(
            [edits[f] for f in edit_features],
            dtype=torch.bfloat16,
            device=self.device,
        )
        # W_dec shape: (d_features, d_in). Select rows for edited features.
        W_dec_rows = self.sae.W_dec[edit_features]  # (n_edits, d_in)
        delta_activation = (edit_deltas.unsqueeze(-1) * W_dec_rows).sum(dim=0)  # (d_in,)

        # Closure state to track which forward pass we are on. The first call
        # in any sequence is the prefill (seq_len = prompt_len); subsequent
        # calls have seq_len = 1 because the KV cache is in use.
        call_count = [0]

        def steering_hook(module, input, output):
            hidden = output[0]  # (batch, seq, d_in)
            call_count[0] += 1
            is_prefill = call_count[0] == 1

            if position_mode == "all":
                new_hidden = hidden + delta_activation
            elif position_mode == "all_prompt":
                if is_prefill:
                    new_hidden = hidden + delta_activation
                else:
                    new_hidden = hidden  # no-op for generation tokens
            elif position_mode == "last_prompt_only":
                if is_prefill:
                    # Modify only the residual at the LAST prompt token.
                    new_hidden = hidden.clone()
                    new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta_activation
                else:
                    new_hidden = hidden  # no-op for generation tokens
            else:
                # Unknown mode → safe default of no modification.
                new_hidden = hidden

            new_hidden = torch.clamp(new_hidden, clamp_min, clamp_max)
            return (new_hidden,) + output[1:]

        self._steering_handle = target_layer.register_forward_hook(steering_hook)
        self._steering_call_count = call_count  # so we can read it for telemetry

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
    def feature_logit_lens(self, feature_id: int, top_k: int = 30):
        """
        Project a feature's decoder direction onto the unembedding matrix.

        Returns the tokens this feature most "promotes" and most "suppresses"
        in the model's next-token distribution. This is the strongest internal
        evidence of what a feature encodes — independent of any prompt corpus.

        Mathematically:
            scores = W_dec[fid] @ W_U  (1, vocab_size)
            promoted = argsort(scores)[-top_k:]   # tokens with highest score
            suppressed = argsort(scores)[:top_k]  # tokens with lowest (most negative) score
        """
        import torch

        # Get the model's unembedding matrix.
        # For Llama, it's the lm_head.weight: shape (vocab_size, d_in).
        W_U = self.model.lm_head.weight  # (vocab_size, d_in)
        decoder_dir = self.sae.W_dec[feature_id].to(W_U.dtype).to(W_U.device)  # (d_in,)

        # scores[v] = W_dec[fid] @ W_U[v].T  (effectively W_dec[fid] @ W_U.T)
        scores = W_U @ decoder_dir  # (vocab_size,)

        # Top-k promoted (highest scores) and suppressed (lowest = most negative).
        top_promoted = torch.topk(scores, top_k)
        top_suppressed = torch.topk(scores, top_k, largest=False)

        promoted = []
        for i in range(top_k):
            tok_id = int(top_promoted.indices[i].item())
            promoted.append({
                "token_id": tok_id,
                "token": self.tokenizer.decode([tok_id]),
                "score": float(top_promoted.values[i].item()),
            })

        suppressed = []
        for i in range(top_k):
            tok_id = int(top_suppressed.indices[i].item())
            suppressed.append({
                "token_id": tok_id,
                "token": self.tokenizer.decode([tok_id]),
                "score": float(top_suppressed.values[i].item()),
            })

        return {
            "feature_id": feature_id,
            "decoder_norm": float(decoder_dir.norm().item()),
            "score_mean": float(scores.mean().item()),
            "score_std": float(scores.std().item()),
            "promoted": promoted,
            "suppressed": suppressed,
        }

    @modal.method()
    def feature_decoder_similarity(self, feature_ids: list, top_k: int = 10):
        """
        For each feature, return its nearest decoder-vector neighbors by cosine
        similarity. Features with high cosine similarity likely encode similar
        concepts.
        """
        import torch
        import torch.nn.functional as F

        # W_dec shape: (d_features, d_in)
        W_dec = self.sae.W_dec.to(torch.float32)
        # Normalize all rows.
        W_dec_norm = F.normalize(W_dec, dim=1)

        results = {}
        for fid in feature_ids:
            target = W_dec_norm[fid]  # (d_in,)
            sims = W_dec_norm @ target  # (d_features,)
            # Top-k+1 because the feature itself has similarity 1.0.
            top = torch.topk(sims, top_k + 1)
            neighbors = []
            for i in range(1, top_k + 1):  # skip self
                neighbors.append({
                    "feature_id": int(top.indices[i].item()),
                    "cosine_sim": float(top.values[i].item()),
                })
            results[fid] = neighbors
        return {"similarities": results}

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
        position_mode: str = "last_prompt_only",
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
            self._install_steering(edits, clamp_min, clamp_max, position_mode=position_mode)

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
