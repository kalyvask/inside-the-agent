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
    semantic-vs-lexical so planning rate should rise without
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
    A100-80GB x 2:              older but reliable, ~$4-5/hr combined
    H100 80GB + 8-bit quant:    cheaper but quantization noise may corrupt
                                SAE encoder activation reads, avoid for the
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
    "Llama-3.3-70B-Instruct-SAE-l50.pt",   # verified 2026-05-23 via HF API
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

        # H200/Hopper workaround: cuDNN's SDPA backend hits "No execution plans
        # support the graph" on Llama-3.3-70B BF16. Disabling cuDNN-SDP via
        # torch.backends.cuda.enable_cudnn_sdp(False) wasn't sufficient — the
        # path still routes through the broken cuDNN frontend. Force eager
        # attention so PyTorch uses pure-tensor attention math. Slower
        # (~1.3-1.8x) but reliable; sae_smoke + benchmark cost dominated by
        # generation length, not attention kernel choice.
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("cuDNN SDP disabled; using attn_implementation=eager (H200 workaround).")

        print(f"Loading {BASE_MODEL_ID} in BF16...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, cache_dir="/cache",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",                # auto-splits across GPUs if multi-GPU
            cache_dir="/cache",
            attn_implementation="eager",      # pure-tensor attention, bypasses SDPA + cuDNN
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

    # -----------------------------------------------------------------------
    # Internals — identical to app.py because the SAE+steering logic is
    # SAE-size-generic. The only thing that changes is which layer we hook
    # (SAE_LAYER_INDEX = 50 instead of 19) and the residual width (d_in
    # auto-detected from the .pth file by sae_loader).
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
        Install a forward hook on the SAE layer that applies feature-level
        deltas to the residual stream. Edits = {feature_id: delta}.

        position_mode is dispatched identically to app.py.
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
        W_dec_rows = self.sae.W_dec[edit_features]  # (n_edits, d_in)
        delta_activation = (edit_deltas.unsqueeze(-1) * W_dec_rows).sum(dim=0)  # (d_in,)

        call_count = [0]

        def steering_hook(module, input, output):
            hidden = output[0]
            call_count[0] += 1
            is_prefill = call_count[0] == 1

            if position_mode == "all":
                new_hidden = hidden + delta_activation
            elif position_mode == "all_prompt":
                if is_prefill:
                    new_hidden = hidden + delta_activation
                else:
                    new_hidden = hidden
            elif position_mode == "last_prompt_only":
                if is_prefill:
                    new_hidden = hidden.clone()
                    new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta_activation
                else:
                    new_hidden = hidden
            else:
                new_hidden = hidden

            new_hidden = torch.clamp(new_hidden, clamp_min, clamp_max)
            return (new_hidden,) + output[1:]

        self._steering_handle = target_layer.register_forward_hook(steering_hook)
        self._steering_call_count = call_count

    def _top_k_features(self, top_k: int = TOP_K_DEFAULT):
        """Given captured layer-50 activations, return top-k SAE features."""
        import torch

        if not self._captured_activations:
            return []
        last_act = self._captured_activations[-1][0, -1, :]  # (d_in,)
        with torch.no_grad():
            features = self.sae.encode(last_act.unsqueeze(0)).squeeze(0)
        values, indices = torch.topk(features, top_k)
        return [
            {"id": int(idx), "activation": float(val)}
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

    # -----------------------------------------------------------------------
    # Endpoints — identical signatures to app.py so bench.runner, HUD,
    # policies/, and verify/ all switch over via BRAIN_APP_NAME alone.
    # -----------------------------------------------------------------------

    @modal.method()
    def health(self):
        return {
            "status": "ok",
            "model": BASE_MODEL_ID,
            "sae": SAE_REPO_ID,
            "layer": SAE_LAYER_INDEX,
            "d_features": int(self.sae.d_features),
            "d_in": int(self.sae.d_in),
        }

    @modal.method()
    def feature_logit_lens(self, feature_id: int, top_k: int = 30):
        """
        Project a feature's decoder direction onto the unembedding matrix.

        Returns the tokens this feature most "promotes" and most "suppresses"
        in the model's next-token distribution. Independent of any prompt
        corpus, this is the strongest internal characterization of what a
        feature encodes.

        For the 70B SAE the absolute magnitudes will differ (larger W_U),
        but the ranking is what matters for labeling.
        """
        import torch

        W_U = self.model.lm_head.weight  # (vocab_size, d_in)
        decoder_dir = self.sae.W_dec[feature_id].to(W_U.dtype).to(W_U.device)  # (d_in,)
        scores = W_U @ decoder_dir  # (vocab_size,)

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
    def sae_validation(self, prompts: list = None, n_random_features: int = 100):
        """
        Full SAE sanity suite. Same metrics as the 8B version. Expected ranges
        for the 70B-l50 SAE per Goodfire's card:

          mean_l0:                 ~60-100 (vs 91 on 8B-l19)
          mean_reconstruction_err: <0.5
          wrong_layer_l0_layer_0:  much higher than mean_l0 (sanity check —
                                   layer-50-trained SAE shouldn't reconstruct
                                   layer-0 activations cleanly)

        If wrong_layer_l0 is comparable to mean_l0, our layer hook is
        misregistered. That's the #1 failure mode for the cross-scale port
        because layer-50 vs layer-19 hook semantics could differ.
        """
        import torch

        if prompts is None:
            prompts = [
                "The cat sat on the mat.",
                "Quantum entanglement is a phenomenon.",
                "Click the buy now button.",
                "Plan three steps before acting.",
                "The page shows a promotional banner.",
                "Goal: buy a USB-C cable. Stay focused.",
                "Hello, how are you today?",
                "Translate this sentence to French.",
            ]

        l0_values = []
        recon_errors = []
        for prompt in prompts:
            self._reset_capture()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            if not self._captured_activations:
                continue
            acts = self._captured_activations[-1][0]  # (seq, d_in)
            with torch.no_grad():
                features = self.sae.encode(acts)
                x_hat = self.sae.decode(features)
            l0 = (features > 0.01).float().sum(dim=-1).mean().item()
            l0_values.append(l0)
            err = (acts - x_hat).norm(dim=-1) / acts.norm(dim=-1).clamp(min=1e-6)
            recon_errors.append(err.mean().item())

        mean_l0 = sum(l0_values) / max(1, len(l0_values))
        mean_recon = sum(recon_errors) / max(1, len(recon_errors))

        with torch.no_grad():
            dec_norms = self.sae.W_dec.norm(dim=-1)
            enc_norms = self.sae.W_enc.norm(dim=0)
        dec_stats = {
            "min": float(dec_norms.min().item()),
            "max": float(dec_norms.max().item()),
            "median": float(dec_norms.median().item()),
            "mean": float(dec_norms.mean().item()),
            "std": float(dec_norms.std().item()),
        }
        enc_stats = {
            "min": float(enc_norms.min().item()),
            "max": float(enc_norms.max().item()),
            "median": float(enc_norms.median().item()),
            "mean": float(enc_norms.mean().item()),
            "std": float(enc_norms.std().item()),
        }

        # Layer-0 sanity: applying the layer-50 SAE to layer-0 acts should
        # be out-of-distribution. If it isn't, the hook is misregistered.
        target_layer_0 = self.model.model.layers[0]
        layer0_capture = []

        def capture0(module, input, output):
            layer0_capture.append(output[0].detach())
            return output

        h = target_layer_0.register_forward_hook(capture0)
        try:
            self._reset_capture()
            inputs = self.tokenizer(prompts[0], return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
        finally:
            h.remove()

        wrong_layer_l0 = None
        if layer0_capture:
            acts0 = layer0_capture[-1][0]
            with torch.no_grad():
                features0 = self.sae.encode(acts0)
            wrong_layer_l0 = (features0 > 0.01).float().sum(dim=-1).mean().item()

        sanity_prompt = "The cat sat on the mat."
        self._reset_capture()
        inputs = self.tokenizer(sanity_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        sanity_top = []
        if self._captured_activations:
            last_act = self._captured_activations[-1][0, -1, :]
            with torch.no_grad():
                f = self.sae.encode(last_act.unsqueeze(0)).squeeze(0)
            top = torch.topk(f, 5)
            sanity_top = [
                {"feature_id": int(top.indices[i].item()),
                 "activation": float(top.values[i].item())}
                for i in range(5)
            ]

        return {
            "mean_l0_per_token": mean_l0,
            "expected_l0_from_card": 80,  # 70B SAE card suggests ~60-100
            "mean_reconstruction_relative_error": mean_recon,
            "decoder_norms": dec_stats,
            "encoder_norms": enc_stats,
            "wrong_layer_l0_layer_0": wrong_layer_l0,
            "sanity_top_features_cat_prompt": sanity_top,
            "n_prompts": len(prompts),
            "model": BASE_MODEL_ID,
            "sae": SAE_REPO_ID,
            "layer": SAE_LAYER_INDEX,
        }

    @modal.method()
    def feature_decoder_similarity(self, feature_ids: list, top_k: int = 10):
        """For each feature, return its nearest decoder-vector neighbors by cosine similarity."""
        import torch
        import torch.nn.functional as F

        W_dec = self.sae.W_dec.to(torch.float32)
        W_dec_norm = F.normalize(W_dec, dim=1)

        results = {}
        for fid in feature_ids:
            target = W_dec_norm[fid]
            sims = W_dec_norm @ target
            top = torch.topk(sims, top_k + 1)
            neighbors = []
            for i in range(1, top_k + 1):  # skip self at index 0
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
        self._remove_steering()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        return {
            "top_features": self._top_k_features(top_k),
            "prompt": prompt,
        }

    @modal.method()
    def steer_act_with_noise(
        self,
        prompt: str,
        noise_seed: int = 0,
        noise_norm: float = 6.0,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = TOP_K_DEFAULT,
        position_mode: str = "last_prompt_only",
        clamp_min: float = -100.0,
        clamp_max: float = 100.0,
    ):
        """
        Matched-norm noise control. Identical to app.py except d_in is 8192
        for the 70B (vs 4096 for the 8B); sae_loader auto-detects this so no
        code change needed. The same noise_norm of 6.0 produces a SMALLER
        relative perturbation on the wider 70B residual stream, so during
        magnitude tuning expect to try noise_norm in {6, 12, 24} to span
        comparable %-of-norm magnitudes.
        """
        import torch

        self._reset_capture()
        self._remove_steering()

        d_in = self.sae.d_in
        gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        noise = torch.randn(d_in, generator=gen).to(self.device).to(torch.bfloat16)
        noise = noise / noise.norm() * float(noise_norm)

        target_layer = self.model.model.layers[SAE_LAYER_INDEX]
        call_count = [0]
        delta_activation = noise

        def steering_hook(module, input, output):
            hidden = output[0]
            call_count[0] += 1
            is_prefill = call_count[0] == 1
            if position_mode == "all":
                new_hidden = hidden + delta_activation
            elif position_mode == "all_prompt":
                new_hidden = hidden + delta_activation if is_prefill else hidden
            elif position_mode == "last_prompt_only":
                if is_prefill:
                    new_hidden = hidden.clone()
                    new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta_activation
                else:
                    new_hidden = hidden
            else:
                new_hidden = hidden
            new_hidden = torch.clamp(new_hidden, clamp_min, clamp_max)
            return (new_hidden,) + output[1:]

        self._steering_handle = target_layer.register_forward_hook(steering_hook)

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
            "noise_norm": float(noise_norm),
            "noise_seed": int(noise_seed),
        }
        self._remove_steering()
        return result

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

        Note: feature indices DO NOT transfer from the 8B SAE. The "ui-selection"
        and "distraction-avoidance" features in the 70B SAE will have different
        IDs — run verify/feature_drill on the 70B brain first to discover them.
        """
        import torch

        self._reset_capture()
        self._remove_steering()

        if edits:
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
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke():
    """Quick smoke test from local machine.
    Run: modal run modal_deploy/app_70b.py::smoke
    """
    server = BrainServer()
    print("Health:", server.health.remote())
    print()
    print("Reading features for 'plan a browser action'...")
    r = server.read_features.remote("plan a browser action")
    for f in r["top_features"][:10]:
        print(f"  feature {f['id']:>6d}: {f['activation']:.3f}")
    print()
    print("Steered generation with a placeholder edit {1234: 5.0}...")
    print("(In real use you'll first discover the 70B feature IDs via "
          "verify/feature_drill; the 8B IDs DO NOT transfer.)")
    r = server.steer_act.remote(
        prompt=(
            "You are a browser agent. The page shows a 'Buy Now' button for "
            "earbuds. The goal is to buy a USB-C cable. What do you do?"
        ),
        edits={1234: 5.0},
        max_new_tokens=80,
    )
    print("Response:", r["response"])
    print("Top features after generation:")
    for f in r["top_features"][:5]:
        print(f"  feature {f['id']:>6d}: {f['activation']:.3f}")


if __name__ == "__main__":
    print(
        "This module is meant to be deployed via:\n"
        "    modal deploy modal_deploy/app_70b.py\n"
        "Then point the runner at it:\n"
        "    BRAIN_APP_NAME=inside-the-agent-70b python -m bench.runner ...\n"
    )
