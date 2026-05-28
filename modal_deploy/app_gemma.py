"""
Alternative Modal brain-server using Gemma 2 9B + Gemma Scope SAE.

Use this if Llama 3.1 license is still pending. To swap:

  modal deploy modal_deploy/app_gemma.py     # instead of modal_deploy/app.py
  # then set BRAIN_APP_NAME=inside-the-agent-gemma in .env

This file mirrors app.py exactly except for the base model and SAE source.
Endpoints, hook logic, and steering primitives are identical.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("modal_deploy/requirements.txt")
    .add_local_python_source("sae")
)

app = modal.App("inside-the-agent-gemma")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Gemma 2 9B is gated (Google updated their policy after this file was first
# written). Accept the license at https://huggingface.co/google/gemma-2-9b-it
# before deploying. Google's grant is usually instant. Gemma Scope SAEs are open.
BASE_MODEL_ID = "google/gemma-2-9b-it"
SAE_REPO_ID = "google/gemma-scope-9b-it-res"
# Specific SAE checkpoint. Layer 20 of 42 (~48% depth, the middle-ish residual),
# width 16k, L0=91. The L0 picks the closest match to Goodfire's Llama-8B SAE
# (L0~91) so cross-model comparisons are apples-to-apples on sparsity.
# Other available L0 values at this (layer, width): 14, 25, 47, 189.
SAE_FILENAME = "layer_20/width_16k/average_l0_91/params.npz"
SAE_LAYER_INDEX = 20
TOP_K_DEFAULT = 20


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/cache": hf_volume},
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],
)
class BrainServer:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download

        from sae.sae_loader_gemma import load_gemma_scope_sae

        print(f"Loading {BASE_MODEL_ID} in BF16...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, cache_dir="/cache"
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

        print(f"Downloading Gemma Scope SAE: {SAE_FILENAME}")
        sae_path = hf_hub_download(
            repo_id=SAE_REPO_ID,
            filename=SAE_FILENAME,
            cache_dir="/cache",
        )
        self.sae = load_gemma_scope_sae(sae_path, device=self.device, dtype=torch.bfloat16)
        print(f"SAE loaded: d_in={self.sae.d_in}, d_features={self.sae.d_features}")

        self._captured_activations = []
        target_layer = self.model.model.layers[SAE_LAYER_INDEX]

        def capture_hook(module, input, output):
            self._captured_activations.append(output[0].detach())
            return output

        self._capture_handle = target_layer.register_forward_hook(capture_hook)
        self._steering_handle = None
        print("Gemma brain-server ready.")

    def _reset_capture(self):
        self._captured_activations = []

    def _remove_steering(self):
        if self._steering_handle is not None:
            self._steering_handle.remove()
            self._steering_handle = None

    def _install_steering(
        self,
        edits: dict,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        position_mode: str = "all",
    ):
        """Install steering hook with per-position dispatch. Mirrors app.py.

        position_mode:
          "all":              modify every position in every forward pass
          "all_prompt":       modify all positions during prefill, skip generation
          "last_prompt_only": modify only the last prefill token, skip generation
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
        W_dec_rows = self.sae.W_dec[edit_features]
        delta_activation = (edit_deltas.unsqueeze(-1) * W_dec_rows).sum(dim=0)

        call_count = [0]

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
        self._steering_call_count = call_count

    def _top_k_features(self, top_k: int = TOP_K_DEFAULT):
        import torch
        if not self._captured_activations:
            return []
        last_act = self._captured_activations[-1][0, -1, :]
        with torch.no_grad():
            features = self.sae.encode(last_act.unsqueeze(0)).squeeze(0)
        values, indices = torch.topk(features, top_k)
        return [
            {"id": int(idx), "activation": float(val)}
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

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
        import torch
        self._reset_capture()
        self._remove_steering()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        return {"top_features": self._top_k_features(top_k), "prompt": prompt}

    @modal.method()
    def steer_act(
        self,
        prompt: str,
        edits: dict = None,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = TOP_K_DEFAULT,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        position_mode: str = "all",
    ):
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
