"""
Alternative SAE loader for Gemma 2 9B + Gemma Scope SAEs (DeepMind).

Use this if Llama 3.1 license is still pending tomorrow morning. Gemma 2 is
NOT gated — anyone can download it. Gemma Scope ships open SAEs on HuggingFace.

Differences from the Goodfire loader:
- Base model:    google/gemma-2-9b-it  (instead of meta-llama/Llama-3.1-8B-Instruct)
- SAE source:    google/gemma-scope-9b-it-res  (residual SAEs on the instruct variant)
- SAE format:    .npz files with JumpReLU activation
- Layers:        many available; we use layer 20 by default (mid-network)
- Activation:    JumpReLU(x) = x * (x > threshold)  rather than ReLU

Swap-in instructions:
1. In modal_deploy/app.py, change BASE_MODEL_ID and SAE_REPO_ID to the Gemma versions
2. Switch sae loader import from `load_goodfire_sae` to `load_gemma_scope_sae`
3. Update SAE_LAYER_INDEX to 20

Tradeoffs:
- Gemma 2 9B is a slightly larger model (~16-18GB BF16) — still fits L40S 48GB
- Gemma Scope SAEs have published feature browsers at neuronpedia.org (gemma-scope)
- L0 is usually around 70-100, comparable to Goodfire's 91
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JumpReLU(nn.Module):
    """Threshold-based sparse activation. f(x) = x * (x > threshold)."""

    def __init__(self, threshold: torch.Tensor):
        super().__init__()
        # threshold shape: (d_features,)
        self.threshold = nn.Parameter(threshold, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > self.threshold).to(x.dtype)


class GemmaScopeSAE(nn.Module):
    """
    Gemma Scope SAE: f(x) = JumpReLU(W_enc @ (x - b_dec) + b_enc).

    Stored on disk as a .npz with arrays:
        W_enc:     (d_in, d_features)
        W_dec:     (d_features, d_in)
        b_enc:     (d_features,)
        b_dec:     (d_in,)            (called "pre_bias" in some releases)
        threshold: (d_features,)      (JumpReLU thresholds)
    """

    def __init__(self, d_in: int, d_features: int, dtype=torch.bfloat16):
        super().__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_features, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_features, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_features, d_in, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))
        self.activation = JumpReLU(torch.zeros(d_features, dtype=dtype))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return self.activation(pre)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        x_hat = self.decode(features)
        return x_hat, features


def load_gemma_scope_sae(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> GemmaScopeSAE:
    """
    Load a Gemma Scope SAE from a .npz file.

    Gemma Scope releases SAEs as compressed numpy arrays. The exact filename
    you want is typically something like:
        layer_20/width_16k/average_l0_71/params.npz

    Browse releases here:
        https://huggingface.co/google/gemma-scope-9b-it-res/tree/main
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Gemma SAE checkpoint not found: {checkpoint_path}")

    print(f"[gemma_loader] Loading {checkpoint_path.name}...")
    npz = np.load(checkpoint_path)

    # Standard key names in Gemma Scope releases.
    aliases = {
        "W_enc": ["W_enc", "encoder.W"],
        "W_dec": ["W_dec", "decoder.W"],
        "b_enc": ["b_enc", "encoder.b"],
        "b_dec": ["b_dec", "pre_bias", "decoder.b"],
        "threshold": ["threshold", "log_threshold"],
    }

    def find(canonical):
        for k in aliases[canonical]:
            if k in npz:
                return npz[k]
        raise KeyError(
            f"Missing {canonical} in Gemma SAE. Keys available: {list(npz.keys())}"
        )

    W_enc = find("W_enc")
    W_dec = find("W_dec")
    b_enc = find("b_enc")
    b_dec = find("b_dec")
    raw_threshold = find("threshold")

    # Threshold may be stored as log; detect by checking if all negative.
    if (raw_threshold < 0).all():
        threshold = np.exp(raw_threshold)
    else:
        threshold = raw_threshold

    # Determine orientation using b_dec as a fixed reference: b_dec is always
    # d_in-sized in Gemma Scope releases. The earlier heuristic compared
    # W_enc's two dims and assumed d_in > d_features, which fails on
    # sparse-overcomplete SAEs where d_features > d_in (the normal case).
    d_in = int(np.asarray(b_dec).reshape(-1).shape[0])
    if W_enc.shape[0] == d_in:
        d_features = W_enc.shape[1]
    elif W_enc.shape[1] == d_in:
        d_features = W_enc.shape[0]
        W_enc = W_enc.T
    else:
        raise ValueError(
            f"Could not align W_enc {W_enc.shape} with b_dec d_in={d_in}. "
            f"Available .npz keys: {list(npz.keys())}"
        )

    print(f"[gemma_loader] d_in={d_in}, d_features={d_features}")

    sae = GemmaScopeSAE(d_in=d_in, d_features=d_features, dtype=dtype)

    if W_dec.shape != (d_features, d_in):
        W_dec = W_dec.T

    sae.W_enc.data = torch.from_numpy(W_enc).to(dtype)
    sae.W_dec.data = torch.from_numpy(W_dec).to(dtype)
    sae.b_enc.data = torch.from_numpy(b_enc).to(dtype).reshape(-1)
    sae.b_dec.data = torch.from_numpy(b_dec).to(dtype).reshape(-1)
    sae.activation.threshold.data = torch.from_numpy(threshold).to(dtype).reshape(-1)

    sae.to(device)
    sae.eval()
    print(f"[gemma_loader] Gemma Scope SAE loaded on {device}")
    return sae
