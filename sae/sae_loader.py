"""
Generic Sparse Autoencoder module + loader for Goodfire's open-weight SAEs.

Verified from HuggingFace repo (Goodfire/Llama-3.1-8B-Instruct-SAE-l19):
    Filename: Llama-3.1-8B-Instruct-SAE-l19.pth  (2.15 GB)
    Trained on: LMSYS-Chat-1M
    Sparsity: L0 = 91 (active features per token)
    Layer: 19 of Llama-3.1-8B-Instruct
    Note: Goodfire's released SAE may use a TopK activation rather than ReLU.
          If our ReLU encoder produces wildly different L0 from 91 on test
          prompts, we may need to wrap with TopK(k=91) post-encoder. Day 1
          verification will tell us.

State dict tensor names: candidates listed in KEY_ALIASES below. Common SAE conventions:
    W_enc:    (d_in, d_features)        encoder weight
    b_enc:    (d_features,)             encoder bias
    W_dec:    (d_features, d_in)        decoder weight
    b_dec:    (d_in,)                   decoder bias (optional, sometimes "pre_bias")

If Goodfire's checkpoint uses different key names, add them to KEY_ALIASES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


KEY_ALIASES = {
    # Goodfire's released SAE uses nn.Linear naming: encoder_linear.weight etc.
    # Other conventions: W_enc directly, or encoder.weight (deprecated).
    "W_enc": ["encoder_linear.weight", "W_enc", "encoder.weight", "encoder.W"],
    "b_enc": ["encoder_linear.bias", "b_enc", "encoder.bias", "encoder.b"],
    "W_dec": ["decoder_linear.weight", "W_dec", "decoder.weight", "decoder.W"],
    "b_dec": ["decoder_linear.bias", "b_dec", "decoder.bias", "decoder.b", "pre_bias"],
}


def _find_key(state_dict: dict, candidates: list[str]) -> Optional[str]:
    for k in candidates:
        if k in state_dict:
            return k
    return None


class SAE(nn.Module):
    """
    SAE matching Goodfire's nn.Linear formulation:
        features = ReLU(x @ W_enc + b_enc)
        x_hat    = features @ W_dec + b_dec

    No b_dec pre-subtraction from x (Goodfire's encoder is plain nn.Linear).
    Use TopK wrapper externally if the model is TopK-trained (Goodfire L0 ≈ 91).
    """

    def __init__(self, d_in: int, d_features: int, dtype=torch.bfloat16):
        super().__init__()
        self.d_in = d_in
        self.d_features = d_features
        # Canonical storage: W_enc (d_in, d_features), W_dec (d_features, d_in).
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_features, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_features, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_features, d_in, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) -> features: (..., d_features)"""
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., d_features) -> x_hat: (..., d_in)"""
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        x_hat = self.decode(features)
        return x_hat, features


def load_goodfire_sae(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> SAE:
    """
    Load Goodfire's open SAE checkpoint.

    Returns an SAE module on `device` with weights loaded from the .pth file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    print(f"[sae_loader] Loading {checkpoint_path.name}...")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # State may be a raw dict or wrapped in {"state_dict": {...}}.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Map aliases.
    mapped = {}
    for canonical, aliases in KEY_ALIASES.items():
        key = _find_key(state, aliases)
        if key is None:
            available = list(state.keys())[:10]
            raise KeyError(
                f"Could not find SAE key for '{canonical}' (tried {aliases}). "
                f"First 10 keys in checkpoint: {available}"
            )
        mapped[canonical] = state[key]

    # Infer dimensions.
    W_enc = mapped["W_enc"]
    if W_enc.ndim != 2:
        raise ValueError(f"W_enc has wrong shape: {W_enc.shape}")

    # SAE expansion: d_features is typically much larger than d_in (e.g., 65536 vs 4096).
    # nn.Linear stores weight as (out_features, in_features), so encoder_linear.weight
    # has shape (d_features, d_in). Canonical W_enc is (d_in, d_features) for x @ W_enc.
    if W_enc.shape[0] > W_enc.shape[1]:
        # Got (d_features, d_in) — transpose to canonical form.
        d_features, d_in = W_enc.shape
        W_enc = W_enc.T
    else:
        # Got (d_in, d_features) — already canonical.
        d_in, d_features = W_enc.shape

    print(f"[sae_loader] Inferred d_in={d_in}, d_features={d_features}")

    sae = SAE(d_in=d_in, d_features=d_features, dtype=dtype)

    # Decoder: nn.Linear(d_features, d_in) stores weight as (d_in, d_features).
    # Canonical W_dec is (d_features, d_in) for features @ W_dec.
    W_dec = mapped["W_dec"]
    if W_dec.shape != (d_features, d_in):
        W_dec = W_dec.T
    assert W_dec.shape == (d_features, d_in), f"W_dec shape mismatch: {W_dec.shape}"

    # Load with dtype conversion.
    sae.W_enc.data = W_enc.to(dtype)
    sae.b_enc.data = mapped["b_enc"].to(dtype).reshape(-1)
    sae.W_dec.data = W_dec.to(dtype)
    sae.b_dec.data = mapped["b_dec"].to(dtype).reshape(-1)

    sae.to(device)
    sae.eval()
    print(f"[sae_loader] SAE loaded on {device}")
    return sae
