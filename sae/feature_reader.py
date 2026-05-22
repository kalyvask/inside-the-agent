"""
FeatureReader: extracts top-k SAE features from a captured activation.

Used inside the brain-server. The capture hook is installed on the base model's
layer 19; this class wraps the SAE encoder + top-k logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TopFeature:
    id: int
    activation: float
    label: str | None = None  # filled in by features.yaml lookup at HUD time


class FeatureReader:
    def __init__(self, sae, device: str = "cuda"):
        self.sae = sae
        self.device = device

    @torch.no_grad()
    def top_k_from_activation(
        self,
        activation: torch.Tensor,
        top_k: int = 20,
    ) -> list[TopFeature]:
        """
        activation: (d_in,) tensor — the LAST token's residual at the SAE layer.

        Returns top-k features with activation values, sorted descending.
        """
        if activation.ndim == 0:
            raise ValueError("Activation must be at least 1D")
        if activation.ndim > 1:
            # If they passed (1, d_in), squeeze.
            activation = activation.reshape(-1)

        features = self.sae.encode(activation.unsqueeze(0)).squeeze(0)
        values, indices = torch.topk(features, top_k)
        return [
            TopFeature(id=int(idx), activation=float(val))
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

    @torch.no_grad()
    def features_per_token(
        self,
        activations_per_token: torch.Tensor,
        top_k: int = 20,
    ) -> list[list[TopFeature]]:
        """
        activations_per_token: (seq_len, d_in)

        Returns per-token top-k features. Useful for trajectory-level analysis.
        """
        out = []
        for t in range(activations_per_token.shape[0]):
            out.append(self.top_k_from_activation(activations_per_token[t], top_k))
        return out
