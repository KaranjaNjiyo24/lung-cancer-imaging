"""Multimodal 3D CNN architectures for NSCLC classification."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


try:  # pragma: no cover - optional torchvideo dependency
    from torchvision.models.video import r3d_18
except ImportError:  # pragma: no cover
    r3d_18 = None  # type: ignore


# ---------------------------------------------------------------------------
class AttentionFusion(nn.Module):
    """Simple attention-based fusion for modality embeddings."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.scale = feature_dim ** 0.5

    def forward(self, inputs: Dict[str, torch.Tensor], availability: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        weights = []
        for modality, tensor in inputs.items():
            mask = availability.get(modality)
            if mask is not None:
                mask = mask.float().view(tensor.size(0), 1)
            else:
                mask = torch.ones(tensor.size(0), 1, device=tensor.device, dtype=tensor.dtype)
            features.append(tensor * mask)
            weights.append(mask)

        stacked = torch.stack(features, dim=1)  # (B, M, D)
        queries = self.query(stacked)
        keys = self.key(stacked)
        values = self.value(stacked)

        scores = torch.einsum("bmd,bnd->bmn", queries, keys) / self.scale
        modal_mask = torch.stack(weights, dim=1)
        scores = scores.masked_fill(modal_mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        fused = torch.einsum("bmn,bnd->bmd", attn, values).sum(dim=1)
        return fused


# ---------------------------------------------------------------------------
def _make_resnet3d_encoder(input_channels: int, pretrained: bool = False) -> nn.Module:
    if r3d_18 is None:
        raise ImportError("torchvision>=0.13 required for r3d_18 3D backbone")

    model = r3d_18(pretrained=pretrained)
    model.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    return nn.Sequential(*list(model.children())[:-1])  # remove classifier


# ---------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, pretrained: bool) -> None:
        super().__init__()
        self.backbone = _make_resnet3d_encoder(input_channels, pretrained=pretrained)
        self.proj = nn.Linear(512, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.flatten(1)
        return self.proj(features)


# ---------------------------------------------------------------------------
class MultimodalLungCancerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        ct_channels: int = 1,
        pet_channels: int = 1,
        hidden_dim: int = 512,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.ct_encoder = ModalityEncoder(ct_channels, hidden_dim, pretrained)
        self.pet_encoder = ModalityEncoder(pet_channels, hidden_dim, pretrained)
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ct = batch.get("ct")
        pet = batch.get("pet")
        ct_available = batch.get("ct_available", torch.ones(ct.size(0), device=ct.device)) if ct is not None else None
        pet_available = batch.get("pet_available", torch.ones(pet.size(0), device=pet.device)) if pet is not None else None

        modality_features: Dict[str, torch.Tensor] = {}
        availability: Dict[str, torch.Tensor] = {}

        if ct is not None:
            modality_features["ct"] = self.ct_encoder(ct)
            availability["ct"] = ct_available
        if pet is not None:
            modality_features["pet"] = self.pet_encoder(pet)
            availability["pet"] = pet_available

        if not modality_features:
            raise ValueError("At least one modality input is required")

        fused = self.fusion(modality_features, availability)
        logits = self.classifier(fused)
        return logits


# ---------------------------------------------------------------------------
def build_multimodal_classifier(
    architecture: str,
    input_channels: Dict[str, int],
    num_classes: int,
    dropout: float,
    fusion: str,
) -> MultimodalLungCancerClassifier:
    if architecture != "resnet3d":
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = MultimodalLungCancerClassifier(
        num_classes=num_classes,
        ct_channels=input_channels.get("ct", 1),
        pet_channels=input_channels.get("pet", 1),
        hidden_dim=512,
        pretrained=False,
    )
    return model


__all__ = ["MultimodalLungCancerClassifier", "build_multimodal_classifier"]
