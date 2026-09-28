"""Reusable cross-modal feature fusion (CMFF) module.

The module expects one EEG feature tensor and two competing speech feature
tensors, each shaped ``[batch_size, feature_dim]``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CMFF(nn.Module):
    """Fuse EEG features with two competing speech-stream features."""

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout1: float = 0.2,
        dropout2: float = 0.1,
        cosine_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cosine_eps = cosine_eps

        self.eeg_transform = nn.Linear(feature_dim, hidden_dim)
        self.audio1_transform = nn.Linear(feature_dim, hidden_dim)
        self.audio2_transform = nn.Linear(feature_dim, hidden_dim)

        # EEG-driven cognitive modulation of the two speech streams.
        self.gate_eeg_to_audio1 = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.Sigmoid()
        )
        self.gate_eeg_to_audio2 = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.Sigmoid()
        )

        # Complementary competition between the modulated speech streams.
        self.gate_audio_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )

        self.fusion_layer1 = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout1),
        )
        self.fusion_layer2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _validate_inputs(self, eeg: Tensor, audio1: Tensor, audio2: Tensor) -> None:
        tensors = {"eeg": eeg, "audio1": audio1, "audio2": audio2}
        for name, tensor in tensors.items():
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [batch_size, feature_dim], "
                    f"but received {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != self.feature_dim:
                raise ValueError(
                    f"{name} feature dimension must be {self.feature_dim}, "
                    f"but received {tensor.shape[1]}"
                )
        if not (eeg.shape[0] == audio1.shape[0] == audio2.shape[0]):
            raise ValueError("eeg, audio1, and audio2 must share the same batch size")

    def forward(self, eeg: Tensor, audio1: Tensor, audio2: Tensor) -> Tensor:
        self._validate_inputs(eeg, audio1, audio2)

        eeg_feat = self.eeg_transform(eeg)
        audio1_feat = self.audio1_transform(audio1)
        audio2_feat = self.audio2_transform(audio2)

        sim1 = F.cosine_similarity(
            eeg, audio1, dim=1, eps=self.cosine_eps
        ).unsqueeze(1)
        sim2 = F.cosine_similarity(
            eeg, audio2, dim=1, eps=self.cosine_eps
        ).unsqueeze(1)

        gate1 = self.gate_eeg_to_audio1(torch.cat([eeg, audio1], dim=1))
        gate2 = self.gate_eeg_to_audio2(torch.cat([eeg, audio2], dim=1))
        gated_audio1 = audio1_feat * gate1
        gated_audio2 = audio2_feat * gate2

        competition_gate = self.gate_audio_interaction(
            torch.cat([gated_audio1, gated_audio2], dim=1)
        )
        refined_audio1 = gated_audio1 * competition_gate
        refined_audio2 = gated_audio2 * (1.0 - competition_gate)

        combined = torch.cat(
            [eeg_feat, refined_audio1, refined_audio2, sim1, sim2], dim=1
        )
        fused = self.fusion_layer1(combined)
        fused = self.fusion_layer2(fused)
        return self.output_layer(fused)


FusionNet = CMFF


if __name__ == "__main__":
    model = CMFF(feature_dim=128, hidden_dim=64, num_classes=2)
    batch = 4
    logits = model(
        torch.randn(batch, 128),
        torch.randn(batch, 128),
        torch.randn(batch, 128),
    )
    print(f"Output shape: {tuple(logits.shape)}")
