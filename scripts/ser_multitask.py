from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModel


def _infer_hidden_size(config) -> int:
    for attr in ("hidden_size", "classifier_proj_size", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer hidden size from backbone config.")


class MultiTaskEmotionModel(nn.Module):
    def __init__(
        self,
        backbone_name_or_path: str,
        num_emotions: int,
        num_intensity: int,
        head_dropout: float = 0.2,
        use_handcrafted_features: bool = False,
        aux_feature_dim: int = 0,
        aux_hidden_dim: int = 128,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name_or_path,
            local_files_only=local_files_only,
        )
        hidden_size = _infer_hidden_size(self.backbone.config)

        self.use_handcrafted_features = bool(use_handcrafted_features and aux_feature_dim > 0)
        self.aux_feature_dim = int(aux_feature_dim)
        self.aux_hidden_dim = int(aux_hidden_dim)

        if self.use_handcrafted_features:
            self.aux_norm = nn.LayerNorm(self.aux_feature_dim)
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_feature_dim, self.aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
            )
            fused_dim = hidden_size + self.aux_hidden_dim
        else:
            self.aux_norm = None
            self.aux_mlp = None
            fused_dim = hidden_size

        self.dropout = nn.Dropout(head_dropout)
        self.emotion_head = nn.Linear(fused_dim, num_emotions)
        self.intensity_head = nn.Linear(fused_dim, num_intensity)

    @staticmethod
    def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        numerator = (last_hidden_state * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp(min=1.0)
        return numerator / denominator

    def _prepare_aux_features(self, pooled: torch.Tensor, aux_features: torch.Tensor | None) -> torch.Tensor:
        if not self.use_handcrafted_features:
            return pooled

        if aux_features is None:
            aux_features = torch.zeros(
                pooled.size(0),
                self.aux_feature_dim,
                device=pooled.device,
                dtype=pooled.dtype,
            )
        aux_features = aux_features.to(device=pooled.device, dtype=pooled.dtype)
        aux = self.aux_mlp(self.aux_norm(aux_features))
        return torch.cat([pooled, aux], dim=-1)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        aux_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        feature_mask = None
        if attention_mask is not None:
            if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
                feature_mask = self.backbone._get_feature_vector_attention_mask(
                    outputs.last_hidden_state.shape[1],
                    attention_mask,
                )
            else:
                feature_mask = attention_mask

        pooled = self.masked_mean_pool(outputs.last_hidden_state, feature_mask)
        fused = self._prepare_aux_features(pooled, aux_features)
        fused = self.dropout(fused)
        emotion_logits = self.emotion_head(fused)
        intensity_logits = self.intensity_head(fused)
        return emotion_logits, intensity_logits


def set_feature_encoder_trainable(model: MultiTaskEmotionModel, trainable: bool) -> None:
    backbone = model.backbone

    if not trainable and hasattr(backbone, "freeze_feature_encoder"):
        backbone.freeze_feature_encoder()
        return
    if trainable and hasattr(backbone, "unfreeze_feature_encoder"):
        backbone.unfreeze_feature_encoder()
        return

    prefixes = (
        "wav2vec2.feature_extractor",
        "hubert.feature_extractor",
        "wavlm.feature_extractor",
        "feature_extractor",
    )
    for name, param in backbone.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = trainable


def load_checkpoint_state(checkpoint_path: str, map_location: torch.device) -> Dict:
    # Our checkpoints store full Python dict payloads, not just raw tensor weights.
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict):
        return payload
    return {"state_dict": payload}


def load_first_valid_checkpoint(checkpoint_paths: Sequence[Path], map_location: torch.device) -> Tuple[Dict, Path]:
    errors: list[str] = []
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            continue
        try:
            return load_checkpoint_state(str(checkpoint_path), map_location=map_location), checkpoint_path
        except Exception as exc:
            errors.append(f"{checkpoint_path.name}: {exc}")
    joined = " | ".join(errors) if errors else "no checkpoint files found"
    raise RuntimeError(f"Could not load any valid checkpoint from artifacts directory: {joined}")


def is_lfs_pointer_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    with path.open("rb") as f:
        prefix = f.read(64)
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
