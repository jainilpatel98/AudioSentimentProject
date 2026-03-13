from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, get_cosine_schedule_with_warmup

from ser_multitask import MultiTaskEmotionModel, load_checkpoint_state, set_feature_encoder_trainable
from ser_pipeline import (
    FULL_EMOTION_LABELS,
    INTENSITY_LABELS,
    AudioConfig,
    AugmentConfig,
    FeatureConfig,
    SampleRecord,
    augment_waveform,
    discover_records,
    extract_handcrafted_features,
    limit_records,
    load_waveform,
    mix_two_waveforms,
    save_training_metadata,
    set_global_seed,
    split_by_actor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a hybrid HuBERT model on full emotions + intensity (multi-task).",
    )
    parser.add_argument("--data-dir", type=str, default="actors_speech")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--model-name", type=str, default="superb/hubert-base-superb-er")

    parser.add_argument("--emotion-scheme", choices=["all8", "ekman7"], default="all8")

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)

    parser.add_argument("--augment-copies", type=int, default=3)
    parser.add_argument("--speaker-mix-prob", type=float, default=0.30)
    parser.add_argument("--speaker-mix-alpha-min", type=float, default=0.35)
    parser.add_argument("--speaker-mix-alpha-max", type=float, default=0.65)

    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)

    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--max-test-records", type=int, default=0)

    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--freeze-feature-encoder-epochs", type=int, default=2)

    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--emotion-loss",
        choices=["ce", "focal"],
        default="ce",
        help="Loss used for emotion head: cross entropy or focal loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal emotion loss (used when --emotion-loss=focal).",
    )
    parser.add_argument("--intensity-label-smoothing", type=float, default=0.02)
    parser.add_argument("--intensity-loss-weight", type=float, default=1.0)
    parser.add_argument("--intensity-metric-weight", type=float, default=0.5)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=0,
        help="If > 0, train only the last N transformer encoder layers (+feature projection) of the backbone.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the transformer backbone and train classification heads only.",
    )

    parser.add_argument("--use-handcrafted-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-hidden-dim", type=int, default=128)
    parser.add_argument("--feature-stats-max-records", type=int, default=0)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use local Hugging Face cache only (no network checks/downloads).",
    )
    parser.add_argument(
        "--resume-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume training from an existing output directory checkpoint when available.",
    )

    return parser.parse_args()


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_emotion_labels(emotion_scheme: str) -> List[str]:
    if emotion_scheme == "all8":
        return list(FULL_EMOTION_LABELS)
    return [label for label in FULL_EMOTION_LABELS if label != "calm"]


def filter_records_by_emotion(records: Sequence[SampleRecord], allowed_labels: Sequence[str]) -> List[SampleRecord]:
    allowed = set(allowed_labels)
    return [record for record in records if record.ravdess_emotion in allowed]


def validate_args(args: argparse.Namespace) -> None:
    if args.speaker_mix_alpha_min > args.speaker_mix_alpha_max:
        raise ValueError("--speaker-mix-alpha-min must be <= --speaker-mix-alpha-max")
    if args.focal_gamma < 0:
        raise ValueError("--focal-gamma must be >= 0.")
    if args.unfreeze_last_n_layers < 0:
        raise ValueError("--unfreeze-last-n-layers must be >= 0.")
    if args.freeze_backbone and args.unfreeze_last_n_layers > 0:
        raise ValueError("--unfreeze-last-n-layers cannot be used with --freeze-backbone.")


def compute_balanced_weights(num_classes: int, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    weights = np.ones(num_classes, dtype=np.float32)
    unique = np.unique(y)
    if unique.size == 0:
        return weights

    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique,
        y=y,
    )
    for cls, weight in zip(unique.tolist(), present_weights.tolist()):
        weights[int(cls)] = float(weight)
    return weights


class WeightedFocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self._ce = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self._ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce
        return loss.mean()


def _get_encoder_layers(backbone: nn.Module) -> Sequence[nn.Module]:
    candidate_paths = (
        ("encoder", "layers"),
        ("wav2vec2", "encoder", "layers"),
        ("hubert", "encoder", "layers"),
        ("wavlm", "encoder", "layers"),
        ("model", "encoder", "layers"),
    )
    for path in candidate_paths:
        ref = backbone
        ok = True
        for part in path:
            if not hasattr(ref, part):
                ok = False
                break
            ref = getattr(ref, part)
        if not ok:
            continue
        if isinstance(ref, (list, tuple, nn.ModuleList)):
            return list(ref)
    return []


def apply_partial_backbone_unfreeze(
    model: MultiTaskEmotionModel,
    unfreeze_last_n_layers: int,
    freeze_feature_encoder: bool,
) -> Dict[str, int]:
    backbone = model.backbone
    for param in backbone.parameters():
        param.requires_grad = False

    if hasattr(backbone, "feature_projection"):
        for param in backbone.feature_projection.parameters():
            param.requires_grad = True

    layers = _get_encoder_layers(backbone)
    total_layers = len(layers)

    if unfreeze_last_n_layers <= 0 or total_layers == 0:
        for param in backbone.parameters():
            param.requires_grad = True
        unfrozen_layers = total_layers
    else:
        n = min(unfreeze_last_n_layers, total_layers)
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        unfrozen_layers = n

    set_feature_encoder_trainable(model, trainable=(not freeze_feature_encoder))
    return {"total_layers": int(total_layers), "unfrozen_layers": int(unfrozen_layers)}


class EmotionIntensityDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        cfg: AudioConfig,
        feature_cfg: FeatureConfig,
        emotion_to_index: Dict[str, int],
        intensity_to_index: Dict[str, int],
        training: bool,
        augment_copies: int,
        augment_cfg: AugmentConfig,
        seed: int,
        use_handcrafted_features: bool,
        feature_mean: np.ndarray | None,
        feature_std: np.ndarray | None,
    ) -> None:
        self.records = list(records)
        self.cfg = cfg
        self.feature_cfg = feature_cfg
        self.emotion_to_index = emotion_to_index
        self.intensity_to_index = intensity_to_index
        self.training = training
        self.augment_copies = max(augment_copies, 0)
        self.augment_cfg = augment_cfg
        self.seed = seed
        self.use_handcrafted_features = use_handcrafted_features
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.sample_map: List[tuple[int, bool]] = []
        for idx in range(len(self.records)):
            self.sample_map.append((idx, False))
            if self.training and self.augment_copies > 0:
                for _ in range(self.augment_copies):
                    self.sample_map.append((idx, True))

        self.label_pair_to_indices: Dict[tuple[str, str], List[int]] = defaultdict(list)
        for idx, record in enumerate(self.records):
            key = (record.ravdess_emotion, record.intensity_label)
            self.label_pair_to_indices[key].append(idx)

    def __len__(self) -> int:
        return len(self.sample_map)

    def _sample_same_label_donor(self, record_idx: int, rng: np.random.Generator) -> SampleRecord | None:
        record = self.records[record_idx]
        key = (record.ravdess_emotion, record.intensity_label)
        candidates = self.label_pair_to_indices.get(key, [])
        if len(candidates) <= 1:
            return None

        cross_actor = [idx for idx in candidates if self.records[idx].actor_id != record.actor_id]
        pool = cross_actor if cross_actor else [idx for idx in candidates if idx != record_idx]
        if not pool:
            return None

        donor_idx = int(pool[int(rng.integers(0, len(pool)))])
        return self.records[donor_idx]

    def _apply_speaker_mix(
        self,
        waveform: np.ndarray,
        record_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        donor = self._sample_same_label_donor(record_idx, rng)
        if donor is None:
            return waveform

        donor_wave = load_waveform(donor.path, self.cfg, random_crop=True)
        if rng.random() < 0.50:
            donor_wave = augment_waveform(donor_wave, self.cfg, self.augment_cfg, rng)

        alpha = float(rng.uniform(self.augment_cfg.speaker_mix_alpha_min, self.augment_cfg.speaker_mix_alpha_max))
        return mix_two_waveforms(waveform, donor_wave, alpha)

    def _extract_aux_features(self, waveform: np.ndarray) -> np.ndarray | None:
        if not self.use_handcrafted_features:
            return None
        feats = extract_handcrafted_features(waveform, self.cfg.sample_rate, self.feature_cfg)
        if self.feature_mean is not None and self.feature_std is not None:
            feats = (feats - self.feature_mean) / self.feature_std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return feats

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record_idx, apply_augmentation = self.sample_map[idx]
        record = self.records[record_idx]

        random_crop = self.training and not apply_augmentation
        waveform = load_waveform(record.path, self.cfg, random_crop=random_crop)

        if apply_augmentation:
            rng = np.random.default_rng(self.seed + (idx * 7919))
            waveform = augment_waveform(waveform, self.cfg, self.augment_cfg, rng)
            if rng.random() < self.augment_cfg.speaker_mix_prob:
                waveform = self._apply_speaker_mix(waveform, record_idx, rng)

        sample = {
            "waveform": waveform,
            "emotion_label": int(self.emotion_to_index[record.ravdess_emotion]),
            "intensity_label": int(self.intensity_to_index[record.intensity_label]),
            "actor_id": int(record.actor_id),
        }

        aux_features = self._extract_aux_features(waveform)
        if aux_features is not None:
            sample["aux_features"] = aux_features

        return sample


class EmotionIntensityCollator:
    def __init__(self, feature_extractor: AutoFeatureExtractor, sample_rate: int) -> None:
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        waveforms = [item["waveform"] for item in batch]
        emotion_labels = torch.tensor([int(item["emotion_label"]) for item in batch], dtype=torch.long)
        intensity_labels = torch.tensor([int(item["intensity_label"]) for item in batch], dtype=torch.long)
        actor_ids = torch.tensor([int(item["actor_id"]) for item in batch], dtype=torch.long)

        encoded = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        encoded["emotion_labels"] = emotion_labels
        encoded["intensity_labels"] = intensity_labels
        encoded["actor_ids"] = actor_ids

        if "aux_features" in batch[0]:
            aux = np.stack([np.asarray(item["aux_features"], dtype=np.float32) for item in batch], axis=0)
            encoded["aux_features"] = torch.from_numpy(aux)

        return encoded


def compute_feature_stats(
    records: Sequence[SampleRecord],
    cfg: AudioConfig,
    feature_cfg: FeatureConfig,
    max_records: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected = limit_records(records, max_records=max_records, seed=seed) if max_records > 0 else list(records)
    features: List[np.ndarray] = []

    for record in tqdm(selected, desc="Feature stats", leave=False):
        waveform = load_waveform(record.path, cfg=cfg, random_crop=False)
        feats = extract_handcrafted_features(waveform, cfg.sample_rate, feature_cfg)
        features.append(feats)

    if not features:
        raise ValueError("No features extracted for feature statistics.")

    matrix = np.stack(features).astype(np.float32)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = (matrix.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std, int(matrix.shape[0])


def run_epoch(
    model: MultiTaskEmotionModel,
    loader: DataLoader,
    device: torch.device,
    emotion_criterion: nn.Module,
    intensity_criterion: nn.Module,
    intensity_loss_weight: float,
    optimizer: AdamW | None,
    scheduler,
    grad_clip: float,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    emotion_preds_all: List[int] = []
    emotion_true_all: List[int] = []
    intensity_preds_all: List[int] = []
    intensity_true_all: List[int] = []

    total_loss = 0.0
    total_emotion_loss = 0.0
    total_intensity_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if is_training else "eval"):
            emotion_labels = batch.pop("emotion_labels").to(device)
            intensity_labels = batch.pop("intensity_labels").to(device)
            batch.pop("actor_ids", None)

            model_inputs = {k: v.to(device) for k, v in batch.items()}

            emotion_logits, intensity_logits = model(**model_inputs)
            emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
            intensity_loss = intensity_criterion(intensity_logits, intensity_labels)
            loss = emotion_loss + (intensity_loss_weight * intensity_loss)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            batch_size = emotion_labels.size(0)
            total_examples += int(batch_size)
            total_loss += float(loss.item()) * batch_size
            total_emotion_loss += float(emotion_loss.item()) * batch_size
            total_intensity_loss += float(intensity_loss.item()) * batch_size

            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            intensity_preds = torch.argmax(intensity_logits, dim=-1)

            emotion_preds_all.extend(emotion_preds.detach().cpu().tolist())
            emotion_true_all.extend(emotion_labels.detach().cpu().tolist())
            intensity_preds_all.extend(intensity_preds.detach().cpu().tolist())
            intensity_true_all.extend(intensity_labels.detach().cpu().tolist())

    emotion_true_np = np.array(emotion_true_all)
    emotion_pred_np = np.array(emotion_preds_all)
    intensity_true_np = np.array(intensity_true_all)
    intensity_pred_np = np.array(intensity_preds_all)

    emotion_accuracy = float(np.mean(emotion_pred_np == emotion_true_np)) if len(emotion_true_np) else 0.0
    intensity_accuracy = float(np.mean(intensity_pred_np == intensity_true_np)) if len(intensity_true_np) else 0.0
    joint_accuracy = (
        float(np.mean((emotion_pred_np == emotion_true_np) & (intensity_pred_np == intensity_true_np)))
        if len(emotion_true_np)
        else 0.0
    )

    emotion_macro_f1 = (
        float(f1_score(emotion_true_np, emotion_pred_np, average="macro", zero_division=0))
        if len(emotion_true_np)
        else 0.0
    )
    intensity_macro_f1 = (
        float(f1_score(intensity_true_np, intensity_pred_np, average="macro", zero_division=0))
        if len(intensity_true_np)
        else 0.0
    )

    return {
        "loss": total_loss / max(total_examples, 1),
        "emotion_loss": total_emotion_loss / max(total_examples, 1),
        "intensity_loss": total_intensity_loss / max(total_examples, 1),
        "emotion_accuracy": emotion_accuracy,
        "intensity_accuracy": intensity_accuracy,
        "joint_accuracy": joint_accuracy,
        "emotion_macro_f1": emotion_macro_f1,
        "intensity_macro_f1": intensity_macro_f1,
    }


def collect_predictions(
    model: MultiTaskEmotionModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    emotion_logits_list: List[np.ndarray] = []
    intensity_logits_list: List[np.ndarray] = []
    emotion_true_list: List[np.ndarray] = []
    intensity_true_list: List[np.ndarray] = []
    actor_ids_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="predict"):
            emotion_labels = batch.pop("emotion_labels").to(device)
            intensity_labels = batch.pop("intensity_labels").to(device)
            actor_ids = batch.pop("actor_ids")
            model_inputs = {k: v.to(device) for k, v in batch.items()}

            emotion_logits, intensity_logits = model(**model_inputs)
            emotion_logits_list.append(emotion_logits.detach().cpu().numpy())
            intensity_logits_list.append(intensity_logits.detach().cpu().numpy())
            emotion_true_list.append(emotion_labels.detach().cpu().numpy())
            intensity_true_list.append(intensity_labels.detach().cpu().numpy())
            actor_ids_list.append(actor_ids.detach().cpu().numpy())

    emotion_logits_np = np.concatenate(emotion_logits_list, axis=0)
    intensity_logits_np = np.concatenate(intensity_logits_list, axis=0)
    emotion_true_np = np.concatenate(emotion_true_list, axis=0)
    intensity_true_np = np.concatenate(intensity_true_list, axis=0)
    actor_ids_np = np.concatenate(actor_ids_list, axis=0)

    emotion_probs_np = torch.softmax(torch.from_numpy(emotion_logits_np), dim=-1).numpy()
    intensity_probs_np = torch.softmax(torch.from_numpy(intensity_logits_np), dim=-1).numpy()

    emotion_pred_np = np.argmax(emotion_probs_np, axis=1)
    intensity_pred_np = np.argmax(intensity_probs_np, axis=1)

    return {
        "emotion_probs": emotion_probs_np,
        "intensity_probs": intensity_probs_np,
        "emotion_pred": emotion_pred_np,
        "intensity_pred": intensity_pred_np,
        "emotion_true": emotion_true_np,
        "intensity_true": intensity_true_np,
        "actor_ids": actor_ids_np,
    }


def compute_actor_metrics(
    actor_ids: np.ndarray,
    emotion_true: np.ndarray,
    emotion_pred: np.ndarray,
    intensity_true: np.ndarray,
    intensity_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for actor in sorted(np.unique(actor_ids).tolist()):
        mask = actor_ids == actor
        e_true = emotion_true[mask]
        e_pred = emotion_pred[mask]
        i_true = intensity_true[mask]
        i_pred = intensity_pred[mask]

        out[str(int(actor))] = {
            "num_samples": int(mask.sum()),
            "emotion_accuracy": float(np.mean(e_true == e_pred)) if mask.any() else 0.0,
            "intensity_accuracy": float(np.mean(i_true == i_pred)) if mask.any() else 0.0,
            "joint_accuracy": float(np.mean((e_true == e_pred) & (i_true == i_pred))) if mask.any() else 0.0,
        }
    return out


def checkpoint_payload(
    model: MultiTaskEmotionModel,
    model_name: str,
    emotion_labels: Sequence[str],
    intensity_labels: Sequence[str],
    head_dropout: float,
    use_handcrafted_features: bool,
    aux_feature_dim: int,
    aux_hidden_dim: int,
) -> Dict:
    return {
        "task_type": "emotion_intensity_multitask",
        "model_name": model_name,
        "head_dropout": float(head_dropout),
        "emotion_labels": list(emotion_labels),
        "intensity_labels": list(intensity_labels),
        "use_handcrafted_features": bool(use_handcrafted_features),
        "aux_feature_dim": int(aux_feature_dim),
        "aux_hidden_dim": int(aux_hidden_dim),
        "state_dict": model.state_dict(),
    }


def load_history_file(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return payload["history"]
    return []


def save_history_file(path: Path, history: List[Dict[str, float]]) -> None:
    save_json(path, {"history": history})


def save_resume_state(
    path: Path,
    base_payload: Dict,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    best_val_score: float,
    early_stop_counter: int,
    history: List[Dict[str, float]],
) -> None:
    payload = dict(base_payload)
    payload.update(
        {
            "epoch": int(epoch),
            "best_val_score": float(best_val_score),
            "early_stop_counter": int(early_stop_counter),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }
    )
    torch.save(payload, path)


def maybe_resume_training(
    output_dir: Path,
    args: argparse.Namespace,
    model: MultiTaskEmotionModel,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> Tuple[int, List[Dict[str, float]], float, int]:
    if not args.resume_if_exists:
        return 1, [], -1.0, 0

    resume_path = output_dir / "resume_state.pt"
    model_state_path = output_dir / "model_state.pt"
    history_path = output_dir / "history.json"

    if resume_path.exists():
        payload = load_checkpoint_state(str(resume_path), map_location=device)
        model.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = payload.get("history")
        if not isinstance(history, list):
            history = load_history_file(history_path)
        completed_epoch = int(payload.get("epoch", len(history)))
        best_val_score = float(payload.get("best_val_score", -1.0))
        early_stop_counter = int(payload.get("early_stop_counter", 0))
        print(
            "Resuming training from resume_state.pt | "
            f"completed_epoch={completed_epoch} next_epoch={completed_epoch + 1}"
        )
        return completed_epoch + 1, history, best_val_score, early_stop_counter

    if model_state_path.exists() and history_path.exists():
        history = load_history_file(history_path)
        if history:
            payload = load_checkpoint_state(str(model_state_path), map_location=device)
            model.load_state_dict(payload["state_dict"])
            completed_epoch = int(max(int(row.get("epoch", 0)) for row in history))
            best_val_score = float(max(row.get("val_composite_score", -1.0) for row in history))
            print(
                "Resuming from model_state.pt + history.json | "
                f"completed_epoch={completed_epoch} next_epoch={completed_epoch + 1} "
                "(optimizer/scheduler restarted)"
            )
            return completed_epoch + 1, history, best_val_score, 0

    return 1, [], -1.0, 0


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cfg = AudioConfig(
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
    )
    feature_cfg = FeatureConfig()
    augment_cfg = AugmentConfig(
        speaker_mix_prob=args.speaker_mix_prob,
        speaker_mix_alpha_min=args.speaker_mix_alpha_min,
        speaker_mix_alpha_max=args.speaker_mix_alpha_max,
    )

    print("Discovering records...")
    all_records = discover_records(args.data_dir)

    emotion_labels = select_emotion_labels(args.emotion_scheme)
    all_records = filter_records_by_emotion(all_records, emotion_labels)
    print(f"Found {len(all_records)} valid wav files after emotion filter: {args.emotion_scheme}")

    train_records, val_records, test_records = split_by_actor(
        all_records,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(
        "Split sizes (pre-limit) | "
        f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
    )

    train_records = limit_records(train_records, args.max_train_records, seed=args.seed)
    val_records = limit_records(val_records, args.max_val_records, seed=args.seed + 1)
    test_records = limit_records(test_records, args.max_test_records, seed=args.seed + 2)
    print(
        "Split sizes (final)    | "
        f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
    )

    emotion_to_index = {label: idx for idx, label in enumerate(emotion_labels)}
    intensity_labels = list(INTENSITY_LABELS)
    intensity_to_index = {label: idx for idx, label in enumerate(intensity_labels)}

    feature_mean = None
    feature_std = None
    feature_stats_payload = None
    aux_feature_dim = 0

    if args.use_handcrafted_features:
        feature_mean, feature_std, num_for_stats = compute_feature_stats(
            train_records,
            cfg=cfg,
            feature_cfg=feature_cfg,
            max_records=args.feature_stats_max_records,
            seed=args.seed,
        )
        aux_feature_dim = int(feature_mean.shape[0])
        feature_stats_payload = {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
            "num_records": int(num_for_stats),
            "dim": int(aux_feature_dim),
        }
        print(
            "Handcrafted features enabled | "
            f"dim={aux_feature_dim} stats_records={num_for_stats}"
        )
    else:
        print("Handcrafted features disabled.")

    save_training_metadata(
        output_dir=output_dir,
        cfg=cfg,
        emotion_to_index=emotion_to_index,
        intensity_to_index=intensity_to_index,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        model_name=args.model_name,
        augment_cfg=augment_cfg,
        feature_cfg=feature_cfg,
        feature_stats=feature_stats_payload,
        extra_metadata={
            "emotion_scheme": args.emotion_scheme,
            "use_handcrafted_features": bool(args.use_handcrafted_features),
            "emotion_loss": str(args.emotion_loss),
            "focal_gamma": float(args.focal_gamma),
            "unfreeze_last_n_layers": int(args.unfreeze_last_n_layers),
            "freeze_backbone": bool(args.freeze_backbone),
            "aux_feature_dim": int(aux_feature_dim),
            "aux_hidden_dim": int(args.aux_hidden_dim),
        },
    )

    print(f"Loading feature extractor + backbone: {args.model_name}")
    used_local_fallback = False
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_name,
            local_files_only=args.offline,
        )
    except Exception:
        if args.offline:
            raise
        print("Feature extractor remote load failed; retrying with local cache only.")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.model_name,
            local_files_only=True,
        )
        used_local_fallback = True
    if int(feature_extractor.sampling_rate) != cfg.sample_rate:
        raise ValueError(
            f"Feature extractor expects {feature_extractor.sampling_rate} Hz; "
            f"requested sample rate was {cfg.sample_rate}."
        )

    model = MultiTaskEmotionModel(
        backbone_name_or_path=args.model_name,
        num_emotions=len(emotion_labels),
        num_intensity=len(intensity_labels),
        head_dropout=args.head_dropout,
        use_handcrafted_features=args.use_handcrafted_features,
        aux_feature_dim=aux_feature_dim,
        aux_hidden_dim=args.aux_hidden_dim,
        local_files_only=(args.offline or used_local_fallback),
    )
    model.to(device)

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Keep the feature encoder frozen too for full backbone freeze mode.
        set_feature_encoder_trainable(model, trainable=False)
        model.backbone.eval()
        print("Backbone frozen: training classification heads only.")
    else:
        layer_info = apply_partial_backbone_unfreeze(
            model=model,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
            freeze_feature_encoder=(args.freeze_feature_encoder_epochs > 0),
        )
        if args.unfreeze_last_n_layers > 0:
            print(
                "Partial backbone unfreezing enabled | "
                f"unfrozen_last_layers={layer_info['unfrozen_layers']}/{layer_info['total_layers']}"
            )

    train_dataset = EmotionIntensityDataset(
        train_records,
        cfg=cfg,
        feature_cfg=feature_cfg,
        emotion_to_index=emotion_to_index,
        intensity_to_index=intensity_to_index,
        training=True,
        augment_copies=args.augment_copies,
        augment_cfg=augment_cfg,
        seed=args.seed,
        use_handcrafted_features=args.use_handcrafted_features,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    val_dataset = EmotionIntensityDataset(
        val_records,
        cfg=cfg,
        feature_cfg=feature_cfg,
        emotion_to_index=emotion_to_index,
        intensity_to_index=intensity_to_index,
        training=False,
        augment_copies=0,
        augment_cfg=augment_cfg,
        seed=args.seed + 10,
        use_handcrafted_features=args.use_handcrafted_features,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_dataset = EmotionIntensityDataset(
        test_records,
        cfg=cfg,
        feature_cfg=feature_cfg,
        emotion_to_index=emotion_to_index,
        intensity_to_index=intensity_to_index,
        training=False,
        augment_copies=0,
        augment_cfg=augment_cfg,
        seed=args.seed + 20,
        use_handcrafted_features=args.use_handcrafted_features,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    collator = EmotionIntensityCollator(feature_extractor=feature_extractor, sample_rate=cfg.sample_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        "Sample sizes | "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    emotion_train_labels = np.array([emotion_to_index[r.ravdess_emotion] for r in train_records], dtype=np.int64)
    intensity_train_labels = np.array([intensity_to_index[r.intensity_label] for r in train_records], dtype=np.int64)

    emotion_weights_arr = compute_balanced_weights(len(emotion_labels), emotion_train_labels)
    intensity_weights_arr = compute_balanced_weights(len(intensity_labels), intensity_train_labels)
    print(f"Emotion class weights: {emotion_weights_arr.tolist()}")
    print(f"Intensity class weights: {intensity_weights_arr.tolist()}")

    emotion_weights = torch.tensor(emotion_weights_arr, dtype=torch.float32, device=device)
    intensity_weights = torch.tensor(intensity_weights_arr, dtype=torch.float32, device=device)

    if args.emotion_loss == "focal":
        emotion_criterion = WeightedFocalLoss(
            class_weights=emotion_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    else:
        emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=args.label_smoothing)
    intensity_criterion = nn.CrossEntropyLoss(
        weight=intensity_weights,
        label_smoothing=args.intensity_label_smoothing,
    )

    if args.freeze_backbone:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Keep all params in optimizer so newly-unfrozen backbone params can train later in the run.
        trainable_params = list(model.parameters())
    if not trainable_params:
        raise ValueError("No trainable parameters remain. Disable --freeze-backbone or enable trainable heads.")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history_path = output_dir / "history.json"
    resume_checkpoint = output_dir / "resume_state.pt"
    history: List[Dict[str, float]] = []
    best_val_score = -1.0
    best_checkpoint = output_dir / "best_model.pt"
    early_stop_counter = 0
    start_epoch = 1

    start_epoch, history, best_val_score, early_stop_counter = maybe_resume_training(
        output_dir=output_dir,
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone:
            set_feature_encoder_trainable(model, trainable=False)
            model.backbone.eval()
        else:
            freeze_feature_encoder = epoch <= args.freeze_feature_encoder_epochs
            apply_partial_backbone_unfreeze(
                model=model,
                unfreeze_last_n_layers=args.unfreeze_last_n_layers,
                freeze_feature_encoder=freeze_feature_encoder,
            )

        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            emotion_criterion=emotion_criterion,
            intensity_criterion=intensity_criterion,
            intensity_loss_weight=args.intensity_loss_weight,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            emotion_criterion=emotion_criterion,
            intensity_criterion=intensity_criterion,
            intensity_loss_weight=args.intensity_loss_weight,
            optimizer=None,
            scheduler=None,
            grad_clip=args.grad_clip,
        )

        val_score = val_metrics["emotion_macro_f1"] + (
            args.intensity_metric_weight * val_metrics["intensity_macro_f1"]
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_emotion_loss": float(train_metrics["emotion_loss"]),
            "train_intensity_loss": float(train_metrics["intensity_loss"]),
            "train_emotion_accuracy": float(train_metrics["emotion_accuracy"]),
            "train_intensity_accuracy": float(train_metrics["intensity_accuracy"]),
            "train_joint_accuracy": float(train_metrics["joint_accuracy"]),
            "train_emotion_macro_f1": float(train_metrics["emotion_macro_f1"]),
            "train_intensity_macro_f1": float(train_metrics["intensity_macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_emotion_loss": float(val_metrics["emotion_loss"]),
            "val_intensity_loss": float(val_metrics["intensity_loss"]),
            "val_emotion_accuracy": float(val_metrics["emotion_accuracy"]),
            "val_intensity_accuracy": float(val_metrics["intensity_accuracy"]),
            "val_joint_accuracy": float(val_metrics["joint_accuracy"]),
            "val_emotion_macro_f1": float(val_metrics["emotion_macro_f1"]),
            "val_intensity_macro_f1": float(val_metrics["intensity_macro_f1"]),
            "val_composite_score": float(val_score),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_score > (best_val_score + args.min_delta):
            best_val_score = float(val_score)
            early_stop_counter = 0
            torch.save(
                checkpoint_payload(
                    model=model,
                    model_name=args.model_name,
                    emotion_labels=emotion_labels,
                    intensity_labels=intensity_labels,
                    head_dropout=args.head_dropout,
                    use_handcrafted_features=args.use_handcrafted_features,
                    aux_feature_dim=aux_feature_dim,
                    aux_hidden_dim=args.aux_hidden_dim,
                ),
                best_checkpoint,
            )
        else:
            early_stop_counter += 1

        save_history_file(history_path, history)
        save_resume_state(
            path=resume_checkpoint,
            base_payload=checkpoint_payload(
                model=model,
                model_name=args.model_name,
                emotion_labels=emotion_labels,
                intensity_labels=intensity_labels,
                head_dropout=args.head_dropout,
                use_handcrafted_features=args.use_handcrafted_features,
                aux_feature_dim=aux_feature_dim,
                aux_hidden_dim=args.aux_hidden_dim,
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_score=best_val_score,
            early_stop_counter=early_stop_counter,
            history=history,
        )

        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_checkpoint.exists():
        best_payload = load_checkpoint_state(str(best_checkpoint), map_location=device)
        model.load_state_dict(best_payload["state_dict"])
    else:
        torch.save(
            checkpoint_payload(
                model=model,
                model_name=args.model_name,
                emotion_labels=emotion_labels,
                intensity_labels=intensity_labels,
                head_dropout=args.head_dropout,
                use_handcrafted_features=args.use_handcrafted_features,
                aux_feature_dim=aux_feature_dim,
                aux_hidden_dim=args.aux_hidden_dim,
            ),
            best_checkpoint,
        )

    pred = collect_predictions(model, test_loader, device)

    emotion_true = pred["emotion_true"]
    emotion_pred = pred["emotion_pred"]
    intensity_true = pred["intensity_true"]
    intensity_pred = pred["intensity_pred"]

    emotion_label_ids = np.arange(len(emotion_labels))
    intensity_label_ids = np.arange(len(intensity_labels))

    emotion_cm = confusion_matrix(emotion_true, emotion_pred, labels=emotion_label_ids)
    intensity_cm = confusion_matrix(intensity_true, intensity_pred, labels=intensity_label_ids)

    emotion_report = classification_report(
        emotion_true,
        emotion_pred,
        labels=emotion_label_ids,
        target_names=emotion_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    intensity_report = classification_report(
        intensity_true,
        intensity_pred,
        labels=intensity_label_ids,
        target_names=intensity_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    joint_labels = [f"{emotion}|{intensity}" for emotion in emotion_labels for intensity in intensity_labels]
    num_intensity = len(intensity_labels)
    joint_true = (emotion_true * num_intensity) + intensity_true
    joint_pred = (emotion_pred * num_intensity) + intensity_pred
    joint_cm = confusion_matrix(joint_true, joint_pred, labels=np.arange(len(joint_labels)))
    joint_report = classification_report(
        joint_true,
        joint_pred,
        labels=np.arange(len(joint_labels)),
        target_names=joint_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    emotion_accuracy = float(np.mean(emotion_true == emotion_pred))
    intensity_accuracy = float(np.mean(intensity_true == intensity_pred))
    joint_accuracy = float(np.mean((emotion_true == emotion_pred) & (intensity_true == intensity_pred)))

    emotion_balanced_acc = float(balanced_accuracy_score(emotion_true, emotion_pred))
    intensity_balanced_acc = float(balanced_accuracy_score(intensity_true, intensity_pred))
    joint_macro_f1 = float(f1_score(joint_true, joint_pred, average="macro", zero_division=0))

    per_actor = compute_actor_metrics(
        actor_ids=pred["actor_ids"],
        emotion_true=emotion_true,
        emotion_pred=emotion_pred,
        intensity_true=intensity_true,
        intensity_pred=intensity_pred,
    )

    metrics = {
        "emotion_accuracy": emotion_accuracy,
        "emotion_balanced_accuracy": emotion_balanced_acc,
        "intensity_accuracy": intensity_accuracy,
        "intensity_balanced_accuracy": intensity_balanced_acc,
        "joint_accuracy": joint_accuracy,
        "joint_macro_f1": joint_macro_f1,
        "emotion_macro_f1": float(emotion_report.get("macro avg", {}).get("f1-score", 0.0)),
        "emotion_weighted_f1": float(emotion_report.get("weighted avg", {}).get("f1-score", 0.0)),
        "intensity_macro_f1": float(intensity_report.get("macro avg", {}).get("f1-score", 0.0)),
        "intensity_weighted_f1": float(intensity_report.get("weighted avg", {}).get("f1-score", 0.0)),
        "best_val_composite_score": best_val_score,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
        "model_name": args.model_name,
        "task_type": "emotion_intensity_multitask",
        "emotion_scheme": args.emotion_scheme,
        "emotion_loss": args.emotion_loss,
        "focal_gamma": float(args.focal_gamma),
        "unfreeze_last_n_layers": int(args.unfreeze_last_n_layers),
        "use_handcrafted_features": bool(args.use_handcrafted_features),
        "freeze_backbone": bool(args.freeze_backbone),
        "aux_feature_dim": int(aux_feature_dim),
        "device": str(device),
        "training_args": vars(args),
        "audio_config": asdict(cfg),
        "augment_config": asdict(augment_cfg),
        "feature_config": asdict(feature_cfg),
        "per_actor": per_actor,
    }

    hf_dir = output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(hf_dir)
    feature_extractor.save_pretrained(hf_dir)

    final_payload = checkpoint_payload(
        model=model,
        model_name=args.model_name,
        emotion_labels=emotion_labels,
        intensity_labels=intensity_labels,
        head_dropout=args.head_dropout,
        use_handcrafted_features=args.use_handcrafted_features,
        aux_feature_dim=aux_feature_dim,
        aux_hidden_dim=args.aux_hidden_dim,
    )
    torch.save(final_payload, output_dir / "model_state.pt")

    np.save(output_dir / "test_emotion_probabilities.npy", pred["emotion_probs"])
    np.save(output_dir / "test_intensity_probabilities.npy", pred["intensity_probs"])

    np.save(output_dir / "emotion_confusion_matrix.npy", emotion_cm)
    np.savetxt(output_dir / "emotion_confusion_matrix.csv", emotion_cm, fmt="%d", delimiter=",")
    np.save(output_dir / "intensity_confusion_matrix.npy", intensity_cm)
    np.savetxt(output_dir / "intensity_confusion_matrix.csv", intensity_cm, fmt="%d", delimiter=",")
    np.save(output_dir / "joint_confusion_matrix.npy", joint_cm)
    np.savetxt(output_dir / "joint_confusion_matrix.csv", joint_cm, fmt="%d", delimiter=",")

    save_json(output_dir / "emotion_classification_report.json", emotion_report)
    save_json(output_dir / "intensity_classification_report.json", intensity_report)
    save_json(output_dir / "joint_classification_report.json", joint_report)
    save_history_file(history_path, history)
    save_json(output_dir / "metrics.json", metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model and artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
