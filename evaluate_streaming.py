from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from emotion2vec_backend import (
    DEFAULT_EMOTION2VEC_MODEL_ID,
    EMOTION2VEC_CANONICAL_LABELS,
    load_emotion2vec_model,
    predict_emotion2vec,
)
from ser_multitask import MultiTaskEmotionModel, is_lfs_pointer_file, load_first_valid_checkpoint
from ser_pipeline import (
    AudioConfig,
    FeatureConfig,
    discover_records,
    ensure_audio_length,
    extract_handcrafted_features,
    split_by_actor,
)


@dataclass
class LoadedModel:
    backend_name: str
    task_type: str
    model: object
    feature_extractor: AutoFeatureExtractor | None
    cfg: AudioConfig
    emotion_labels: list[str]
    intensity_labels: list[str]
    device: torch.device
    use_handcrafted_features: bool
    feature_cfg: FeatureConfig | None
    feature_mean: np.ndarray | None
    feature_std: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate streaming SER on test split using chunked inference.")
    parser.add_argument("--backend", choices=["artifacts", "emotion2vec"], default="artifacts")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--emotion2vec-model-id", type=str, default=DEFAULT_EMOTION2VEC_MODEL_ID)
    parser.add_argument("--data-dir", type=str, default="actors_speech")
    parser.add_argument("--chunk-seconds", type=float, default=0.50)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--min-buffer-seconds", type=float, default=0.8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


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


def _build_feature_config(metadata: dict) -> FeatureConfig | None:
    cfg = metadata.get("feature_config")
    if not isinstance(cfg, dict):
        return None
    allowed_keys = {
        "n_mfcc",
        "n_mels",
        "frame_length",
        "hop_length",
        "n_fft",
        "mel_fmin",
        "mel_fmax",
        "pitch_fmin",
        "pitch_fmax",
    }
    filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
    try:
        return FeatureConfig(**filtered)
    except Exception:
        return None


def _build_feature_stats(metadata: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    stats = metadata.get("feature_stats")
    if not isinstance(stats, dict):
        return None, None

    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        return None, None

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if mean_arr.ndim != 1 or std_arr.ndim != 1 or mean_arr.size != std_arr.size:
        return None, None
    std_arr = np.where(std_arr <= 0.0, 1.0, std_arr)
    return mean_arr, std_arr


def _load_emotion2vec_bundle(model_id: str, device: torch.device) -> tuple[LoadedModel, dict]:
    model = load_emotion2vec_model(model_id=model_id)
    loaded = LoadedModel(
        backend_name="emotion2vec",
        task_type="single_task",
        model=model,
        feature_extractor=None,
        cfg=AudioConfig(sample_rate=16_000, duration_seconds=4.0),
        emotion_labels=list(EMOTION2VEC_CANONICAL_LABELS),
        intensity_labels=[],
        device=device,
        use_handcrafted_features=False,
        feature_cfg=None,
        feature_mean=None,
        feature_std=None,
    )
    return loaded, {}


def load_model_bundle(backend: str, artifacts_dir: Path, device: torch.device, emotion2vec_model_id: str) -> tuple[LoadedModel, dict]:
    if backend == "emotion2vec":
        return _load_emotion2vec_bundle(emotion2vec_model_id, device)

    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    cfg_dict = metadata["audio_config"]
    cfg = AudioConfig(
        sample_rate=int(cfg_dict["sample_rate"]),
        duration_seconds=float(cfg_dict["duration_seconds"]),
    )

    task_type = str(metadata.get("task_type", "single_task"))
    hf_model_path = artifacts_dir / "hf_model"

    if task_type == "emotion_intensity_multitask":
        checkpoint_candidates = [artifacts_dir / "model_state.pt", artifacts_dir / "best_model.pt"]
        payload, state_path = load_first_valid_checkpoint(checkpoint_candidates, map_location=device)
        model_name_fallback = str(payload.get("model_name", metadata.get("model_name", ""))).strip()
        feature_extractor_source = str(hf_model_path)
        backbone_source = str(hf_model_path)
        local_only_fallback = False
        hf_config_path = hf_model_path / "config.json"
        hf_preproc_path = hf_model_path / "preprocessor_config.json"
        hf_weights_path = hf_model_path / "model.safetensors"
        hf_bundle_invalid = (
            not hf_config_path.exists()
            or not hf_preproc_path.exists()
            or is_lfs_pointer_file(hf_config_path)
            or is_lfs_pointer_file(hf_preproc_path)
            or is_lfs_pointer_file(hf_weights_path)
        )
        if hf_bundle_invalid:
            if not model_name_fallback:
                raise FileNotFoundError("No valid hf_model bundle found and no model_name fallback is available.")
            feature_extractor_source = model_name_fallback
            backbone_source = model_name_fallback
            local_only_fallback = True
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_source,
            local_files_only=local_only_fallback,
        )

        emotion_labels = list(metadata.get("emotion_labels", payload.get("emotion_labels", [])))
        intensity_labels = list(metadata.get("intensity_labels", payload.get("intensity_labels", ["normal", "strong"])))

        use_handcrafted_features = bool(
            payload.get("use_handcrafted_features", metadata.get("use_handcrafted_features", False))
        )
        aux_feature_dim = int(payload.get("aux_feature_dim", metadata.get("aux_feature_dim", 0)))
        aux_hidden_dim = int(payload.get("aux_hidden_dim", metadata.get("aux_hidden_dim", 128)))

        model = MultiTaskEmotionModel(
            backbone_name_or_path=backbone_source,
            num_emotions=len(emotion_labels),
            num_intensity=len(intensity_labels),
            head_dropout=float(payload.get("head_dropout", 0.2)),
            use_handcrafted_features=use_handcrafted_features,
            aux_feature_dim=aux_feature_dim,
            aux_hidden_dim=aux_hidden_dim,
            local_files_only=local_only_fallback,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()

        feature_cfg = _build_feature_config(metadata)
        feature_mean, feature_std = _build_feature_stats(metadata)

        loaded = LoadedModel(
            backend_name="custom",
            task_type=task_type,
            model=model,
            feature_extractor=feature_extractor,
            cfg=cfg,
            emotion_labels=emotion_labels,
            intensity_labels=intensity_labels,
            device=device,
            use_handcrafted_features=use_handcrafted_features,
            feature_cfg=feature_cfg,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        return loaded, metadata

    # Legacy single-task path.
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_path)
    model = AutoModelForAudioClassification.from_pretrained(hf_model_path)
    model.to(device)
    model.eval()

    loaded = LoadedModel(
        backend_name="custom",
        task_type="single_task",
        model=model,
        feature_extractor=feature_extractor,
        cfg=cfg,
        emotion_labels=list(metadata.get("target_labels", metadata.get("emotion_labels", []))),
        intensity_labels=[],
        device=device,
        use_handcrafted_features=False,
        feature_cfg=None,
        feature_mean=None,
        feature_std=None,
    )
    return loaded, metadata


def trim_to_latest_samples(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) >= target_samples:
        return y[-target_samples:].astype(np.float32)
    pad = target_samples - len(y)
    return np.pad(y, (pad, 0), mode="constant").astype(np.float32)

def prepare_aux_features(bundle: LoadedModel, waveform: np.ndarray) -> torch.Tensor | None:
    if not bundle.use_handcrafted_features:
        return None
    if bundle.feature_cfg is None:
        return None

    feats = extract_handcrafted_features(waveform, bundle.cfg.sample_rate, bundle.feature_cfg)
    if bundle.feature_mean is not None and bundle.feature_std is not None:
        feats = (feats - bundle.feature_mean) / bundle.feature_std
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(feats[None, :])


def predict_single(bundle: LoadedModel, waveform: np.ndarray) -> np.ndarray:
    if bundle.backend_name == "emotion2vec":
        return predict_emotion2vec(bundle.model, waveform, bundle.cfg.sample_rate)

    if bundle.feature_extractor is None:
        raise ValueError("feature_extractor is required for the current backend.")
    encoded = bundle.feature_extractor(
        waveform,
        sampling_rate=bundle.cfg.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = bundle.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()


def predict_multi(bundle: LoadedModel, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = bundle.feature_extractor(
        waveform,
        sampling_rate=bundle.cfg.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}

    aux = prepare_aux_features(bundle, waveform)
    if aux is not None:
        encoded["aux_features"] = aux.to(bundle.device)

    with torch.no_grad():
        emotion_logits, intensity_logits = bundle.model(**encoded)
        emotion_probs = torch.softmax(emotion_logits[0], dim=-1)
        intensity_probs = torch.softmax(intensity_logits[0], dim=-1)
    return emotion_probs.detach().cpu().numpy(), intensity_probs.detach().cpu().numpy()


def evaluate_streaming(args: argparse.Namespace) -> Dict:
    artifacts_dir = Path(args.artifacts_dir)
    device = resolve_device(args.device)
    bundle, metadata = load_model_bundle(
        backend=args.backend,
        artifacts_dir=artifacts_dir,
        device=device,
        emotion2vec_model_id=args.emotion2vec_model_id,
    )

    records = discover_records(args.data_dir)
    test_actor_ids = set(int(x) for x in metadata.get("test_actor_ids", []))
    if not test_actor_ids:
        _, _, test_records = split_by_actor(records, val_size=0.15, test_size=0.15, seed=42)
        test_actor_ids = {record.actor_id for record in test_records}

    if bundle.task_type == "emotion_intensity_multitask":
        allowed_emotions = set(bundle.emotion_labels)
        records = [r for r in records if r.actor_id in test_actor_ids and r.ravdess_emotion in allowed_emotions]
    else:
        records = [r for r in records if r.actor_id in test_actor_ids and r.ravdess_emotion in set(bundle.emotion_labels)]

    if args.max_records > 0:
        records = records[: args.max_records]

    if not records:
        raise ValueError("No records available for streaming evaluation.")

    emotion_to_idx = {label: i for i, label in enumerate(bundle.emotion_labels)}
    intensity_to_idx = {label: i for i, label in enumerate(bundle.intensity_labels)}

    chunk_samples = int(args.chunk_seconds * bundle.cfg.sample_rate)
    analysis_window_samples = int(max(args.window_seconds, bundle.cfg.duration_seconds) * bundle.cfg.sample_rate)
    min_buffer_samples = int(max(0.1, args.min_buffer_seconds) * bundle.cfg.sample_rate)

    final_emotion_true: List[int] = []
    final_emotion_pred: List[int] = []
    online_emotion_true: List[int] = []
    online_emotion_pred: List[int] = []

    final_intensity_true: List[int] = []
    final_intensity_pred: List[int] = []
    online_intensity_true: List[int] = []
    online_intensity_pred: List[int] = []

    inference_latencies_ms: List[float] = []
    stream_steps_per_clip: List[int] = []

    for record in records:
        y, _ = librosa.load(record.path, sr=bundle.cfg.sample_rate, mono=True)
        y = y.astype(np.float32)

        buffer = np.zeros(0, dtype=np.float32)
        last_emotion_pred = None
        last_intensity_pred = None
        steps = 0

        for start in range(0, len(y), max(1, chunk_samples)):
            chunk = y[start : start + chunk_samples]
            buffer = np.concatenate([buffer, chunk])
            if len(buffer) > (analysis_window_samples + bundle.cfg.target_num_samples):
                buffer = buffer[-(analysis_window_samples + bundle.cfg.target_num_samples) :]

            if len(buffer) < min_buffer_samples:
                continue

            analysis = buffer[-analysis_window_samples:]
            waveform = trim_to_latest_samples(analysis, bundle.cfg.target_num_samples)

            t0 = time.perf_counter()
            if bundle.task_type == "emotion_intensity_multitask":
                emotion_probs, intensity_probs = predict_multi(bundle, waveform)
                e_pred = int(np.argmax(emotion_probs))
                i_pred = int(np.argmax(intensity_probs))

                online_emotion_true.append(emotion_to_idx[record.ravdess_emotion])
                online_emotion_pred.append(e_pred)
                online_intensity_true.append(intensity_to_idx[record.intensity_label])
                online_intensity_pred.append(i_pred)

                last_emotion_pred = e_pred
                last_intensity_pred = i_pred
            else:
                emotion_probs = predict_single(bundle, waveform)
                e_pred = int(np.argmax(emotion_probs))
                online_emotion_true.append(emotion_to_idx[record.target_label])
                online_emotion_pred.append(e_pred)
                last_emotion_pred = e_pred

            inference_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            steps += 1

        if last_emotion_pred is None:
            full_wave = ensure_audio_length(y, bundle.cfg.target_num_samples, random_crop=False)
            if bundle.task_type == "emotion_intensity_multitask":
                emotion_probs, intensity_probs = predict_multi(bundle, full_wave)
                last_emotion_pred = int(np.argmax(emotion_probs))
                last_intensity_pred = int(np.argmax(intensity_probs))
            else:
                emotion_probs = predict_single(bundle, full_wave)
                last_emotion_pred = int(np.argmax(emotion_probs))

        stream_steps_per_clip.append(steps)

        if bundle.task_type == "emotion_intensity_multitask":
            final_emotion_true.append(emotion_to_idx[record.ravdess_emotion])
            final_emotion_pred.append(int(last_emotion_pred))
            final_intensity_true.append(intensity_to_idx[record.intensity_label])
            final_intensity_pred.append(int(last_intensity_pred))
        else:
            final_emotion_true.append(emotion_to_idx[record.target_label])
            final_emotion_pred.append(int(last_emotion_pred))

    result: Dict[str, object] = {
        "artifacts_dir": str(artifacts_dir.resolve()),
        "backend": args.backend,
        "task_type": bundle.task_type,
        "num_records": len(records),
        "chunk_seconds": float(args.chunk_seconds),
        "window_seconds": float(args.window_seconds),
        "min_buffer_seconds": float(args.min_buffer_seconds),
        "latency_ms_mean": float(np.mean(inference_latencies_ms)) if inference_latencies_ms else 0.0,
        "latency_ms_p95": float(np.percentile(inference_latencies_ms, 95)) if inference_latencies_ms else 0.0,
        "latency_ms_max": float(np.max(inference_latencies_ms)) if inference_latencies_ms else 0.0,
        "avg_stream_steps_per_clip": float(np.mean(stream_steps_per_clip)) if stream_steps_per_clip else 0.0,
    }

    emotion_label_ids = np.arange(len(bundle.emotion_labels))

    emotion_report_final = classification_report(
        final_emotion_true,
        final_emotion_pred,
        labels=emotion_label_ids,
        target_names=bundle.emotion_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    emotion_report_online = classification_report(
        online_emotion_true,
        online_emotion_pred,
        labels=emotion_label_ids,
        target_names=bundle.emotion_labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    result.update(
        {
            "emotion_final_accuracy": float(np.mean(np.array(final_emotion_true) == np.array(final_emotion_pred))),
            "emotion_online_accuracy": float(np.mean(np.array(online_emotion_true) == np.array(online_emotion_pred))),
            "emotion_final_macro_f1": float(
                f1_score(final_emotion_true, final_emotion_pred, average="macro", zero_division=0)
            ),
            "emotion_online_macro_f1": float(
                f1_score(online_emotion_true, online_emotion_pred, average="macro", zero_division=0)
            ),
            "emotion_final_report": emotion_report_final,
            "emotion_online_report": emotion_report_online,
            "emotion_final_confusion_matrix": confusion_matrix(
                final_emotion_true,
                final_emotion_pred,
                labels=emotion_label_ids,
            ).tolist(),
        }
    )

    if bundle.task_type == "emotion_intensity_multitask":
        intensity_label_ids = np.arange(len(bundle.intensity_labels))
        intensity_report_final = classification_report(
            final_intensity_true,
            final_intensity_pred,
            labels=intensity_label_ids,
            target_names=bundle.intensity_labels,
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        intensity_report_online = classification_report(
            online_intensity_true,
            online_intensity_pred,
            labels=intensity_label_ids,
            target_names=bundle.intensity_labels,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        result.update(
            {
                "intensity_final_accuracy": float(
                    np.mean(np.array(final_intensity_true) == np.array(final_intensity_pred))
                ),
                "intensity_online_accuracy": float(
                    np.mean(np.array(online_intensity_true) == np.array(online_intensity_pred))
                ),
                "intensity_final_macro_f1": float(
                    f1_score(final_intensity_true, final_intensity_pred, average="macro", zero_division=0)
                ),
                "intensity_online_macro_f1": float(
                    f1_score(online_intensity_true, online_intensity_pred, average="macro", zero_division=0)
                ),
                "intensity_final_report": intensity_report_final,
                "intensity_online_report": intensity_report_online,
                "intensity_final_confusion_matrix": confusion_matrix(
                    final_intensity_true,
                    final_intensity_pred,
                    labels=intensity_label_ids,
                ).tolist(),
            }
        )

    return result


def main() -> None:
    args = parse_args()
    result = evaluate_streaming(args)

    output_path = Path(args.output_json) if args.output_json else Path(args.artifacts_dir) / "streaming_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved streaming evaluation to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
