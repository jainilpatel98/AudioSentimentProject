from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


RAVDESS_MODALITY_BY_CODE: Dict[int, str] = {
    1: "full_av",
    2: "video_only",
    3: "audio_only",
}

RAVDESS_CHANNEL_BY_CODE: Dict[int, str] = {
    1: "speech",
    2: "song",
}

RAVDESS_EMOTION_BY_CODE: Dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

RAVDESS_INTENSITY_BY_CODE: Dict[int, str] = {
    1: "normal",
    2: "strong",
}

RAVDESS_STATEMENT_BY_CODE: Dict[int, str] = {
    1: "kids_are_talking_by_the_door",
    2: "dogs_are_sitting_by_the_door",
}

RAVDESS_REPETITION_BY_CODE: Dict[int, str] = {
    1: "first_repetition",
    2: "second_repetition",
}

FULL_EMOTION_LABELS: Tuple[str, ...] = tuple(RAVDESS_EMOTION_BY_CODE[idx] for idx in sorted(RAVDESS_EMOTION_BY_CODE))
INTENSITY_LABELS: Tuple[str, ...] = tuple(RAVDESS_INTENSITY_BY_CODE[idx] for idx in sorted(RAVDESS_INTENSITY_BY_CODE))
TARGET_LABELS: Tuple[str, ...] = ("happy", "sad", "other")  # legacy mapping (kept for compatibility)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration_seconds: float = 4.0

    @property
    def target_num_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)


@dataclass(frozen=True)
class FeatureConfig:
    n_mfcc: int = 13
    n_mels: int = 64
    frame_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    mel_fmin: float = 20.0
    mel_fmax: float = 8_000.0
    pitch_fmin: float = 50.0
    pitch_fmax: float = 500.0


@dataclass(frozen=True)
class AugmentConfig:
    noise_prob: float = 0.80
    noise_scale: float = 0.006
    shift_prob: float = 0.70
    shift_max_fraction: float = 0.20
    pitch_prob: float = 0.45
    pitch_max_steps: float = 2.5
    stretch_prob: float = 0.35
    stretch_min_rate: float = 0.85
    stretch_max_rate: float = 1.15
    gain_prob: float = 0.35
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0
    speaker_mix_prob: float = 0.30
    speaker_mix_alpha_min: float = 0.35
    speaker_mix_alpha_max: float = 0.65


@dataclass(frozen=True)
class SampleRecord:
    path: str
    actor_id: int
    modality_code: int
    modality_label: str
    channel_code: int
    channel_label: str
    emotion_code: int
    ravdess_emotion: str
    intensity_code: int
    intensity_label: str
    statement_code: int
    statement_label: str
    repetition_code: int
    repetition_label: str
    target_label: str


def map_to_target_label(ravdess_emotion: str) -> str:
    if ravdess_emotion == "happy":
        return "happy"
    if ravdess_emotion == "sad":
        return "sad"
    return "other"


def parse_ravdess_file(path: Path) -> SampleRecord | None:
    # RAVDESS naming convention: MM-VC-EE-II-SS-RR-AA
    # MM=modality, VC=vocal channel, EE=emotion, II=intensity,
    # SS=statement, RR=repetition, AA=actor.
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None

    try:
        modality_code = int(parts[0])
        channel_code = int(parts[1])
        emotion_code = int(parts[2])
        intensity_code = int(parts[3])
        statement_code = int(parts[4])
        repetition_code = int(parts[5])
        actor_id = int(parts[6])
    except ValueError:
        return None

    modality_label = RAVDESS_MODALITY_BY_CODE.get(modality_code)
    channel_label = RAVDESS_CHANNEL_BY_CODE.get(channel_code)
    ravdess_emotion = RAVDESS_EMOTION_BY_CODE.get(emotion_code)
    intensity_label = RAVDESS_INTENSITY_BY_CODE.get(intensity_code)
    statement_label = RAVDESS_STATEMENT_BY_CODE.get(statement_code)
    repetition_label = RAVDESS_REPETITION_BY_CODE.get(repetition_code)

    if (
        modality_label is None
        or channel_label is None
        or ravdess_emotion is None
        or intensity_label is None
        or statement_label is None
        or repetition_label is None
    ):
        return None

    target_label = map_to_target_label(ravdess_emotion)
    return SampleRecord(
        path=str(path),
        actor_id=actor_id,
        modality_code=modality_code,
        modality_label=modality_label,
        channel_code=channel_code,
        channel_label=channel_label,
        emotion_code=emotion_code,
        ravdess_emotion=ravdess_emotion,
        intensity_code=intensity_code,
        intensity_label=intensity_label,
        statement_code=statement_code,
        statement_label=statement_label,
        repetition_code=repetition_code,
        repetition_label=repetition_label,
        target_label=target_label,
    )


def discover_records(data_dir: str | Path) -> List[SampleRecord]:
    root = Path(data_dir)
    wav_paths = sorted(root.glob("Actor_*/*.wav"))
    records: List[SampleRecord] = []

    for wav_path in wav_paths:
        record = parse_ravdess_file(wav_path)
        if record is not None:
            records.append(record)

    if not records:
        raise ValueError(f"No valid RAVDESS wav files found in: {root}")
    return records


def split_by_actor(
    records: Sequence[SampleRecord],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 0.8:
        raise ValueError("Use val/test sizes in a reasonable range, e.g. 0.15 each.")

    all_indices = np.arange(len(records))
    groups = np.array([r.actor_id for r in records])

    first_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(first_splitter.split(all_indices, groups=groups))

    val_relative_size = val_size / (1.0 - test_size)
    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_relative_size,
        random_state=seed + 1,
    )
    train_idx_local, val_idx_local = next(
        second_splitter.split(train_val_idx, groups=groups[train_val_idx])
    )

    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]
    return train_records, val_records, test_records


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_audio_length(
    y: np.ndarray,
    target_num_samples: int,
    rng: np.random.Generator | None = None,
    random_crop: bool = False,
) -> np.ndarray:
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if len(y) > target_num_samples:
        if random_crop and rng is not None:
            max_start = len(y) - target_num_samples
            start = int(rng.integers(0, max_start + 1))
        else:
            start = (len(y) - target_num_samples) // 2
        y = y[start : start + target_num_samples]
    elif len(y) < target_num_samples:
        pad_total = target_num_samples - len(y)
        left = pad_total // 2
        right = pad_total - left
        y = np.pad(y, (left, right), mode="constant")

    return y.astype(np.float32)


def add_noise(y: np.ndarray, rng: np.random.Generator, noise_scale: float = 0.006) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-8)
    return y + (noise_scale * peak * noise)


def time_shift(y: np.ndarray, rng: np.random.Generator, max_fraction: float = 0.2) -> np.ndarray:
    max_shift = int(len(y) * max_fraction)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(y, shift)


def random_gain(
    y: np.ndarray,
    rng: np.random.Generator,
    min_db: float = -6.0,
    max_db: float = 6.0,
) -> np.ndarray:
    gain_db = float(rng.uniform(min_db, max_db))
    gain = float(10.0 ** (gain_db / 20.0))
    return y * gain


def pitch_shift(y: np.ndarray, sr: int, rng: np.random.Generator, max_steps: float = 2.5) -> np.ndarray:
    n_steps = float(rng.uniform(-max_steps, max_steps))
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(
    y: np.ndarray,
    rng: np.random.Generator,
    min_rate: float = 0.85,
    max_rate: float = 1.15,
) -> np.ndarray:
    rate = float(rng.uniform(min_rate, max_rate))
    return librosa.effects.time_stretch(y=y, rate=rate)


def augment_waveform(
    y: np.ndarray,
    cfg: AudioConfig,
    augment_cfg: AugmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = y.copy()

    if rng.random() < augment_cfg.noise_prob:
        out = add_noise(out, rng, noise_scale=augment_cfg.noise_scale)
    if rng.random() < augment_cfg.shift_prob:
        out = time_shift(out, rng, max_fraction=augment_cfg.shift_max_fraction)
    if rng.random() < augment_cfg.gain_prob:
        out = random_gain(
            out,
            rng,
            min_db=augment_cfg.gain_db_min,
            max_db=augment_cfg.gain_db_max,
        )
    if rng.random() < augment_cfg.pitch_prob:
        out = pitch_shift(out, cfg.sample_rate, rng, max_steps=augment_cfg.pitch_max_steps)
    if rng.random() < augment_cfg.stretch_prob:
        out = time_stretch(
            out,
            rng,
            min_rate=augment_cfg.stretch_min_rate,
            max_rate=augment_cfg.stretch_max_rate,
        )

    out = ensure_audio_length(out, cfg.target_num_samples, rng=rng, random_crop=True)
    return np.clip(out, -1.0, 1.0)


def mix_two_waveforms(
    wave_a: np.ndarray,
    wave_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mixed = (alpha * wave_a) + ((1.0 - alpha) * wave_b)
    peak = float(np.max(np.abs(mixed)) + 1e-8)
    if peak > 1.0:
        mixed = mixed / peak
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def load_waveform(path: str, cfg: AudioConfig, random_crop: bool = False) -> np.ndarray:
    y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    rng = np.random.default_rng() if random_crop else None
    return ensure_audio_length(y, cfg.target_num_samples, rng=rng, random_crop=random_crop)


def _safe_stats(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
        ],
        dtype=np.float32,
    )


def _finite_or_empty(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    return x[np.isfinite(x)]


def extract_handcrafted_features(
    y: np.ndarray,
    sr: int,
    feature_cfg: FeatureConfig,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=feature_cfg.frame_length,
        hop_length=feature_cfg.hop_length,
    )[0]
    rms = librosa.feature.rms(
        y=y,
        frame_length=feature_cfg.frame_length,
        hop_length=feature_cfg.hop_length,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=y,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )[0]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=feature_cfg.n_mfcc,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )
    mfcc_delta = librosa.feature.delta(mfcc)

    mel_fmax = float(min(feature_cfg.mel_fmax, (sr / 2.0) - 1e-3))
    mel_fmax = max(mel_fmax, feature_cfg.mel_fmin + 1.0)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        win_length=feature_cfg.frame_length,
        n_mels=feature_cfg.n_mels,
        fmin=feature_cfg.mel_fmin,
        fmax=mel_fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel_delta = librosa.feature.delta(log_mel)
    log_mel_delta2 = librosa.feature.delta(log_mel, order=2)

    # Chroma gives a rough harmonic fingerprint that can help disambiguate affective tone.
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
    )

    try:
        f0 = librosa.yin(
            y,
            fmin=feature_cfg.pitch_fmin,
            fmax=feature_cfg.pitch_fmax,
            sr=sr,
            frame_length=max(2048, feature_cfg.frame_length),
            hop_length=feature_cfg.hop_length,
        )
    except Exception:
        f0 = np.array([], dtype=np.float32)

    f0 = _finite_or_empty(np.asarray(f0, dtype=np.float32))
    voiced_ratio = float(f0.size) / float(max(1, int(np.ceil(len(y) / feature_cfg.hop_length))))

    feature_parts: List[np.ndarray] = [
        _safe_stats(np.asarray(zcr, dtype=np.float32)),
        _safe_stats(np.asarray(rms, dtype=np.float32)),
        _safe_stats(np.asarray(centroid, dtype=np.float32)),
        _safe_stats(np.asarray(bandwidth, dtype=np.float32)),
        _safe_stats(np.asarray(rolloff, dtype=np.float32)),
        _safe_stats(np.asarray(flatness, dtype=np.float32)),
    ]

    feature_parts.extend(
        [
            np.mean(mfcc, axis=1).astype(np.float32),
            np.std(mfcc, axis=1).astype(np.float32),
            np.mean(mfcc_delta, axis=1).astype(np.float32),
            np.std(mfcc_delta, axis=1).astype(np.float32),
            np.mean(log_mel, axis=1).astype(np.float32),
            np.std(log_mel, axis=1).astype(np.float32),
            np.mean(log_mel_delta, axis=1).astype(np.float32),
            np.std(log_mel_delta, axis=1).astype(np.float32),
            np.mean(log_mel_delta2, axis=1).astype(np.float32),
            np.std(log_mel_delta2, axis=1).astype(np.float32),
            np.mean(chroma, axis=1).astype(np.float32),
            np.std(chroma, axis=1).astype(np.float32),
            _safe_stats(f0),
            np.array([voiced_ratio], dtype=np.float32),
        ]
    )

    out = np.concatenate(feature_parts, axis=0).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_legacy_label_index() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(TARGET_LABELS)}


def build_emotion_label_index() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(FULL_EMOTION_LABELS)}


def build_intensity_label_index() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(INTENSITY_LABELS)}


def limit_records(records: Sequence[SampleRecord], max_records: int, seed: int) -> List[SampleRecord]:
    if max_records <= 0 or len(records) <= max_records:
        return list(records)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    chosen = sorted(idx[:max_records].tolist())
    return [records[i] for i in chosen]


def split_summary(records: Sequence[SampleRecord], field: str) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        key = str(getattr(record, field))
        summary[key] = summary.get(key, 0) + 1
    return summary


def actor_summary(records: Sequence[SampleRecord]) -> List[int]:
    return sorted({r.actor_id for r in records})


def save_training_metadata(
    output_dir: str | Path,
    cfg: AudioConfig,
    emotion_to_index: Dict[str, int],
    intensity_to_index: Dict[str, int],
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    model_name: str,
    augment_cfg: AugmentConfig,
    feature_cfg: FeatureConfig | None = None,
    feature_stats: Dict[str, List[float]] | None = None,
    extra_metadata: Dict | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifacts_format": "hf_audio_multitask_v3",
        "task_type": "emotion_intensity_multitask",
        "audio_config": asdict(cfg),
        "augment_config": asdict(augment_cfg),
        "feature_config": asdict(feature_cfg) if feature_cfg is not None else None,
        "feature_stats": feature_stats,
        "emotion_labels": list(emotion_to_index.keys()),
        "emotion_to_index": emotion_to_index,
        "intensity_labels": list(intensity_to_index.keys()),
        "intensity_to_index": intensity_to_index,
        "base_model_name": model_name,
        "train_actor_ids": actor_summary(train_records),
        "val_actor_ids": actor_summary(val_records),
        "test_actor_ids": actor_summary(test_records),
        "train_emotion_distribution": split_summary(train_records, field="ravdess_emotion"),
        "val_emotion_distribution": split_summary(val_records, field="ravdess_emotion"),
        "test_emotion_distribution": split_summary(test_records, field="ravdess_emotion"),
        "train_intensity_distribution": split_summary(train_records, field="intensity_label"),
        "val_intensity_distribution": split_summary(val_records, field="intensity_label"),
        "test_intensity_distribution": split_summary(test_records, field="intensity_label"),
        "ravdess_naming_convention": {
            "format": "MM-VC-EE-II-SS-RR-AA",
            "modality_map": RAVDESS_MODALITY_BY_CODE,
            "vocal_channel_map": RAVDESS_CHANNEL_BY_CODE,
            "emotion_map": RAVDESS_EMOTION_BY_CODE,
            "intensity_map": RAVDESS_INTENSITY_BY_CODE,
            "statement_map": RAVDESS_STATEMENT_BY_CODE,
            "repetition_map": RAVDESS_REPETITION_BY_CODE,
        },
    }
    if extra_metadata:
        payload.update(extra_metadata)

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metadata_path
