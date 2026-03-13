from __future__ import annotations

import importlib.util
from typing import Sequence

import numpy as np

DEFAULT_EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_seed"
EMOTION2VEC_CANONICAL_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

_RAW_TO_CANONICAL = {
    "angry": "angry",
    "disgusted": "disgust",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "other": "other",
    "sad": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
    "<unk>": "unknown",
    "unknown": "unknown",
}


def emotion2vec_available() -> bool:
    return importlib.util.find_spec("funasr") is not None


def canonicalize_emotion2vec_label(label: str) -> str:
    raw = str(label).strip().lower()
    if "/" in raw:
        parts = [part.strip() for part in raw.split("/") if part.strip()]
        # emotion2vec labels are often bilingual, e.g. "开心/happy".
        if parts:
            raw = parts[-1]
    return _RAW_TO_CANONICAL.get(raw, "unknown")


def load_emotion2vec_model(model_id: str = DEFAULT_EMOTION2VEC_MODEL_ID):
    if not emotion2vec_available():
        raise ImportError(
            "emotion2vec requires the optional `funasr` package. "
            "Install with: `pip install -U funasr modelscope`"
        )
    from funasr import AutoModel

    return AutoModel(model=model_id, hub="hf")


def _normalize_scores(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("emotion2vec scores must be a non-empty 1D sequence.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(arr < 0.0):
        exp = np.exp(arr - np.max(arr))
        denom = float(np.sum(exp)) + 1e-8
        return (exp / denom).astype(np.float32)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    return (arr / total).astype(np.float32)


def parse_emotion2vec_result(result) -> np.ndarray:
    payload = result
    if isinstance(payload, list):
        if not payload:
            raise ValueError("emotion2vec returned an empty result list.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported emotion2vec result type: {type(payload)!r}")

    probs = np.zeros(len(EMOTION2VEC_CANONICAL_LABELS), dtype=np.float32)
    label_to_idx = {label: idx for idx, label in enumerate(EMOTION2VEC_CANONICAL_LABELS)}

    labels = payload.get("labels")
    scores = payload.get("scores")
    if isinstance(labels, (list, tuple)) and isinstance(scores, (list, tuple)) and len(labels) == len(scores):
        norm_scores = _normalize_scores(scores)
        for raw_label, score in zip(labels, norm_scores.tolist()):
            canonical = canonicalize_emotion2vec_label(str(raw_label))
            probs[label_to_idx[canonical]] += float(score)
        total = float(np.sum(probs))
        if total > 0.0:
            probs /= total
        return probs.astype(np.float32)

    top_label = payload.get("text") or payload.get("label") or payload.get("labels")
    top_score = payload.get("score", 1.0)
    if isinstance(top_label, str):
        probs[label_to_idx[canonicalize_emotion2vec_label(top_label)]] = float(top_score)
        return _normalize_scores(probs)

    raise ValueError(f"Unsupported emotion2vec payload structure: {payload}")


def predict_emotion2vec(model, waveform: np.ndarray, sr: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    try:
        result = model.generate(
            input=waveform,
            sample_rate=sr,
            granularity="utterance",
            extract_embedding=False,
        )
    except Exception:
        # Some FunASR model wrappers only accept batched inputs.
        result = model.generate(
            input=[waveform],
            sample_rate=sr,
            granularity="utterance",
            extract_embedding=False,
        )
    return parse_emotion2vec_result(result)
