from __future__ import annotations

import io
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from emotion2vec_backend import (
    DEFAULT_EMOTION2VEC_MODEL_ID,
    EMOTION2VEC_CANONICAL_LABELS,
    emotion2vec_available,
    load_emotion2vec_model,
    predict_emotion2vec,
)
from ser_multitask import MultiTaskEmotionModel, is_lfs_pointer_file, load_first_valid_checkpoint
from ser_pipeline import AudioConfig, FeatureConfig, ensure_audio_length, extract_handcrafted_features

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False


st.set_page_config(page_title="Speech Emotion Detector", page_icon="🎙️", layout="centered")

DEFAULT_AUDIO_RECEIVER_SIZE = 256
DEFAULT_LIVE_UPDATE_SECONDS = 0.9
DEFAULT_LIVE_SMOOTHING = 0.25
DEFAULT_LIVE_VAD_RMS = 0.008
DEFAULT_SPECTROGRAM_REFRESH_SECONDS = 1.5


@dataclass
class InferenceBundle:
    backend_name: str
    task_type: str  # "single_task" or "emotion_intensity_multitask"
    model: object
    feature_extractor: AutoFeatureExtractor | None
    cfg: AudioConfig
    emotion_labels: list[str]
    intensity_labels: list[str]
    device: torch.device
    use_handcrafted_features: bool = False
    feature_cfg: FeatureConfig | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None


def _artifacts_ready(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "metadata.json").exists() and (artifacts_dir / "hf_model").exists()


def default_artifacts_dir() -> str:
    candidates = ["artifacts", "artifacts_multitask_smoke", "artifacts_hubert_smoke", "artifacts_smoke"]
    for candidate in candidates:
        if _artifacts_ready(Path(candidate)):
            return candidate
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "artifacts"


def resolve_device() -> torch.device:
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


def _load_multitask_bundle(base: Path, metadata: dict, device: torch.device) -> InferenceBundle:
    cfg_dict = metadata["audio_config"]
    cfg = AudioConfig(
        sample_rate=int(cfg_dict["sample_rate"]),
        duration_seconds=float(cfg_dict["duration_seconds"]),
    )

    emotion_labels = list(metadata["emotion_labels"])
    intensity_labels = list(metadata.get("intensity_labels", ["normal", "strong"]))

    checkpoint_candidates = [base / "model_state.pt", base / "best_model.pt"]
    if not any(path.exists() for path in checkpoint_candidates):
        raise FileNotFoundError(
            "Multitask artifacts require `model_state.pt` (or `best_model.pt`) in artifacts directory."
        )

    payload, state_path = load_first_valid_checkpoint(checkpoint_candidates, map_location=device)
    hf_model_path = base / "hf_model"
    backbone_source = str(hf_model_path)
    feature_extractor_source = str(hf_model_path)
    local_only_fallback = False
    model_name_fallback = str(payload.get("model_name", metadata.get("model_name", ""))).strip()
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
    if hf_bundle_invalid and model_name_fallback:
        backbone_source = model_name_fallback
        feature_extractor_source = model_name_fallback
        local_only_fallback = True

    head_dropout = float(payload.get("head_dropout", metadata.get("head_dropout", 0.2)))
    use_handcrafted_features = bool(
        payload.get("use_handcrafted_features", metadata.get("use_handcrafted_features", False))
    )
    aux_feature_dim = int(payload.get("aux_feature_dim", metadata.get("aux_feature_dim", 0)))
    aux_hidden_dim = int(payload.get("aux_hidden_dim", metadata.get("aux_hidden_dim", 128)))

    feature_cfg = _build_feature_config(metadata)
    feature_mean, feature_std = _build_feature_stats(metadata)

    model = MultiTaskEmotionModel(
        backbone_name_or_path=backbone_source,
        num_emotions=len(emotion_labels),
        num_intensity=len(intensity_labels),
        head_dropout=head_dropout,
        use_handcrafted_features=use_handcrafted_features,
        aux_feature_dim=aux_feature_dim,
        aux_hidden_dim=aux_hidden_dim,
        local_files_only=local_only_fallback,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        feature_extractor_source,
        local_files_only=local_only_fallback,
    )

    return InferenceBundle(
        backend_name="custom",
        task_type="emotion_intensity_multitask",
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


def _load_single_task_bundle(base: Path, metadata: dict, device: torch.device) -> InferenceBundle:
    cfg_dict = metadata["audio_config"]
    cfg = AudioConfig(
        sample_rate=int(cfg_dict["sample_rate"]),
        duration_seconds=float(cfg_dict["duration_seconds"]),
    )

    emotion_labels = list(metadata.get("target_labels", metadata.get("emotion_labels", [])))
    hf_model_path = base / "hf_model"
    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_path)
    model = AutoModelForAudioClassification.from_pretrained(hf_model_path)
    model.eval()
    model.to(device)

    return InferenceBundle(
        backend_name="custom",
        task_type="single_task",
        model=model,
        feature_extractor=feature_extractor,
        cfg=cfg,
        emotion_labels=emotion_labels,
        intensity_labels=[],
        device=device,
    )


def _load_emotion2vec_bundle(model_id: str, device: torch.device) -> InferenceBundle:
    model = load_emotion2vec_model(model_id=model_id)
    return InferenceBundle(
        backend_name="emotion2vec",
        task_type="single_task",
        model=model,
        feature_extractor=None,
        cfg=AudioConfig(sample_rate=16_000, duration_seconds=4.0),
        emotion_labels=list(EMOTION2VEC_CANONICAL_LABELS),
        intensity_labels=[],
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle(backend_name: str, artifacts_dir: str, emotion2vec_model_id: str) -> InferenceBundle:
    device = resolve_device()
    if backend_name == "emotion2vec":
        return _load_emotion2vec_bundle(emotion2vec_model_id, device)

    base = Path(artifacts_dir)
    metadata_path = base / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    task_type = str(metadata.get("task_type", "")).strip()

    if task_type == "emotion_intensity_multitask":
        return _load_multitask_bundle(base, metadata, device)

    # Backward compatibility for earlier single-task artifacts.
    return _load_single_task_bundle(base, metadata, device)


def read_audio_from_uploaded(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Tuple[np.ndarray, int]:
    raw = uploaded_file.getvalue()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), int(sr)


def trim_to_latest_samples(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) >= target_samples:
        return y[-target_samples:].astype(np.float32)
    pad = target_samples - len(y)
    return np.pad(y, (pad, 0), mode="constant").astype(np.float32)

def preprocess_waveform(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    if sr != cfg.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate)
    y = ensure_audio_length(y, cfg.target_num_samples, random_crop=False)
    return y.astype(np.float32)


def preprocess_live_waveform(y: np.ndarray, sr: int, cfg: AudioConfig) -> np.ndarray:
    if sr != cfg.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate)
    y = trim_to_latest_samples(y, cfg.target_num_samples)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _prepare_aux_features(bundle: InferenceBundle, waveform: np.ndarray) -> torch.Tensor | None:
    if not bundle.use_handcrafted_features:
        return None
    if bundle.feature_cfg is None:
        return None

    feats = extract_handcrafted_features(waveform, bundle.cfg.sample_rate, bundle.feature_cfg)
    if bundle.feature_mean is not None and bundle.feature_std is not None:
        feats = (feats - bundle.feature_mean) / bundle.feature_std
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(feats[None, :])


def predict_single_task(bundle: InferenceBundle, waveform: np.ndarray) -> np.ndarray:
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
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = bundle.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()


def predict_multitask(bundle: InferenceBundle, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = bundle.feature_extractor(
        waveform,
        sampling_rate=bundle.cfg.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}

    aux = _prepare_aux_features(bundle, waveform)
    if aux is not None:
        encoded["aux_features"] = aux.to(bundle.device)

    with torch.no_grad():
        emotion_logits, intensity_logits = bundle.model(**encoded)
        emotion_probs = torch.softmax(emotion_logits[0], dim=-1)
        intensity_probs = torch.softmax(intensity_logits[0], dim=-1)
    return emotion_probs.detach().cpu().numpy(), intensity_probs.detach().cpu().numpy()


def normalize_pcm(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_mag = float(max(abs(info.min), abs(info.max)))
        arr = arr.astype(np.float32) / max_mag
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def frame_to_mono_float32(frame) -> Tuple[np.ndarray, int]:
    raw = frame.to_ndarray()

    if raw.ndim == 2:
        if raw.shape[0] <= raw.shape[1]:
            raw = np.mean(raw, axis=0)
        else:
            raw = np.mean(raw, axis=1)

    mono = normalize_pcm(np.asarray(raw).reshape(-1))
    return mono.astype(np.float32), int(frame.sample_rate)


def state_key(prefix: str, key: str) -> str:
    return f"{prefix}_{key}"


def reset_live_state(buffer_key: str, emotion_key: str, intensity_key: str, silence_key: str) -> None:
    st.session_state[buffer_key] = np.zeros(0, dtype=np.float32)
    st.session_state[emotion_key] = None
    st.session_state[intensity_key] = None
    st.session_state[silence_key] = None


def render_single_task_outputs(probs: np.ndarray, labels: list[str], heading_prefix: str = "") -> None:
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    confidence = float(probs[top_idx])

    title = f"{heading_prefix}Prediction" if heading_prefix else "Prediction"
    st.success(f"{title}: **{top_label.upper()}** ({confidence:.1%} confidence)")

    score_rows = [{"label": label, "probability": float(prob)} for label, prob in zip(labels, probs)]
    st.dataframe(score_rows, width="stretch", hide_index=True)
    st.bar_chart({label: float(prob) for label, prob in zip(labels, probs)})


def render_multitask_outputs(
    emotion_probs: np.ndarray,
    intensity_probs: np.ndarray,
    emotion_labels: list[str],
    intensity_labels: list[str],
    heading_prefix: str = "",
) -> None:
    emotion_idx = int(np.argmax(emotion_probs))
    intensity_idx = int(np.argmax(intensity_probs))

    emotion_label = emotion_labels[emotion_idx]
    intensity_label = intensity_labels[intensity_idx]
    emotion_conf = float(emotion_probs[emotion_idx])
    intensity_conf = float(intensity_probs[intensity_idx])

    title = f"{heading_prefix}Prediction" if heading_prefix else "Prediction"
    st.success(
        f"{title}: **{emotion_label.upper()}** | intensity: **{intensity_label.upper()}** "
        f"(emotion {emotion_conf:.1%}, intensity {intensity_conf:.1%})"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emotion probabilities**")
        emotion_rows = [
            {"emotion": label, "probability": float(prob)} for label, prob in zip(emotion_labels, emotion_probs)
        ]
        st.dataframe(emotion_rows, width="stretch", hide_index=True)
        st.bar_chart({label: float(prob) for label, prob in zip(emotion_labels, emotion_probs)})
    with col2:
        st.markdown("**Intensity probabilities**")
        intensity_rows = [
            {"intensity": label, "probability": float(prob)}
            for label, prob in zip(intensity_labels, intensity_probs)
        ]
        st.dataframe(intensity_rows, width="stretch", hide_index=True)
        st.bar_chart({label: float(prob) for label, prob in zip(intensity_labels, intensity_probs)})


def render_live_spectrogram(waveform: np.ndarray, sr: int) -> None:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    duration_seconds = len(waveform) / float(sr) if sr > 0 else 0.0
    ax.set_title(f"Live Buffer Mel Spectrogram ({duration_seconds:.2f}s)")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_live_mode(bundle: InferenceBundle, artifacts_dir: Path) -> None:
    st.markdown("### Live microphone mode")
    st.caption("Start the stream and talk. The app continuously classifies a rolling audio window.")

    if not HAS_WEBRTC:
        st.warning(
            "Real-time mode requires `streamlit-webrtc`. Install with: "
            "`pip install streamlit-webrtc`"
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        window_seconds = st.slider(
            "Window (s)",
            min_value=1.5,
            max_value=8.0,
            value=float(max(2.0, bundle.cfg.duration_seconds)),
            step=0.5,
        )
    with col2:
        update_interval = st.slider(
            "Update (s)",
            min_value=0.25,
            max_value=2.0,
            value=DEFAULT_LIVE_UPDATE_SECONDS,
            step=0.05,
        )
    with col3:
        vad_rms = st.slider(
            "Voice threshold",
            min_value=0.0,
            max_value=0.05,
            value=DEFAULT_LIVE_VAD_RMS,
            step=0.001,
        )

    smoothing = st.slider(
        "Prediction smoothing",
        min_value=0.0,
        max_value=0.95,
        value=DEFAULT_LIVE_SMOOTHING,
        step=0.05,
        help="Higher values reduce jitter but respond slower.",
    )
    show_spectrogram = st.checkbox("Show live spectrogram", value=True)
    show_probability_details = st.checkbox("Show live probability details", value=False)

    stream_key = f"live_{bundle.backend_name}_{str(artifacts_dir).replace('/', '_')}"
    buf_key = state_key(stream_key, "buffer")
    emo_ema_key = state_key(stream_key, "emotion_ema")
    int_ema_key = state_key(stream_key, "intensity_ema")
    silence_key = state_key(stream_key, "silence_since")
    spec_key = state_key(stream_key, "last_spectrogram_time")

    if buf_key not in st.session_state:
        st.session_state[buf_key] = np.zeros(0, dtype=np.float32)
    if emo_ema_key not in st.session_state:
        st.session_state[emo_ema_key] = None
    if int_ema_key not in st.session_state:
        st.session_state[int_ema_key] = None
    if silence_key not in st.session_state:
        st.session_state[silence_key] = None
    if spec_key not in st.session_state:
        st.session_state[spec_key] = 0.0

    ctx = webrtc_streamer(
        key=stream_key,
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_receiver_size=DEFAULT_AUDIO_RECEIVER_SIZE,
    )

    status_placeholder = st.empty()
    results_placeholder = st.empty()
    spectrogram_placeholder = st.empty()

    if not ctx.state.playing:
        reset_live_state(buf_key, emo_ema_key, int_ema_key, silence_key)
        status_placeholder.info("Click Start and begin speaking.")
        return

    if ctx.audio_receiver is None:
        status_placeholder.warning("Waiting for microphone stream...")
        return

    status_placeholder.info("Listening...")
    max_buffer_samples = int((max(window_seconds, bundle.cfg.duration_seconds) + 2.0) * bundle.cfg.sample_rate)
    min_samples_for_inference = int(max(0.6, min(window_seconds, bundle.cfg.duration_seconds)) * bundle.cfg.sample_rate)
    last_predict_time = 0.0

    while ctx.state.playing:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            continue

        if len(frames) > 24:
            frames = frames[-24:]

        for frame in frames:
            chunk, frame_sr = frame_to_mono_float32(frame)
            if chunk.size == 0:
                continue

            if frame_sr != bundle.cfg.sample_rate:
                chunk = librosa.resample(chunk, orig_sr=frame_sr, target_sr=bundle.cfg.sample_rate)

            combined = np.concatenate([st.session_state[buf_key], chunk.astype(np.float32)])
            st.session_state[buf_key] = combined[-max_buffer_samples:]

        now = time.time()
        if (now - last_predict_time) < update_interval:
            continue
        last_predict_time = now

        live_buffer = st.session_state[buf_key]
        if live_buffer.size < min_samples_for_inference:
            buffered_seconds = live_buffer.size / float(bundle.cfg.sample_rate)
            status_placeholder.info(f"Listening... ({buffered_seconds:.2f}s buffered)")
            continue

        window_size_samples = int(max(window_seconds, bundle.cfg.duration_seconds) * bundle.cfg.sample_rate)
        analysis_window = live_buffer[-window_size_samples:]
        rms = float(np.sqrt(np.mean(np.square(analysis_window)) + 1e-12))
        if rms < vad_rms:
            silent_since = st.session_state[silence_key]
            if silent_since is None:
                st.session_state[silence_key] = now
                silent_for = 0.0
            else:
                silent_for = now - float(silent_since)
            if silent_for >= max(1.0, update_interval * 2.0):
                reset_live_state(buf_key, emo_ema_key, int_ema_key, silence_key)
            status_placeholder.info("Listening... no speech detected in current window.")
            continue
        st.session_state[silence_key] = None

        waveform = preprocess_live_waveform(analysis_window, bundle.cfg.sample_rate, bundle.cfg)

        with results_placeholder.container():
            if bundle.task_type == "emotion_intensity_multitask":
                emotion_probs, intensity_probs = predict_multitask(bundle, waveform)

                prev_emo = st.session_state[emo_ema_key]
                if prev_emo is not None and len(prev_emo) == len(emotion_probs):
                    emotion_probs = (smoothing * prev_emo) + ((1.0 - smoothing) * emotion_probs)
                    emotion_probs = emotion_probs / (np.sum(emotion_probs) + 1e-8)
                st.session_state[emo_ema_key] = emotion_probs

                prev_int = st.session_state[int_ema_key]
                if prev_int is not None and len(prev_int) == len(intensity_probs):
                    intensity_probs = (smoothing * prev_int) + ((1.0 - smoothing) * intensity_probs)
                    intensity_probs = intensity_probs / (np.sum(intensity_probs) + 1e-8)
                st.session_state[int_ema_key] = intensity_probs

                e_idx = int(np.argmax(emotion_probs))
                i_idx = int(np.argmax(intensity_probs))
                status_placeholder.success(
                    f"Live: **{bundle.emotion_labels[e_idx].upper()}** + **{bundle.intensity_labels[i_idx].upper()}** "
                    f"| RMS: {rms:.4f}"
                )
                if show_probability_details:
                    render_multitask_outputs(
                        emotion_probs,
                        intensity_probs,
                        bundle.emotion_labels,
                        bundle.intensity_labels,
                    )
                else:
                    st.metric("Emotion", bundle.emotion_labels[e_idx].upper(), f"{float(emotion_probs[e_idx]):.1%}")
                    st.metric(
                        "Intensity",
                        bundle.intensity_labels[i_idx].upper(),
                        f"{float(intensity_probs[i_idx]):.1%}",
                    )
            else:
                probs = predict_single_task(bundle, waveform)
                prev_emo = st.session_state[emo_ema_key]
                if prev_emo is not None and len(prev_emo) == len(probs):
                    probs = (smoothing * prev_emo) + ((1.0 - smoothing) * probs)
                    probs = probs / (np.sum(probs) + 1e-8)
                st.session_state[emo_ema_key] = probs

                top_idx = int(np.argmax(probs))
                status_placeholder.success(
                    f"Live: **{bundle.emotion_labels[top_idx].upper()}** ({float(probs[top_idx]):.1%}) | RMS: {rms:.4f}"
                )
                if show_probability_details:
                    render_single_task_outputs(probs, bundle.emotion_labels)
                else:
                    st.metric("Emotion", bundle.emotion_labels[top_idx].upper(), f"{float(probs[top_idx]):.1%}")

        if show_spectrogram and (now - float(st.session_state[spec_key])) >= DEFAULT_SPECTROGRAM_REFRESH_SECONDS:
            with spectrogram_placeholder.container():
                render_live_spectrogram(analysis_window, bundle.cfg.sample_rate)
            st.session_state[spec_key] = now
        else:
            if not show_spectrogram:
                spectrogram_placeholder.empty()


def render_clip_mode(bundle: InferenceBundle) -> None:
    st.markdown("### Clip mode")
    st.markdown(
        "- Record and stop using microphone input, then run one-shot prediction.\n"
        "- Or upload a `.wav`, `.mp3`, `.m4a`, or `.ogg` clip."
    )

    mic_audio = st.audio_input("Record speech")
    file_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"])

    selected_audio = mic_audio if mic_audio is not None else file_audio
    if selected_audio is None:
        st.info("Add audio to run one-shot prediction.")
        return

    st.audio(selected_audio)

    if st.button("Predict Emotion", type="primary"):
        y, sr = read_audio_from_uploaded(selected_audio)
        waveform = preprocess_waveform(y, sr, bundle.cfg)

        if bundle.task_type == "emotion_intensity_multitask":
            emotion_probs, intensity_probs = predict_multitask(bundle, waveform)
            render_multitask_outputs(
                emotion_probs,
                intensity_probs,
                bundle.emotion_labels,
                bundle.intensity_labels,
            )
        else:
            probs = predict_single_task(bundle, waveform)
            render_single_task_outputs(probs, bundle.emotion_labels)


def render_app() -> None:
    st.title("🎙️ Speech Emotion Detector")
    st.caption("Hybrid HuBERT + engineered-feature model with live inference.")

    backend_label = st.selectbox(
        "Inference backend",
        options=["Custom artifacts", "emotion2vec_plus_seed"],
        help="Switch between the local trained multitask model and the optional off-the-shelf emotion2vec model.",
    )
    backend_name = "emotion2vec" if backend_label == "emotion2vec_plus_seed" else "custom"

    artifacts_dir = Path(default_artifacts_dir())
    emotion2vec_model_id = DEFAULT_EMOTION2VEC_MODEL_ID
    if backend_name == "custom":
        artifacts_dir = Path(
            st.text_input("Artifacts directory", value=str(artifacts_dir), help="Folder containing model + metadata.")
        )
        if not _artifacts_ready(artifacts_dir):
            st.error(
                "Trained artifacts not found. Train first with:\n\n"
                "`python train_model.py --data-dir actors_speech --output-dir artifacts`"
            )
            return
    else:
        emotion2vec_model_id = st.text_input(
            "emotion2vec model id",
            value=DEFAULT_EMOTION2VEC_MODEL_ID,
            help="FunASR/Hugging Face model id used for the off-the-shelf backend.",
        )
        if not emotion2vec_available():
            st.warning(
                "emotion2vec backend needs optional dependencies. Install with:\n\n"
                "`pip install -U funasr modelscope`"
            )
            return

    try:
        bundle = load_model_bundle(backend_name, str(artifacts_dir), emotion2vec_model_id)
    except Exception as exc:
        st.error(f"Failed to load backend: {exc}")
        return

    if bundle.backend_name == "emotion2vec":
        st.info(
            "Loaded off-the-shelf emotion2vec backend: emotion-only inference "
            f"({', '.join(bundle.emotion_labels)}). Intensity prediction is not available in this mode."
        )
    elif bundle.task_type == "emotion_intensity_multitask":
        aux_info = "with engineered features" if bundle.use_handcrafted_features else "without engineered features"
        st.info(
            "Loaded multi-task model: predicts both emotion class and intensity degree "
            f"({', '.join(bundle.intensity_labels)}), {aux_info}."
        )
    else:
        st.info("Loaded single-task model: emotion class only.")

    live_tab, clip_tab = st.tabs(["Live", "Clip"])
    with live_tab:
        render_live_mode(bundle, artifacts_dir)
    with clip_tab:
        render_clip_mode(bundle)


if __name__ == "__main__":
    render_app()
