from __future__ import annotations

import io
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from ser_multitask import MultiTaskEmotionModel, load_checkpoint_state
from ser_pipeline import AudioConfig, FeatureConfig, ensure_audio_length, extract_handcrafted_features

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False


st.set_page_config(page_title="Speech Emotion Detector", page_icon="🎙️", layout="centered")


@dataclass
class InferenceBundle:
    task_type: str  # "single_task" or "emotion_intensity_multitask"
    model: object
    feature_extractor: AutoFeatureExtractor
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

    hf_model_path = base / "hf_model"
    state_path = base / "model_state.pt"
    if not state_path.exists():
        state_path = base / "best_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            "Multitask artifacts require `model_state.pt` (or `best_model.pt`) in artifacts directory."
        )

    payload = load_checkpoint_state(str(state_path), map_location=device)

    head_dropout = float(payload.get("head_dropout", metadata.get("head_dropout", 0.2)))
    use_handcrafted_features = bool(
        payload.get("use_handcrafted_features", metadata.get("use_handcrafted_features", False))
    )
    aux_feature_dim = int(payload.get("aux_feature_dim", metadata.get("aux_feature_dim", 0)))
    aux_hidden_dim = int(payload.get("aux_hidden_dim", metadata.get("aux_hidden_dim", 128)))

    feature_cfg = _build_feature_config(metadata)
    feature_mean, feature_std = _build_feature_stats(metadata)

    model = MultiTaskEmotionModel(
        backbone_name_or_path=str(hf_model_path),
        num_emotions=len(emotion_labels),
        num_intensity=len(intensity_labels),
        head_dropout=head_dropout,
        use_handcrafted_features=use_handcrafted_features,
        aux_feature_dim=aux_feature_dim,
        aux_hidden_dim=aux_hidden_dim,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_path)

    return InferenceBundle(
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
        task_type="single_task",
        model=model,
        feature_extractor=feature_extractor,
        cfg=cfg,
        emotion_labels=emotion_labels,
        intensity_labels=[],
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_model_bundle(artifacts_dir: str) -> InferenceBundle:
    base = Path(artifacts_dir)
    metadata_path = base / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = resolve_device()
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


def render_single_task_outputs(probs: np.ndarray, labels: list[str], heading_prefix: str = "") -> None:
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    confidence = float(probs[top_idx])

    title = f"{heading_prefix}Prediction" if heading_prefix else "Prediction"
    st.success(f"{title}: **{top_label.upper()}** ({confidence:.1%} confidence)")

    score_rows = [{"label": label, "probability": float(prob)} for label, prob in zip(labels, probs)]
    st.dataframe(score_rows, use_container_width=True, hide_index=True)
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
        st.dataframe(emotion_rows, use_container_width=True, hide_index=True)
        st.bar_chart({label: float(prob) for label, prob in zip(emotion_labels, emotion_probs)})
    with col2:
        st.markdown("**Intensity probabilities**")
        intensity_rows = [
            {"intensity": label, "probability": float(prob)}
            for label, prob in zip(intensity_labels, intensity_probs)
        ]
        st.dataframe(intensity_rows, use_container_width=True, hide_index=True)
        st.bar_chart({label: float(prob) for label, prob in zip(intensity_labels, intensity_probs)})


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
            value=0.6,
            step=0.05,
        )
    with col3:
        vad_rms = st.slider(
            "Voice threshold",
            min_value=0.0,
            max_value=0.05,
            value=0.004,
            step=0.001,
        )

    smoothing = st.slider(
        "Prediction smoothing",
        min_value=0.0,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Higher values reduce jitter but respond slower.",
    )

    stream_key = f"live_{str(artifacts_dir).replace('/', '_')}"
    buf_key = state_key(stream_key, "buffer")
    emo_ema_key = state_key(stream_key, "emotion_ema")
    int_ema_key = state_key(stream_key, "intensity_ema")

    if buf_key not in st.session_state:
        st.session_state[buf_key] = np.zeros(0, dtype=np.float32)
    if emo_ema_key not in st.session_state:
        st.session_state[emo_ema_key] = None
    if int_ema_key not in st.session_state:
        st.session_state[int_ema_key] = None

    ctx = webrtc_streamer(
        key=stream_key,
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    status_placeholder = st.empty()
    results_placeholder = st.empty()

    if not ctx.state.playing:
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
            status_placeholder.info("Listening... no speech detected in current window.")
            continue

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
                render_multitask_outputs(
                    emotion_probs,
                    intensity_probs,
                    bundle.emotion_labels,
                    bundle.intensity_labels,
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
                render_single_task_outputs(probs, bundle.emotion_labels)


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

    default_artifacts = default_artifacts_dir()
    artifacts_dir = Path(
        st.text_input("Artifacts directory", value=default_artifacts, help="Folder containing model + metadata.")
    )

    if not _artifacts_ready(artifacts_dir):
        st.error(
            "Trained artifacts not found. Train first with:\n\n"
            "`python train_model.py --data-dir actors_speech --output-dir artifacts`"
        )
        return

    try:
        bundle = load_model_bundle(str(artifacts_dir))
    except Exception as exc:
        st.error(f"Failed to load artifacts: {exc}")
        return

    if bundle.task_type == "emotion_intensity_multitask":
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
