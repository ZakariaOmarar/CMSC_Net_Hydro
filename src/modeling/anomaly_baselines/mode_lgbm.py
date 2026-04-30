"""V0 LightGBM mode classifier on hand-engineered features.

Produces the **upper-bound reference row** for the RQ1 cluster-purity table:
how well could a classifier do if it *were* allowed to use mode labels?  V1's
per-modality SSL and V2's full multimodal SSL both target this number from
below without using labels at training time.

Per-window features:
  Acoustic (mean-pooled across mics):
    - RMS
    - kurtosis
    - spectral centroid
    - n_mels log-mel band means (default n_mels=64)
  Vibration (mean-pooled across vibration channels):
    - amplitude RMS, kurtosis, mean, std

Training:
  - 4-class: {Pump, Standstill, Turbine, RandomFault} on D1+D2 folder labels.
  - Held-out *recordings* (not held-out windows) per the user-set sanity-gate
    contract.  No window-level leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch  # noqa: F401  # keeps deterministic seeding consistent with lstm_ae.py

from scipy.stats import kurtosis as _kurtosis

from ...features.audio_spectral import compute_log_mel_spectrogram
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V0ModeConfig:
    """Hyperparameters for the V0 LightGBM mode classifier."""

    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    window_seconds: float = 1.0
    window_overlap: float = 0.5
    target_classes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
        "RandomFault",
    )
    val_ratio: float = 0.5  # held-out recordings (D1/D2 each have 1 rec per class)
    seed: int = 42
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 5  # tiny dataset; loosen the LightGBM defaults
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_mode_features(
    segment: TestDatasetSegment, cfg: V0ModeConfig
) -> tuple[np.ndarray, list[str]]:
    """Compute per-window hand-engineered features for one segment.

    Returns
    -------
    features : (n_windows, n_features) float32
    feature_names : list of human-readable names, length n_features
    """
    fs = int(segment.segment.mic_sample_rate)

    # --- Acoustic side: per-mic log-mel + RMS-style stats, then mean over mics.
    mel_per_mic: list[np.ndarray] = []
    rms_per_mic: list[float] = []
    kurt_per_mic: list[float] = []
    centroid_per_mic: list[float] = []
    raw_mics = segment.segment.mic_data

    for ch in range(segment.segment.n_mic_channels):
        x = raw_mics[ch].astype(np.float64)
        rms_per_mic.append(float(np.sqrt(np.mean(x * x) + 1e-12)))
        kurt_per_mic.append(float(_kurtosis(x, fisher=True, bias=True)))
        # Spectral centroid via FFT magnitude
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
        centroid_per_mic.append(centroid)
        mel_per_mic.append(
            compute_log_mel_spectrogram(
                x.astype(np.float32),
                fs=fs,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
            )
        )

    # mel_pool: (n_mels, n_frames) — mean over mics
    mel_pool = np.stack(mel_per_mic, axis=0).mean(axis=0)
    mic_rms = float(np.mean(rms_per_mic))
    mic_kurt = float(np.mean(kurt_per_mic))
    mic_centroid = float(np.mean(centroid_per_mic))

    frames_per_window = max(1, int(round(cfg.window_seconds * fs / cfg.hop_length)))
    step = max(1, int(round(frames_per_window * (1.0 - cfg.window_overlap))))
    n_frames = mel_pool.shape[1]
    if n_frames < frames_per_window:
        return (
            np.zeros((0, cfg.n_mels + 3 + 4), dtype=np.float32),
            _feature_names(cfg.n_mels),
        )

    # --- Vibration side: per-channel stats, mean over channels.  Computed once
    # per segment (vibration is at ~Hz cadence; per-window slicing of vibration
    # would give too few samples to estimate kurtosis reliably).
    vib = segment.segment.accel_data.astype(np.float64)
    vib_rms = float(np.sqrt(np.mean(vib * vib) + 1e-12))
    vib_kurt = float(np.mean([_kurtosis(v, fisher=True, bias=True) for v in vib]))
    vib_mean = float(np.mean(vib))
    vib_std = float(np.std(vib))

    rows: list[np.ndarray] = []
    for start in range(0, n_frames - frames_per_window + 1, step):
        mel_window = mel_pool[:, start : start + frames_per_window]
        mel_means = mel_window.mean(axis=1)  # (n_mels,)
        feats = np.concatenate(
            [
                mel_means,
                np.array([mic_rms, mic_kurt, mic_centroid], dtype=np.float64),
                np.array([vib_rms, vib_kurt, vib_mean, vib_std], dtype=np.float64),
            ]
        ).astype(np.float32)
        rows.append(feats)

    if not rows:
        return (
            np.zeros((0, cfg.n_mels + 3 + 4), dtype=np.float32),
            _feature_names(cfg.n_mels),
        )
    return np.stack(rows, axis=0), _feature_names(cfg.n_mels)


def _feature_names(n_mels: int) -> list[str]:
    return (
        [f"mel_mean_{i}" for i in range(n_mels)]
        + ["mic_rms", "mic_kurtosis", "mic_spectral_centroid"]
        + ["vib_rms", "vib_kurtosis", "vib_mean", "vib_std"]
    )


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------


@dataclass
class ModeTrainResult:
    booster: Any  # lightgbm.Booster
    classes: tuple[str, ...]
    feature_names: list[str]
    standardiser_mean: np.ndarray
    standardiser_std: np.ndarray
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    val_macro_f1: float
    val_per_class_f1: dict[str, float]
    val_confusion: np.ndarray  # (n_classes, n_classes)


def _gather_labelled_windows(
    segments: Iterable[TestDatasetSegment], cfg: V0ModeConfig
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Stack windows from labelled recordings; return X, y, recording_id_per_window."""
    feature_names: list[str] | None = None
    Xs: list[np.ndarray] = []
    ys: list[int] = []
    recs: list[str] = []
    label_to_idx = {c: i for i, c in enumerate(cfg.target_classes)}
    for s in segments:
        if (s.mode_label or "") not in label_to_idx:
            continue
        feats, names = extract_mode_features(s, cfg)
        if feats.shape[0] == 0:
            continue
        if feature_names is None:
            feature_names = names
        Xs.append(feats)
        ys.extend([label_to_idx[s.mode_label]] * feats.shape[0])
        recs.extend([s.recording_id] * feats.shape[0])
    if not Xs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            [],
            feature_names or [],
        )
    return (
        np.concatenate(Xs, axis=0),
        np.asarray(ys, dtype=np.int32),
        recs,
        feature_names or _feature_names(cfg.n_mels),
    )


def _split_by_recording(
    rec_ids: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(rec_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_ids = unique[:n_val]
    train_ids = unique[n_val:]
    if not train_ids:
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, list[float], np.ndarray]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    f1_per: list[float] = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1_per.append(f1)
    return float(np.mean(f1_per)), f1_per, cm


def train_v0_mode_lgbm(
    loader: TestDatasetLoader, cfg: V0ModeConfig | None = None
) -> ModeTrainResult:
    """Train the V0 LightGBM mode classifier on labelled recordings."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required for the V0 mode classifier; install via 'pip install lightgbm'"
        ) from exc

    cfg = cfg or V0ModeConfig()
    np.random.seed(cfg.seed)

    segments = loader.list_segments()
    X, y, rec_ids, feature_names = _gather_labelled_windows(segments, cfg)
    if X.shape[0] == 0:
        raise RuntimeError("no labelled windows found for V0 mode classifier")

    train_ids, val_ids = _split_by_recording(rec_ids, cfg.val_ratio, cfg.seed)
    train_mask = np.array([r in train_ids for r in rec_ids], dtype=bool)
    val_mask = np.array([r in val_ids for r in rec_ids], dtype=bool)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": len(cfg.target_classes),
        "metric": "multi_logloss",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "verbose": -1,
        "seed": cfg.seed,
        "class_weight": "balanced",
    }
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    if X_val.shape[0] > 0:
        probs = booster.predict(X_val, num_iteration=booster.best_iteration)
        y_pred = np.argmax(probs, axis=1)
        macro_f1, per_class, cm = _macro_f1(y_val, y_pred, len(cfg.target_classes))
        per_class_dict = {c: f for c, f in zip(cfg.target_classes, per_class)}
    else:
        macro_f1, per_class_dict, cm = 0.0, {c: 0.0 for c in cfg.target_classes}, np.zeros(
            (len(cfg.target_classes), len(cfg.target_classes)), dtype=np.int64
        )

    return ModeTrainResult(
        booster=booster,
        classes=tuple(cfg.target_classes),
        feature_names=feature_names,
        standardiser_mean=mean.astype(np.float32),
        standardiser_std=std.astype(np.float32),
        train_recording_ids=sorted(train_ids),
        val_recording_ids=sorted(val_ids),
        val_macro_f1=macro_f1,
        val_per_class_f1=per_class_dict,
        val_confusion=cm,
    )


def predict_modes(
    result: ModeTrainResult,
    segments: Iterable[TestDatasetSegment],
    cfg: V0ModeConfig,
) -> list[dict]:
    """Predict per-window mode probabilities + argmax for each segment."""
    out: list[dict] = []
    for s in segments:
        feats, _ = extract_mode_features(s, cfg)
        if feats.shape[0] == 0:
            continue
        norm = (feats - result.standardiser_mean) / result.standardiser_std
        probs = result.booster.predict(norm, num_iteration=result.booster.best_iteration)
        preds = np.argmax(probs, axis=1)
        out.append(
            {
                "dataset_id": s.dataset_id,
                "recording_id": s.recording_id,
                "mode_label": s.mode_label,
                "n_windows": int(feats.shape[0]),
                "probs": probs.astype(np.float32),
                "predicted_class": np.array(
                    [result.classes[int(c)] for c in preds], dtype=object
                ),
            }
        )
    return out


__all__ = [
    "V0ModeConfig",
    "ModeTrainResult",
    "extract_mode_features",
    "train_v0_mode_lgbm",
    "predict_modes",
]
