"""Baseline anomaly models over latent windows: OC-SVM, LSTM-AE, and CNN-AE.

All three baselines consume the same latent cache inputs (z, c) as the
conditional flow head, enabling a direct controlled comparison. The baselines
serve as ablation references: they expose how much of the performance gain from
the CNF comes from the normalizing flow architecture vs. the latent representation.

OC-SVM  — classical one-class SVM; fast, interpretable, no training epochs.
LSTM-AE — sequence autoencoder over sliding windows of latent vectors;
           reconstruction error is the anomaly score.
CNN-AE  — 2D convolutional autoencoder treating the window (time × feature) grid as
           an image; reconstruction error is the anomaly score.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import numpy as np
import torch
import torch.nn as nn

from ..flow.data import LatentDataset, load_latent_dataset
from ..flow.eval import ModeArtifactBundle, load_mode_artifact, predict_modes
from ..core.artifact_contracts import ARTIFACT_SCHEMA_VERSION
from ..core.artifact_contracts import stamp_artifact_metadata
from ..core.artifact_contracts import validate_artifact_metadata
from .data import (
    FeatureSet,
    ModelType,
    aggregate_sequence_errors,
    build_features,
    build_sequences,
    prepare_autoencoder_input,
    split_recording_ids,
)
from ..models import (
    CNNAutoencoder,
    LSTMAutoencoder,
    build_one_class_svm,
    ensure_ocsvm_available,
)
from ..core.runtime_utils import compute_window_step_s
from ..core.runtime_utils import fit_standardizer
from ..core.runtime_utils import is_healthy_recording_id
from ..core.runtime_utils import majority_smooth_labels
from ..core.runtime_utils import apply_standardizer
from ..core.runtime_utils import enable_global_determinism
from ..core.runtime_utils import run_lengths


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

OCSVMKernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]
OCSVMGamma = Literal["scale", "auto"] | float


# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineTrainingArtifacts:
    """Paths of the two files written by train_baseline_model."""

    artifact_path: Path
    summary_path: Path


@dataclass(frozen=True)
class BaselineInferenceResult:
    """Packed inference outputs for one baseline model over a latent dataset.

    Attributes:
        scores: Per-window reconstruction errors or SVM decision distances.
        flags: Boolean anomaly flags after thresholding.
        thresholds: Per-window effective threshold (scalar broadcast to array).
    """

    scores: np.ndarray
    flags: np.ndarray
    thresholds: np.ndarray


# ---------------------------------------------------------------------------
# Internal-only containers
# ---------------------------------------------------------------------------


@dataclass
class _AEParams:
    """Hyperparameters shared between AE training, scoring, and artifact I/O.

    Grouping these avoids passing them individually through multiple call
    layers and eliminates the parallel dict packing/unpacking that existed
    in the original code.
    """

    seq_len: int
    seq_stride: int
    hidden_dim: int
    latent_dim: int
    n_layers: int
    batch_size: int
    cnn_spec_n_fft: int
    cnn_spec_hop_length: int

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation for JSON/pickle storage."""
        return {
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "n_layers": self.n_layers,
            "batch_size": self.batch_size,
            "cnn_spec_n_fft": self.cnn_spec_n_fft,
            "cnn_spec_hop_length": self.cnn_spec_hop_length,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "_AEParams":
        """Reconstruct from a stored params dict, applying safe defaults."""

        def _i(key: str, default: int) -> int:
            v = d.get(key)
            return int(v) if v is not None else default

        return cls(
            seq_len=_i("seq_len", 16),
            seq_stride=_i("seq_stride", 4),
            hidden_dim=_i("hidden_dim", 128),
            latent_dim=_i("latent_dim", 64),
            n_layers=_i("n_layers", 1),
            batch_size=_i("batch_size", 128),
            cnn_spec_n_fft=_i("cnn_spec_n_fft", 64),
            cnn_spec_hop_length=_i("cnn_spec_hop_length", 16),
        )


@dataclass
class _BranchResult:
    model_blob: dict[str, object]
    healthy_train_scores: np.ndarray
    healthy_val_scores: np.ndarray
    history: list[dict[str, float]]
    threshold: float


def _parse_ocsvm_kernel(value: str) -> OCSVMKernel:
    kernel = str(value).strip().lower()
    allowed = {"linear", "poly", "rbf", "sigmoid", "precomputed"}
    if kernel not in allowed:
        raise ValueError(
            f"Unsupported OC-SVM kernel {value!r}. Allowed: {sorted(allowed)}"
        )
    return cast(OCSVMKernel, kernel)


def _parse_ocsvm_gamma(value: str | float) -> OCSVMGamma:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    gamma = str(value).strip().lower()
    if gamma in {"scale", "auto"}:
        return cast(OCSVMGamma, gamma)
    return float(gamma)


def _score_stats(arr: np.ndarray, key: str) -> dict[str, float]:
    """Return mean and std of *arr* under prefixed keys, or NaN if empty."""
    if arr.size:
        return {
            f"{key}_mean": float(np.mean(arr)),
            f"{key}_std": float(np.std(arr)),
        }
    return {f"{key}_mean": float("nan"), f"{key}_std": float("nan")}


# ---------------------------------------------------------------------------
# Sequence / batch helpers
# ---------------------------------------------------------------------------


def _iter_seq_batches(
    sequences: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> Iterable[np.ndarray]:
    idx = np.arange(sequences.shape[0], dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, idx.shape[0], batch_size):
        yield sequences[idx[start : start + batch_size]]


def build_anomaly_events(
    result: BaselineInferenceResult,
    *,
    window_s: float,
    overlap: float,
) -> list[dict[str, float | int | str]]:
    step_s = compute_window_step_s(window_s=window_s, overlap=overlap)
    return [
        {
            "window_index": int(idx),
            "timestamp_s": float(idx * step_s),
            "score": float(result.scores[idx]),
            "threshold": float(result.thresholds[idx]),
        }
        for idx, is_anomaly in enumerate(result.flags.astype(bool).tolist())
        if is_anomaly
    ]


# ---------------------------------------------------------------------------
# Mode artifact helpers
# ---------------------------------------------------------------------------


def build_mode_predictions(
    *,
    artifact_path: Path,
    dataset: LatentDataset,
    anomaly_events: list[dict[str, float | int | str]],
    mode_consistency_window: int = 5,
    device: str = "cpu",
) -> dict[str, object]:
    bundle = load_mode_artifact(artifact_path, device=device)
    mode_labels, mode_probs = predict_modes(bundle, dataset, device=device)
    mode_labels_smoothed = majority_smooth_labels(
        mode_labels, window=max(1, int(mode_consistency_window))
    )
    mode_conf = np.max(mode_probs, axis=1).astype(np.float32)

    for item in anomaly_events:
        win = int(item["window_index"])
        item["mode"] = str(mode_labels_smoothed[win])
        item["mode_confidence"] = float(mode_conf[win])

    return {
        "mode_detection_enabled": True,
        "mode_artifact_path": str(artifact_path),
        "mode_consistency_window": int(mode_consistency_window),
        "mode_labels": mode_labels.tolist(),
        "mode_labels_smoothed": mode_labels_smoothed.tolist(),
        "mode_confidence": mode_conf.tolist(),
        "mode_run_lengths": run_lengths(mode_labels_smoothed).astype(int).tolist(),
        "anomaly_modes": [item["mode"] for item in anomaly_events],
    }


# ---------------------------------------------------------------------------
# Autoencoder helpers
# ---------------------------------------------------------------------------


def _build_ae_model(
    *,
    model_type: ModelType,
    input_dim: int,
    p: _AEParams,
) -> nn.Module:
    if model_type == "lstm_ae":
        return LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=p.hidden_dim,
            latent_dim=p.latent_dim,
            n_layers=p.n_layers,
        )
    if model_type == "cnn_ae":
        return CNNAutoencoder(
            input_dim=input_dim,
            hidden_dim=p.hidden_dim,
            latent_dim=p.latent_dim,
        )
    raise ValueError(f"Unsupported autoencoder model_type: {model_type!r}")


def _ae_sequence_errors(
    model: nn.Module,
    sequences: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _iter_seq_batches(sequences, batch_size=batch_size, shuffle=False):
            x_t = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
            err_t = torch.mean((model(x_t) - x_t) ** 2, dim=(1, 2))
            errors.append(err_t.cpu().numpy().astype(np.float32))
    return np.concatenate(errors, axis=0)


def _train_autoencoder(
    *,
    model_type: ModelType,
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    input_dim: int,
    p: _AEParams,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
    seed: int,
) -> tuple[nn.Module, list[dict[str, float]], float]:
    torch_device = torch.device(device)
    model = _build_ae_model(model_type=model_type, input_dim=input_dim, p=p).to(
        torch_device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    # Cosine annealing is only beneficial for LSTMs, which are sensitive to
    # the learning-rate schedule.  CNNs use a fixed rate.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, min(50, epochs))
        )
        if model_type == "lstm_ae"
        else None
    )
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val: float = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve: int = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch in _iter_seq_batches(
            train_sequences, batch_size=p.batch_size, shuffle=True, seed=seed + epoch
        ):
            x_t = torch.from_numpy(batch).to(device=torch_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_t), x_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu().item()) * batch.shape[0]
            train_n += batch.shape[0]

        if scheduler is not None:
            scheduler.step()

        val_err = _ae_sequence_errors(
            model, val_sequences, batch_size=p.batch_size, device=torch_device
        )
        val_loss = float(np.mean(val_err))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / max(1, train_n),
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val


def _score_autoencoder_windows(
    *,
    model_type: ModelType,
    model_blob: dict[str, object],
    x_norm: np.ndarray,
    recording_ids: np.ndarray,
    p: _AEParams,
    device: str,
) -> np.ndarray:
    seq, seq_idx = build_sequences(
        x_norm, recording_ids.astype(str), seq_len=p.seq_len, stride=p.seq_stride
    )

    seq_ae = prepare_autoencoder_input(
        model_type=model_type,
        sequences=seq,
        cnn_spec_n_fft=p.cnn_spec_n_fft,
        cnn_spec_hop_length=p.cnn_spec_hop_length,
    )

    model = _build_ae_model(
        model_type=model_type,
        input_dim=int(seq_ae.shape[-1]),
        p=p,
    )
    state_np = cast(dict[str, np.ndarray], model_blob.get("state_dict", {}))
    if not state_np:
        raise ValueError("Invalid AE artifact: 'state_dict' is missing or empty")
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state_np.items()})
    model.to(torch.device(device))

    seq_err = _ae_sequence_errors(
        model, seq_ae, batch_size=p.batch_size, device=torch.device(device)
    )
    return aggregate_sequence_errors(
        errors=seq_err, seq_indices=seq_idx, n_windows=x_norm.shape[0]
    )


# ---------------------------------------------------------------------------
# Training branch helpers
# ---------------------------------------------------------------------------


def _run_ocsvm_branch(
    x_train_n: np.ndarray,
    x_val_n: np.ndarray,
    *,
    ocsvm_kernel: str,
    ocsvm_gamma: str | float,
    ocsvm_nu: float,
    score_percentile: float,
) -> _BranchResult:
    ensure_ocsvm_available()
    model = build_one_class_svm(
        kernel=str(_parse_ocsvm_kernel(str(ocsvm_kernel))),
        gamma=_parse_ocsvm_gamma(ocsvm_gamma),
        nu=float(ocsvm_nu),
    )
    model.fit(x_train_n)

    train_scores = -np.asarray(
        model.decision_function(x_train_n), dtype=np.float32
    ).reshape(-1)
    val_scores = -np.asarray(
        model.decision_function(x_val_n), dtype=np.float32
    ).reshape(-1)
    threshold = float(np.percentile(val_scores, score_percentile))

    return _BranchResult(
        model_blob={"ocsvm": model},
        healthy_train_scores=train_scores,
        healthy_val_scores=val_scores,
        history=[],
        threshold=threshold,
    )


def _run_ae_branch(
    model_type: ModelType,
    x_train_n: np.ndarray,
    x_val_n: np.ndarray,
    rid_train: np.ndarray,
    rid_val: np.ndarray,
    *,
    p: _AEParams,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    score_percentile: float,
    device: str,
    seed: int,
) -> _BranchResult:
    train_seq, _ = build_sequences(
        x_train_n, rid_train, seq_len=p.seq_len, stride=p.seq_stride
    )
    val_seq, val_seq_idx = build_sequences(
        x_val_n, rid_val, seq_len=p.seq_len, stride=p.seq_stride
    )

    train_seq_ae = prepare_autoencoder_input(
        model_type=model_type,
        sequences=train_seq,
        cnn_spec_n_fft=p.cnn_spec_n_fft,
        cnn_spec_hop_length=p.cnn_spec_hop_length,
    )
    val_seq_ae = prepare_autoencoder_input(
        model_type=model_type,
        sequences=val_seq,
        cnn_spec_n_fft=p.cnn_spec_n_fft,
        cnn_spec_hop_length=p.cnn_spec_hop_length,
    )

    ae_model, history, best_val = _train_autoencoder(
        model_type=model_type,
        train_sequences=train_seq_ae,
        val_sequences=val_seq_ae,
        input_dim=int(train_seq_ae.shape[-1]),
        p=p,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        seed=seed,
    )

    model_blob: dict[str, object] = {
        "state_dict": {
            k: v.detach().cpu().numpy() for k, v in ae_model.state_dict().items()
        },
        "best_val_loss": float(best_val),
    }

    torch_device = torch.device(device)
    train_scores = _score_autoencoder_windows(
        model_type=model_type,
        model_blob=model_blob,
        x_norm=x_train_n,
        recording_ids=rid_train,
        p=p,
        device=device,
    )
    val_err = _ae_sequence_errors(
        ae_model, val_seq_ae, batch_size=p.batch_size, device=torch_device
    )
    val_scores = aggregate_sequence_errors(
        errors=val_err, seq_indices=val_seq_idx, n_windows=x_val_n.shape[0]
    )
    threshold = float(np.percentile(val_err, score_percentile))

    return _BranchResult(
        model_blob=model_blob,
        healthy_train_scores=train_scores,
        healthy_val_scores=val_scores,
        history=history,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def load_artifact(path: Path) -> dict[str, object]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(
            f"Invalid baseline artifact at {path}: expected dict, got {type(obj).__name__}"
        )
    validate_artifact_metadata(blob=obj, expected_type="baseline")
    return cast(dict[str, object], obj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_baseline_model(
    *,
    latent_paths: Iterable[Path],
    output_dir: Path,
    model_type: ModelType,
    feature_set: FeatureSet = "zc",
    val_ratio: float = 0.2,
    score_percentile: float = 99.0,
    ocsvm_kernel: str = "rbf",
    ocsvm_gamma: str | float = "scale",
    ocsvm_nu: float = 0.05,
    seq_len: int = 16,
    seq_stride: int = 4,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    n_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    cnn_spec_n_fft: int = 64,
    cnn_spec_hop_length: int = 16,
    device: str = "cpu",
    seed: int = 42,
    artifact_name: str | None = None,
) -> BaselineTrainingArtifacts:
    if not (90.0 <= score_percentile < 100.0):
        raise ValueError(
            f"score_percentile must be in [90, 100), got {score_percentile!r}"
        )

    enable_global_determinism(int(seed))

    full_dataset = load_latent_dataset(latent_paths)

    # Remove RandomFault recordings before fitting — they are anomalous by
    # definition and must not influence the healthy boundary.
    mask = np.asarray(
        [is_healthy_recording_id(str(r)) for r in full_dataset.recording_id],
        dtype=bool,
    )
    if not np.any(mask):
        raise ValueError("No healthy windows remain after RandomFault exclusion.")
    dataset = LatentDataset(
        z=full_dataset.z[mask],
        c=full_dataset.c[mask],
        recording_id=full_dataset.recording_id[mask],
        is_transition_window=full_dataset.is_transition_window[mask],
    )

    x_all = build_features(dataset, feature_set=feature_set)
    rid_all = dataset.recording_id.astype(str)
    train_ids, val_ids = split_recording_ids(rid_all, val_ratio=val_ratio, seed=seed)

    train_mask = np.asarray([r in train_ids for r in rid_all], dtype=bool)
    val_mask = np.asarray([r in val_ids for r in rid_all], dtype=bool)

    # If any recording is entirely absent from training (every window went to
    # val), fall back to a within-recording split so all modes are represented
    # in both train and val regardless of how many recordings are present.
    unique_rids = np.unique(rid_all)
    if len(unique_rids) > 1 and not np.all(np.isin(unique_rids, list(train_ids))):
        rng = np.random.default_rng(seed)
        train_mask = np.ones(rid_all.shape[0], dtype=bool)
        val_mask = np.zeros(rid_all.shape[0], dtype=bool)
        for rid_name in unique_rids:
            idx = np.where(rid_all == rid_name)[0]
            # Shuffle within recording then take last val_ratio fraction as val.
            shuffled_idx = rng.permutation(idx)
            n_val_rec = max(1, int(round(len(shuffled_idx) * val_ratio)))
            val_mask[shuffled_idx[:n_val_rec]] = True
            train_mask[shuffled_idx[:n_val_rec]] = False

    if not np.any(val_mask):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(rid_all.shape[0], dtype=np.int64))
        n_val = max(1, int(round(idx.shape[0] * float(val_ratio))))
        val_mask = np.zeros(rid_all.shape[0], dtype=bool)
        val_mask[idx[:n_val]] = True
        train_mask = ~val_mask

    x_train, x_val = x_all[train_mask], x_all[val_mask]
    rid_train, rid_val = rid_all[train_mask], rid_all[val_mask]

    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        raise ValueError("Train or validation split for healthy windows is empty.")

    mean, std = fit_standardizer(x_train)
    x_train_n = apply_standardizer(x_train, mean=mean, std=std)
    x_val_n = apply_standardizer(x_val, mean=mean, std=std)

    p = _AEParams(
        seq_len=seq_len,
        seq_stride=seq_stride,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        batch_size=batch_size,
        cnn_spec_n_fft=cnn_spec_n_fft,
        cnn_spec_hop_length=cnn_spec_hop_length,
    )

    if model_type == "ocsvm":
        branch = _run_ocsvm_branch(
            x_train_n,
            x_val_n,
            ocsvm_kernel=ocsvm_kernel,
            ocsvm_gamma=ocsvm_gamma,
            ocsvm_nu=ocsvm_nu,
            score_percentile=score_percentile,
        )
    else:
        branch = _run_ae_branch(
            model_type,
            x_train_n,
            x_val_n,
            rid_train,
            rid_val,
            p=p,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            score_percentile=score_percentile,
            device=device,
            seed=seed,
        )

    # Score the *full* dataset (incl. any anomalous recordings) for the summary.
    x_eval = build_features(full_dataset, feature_set=feature_set)
    x_eval_n = apply_standardizer(x_eval, mean=mean, std=std)

    if model_type == "ocsvm":
        ocsvm_model = cast(Any, branch.model_blob.get("ocsvm"))
        if ocsvm_model is None:
            raise ValueError("Missing OC-SVM model object in branch result")
        eval_scores = -np.asarray(
            ocsvm_model.decision_function(x_eval_n), dtype=np.float32
        ).reshape(-1)
    else:
        eval_scores = _score_autoencoder_windows(
            model_type=model_type,
            model_blob=branch.model_blob,
            x_norm=x_eval_n,
            recording_ids=full_dataset.recording_id.astype(str),
            p=p,
            device=device,
        )

    n_full_anomalies = int(np.sum(eval_scores > branch.threshold))
    full_anomaly_rate = (
        float(n_full_anomalies) / float(eval_scores.shape[0])
        if eval_scores.shape[0] > 0
        else 0.0
    )

    # Per-recording-class score statistics: separates healthy FPR from RF
    # detection rate so the two are not conflated in the aggregate metric.
    full_rids = full_dataset.recording_id.astype(str)
    recording_class_stats: dict[str, dict[str, float | int]] = {}
    for uid in sorted(np.unique(full_rids)):
        uid_mask = full_rids == uid
        uid_scores = eval_scores[uid_mask]
        uid_n_flagged = int(np.sum(uid_scores > branch.threshold))
        uid_n = int(uid_scores.shape[0])
        recording_class_stats[uid] = {
            "n_windows": uid_n,
            "score_mean": float(np.mean(uid_scores)),
            "score_std": float(np.std(uid_scores)),
            "n_flagged": uid_n_flagged,
            "flag_rate": float(uid_n_flagged) / float(uid_n) if uid_n > 0 else 0.0,
        }
    # FPR is measured only on healthy (non-RandomFault) windows.
    healthy_mask_full = np.asarray(
        [is_healthy_recording_id(r) for r in full_rids], dtype=bool
    )
    healthy_eval_scores = eval_scores[healthy_mask_full]
    n_healthy_flagged = int(np.sum(healthy_eval_scores > branch.threshold))
    healthy_fpr = (
        float(n_healthy_flagged) / float(healthy_eval_scores.shape[0])
        if healthy_eval_scores.shape[0] > 0
        else 0.0
    )

    # ----- Persist artifact ------------------------------------------------
    artifact: dict[str, object] = {
        "_meta": stamp_artifact_metadata(artifact_type="baseline"),
        "model_type": str(model_type),
        "feature_set": str(feature_set),
        "mean": mean,
        "std": std,
        "threshold": float(branch.threshold),
        "score_percentile": float(score_percentile),
        "input_dim": int(x_train_n.shape[1]),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "params": {
            "ocsvm_kernel": str(ocsvm_kernel),
            "ocsvm_gamma": ocsvm_gamma,
            "ocsvm_nu": float(ocsvm_nu),
            **p.to_dict(),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "seed": int(seed),
        },
        "model_blob": branch.model_blob,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_stub = artifact_name or f"anomaly_baseline_{model_type}"
    artifact_path = output / f"{model_stub}.pkl"
    summary_path = output / f"{model_stub}_summary.json"

    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)

    summary: dict[str, object] = {
        "model_type": str(model_type),
        "feature_set": str(feature_set),
        "n_windows": int(x_all.shape[0]),
        "n_full_windows": int(x_eval.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "threshold": float(branch.threshold),
        "score_percentile": float(score_percentile),
        **_score_stats(branch.healthy_train_scores, "healthy_train_score"),
        **_score_stats(branch.healthy_val_scores, "healthy_val_score"),
        **_score_stats(eval_scores, "full_score"),
        "n_full_anomalies": int(n_full_anomalies),
        "full_anomaly_rate": float(full_anomaly_rate),
        "healthy_fpr": float(healthy_fpr),
        "recording_class_stats": recording_class_stats,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "history": branch.history,
        "artifact_path": str(artifact_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return BaselineTrainingArtifacts(
        artifact_path=artifact_path,
        summary_path=summary_path,
    )


def infer_baseline_model(
    *,
    artifact_path: Path,
    latent_paths: Iterable[Path],
    score_threshold: float | None = None,
    device: str = "cpu",
) -> BaselineInferenceResult:
    artifact = load_artifact(artifact_path)
    model_type = cast(ModelType, str(artifact.get("model_type", "")))
    feature_set = cast(FeatureSet, str(artifact.get("feature_set", "zc")))
    mean = np.asarray(artifact.get("mean"), dtype=np.float32)
    std = np.asarray(artifact.get("std"), dtype=np.float32)
    _thresh = artifact.get("threshold")
    threshold = float(
        score_threshold
        if score_threshold is not None
        else (_thresh if _thresh is not None else 0.0)
    )

    dataset = load_latent_dataset(latent_paths)
    x_norm = apply_standardizer(
        build_features(dataset, feature_set=feature_set), mean=mean, std=std
    )

    params_dict = cast(dict[str, object], artifact.get("params", {}))
    model_blob = cast(dict[str, object], artifact.get("model_blob", {}))

    if model_type == "ocsvm":
        ocsvm = model_blob.get("ocsvm")
        if ocsvm is None:
            raise ValueError("Invalid OC-SVM artifact: 'ocsvm' key is missing")
        scores = -np.asarray(
            cast(Any, ocsvm).decision_function(x_norm), dtype=np.float32
        ).reshape(-1)

    elif model_type in {"lstm_ae", "cnn_ae"}:
        p = _AEParams.from_dict(params_dict)
        scores = _score_autoencoder_windows(
            model_type=model_type,
            model_blob=model_blob,
            x_norm=x_norm,
            recording_ids=dataset.recording_id.astype(str),
            p=p,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model_type in artifact: {model_type!r}")

    thresholds = np.full(scores.shape, float(threshold), dtype=np.float32)
    return BaselineInferenceResult(
        scores=scores.astype(np.float32),
        flags=(scores > thresholds).astype(bool),
        thresholds=thresholds,
    )
