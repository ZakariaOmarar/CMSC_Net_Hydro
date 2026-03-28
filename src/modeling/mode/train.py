"""Train an epoch-based mode classifier over latent cache windows.

The mode classifier labels each analysis window as Pump, Turbine, or Standstill
based on the concatenated latent vector [z, c]. Correct mode labeling is the
prerequisite for the transition-aware anomaly detection logic: if the mode
classifier cannot distinguish operating states, false positives during mode
transitions cannot be suppressed.

Training procedure:
1. Load latent windows and assign labels from recording ID naming conventions.
2. Apply optional Mixup augmentation within the z component only.
3. Train ModeCNN2DClassifier with cross-entropy + optional class weighting.
4. Select the best checkpoint by validation macro-F1.
5. Export mode_classifier.pt, mode_training_summary.json, mode_test_predictions.json.

Mode label mapping from recording ID:
  contains 'Pump'       → Pump
  contains 'Turbine'    → Turbine
  contains 'RandomFault'→ Turbine (faulty turbine windows keep the mode label)
  contains 'StandStill' → Standstill
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..flow.data import load_latent_dataset
from ..core.artifact_contracts import ARTIFACT_SCHEMA_VERSION
from ..core.artifact_contracts import stamp_artifact_metadata
from .data import batch_iterator
from .data import split_stratified_indices
from ..core.runtime_utils import enable_global_determinism
from .eval import build_cv_splits
from .eval import evaluate
from .eval import macro_f1_score
from .eval import resolve_class_weights
from ..models import ModeCNN2DClassifier
from ..core.runtime_utils import emit_event as _emit_event
from ..core.runtime_utils import is_randomfault_recording as _is_randomfault_recording
from ..core.runtime_utils import resolve_mode_label as _resolve_mode_label


@dataclass(frozen=True)
class ModeTrainingArtifacts:
    """Paths of the three files written by train_mode_classifier."""
    artifact_path: Path
    summary_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class _TrainEvalResult:
    best_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    best_selection_value: float
    trained_epochs: int


class FeatureAugmenter:
    """Mixup-style augmentation applied to the z portion of the feature vector.

    Only the diagnostic features z are augmented; the context features c are
    left intact. Augmentation is applied only during training to improve
    generalization across recording nodes that have different signal levels.

    Args:
        noise_std: Standard deviation of additive Gaussian noise on z.
        scale_range: Uniform random multiplicative scale range on z.
        dropout_p: Probability of zeroing each z feature independently.
    """

    def __init__(
        self,
        *,
        noise_std: float = 0.02,
        scale_range: tuple[float, float] = (0.95, 1.05),
        dropout_p: float = 0.05,
    ) -> None:
        self.noise_std = float(max(0.0, noise_std))
        lo = float(scale_range[0])
        hi = float(scale_range[1])
        self.scale_lo = float(min(lo, hi))
        self.scale_hi = float(max(lo, hi))
        self.dropout_p = float(min(1.0, max(0.0, dropout_p)))

    def __call__(self, x: torch.Tensor, *, z_dim: int) -> torch.Tensor:
        if x.ndim != 2:
            return x
        dz = int(max(0, min(int(z_dim), int(x.shape[1]))))
        if dz <= 0:
            return x

        z = x[:, :dz]
        if self.noise_std > 0.0:
            z = z + torch.randn_like(z) * float(self.noise_std)

        if self.scale_hi > 0.0:
            scales = torch.empty(
                (z.shape[0], 1),
                device=z.device,
                dtype=z.dtype,
            ).uniform_(self.scale_lo, self.scale_hi)
            z = z * scales

        if self.dropout_p > 0.0:
            keep = (torch.rand_like(z) > self.dropout_p).to(dtype=z.dtype)
            z = z * keep

        if dz == int(x.shape[1]):
            return z
        return torch.cat([z, x[:, dz:]], dim=1)


def _mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_a = y
    y_b = y[idx]
    return x_mix, y_a, y_b, lam


def _train_single_split(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    hidden_dim: int,
    classifier_dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    label_smoothing: float,
    z_dim: int,
    feature_augment: bool,
    augment_noise_std: float,
    augment_scale_min: float,
    augment_scale_max: float,
    augment_dropout_p: float,
    mixup_alpha: float,
    use_class_weights: bool,
    selection_metric: str,
    seed: int,
    device: torch.device,
    log_every: int,
    quiet: bool,
    event_prefix: str,
) -> _TrainEvalResult:
    model = ModeCNN2DClassifier(
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(hidden_dim),
        n_classes=int(n_classes),
        dropout_p=float(classifier_dropout),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
    )

    class_weights_t: torch.Tensor | None = None
    if use_class_weights:
        cw = resolve_class_weights(y_train, n_classes=n_classes)
        class_weights_t = torch.from_numpy(cw).to(device, dtype=torch.float32)

    augmenter = FeatureAugmenter(
        noise_std=float(augment_noise_std),
        scale_range=(float(augment_scale_min), float(augment_scale_max)),
        dropout_p=float(augment_dropout_p),
    )

    history: list[dict[str, float]] = []
    best_selection_value = float("-inf")
    best_val_acc = float("nan")
    best_val_macro_f1 = float("nan")
    best_epoch = 1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improve = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: list[float] = []

        for x_b, y_b in batch_iterator(
            x_train,
            y_train,
            batch_size=int(batch_size),
            device=device,
            shuffle=True,
            seed=int(seed) + int(epoch),
        ):
            optimizer.zero_grad(set_to_none=True)

            x_in = augmenter(x_b, z_dim=int(z_dim)) if bool(feature_augment) else x_b

            if float(mixup_alpha) > 0.0 and int(x_in.shape[0]) >= 2:
                x_mix, y_a, y_b_mix, lam = _mixup_batch(
                    x_in, y_b, alpha=float(mixup_alpha)
                )
                logits = model(x_mix)
                loss_a = F.cross_entropy(
                    logits,
                    y_a,
                    weight=class_weights_t,
                    label_smoothing=float(label_smoothing),
                )
                loss_b = F.cross_entropy(
                    logits,
                    y_b_mix,
                    weight=class_weights_t,
                    label_smoothing=float(label_smoothing),
                )
                loss = float(lam) * loss_a + (1.0 - float(lam)) * loss_b
            else:
                logits = model(x_in)
                loss = F.cross_entropy(
                    logits,
                    y_b,
                    weight=class_weights_t,
                    label_smoothing=float(label_smoothing),
                )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_loss_eval, train_acc, y_train_pred, _ = evaluate(
            model, x_train, y_train, batch_size=int(batch_size), device=device
        )
        val_loss, val_acc, y_val_pred, _ = evaluate(
            model, x_val, y_val, batch_size=int(batch_size), device=device
        )

        train_macro_f1 = macro_f1_score(y_train, y_train_pred, n_classes=int(n_classes))
        val_macro_f1 = macro_f1_score(y_val, y_val_pred, n_classes=int(n_classes))

        selection_value = (
            float(val_macro_f1)
            if selection_metric == "val_macro_f1"
            else float(val_acc)
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(
                    train_loss if np.isfinite(train_loss) else train_loss_eval
                ),
                "train_accuracy": float(train_acc),
                "train_macro_f1": float(train_macro_f1),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = selection_value > best_selection_value
        if improved:
            best_selection_value = float(selection_value)
            best_val_acc = float(val_acc)
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch == 1 or epoch == int(epochs) or (epoch % int(log_every) == 0):
            _emit_event(
                f"{event_prefix}_epoch",
                quiet=quiet,
                epoch=int(epoch),
                epochs=int(epochs),
                train_accuracy=float(train_acc),
                val_accuracy=float(val_acc),
                train_macro_f1=float(train_macro_f1),
                val_macro_f1=float(val_macro_f1),
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        if epochs_without_improve >= int(patience):
            _emit_event(
                f"{event_prefix}_early_stop",
                quiet=quiet,
                epoch=int(epoch),
                best_epoch=int(best_epoch),
                patience=int(patience),
            )
            scheduler.step()
            break

        scheduler.step()

    return _TrainEvalResult(
        best_state=best_state,
        history=history,
        best_epoch=int(best_epoch),
        best_val_accuracy=float(best_val_acc),
        best_val_macro_f1=float(best_val_macro_f1),
        best_selection_value=float(best_selection_value),
        trained_epochs=int(len(history)),
    )


def train_mode_classifier(
    *,
    latent_paths: Iterable[Path],
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    hidden_dim: int = 256,
    classifier_dropout: float = 0.2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cv_folds: int = 5,
    patience: int = 30,
    label_smoothing: float = 0.1,
    feature_augment: bool = True,
    augment_noise_std: float = 0.02,
    augment_scale_min: float = 0.95,
    augment_scale_max: float = 1.05,
    augment_dropout_p: float = 0.05,
    mixup_alpha: float = 0.2,
    exclude_randomfault: bool = False,
    use_class_weights: bool = True,
    selection_metric: str = "val_macro_f1",
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
    quiet: bool = False,
) -> ModeTrainingArtifacts:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if selection_metric not in {"val_accuracy", "val_macro_f1"}:
        raise ValueError("selection_metric must be one of: val_accuracy, val_macro_f1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1)")
    if augment_noise_std < 0.0:
        raise ValueError("augment_noise_std must be >= 0")
    if not (0.0 <= augment_dropout_p < 1.0):
        raise ValueError("augment_dropout_p must be in [0, 1)")
    if mixup_alpha < 0.0:
        raise ValueError("mixup_alpha must be >= 0")
    if not (0.0 <= classifier_dropout < 1.0):
        raise ValueError("classifier_dropout must be in [0, 1)")

    enable_global_determinism(int(seed))

    dataset = load_latent_dataset(latent_paths)

    excluded_randomfault_windows = 0
    if exclude_randomfault:
        keep_mask = np.asarray(
            [not _is_randomfault_recording(str(rid)) for rid in dataset.recording_id],
            dtype=bool,
        )
        excluded_randomfault_windows = int(np.sum(~keep_mask))
        if not np.any(keep_mask):
            raise ValueError(
                "All windows were excluded by --exclude-randomfault. "
                "Provide latent data with healthy mode recordings."
            )
        dataset = dataset.__class__(
            z=dataset.z[keep_mask],
            c=dataset.c[keep_mask],
            recording_id=dataset.recording_id[keep_mask],
            is_transition_window=dataset.is_transition_window[keep_mask],
        )

    x = np.concatenate([dataset.z, dataset.c], axis=1).astype(np.float32)
    z_dim = int(dataset.z.shape[1])
    mode_labels = np.asarray(
        [_resolve_mode_label(str(rid)) for rid in dataset.recording_id], dtype=str
    )

    classes = sorted(set(mode_labels.tolist()))
    if len(classes) < 2:
        raise ValueError(
            "Mode training requires at least 2 classes. Found: "
            f"{classes}. Check recording_id labels in latent cache."
        )

    class_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.asarray([class_to_idx[name] for name in mode_labels], dtype=np.int64)

    train_idx, val_idx, test_idx = split_stratified_indices(
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    torch_device = torch.device(device)

    # Cross-validation on the development set for robust model comparison.
    dev_idx = np.concatenate([train_idx, val_idx], axis=0)
    x_dev = x[dev_idx]
    y_dev = y[dev_idx]
    rid_dev = dataset.recording_id[dev_idx].astype(str)

    cv_splits, cv_strategy, cv_warning = build_cv_splits(
        y_dev, rid_dev, n_splits=int(cv_folds), seed=int(seed)
    )

    cv_folds_payload: list[dict[str, float | int]] = []
    cv_macro_f1_vals: list[float] = []
    cv_acc_vals: list[float] = []
    for fold_i, (tr_local, va_local) in enumerate(cv_splits, start=1):
        fold_x_train = x_dev[tr_local]
        fold_y_train = y_dev[tr_local]
        fold_x_val = x_dev[va_local]
        fold_y_val = y_dev[va_local]

        fold_mean = fold_x_train.mean(axis=0, keepdims=True)
        fold_std = fold_x_train.std(axis=0, keepdims=True)
        fold_std = np.where(fold_std < 1e-6, 1.0, fold_std)

        fold_train_n = ((fold_x_train - fold_mean) / fold_std).astype(np.float32)
        fold_val_n = ((fold_x_val - fold_mean) / fold_std).astype(np.float32)

        fold_result = _train_single_split(
            x_train=fold_train_n,
            y_train=fold_y_train,
            x_val=fold_val_n,
            y_val=fold_y_val,
            n_classes=len(classes),
            hidden_dim=int(hidden_dim),
            classifier_dropout=float(classifier_dropout),
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            patience=int(patience),
            label_smoothing=float(label_smoothing),
            z_dim=int(z_dim),
            feature_augment=bool(feature_augment),
            augment_noise_std=float(augment_noise_std),
            augment_scale_min=float(augment_scale_min),
            augment_scale_max=float(augment_scale_max),
            augment_dropout_p=float(augment_dropout_p),
            mixup_alpha=float(mixup_alpha),
            use_class_weights=bool(use_class_weights),
            selection_metric=str(selection_metric),
            seed=int(seed) + (1000 * int(fold_i)),
            device=torch_device,
            log_every=int(log_every),
            quiet=bool(quiet),
            event_prefix=f"mode_cv_fold_{fold_i}",
        )

        cv_macro_f1_vals.append(float(fold_result.best_val_macro_f1))
        cv_acc_vals.append(float(fold_result.best_val_accuracy))
        cv_folds_payload.append(
            {
                "fold": int(fold_i),
                "n_train": int(fold_train_n.shape[0]),
                "n_val": int(fold_val_n.shape[0]),
                "best_epoch": int(fold_result.best_epoch),
                "best_val_accuracy": float(fold_result.best_val_accuracy),
                "best_val_macro_f1": float(fold_result.best_val_macro_f1),
                "trained_epochs": int(fold_result.trained_epochs),
            }
        )

    cv_macro_f1_mean = (
        float(np.mean(cv_macro_f1_vals)) if cv_macro_f1_vals else float("nan")
    )
    cv_macro_f1_std = (
        float(np.std(cv_macro_f1_vals)) if cv_macro_f1_vals else float("nan")
    )
    cv_accuracy_mean = float(np.mean(cv_acc_vals)) if cv_acc_vals else float("nan")
    cv_accuracy_std = float(np.std(cv_acc_vals)) if cv_acc_vals else float("nan")

    # Final training on train split; validation split used for checkpoint selection.
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_n = ((x_train - mean) / std).astype(np.float32)
    x_val_n = ((x_val - mean) / std).astype(np.float32)
    x_test_n = ((x_test - mean) / std).astype(np.float32)

    _emit_event(
        "mode_training_start",
        quiet=quiet,
        n_samples=int(x.shape[0]),
        n_train=int(x_train_n.shape[0]),
        n_val=int(x_val_n.shape[0]),
        n_test=int(test_idx.shape[0]),
        classes=classes,
        exclude_randomfault=bool(exclude_randomfault),
        excluded_randomfault_windows=int(excluded_randomfault_windows),
        use_class_weights=bool(use_class_weights),
        selection_metric=str(selection_metric),
        cv_folds=int(len(cv_splits)),
        cv_strategy=str(cv_strategy),
        seed=int(seed),
        deterministic=True,
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=str(torch_device),
    )
    final_result = _train_single_split(
        x_train=x_train_n,
        y_train=y_train,
        x_val=x_val_n,
        y_val=y_val,
        n_classes=len(classes),
        hidden_dim=int(hidden_dim),
        classifier_dropout=float(classifier_dropout),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        patience=int(patience),
        label_smoothing=float(label_smoothing),
        z_dim=int(z_dim),
        feature_augment=bool(feature_augment),
        augment_noise_std=float(augment_noise_std),
        augment_scale_min=float(augment_scale_min),
        augment_scale_max=float(augment_scale_max),
        augment_dropout_p=float(augment_dropout_p),
        mixup_alpha=float(mixup_alpha),
        use_class_weights=bool(use_class_weights),
        selection_metric=str(selection_metric),
        seed=int(seed),
        device=torch_device,
        log_every=int(log_every),
        quiet=bool(quiet),
        event_prefix="mode_train",
    )

    model = ModeCNN2DClassifier(
        input_dim=int(x_train_n.shape[1]),
        hidden_dim=int(hidden_dim),
        n_classes=int(len(classes)),
        dropout_p=float(classifier_dropout),
    ).to(torch_device)
    model.load_state_dict(final_result.best_state)

    test_loss, test_acc, y_test_pred, y_test_prob = evaluate(
        model, x_test_n, y_test, batch_size=int(batch_size), device=torch_device
    )
    test_macro_f1 = macro_f1_score(y_test, y_test_pred, n_classes=len(classes))

    y_test_true_names = [classes[int(i)] for i in y_test.tolist()]
    y_test_pred_names = [classes[int(i)] for i in y_test_pred.tolist()]

    predictions_payload = {
        "n_test": int(test_idx.shape[0]),
        "items": [
            {
                "sample_index": int(test_idx[i]),
                "recording_id": str(dataset.recording_id[test_idx[i]]),
                "y_true": str(y_test_true_names[i]),
                "y_pred": str(y_test_pred_names[i]),
                "correct": bool(y_test_true_names[i] == y_test_pred_names[i]),
                "probabilities": {
                    classes[j]: float(y_test_prob[i, j]) for j in range(len(classes))
                },
            }
            for i in range(test_idx.shape[0])
        ],
    }

    _emit_event(
        "mode_training_done",
        quiet=quiet,
        best_epoch=int(final_result.best_epoch),
        best_val_accuracy=float(final_result.best_val_accuracy),
        best_val_macro_f1=float(final_result.best_val_macro_f1),
        best_selection_metric=str(selection_metric),
        best_selection_value=float(final_result.best_selection_value),
        test_accuracy=float(test_acc),
        test_macro_f1=float(test_macro_f1),
        test_loss=float(test_loss),
        cv_macro_f1_mean=float(cv_macro_f1_mean),
        cv_macro_f1_std=float(cv_macro_f1_std),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / "mode_classifier.pt"
    summary_path = output_dir / "mode_training_summary.json"
    predictions_path = output_dir / "mode_test_predictions.json"

    torch.save(
        {
            "_meta": stamp_artifact_metadata(artifact_type="mode"),
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": int(hidden_dim),
            "classes": classes,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "best_epoch": int(final_result.best_epoch),
            "best_val_accuracy": float(final_result.best_val_accuracy),
            "best_val_macro_f1": float(final_result.best_val_macro_f1),
            "best_selection_metric": str(selection_metric),
            "best_selection_value": float(final_result.best_selection_value),
            "use_class_weights": bool(use_class_weights),
            "mode_architecture": "cnn2d",
        },
        artifact_path,
    )

    summary_payload = {
        "n_samples": int(x.shape[0]),
        "exclude_randomfault": bool(exclude_randomfault),
        "excluded_randomfault_windows": int(excluded_randomfault_windows),
        "use_class_weights": bool(use_class_weights),
        "selection_metric": str(selection_metric),
        "label_smoothing": float(label_smoothing),
        "patience": int(patience),
        "feature_augment": bool(feature_augment),
        "augment_noise_std": float(augment_noise_std),
        "augment_scale_min": float(augment_scale_min),
        "augment_scale_max": float(augment_scale_max),
        "augment_dropout_p": float(augment_dropout_p),
        "mixup_alpha": float(mixup_alpha),
        "classifier_dropout": float(classifier_dropout),
        "mode_architecture": "cnn2d",
        "seed": int(seed),
        "deterministic": True,
        "n_train": int(train_idx.shape[0]),
        "n_val": int(val_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "classes": classes,
        "split_ratio": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "epochs": int(epochs),
        "trained_epochs": int(final_result.trained_epochs),
        "best_epoch": int(final_result.best_epoch),
        "best_val_accuracy": float(final_result.best_val_accuracy),
        "best_val_macro_f1": float(final_result.best_val_macro_f1),
        "best_selection_value": float(final_result.best_selection_value),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_loss": float(test_loss),
        "validation_accuracy": float(cv_accuracy_mean),
        "validation_macro_f1": float(cv_macro_f1_mean),
        "cv": {
            "n_folds": int(len(cv_splits)),
            "requested_folds": int(cv_folds),
            "strategy": str(cv_strategy),
            "warning": cv_warning,
            "accuracy_mean": float(cv_accuracy_mean),
            "accuracy_std": float(cv_accuracy_std),
            "macro_f1_mean": float(cv_macro_f1_mean),
            "macro_f1_std": float(cv_macro_f1_std),
            "folds": cv_folds_payload,
        },
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "history": final_result.history,
    }

    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    predictions_path.write_text(
        json.dumps(predictions_payload, indent=2),
        encoding="utf-8",
    )

    return ModeTrainingArtifacts(
        artifact_path=artifact_path,
        summary_path=summary_path,
        predictions_path=predictions_path,
    )
