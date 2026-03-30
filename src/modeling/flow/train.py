"""End-to-end flow model training and calibration pipeline.

The training procedure:
1. Load all healthy latent windows from .npz cache files (RandomFault recordings
   are excluded from training to avoid contaminating the normal distribution).
2. Split by recording ID into train / validation / test sets.
3. Pre-train the LightweightContextEncoder via NT-Xent contrastive loss to produce
   an operational context embedding without operating mode labels.
4. Train the ConditionalRealNVP by maximizing log p(z | c) on training windows.
5. Calibrate the anomaly threshold at a configurable score percentile on val data.
6. Serialize model weights, threshold, and summary statistics to output_dir.

Outputs (under output_dir):
  flow_model.pt   — model weights + context encoder + standardizer.
  thresholds.json — calibrated anomaly threshold.
  flow_summary.json — training history and score statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .detection_head import (
    FlowConfig,
    LightweightContextEncoder,
    calibrate_threshold,
    filter_healthy_recording_ids,
    nt_xent_loss,
    train_flow_epoch,
)
from ..core.artifact_contracts import ARTIFACT_SCHEMA_VERSION
from ..core.artifact_contracts import stamp_artifact_metadata
from .data import (
    filter_healthy_latents,
    load_latent_dataset,
    split_healthy_train_val_test_indices,
)
from .train_core import augment_feature_views
from .train_core import build_flat_features
from .train_core import cluster_score_stats
from .train_core import collect_context_vectors
from .train_core import collect_scores_with_context
from .train_core import feature_batch_iterator
from .train_core import flow_batches_with_context
from .train_core import pretrain_context_encoder
from ..models import build_conditional_flow
from ..core.runtime_utils import apply_standardizer
from ..core.runtime_utils import enable_global_determinism
from ..core.runtime_utils import emit_event
from ..core.runtime_utils import fit_standardizer
from ..core.runtime_utils import is_healthy_recording_id


@dataclass(frozen=True)
class FlowTrainingArtifacts:
    """Paths of the three files written by train_and_calibrate_flow."""

    artifact_path: Path
    threshold_path: Path
    summary_path: Path


def train_and_calibrate_flow(
    *,
    latent_paths: Iterable[Path],
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    n_coupling_layers: int = 8,
    n_layers: int | None = None,
    hidden_dim: int = 256,
    context_dim: int = 32,
    dropout: float = 0.0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    score_percentile: float = 99.0,
    pretrain_context_epochs: int = 25,
    joint_finetune_epochs: int = 0,
    joint_ntxent_weight: float = 0.1,
    contrastive_temperature: float = 0.2,
    patience: int = 0,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
    quiet: bool = False,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = False,
    checkpoint_every: int = 5,
) -> FlowTrainingArtifacts:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")
    if pretrain_context_epochs < 0:
        raise ValueError("pretrain_context_epochs must be >= 0")
    if joint_finetune_epochs < 0:
        raise ValueError("joint_finetune_epochs must be >= 0")
    if context_dim <= 0:
        raise ValueError("context_dim must be > 0")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if patience < 0:
        raise ValueError("patience must be >= 0")

    determinism = enable_global_determinism(int(seed))

    full_dataset = load_latent_dataset(latent_paths)
    dataset = filter_healthy_latents(full_dataset)

    x_full = build_flat_features(full_dataset.z, full_dataset.c)
    x_all = build_flat_features(dataset.z, dataset.c)
    if not np.isfinite(x_all).all() or not np.isfinite(x_full).all():
        raise ValueError("Non-finite values detected in latent features (NaN/Inf)")

    train_idx, val_idx, test_idx = split_healthy_train_val_test_indices(
        dataset.recording_id,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    x_train = x_all[train_idx]
    x_val = x_all[val_idx]
    x_test = x_all[test_idx]

    # Drop near-constant train features to stabilize CNF likelihood scaling.
    train_std_raw = np.std(x_train, axis=0).astype(np.float32)
    keep_mask = np.asarray(train_std_raw > 1e-2, dtype=bool)
    dropped_feature_count = int(np.sum(~keep_mask))
    if not np.any(keep_mask):
        raise ValueError("All CNF input features were near-constant on train split")

    if dropped_feature_count > 0:
        x_train = x_train[:, keep_mask]
        x_val = x_val[:, keep_mask]
        x_test = x_test[:, keep_mask]
        x_full = x_full[:, keep_mask]

    mean, std = fit_standardizer(x_train)
    x_train_n = apply_standardizer(x_train, mean=mean, std=std)
    x_val_n = apply_standardizer(x_val, mean=mean, std=std)
    x_test_n = apply_standardizer(x_test, mean=mean, std=std)
    x_full_n = apply_standardizer(x_full, mean=mean, std=std)

    layer_count = int(n_layers if n_layers is not None else n_coupling_layers)

    cfg = FlowConfig(
        feature_dim=int(x_train_n.shape[1]),
        d_ctx=int(context_dim),
        n_layers=int(layer_count),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )

    torch_device = torch.device(device)
    flow = build_conditional_flow(cfg).to(torch_device)
    context_encoder = LightweightContextEncoder(
        feature_dim=int(x_train_n.shape[1]),
        d_ctx=int(context_dim),
    ).to(torch_device)

    pretrain_optimizer = torch.optim.Adam(
        context_encoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pretrain_optimizer,
        T_max=max(pretrain_context_epochs, 1),
    )

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )

    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    pretrain_epoch_done = 0
    flow_epoch_done = 0
    joint_epoch_done = 0

    history: list[dict[str, object]] = []

    def _save_checkpoint(
        *,
        pretrain_done: int,
        flow_done: int,
        joint_done: int,
        joint_optimizer_state: dict[str, object] | None,
        joint_scheduler_state: dict[str, object] | None,
    ) -> None:
        if checkpoint_file is None:
            return
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "_meta": stamp_artifact_metadata(artifact_type="flow_checkpoint"),
                "history": history,
                "pretrain_epoch_done": int(pretrain_done),
                "flow_epoch_done": int(flow_done),
                "joint_epoch_done": int(joint_done),
                "flow_state_dict": flow.state_dict(),
                "context_encoder_state_dict": context_encoder.state_dict(),
                "pretrain_optimizer_state": pretrain_optimizer.state_dict(),
                "pretrain_scheduler_state": pretrain_scheduler.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "joint_optimizer_state": joint_optimizer_state,
                "joint_scheduler_state": joint_scheduler_state,
            },
            checkpoint_file,
        )

    if checkpoint_file is not None and resume_checkpoint and checkpoint_file.exists():
        try:
            checkpoint_blob = torch.load(
                checkpoint_file,
                map_location=torch_device,
                weights_only=False,
            )
        except TypeError:
            checkpoint_blob = torch.load(checkpoint_file, map_location=torch_device)

        if not isinstance(checkpoint_blob, dict):
            raise ValueError("Invalid flow checkpoint payload")

        flow_state = checkpoint_blob.get("flow_state_dict")
        context_state = checkpoint_blob.get("context_encoder_state_dict")
        if not isinstance(flow_state, dict) or not isinstance(context_state, dict):
            raise ValueError("Invalid flow checkpoint: missing model states")

        flow.load_state_dict(flow_state)
        context_encoder.load_state_dict(context_state)

        pretrain_optimizer_state = checkpoint_blob.get("pretrain_optimizer_state")
        pretrain_scheduler_state = checkpoint_blob.get("pretrain_scheduler_state")
        optimizer_state = checkpoint_blob.get("optimizer_state")
        scheduler_state = checkpoint_blob.get("scheduler_state")

        if isinstance(pretrain_optimizer_state, dict):
            pretrain_optimizer.load_state_dict(pretrain_optimizer_state)
        if isinstance(pretrain_scheduler_state, dict):
            pretrain_scheduler.load_state_dict(pretrain_scheduler_state)
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)

        history_raw = checkpoint_blob.get("history")
        if isinstance(history_raw, list):
            history = [item for item in history_raw if isinstance(item, dict)]

        pretrain_epoch_done = int(checkpoint_blob.get("pretrain_epoch_done", 0))
        flow_epoch_done = int(checkpoint_blob.get("flow_epoch_done", 0))
        joint_epoch_done = int(checkpoint_blob.get("joint_epoch_done", 0))

        emit_event(
            "checkpoint_resumed",
            quiet=quiet,
            checkpoint_path=str(checkpoint_file),
            pretrain_epoch_done=int(pretrain_epoch_done),
            flow_epoch_done=int(flow_epoch_done),
            joint_epoch_done=int(joint_epoch_done),
            history_points=int(len(history)),
        )

    emit_event(
        "training_start",
        quiet=quiet,
        n_total_windows=int(dataset.z.shape[0]),
        n_full_windows=int(full_dataset.z.shape[0]),
        n_train_windows=int(x_train_n.shape[0]),
        n_val_windows=int(x_val_n.shape[0]),
        n_test_windows=int(x_test_n.shape[0]),
        n_healthy_recordings=int(
            len(set(filter_healthy_recording_ids(dataset.recording_id.tolist())))
        ),
        dropped_constant_features=int(dropped_feature_count),
        feature_dim=int(x_train_n.shape[1]),
        context_dim=int(context_dim),
        context_pretrain_epochs=int(pretrain_context_epochs),
        flow_train_epochs=int(epochs),
        joint_finetune_epochs=int(joint_finetune_epochs),
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=str(torch_device),
        determinism=determinism,
        checkpoint_path=(str(checkpoint_file) if checkpoint_file is not None else None),
        resume_checkpoint=bool(resume_checkpoint),
    )

    for epoch in range(int(pretrain_epoch_done) + 1, pretrain_context_epochs + 1):
        pre_loss = pretrain_context_encoder(
            context_encoder,
            x_train=x_train_n,
            batch_size=int(batch_size),
            device=torch_device,
            optimizer=pretrain_optimizer,
            grad_clip=float(grad_clip),
            seed=int(seed),
            epoch=int(epoch),
            temperature=float(contrastive_temperature),
        )
        pretrain_scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "phase": "context_pretrain",
                "contrastive_loss": float(pre_loss),
                "lr": float(pretrain_optimizer.param_groups[0]["lr"]),
            }
        )

        if epoch == 1 or epoch == pretrain_context_epochs or (epoch % log_every == 0):
            emit_event(
                "context_pretrain_epoch",
                quiet=quiet,
                epoch=int(epoch),
                epochs=int(pretrain_context_epochs),
                contrastive_loss=float(pre_loss),
                lr=float(pretrain_optimizer.param_groups[0]["lr"]),
            )

        pretrain_epoch_done = int(epoch)
        if (epoch % int(checkpoint_every) == 0) or (epoch == pretrain_context_epochs):
            _save_checkpoint(
                pretrain_done=pretrain_epoch_done,
                flow_done=flow_epoch_done,
                joint_done=joint_epoch_done,
                joint_optimizer_state=None,
                joint_scheduler_state=None,
            )

    for p in context_encoder.parameters():
        p.requires_grad = False

    # Track the best-val-NLL checkpoint in memory so we always save the
    # generalization-optimal model rather than the final (often overfit) epoch.
    _best_val_nll: float = float("inf")
    _best_flow_state: dict | None = None
    _best_ctx_state: dict | None = None
    _epochs_without_improvement: int = 0

    for epoch in range(int(flow_epoch_done) + 1, epochs + 1):
        train_loss = train_flow_epoch(
            flow,
            batches=flow_batches_with_context(
                x=x_train_n,
                context_encoder=context_encoder,
                batch_size=batch_size,
                device=torch_device,
                shuffle=True,
                seed=seed + epoch,
            ),
            optimizer=optimizer,
            grad_clip=grad_clip,
        )

        val_scores = collect_scores_with_context(
            flow,
            context_encoder,
            x_val_n,
            batch_size=batch_size,
            device=torch_device,
        )
        val_nll = float(np.mean(val_scores)) if val_scores.size else float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "phase": "flow_train",
                "train_nll": float(train_loss),
                "val_nll": float(val_nll),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        # Best-model tracking: keep a copy of the weights with the lowest val NLL.
        if np.isfinite(val_nll) and float(val_nll) < _best_val_nll:
            _best_val_nll = float(val_nll)
            _best_flow_state = {k: v.clone() for k, v in flow.state_dict().items()}
            _best_ctx_state = {
                k: v.clone() for k, v in context_encoder.state_dict().items()
            }
            _epochs_without_improvement = 0
        else:
            _epochs_without_improvement += 1

        if epoch == 1 or epoch == epochs or (epoch % log_every == 0):
            emit_event(
                "flow_train_epoch",
                quiet=quiet,
                epoch=int(epoch),
                epochs=int(epochs),
                train_nll=float(train_loss),
                val_nll=float(val_nll),
                best_val_nll=float(_best_val_nll),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        scheduler.step()

        flow_epoch_done = int(epoch)
        if (epoch % int(checkpoint_every) == 0) or (epoch == epochs):
            _save_checkpoint(
                pretrain_done=pretrain_epoch_done,
                flow_done=flow_epoch_done,
                joint_done=joint_epoch_done,
                joint_optimizer_state=None,
                joint_scheduler_state=None,
            )

        # Early stopping: halt if val NLL has not improved for `patience` epochs.
        if patience > 0 and _epochs_without_improvement >= int(patience):
            emit_event(
                "early_stopping",
                quiet=quiet,
                epoch=int(epoch),
                patience=int(patience),
                best_val_nll=float(_best_val_nll),
            )
            break

    # Restore the best generalising weights before final scoring and saving.
    if _best_flow_state is not None:
        flow.load_state_dict(_best_flow_state)
    if _best_ctx_state is not None:
        context_encoder.load_state_dict(_best_ctx_state)

    if joint_finetune_epochs > 0:
        for p in context_encoder.parameters():
            p.requires_grad = True

        joint_optimizer = torch.optim.Adam(
            list(flow.parameters()) + list(context_encoder.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            joint_optimizer,
            T_max=max(joint_finetune_epochs, 1),
        )

        if (
            checkpoint_file is not None
            and resume_checkpoint
            and checkpoint_file.exists()
        ):
            try:
                checkpoint_blob = torch.load(
                    checkpoint_file,
                    map_location=torch_device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint_blob = torch.load(checkpoint_file, map_location=torch_device)

            if isinstance(checkpoint_blob, dict):
                joint_optimizer_state = checkpoint_blob.get("joint_optimizer_state")
                joint_scheduler_state = checkpoint_blob.get("joint_scheduler_state")
                if isinstance(joint_optimizer_state, dict):
                    joint_optimizer.load_state_dict(joint_optimizer_state)
                if isinstance(joint_scheduler_state, dict):
                    joint_scheduler.load_state_dict(joint_scheduler_state)

        for epoch in range(int(joint_epoch_done) + 1, joint_finetune_epochs + 1):
            flow.train()
            context_encoder.train()
            loss_vals: list[float] = []
            flow_vals: list[float] = []
            contrastive_vals: list[float] = []

            for x_t in feature_batch_iterator(
                x_train_n,
                batch_size=batch_size,
                device=torch_device,
                shuffle=True,
                seed=seed + 9000 + epoch,
            ):
                if int(x_t.shape[0]) < 2:
                    continue

                joint_optimizer.zero_grad(set_to_none=True)
                c_t = context_encoder(x_t)
                flow_loss = -flow.log_likelihood(x_t, c_t).mean()

                view_a, view_b = augment_feature_views(x_t)
                emb_a = context_encoder(view_a)
                emb_b = context_encoder(view_b)
                contrastive = nt_xent_loss(
                    emb_a,
                    emb_b,
                    temperature=float(contrastive_temperature),
                )

                total = flow_loss + float(joint_ntxent_weight) * contrastive
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(flow.parameters()) + list(context_encoder.parameters()),
                    max_norm=float(grad_clip),
                )
                joint_optimizer.step()

                loss_vals.append(float(total.detach().cpu().item()))
                flow_vals.append(float(flow_loss.detach().cpu().item()))
                contrastive_vals.append(float(contrastive.detach().cpu().item()))

            if not loss_vals:
                raise ValueError("Joint fine-tuning produced no valid batches")

            joint_scheduler.step()

            val_scores_joint = collect_scores_with_context(
                flow,
                context_encoder,
                x_val_n,
                batch_size=batch_size,
                device=torch_device,
            )
            val_nll_joint = (
                float(np.mean(val_scores_joint))
                if val_scores_joint.size
                else float("nan")
            )

            history.append(
                {
                    "epoch": float(epoch),
                    "phase": "joint_finetune",
                    "train_total": float(np.mean(loss_vals)),
                    "train_nll": float(np.mean(flow_vals)),
                    "train_contrastive": float(np.mean(contrastive_vals)),
                    "val_nll": float(val_nll_joint),
                    "lr": float(joint_optimizer.param_groups[0]["lr"]),
                }
            )

            if epoch == 1 or epoch == joint_finetune_epochs or (epoch % log_every == 0):
                emit_event(
                    "joint_finetune_epoch",
                    quiet=quiet,
                    epoch=int(epoch),
                    epochs=int(joint_finetune_epochs),
                    train_total=float(np.mean(loss_vals)),
                    train_nll=float(np.mean(flow_vals)),
                    train_contrastive=float(np.mean(contrastive_vals)),
                    val_nll=float(val_nll_joint),
                    lr=float(joint_optimizer.param_groups[0]["lr"]),
                )

            joint_epoch_done = int(epoch)
            if (epoch % int(checkpoint_every) == 0) or (epoch == joint_finetune_epochs):
                _save_checkpoint(
                    pretrain_done=pretrain_epoch_done,
                    flow_done=flow_epoch_done,
                    joint_done=joint_epoch_done,
                    joint_optimizer_state=joint_optimizer.state_dict(),
                    joint_scheduler_state=joint_scheduler.state_dict(),
                )

    val_scores = collect_scores_with_context(
        flow,
        context_encoder,
        x_val_n,
        batch_size=batch_size,
        device=torch_device,
    )
    threshold = calibrate_threshold(val_scores.tolist(), percentile=score_percentile)

    test_scores = collect_scores_with_context(
        flow,
        context_encoder,
        x_test_n,
        batch_size=batch_size,
        device=torch_device,
    )
    full_scores = collect_scores_with_context(
        flow,
        context_encoder,
        x_full_n,
        batch_size=batch_size,
        device=torch_device,
    )

    val_context_vecs = collect_context_vectors(
        context_encoder,
        x_val_n,
        batch_size=batch_size,
        device=torch_device,
    )
    mode_cluster_stats = cluster_score_stats(
        val_context_vecs,
        val_scores,
        seed=int(seed),
        max_clusters=max(1, int(np.unique(dataset.recording_id[val_idx]).shape[0])),
    )

    n_full_anomalies = int(np.sum(full_scores > float(threshold)))
    full_anomaly_rate = (
        float(n_full_anomalies) / float(full_scores.shape[0])
        if full_scores.shape[0] > 0
        else 0.0
    )

    # Per-recording-class score statistics: separates healthy FPR from RF
    # detection rate so the two are not conflated in the aggregate metric.
    full_rids = full_dataset.recording_id.astype(str)
    recording_class_stats: dict[str, dict[str, float | int]] = {}
    for uid in sorted(np.unique(full_rids)):
        uid_mask = full_rids == uid
        uid_scores = full_scores[uid_mask]
        uid_n_flagged = int(np.sum(uid_scores > float(threshold)))
        uid_n = int(uid_scores.shape[0])
        recording_class_stats[uid] = {
            "n_windows": uid_n,
            "score_mean": float(np.mean(uid_scores)),
            "score_std": float(np.std(uid_scores)),
            "n_flagged": uid_n_flagged,
            "flag_rate": float(uid_n_flagged) / float(uid_n) if uid_n > 0 else 0.0,
        }
    healthy_mask = np.asarray(
        [is_healthy_recording_id(r) for r in full_rids], dtype=bool
    )
    healthy_full_scores = full_scores[healthy_mask]
    n_healthy_flagged = int(np.sum(healthy_full_scores > float(threshold)))
    healthy_fpr = (
        float(n_healthy_flagged) / float(healthy_full_scores.shape[0])
        if healthy_full_scores.shape[0] > 0
        else 0.0
    )

    emit_event(
        "training_done",
        quiet=quiet,
        threshold=float(threshold),
        score_percentile=float(score_percentile),
        n_full_anomalies=int(n_full_anomalies),
        full_anomaly_rate=float(full_anomaly_rate),
        n_history_points=int(len(history)),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / "flow.pt"
    threshold_path = output_dir / "threshold.json"
    summary_path = output_dir / "training_summary.json"

    torch.save(
        {
            "_meta": stamp_artifact_metadata(artifact_type="flow"),
            "flow_config": asdict(cfg),
            "state_dict": flow.state_dict(),
            "context_encoder_state_dict": context_encoder.state_dict(),
            "context_encoder_type": "lightweight_mlp",
            "feature_keep_indices": np.flatnonzero(keep_mask).astype(np.int64),
            "scaler_mean": mean.astype(np.float32),
            "scaler_std": std.astype(np.float32),
            "score_percentile": float(score_percentile),
            "threshold": float(threshold),
            "seed": int(seed),
        },
        artifact_path,
    )

    threshold_path.write_text(
        json.dumps(
            {
                "threshold": float(threshold),
                "percentile": float(score_percentile),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "n_total_windows": int(dataset.z.shape[0]),
        "n_full_windows": int(full_dataset.z.shape[0]),
        "n_train_windows": int(x_train_n.shape[0]),
        "n_val_windows": int(x_val_n.shape[0]),
        "n_test_windows": int(x_test_n.shape[0]),
        "n_healthy_recordings": int(
            len(set(filter_healthy_recording_ids(dataset.recording_id.tolist())))
        ),
        "feature_pipeline": {
            "input": "concat(z, c)",
            "constant_feature_filter": {
                "enabled": True,
                "std_threshold": 1e-2,
                "dropped_count": int(dropped_feature_count),
            },
            "scaler": "StandardScaler(train-only)",
            "context_encoder": "LightweightContextEncoder",
            "context_dim": int(context_dim),
            "contrastive_temperature": float(contrastive_temperature),
            "pretrain_context_epochs": int(pretrain_context_epochs),
            "joint_finetune_epochs": int(joint_finetune_epochs),
            "joint_ntxent_weight": float(joint_ntxent_weight),
        },
        "flow_config": asdict(cfg),
        "epochs": int(epochs),
        "trained_epochs": int(len(history)),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "best_val_nll": float(_best_val_nll),
        "optimizer": {
            "name": "Adam",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
            "t_max": int(epochs),
            "grad_clip": float(grad_clip),
        },
        "threshold": float(threshold),
        "score_percentile": float(score_percentile),
        "healthy_val_mode_cluster_score_stats": mode_cluster_stats,
        "healthy_val_score_mean": (
            float(np.mean(val_scores)) if val_scores.size else float("nan")
        ),
        "healthy_val_score_std": (
            float(np.std(val_scores)) if val_scores.size else float("nan")
        ),
        "healthy_test_score_mean": (
            float(np.mean(test_scores)) if test_scores.size else float("nan")
        ),
        "healthy_test_score_std": (
            float(np.std(test_scores)) if test_scores.size else float("nan")
        ),
        "full_score_mean": (
            float(np.mean(full_scores)) if full_scores.size else float("nan")
        ),
        "full_score_std": (
            float(np.std(full_scores)) if full_scores.size else float("nan")
        ),
        "n_full_anomalies": int(n_full_anomalies),
        "full_anomaly_rate": float(full_anomaly_rate),
        "healthy_fpr": float(healthy_fpr),
        "recording_class_stats": recording_class_stats,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "checkpoint": {
            "path": str(checkpoint_file) if checkpoint_file is not None else None,
            "resume_enabled": bool(resume_checkpoint),
            "checkpoint_every": int(checkpoint_every),
            "pretrain_epoch_done": int(pretrain_epoch_done),
            "flow_epoch_done": int(flow_epoch_done),
            "joint_epoch_done": int(joint_epoch_done),
        },
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return FlowTrainingArtifacts(
        artifact_path=artifact_path,
        threshold_path=threshold_path,
        summary_path=summary_path,
    )
