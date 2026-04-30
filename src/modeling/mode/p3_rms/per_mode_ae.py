"""Layer 3 — Per-mode autoencoders for anomaly scoring.

One small 1D causal conv autoencoder per steady mode (ST, TU, PU, PH).
Trained on 60-second windows of oracle-HIGH RMS features for that mode only.
Output: per-second MSE reconstruction error → used as anomaly score in Layer 5.

Architecture (each AE, ~30k params)
------------------------------------
Encoder: CausalConv1d(D→32, k=3) → GELU → CausalConv1d(32→16, k=3, dilation=2) → GELU → mean pool
Decoder: Linear(16→32) → GELU → CausalConv1d(32→D, k=3) → output
D = 33 (feature dimension from features.py)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("torch not installed. Per-mode autoencoders unavailable. "
                  "Install via: pip install torch --index-url https://download.pytorch.org/whl/cpu")

MODES = ["ST", "TU", "PU", "PH"]
WINDOW_S = 60


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def _causal_pad(kernel_size: int, dilation: int) -> int:
    return (kernel_size - 1) * dilation


if _HAS_TORCH:
    class _CausalConv1d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
            super().__init__()
            self.pad = _causal_pad(kernel_size, dilation)
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = nn.functional.pad(x, (self.pad, 0))
            return self.conv(x)

    class ModeAutoencoder(nn.Module):
        """Causal 1D conv autoencoder for a single operating mode."""

        def __init__(self, d_feat: int = 33, hidden: int = 32, bottleneck: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(
                _CausalConv1d(d_feat, hidden, kernel_size=3, dilation=1),
                nn.GELU(),
                _CausalConv1d(hidden, bottleneck, kernel_size=3, dilation=2),
                nn.GELU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_feat),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, D, T)
            z = self.encoder(x)           # (B, bottleneck, T)
            z = z.permute(0, 2, 1)        # (B, T, bottleneck)
            out = self.decoder(z)          # (B, T, D)
            return out.permute(0, 2, 1)   # (B, D, T)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _extract_windows(
    X_feat: np.ndarray,
    mask: np.ndarray,
    window_s: int = WINDOW_S,
    stride: int = 15,
) -> np.ndarray:
    """Extract non-overlapping (stride) windows from masked timesteps.

    Returns (N_windows, D, window_s) float32.
    """
    T, D = X_feat.shape
    windows = []
    t = 0
    while t + window_s <= T:
        # Use window only if the majority of its timesteps are in mask
        if np.mean(mask[t:t + window_s]) > 0.8:
            windows.append(X_feat[t:t + window_s].T.astype(np.float32))  # (D, window_s)
        t += stride
    if not windows:
        return np.empty((0, D, window_s), dtype=np.float32)
    return np.stack(windows, axis=0)  # (N, D, window_s)


def train_per_mode_ae(
    X_feat: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_confidence: np.ndarray,
    *,
    d_feat: int = 33,
    hidden: int = 32,
    bottleneck: int = 16,
    window_s: int = WINDOW_S,
    window_stride: int = 15,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    random_seed: int = 42,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train one autoencoder per mode.

    Returns dict mapping mode name → trained ModeAutoencoder (or None if not enough data).
    """
    if not _HAS_TORCH:
        raise ImportError("torch required for per-mode autoencoders")

    from src.modeling.mode.p1_physics.oracle import CONFIDENCE_CODE, LABEL_CODE

    torch.manual_seed(random_seed)
    high_code = CONFIDENCE_CODE["HIGH"]
    aes: dict[str, Any] = {}

    for mode in MODES:
        code = LABEL_CODE[mode]
        mask = (oracle_labels == code) & (oracle_confidence == high_code)
        windows = _extract_windows(X_feat, mask, window_s=window_s, stride=window_stride)

        if len(windows) < 20:
            warnings.warn(f"Mode {mode}: only {len(windows)} windows — skipping AE training")
            aes[mode] = None
            continue

        if not quiet:
            print(f"  [{mode}] {len(windows)} windows → training AE ...")

        # Per-feature standardization so all features contribute equally to MSE.
        # Computed from training windows; stored on the AE for inference.
        feat_mean = windows.mean(axis=(0, 2))  # (D,)
        feat_std  = windows.std(axis=(0, 2)) + 1e-6  # (D,)
        windows_norm = (windows - feat_mean[:, None]) / feat_std[:, None]

        ae = ModeAutoencoder(d_feat=d_feat, hidden=hidden, bottleneck=bottleneck)
        ae.feat_mean = feat_mean.astype(np.float32)
        ae.feat_std  = feat_std.astype(np.float32)
        optimizer = optim.Adam(ae.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
        loss_fn = nn.MSELoss()

        rng = np.random.default_rng(random_seed)
        ae.train()
        for epoch in range(epochs):
            order = rng.permutation(len(windows_norm))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(windows_norm), batch_size):
                batch_idx = order[i:i + batch_size]
                x = torch.from_numpy(windows_norm[batch_idx])
                optimizer.zero_grad()
                x_hat = ae(x)
                loss = loss_fn(x_hat, x)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            scheduler.step()
            if not quiet and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{epochs}  loss={epoch_loss/n_batches:.4f}")

        ae.eval()
        aes[mode] = ae

    return aes


# ---------------------------------------------------------------------------
# Inference: per-second reconstruction error
# ---------------------------------------------------------------------------


def compute_reconstruction_errors(
    aes: dict[str, Any],
    X_feat: np.ndarray,
    smoothed_labels: np.ndarray,
    *,
    window_s: int = WINDOW_S,
    mode_label_map: dict[int, str] | None = None,
) -> np.ndarray:
    """Compute per-second MSE reconstruction error using the mode's AE.

    Processes each contiguous mode segment in non-overlapping window_s blocks.
    All seconds within a block share the same block-average MSE, which is
    directly comparable to the training loss and gives independent observations.

    Returns (T,) float32 — reconstruction error array (NaN for unscored seconds).
    """
    if not _HAS_TORCH:
        raise ImportError("torch required")

    from src.modeling.mode.p1_physics.oracle import LABEL_CODE

    T, D = X_feat.shape
    errors = np.full(T, np.nan, dtype=np.float32)

    for mode in MODES:
        ae = aes.get(mode)
        if ae is None:
            continue

        mode_code = LABEL_CODE[mode]
        mode_idx = np.where(smoothed_labels == mode_code)[0]
        if len(mode_idx) == 0:
            continue

        ae.eval()
        feat_mean = getattr(ae, "feat_mean", np.zeros(D, dtype=np.float32))
        feat_std  = getattr(ae, "feat_std",  np.ones(D, dtype=np.float32))

        # Split mode_idx into contiguous segments (mode may be interrupted by other modes)
        breaks = np.where(np.diff(mode_idx) > 1)[0] + 1
        segments = np.split(mode_idx, breaks)

        with torch.no_grad():
            for seg_t in segments:
                # Skip first and last window_s seconds: they may overlap with mode transitions
                # (oracle fires with a lag, so the physical transition begins before the label changes).
                # Only interior blocks with clean within-mode features are scored.
                for i in range(window_s, len(seg_t) - window_s, window_s):
                    block_t = seg_t[i:i + window_s]
                    if len(block_t) < window_s:
                        continue
                    window_raw = X_feat[block_t].T.astype(np.float32)  # (D, window_s)
                    window_norm = (window_raw - feat_mean[:, None]) / feat_std[:, None]
                    x = torch.from_numpy(window_norm[None])  # (1, D, window_s)
                    x_hat = ae(x)
                    # Window-average MSE — same metric as training loss, independent across blocks
                    err = float(torch.mean((x_hat - x) ** 2))
                    errors[block_t] = err  # all 60 seconds in the block share this value

    return errors


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_aes(aes: dict[str, Any], output_dir: str | Path) -> None:
    if not _HAS_TORCH:
        raise ImportError("torch required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, ae in aes.items():
        if ae is not None:
            checkpoint = {
                "state_dict": ae.state_dict(),
                "feat_mean": getattr(ae, "feat_mean", None),
                "feat_std":  getattr(ae, "feat_std",  None),
            }
            torch.save(checkpoint, output_dir / f"ae_{mode}.pt")


def load_aes(
    output_dir: str | Path,
    *,
    d_feat: int = 33,
    hidden: int = 32,
    bottleneck: int = 16,
) -> dict[str, Any]:
    if not _HAS_TORCH:
        raise ImportError("torch required")
    output_dir = Path(output_dir)
    aes: dict[str, Any] = {}
    for mode in MODES:
        path = output_dir / f"ae_{mode}.pt"
        if path.exists():
            ae = ModeAutoencoder(d_feat=d_feat, hidden=hidden, bottleneck=bottleneck)
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                ae.load_state_dict(checkpoint["state_dict"])
                if checkpoint.get("feat_mean") is not None:
                    ae.feat_mean = checkpoint["feat_mean"]
                    ae.feat_std  = checkpoint["feat_std"]
            else:
                ae.load_state_dict(checkpoint)  # legacy format
            ae.eval()
            aes[mode] = ae
        else:
            aes[mode] = None
    return aes
