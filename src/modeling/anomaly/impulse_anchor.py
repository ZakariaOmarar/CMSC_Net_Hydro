"""Impulse + spectral anchor features for the conditional V3 flow (RQ2).

Motivation: the SSL/CMA encoder optimises away the impulsive/spectral cues a
knock produces (the flow-input embedding cannot even predict crest factor;
Ridge R^2 ~ 0).  Rather than abandon context-conditioning (RQ2's whole point),
we *augment* the conditional-flow input with a small, fixed vector of
condition-monitoring statistics so the conditional density model can see the
anomaly while still being conditioned on the operating context ``c_t``.

The anchor is computed from the SAME windowed log-mel + CWT feature tensors the
V2 encoder consumes, so it is identical at every site that builds the flow input
(V3 training, V3 eval, and the V4 ``x_for_v3`` used for gating) — no raw
re-windowing, no recording-id / time-coordinate drift.

Acoustic features are ``(N_mic, 2, F, T)`` — channel 0 = log-mel, channel 1 =
CWT (transient-rich).  Vibration features are ``(N_vib, C, T)``.  Stats are
pooled over sensors so the anchor is sensor-count agnostic.
"""
from __future__ import annotations

import numpy as np

# Order is fixed and must never change (persisted standardization stats and the
# flow input dimension depend on it).
ANCHOR_FEATURES = (
    "ac_crest", "ac_kurtosis", "ac_flatness", "ac_centroid",
    "vib_crest", "vib_kurtosis", "vib_energy_std", "vib_peak_over_med",
)
N_ANCHOR = len(ANCHOR_FEATURES)


def _impulse_on_envelope(env: np.ndarray) -> tuple[float, float]:
    """crest factor + excess kurtosis of a 1-D energy envelope."""
    env = np.asarray(env, dtype=np.float64)
    if env.size < 4 or not np.any(env):
        return 0.0, 0.0
    rms = np.sqrt(np.mean(env * env)) + 1e-12
    crest = float(env.max() / rms)
    mu, sd = env.mean(), env.std() + 1e-12
    kurt = float(np.mean(((env - mu) / sd) ** 4) - 3.0)
    return crest, kurt


def _anchor_one(ac: np.ndarray, vib: np.ndarray) -> np.ndarray:
    """Anchor vector for ONE window. ac=(N_mic,2,F,T), vib=(N_vib,C,T)."""
    ac = np.asarray(ac, dtype=np.float64)
    vib = np.asarray(vib, dtype=np.float64)
    # --- acoustic: impulse on the CWT channel, spectral on the log-mel channel ---
    cwt = ac[:, 1] if ac.ndim == 4 and ac.shape[1] >= 2 else ac.reshape(ac.shape[0], -1, ac.shape[-1])[:, 0]
    cwt_env = cwt.mean(axis=tuple(range(cwt.ndim - 1)))          # -> (T,)
    ac_crest, ac_kurt = _impulse_on_envelope(cwt_env)
    logmel = ac[:, 0] if ac.ndim == 4 else ac
    spec = np.clip(logmel.mean(axis=0), 1e-12, None)             # (F,T) avg over mics
    spec_f = spec.mean(axis=-1)                                  # (F,) avg over time
    spec_f = spec_f / (spec_f.sum() + 1e-12)
    F = spec_f.shape[0]
    freqs = np.arange(F, dtype=np.float64)
    ac_centroid = float((freqs * spec_f).sum() / (F - 1 + 1e-12))      # normalised [0,1]
    ac_flatness = float(np.exp(np.mean(np.log(spec_f + 1e-12))) / (spec_f.mean() + 1e-12))
    # --- vibration: impulse + energy spread on the per-time energy envelope ---
    vib_env = (vib ** 2).mean(axis=tuple(range(vib.ndim - 1)))   # -> (T,)
    vib_crest, vib_kurt = _impulse_on_envelope(np.sqrt(vib_env))
    vib_estd = float(np.sqrt(vib_env).std())
    aw = np.sqrt(vib_env)
    vib_pom = float(aw.max() / (np.median(aw) + 1e-12)) if aw.size else 0.0
    return np.array([ac_crest, ac_kurt, ac_flatness, ac_centroid,
                     vib_crest, vib_kurt, vib_estd, vib_pom], dtype=np.float64)


def impulse_spectral_anchor(ac_feat, vib_feat) -> np.ndarray:
    """Batched anchor.  ac_feat=(B,N_mic,2,F,T), vib_feat=(B,N_vib,C,T) -> (B,N_ANCHOR).

    Accepts numpy arrays or torch tensors (converted to numpy).
    """
    ac = np.asarray(_to_numpy(ac_feat), dtype=np.float64)
    vib = np.asarray(_to_numpy(vib_feat), dtype=np.float64)
    return np.stack([_anchor_one(ac[i], vib[i]) for i in range(ac.shape[0])], axis=0)


def _to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def append_anchor(x, ac_feat, vib_feat, anchor_norm):
    """Concatenate the standardized impulse+spectral anchor to a pooled flow input.

    THE single source of truth used by every V3-flow scoring site (V3 trainer,
    RQ2 eval, V4 gating ``x_for_v3``, sliding-window inference, ablations) so the
    anchor is computed and standardized identically everywhere.

    Args:
      x: pooled flow input, ``(B, d)`` torch.Tensor or np.ndarray.
      ac_feat, vib_feat: the SAME windowed log-mel+CWT feature tensors the V2
        encoder consumed for those windows, batched ``(B, ...)``.
      anchor_norm: ``(mean, std)`` healthy standardization from V3 training, or
        ``None`` to pass ``x`` through unchanged (anchor disabled).
    """
    if anchor_norm is None:
        return x
    mean, std = anchor_norm
    a = (impulse_spectral_anchor(ac_feat, vib_feat) - mean) / std
    if hasattr(x, "detach"):  # torch.Tensor
        import torch
        return torch.cat([x, torch.as_tensor(a, dtype=x.dtype, device=x.device)], dim=1)
    return np.concatenate([np.asarray(x), a.astype(np.asarray(x).dtype)], axis=1)


__all__ = ["impulse_spectral_anchor", "append_anchor", "ANCHOR_FEATURES", "N_ANCHOR"]
