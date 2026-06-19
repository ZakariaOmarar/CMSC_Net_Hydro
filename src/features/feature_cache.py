"""Disk memoization for the expensive, deterministic feature extractors.

The CWT acoustic stack (`compute_encoder_input_stack`) and the vibration stack
(`compute_vibration_input_stack`) are *pure functions* of their input waveform
and a handful of scalar parameters.  Yet the full pipeline recomputes them many
times over: V1 acoustic, V1 vibration, V2, the V2 A1 ablation (identical
features — `drop_vibration` is applied later, in the forward pass), and the V4
sample builders all re-extract the same per-recording features.  Each extraction
of a 96-mel/64-scale CWT over a multi-minute recording costs seconds; the A1
ablation alone re-pays the entire V2 feature bill.

This module caches those outputs to disk **only when** the environment variable
``HYDRO_FEATURE_CACHE_DIR`` is set, so the default behaviour is byte-identical to
no caching.  Crucially it is *result-neutral*: the cache key is a SHA-256 over

  * a cache-format version constant,
  * the function's qualified name,
  * the exact bytes (shape + dtype + raw buffer) of every ndarray argument, and
  * the repr of every scalar / tuple / dict argument,

so any change to the input waveform or any parameter — anything that could
change the output — produces a different key and forces a recompute.  A stale
hit is therefore impossible short of a hash collision; the cached value is the
function's own output, never an approximation.

Enable it for a run with, e.g.::

    HYDRO_FEATURE_CACHE_DIR=.feature_cache python -m src.modeling.orchestration.full_run
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Bump when the on-disk format or the set of cached functions' semantics change,
# to invalidate every existing cache entry.
_CACHE_VERSION = 1


def _cache_dir() -> Path | None:
    """Return the cache directory from the env var, or None (caching off)."""
    raw = os.environ.get("HYDRO_FEATURE_CACHE_DIR")
    if not raw:
        return None
    return Path(raw)


def _hash_obj(h: "hashlib._Hash", obj: object) -> None:
    """Fold one argument into the running hash, exactly and recursively."""
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(b"ndarray|")
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    elif isinstance(obj, (tuple, list)):
        h.update(b"seq|")
        for x in obj:
            _hash_obj(h, x)
    elif isinstance(obj, dict):
        h.update(b"dict|")
        for k in sorted(obj, key=repr):
            h.update(repr(k).encode())
            _hash_obj(h, obj[k])
    else:
        h.update(("scalar|" + repr(obj)).encode())


def _make_key(qualname: str, args: tuple, kwargs: dict) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}|{qualname}|".encode())
    for a in args:
        _hash_obj(h, a)
    h.update(b"||kwargs||")
    for k in sorted(kwargs):
        h.update(repr(k).encode())
        _hash_obj(h, kwargs[k])
    return h.hexdigest()


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write `arr` to `path` atomically so a crash never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        np.save(tmp, arr, allow_pickle=False)
        # np.save appends .npy to a path without one; normalise then replace.
        tmp_npy = tmp + ".npy" if not tmp.endswith(".npy") else tmp
        os.replace(tmp_npy, path)
    except Exception:
        for p in (tmp, tmp + ".npy"):
            try:
                os.remove(p)
            except OSError:
                pass
        raise


def disk_cached_feature(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Memoize a pure ``(*arrays, **params) -> np.ndarray`` extractor to disk.

    No-op unless ``HYDRO_FEATURE_CACHE_DIR`` is set.  Only ``np.ndarray`` return
    values are cached (a `None` return — e.g. a too-short signal — is passed
    through uncached).  A corrupt cache file is treated as a miss and overwritten.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cdir = _cache_dir()
        if cdir is None:
            return fn(*args, **kwargs)
        key = _make_key(fn.__qualname__, args, kwargs)
        path = cdir / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path, allow_pickle=False)
            except Exception:
                pass  # corrupt / partial → recompute and overwrite
        out = fn(*args, **kwargs)
        if isinstance(out, np.ndarray):
            try:
                _atomic_save(path, out)
            except Exception:
                pass  # cache is best-effort; never fail the computation
        return out

    return wrapper


__all__ = ["disk_cached_feature"]
