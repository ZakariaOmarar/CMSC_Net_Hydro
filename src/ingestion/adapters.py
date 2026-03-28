"""Format adapters for thesis multimodal recordings (WAV + vibration CSV)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..config.constants import (
    ACCEL_COUNT,
    ACCEL_SAMPLE_RATE_TARGET,
    MIC_SAMPLE_RATE,
)
from ..data import DataSegment
from ..exceptions import IngestionError

_TIME_KEYS = ("esp_time_us", "timestamp_us", "time_us", "timestamp", "time")
_AMPLITUDE_KEYS = ("amplitude", "amp", "fft_amplitude", "peak_amplitude")
_FREQUENCY_KEYS = ("frequency", "freq", "dominant_frequency", "peak_frequency")


class WavVibrationAdapter:
    """Read one recording directory into a DataSegment.

    Expected files by default:
    - `recorded_*.wav` microphone files
    - `vibration_*.csv` vibration files with timestamp, amplitude, frequency columns
    """

    def __init__(
        self,
        expected_mic_count: int | None = None,
        expected_accel_count: int = ACCEL_COUNT,
        mic_glob: str = "recorded_*.wav",
        vibration_glob: str = "vibration_*.csv",
        accel_target_sr: int = ACCEL_SAMPLE_RATE_TARGET,
        allowed_mic_counts: tuple[int, ...] = (4, 9),
    ) -> None:
        self._expected_mic_count = expected_mic_count
        self._expected_accel_count = expected_accel_count
        self._mic_glob = mic_glob
        self._vibration_glob = vibration_glob
        self._accel_target_sr = accel_target_sr
        self._allowed_mic_counts = tuple(
            sorted(set(int(c) for c in allowed_mic_counts))
        )

        if self._expected_mic_count is None and not self._allowed_mic_counts:
            raise IngestionError(
                "allowed_mic_counts must be non-empty when expected_mic_count is None"
            )

        if self._expected_mic_count is not None and int(self._expected_mic_count) <= 0:
            raise IngestionError("expected_mic_count must be positive")

    def read_recording_directory(self, recording_dir: Path) -> DataSegment:
        """Read one recording folder and return a synchronized DataSegment."""
        recording_dir = Path(recording_dir)
        if not recording_dir.exists() or not recording_dir.is_dir():
            raise IngestionError(f"Recording directory not found: {recording_dir}")

        mic_files = sorted(recording_dir.glob(self._mic_glob))
        vibration_files = sorted(recording_dir.glob(self._vibration_glob))
        return self.read_recording_files(
            recording_dir=recording_dir,
            mic_files=mic_files,
            vibration_files=vibration_files,
            recording_id=recording_dir.name,
        )

    def read_recording_files(
        self,
        recording_dir: Path,
        mic_files: list[Path] | tuple[Path, ...],
        vibration_files: list[Path] | tuple[Path, ...],
        recording_id: str | None = None,
    ) -> DataSegment:
        """Read one recording from explicit mic and vibration file lists."""
        recording_dir = Path(recording_dir)

        mic_data, mic_sr, mic_paths = self._read_wav_channels(mic_files, recording_dir)
        vib_amp, vib_freq, vib_sr_raw, vib_paths = self._read_vibration_channels(
            vibration_files,
            recording_dir,
        )

        mic_duration = mic_data.shape[1] / mic_sr
        vib_duration = vib_amp.shape[1] / vib_sr_raw
        common_duration = min(mic_duration, vib_duration)

        mic_samples = int(round(common_duration * mic_sr))
        mic_data = mic_data[:, :mic_samples]

        actual_duration = mic_samples / mic_sr
        accel_samples = max(1, int(round(actual_duration * self._accel_target_sr)))

        accel_data = _resample_channels(
            vib_amp, vib_sr_raw, accel_samples, self._accel_target_sr
        )
        vib_freq_resampled = _resample_channels(
            vib_freq, vib_sr_raw, accel_samples, self._accel_target_sr
        )

        metadata = {
            "source": "wav_vibration_csv",
            "recording_dir": str(recording_dir),
            "recording_id": (
                recording_id if recording_id is not None else recording_dir.name
            ),
            "mic_files": [str(p) for p in mic_paths],
            "vibration_files": [str(p) for p in vib_paths],
            "mic_sample_rate_original": mic_sr,
            "vibration_sample_rate_raw": float(vib_sr_raw),
            "vibration_frequencies": vib_freq_resampled,
        }

        return DataSegment.from_arrays(
            mic_data=mic_data,
            accel_data=accel_data,
            start_time=datetime.now(timezone.utc),
            mic_sr=mic_sr,
            accel_sr=self._accel_target_sr,
            metadata=metadata,
        )

    def _validate_mic_file_count(self, found_count: int, recording_dir: Path) -> None:
        if self._expected_mic_count is not None:
            if found_count != self._expected_mic_count:
                raise IngestionError(
                    "Expected "
                    f"{self._expected_mic_count} WAV files, found {found_count} in {recording_dir}"
                )
            return

        if found_count not in self._allowed_mic_counts:
            raise IngestionError(
                "Expected WAV count in "
                f"{self._allowed_mic_counts}, found {found_count} in {recording_dir}"
            )

    def _read_wav_channels(
        self,
        wav_paths: list[Path] | tuple[Path, ...],
        recording_dir: Path,
    ) -> tuple[np.ndarray, int, list[Path]]:
        wav_paths = sorted(Path(p) for p in wav_paths)
        self._validate_mic_file_count(len(wav_paths), recording_dir)

        channels: list[np.ndarray] = []
        sample_rate: int | None = None

        for wav_path in wav_paths:
            sr, data = wavfile.read(wav_path)
            if sample_rate is None:
                sample_rate = int(sr)
            elif int(sr) != sample_rate:
                raise IngestionError(
                    f"Sample-rate mismatch: {wav_path.name} has {sr}, expected {sample_rate}"
                )

            if data.ndim != 1:
                raise IngestionError(
                    f"WAV must be mono, got shape {data.shape} in {wav_path.name}"
                )

            if data.dtype == np.int16:
                arr = data.astype(np.float64) / 32768.0
            elif np.issubdtype(data.dtype, np.integer):
                max_abs = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                arr = data.astype(np.float64) / float(max_abs)
            else:
                arr = data.astype(np.float64)

            channels.append(arr)

        assert sample_rate is not None
        if sample_rate != MIC_SAMPLE_RATE:
            # Keep strict default expectation for thesis recording protocol.
            raise IngestionError(
                f"Expected mic sample rate {MIC_SAMPLE_RATE} Hz, got {sample_rate} Hz"
            )

        min_len = min(len(ch) for ch in channels)
        mic_data = np.stack([ch[:min_len] for ch in channels])
        return mic_data, sample_rate, wav_paths

    def _read_vibration_channels(
        self,
        csv_paths: list[Path] | tuple[Path, ...],
        recording_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, float, list[Path]]:
        csv_paths = sorted(Path(p) for p in csv_paths)
        if len(csv_paths) != self._expected_accel_count:
            raise IngestionError(
                f"Expected {self._expected_accel_count} vibration CSV files, found {len(csv_paths)} in {recording_dir}"
            )

        amp_channels: list[np.ndarray] = []
        freq_channels: list[np.ndarray] = []
        timestamp_channels: list[np.ndarray] = []

        for csv_path in csv_paths:
            amps, freqs, timestamps = _read_vibration_csv(csv_path)
            amp_channels.append(amps)
            freq_channels.append(freqs)
            timestamp_channels.append(timestamps)

        min_len = min(len(ch) for ch in amp_channels)
        amp_data = np.stack([ch[:min_len] for ch in amp_channels])
        freq_data = np.stack([ch[:min_len] for ch in freq_channels])

        ts0 = timestamp_channels[0][:min_len]
        if len(ts0) > 1:
            dt_us = float(np.mean(np.diff(ts0)))
            vib_sr_raw = (
                1_000_000.0 / dt_us if dt_us > 0 else float(self._accel_target_sr)
            )
        else:
            vib_sr_raw = float(self._accel_target_sr)

        if vib_sr_raw <= 0:
            vib_sr_raw = float(self._accel_target_sr)

        return amp_data, freq_data, vib_sr_raw, csv_paths


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def _read_vibration_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise IngestionError(f"CSV has no header: {path}")

        fieldnames = [f.strip() for f in reader.fieldnames]
        time_col = _pick_column(fieldnames, _TIME_KEYS)
        amp_col = _pick_column(fieldnames, _AMPLITUDE_KEYS)
        freq_col = _pick_column(fieldnames, _FREQUENCY_KEYS)

        if amp_col is None or freq_col is None:
            raise IngestionError(
                f"CSV {path.name} must include amplitude and frequency columns"
            )

        timestamps: list[int] = []
        amplitudes: list[float] = []
        frequencies: list[float] = []

        for i, row in enumerate(reader):
            try:
                amplitudes.append(float(row[amp_col]))
                frequencies.append(float(row[freq_col]))
                if time_col is not None and row.get(time_col, "") != "":
                    raw_t = float(row[time_col])
                    t_us = int(raw_t if "us" in time_col else raw_t * 1_000_000.0)
                else:
                    t_us = i * 250_000
                timestamps.append(t_us)
            except (TypeError, ValueError) as exc:
                raise IngestionError(
                    f"Invalid numeric row in {path.name}: {exc}"
                ) from exc

    if len(amplitudes) == 0:
        raise IngestionError(f"CSV {path.name} contains no data rows")

    return (
        np.asarray(amplitudes, dtype=np.float64),
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(timestamps, dtype=np.int64),
    )


def _resample_channels(
    data: np.ndarray, src_rate: float, n_out: int, dst_rate: int
) -> np.ndarray:
    n_channels, n_in = data.shape
    t_in = np.arange(n_in) / src_rate
    t_out = np.arange(n_out) / dst_rate
    t_out = np.clip(t_out, t_in[0], t_in[-1])

    out = np.empty((n_channels, n_out), dtype=np.float64)
    for ch in range(n_channels):
        out[ch] = np.interp(t_out, t_in, data[ch])
    return out
