"""Windowed segmentation for fixed-length analysis windows."""

from __future__ import annotations

from datetime import timedelta

from ..data import DataSegment
from ..exceptions import PreprocessingError


class WindowedSegmenter:
    """Split a DataSegment into overlapping windows."""

    def __init__(self, window_s: float = 5.0, overlap: float = 0.5) -> None:
        if window_s <= 0:
            raise PreprocessingError("window_s must be > 0")
        if not (0.0 <= overlap < 1.0):
            raise PreprocessingError("overlap must be in [0, 1)")

        self._window_s = window_s
        self._overlap = overlap

    def segment(self, segment: DataSegment) -> list[DataSegment]:
        mic_win = int(round(self._window_s * segment.mic_sample_rate))
        accel_win = int(round(self._window_s * segment.accel_sample_rate))

        if segment.n_mic_samples < mic_win:
            raise PreprocessingError("Segment is shorter than window size")

        mic_hop = max(1, int(round(mic_win * (1.0 - self._overlap))))
        accel_hop = max(1, int(round(accel_win * (1.0 - self._overlap))))

        windows: list[DataSegment] = []
        i = 0
        while True:
            mic_start = i * mic_hop
            mic_end = mic_start + mic_win
            accel_start = i * accel_hop
            accel_end = accel_start + accel_win

            if mic_end > segment.n_mic_samples or accel_end > segment.n_accel_samples:
                break

            win = DataSegment(
                mic_data=segment.mic_data[:, mic_start:mic_end].copy(),
                accel_data=segment.accel_data[:, accel_start:accel_end].copy(),
                mic_sample_rate=segment.mic_sample_rate,
                accel_sample_rate=segment.accel_sample_rate,
                start_time=segment.start_time
                + timedelta(seconds=mic_start / segment.mic_sample_rate),
                duration_s=self._window_s,
                channel_names=segment.channel_names,
                metadata={
                    **segment.metadata,
                    "window_index": i,
                    "window_s": self._window_s,
                    "overlap": self._overlap,
                },
            )
            windows.append(win)
            i += 1

        return windows
