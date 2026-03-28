"""Sensitivity calibration utilities."""

from __future__ import annotations

from ..data import DataSegment
from ..exceptions import PreprocessingError


class SensitivityCalibrator:
    """Scale raw channels into physical-ish units using known sensitivities."""

    def __init__(
        self,
        mic_sensitivity_v_per_pa: float = 1.0,
        accel_sensitivity_v_per_g: float = 1.0,
    ) -> None:
        if mic_sensitivity_v_per_pa == 0:
            raise PreprocessingError("mic_sensitivity_v_per_pa must be non-zero")
        if accel_sensitivity_v_per_g == 0:
            raise PreprocessingError("accel_sensitivity_v_per_g must be non-zero")
        self._mic = mic_sensitivity_v_per_pa
        self._accel = accel_sensitivity_v_per_g

    def process(self, segment: DataSegment) -> DataSegment:
        mic_out = segment.mic_data / self._mic
        accel_out = segment.accel_data / self._accel

        return DataSegment(
            mic_data=mic_out,
            accel_data=accel_out,
            mic_sample_rate=segment.mic_sample_rate,
            accel_sample_rate=segment.accel_sample_rate,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            channel_names=segment.channel_names,
            metadata={
                **segment.metadata,
                "calibrated": True,
                "mic_unit": "Pa",
                "accel_unit": "g",
            },
        )
