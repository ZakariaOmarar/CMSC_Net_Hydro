"""Feature contracts and multimodal extractors."""

from .acoustic_representations import (
    build_cwt_mfcc_encoder_input,
    compute_cwt_scalogram,
    compute_cwt_scalogram_stack,
    compute_log_stft_spectrogram,
    compute_mfcc_stack,
    compute_mfcc_with_deltas,
    compute_stft_stack,
)
from .cross_channel import CrossChannelExtractor
from .feature_frame import FeatureFrame
from .frequency_domain import FrequencyDomainExtractor
from .time_domain import TimeDomainExtractor
from .vibration_frequency import VibrationFrequencyExtractor
from .vibration_envelope import VibrationEnvelopeExtractor

__all__ = [
    "FeatureFrame",
    "compute_cwt_scalogram",
    "compute_cwt_scalogram_stack",
    "compute_mfcc_with_deltas",
    "compute_mfcc_stack",
    "build_cwt_mfcc_encoder_input",
    "compute_log_stft_spectrogram",
    "compute_stft_stack",
    "TimeDomainExtractor",
    "FrequencyDomainExtractor",
    "VibrationFrequencyExtractor",
    "VibrationEnvelopeExtractor",
    "CrossChannelExtractor",
]
