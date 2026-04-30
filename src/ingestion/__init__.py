"""Ingestion utilities for thesis WAV + vibration CSV datasets and Gantner UDBF files."""

from .adapters import WavVibrationAdapter
from .loader import SegmentLoader
from .scanner import RecordingGroup, RecordingScanner
from .udbf_reader import UDBFFile, concat_udbf, read_udbf, read_udbf_folder
from .illwerke_loader import (
    IllwerkeCampaign,
    load_rms_campaign,
    load_allg_campaign,
    load_campaign,
)

__all__ = [
    "RecordingGroup",
    "RecordingScanner",
    "WavVibrationAdapter",
    "SegmentLoader",
    "UDBFFile",
    "read_udbf",
    "read_udbf_folder",
    "concat_udbf",
    "IllwerkeCampaign",
    "load_rms_campaign",
    "load_allg_campaign",
    "load_campaign",
]
