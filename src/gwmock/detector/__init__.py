"""Detector module for GWSim."""

from __future__ import annotations

from gwmock.detector.base import Detector
from gwmock.detector.utils import DEFAULT_DETECTOR_BASE_PATH, load_interferometer_config

__all__ = ["DEFAULT_DETECTOR_BASE_PATH", "Detector", "load_interferometer_config"]
