"""Detector mixin for simulators."""

from __future__ import annotations

from pathlib import Path

from gwsim.detector.base import Detector
from gwsim.detector.utils import DEFAULT_DETECTOR_BASE_PATH


class DetectorMixin:  # pylint: disable=too-few-public-methods
    """Mixin class to add detector information to simulators."""

    def __init__(self, detectors: list[str] | None = None, **kwargs):  # pylint: disable=unused-argument
        """Initialize the DetectorMixin.

        Args:
            detectors (list[str] | None): List of detector names. If None, use all available detectors.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.detectors = detectors

    @property
    def detectors(self) -> list[str] | list[Detector] | None:
        """Get the list of detectors.

        Returns:
            List of detector names or Detector instances, or None if not set.
        """
        return self._detectors

    @detectors.setter
    def detectors(self, value: list[str] | None) -> None:
        if value is None:
            self._detectors = None
        else:
            # Check whether the label is a file name
            detectors = []
            for det in value:
                det_path = Path(det)
                if det_path.is_file() or (DEFAULT_DETECTOR_BASE_PATH / det_path).is_file():
                    detectors.append(Detector(config_file=det))
                else:
                    detectors.append(Detector(name=det))
            self._detectors = detectors

    @property
    def metadata(self) -> dict:
        """Get metadata including detector information.

        Returns:
            Dictionary containing the list of detectors.
        """
        metadata = {
            "detectors": self.detectors,
        }
        return metadata
