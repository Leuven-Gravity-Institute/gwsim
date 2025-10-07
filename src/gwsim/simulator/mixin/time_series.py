"""Mixins for simulator classes providing optional functionality."""

from __future__ import annotations


class TimeSeriesMixin:  # pylint: disable=too-few-public-methods
    """Mixin providing timing and duration management.

    This mixin adds time-based parameters commonly used
    in gravitational wave simulations.
    """

    def __init__(self, duration: float | None = None, sampling_frequency: float | None = None):
        """Initialize timing parameters.

        Args:
            duration: Duration of simulation in seconds.
            sampling_frequency: Sampling frequency in Hz.
            **kwargs: Additional arguments passed to parent classes.
        """
        self.duration = duration
        self.sampling_frequency = sampling_frequency

    @property
    def metadata(self) -> dict:
        """Get metadata including timing information.

        Returns:
            Dictionary containing timing parameters and other metadata.
        """
        metadata = {
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
        }
        return metadata
