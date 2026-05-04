"""Base class for signal simulators backed by ``gwmock_signal``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.population_reader import PopulationReaderMixin
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.signal.adapter import SignalAdapter
from gwmock.simulator.base import Simulator

logger = logging.getLogger("gwmock")


class SignalSimulator(PopulationReaderMixin, TimeSeriesMixin, Simulator):
    """Base class for signal simulators."""

    def __init__(  # noqa: PLR0913
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_sort_by: str | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        waveform_model: str | Callable[..., Any] | None = None,
        waveform_arguments: dict[str, Any] | None = None,
        start_time: int = 0,
        duration: float = 1024,
        sampling_frequency: float = 4096,
        max_samples: int | None = None,
        dtype: type = np.float64,
        detectors: list[str] | None = None,
        minimum_frequency: float = 5,
        source_type: str = "bbh",
        earth_rotation: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the base signal simulator.

        Args:
            population_file: Path to the population file.
            population_parameter_name_mapper: Dict mapping population column names to simulator parameter names.
            population_sort_by: Column name to sort the population by.
            population_cache_dir: Directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300.
            waveform_model: Name (from registry), custom callable, or ``None`` for
                ``IMRPhenomXPHM``. If a callable, it is registered on the backend's
                ``WaveformFactory`` and invoked with merged injection parameters,
                ``waveform_arguments``, ``tc``, ``sampling_frequency``, and
                ``minimum_frequency`` (see ``gwmock_signal`` waveform docs).
            waveform_arguments: Fixed parameters to pass to waveform model.
            start_time: Start time of the first signal segment in GPS seconds. Default is 0.
            duration: Duration of each signal segment in seconds. Default is 1024.
            sampling_frequency: Sampling frequency of the signals in Hz. Default is 4096.
            max_samples: Maximum number of samples to generate. None means infinite.
            dtype: Data type for the time series data. Default is np.float64.
            detectors: List of detector names. Default is None.
            minimum_frequency: Minimum GW frequency for waveform generation. Default is 5 Hz.
            source_type: Public gwmock-pop source-type routing key for backend lookup.
            earth_rotation: Whether to use time-dependent detector projection in gwmock-signal.
            **kwargs: Additional arguments absorbed by subclasses and mixins.
        """
        if waveform_model is None:
            waveform_model = "IMRPhenomXPHM"
        if not (isinstance(waveform_model, str) or callable(waveform_model)):
            raise TypeError("waveform_model must be a str, a callable waveform generator, or None.")

        self.waveform_model = waveform_model
        self.waveform_arguments = waveform_arguments or {}
        self.minimum_frequency = minimum_frequency
        self.source_type = source_type
        self.earth_rotation = earth_rotation
        self.signal_adapter = SignalAdapter.from_source_type(
            source_type=source_type,
            waveform_model=waveform_model,
            detectors=detectors or [],
        )
        self.detectors = list(self.signal_adapter.detector_names)

        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_sort_by=population_sort_by,
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            num_of_channels=len(self.detectors),
            max_samples=max_samples,
            dtype=dtype,
            **kwargs,
        )

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate signals for the current segment.

        Returns:
            TimeSeriesList: List of simulated signals.
        """
        output = []

        while True:
            # Get the next injection parameters
            parameters = self.get_next_injection_parameters()

            # If the parameters are None, break the loop
            if parameters is None:
                break

            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                earth_rotation=self.earth_rotation,
            )

            # Register the parameters
            strain.metadata.update({"injection_parameters": parameters})

            output.append(strain)

            # Check whether the start time of the strain is at or after the end time of the current segment
            if strain.start_time >= self.end_time:
                break
        return TimeSeriesList(output)

    @property
    def metadata(self) -> dict:
        """Get the metadata of the simulator.

        Returns:
            Metadata dictionary.
        """
        wm_meta: str
        if isinstance(self.waveform_model, str):
            wm_meta = self.waveform_model
        else:
            qual = getattr(self.waveform_model, "__qualname__", repr(self.waveform_model))
            mod = getattr(self.waveform_model, "__module__", "")
            wm_meta = f"{mod}.{qual}" if mod else qual

        meta = {
            **Simulator.metadata.fget(self),
            **TimeSeriesMixin.metadata.fget(self),
            **PopulationReaderMixin.metadata.fget(self),
            "signal": {
                "arguments": {
                    "waveform_model": wm_meta,
                    "waveform_arguments": self.waveform_arguments,
                    "minimum_frequency": self.minimum_frequency,
                    "source_type": self.source_type,
                    "earth_rotation": self.earth_rotation,
                    "detectors": self.detectors,
                }
            },
        }
        return meta

    def update_state(self) -> None:
        """Update internal state after each sample generation.

        This method can be overridden by subclasses to update any internal state
        after generating a sample. The default implementation does nothing.
        """
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
