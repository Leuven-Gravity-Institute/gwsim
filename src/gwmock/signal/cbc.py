"""Compact Binary Coalescence (CBC) signal simulation module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.cbc_population_reader import CBCPopulationReaderMixin
from gwmock.mixin.population_reader import PopulationReaderMixin
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.signal.adapter import SignalAdapter
from gwmock.simulator.base import Simulator

DEFAULT_SIGNAL_SIMULATOR_DETECTORS: tuple[str, ...] = ("H1", "L1", "V1")


class CBCSignalSimulator(CBCPopulationReaderMixin, TimeSeriesMixin, Simulator):  # pylint: disable=too-many-ancestors
    """CBC Signal Simulator class."""

    def __init__(  # noqa: PLR0913
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
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
        """Initialize the CBC signal simulator.

        Args:
            population_file: Path to the population file.
            population_parameter_name_mapper: Dict mapping population column names to simulator parameter names.
            population_cache_dir: Directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300.
            waveform_model: Name (from registry), custom callable, or ``None`` for
                ``IMRPhenomXPHM`` (same default as ``SignalSimulator``).
            waveform_arguments: Fixed parameters to pass to waveform model.
            start_time: Start time of the first signal segment in GPS seconds. Default is 0.
            duration: Duration of each signal segment in seconds. Default is 1024.
            sampling_frequency: Sampling frequency of the signals in Hz. Default is 4096.
            max_samples: Maximum number of samples to generate. None means infinite.
            dtype: Data type for the time series data. Default is np.float64.
            detectors: List of detector names. If ``None`` or empty, defaults to the
                same three-detector network as ``SignalSimulator`` (``H1``, ``L1``, ``V1``).
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
        detectors = list(DEFAULT_SIGNAL_SIMULATOR_DETECTORS) if not detectors else list(detectors)
        self.signal_adapter = SignalAdapter.from_source_type(
            source_type=source_type,
            waveform_model=waveform_model,
            detectors=detectors,
        )
        self.detectors = list(self.signal_adapter.detector_names)

        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            waveform_model=waveform_model,
            waveform_arguments=waveform_arguments,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            max_samples=max_samples,
            dtype=dtype,
            detectors=detectors,
            minimum_frequency=minimum_frequency,
            source_type=source_type,
            earth_rotation=earth_rotation,
            **kwargs,
        )

    def _simulate(self, *args, **kwargs) -> TimeSeriesList:
        """Simulate signals for the current segment."""
        output = []

        while True:
            parameters = self.get_next_injection_parameters()
            if parameters is None:
                break

            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                earth_rotation=self.earth_rotation,
            )
            strain.metadata.update({"injection_parameters": parameters})
            output.append(strain)

            if strain.start_time >= self.end_time:
                break
        return TimeSeriesList(output)

    @property
    def metadata(self) -> dict:
        """Get the metadata of the simulator."""
        wm_meta: str
        if isinstance(self.waveform_model, str):
            wm_meta = self.waveform_model
        else:
            qual = getattr(self.waveform_model, "__qualname__", repr(self.waveform_model))
            mod = getattr(self.waveform_model, "__module__", "")
            wm_meta = f"{mod}.{qual}" if mod else qual

        return {
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

    def update_state(self) -> None:
        """Advance to the next segment."""
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration
