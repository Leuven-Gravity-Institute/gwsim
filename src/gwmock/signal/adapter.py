"""Adapter from gwmock orchestration to public ``gwmock_signal`` APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gwmock_signal import DetectorStrainStack, GWSimulator, Network, resolve_simulator_backend
from gwmock_signal.projection import project_polarizations_to_network

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.detector.utils import (
    DEFAULT_DETECTOR_BASE_PATH,
    interferometer_config_to_custom_detector,
)

_DEFAULT_WAVEFORM_MODEL = "IMRPhenomXPHM"


def _callable_waveform_registry_key(func: Callable[..., Any]) -> str:
    """Return a unique registry name for *func* on a ``WaveformFactory`` instance."""
    qual = getattr(func, "__qualname__", type(func).__name__)
    mod = getattr(func, "__module__", "")
    return f"__gwmock_custom__{mod}:{qual}__{id(func):#x}"


def _register_callable_waveform(backend: Any, registry_key: str, factory: Callable[..., Any]) -> None:
    """Register *factory* under *registry_key* on the backend's internal ``WaveformFactory``."""
    try:
        waveform_factory = backend._waveform_factory
    except AttributeError as exc:
        msg = (
            "Callable waveform_model is only supported when the resolved gwmock_signal "
            "backend exposes _waveform_factory (e.g. CBCSimulator)."
        )
        raise TypeError(msg) from exc
    waveform_factory.register_model(registry_key, factory)


class SignalAdapter:
    """Bridge gwmock population/orchestration state to public gwmock-signal APIs."""

    def __init__(
        self,
        *,
        source_type: str,
        backend: GWSimulator,
        detector_specs: Sequence[str],
    ) -> None:
        """Store the resolved backend and detector network."""
        if not detector_specs:
            raise ValueError("detectors must be a non-empty sequence.")

        self._source_type = source_type
        self._backend = backend
        self._network = Network.from_detectors(self._normalize_detector_specs(detector_specs))
        self._detector_names = tuple(
            detector if isinstance(detector, str) else detector.name for detector in self._network.detector_names
        )

    @classmethod
    def from_source_type(
        cls,
        *,
        source_type: str,
        waveform_model: str | Callable[..., Any] | None,
        detectors: Sequence[str],
    ) -> SignalAdapter:
        """Resolve the public gwmock-signal backend for one source type."""
        backend_class = resolve_simulator_backend(source_type)
        if waveform_model is None:
            backend = backend_class(waveform_model=_DEFAULT_WAVEFORM_MODEL)
        elif callable(waveform_model):
            registry_key = _callable_waveform_registry_key(waveform_model)
            backend = backend_class(waveform_model=registry_key)
            _register_callable_waveform(backend, registry_key, waveform_model)
        else:
            backend = backend_class(waveform_model=waveform_model)
        return cls(
            source_type=source_type,
            backend=backend,
            detector_specs=detectors,
        )

    @staticmethod
    def _normalize_detector_specs(detector_specs: Sequence[str]) -> tuple[str | Any, ...]:
        normalized_specs: list[str | Any] = []
        for detector_spec in detector_specs:
            detector_path = Path(detector_spec)
            if detector_path.is_file() or (DEFAULT_DETECTOR_BASE_PATH / detector_path).is_file():
                normalized_specs.append(interferometer_config_to_custom_detector(detector_spec))
            else:
                normalized_specs.append(str(detector_spec))
        return tuple(normalized_specs)

    @property
    def source_type(self) -> str:
        """Return the source-type routing key used for backend resolution."""
        return self._source_type

    @property
    def detector_names(self) -> tuple[str, ...]:
        """Return the ordered detector names used for output stacking."""
        return self._detector_names

    def simulate(
        self,
        parameters: Mapping[str, Any],
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: Mapping[str, Any] | None = None,
        earth_rotation: bool = True,
    ) -> TimeSeries:
        """Generate and project one signal chunk via public gwmock-signal APIs."""
        if not hasattr(self._backend, "generate_polarizations"):
            raise TypeError(
                "Resolved gwmock_signal backend does not expose generate_polarizations; "
                "the current gwmock adapter only supports transient-style backends."
            )

        backend_parameters = {**(waveform_arguments or {}), **dict(parameters)}
        polarizations = self._backend.generate_polarizations(  # type: ignore[attr-defined]
            backend_parameters,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
        )
        projected = project_polarizations_to_network(
            {"plus": polarizations[0], "cross": polarizations[1]},
            self._network.detector_names,
            right_ascension=backend_parameters["right_ascension"],
            declination=backend_parameters["declination"],
            polarization_angle=backend_parameters["polarization_angle"],
            earth_rotation=earth_rotation,
        )
        strain_stack = DetectorStrainStack.from_mapping(self.detector_names, projected)
        return TimeSeries(
            data=strain_stack.data,
            start_time=strain_stack.t0,
            sampling_frequency=strain_stack.sample_rate,
        )
