"""Adapter from gwmock orchestration to public ``gwmock_signal`` APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gwmock_signal import Network, resolve_simulator_backend

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.detector.utils import DEFAULT_DETECTOR_BASE_PATH

_DEFAULT_WAVEFORM_MODEL = "IMRPhenomXPHM"


def _callable_waveform_registry_key(func: Callable[..., Any]) -> str:
    """Return a unique registry name for *func* on a ``WaveformFactory`` instance."""
    qual = getattr(func, "__qualname__", type(func).__name__)
    mod = getattr(func, "__module__", "")
    return f"__gwmock_custom__{mod}:{qual}__{id(func):#x}"


def _register_callable_waveform(backend: Any, registry_key: str, factory: Callable[..., Any]) -> None:
    """Register *factory* under *registry_key* through the public backend API."""
    backend.register_waveform_model(registry_key, factory)


def _resolve_detector_path(detector_spec: str) -> Path | None:
    """Resolve a detector spec to an on-disk network file when one exists."""
    detector_path = Path(detector_spec)
    if detector_path.is_file():
        return detector_path

    bundled_path = DEFAULT_DETECTOR_BASE_PATH / detector_path
    if bundled_path.is_file():
        return bundled_path

    return None


class SignalAdapter:
    """Bridge gwmock population/orchestration state to public gwmock-signal APIs."""

    def __init__(
        self,
        *,
        source_type: str,
        backend: Any,
        detector_specs: Sequence[str],
    ) -> None:
        """Store the resolved backend and detector network."""
        if not detector_specs:
            raise ValueError("detectors must be a non-empty sequence.")

        self._source_type = source_type
        self._backend = backend
        self._network = self._resolve_detector_network(detector_specs)
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
        backend = cls.instantiate_backend(backend_class, waveform_model=waveform_model)
        return cls(
            source_type=source_type,
            backend=backend,
            detector_specs=detectors,
        )

    @classmethod
    def from_backend(
        cls,
        *,
        source_type: str,
        backend: Any,
        detectors: Sequence[str],
    ) -> SignalAdapter:
        """Build an adapter from an already-instantiated backend."""
        return cls(
            source_type=source_type,
            backend=backend,
            detector_specs=detectors,
        )

    @staticmethod
    def instantiate_backend(
        backend_class: type[Any],
        *,
        waveform_model: str | Callable[..., Any] | None,
    ) -> Any:
        """Instantiate a signal backend class while preserving callable waveform support."""
        if waveform_model is None:
            try:
                return backend_class(waveform_model=_DEFAULT_WAVEFORM_MODEL)
            except TypeError:
                return backend_class()

        if callable(waveform_model):
            registry_key = _callable_waveform_registry_key(waveform_model)
            backend = backend_class(waveform_model=registry_key)
            _register_callable_waveform(backend, registry_key, waveform_model)
            return backend

        return backend_class(waveform_model=waveform_model)

    @staticmethod
    def _resolve_detector_network(detector_specs: Sequence[str]) -> Network:
        resolved_detectors: list[str | Any] = []
        for detector_spec in detector_specs:
            detector_path = _resolve_detector_path(str(detector_spec))
            if detector_path is not None:
                resolved_detectors.extend(Network.from_file(detector_path).detector_names)
                continue

            try:
                resolved_detectors.extend(Network.from_name(str(detector_spec)).detector_names)
            except ValueError:
                resolved_detectors.append(str(detector_spec))

        return Network.from_detectors(tuple(resolved_detectors))

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
        """Generate one signal chunk via the public gwmock-signal ``simulate`` API."""
        backend_parameters = {**(waveform_arguments or {}), **dict(parameters)}
        strain_stack = self._backend.simulate(
            backend_parameters,
            self._network.detector_names,
            background=None,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            earth_rotation=earth_rotation,
        )

        return TimeSeries(
            data=strain_stack.data,
            start_time=strain_stack.t0,
            sampling_frequency=strain_stack.sample_rate,
        )
