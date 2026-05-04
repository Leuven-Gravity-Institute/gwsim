"""Adapter-backed orchestration for the primary gwmock CLI path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from gwmock_noise import SimulationResult
from gwmock_pop import (
    BBHSimulator,
    BNSPriorSimulator,
    CBCPriorSimulator,
    FilePopulationLoader,
    GWPopSimulator,
    NSBHPriorSimulator,
)

from gwmock.cli.utils.config import OrchestrationConfig
from gwmock.cli.utils.config_resolution import resolve_max_samples
from gwmock.cli.utils.template import expand_template_variables
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.mixin.time_series import TimeSeriesMixin
from gwmock.noise import NoiseAdapter
from gwmock.population import PopulationAdapter
from gwmock.signal import SignalAdapter
from gwmock.simulator.base import Simulator
from gwmock.simulator.state import StateAttribute

_DETECTOR_PLACEHOLDER = "{{ detectors }}"

_POPULATION_BACKENDS: dict[str, type[GWPopSimulator]] = {
    "BBHSimulator": BBHSimulator,
    "bbh": BBHSimulator,
    "CBCPriorSimulator": CBCPriorSimulator,
    "cbc_prior": CBCPriorSimulator,
    "BNSPriorSimulator": BNSPriorSimulator,
    "bns_prior": BNSPriorSimulator,
    "NSBHPriorSimulator": NSBHPriorSimulator,
    "nsbh_prior": NSBHPriorSimulator,
    "FilePopulationLoader": FilePopulationLoader,
    "file": FilePopulationLoader,
}


@dataclass(slots=True)
class AdapterOrchestrationResult:
    """Artifacts produced by one adapter-backed orchestration batch."""

    signal_segment: TimeSeries
    noise_result: SimulationResult


def _normalize_keys(values: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML-style hyphenated keys into Python identifiers."""
    return {key.replace("-", "_"): value for key, value in values.items()}


def _flatten_first(value: str | list[str] | list[list[str]]) -> str:
    """Return the first scalar string from an expanded template."""
    while isinstance(value, list):
        if not value:
            raise ValueError("Template expansion produced an empty list; cannot derive a scalar string.")
        value = value[0]
    return value


class AdapterOrchestrator(TimeSeriesMixin, Simulator):
    """Compose population, signal, and noise adapters inside gwmock."""

    population_index = StateAttribute(0)

    def __init__(  # noqa: PLR0913
        self,
        *,
        population_events: list[dict[str, Any]],
        source_type: str,
        waveform_model: str | None,
        waveform_arguments: dict[str, Any],
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        start_time: float,
        max_samples: int,
        minimum_frequency: float,
        earth_rotation: bool,
        noise_arguments: dict[str, Any],
        orchestration_config: OrchestrationConfig,
    ) -> None:
        self._population_events = tuple(population_events)
        self._source_type = source_type
        self.waveform_model = waveform_model
        self.waveform_arguments = waveform_arguments
        self.minimum_frequency = minimum_frequency
        self.earth_rotation = earth_rotation
        self.orchestration_config = orchestration_config
        self.signal_adapter = SignalAdapter.from_source_type(
            source_type=source_type,
            waveform_model=waveform_model,
            detectors=detectors,
        )
        self.detectors = list(self.signal_adapter.detector_names)
        self.noise_arguments = noise_arguments
        self.noise_adapter = NoiseAdapter.from_backend()
        self._active_signal_output_directory = Path("signal")
        self._active_noise_output_directory = Path("noise")
        self._active_noise_output_arguments: dict[str, Any] = {}
        self._active_overwrite = False

        super().__init__(
            max_samples=max_samples,
            start_time=start_time,
            duration=duration,
            sampling_frequency=sampling_frequency,
            num_of_channels=len(self.detectors),
        )

    @classmethod
    def from_config(
        cls,
        orchestration_config: OrchestrationConfig,
        global_simulator_arguments: dict[str, Any] | None = None,
    ) -> AdapterOrchestrator:
        """Instantiate the composite adapter-backed orchestration path."""
        global_args = _normalize_keys(global_simulator_arguments or {})
        max_samples = resolve_max_samples(simulator_args={}, global_args=global_args)
        duration = float(global_args.get("duration", 4.0))
        sampling_frequency = float(global_args.get("sampling_frequency", 4096.0))
        start_time = float(global_args.get("start_time", 0.0))

        population_backend = cls._instantiate_population_backend(orchestration_config.population)
        population_adapter = PopulationAdapter.from_backend(
            population_backend,
            n_samples=orchestration_config.population.n_samples,
        )
        population_events = list(population_adapter.iter_event_parameters())
        if orchestration_config.population.sort_by:
            sort_key = orchestration_config.population.sort_by
            if population_events and any(sort_key not in event for event in population_events):
                raise ValueError(f"Population event ordering key '{sort_key}' is missing from one or more events.")
            population_events.sort(key=lambda event: event[sort_key])

        noise_arguments = _normalize_keys(orchestration_config.noise.arguments)
        noise_arguments.setdefault("detectors", list(orchestration_config.signal.detectors))

        return cls(
            population_events=population_events,
            source_type=population_adapter.source_type,
            waveform_model=orchestration_config.signal.waveform_model,
            waveform_arguments=orchestration_config.signal.waveform_arguments,
            detectors=list(orchestration_config.signal.detectors),
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            max_samples=max_samples,
            minimum_frequency=orchestration_config.signal.minimum_frequency,
            earth_rotation=orchestration_config.signal.earth_rotation,
            noise_arguments=noise_arguments,
            orchestration_config=orchestration_config,
        )

    @staticmethod
    def _instantiate_population_backend(population_config) -> GWPopSimulator:
        backend_name = population_config.backend
        try:
            backend_cls = _POPULATION_BACKENDS[backend_name]
        except KeyError as exc:
            available = ", ".join(sorted(_POPULATION_BACKENDS))
            raise ValueError(
                f"Unknown population backend '{backend_name}'. Available public backends: {available}."
            ) from exc

        backend_arguments = dict(population_config.arguments)
        if population_config.source_type is not None:
            backend_arguments.setdefault("source_type", population_config.source_type)

        return backend_cls(**backend_arguments)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return orchestration metadata for reproducibility."""
        return {
            **super().metadata,
            "orchestration": {
                "source_type": self._source_type,
                "population_events_total": len(self._population_events),
                "population_events_remaining": len(self._population_events) - int(self.population_index),
                "signal": {
                    "waveform_model": self.waveform_model,
                    "waveform_arguments": self.waveform_arguments,
                    "minimum_frequency": self.minimum_frequency,
                    "earth_rotation": self.earth_rotation,
                    "detectors": self.detectors,
                },
                "noise": {
                    "arguments": self.noise_arguments,
                },
            },
        }

    def set_batch_context(self, *, batch: Any, output_directory: Path, overwrite: bool) -> None:
        """Resolve per-batch output directories and runtime arguments."""
        signal_output = batch.simulator_config.signal.output
        noise_output = batch.simulator_config.noise.output
        self._active_signal_output_directory = self._resolve_output_directory(
            output_directory,
            signal_output.output_directory,
            fallback_subdir="signal",
        )
        self._active_noise_output_directory = self._resolve_output_directory(
            output_directory,
            noise_output.output_directory,
            fallback_subdir="noise",
        )
        self._active_noise_output_arguments = expand_template_variables(noise_output.arguments or {}, self)
        self._active_overwrite = overwrite

    def simulate(self) -> AdapterOrchestrationResult:
        """Generate one signal segment and one noise segment for the current batch."""
        signal_segment = TimeSeriesMixin.simulate(self)
        noise_result = self._run_noise_batch()
        return AdapterOrchestrationResult(signal_segment=signal_segment, noise_result=noise_result)

    def _simulate(self) -> TimeSeriesList:
        """Generate signal chunks for the current segment from population events."""
        chunks = TimeSeriesList()
        while self.population_index < len(self._population_events):
            parameters = self._population_events[int(self.population_index)]
            strain = self.signal_adapter.simulate(
                parameters,
                sampling_frequency=float(self.sampling_frequency.value),
                minimum_frequency=self.minimum_frequency,
                waveform_arguments=self.waveform_arguments,
                earth_rotation=self.earth_rotation,
            )
            strain.metadata.update({"injection_parameters": dict(parameters)})
            chunks.append(strain)
            self.population_index = cast(int, self.population_index) + 1
            if strain.start_time >= self.end_time:
                break
        return chunks

    def update_state(self) -> None:
        """Advance to the next segment."""
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration

    def signal_output_directory(self) -> Path:
        """Return the active signal output directory."""
        return self._active_signal_output_directory

    def signal_output_template(self) -> str:
        """Return the active signal file-name template."""
        return self.orchestration_config.signal.output.file_name

    def signal_output_arguments(self) -> dict[str, Any]:
        """Return the active signal output keyword arguments."""
        return dict(self.orchestration_config.signal.output.arguments or {})

    def _run_noise_batch(self) -> SimulationResult:
        output_format = self._infer_noise_output_format(self.orchestration_config.noise.output.file_name)
        output_prefix = self._derive_noise_output_prefix(self.orchestration_config.noise.output.file_name)
        channel_prefix = str(self._active_noise_output_arguments.pop("channel_prefix", "MOCK"))
        gps_start = float(self._active_noise_output_arguments.pop("gps_start", float(self.start_time.value)))
        if self._active_noise_output_arguments:
            unsupported = ", ".join(sorted(self._active_noise_output_arguments))
            raise ValueError(
                "Noise orchestration output arguments only support channel_prefix and gps_start. "
                f"Unsupported keys: {unsupported}."
            )

        return self.noise_adapter.run(
            detectors=list(self.noise_arguments["detectors"]),
            duration=float(self.duration.value),
            sampling_frequency=float(self.sampling_frequency.value),
            output_directory=self._active_noise_output_directory,
            output_prefix=output_prefix,
            output_format=output_format,
            gps_start=gps_start,
            channel_prefix=channel_prefix,
            seed=self._segment_seed(),
            psd_file=self.noise_arguments.get("psd_file"),
            psd_schedule=self.noise_arguments.get("psd_schedule"),
            psd_files=self.noise_arguments.get("psd_files"),
            csd_files=self.noise_arguments.get("csd_files"),
            low_frequency_cutoff=self.noise_arguments.get("low_frequency_cutoff", 2.0),
            high_frequency_cutoff=self.noise_arguments.get("high_frequency_cutoff"),
            spectral_lines=self.noise_arguments.get("spectral_lines"),
            glitches=self.noise_arguments.get("glitches"),
        )

    def _segment_seed(self) -> int | None:
        base_seed = self.noise_arguments.get("seed")
        if base_seed is None:
            return None
        return int(base_seed) + int(self.counter)

    def _infer_noise_output_format(self, file_name_template: str) -> Literal["npy", "gwf"]:
        suffix = Path(file_name_template).suffix.lower().lstrip(".")
        if suffix not in {"npy", "gwf"}:
            raise ValueError("Noise output templates must end with .npy or .gwf.")
        return cast(Literal["npy", "gwf"], suffix)

    def _derive_noise_output_prefix(self, file_name_template: str) -> str:
        template_without_detector = file_name_template.replace(_DETECTOR_PLACEHOLDER, "")
        prefix = _flatten_first(expand_template_variables(str(Path(template_without_detector).with_suffix("")), self))
        prefix = prefix.strip("-_ ")
        return prefix or f"noise-{self.counter}"

    @staticmethod
    def _resolve_output_directory(
        base_output_directory: Path, configured_directory: str | None, fallback_subdir: str
    ) -> Path:
        if configured_directory is None:
            return base_output_directory / fallback_subdir
        configured_path = Path(configured_directory)
        return configured_path if configured_path.is_absolute() else base_output_directory / configured_path
