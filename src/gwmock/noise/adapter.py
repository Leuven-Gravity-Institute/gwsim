"""Adapter from gwmock orchestration to public ``gwmock_noise`` APIs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, cast

from gwmock_noise import BaseNoiseSimulator, DefaultNoiseSimulator, NoiseConfig, OutputConfig, SimulationResult

from gwmock.cli.utils.template import expand_template_variables
from gwmock.simulator.base import Simulator
from gwmock.simulator.state import StateAttribute

_SUPPORTED_OUTPUT_FORMATS = {"npy", "gwf"}
_DETECTOR_PLACEHOLDER = re.compile(r"\{\{\s*detectors?\s*\}\}")


def _coerce_path(value: str | Path | None) -> Path | None:
    """Normalize a path-like input."""
    if value is None:
        return None
    return Path(value)


def _coerce_path_mapping(values: dict[str, str | Path] | None) -> dict[str, Path] | None:
    """Normalize mapping values to ``Path`` objects."""
    if values is None:
        return None
    return {key: Path(value) for key, value in values.items()}


def _coerce_path_schedule(values: list[tuple[float, str | Path]] | None) -> list[tuple[float, Path]] | None:
    """Normalize scheduled path values to ``Path`` objects."""
    if values is None:
        return None
    return [(offset, Path(path)) for offset, path in values]


def _flatten_first(value: str | list[str] | list[list[str]]) -> str:
    """Return the first scalar string from an expanded template value."""
    while isinstance(value, list):
        if not value:
            raise ValueError("Template expansion produced an empty list; cannot derive a scalar string.")
        value = value[0]
    return value


class NoiseAdapter:
    """Bridge gwmock orchestration state to public ``gwmock_noise`` APIs."""

    def __init__(self, *, backend: BaseNoiseSimulator) -> None:
        """Store the resolved public gwmock-noise backend."""
        self._backend = backend

    @classmethod
    def from_backend(cls, backend: BaseNoiseSimulator | None = None) -> NoiseAdapter:
        """Build an adapter from a public gwmock-noise backend."""
        return cls(backend=DefaultNoiseSimulator() if backend is None else backend)

    @property
    def backend(self) -> BaseNoiseSimulator:
        """Return the public backend used by the adapter."""
        return self._backend

    def run(  # noqa: PLR0913
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        output_directory: str | Path,
        output_prefix: str,
        output_format: Literal["npy", "gwf"],
        gps_start: float,
        channel_prefix: str,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
    ) -> SimulationResult:
        """Run one noise batch through the public gwmock-noise boundary."""
        config = NoiseConfig(
            detectors=list(detectors),
            duration=duration,
            sampling_frequency=sampling_frequency,
            output=OutputConfig(
                directory=Path(output_directory),
                prefix=output_prefix,
                format=output_format,
                gps_start=gps_start,
                channel_prefix=channel_prefix,
            ),
            seed=seed,
            psd_file=_coerce_path(psd_file),
            psd_schedule=_coerce_path_schedule(psd_schedule),
            psd_files=_coerce_path_mapping(psd_files),
            csd_files=_coerce_path_mapping(csd_files),
            low_frequency_cutoff=low_frequency_cutoff,
            high_frequency_cutoff=high_frequency_cutoff,
            spectral_lines=spectral_lines,
            glitches=glitches,
        )
        return self._backend.run(config)


class UpstreamNoiseSimulator(Simulator):
    """Stateful gwmock wrapper around the public ``gwmock_noise`` run boundary."""

    start_time = StateAttribute(0.0)

    def __init__(  # noqa: PLR0913
        self,
        *,
        duration: float = 4.0,
        sampling_frequency: float = 4096.0,
        detectors: list[str] | None = None,
        start_time: float = 0.0,
        max_samples: int | None = None,
        seed: int | None = None,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        spectral_lines: list[Any] | None = None,
        glitches: list[Any] | None = None,
        output_directory: str | Path | None = None,
        output_prefix: str | None = None,
        output_format: Literal["npy", "gwf"] = "npy",
        channel_prefix: str = "MOCK",
        noise_adapter: NoiseAdapter | None = None,
        noise_backend: BaseNoiseSimulator | None = None,
        **kwargs,
    ) -> None:
        """Initialize the gwmock orchestration wrapper."""
        if noise_adapter is not None and noise_backend is not None:
            raise ValueError("Pass either noise_adapter or noise_backend, not both.")
        if duration <= 0:
            raise ValueError("duration must be positive.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive.")

        super().__init__(max_samples=max_samples, **kwargs)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.start_time = float(start_time)
        self.base_seed = seed
        self.psd_file = _coerce_path(psd_file)
        self.psd_schedule = _coerce_path_schedule(psd_schedule)
        self.psd_files = _coerce_path_mapping(psd_files)
        self.csd_files = _coerce_path_mapping(csd_files)
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.spectral_lines = spectral_lines
        self.glitches = glitches
        self.output_directory = _coerce_path(output_directory)
        self.output_prefix = output_prefix
        self.output_format = output_format
        self.channel_prefix = channel_prefix
        self.noise_adapter = noise_adapter if noise_adapter is not None else NoiseAdapter.from_backend(noise_backend)

        self._active_output_directory = self.output_directory
        self._active_output_prefix = output_prefix
        self._active_output_format = output_format
        self._active_channel_prefix = channel_prefix
        self._active_overwrite = False
        self._last_result: SimulationResult | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the adapter-backed orchestration state."""
        metadata = {
            **super().metadata,
            "noise": {
                "arguments": {
                    "duration": self.duration,
                    "sampling_frequency": self.sampling_frequency,
                    "detectors": self.detectors,
                    "seed": self.base_seed,
                    "psd_file": None if self.psd_file is None else str(self.psd_file),
                    "psd_schedule": (
                        None
                        if self.psd_schedule is None
                        else [(offset, str(path)) for offset, path in self.psd_schedule]
                    ),
                    "psd_files": None if self.psd_files is None else {k: str(v) for k, v in self.psd_files.items()},
                    "csd_files": None if self.csd_files is None else {k: str(v) for k, v in self.csd_files.items()},
                    "low_frequency_cutoff": self.low_frequency_cutoff,
                    "high_frequency_cutoff": self.high_frequency_cutoff,
                    "spectral_lines": self.spectral_lines,
                    "glitches": self.glitches,
                },
                "output": {
                    "directory": None if self._active_output_directory is None else str(self._active_output_directory),
                    "prefix": self._active_output_prefix,
                    "format": self._active_output_format,
                    "channel_prefix": self._active_channel_prefix,
                },
                "state_model": (
                    "gwmock tracks counter and gps_start locally; each batch reseeds gwmock_noise "
                    "deterministically as base_seed + counter because BaseNoiseSimulator.run() is stateless."
                ),
            },
        }
        if self._last_result is not None:
            metadata["noise"]["last_output_paths"] = {
                detector: str(path) for detector, path in self._last_result.output_paths.items()
            }
        return metadata

    def set_batch_context(self, *, batch: Any, output_directory: Path, overwrite: bool) -> None:
        """Attach per-batch output settings before ``simulate()`` runs."""
        output_config = batch.simulator_config.output
        expanded_output_args = expand_template_variables(output_config.arguments or {}, self)

        prefix = expanded_output_args.pop("prefix", None) or self.output_prefix
        if prefix is None:
            prefix = self._derive_output_prefix(output_config.file_name)

        output_format = expanded_output_args.pop("format", None) or self._infer_output_format(output_config.file_name)
        channel_prefix = expanded_output_args.pop("channel_prefix", self.channel_prefix)
        gps_start = expanded_output_args.pop("gps_start", self.start_time)
        if expanded_output_args:
            unsupported = ", ".join(sorted(expanded_output_args))
            raise ValueError(
                "Noise adapter output arguments only support prefix, format, gps_start, and channel_prefix. "
                f"Unsupported keys: {unsupported}."
            )

        self._active_output_directory = Path(output_directory)
        self._active_output_prefix = str(prefix)
        self._active_output_format = cast(Literal["npy", "gwf"], str(output_format))
        self._active_channel_prefix = str(channel_prefix)
        self._active_overwrite = overwrite
        self.start_time = float(gps_start)

    def simulate(self) -> SimulationResult:
        """Generate one batch by delegating to the public gwmock-noise run API."""
        if self._active_output_directory is None:
            raise ValueError("Noise adapter batch context is missing an output directory.")
        if self._active_output_prefix is None:
            raise ValueError("Noise adapter batch context is missing an output prefix.")

        expected_paths = self._expected_artifact_paths()
        if not self._active_overwrite:
            existing = [path for path in expected_paths if path.exists()]
            if existing:
                raise FileExistsError(
                    f"Noise adapter output(s) already exist: {', '.join(str(path) for path in existing)}. "
                    "Use overwrite=True to overwrite them."
                )

        result = self.noise_adapter.run(
            detectors=self.detectors,
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            output_directory=self._active_output_directory,
            output_prefix=self._active_output_prefix,
            output_format=self._active_output_format,
            gps_start=float(self.start_time),
            channel_prefix=self._active_channel_prefix,
            seed=self._segment_seed(),
            psd_file=self.psd_file,
            psd_schedule=self.psd_schedule,
            psd_files=self.psd_files,
            csd_files=self.csd_files,
            low_frequency_cutoff=self.low_frequency_cutoff,
            high_frequency_cutoff=self.high_frequency_cutoff,
            spectral_lines=self.spectral_lines,
            glitches=self.glitches,
        )
        self._last_result = result
        return result

    def update_state(self) -> None:
        """Advance the gwmock-owned orchestration state to the next segment."""
        self.counter = cast(int, self.counter) + 1
        self.start_time += self.duration

    def _save_data(self, data: Any, file_name: str | Path, **kwargs) -> None:
        """The upstream adapter writes data itself; gwmock must not rewrite it."""
        raise TypeError("UpstreamNoiseSimulator writes outputs through gwmock_noise and does not support save_data().")

    def _segment_seed(self) -> int | None:
        """Return the deterministic seed for the current batch."""
        if self.base_seed is None:
            return None
        return int(self.base_seed) + int(self.counter)

    def _infer_output_format(self, file_name_template: str) -> Literal["npy", "gwf"]:
        """Infer the upstream output format from the template suffix or constructor default."""
        suffix = Path(file_name_template).suffix.lower().lstrip(".")
        if suffix in _SUPPORTED_OUTPUT_FORMATS:
            return cast(Literal["npy", "gwf"], suffix)
        if suffix and suffix not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                "Noise adapter output templates must use a .npy or .gwf suffix, or specify "
                "`output.arguments.format` explicitly."
            )
        return self.output_format

    def _derive_output_prefix(self, file_name_template: str) -> str:
        """Derive an upstream prefix from the configured gwmock output template."""
        template_without_detector = _DETECTOR_PLACEHOLDER.sub("", file_name_template)
        expanded = expand_template_variables(str(Path(template_without_detector).with_suffix("")), self)
        prefix = _flatten_first(expanded).strip("-_ ")
        return prefix or f"noise-{self.counter}"

    def _expected_artifact_paths(self) -> list[Path]:
        """Return the detector artifacts and sidecars the upstream run will create."""
        if self._active_output_directory is None or self._active_output_prefix is None:
            return []

        output_paths: list[Path] = []
        start_token = self._format_time_token(float(self.start_time))
        duration_token = self._format_time_token(self.duration)
        for detector in self.detectors:
            if self._active_output_format == "npy":
                artifact = self._active_output_directory / f"{self._active_output_prefix}_{detector}.npy"
            else:
                channel = f"{detector}:{self._active_channel_prefix}_NOISE"
                artifact_name = f"{detector[0]}-{channel}_{start_token}-{duration_token}.gwf"
                artifact = self._active_output_directory / f"{self._active_output_prefix}_{artifact_name}"
            output_paths.append(artifact)
            output_paths.append(self._active_output_directory / f"{self._active_output_prefix}_{detector}.json")
        return output_paths

    @staticmethod
    def _format_time_token(value: float) -> str:
        """Return the filename token used by the upstream frame writer."""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")
