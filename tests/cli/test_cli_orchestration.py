"""Focused tests for the adapter-backed CLI orchestration path."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import yaml
from gwmock_signal import DetectorStrainStack
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.cli.simulate import _simulate_impl
from gwmock.cli.utils.config import (
    Config,
    GlobalsConfig,
    NoiseAdapterConfig,
    OrchestrationConfig,
    PopulationConfig,
    SignalConfig,
    SimulatorOutputConfig,
)
from gwmock.cli.utils.simulation_plan import create_plan_from_config

EXPECTED_BATCHES = 2


class FakePopulationBackend:
    """Minimal public-style population backend for orchestration tests."""

    parameter_names: ClassVar[tuple[str, ...]] = ("detector_frame_mass_1", "detector_frame_mass_2", "coa_time")
    metadata: ClassVar[dict[str, object]] = {
        "fetch": {"scheme": "https"},
        "resolved_path": str(Path(tempfile.gettempdir()) / "catalog.h5"),
    }

    def __init__(self, path: str, source_type: str = "bbh") -> None:
        self.path = path
        self.source_type = source_type

    def simulate(self, n_samples: int, **_kwargs):
        if n_samples != EXPECTED_BATCHES:
            raise AssertionError("Unexpected population sample count for test.")
        return {
            "detector_frame_mass_1": np.array([30.0, 31.0]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "coa_time": np.array([100.5, 104.5]),
        }


class FakeSignalAdapter:
    """Minimal signal backend returning deterministic strain stacks."""

    required_params = frozenset({"detector_frame_mass_1", "coa_time"})

    def __init__(self, waveform_model: str = "IMRPhenomXPHM") -> None:
        self.waveform_model = waveform_model

    def simulate(
        self,
        parameters: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = background, minimum_frequency, earth_rotation, interpolate_if_offset
        return DetectorStrainStack.from_mapping(
            detector_names,
            {
                detector: GWpyTimeSeries(
                    np.full(4, parameters["detector_frame_mass_1"]),
                    t0=parameters["coa_time"],
                    sample_rate=sampling_frequency,
                )
                for detector in detector_names
            },
        )


class FakeNoiseAdapter:
    """Minimal noise protocol backend that materializes deterministic arrays."""

    def __init__(
        self,
        duration: float = 4.0,
        sampling_frequency: float = 4.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = ["H1"] if detectors is None else detectors
        self.seed = seed

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        _ = duration, sampling_frequency, seed
        return {detector: np.zeros(4) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        yield self.generate(
            duration=chunk_duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {"kind": "fake-noise"}


def _write_signal_file(self, data, file_name, channel=None, **kwargs):
    _ = data, kwargs
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    Path(file_name).write_text(channel or "STRAIN")


def _fake_orchestration_config(tmp_path: Path) -> Config:
    return Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 4,
                "duration": 4,
                "start-time": 100,
                "max-samples": 2,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend="tests.cli.test_cli_orchestration:FakePopulationBackend",
                source_type="bbh",
                n_samples=2,
                arguments={"path": str(tmp_path / "population.h5")},
            ),
            signal=SignalConfig(
                backend="tests.cli.test_cli_orchestration:FakeSignalAdapter",
                detectors=["H1"],
                output=SimulatorOutputConfig(
                    file_name="signal-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "H1:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                backend="tests.cli.test_cli_orchestration:FakeNoiseAdapter",
                arguments={"seed": 7, "detectors": ["H1"], "duration": 4.0, "sampling_frequency": 4.0},
                output=SimulatorOutputConfig(
                    file_name="noise-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )


def _orchestration_config(tmp_path: Path) -> Config:
    return _fake_orchestration_config(tmp_path)


def _assert_noise_outputs_exist(output_directory: Path) -> None:
    for counter in range(EXPECTED_BATCHES):
        for detector in ["H1"]:
            assert (output_directory / f"noise-{counter}_{detector}.npy").exists()


def _write_real_population_catalog(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("detector_frame_mass_1", data=[30.0])
        group.create_dataset("detector_frame_mass_2", data=[20.0])
        group.create_dataset("coa_time", data=[1001.0])
        group.create_dataset("distance", data=[400.0])
        group.create_dataset("inclination", data=[0.3])
        group.create_dataset("right_ascension", data=[1.1])
        group.create_dataset("declination", data=[-0.5])
        group.create_dataset("polarization_angle", data=[0.2])


def _real_orchestration_config(tmp_path: Path, population_path: Path) -> Config:
    return Config(
        globals=GlobalsConfig(
            working_directory=str(tmp_path),
            output_directory="output",
            metadata_directory="metadata",
            simulator_arguments={
                "sampling-frequency": 64,
                "duration": 4,
                "start-time": 1000,
                "max-samples": 1,
            },
        ),
        orchestration=OrchestrationConfig(
            population=PopulationConfig(
                backend="file",
                source_type="bbh",
                n_samples=1,
                arguments={"path": str(population_path)},
            ),
            signal=SignalConfig(
                detectors=["H1"],
                waveform_model="IMRPhenomD",
                minimum_frequency=20.0,
                output=SimulatorOutputConfig(
                    file_name="signal-{{ counter }}.gwf",
                    output_directory="signal",
                    arguments={"channel": "H1:STRAIN"},
                ),
            ),
            noise=NoiseAdapterConfig(
                arguments={"seed": 7},
                output=SimulatorOutputConfig(
                    file_name="noise-{{ counter }}.npy",
                    output_directory="noise",
                ),
            ),
        ),
    )


def test_create_plan_from_orchestration_config(tmp_path: Path):
    """Batch planning should respect the new orchestration config surface."""
    config = _orchestration_config(tmp_path)

    plan = create_plan_from_config(config, tmp_path / "checkpoints")

    assert plan.total_batches == EXPECTED_BATCHES
    assert all(batch.simulator_name == "orchestration" for batch in plan.batches)
    assert all(isinstance(batch.simulator_config, OrchestrationConfig) for batch in plan.batches)


def test_simulate_command_runs_adapter_orchestration(monkeypatch, tmp_path: Path):
    """The CLI should execute the adapter-backed orchestration path end to end."""
    config = _orchestration_config(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr("gwmock.mixin.time_series.TimeSeriesMixin._save_gwf_data", _write_signal_file)

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    assert (tmp_path / "output" / "signal" / "signal-0.gwf").exists()
    assert (tmp_path / "output" / "signal" / "signal-1.gwf").exists()
    _assert_noise_outputs_exist(tmp_path / "output" / "noise")
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.yaml").read_text())
    assert (
        metadata["simulator_config"]["population"]["backend"]
        == "tests.cli.test_cli_orchestration:FakePopulationBackend"
    )
    assert metadata["simulator_config"]["signal"]["detectors"] == ["H1"]
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"] == FakePopulationBackend.metadata


def test_simulate_command_runs_real_public_subpackages(monkeypatch, tmp_path: Path):
    """The orchestration path should work against the real public subpackage contracts."""
    population_path = tmp_path / "population.h5"
    _write_real_population_catalog(population_path)
    config = _real_orchestration_config(tmp_path, population_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True), sort_keys=False))

    monkeypatch.setattr(
        "gwmock.mixin.time_series.TimeSeriesMixin._save_gwf_data",
        _write_signal_file,
    )

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    signal_path = tmp_path / "output" / "signal" / "signal-0.gwf"
    noise_path = tmp_path / "output" / "noise" / "noise-0_H1.npy"
    metadata_path = tmp_path / "metadata" / "orchestration-0.metadata.yaml"

    assert signal_path.read_text() == "H1:STRAIN"
    assert noise_path.exists()
    assert np.load(noise_path).shape == (256,)
    metadata = yaml.safe_load(metadata_path.read_text())
    assert metadata["simulator_config"]["population"]["backend"] == "file"
    assert metadata["simulator_config"]["signal"]["waveform_model"] == "IMRPhenomD"
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"]["original_path"] == str(
        population_path
    )
    assert metadata["simulator_metadata"]["orchestration"]["population"]["metadata"]["resolved_path"] == str(
        population_path
    )
    assert metadata["versions"]["gwmock-pop"] >= "0.6.0"
    assert metadata["versions"]["gwmock-signal"] >= "0.5.0"
    assert metadata["versions"]["gwmock-noise"] >= "0.1.2"
