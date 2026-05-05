"""Focused tests for the adapter-backed CLI orchestration path."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import yaml

from gwmock.cli import adapter_orchestration
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
from gwmock.data.time_series.time_series import TimeSeries

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
    """Minimal signal adapter stub returning deterministic strains."""

    detector_names = ("H1",)

    def simulate(
        self,
        parameters: dict,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        waveform_arguments: dict | None = None,
        earth_rotation: bool = True,
    ) -> TimeSeries:
        _ = minimum_frequency, waveform_arguments, earth_rotation
        return TimeSeries(
            data=np.full((1, 4), parameters["detector_frame_mass_1"]),
            start_time=parameters["coa_time"],
            sampling_frequency=sampling_frequency,
        )


class FakeNoiseAdapter:
    """Minimal noise adapter stub that materializes declared outputs."""

    def run(
        self,
        *,
        detectors: list[str],
        output_directory: Path,
        output_prefix: str,
        output_format: str,
        **_kwargs,
    ):
        if output_format != "npy":
            raise AssertionError("Test noise adapter expects npy outputs.")
        output_directory.mkdir(parents=True, exist_ok=True)
        output_paths = {}
        for detector in detectors:
            output_path = output_directory / f"{output_prefix}_{detector}.npy"
            output_path.write_bytes(b"noise")
            output_paths[detector] = output_path
        return type("SimulationResultStub", (), {"output_paths": output_paths})()


def _orchestration_config(tmp_path: Path) -> Config:
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
                backend="file",
                source_type="bbh",
                n_samples=2,
                arguments={"path": str(tmp_path / "population.h5")},
            ),
            signal=SignalConfig(
                detectors=["H1"],
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

    monkeypatch.setitem(
        adapter_orchestration._POPULATION_BACKENDS,
        "file",
        FakePopulationBackend,
    )
    monkeypatch.setattr(
        "gwmock.cli.adapter_orchestration.SignalAdapter.from_source_type",
        lambda **_kwargs: FakeSignalAdapter(),
    )
    monkeypatch.setattr(
        "gwmock.cli.adapter_orchestration.NoiseAdapter.from_backend",
        lambda *args, **kwargs: FakeNoiseAdapter(),
    )
    monkeypatch.setattr(
        "gwmock.mixin.time_series.TimeSeriesMixin._save_gwf_data",
        lambda self, data, file_name, channel=None, **kwargs: (
            Path(file_name).parent.mkdir(parents=True, exist_ok=True),
            Path(file_name).write_text(channel or "STRAIN"),
        )[-1],
    )

    _simulate_impl(str(config_file), overwrite=True, metadata=True)

    assert (tmp_path / "output" / "signal" / "signal-0.gwf").exists()
    assert (tmp_path / "output" / "signal" / "signal-1.gwf").exists()
    assert (tmp_path / "output" / "noise" / "noise-0_H1.npy").exists()
    assert (tmp_path / "output" / "noise" / "noise-1_H1.npy").exists()
    metadata = yaml.safe_load((tmp_path / "metadata" / "orchestration-0.metadata.yaml").read_text())
    assert metadata["simulator_config"]["population"]["backend"] == "file"
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
        lambda self, data, file_name, channel=None, **kwargs: (
            Path(file_name).parent.mkdir(parents=True, exist_ok=True),
            Path(file_name).write_text(channel or "STRAIN"),
        )[-1],
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
