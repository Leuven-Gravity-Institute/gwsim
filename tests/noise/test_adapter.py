"""Tests for the gwmock-side gwmock_noise adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock.cli.utils.config import SimulatorConfig, SimulatorOutputConfig
from gwmock.cli.utils.simulation_plan import SimulationBatch
from gwmock.noise import NoiseAdapter, UpstreamNoiseSimulator

TEST_DURATION = 8.0
TEST_SAMPLING_FREQUENCY = 256.0
TEST_GPS_START = 1234.5
TEST_SEED = 11
SEGMENT_DURATION = 4.0
BASE_SEED = 7


class FakeNoiseBackend:
    """Minimal public gwmock-noise backend for adapter tests."""

    def __init__(self) -> None:
        self.run_calls = []

    def run(self, config):
        """Record the config and materialize the declared outputs."""
        self.run_calls.append(config)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        output_paths = {}
        for detector in config.detectors:
            suffix = ".npy" if config.output.format == "npy" else ".gwf"
            artifact_path = config.output.directory / f"{config.output.prefix}_{detector}{suffix}"
            artifact_path.write_text(f"{detector}:{config.output.format}")
            (config.output.directory / f"{config.output.prefix}_{detector}.json").write_text("{}")
            output_paths[detector] = artifact_path
        return type("SimulationResultStub", (), {"output_paths": output_paths, "config": config})()


class TestNoiseAdapter:
    """Tests for direct adapter behavior."""

    def test_run_builds_public_noise_config(self, tmp_path: Path):
        """The adapter should pass gwmock orchestration inputs through NoiseConfig."""
        backend = FakeNoiseBackend()
        adapter = NoiseAdapter.from_backend(backend)
        psd_path = tmp_path / "psd.txt"
        psd_path.write_text("0 1\n1 1\n")

        result = adapter.run(
            detectors=["H1", "L1"],
            duration=TEST_DURATION,
            sampling_frequency=TEST_SAMPLING_FREQUENCY,
            output_directory=tmp_path,
            output_prefix="segment-0",
            output_format="npy",
            gps_start=TEST_GPS_START,
            channel_prefix="TEST",
            seed=TEST_SEED,
            psd_file=psd_path,
            low_frequency_cutoff=10.0,
            high_frequency_cutoff=100.0,
        )

        config = backend.run_calls[0]
        assert config.detectors == ["H1", "L1"]
        assert config.duration == TEST_DURATION
        assert config.sampling_frequency == TEST_SAMPLING_FREQUENCY
        assert config.output.directory == tmp_path
        assert config.output.prefix == "segment-0"
        assert config.output.format == "npy"
        assert config.output.gps_start == TEST_GPS_START
        assert config.output.channel_prefix == "TEST"
        assert config.seed == TEST_SEED
        assert config.psd_file == psd_path
        assert result.output_paths["H1"] == tmp_path / "segment-0_H1.npy"


class TestUpstreamNoiseSimulator:
    """Tests for the gwmock orchestration wrapper."""

    @staticmethod
    def _batch(file_name: str, arguments: dict | None = None) -> SimulationBatch:
        return SimulationBatch(
            simulator_name="noise",
            simulator_config=SimulatorConfig(
                class_="gwmock.noise.UpstreamNoiseSimulator",
                arguments={},
                output=SimulatorOutputConfig(file_name=file_name, arguments=arguments or {}),
            ),
            globals_config=type(
                "GlobalsStub",
                (),
                {
                    "working_directory": ".",
                    "output_directory": None,
                    "metadata_directory": None,
                    "simulator_arguments": {},
                    "output_arguments": {},
                },
            )(),
            batch_index=0,
        )

    def test_simulate_uses_batch_context_and_deterministic_segment_seed(self, tmp_path: Path):
        """Each batch should map to one deterministic public run(config) call."""
        backend = FakeNoiseBackend()
        simulator = UpstreamNoiseSimulator(
            duration=SEGMENT_DURATION,
            sampling_frequency=128.0,
            detectors=["H1", "L1"],
            seed=BASE_SEED,
            psd_file=tmp_path / "psd.npy",
            noise_adapter=NoiseAdapter.from_backend(backend),
        )

        batch0 = self._batch("noise-{{counter}}.npy")
        batch1 = self._batch("noise-{{counter}}.npy")

        simulator.set_batch_context(batch=batch0, output_directory=tmp_path, overwrite=False)
        result0 = simulator.simulate()
        simulator.update_state()

        simulator.set_batch_context(batch=batch1, output_directory=tmp_path, overwrite=False)
        result1 = simulator.simulate()

        config0, config1 = backend.run_calls
        assert config0.seed == BASE_SEED
        assert config1.seed == BASE_SEED + 1
        assert config0.output.prefix == "noise-0"
        assert config1.output.prefix == "noise-1"
        assert config0.output.gps_start == 0.0
        assert config1.output.gps_start == SEGMENT_DURATION
        assert result0.output_paths["H1"] == tmp_path / "noise-0_H1.npy"
        assert result1.output_paths["L1"] == tmp_path / "noise-1_L1.npy"

    def test_set_batch_context_rejects_unsupported_output_arguments(self, tmp_path: Path):
        """Unsupported gwmock-only output arguments should fail loudly for the adapter path."""
        simulator = UpstreamNoiseSimulator(noise_adapter=NoiseAdapter.from_backend(FakeNoiseBackend()))
        batch = self._batch("noise.npy", {"channel": "H1:STRAIN"})

        with pytest.raises(ValueError, match="Unsupported keys: channel"):
            simulator.set_batch_context(batch=batch, output_directory=tmp_path, overwrite=False)
