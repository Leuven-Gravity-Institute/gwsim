"""Unit tests for CBCSignalSimulator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from astropy.units import Quantity

from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList
from gwmock.signal.cbc import DEFAULT_SIGNAL_SIMULATOR_DETECTORS, CBCSignalSimulator


def _write_catalogue(path: Path, data: dict[str, list[float]]) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        for name, values in data.items():
            group.create_dataset(name, data=values)


@pytest.fixture
def cbc_signal_simulator_with_mocks(tmp_path):
    """Create a CBCSignalSimulator with mixin initialization patched out."""
    dummy_file = tmp_path / "dummy_file.h5"
    _write_catalogue(dummy_file, {"tc": [100.0], "m1": [30.0], "m2": [25.0], "z": [0.1]})
    mock_adapter = MagicMock()
    mock_adapter.detector_names = ("H1",)

    with (
        patch("gwmock.mixin.population_reader.PopulationReaderMixin.__init__", return_value=None),
        patch("gwmock.mixin.time_series.TimeSeriesMixin.__init__", return_value=None),
        patch("gwmock.simulator.base.Simulator.__init__", return_value=None),
        patch("gwmock.signal.cbc.SignalAdapter.from_source_type", return_value=mock_adapter) as mock_from_source_type,
    ):
        simulator = CBCSignalSimulator(
            population_file=str(dummy_file),
            waveform_model="IMRPhenomD",
            waveform_arguments={"reference_frequency": 50.0},
            start_time=0,
            duration=100.0,
            sampling_frequency=4096,
            detectors=["H1"],
            minimum_frequency=5.0,
            source_type="bbh",
        )

    simulator.signal_adapter = mock_adapter
    simulator.waveform_model = "IMRPhenomD"
    simulator.waveform_arguments = {"reference_frequency": 50.0}
    simulator.start_time = Quantity(0, unit="s")
    simulator.duration = Quantity(100.0, unit="s")
    simulator.sampling_frequency = Quantity(4096, unit="Hz")
    simulator.minimum_frequency = 5.0
    simulator.source_type = "bbh"
    simulator.earth_rotation = True
    simulator.counter = 0
    simulator.detectors = ["H1"]
    simulator.population_file = Path(dummy_file)
    simulator.population_parameter_name_mapper = {}
    simulator.population_sort_by = "coa_time"
    simulator.population_cache_dir = Path.home() / ".gwmock" / "population"
    simulator.population_download_timeout = 300
    simulator._population_loader_metadata = {}
    simulator._population_loaded_by_gwmock_pop = True
    simulator.max_samples = np.inf
    simulator.num_of_channels = 1
    simulator.dtype = np.float64

    return simulator, mock_adapter, mock_from_source_type


class TestCBCSignalSimulator:
    """Test CBCSignalSimulator initialization and population mapping."""

    def test_init_success_sorts_by_coalescence_time(self, tmp_path: Path):
        """CBC populations should be sorted by canonical ``coa_time``."""
        dummy_file = tmp_path / "cbc_test.h5"
        _write_catalogue(dummy_file, {"tc": [100.0, 50.0], "m1": [30.0, 25.0], "m2": [25.0, 20.0], "z": [0.2, 0.1]})
        mock_adapter = MagicMock()
        mock_adapter.detector_names = ("H1",)

        with patch("gwmock.signal.cbc.SignalAdapter.from_source_type", return_value=mock_adapter):
            simulator = CBCSignalSimulator(
                population_file=str(dummy_file),
                waveform_model="IMRPhenomD",
                start_time=0,
                duration=100.0,
                sampling_frequency=4096,
                detectors=["H1"],
                minimum_frequency=5.0,
            )

        assert simulator is not None
        assert list(simulator.population_data["coa_time"]) == [50.0, 100.0]
        assert simulator.source_type == "bbh"

    def test_init_maps_population_columns_to_canonical_signal_names(self, tmp_path: Path):
        """Legacy CBC population columns should be renamed to gwmock-signal canonical names."""
        dummy_file = tmp_path / "cbc_mapping.h5"
        _write_catalogue(dummy_file, {"tc": [100.0], "m1": [30.0], "m2": [25.0], "z": [0.1]})
        mock_adapter = MagicMock()
        mock_adapter.detector_names = ("H1",)

        with patch("gwmock.signal.cbc.SignalAdapter.from_source_type", return_value=mock_adapter):
            simulator = CBCSignalSimulator(
                population_file=str(dummy_file),
                waveform_model="IMRPhenomD",
                start_time=0,
                duration=100.0,
                sampling_frequency=4096,
                detectors=["H1"],
                minimum_frequency=5.0,
            )

        assert "detector_frame_mass_1" in simulator.population_data.columns
        assert "detector_frame_mass_2" in simulator.population_data.columns
        assert "coa_time" in simulator.population_data.columns
        assert "redshift" in simulator.population_data.columns
        assert "m1" not in simulator.population_data.columns

    def test_init_uses_default_detector_network_when_detectors_missing(self, tmp_path: Path):
        """Omitted or empty detectors should fall back to a non-empty HLV network."""
        dummy_file = tmp_path / "dummy_file.h5"
        _write_catalogue(dummy_file, {"tc": [100.0], "m1": [30.0], "m2": [25.0], "z": [0.1]})
        mock_adapter = MagicMock()
        mock_adapter.detector_names = tuple(DEFAULT_SIGNAL_SIMULATOR_DETECTORS)

        expected = list(DEFAULT_SIGNAL_SIMULATOR_DETECTORS)
        with (
            patch("gwmock.mixin.population_reader.PopulationReaderMixin.__init__", return_value=None),
            patch("gwmock.mixin.time_series.TimeSeriesMixin.__init__", return_value=None),
            patch("gwmock.simulator.base.Simulator.__init__", return_value=None),
            patch(
                "gwmock.signal.cbc.SignalAdapter.from_source_type", return_value=mock_adapter
            ) as mock_from_source_type,
        ):
            for detectors in (None, []):
                mock_from_source_type.reset_mock()
                CBCSignalSimulator(
                    population_file=str(dummy_file),
                    waveform_model="IMRPhenomD",
                    detectors=detectors,
                    start_time=0,
                    duration=100.0,
                    sampling_frequency=4096,
                    minimum_frequency=5.0,
                    source_type="bbh",
                )
                mock_from_source_type.assert_called_once_with(
                    source_type="bbh",
                    waveform_model="IMRPhenomD",
                    detectors=expected,
                )


class TestCBCSignalSimulatorSimulate:
    """Test the adapter-backed `_simulate` method."""

    def test_simulate_returns_timeseries_list(self, cbc_signal_simulator_with_mocks):
        """`_simulate` should return an empty list when the population is exhausted."""
        simulator, _, _ = cbc_signal_simulator_with_mocks
        with patch.object(simulator, "get_next_injection_parameters", return_value=None):
            result = simulator._simulate()
        assert isinstance(result, TimeSeriesList)
        assert len(result) == 0

    def test_simulate_calls_signal_adapter_with_segment_context(self, cbc_signal_simulator_with_mocks):
        """`_simulate` should forward event parameters and segment settings to the adapter."""
        simulator, mock_adapter, _ = cbc_signal_simulator_with_mocks
        parameters = {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "coa_time": 50.0,
            "distance": 1000.0,
            "inclination": 0.5,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }
        strain = TimeSeries(
            data=np.ones((1, 8), dtype=np.float64),
            start_time=40.0,
            sampling_frequency=4.0,
        )

        with patch.object(simulator, "get_next_injection_parameters", side_effect=[parameters, None]):
            mock_adapter.simulate.return_value = strain
            result = simulator._simulate()

        assert len(result) == 1
        call = mock_adapter.simulate.call_args
        assert call.args[0] == parameters
        assert call.kwargs["sampling_frequency"] == 4096.0
        assert call.kwargs["minimum_frequency"] == 5.0
        assert call.kwargs["waveform_arguments"] == {"reference_frequency": 50.0}
        assert call.kwargs["earth_rotation"] is True
        assert result[0].metadata["injection_parameters"] == parameters

    def test_simulate_stops_after_first_chunk_starting_outside_segment(self, cbc_signal_simulator_with_mocks):
        """Once a chunk starts at or after the current segment end, `_simulate` should stop."""
        simulator, mock_adapter, _ = cbc_signal_simulator_with_mocks
        first_parameters = {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "coa_time": 50.0,
            "distance": 1000.0,
            "inclination": 0.5,
            "right_ascension": 1.97,
            "declination": -1.21,
            "polarization_angle": 1.6,
        }
        second_parameters = {
            "detector_frame_mass_1": 31.0,
            "detector_frame_mass_2": 24.0,
            "coa_time": 120.0,
            "distance": 900.0,
            "inclination": 0.4,
            "right_ascension": 1.80,
            "declination": -1.11,
            "polarization_angle": 1.2,
        }
        in_segment = TimeSeries(data=np.ones((1, 8), dtype=np.float64), start_time=40.0, sampling_frequency=4.0)
        outside_segment = TimeSeries(data=np.ones((1, 8), dtype=np.float64), start_time=100.0, sampling_frequency=4.0)

        with patch.object(
            simulator,
            "get_next_injection_parameters",
            side_effect=[first_parameters, second_parameters, None],
        ):
            mock_adapter.simulate.side_effect = [in_segment, outside_segment]
            result = simulator._simulate()

        assert len(result) == 2
        assert result[0].start_time == Quantity(40.0, unit="s")
        assert result[1].start_time == Quantity(100.0, unit="s")


class TestCBCSignalSimulatorMetadata:
    """Test metadata generation."""

    def test_metadata_includes_signal_configuration(self, cbc_signal_simulator_with_mocks):
        """The metadata should expose the adapter-facing signal configuration."""
        simulator, _, _ = cbc_signal_simulator_with_mocks
        metadata = simulator.metadata
        assert metadata["signal"]["arguments"]["waveform_model"] == "IMRPhenomD"
        assert metadata["signal"]["arguments"]["source_type"] == "bbh"
        assert metadata["signal"]["arguments"]["detectors"] == ["H1"]


class TestCBCSignalSimulatorUpdateState:
    """Test state update functionality."""

    def test_update_state_advances_counter_and_start_time(self, cbc_signal_simulator_with_mocks):
        """`update_state` should advance both the segment counter and GPS start time."""
        simulator, _, _ = cbc_signal_simulator_with_mocks
        simulator.update_state()
        assert simulator.counter == 1
        assert simulator.start_time == Quantity(100.0, unit="s")
