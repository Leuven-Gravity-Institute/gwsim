"""Unit tests for CBCPopulationReaderMixin."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import pytest
from gwmock_pop import PopulationValidationError

from gwmock.mixin.cbc_population_reader import CBCPopulationReaderMixin
from gwmock.simulator.base import Simulator

EXPECTED_FIRST_COA_TIME = 50.0
EXPECTED_FIRST_DETECTOR_MASS_1 = 25.0
EXPECTED_FIRST_DETECTOR_MASS_2 = 20.0
EXPECTED_FIRST_REDSHIFT = 0.05


class MockCBCSimulator(CBCPopulationReaderMixin, Simulator):
    """Mock simulator combining CBCPopulationReaderMixin and Simulator for integration testing."""

    def simulate(self):
        """Mock simulate method."""
        return "mock_signal"

    def _save_data(self, data, file_name, **kwargs):
        """Mock save data method."""
        pass

    @property
    def metadata(self):
        return super().metadata


def _write_catalogue(path: Path, data: dict[str, list[float]]) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        for name, values in data.items():
            group.create_dataset(name, data=values)


class TestCBCPopulationReaderMixin:
    """Test suite for CBCPopulationReaderMixin."""

    @pytest.fixture
    def detector_frame_catalogue(self):
        """Fixture for detector-frame CBC data."""
        return {
            "tc": [100.0, 50.0, 200.0],
            "m1": [30.0, 25.0, 35.0],
            "m2": [25.0, 20.0, 30.0],
            "z": [0.1, 0.05, 0.2],
        }

    @pytest.fixture
    def source_frame_catalogue(self):
        """Fixture for source-frame CBC data."""
        return {
            "tc": [100.0, 50.0],
            "m1_source": [28.0, 24.0],
            "m2_source": [23.0, 19.0],
            "z": [0.1, 0.05],
        }

    def test_init_success(self, detector_frame_catalogue, tmp_path: Path):
        """Test successful initialization with a valid CBC file."""
        file_path = tmp_path / "cbc_test.h5"
        _write_catalogue(file_path, detector_frame_catalogue)

        simulator = MockCBCSimulator(file_path, start_time=0, duration=100)

        assert simulator.population_data is not None
        assert list(simulator.population_data["coa_time"]) == [50.0, 100.0, 200.0]
        assert "detector_frame_mass_1" in simulator.population_data.columns
        assert "detector_frame_mass_2" in simulator.population_data.columns
        assert "redshift" in simulator.population_data.columns
        assert "m1" not in simulator.population_data.columns

    def test_parameter_name_mapping(self, detector_frame_catalogue, tmp_path: Path):
        """Test CBC parameter name mapping."""
        file_path = tmp_path / "cbc_test.h5"
        _write_catalogue(file_path, detector_frame_catalogue)

        simulator = MockCBCSimulator(file_path, start_time=0, duration=100)

        assert simulator.population_data["detector_frame_mass_1"].iloc[0] == EXPECTED_FIRST_DETECTOR_MASS_1
        assert simulator.population_data["detector_frame_mass_2"].iloc[0] == EXPECTED_FIRST_DETECTOR_MASS_2
        assert simulator.population_data["redshift"].iloc[0] == EXPECTED_FIRST_REDSHIFT

    def test_post_process_compute_masses(self, source_frame_catalogue, tmp_path: Path):
        """Test gwmock-pop computes detector-frame masses from source masses and redshift."""
        file_path = tmp_path / "cbc_srcmass.h5"
        _write_catalogue(file_path, source_frame_catalogue)

        simulator = MockCBCSimulator(file_path, start_time=0, duration=100)

        assert simulator.population_data["detector_frame_mass_1"].iloc[0] == pytest.approx(24.0 * 1.05)
        assert simulator.population_data["detector_frame_mass_2"].iloc[0] == pytest.approx(19.0 * 1.05)

    def test_post_process_missing_srcmass_raises_error(self, tmp_path: Path):
        """Missing mass information should surface gwmock-pop validation errors."""
        file_path = tmp_path / "cbc_error.h5"
        _write_catalogue(file_path, {"tc": [100.0], "z": [0.1]})

        with pytest.raises(PopulationValidationError, match="detector_frame_mass_1"):
            MockCBCSimulator(file_path, start_time=0, duration=100)

    def test_url_population_file(self):
        """URL paths should be delegated directly to gwmock-pop."""
        url = "https://example.com/cbc_population.h5"

        with patch("gwmock.mixin.population_reader.FilePopulationLoader") as mock_loader:
            mock_loader.return_value.parameter_names = ["coa_time", "detector_frame_mass_1", "detector_frame_mass_2"]
            tmp_file_path = Path(tempfile.gettempdir()) / "downloaded_cbc.h5"
            mock_loader.return_value.metadata = {
                "original_path": url,
                "resolved_path": tmp_file_path,
                "source_type": "bbh",
                "fetch": {"scheme": "https"},
            }
            mock_loader.return_value._catalogue = {
                "coa_time": [100.0],
                "detector_frame_mass_1": [30.0],
                "detector_frame_mass_2": [25.0],
            }

            simulator = MockCBCSimulator(url, start_time=0, duration=100)

            assert simulator.population_data is not None
            assert simulator.population_file == tmp_file_path
            mock_loader.assert_called_once_with(
                source_type="bbh",
                path=url,
                column_map=None,
                cache_dir=simulator.population_cache_dir,
                download_timeout=300,
            )

    def test_get_injection_parameters(self, detector_frame_catalogue, tmp_path: Path):
        """Test getting injection parameters for CBC."""
        file_path = tmp_path / "cbc_test.h5"
        _write_catalogue(file_path, detector_frame_catalogue)

        simulator = MockCBCSimulator(file_path, start_time=0, duration=100)
        params = simulator.get_next_injection_parameters()

        assert params is not None
        assert params["coa_time"] == EXPECTED_FIRST_COA_TIME
        assert "detector_frame_mass_1" in params
        assert "detector_frame_mass_2" in params
        assert "redshift" in params
