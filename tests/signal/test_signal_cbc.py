"""Unit tests for CBCSignalSimulator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py

from gwmock.signal.cbc import CBCSignalSimulator


def _write_catalogue(path: Path, data: dict[str, list[float]]) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        for name, values in data.items():
            group.create_dataset(name, data=values)


class TestCBCSignalSimulator:
    """Test CBCSignalSimulator initialization and population mapping."""

    def test_init_success_sorts_by_coalescence_time(self, tmp_path: Path):
        """CBC populations should be sorted by canonical ``coa_time``."""
        dummy_file = tmp_path / "cbc_test.h5"
        _write_catalogue(dummy_file, {"tc": [100.0, 50.0], "m1": [30.0, 25.0], "m2": [25.0, 20.0], "z": [0.2, 0.1]})
        mock_adapter = MagicMock()
        mock_adapter.detector_names = ("H1",)

        with patch("gwmock.signal.base.SignalAdapter.from_source_type", return_value=mock_adapter):
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

        with patch("gwmock.signal.base.SignalAdapter.from_source_type", return_value=mock_adapter):
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
