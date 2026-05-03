"""Unit tests for CBCSignalSimulator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gwmock.signal.cbc import CBCSignalSimulator


class TestCBCSignalSimulator:
    """Test CBCSignalSimulator initialization and population mapping."""

    def test_init_success_sorts_by_coalescence_time(self, tmp_path):
        """CBC populations should be sorted by canonical ``coa_time``."""
        dummy_file = tmp_path / "cbc_test.h5"
        dummy_file.write_bytes(b"dummy")
        mock_adapter = MagicMock()
        mock_adapter.detector_names = ("H1",)

        with (
            patch("h5py.File") as mock_file,
            patch("gwmock.signal.base.SignalAdapter.from_source_type", return_value=mock_adapter),
        ):
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                ("tc", type("MockDataset", (), {"__getitem__": lambda self, idx: [100.0, 50.0]})()),
                ("m1", type("MockDataset", (), {"__getitem__": lambda self, idx: [30.0, 25.0]})()),
                ("m2", type("MockDataset", (), {"__getitem__": lambda self, idx: [25.0, 20.0]})()),
            ]
            mock_f.attrs.items.return_value = []

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

    def test_init_maps_population_columns_to_canonical_signal_names(self, tmp_path):
        """Legacy CBC population columns should be renamed to gwmock-signal canonical names."""
        dummy_file = tmp_path / "cbc_mapping.h5"
        dummy_file.write_bytes(b"dummy")
        mock_adapter = MagicMock()
        mock_adapter.detector_names = ("H1",)

        with (
            patch("h5py.File") as mock_file,
            patch("gwmock.signal.base.SignalAdapter.from_source_type", return_value=mock_adapter),
        ):
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                ("tc", type("MockDataset", (), {"__getitem__": lambda self, idx: [100.0]})()),
                ("m1", type("MockDataset", (), {"__getitem__": lambda self, idx: [30.0]})()),
                ("m2", type("MockDataset", (), {"__getitem__": lambda self, idx: [25.0]})()),
                ("z", type("MockDataset", (), {"__getitem__": lambda self, idx: [0.1]})()),
            ]
            mock_f.attrs.items.return_value = []

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
