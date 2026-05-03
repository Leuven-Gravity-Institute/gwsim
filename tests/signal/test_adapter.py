"""Tests for the gwmock-side gwmock_signal adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.signal.adapter import SignalAdapter


class FakeBackend:
    """Minimal backend exposing the transient-style gwmock_signal interface."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.generate_calls: list[tuple[dict, float, float]] = []

    def generate_polarizations(
        self,
        params: dict,
        sampling_frequency: float,
        minimum_frequency: float,
    ) -> tuple[GWpyTimeSeries, GWpyTimeSeries]:
        """Record the call and return simple test polarizations."""
        self.generate_calls.append((params, sampling_frequency, minimum_frequency))
        hp = GWpyTimeSeries([1.0, 2.0, 3.0, 4.0], t0=8.0, sample_rate=sampling_frequency)
        hc = GWpyTimeSeries([0.5, 0.0, -0.5, -1.0], t0=8.0, sample_rate=sampling_frequency)
        return hp, hc


class TestSignalAdapter:
    """Test the gwmock-side signal adapter."""

    def test_from_source_type_merges_fixed_waveform_arguments(self):
        """The adapter should resolve the backend via source_type and merge fixed waveform args into params."""
        reference_frequency = 50.0
        detector_frame_mass_1 = 30.0
        projected = {
            "H1": GWpyTimeSeries([10.0, 20.0, 30.0, 40.0], t0=8.0, sample_rate=4.0),
            "L1": GWpyTimeSeries([11.0, 21.0, 31.0, 41.0], t0=8.0, sample_rate=4.0),
        }
        parameters = {
            "detector_frame_mass_1": detector_frame_mass_1,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch("gwmock.signal.adapter.project_polarizations_to_network", return_value=projected) as mock_project,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=["H1", "L1"],
            )
            strain = adapter.simulate(
                parameters,
                sampling_frequency=4.0,
                minimum_frequency=5.0,
                waveform_arguments={"reference_frequency": reference_frequency},
            )

        backend = adapter._backend
        assert adapter.detector_names == ("H1", "L1")
        assert backend.generate_calls[0][0]["reference_frequency"] == reference_frequency
        assert backend.generate_calls[0][0]["detector_frame_mass_1"] == detector_frame_mass_1
        assert strain.shape == (2, 4)
        np.testing.assert_allclose(strain[0].value, [10.0, 20.0, 30.0, 40.0])
        mock_project.assert_called_once()

    def test_from_source_type_converts_detector_config_files(self, tmp_path):
        """Detector config paths should be converted into public gwmock_signal detector specs."""
        config_path = tmp_path / "custom.interferometer"
        config_path.write_text("# dummy\n")
        sentinel_detector = SimpleNamespace(name="custom")

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=FakeBackend),
            patch("gwmock.signal.adapter.interferometer_config_to_custom_detector", return_value=sentinel_detector),
            patch(
                "gwmock.signal.adapter.Network.from_detectors",
                return_value=SimpleNamespace(detector_names=(sentinel_detector,)),
            ) as mock_network,
        ):
            adapter = SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model="IMRPhenomD",
                detectors=[str(config_path)],
            )

        assert adapter.detector_names == ("custom",)
        mock_network.assert_called_once_with((sentinel_detector,))
