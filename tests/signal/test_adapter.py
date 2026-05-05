"""Tests for the gwmock-side gwmock_signal adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from gwmock_signal import DetectorStrainStack
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwmock.signal.adapter import SignalAdapter


class FakeBackend:
    """Minimal backend exposing the public gwmock_signal ``simulate`` interface."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model
        self.required_params = frozenset({"coa_time"})
        self.simulate_calls: list[tuple[dict, tuple[str, ...], float, float]] = []

    def register_waveform_model(self, name: str, factory) -> None:
        self.waveform_model = name
        self._registered = (name, factory)

    def simulate(
        self,
        params: dict,
        detector_names: tuple[str, ...],
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        """Record the call and return simple per-detector strains."""
        _ = background, earth_rotation, interpolate_if_offset
        self.simulate_calls.append((params, detector_names, sampling_frequency, minimum_frequency))
        return DetectorStrainStack.from_mapping(
            detector_names,
            {
                detector: GWpyTimeSeries(
                    [10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index], t0=8.0, sample_rate=sampling_frequency
                )
                for index, detector in enumerate(detector_names)
            },
        )


class NoRegistrationBackend:
    """Backend without waveform registration support."""

    def __init__(self, *, waveform_model: str) -> None:
        self.waveform_model = waveform_model


class TestSignalAdapter:
    """Test the gwmock-side signal adapter."""

    def test_from_source_type_merges_fixed_waveform_arguments(self):
        """The adapter should resolve the backend via source_type and merge fixed waveform args into params."""
        reference_frequency = 50.0
        detector_frame_mass_1 = 30.0
        projected = {
            "H1": [10.0, 20.0, 30.0, 40.0],
            "L1": [11.0, 21.0, 31.0, 41.0],
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
        assert backend.simulate_calls[0][0]["reference_frequency"] == reference_frequency
        assert backend.simulate_calls[0][0]["detector_frame_mass_1"] == detector_frame_mass_1
        assert strain.shape == (2, 4)
        np.testing.assert_allclose(strain[0].value, projected["H1"])

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

    def test_from_source_type_callable_requires_waveform_factory(self):
        """Backends without ``_waveform_factory`` cannot host a callable ``waveform_model``."""

        def _dummy(**_kwargs):
            return {"plus": None, "cross": None}

        with (
            patch("gwmock.signal.adapter.resolve_simulator_backend", return_value=NoRegistrationBackend),
            pytest.raises(TypeError, match="Callable waveform_model"),
        ):
            SignalAdapter.from_source_type(
                source_type="bbh",
                waveform_model=_dummy,
                detectors=["H1"],
            )

    def test_from_source_type_callable_waveform_with_real_backend(self):
        """A callable should be registered and invoked on the public CBC backend."""
        from gwpy.timeseries import TimeSeries

        n = 32
        merged_calls: list[dict] = []

        def custom_waveform(*, waveform_model, tc, sampling_frequency, minimum_frequency, **params):
            merged_calls.append(
                {"waveform_model": waveform_model, "tc": tc, "minimum_frequency": minimum_frequency, **params}
            )
            t0 = float(tc) - (n / 2) / sampling_frequency
            hp = TimeSeries(np.linspace(0, 1, n), t0=t0, sample_rate=sampling_frequency)
            hc = TimeSeries(np.linspace(1, 0, n), t0=t0, sample_rate=sampling_frequency)
            return {"plus": hp, "cross": hc}

        parameters = {
            "detector_frame_mass_1": 30.0,
            "detector_frame_mass_2": 25.0,
            "coa_time": 12.0,
            "distance": 800.0,
            "inclination": 0.3,
            "right_ascension": 1.1,
            "declination": -0.8,
            "polarization_angle": 0.2,
        }

        adapter = SignalAdapter.from_source_type(
            source_type="bbh",
            waveform_model=custom_waveform,
            detectors=["H1", "L1"],
        )
        strain = adapter.simulate(
            parameters,
            sampling_frequency=4.0,
            minimum_frequency=5.0,
            waveform_arguments={"reference_frequency": 50.0},
        )

        assert strain.shape == (2, n)
        backend = adapter._backend
        assert isinstance(backend.waveform_model, str)
        assert backend.waveform_model.startswith("__gwmock_custom__")
        assert merged_calls[0]["reference_frequency"] == 50.0
        assert merged_calls[0]["detector_frame_mass_1"] == 30.0
