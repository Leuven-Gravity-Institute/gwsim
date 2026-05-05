"""Tests for the gwmock population adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from gwmock_pop import CBC_PARAMETER_NAMES, ExternalPopulationLoader, GWPopSimulator

from gwmock.population import PopulationAdapter

EXPECTED_SAMPLE_COUNT = 2


class MockGWPopBackend:
    """Protocol-compatible simulator backend for adapter tests."""

    parameter_names = (
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "redshift",
        "coa_time",
    )
    source_type = "bbh"

    def simulate(self, n_samples: int, **kwargs):
        if n_samples != EXPECTED_SAMPLE_COUNT:
            raise AssertionError("Unexpected n_samples for test backend.")
        return {
            "detector_frame_mass_1": np.array([30.0, 32.0]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "redshift": np.array([0.1, 0.2]),
            "coa_time": np.array([1000.0, 1001.0]),
        }


class MockExternalPopulationLoader:
    """Protocol-compatible file loader backend for adapter tests."""

    parameter_names: ClassVar[tuple[str, ...]] = (
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "inclination",
        "coa_time",
    )
    source_type = "bns"
    metadata: ClassVar[dict[str, str]] = {"resolved_path": str(Path(tempfile.gettempdir()) / "catalog.h5")}

    def simulate(self, n_samples: int, **kwargs):
        if n_samples != EXPECTED_SAMPLE_COUNT:
            raise AssertionError("Unexpected n_samples for test loader.")
        return {
            "detector_frame_mass_1": np.array([1.4, 1.35]),
            "detector_frame_mass_2": np.array([1.3, 1.25]),
            "inclination": np.array([0.2, 0.3]),
            "coa_time": np.array([2000.0, 2001.0]),
        }


class TestPopulationAdapter:
    """Test suite for population adapter behavior."""

    def test_from_backend_accepts_gwpop_simulator(self):
        """Simulator-backed batches are sliced into per-event dictionaries."""
        backend = MockGWPopBackend()

        assert isinstance(backend, GWPopSimulator)

        adapter = PopulationAdapter.from_backend(backend, n_samples=EXPECTED_SAMPLE_COUNT)

        events = list(adapter)

        assert len(adapter) == EXPECTED_SAMPLE_COUNT
        assert adapter.source_type == "bbh"
        assert adapter.parameter_names == backend.parameter_names
        assert list(events[0].keys()) == list(backend.parameter_names)
        assert set(adapter.parameter_names).issubset(CBC_PARAMETER_NAMES)
        assert events == [
            {
                "detector_frame_mass_1": 30.0,
                "detector_frame_mass_2": 20.0,
                "redshift": 0.1,
                "coa_time": 1000.0,
            },
            {
                "detector_frame_mass_1": 32.0,
                "detector_frame_mass_2": 21.0,
                "redshift": 0.2,
                "coa_time": 1001.0,
            },
        ]

    def test_from_backend_accepts_external_population_loader(self):
        """Loader-backed batches use the same adapter boundary."""
        loader = MockExternalPopulationLoader()

        assert isinstance(loader, GWPopSimulator)
        assert isinstance(loader, ExternalPopulationLoader)

        adapter = PopulationAdapter.from_backend(loader, n_samples=EXPECTED_SAMPLE_COUNT)

        assert adapter.source_type == "bns"
        assert adapter.metadata == loader.metadata
        assert list(adapter.iter_event_parameters()) == [
            {
                "detector_frame_mass_1": 1.4,
                "detector_frame_mass_2": 1.3,
                "inclination": 0.2,
                "coa_time": 2000.0,
            },
            {
                "detector_frame_mass_1": 1.35,
                "detector_frame_mass_2": 1.25,
                "inclination": 0.3,
                "coa_time": 2001.0,
            },
        ]

    def test_from_mapping_preserves_mapping_order_without_renaming(self):
        """Direct population mappings keep their canonical key order untouched."""
        population_mapping = {
            "detector_frame_mass_1": np.array([35.0, 36.0]),
            "detector_frame_mass_2": np.array([25.0, 26.0]),
            "polarization_angle": np.array([0.4, 0.5]),
        }

        adapter = PopulationAdapter.from_mapping(population_mapping, source_type="bbh")

        assert adapter.parameter_names == tuple(population_mapping.keys())
        assert list(adapter.get_event_parameters(1).keys()) == list(population_mapping.keys())
        assert adapter.get_event_parameters(1) == {
            "detector_frame_mass_1": 36.0,
            "detector_frame_mass_2": 26.0,
            "polarization_angle": 0.5,
        }

    def test_from_mapping_rejects_mismatched_lengths(self):
        """All parameter arrays must describe the same number of events."""
        with pytest.raises(ValueError, match="same number of samples"):
            PopulationAdapter.from_mapping(
                {
                    "detector_frame_mass_1": np.array([30.0, 32.0]),
                    "detector_frame_mass_2": np.array([20.0]),
                },
                source_type="bbh",
            )

    def test_from_mapping_rejects_key_order_mismatches(self):
        """Explicit parameter order must match the mapping keys exactly."""
        with pytest.raises(ValueError, match="match parameter_names in the same order"):
            PopulationAdapter.from_mapping(
                {
                    "detector_frame_mass_1": np.array([30.0]),
                    "detector_frame_mass_2": np.array([20.0]),
                },
                source_type="bbh",
                parameter_names=("detector_frame_mass_2", "detector_frame_mass_1"),
            )
