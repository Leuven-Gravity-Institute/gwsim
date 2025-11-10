"""Unit tests for the Simulator base class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gwsim.simulator.base import Simulator


class MockSimulator(Simulator):
    """Mock simulator for testing base functionality."""

    def simulate(self) -> int:
        """Return the current counter value."""
        return self.counter

    def _save_data(self, data: int, file_name: str | Path, **kwargs) -> None:
        """Save data as JSON."""
        with open(file_name, "w") as f:
            json.dump(data, f)


@pytest.fixture
def simulator() -> MockSimulator:
    """Fixture for a basic simulator instance."""
    return MockSimulator(max_samples=5)


@pytest.fixture
def simulator_with_attrs() -> MockSimulator:
    """Fixture for simulator with additional attributes for template testing."""
    sim = MockSimulator(max_samples=10)
    sim.detector = np.array(["H1", "L1"])
    sim.duration = np.array([4, 8])
    return sim


class TestSimulatorInitialization:
    """Test Simulator initialization."""

    def test_init_with_max_samples(self):
        """Test initialization with max_samples."""
        sim = MockSimulator(max_samples=10)
        assert sim.max_samples == 10

    def test_init_with_none_max_samples(self):
        """Test initialization with None max_samples (infinite)."""
        sim = MockSimulator(max_samples=None)
        assert sim.max_samples == np.inf

    def test_init_without_max_samples(self):
        """Test initialization without max_samples defaults to infinite."""
        sim = MockSimulator()
        assert sim.max_samples == np.inf


class TestSimulatorProperties:
    """Test Simulator properties."""

    def test_max_samples_getter_setter(self, simulator: MockSimulator):
        """Test max_samples property."""
        assert simulator.max_samples == 5
        simulator.max_samples = 10
        assert simulator.max_samples == 10

    def test_max_samples_setter_validation(self, simulator: MockSimulator):
        """Test max_samples setter validation."""
        with pytest.raises(ValueError, match="Max samples cannot be negative"):
            simulator.max_samples = -1

    def test_state_property(self, simulator: MockSimulator):
        """Test state property includes counter."""
        state = simulator.state
        assert "counter" in state
        assert state["counter"] == 0

    def test_state_setter(self, simulator: MockSimulator):
        """Test state setter."""
        simulator.state = {"counter": 5}
        assert simulator.counter == 5

    def test_state_setter_invalid_key(self, simulator: MockSimulator):
        """Test state setter with invalid key."""
        with pytest.raises(ValueError, match="not registered as a state attribute"):
            simulator.state = {"invalid_key": 42}

    def test_metadata_property(self, simulator: MockSimulator):
        """Test metadata property."""
        metadata = simulator.metadata
        assert metadata["max_samples"] == 5
        assert metadata["counter"] == 0
        assert "version" in metadata


class TestSimulatorIterator:
    """Test Simulator iterator protocol."""

    def test_iterator_protocol(self, simulator: MockSimulator):
        """Test __iter__ and __next__."""
        iterator = iter(simulator)
        assert iterator is simulator

        # Generate samples
        samples = list(simulator)
        assert len(samples) == 5
        assert samples == [0, 1, 2, 3, 4]

        # Counter should be updated
        assert simulator.counter == 5

    def test_iterator_stopiteration(self, simulator: MockSimulator):
        """Test StopIteration when max_samples reached."""
        samples = []
        for sample in simulator:
            samples.append(sample)
        assert len(samples) == 5

        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(simulator)

    def test_iterator_infinite(self):
        """Test iterator with infinite max_samples."""
        sim = MockSimulator(max_samples=None)
        iterator = iter(sim)
        # Generate a few samples
        for i in range(3):
            assert next(iterator) == i
        assert sim.counter == 3


class TestSimulatorFileIO:
    """Test Simulator file I/O methods."""

    def test_save_state(self, simulator: MockSimulator):
        """Test save_state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.json"

            simulator.save_state(file_path)
            assert file_path.exists()

            with open(file_path) as f:
                state = json.load(f)
            assert state["counter"] == 0

    def test_save_state_invalid_extension(self, simulator: MockSimulator):
        """Test save_state with invalid file extension."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            simulator.save_state("test.txt")

    def test_save_state_overwrite_false(self, simulator: MockSimulator):
        """Test save_state overwrite=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.json"

            simulator.save_state(file_path)
            with pytest.raises(FileExistsError):
                simulator.save_state(file_path, overwrite=False)

    def test_save_state_overwrite_true(self, simulator: MockSimulator):
        """Test save_state overwrite=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.json"

            simulator.save_state(file_path)
            simulator.counter = 1
            simulator.save_state(file_path, overwrite=True)

            with open(file_path) as f:
                state = json.load(f)
            assert state["counter"] == 1

    def test_load_state(self, simulator: MockSimulator):
        """Test load_state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "state.json"

            # Save state
            simulator.counter = 3
            simulator.save_state(file_path)

            # Load into new simulator
            new_sim = MockSimulator()
            new_sim.load_state(file_path)
            assert new_sim.counter == 3

    def test_load_state_file_not_found(self, simulator: MockSimulator):
        """Test load_state with non-existent file."""
        with pytest.raises(FileNotFoundError):
            simulator.load_state("nonexistent.json")

    def test_load_state_invalid_extension(self, simulator: MockSimulator):
        """Test load_state with invalid extension."""

        with pytest.raises(ValueError, match="Unsupported file format"):
            simulator.load_state("test.txt")

    def test_save_metadata(self, simulator: MockSimulator):
        """Test save_metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "metadata.json"

            simulator.save_metadata(file_path)
            assert file_path.exists()

            with open(file_path) as f:
                metadata = json.load(f)
            assert metadata["counter"] == 0
            assert "version" in metadata


class TestSimulatorSaveData:
    """Test Simulator save_data method."""

    def test_save_data_single_file(self, simulator: MockSimulator):
        """Test save_data with single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.json"

            data = 42
            simulator.save_data(data, file_path)
            assert file_path.exists()

            with open(file_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == 42

    def test_save_data_with_template(self, simulator: MockSimulator):
        """Test save_data with template resolution."""
        simulator.detector = "H1"
        simulator.duration = 4

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.json"
            data = 100
            simulator.save_data(data, template)

            expected_path = Path(f"{temp_dir}/H1-4.json")
            assert expected_path.exists()

            with open(expected_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == 100

    def test_save_data_array_files(self, simulator_with_attrs: MockSimulator):
        """Test save_data with array of files."""
        sim = simulator_with_attrs
        # data shape should match file_name array shape (2, 2)
        data = np.array([[1, 2], [3, 4]], dtype=object)

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.json"
            sim.save_data(data, template)

            # Check files exist
            assert Path(f"{temp_dir}/H1-4.json").exists()
            assert Path(f"{temp_dir}/H1-8.json").exists()
            assert Path(f"{temp_dir}/L1-4.json").exists()
            assert Path(f"{temp_dir}/L1-8.json").exists()

            # Check contents
            with open(f"{temp_dir}/H1-4.json") as f:
                assert json.load(f) == 1
            with open(f"{temp_dir}/L1-8.json") as f:
                assert json.load(f) == 4

    def test_save_data_array_shape_mismatch(self, simulator_with_attrs: MockSimulator):
        """Test save_data with mismatched data shape."""
        sim = simulator_with_attrs
        # Wrong shape: should be (2, 2) but is (2,)
        data = np.array([1, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            template = f"{temp_dir}/{{{{detector}}}}-{{{{duration}}}}.json"
            with pytest.raises(
                ValueError, match="Data must have equal or more dimensions than the resolved file names"
            ):
                sim.save_data(data, template)

    def test_save_data_overwrite_false(self, simulator: MockSimulator):
        """Test save_data overwrite=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.json"

            simulator.save_data(1, file_path)
            with pytest.raises(FileExistsError):
                simulator.save_data(2, file_path, overwrite=False)

    def test_save_data_overwrite_true(self, simulator: MockSimulator):
        """Test save_data overwrite=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.json"

            simulator.save_data(1, file_path)
            simulator.save_data(2, file_path, overwrite=True)

            with open(file_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == 2
