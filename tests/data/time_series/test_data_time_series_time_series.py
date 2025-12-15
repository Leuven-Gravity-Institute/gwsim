"""Unit tests for the TimeSeries class."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.units import Quantity

from gwsim.data.time_series.time_series import TimeSeries
from gwsim.data.time_series.time_series_list import TimeSeriesList


@pytest.fixture
def sample_timeseries() -> TimeSeries:
    """Fixture for a sample TimeSeries instance."""
    np.random.seed(42)
    data = np.random.rand(2, 1024)  # 2 channels, 1024 samples
    start_time = Quantity(1234567890, unit="s")
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


@pytest.fixture
def small_timeseries() -> TimeSeries:
    """Fixture for a smaller TimeSeries for injection tests."""
    data = np.ones((2, 512))  # 2 channels, 512 samples
    start_time = Quantity(1234567890.1, unit="s")  # Overlaps with sample
    sampling_frequency = Quantity(4096, unit="Hz")
    return TimeSeries(data=data, start_time=start_time, sampling_frequency=sampling_frequency)


class TestTimeSeriesInitialization:
    """Test TimeSeries initialization."""

    def test_init_with_valid_data(self, sample_timeseries: TimeSeries):
        """Test initialization with valid 2D array."""
        assert sample_timeseries.num_of_channels == 2
        assert sample_timeseries.dtype == np.float64
        assert len(sample_timeseries) == 2

    def test_init_with_int_start_time(self):
        """Test initialization with int start_time."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100)
        assert ts.start_time == Quantity(1000, unit="s")

    def test_init_with_float_sampling_freq(self):
        """Test initialization with float sampling_frequency."""
        data = np.random.rand(1, 100)
        ts = TimeSeries(data, start_time=1000, sampling_frequency=100.0)
        assert ts.sampling_frequency == Quantity(100.0, unit="Hz")

    def test_init_raises_for_1d_data(self):
        """Test that 1D data raises ValueError."""
        data = np.random.rand(100)
        with pytest.raises(ValueError, match="Data must be a 2D"):
            TimeSeries(data, start_time=1000, sampling_frequency=100)


class TestTimeSeriesProperties:
    """Test TimeSeries properties."""

    def test_start_time_property(self, sample_timeseries: TimeSeries):
        """Test start_time property."""
        assert sample_timeseries.start_time == Quantity(1234567890, unit="s")

    def test_duration_property(self, sample_timeseries: TimeSeries):
        """Test duration property."""
        expected_duration = Quantity(1024 / 4096, unit="s")  # samples / freq
        assert sample_timeseries.duration == expected_duration

    def test_end_time_property(self, sample_timeseries: TimeSeries):
        """Test end_time property."""
        assert sample_timeseries.end_time == sample_timeseries.start_time + sample_timeseries.duration

    def test_sampling_frequency_property(self, sample_timeseries: TimeSeries):
        """Test sampling_frequency property."""
        assert sample_timeseries.sampling_frequency == Quantity(4096, unit="Hz")

    def test_time_array_property(self, sample_timeseries: TimeSeries):
        """Test time_array property."""
        times = sample_timeseries.time_array
        assert len(times) == 1024
        assert times[0] == sample_timeseries.start_time


class TestTimeSeriesIndexing:
    """Test TimeSeries indexing and iteration."""

    def test_getitem(self, sample_timeseries: TimeSeries):
        """Test __getitem__."""
        channel = sample_timeseries[0]
        assert hasattr(channel, "value")  # GWpy TimeSeries

    def test_len(self, sample_timeseries: TimeSeries):
        """Test __len__."""
        assert len(sample_timeseries) == 2

    def test_iter(self, sample_timeseries: TimeSeries):
        """Test __iter__."""
        channels = list(sample_timeseries)
        assert len(channels) == 2


class TestTimeSeries1DGetItem:
    """Test 1D __getitem__ (channel indexing)."""

    def test_getitem_single_channel(self, sample_timeseries: TimeSeries):
        """Test getting a single channel returns GWpyTimeSeries."""
        channel = sample_timeseries[0]
        assert hasattr(channel, "value")
        assert hasattr(channel, "t0")
        assert channel.size == 1024

    def test_getitem_negative_index(self, sample_timeseries: TimeSeries):
        """Test negative indexing for channels."""
        last_channel = sample_timeseries[-1]
        assert last_channel is sample_timeseries[1]

    def test_getitem_out_of_bounds(self, sample_timeseries: TimeSeries):
        """Test that out of bounds raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            sample_timeseries[5]

    def test_getitem_negative_out_of_bounds(self, sample_timeseries: TimeSeries):
        """Test that negative out of bounds raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            sample_timeseries[-5]

    def test_getitem_slice(self, sample_timeseries: TimeSeries):
        """Test slicing channels returns new TimeSeries."""
        sliced = sample_timeseries[0:2]
        assert isinstance(sliced, TimeSeries)
        assert sliced.num_of_channels == 2

    def test_getitem_slice_single_channel(self, sample_timeseries: TimeSeries):
        """Test slicing single channel returns TimeSeries."""
        sliced = sample_timeseries[0:1]
        assert isinstance(sliced, TimeSeries)
        assert sliced.num_of_channels == 1

    def test_getitem_slice_with_step(self, sample_timeseries: TimeSeries):
        """Test slicing with step."""
        ts = TimeSeries(
            np.random.rand(4, 100),
            start_time=0,
            sampling_frequency=100,
        )
        sliced = ts[::2]
        assert sliced.num_of_channels == 2

    def test_getitem_slice_preserves_properties(self, sample_timeseries: TimeSeries):
        """Test that sliced TimeSeries preserves start_time and sampling_frequency."""
        sliced = sample_timeseries[0:1]
        assert sliced.start_time == sample_timeseries.start_time
        assert sliced.sampling_frequency == sample_timeseries.sampling_frequency
        assert sliced.duration == sample_timeseries.duration


class TestTimeSeries2DGetItem:
    """Test 2D __getitem__ (channel, sample indexing)."""

    def test_getitem_2d_scalar(self, sample_timeseries: TimeSeries):
        """Test getting scalar value with [channel, sample]."""
        value = sample_timeseries[0, 5]
        assert isinstance(value, (float, np.floating))
        np.testing.assert_equal(value, sample_timeseries[0].value[5])

    def test_getitem_2d_scalar_negative_channel(self, sample_timeseries: TimeSeries):
        """Test getting scalar with negative channel index."""
        value = sample_timeseries[-1, 5]
        np.testing.assert_equal(value, sample_timeseries[1].value[5])

    def test_getitem_2d_scalar_negative_sample(self, sample_timeseries: TimeSeries):
        """Test getting scalar with negative sample index."""
        value = sample_timeseries[0, -1]
        np.testing.assert_equal(value, sample_timeseries[0].value[-1])

    def test_getitem_2d_scalar_out_of_bounds(self, sample_timeseries: TimeSeries):
        """Test that out of bounds raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            sample_timeseries[0, 5000]

    def test_getitem_2d_row_slice(self, sample_timeseries: TimeSeries):
        """Test getting all samples of a channel with [channel, :]."""
        row = sample_timeseries[0, :]
        assert isinstance(row, np.ndarray)
        assert row.ndim == 1
        assert len(row) == 1024
        np.testing.assert_array_equal(row, sample_timeseries[0].value)

    def test_getitem_2d_row_slice_partial(self, sample_timeseries: TimeSeries):
        """Test getting partial samples with [channel, start:end]."""
        row = sample_timeseries[0, 100:200]
        assert isinstance(row, np.ndarray)
        assert len(row) == 100
        np.testing.assert_array_equal(row, sample_timeseries[0].value[100:200])

    def test_getitem_2d_column_slice(self, sample_timeseries: TimeSeries):
        """Test getting same sample across channels with [:, sample]."""
        column = sample_timeseries[:, 5]
        assert isinstance(column, np.ndarray)
        assert column.ndim == 1
        assert len(column) == 2
        np.testing.assert_array_equal(column[0], sample_timeseries[0].value[5])
        np.testing.assert_array_equal(column[1], sample_timeseries[1].value[5])

    def test_getitem_2d_block_slice(self, sample_timeseries: TimeSeries):
        """Test getting 2D block with [channel_slice, sample_slice]."""
        block = sample_timeseries[0:2, 100:200]
        assert isinstance(block, np.ndarray)
        assert block.ndim == 2
        assert block.shape == (2, 100)
        np.testing.assert_array_equal(block[0], sample_timeseries[0].value[100:200])
        np.testing.assert_array_equal(block[1], sample_timeseries[1].value[100:200])

    def test_getitem_2d_block_negative_indices(self, sample_timeseries: TimeSeries):
        """Test 2D slicing with negative indices."""
        block = sample_timeseries[-2:, -100:]
        assert block.shape == (2, 100)

    def test_getitem_2d_invalid_index_type(self, sample_timeseries: TimeSeries):
        """Test that invalid index types raise TypeError."""
        with pytest.raises(TypeError, match="must be int or slice"):
            sample_timeseries["invalid", 5]

    def test_getitem_2d_invalid_sample_type(self, sample_timeseries: TimeSeries):
        """Test that invalid sample index type raises TypeError."""
        with pytest.raises(TypeError, match="must be int or slice"):
            sample_timeseries[0, "invalid"]


class TestTimeSeries1DSetItem:
    """Test 1D __setitem__ (channel assignment)."""

    def test_setitem_single_channel(self, sample_timeseries: TimeSeries):
        """Test setting a single channel with GWpyTimeSeries."""
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries

        new_channel = GWpyTimeSeries(
            data=np.ones(1024),
            t0=sample_timeseries.start_time,
            sample_rate=sample_timeseries.sampling_frequency,
        )
        sample_timeseries[0] = new_channel
        np.testing.assert_array_equal(sample_timeseries[0].value, np.ones(1024))

    def test_setitem_channel_wrong_type(self, sample_timeseries: TimeSeries):
        """Test that assigning non-GWpyTimeSeries to channel raises TypeError."""
        with pytest.raises(TypeError, match="must be GWpyTimeSeries"):
            sample_timeseries[0] = np.ones(1024)

    def test_setitem_channel_mismatched_start_time(self, sample_timeseries: TimeSeries):
        """Test that mismatched start_time raises ValueError."""
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries

        wrong_channel = GWpyTimeSeries(
            data=np.ones(1024),
            t0=Quantity(0, unit="s"),  # Different start time
            sample_rate=sample_timeseries.sampling_frequency,
        )
        with pytest.raises(ValueError, match="Start time mismatch"):
            sample_timeseries[0] = wrong_channel

    def test_setitem_channel_slice(self, sample_timeseries: TimeSeries):
        """Test setting multiple channels with slice."""
        new_ts = TimeSeries(
            data=np.ones((2, 1024)),
            start_time=sample_timeseries.start_time,
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        sample_timeseries[0:2] = new_ts
        np.testing.assert_array_equal(sample_timeseries[0].value, np.ones(1024))
        np.testing.assert_array_equal(sample_timeseries[1].value, np.ones(1024))

    def test_setitem_channel_slice_size_mismatch(self, sample_timeseries: TimeSeries):
        """Test that channel count mismatch in slice raises ValueError."""
        wrong_ts = TimeSeries(
            data=np.ones((1, 1024)),
            start_time=sample_timeseries.start_time,
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        with pytest.raises(ValueError, match="selects 2 channels"):
            sample_timeseries[0:2] = wrong_ts


class TestTimeSeries2DSetItem:
    """Test 2D __setitem__ (channel, sample assignment)."""

    def test_setitem_2d_scalar(self, sample_timeseries: TimeSeries):
        """Test setting scalar value with [channel, sample]."""
        sample_timeseries[0, 5] = 42.0
        assert sample_timeseries[0, 5] == 42.0

    def test_setitem_2d_scalar_negative_indices(self, sample_timeseries: TimeSeries):
        """Test setting scalar with negative indices."""
        sample_timeseries[-1, -1] = 99.0
        assert sample_timeseries[-1, -1] == 99.0

    def test_setitem_2d_row_slice(self, sample_timeseries: TimeSeries):
        """Test setting all samples of a channel with [channel, :]."""
        new_data = np.arange(1024, dtype=np.float64)
        sample_timeseries[0, :] = new_data
        np.testing.assert_array_equal(sample_timeseries[0, :], new_data)

    def test_setitem_2d_row_slice_partial(self, sample_timeseries: TimeSeries):
        """Test setting partial samples with [channel, start:end]."""
        new_data = np.ones(100)
        sample_timeseries[0, 100:200] = new_data
        np.testing.assert_array_equal(sample_timeseries[0, 100:200], new_data)

    def test_setitem_2d_column_slice(self, sample_timeseries: TimeSeries):
        """Test setting same sample across channels with [:, sample]."""
        new_values = np.array([1.0, 2.0])
        sample_timeseries[:, 5] = new_values
        assert sample_timeseries[0, 5] == 1.0
        assert sample_timeseries[1, 5] == 2.0

    def test_setitem_2d_column_slice_size_mismatch(self, sample_timeseries: TimeSeries):
        """Test that column slice with wrong size raises ValueError."""
        wrong_values = np.array([1.0, 2.0, 3.0])  # 3 values for 2 channels
        with pytest.raises(ValueError, match="has 3 elements"):
            sample_timeseries[:, 5] = wrong_values

    def test_setitem_2d_block_slice(self, sample_timeseries: TimeSeries):
        """Test setting 2D block with [channel_slice, sample_slice]."""
        new_block = np.arange(200, dtype=np.float64).reshape(2, 100)
        sample_timeseries[0:2, 100:200] = new_block
        np.testing.assert_array_equal(sample_timeseries[0:2, 100:200], new_block)

    def test_setitem_2d_block_shape_mismatch_channels(self, sample_timeseries: TimeSeries):
        """Test that block with wrong channel count raises ValueError."""
        wrong_block = np.ones((3, 100))  # 3 channels for 2 selected
        with pytest.raises(ValueError, match="has 3 channels"):
            sample_timeseries[0:2, 100:200] = wrong_block

    def test_setitem_2d_block_not_2d(self, sample_timeseries: TimeSeries):
        """Test that non-2D value raises ValueError."""
        wrong_data = np.ones(100)  # 1D instead of 2D
        with pytest.raises(ValueError, match="must be 2D"):
            sample_timeseries[0:2, 100:200] = wrong_data

    def test_setitem_2d_scalar_wrong_type(self, sample_timeseries: TimeSeries):
        """Test that non-numeric value for scalar raises TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            sample_timeseries[0, 5] = "string"

    def test_setitem_2d_row_wrong_type(self, sample_timeseries: TimeSeries):
        """Test that non-ndarray value for row raises TypeError."""
        with pytest.raises(TypeError, match=r"must be np.ndarray"):
            sample_timeseries[0, :] = [1.0, 2.0]

    def test_setitem_2d_column_wrong_type(self, sample_timeseries: TimeSeries):
        """Test that non-ndarray value for column raises TypeError."""
        with pytest.raises(TypeError, match=r"must be np.ndarray"):
            sample_timeseries[:, 5] = [1.0, 2.0]

    def test_setitem_2d_out_of_bounds(self, sample_timeseries: TimeSeries):
        """Test that out of bounds assignment raises IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            sample_timeseries[0, 5000] = 1.0


class TestTimeSeriesIndexingRoundTrip:
    """Test round-trip get/set operations."""

    def test_roundtrip_scalar(self, sample_timeseries: TimeSeries):
        """Test that set and get scalar return same value."""
        test_value = 123.456
        sample_timeseries[1, 500] = test_value
        retrieved = sample_timeseries[1, 500]
        assert retrieved == test_value

    def test_roundtrip_row(self, sample_timeseries: TimeSeries):
        """Test that set and get row return same array."""
        test_data = np.random.rand(1024)
        sample_timeseries[0, :] = test_data
        retrieved = sample_timeseries[0, :]
        np.testing.assert_array_equal(retrieved, test_data)

    def test_roundtrip_column(self, sample_timeseries: TimeSeries):
        """Test that set and get column return same array."""
        test_data = np.array([10.0, 20.0])
        sample_timeseries[:, 256] = test_data
        retrieved = sample_timeseries[:, 256]
        np.testing.assert_array_equal(retrieved, test_data)

    def test_roundtrip_block(self, sample_timeseries: TimeSeries):
        """Test that set and get block return same array."""
        test_block = np.random.rand(2, 512)
        sample_timeseries[:, 256:768] = test_block
        retrieved = sample_timeseries[:, 256:768]
        np.testing.assert_array_almost_equal(retrieved, test_block)


class TestTimeSeriesCrop:
    """Test TimeSeries crop method."""

    def test_crop_with_start_end(self, sample_timeseries: TimeSeries):
        """Test cropping with start and end times."""
        original_start = sample_timeseries.start_time
        original_duration = sample_timeseries.duration
        cropped = sample_timeseries.crop(
            start_time=original_start + Quantity(0.1, unit="s"), end_time=original_start + Quantity(0.2, unit="s")
        )
        assert cropped.start_time > original_start
        assert cropped.duration < original_duration

    def test_crop_returns_self(self, sample_timeseries: TimeSeries):
        """Test that crop returns self."""
        result = sample_timeseries.crop()
        assert result is sample_timeseries


class TestTimeSeriesInject:
    """Test TimeSeries inject method."""

    def test_inject_overlapping(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting an overlapping TimeSeries."""
        original_value = sample_timeseries[0].value[512]  # Middle sample
        sample_timeseries.inject(small_timeseries)

        # Check that injection modified the data (assuming small_timeseries has ones)
        assert sample_timeseries[0].value[512] != original_value

    def test_inject_mismatched_channels_raises(self, sample_timeseries: TimeSeries):
        """Test that mismatched channel count raises ValueError."""
        wrong_channels = TimeSeries(
            np.ones((3, 100)),  # 3 channels vs 2
            start_time=sample_timeseries.start_time,
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        with pytest.raises(ValueError, match="Number of channels"):
            sample_timeseries.inject(wrong_channels)

    def test_inject_extending_beyond(self, sample_timeseries: TimeSeries):
        """Test injecting a TimeSeries that extends beyond the end."""
        extending_ts = TimeSeries(
            np.ones((2, 100)),
            start_time=sample_timeseries.end_time - Quantity(50 / 4096, unit="s"),  # Overlaps end
            sampling_frequency=sample_timeseries.sampling_frequency,
        )
        remaining = sample_timeseries.inject(extending_ts)
        assert remaining is not None
        assert isinstance(remaining, TimeSeries)


class TestTimeSeriesInjectFromList:
    """Test TimeSeries inject_from_list method."""

    def test_inject_from_list(self, sample_timeseries: TimeSeries, small_timeseries: TimeSeries):
        """Test injecting from a TimeSeriesList."""
        ts_list = TimeSeriesList([small_timeseries])
        remaining_list = sample_timeseries.inject_from_list(ts_list)
        assert isinstance(remaining_list, TimeSeriesList)


class TestTimeSeriesSerialization:
    """Test TimeSeries serialization."""

    def test_to_json_dict(self, sample_timeseries: TimeSeries):
        """Test to_json_dict produces correct structure."""
        data = sample_timeseries.to_json_dict()
        assert data["__type__"] == "TimeSeries"
        assert "data" in data
        assert "start_time" in data
        assert "start_time_unit" in data
        assert "sampling_frequency" in data
        assert "sampling_frequency_unit" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == sample_timeseries.num_of_channels

    def test_from_json_dict_round_trip(self, sample_timeseries: TimeSeries):
        """Test round-trip serialization."""
        json_data = sample_timeseries.to_json_dict()
        reconstructed = TimeSeries.from_json_dict(json_data)
        assert reconstructed.num_of_channels == sample_timeseries.num_of_channels
        assert reconstructed.start_time == sample_timeseries.start_time
        assert reconstructed.sampling_frequency == sample_timeseries.sampling_frequency
        np.testing.assert_array_equal(reconstructed[0].value, sample_timeseries[0].value)
