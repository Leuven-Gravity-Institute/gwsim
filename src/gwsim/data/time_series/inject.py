"""Module to handle injection of one TimeSeries into another, with support for time offsets."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from astropy.units import second  # pylint: disable=no-name-in-module
from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d

logger = logging.getLogger("gwsim")


def inject(timeseries: TimeSeries, other: TimeSeries, interpolate_if_offset: bool = True) -> TimeSeries:
    """Inject one TimeSeries into another, handling time offsets.

    Args:
        timeseries: The target TimeSeries to inject into.
        other: The TimeSeries to be injected.
        interpolate_if_offset: Whether to interpolate if there is a non-integer sample offset.

    Returns:
        TimeSeries: The resulting TimeSeries after injection.
    """
    # Check whether timeseries is compatible with other
    timeseries.is_compatible(other)

    # crop to fit
    if (timeseries.xunit == second) and (other.xspan[0] < timeseries.xspan[0]):
        other = cast(TimeSeries, other.crop(start=timeseries.xspan[0]))
    if (timeseries.xunit == second) and (other.xspan[1] > timeseries.xspan[1]):
        other = cast(TimeSeries, other.crop(end=timeseries.xspan[1]))

    target_times = timeseries.times.value
    other_times = other.times.value

    # determine the slice of target times that overlaps the other file
    start_idx = int(np.searchsorted(target_times, other_times[0], side="left"))
    end_idx = int(np.searchsorted(target_times, other_times[-1], side="right") - 1)

    sample_spacing = float(timeseries.dt.value)

    offset = (other_times[0] - target_times[0]) / sample_spacing
    if not np.isclose(offset, round(offset)):
        if not interpolate_if_offset:
            # If not interpolating, and offset, but since cropped, perhaps add directly if times match
            # But since offset not integer, times don't match, so can't add. Return timeseries unchanged.
            return timeseries

        if start_idx >= len(target_times) or end_idx < 0 or start_idx > end_idx:
            # No overlap, return timeseries unchanged
            return timeseries

        logger.debug("Injecting with interpolation due to non-integer offset of %s samples", offset)

        interp_func = interp1d(other_times, other.value, kind="cubic", axis=0, bounds_error=False, fill_value=0.0)
        resampled = interp_func(target_times[start_idx : end_idx + 1])

        # Create injected timeseries
        injected = timeseries.copy()
        injected[start_idx : end_idx + 1] += resampled
        return injected

    # Aligned, add directly
    injected = timeseries.copy()
    injected.value[start_idx : end_idx + 1] += other.value
    return injected
