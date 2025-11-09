"""Time series JSON decoder."""

from __future__ import annotations

from typing import Any

import numpy as np

from gwsim.data.time_series import TimeSeries


def time_series_decoder(obj: dict[str, Any]) -> Any:
    """Custom JSON decoder for TimeSeries objects.

    Args:
        obj: Dictionary representation of the object.

    Returns:
        Decoded TimeSeries object or the original object.
    """
    if obj.get("__timeseries__"):
        data_array = np.array(obj["data"])
        start_time = obj["start_time"]
        sampling_frequency = obj["sampling_frequency"]

        return TimeSeries(data=data_array, start_time=start_time, sampling_frequency=sampling_frequency)
    return obj
