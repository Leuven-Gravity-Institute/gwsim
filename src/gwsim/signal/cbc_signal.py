from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import bilby
from scipy.interpolate import interp1d
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries

from ..utils import read_pycbc_population_file
from ..detectors import Detector
from .base import BaseSignal


class CBCSignal(BaseSignal):

    """
    Generate time series frames with gravitational wave signals from CBC using a population file
    """

    def __init__(
        self,
        detector_names: list[str],
        population_file: str,
        approximant: str,
        flow: float,
        sampling_frequency: float,
        duration: float,
        earth_rotation: bool = False,
        earth_rotation_timestep: float = 100,
        time_dependent_timedelay: bool = False,
        start_time: float = 0,
        max_samples: int | None = None,
    ):
        """
        Initialize the CBCSignal generator for gravitational wave signals.

        Args:
            detector_names (list[str]): List of detector names.
            population_file (str): Path to the population file containing CBC event parameters.
            approximant (str): Waveform approximant.
            flow (float): Low-frequency cutoff in Hz.
            sampling_frequency (float): Sampling frequency in Hz.
            duration (float): Duration of the frame in seconds.
            earth_rotation (bool, optional): If True, account for Earth's rotation effects in antenna patterns and time delays. Defaults to False.
            earth_rotation_timestep (float, optional): Time step in seconds for approximating time-varying antenna patterns and delays when earth_rotation is True. Defaults to 100.
            time_dependent_timedelay (bool, optional): If True, include time-dependent time delays due to Earth's rotation (only applicable if earth_rotation is True). Defaults to False.
            start_time (float, optional): Start time of the frame in seconds. Defaults to 0.0.
            max_samples (int, optional): Maximum number of samples to generate. Defaults to None.
        """

        super().__init__(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
            max_samples=max_samples,
        )
        self.detector_names = detector_names
        self.N_det = len(detector_names)
        if self.N_det == 0:
            raise ValueError("detector_names must contain at least one detector.")
        self.detectors = [Detector(det_name) for det_name in detector_names]
        self.waveform_arguments = dict(
            approximant=approximant,
            flow=flow,
            sampling_frequency=sampling_frequency,
            earth_rotation=earth_rotation,
            earth_rotation_timestep=earth_rotation_timestep,
            time_dependent_timedelay=time_dependent_timedelay
        )
        if not Path(population_file).is_file():
            raise FileNotFoundError(f"Population file {population_file} not found.")
        self.population_df = self._read_population_file(population_file)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def _read_population_file(self, filename: str) -> pd.DataFrame:
        """
        Read the PyCBC population file into a pandas DataFrame.

        Args:
            filename (str): Path to the PyCBC population file.

        Returns:
            pandas.DataFrame: DataFrame with event parameters as columns and events as rows.
        """

        # TODO: the output data frame must have as column the parameters name, and as rows the event names. Each row corresponds to an event
        return read_pycbc_population_file(filename)

    def select_events_in_frame(self, events_df: pandas.DataFrame, start_time: float, end_time: float) -> pandas.DataFrame:
        """
        Function to select the events that overlap with the frame. This takes events with a piece of signal or full duration in the frame.

        Args:
            events_df (pandas.DataFrame): Data frame of all the event
            start_time (float): Start time of the frame
            end_time (float): End time of the frame

        Returns:
            pandas.DataFrame: Data frame containing the events in the frame, sorted by geocentric time

        Raises:
            ValueError: If the signal durations reported in the population file do not match the expected ones computed with the given flow
        """
        # Check that signals duration is close to the expected one
        expected_durations = np.array([
            bilby.gw.utils.calculate_time_to_merger(
                self.waveform_arguments['flow'],
                row['mass_1'] * (1 + row.get('redshift', 0.0)),
                row['mass_2'] * (1 + row.get('redshift', 0.0)),
                safety=1.2
            ) for _, row in events_df.iterrows()
        ])
        if not np.allclose(events_df['duration'].values, expected_durations, rtol=1e-3):
            raise ValueError(
                f"Reported signal durations in population file do not match the expected durations computed with the given flow = {self.waveform_arguments['flow']} Hz.")

        # An event overlaps the frame if it starts before the frame ends and ends after the frame starts
        time_mask = (events_df['geocent_time'] > start_time) & (
            events_df['geocent_time'] - events_df['duration'] < end_time)

        return events_df[time_mask].sort_values(by='geocent_time', ascending=True)

    def _adjust_parameters_to_pycbc_convention(self, parameters: dict) -> dict:
        """
        Auxiliary function to adjust the parameters of the event according to the PyCBC convention. (#TODO)

        Args:
            parameters (dict): Dictionary with the parameters of the event

        Returns:
            dict: Dictionary with the parameters of the event adjusted according to the PyCBC convention
        """
        parameters_adjusted = parameters  # TODO

        return parameters_adjusted

    def get_polarization_at_time(self, parameters: dict, waveform_arguments: dict) -> (pycbc.TimeSeries, pycbc.TimeSeries):
        """
        Function to make the polarization of the events at the correct time

        Args:
            parameters (dict): Dictionary with the parameters of the event
            waveform_arguments (dict): Dictionary with the waveform arguments

        Returns:
            (TimeSeries, TimeSeries): PyCBC TimeSeries of the plus and cross polarization, hp and hc
        """

        kwargs = {
            'approximant': waveform_arguments['approximant'],
            'mass1': parameters['mass_1'] * (1 + parameters['redshift']),
            'mass2': parameters['mass_2'] * (1 + parameters['redshift']),
            'spin1x': parameters['spin_1x'],
            'spin1y': parameters['spin_1y'],
            'spin1z': parameters['spin_1z'],
            'spin2x': parameters['spin_2x'],
            'spin2y': parameters['spin_2y'],
            'spin2z': parameters['spin_2z'],
            'distance': parameters['luminosity_distance'],
            'coa_phase': parameters['phase'],
            'inclination': parameters['iota'],
            'polarization_angle': parameters['polarization_angle'],
            'f_lower': waveform_arguments['flow'],
            'delta_t': 1. / waveform_arguments['sampling_frequency']
        }

        if 'lambda_2' in parameters:
            kwargs['lambda2'] = parameters['lambda_2']
            kwargs['lambda1'] = parameters.get('lambda_1', 0)

        hp, hc = get_td_waveform(**kwargs)

        hp.start_time += parameters['geocent_time']
        hc.start_time += parameters['geocent_time']

        return hp, hc

    def inject_signal_in_frame(self,
                               hp: pycbc.TimeSeries,
                               hc: pycbc.TimeSeries,
                               parameters: dict,
                               detector: Detector,
                               frame_data: np.ndarray,
                               sampling_frequency: float,
                               frame_start_time: float,
                               frame_end_time: float) -> np.ndarray:
        """
        Inject a signal into the frame data.

        Parameters:
            hp (TimeSeries): TimeSeries of the plus polarization
            hc (TimeSeries): TimeSeries of the cross polarization
            parameters (dict): Dictionary with the parameters of the event
            detector(Detector): Detector object in which the injection should happen
            frame_data (np.ndarray): Frame data to which the signal should be added
            sampling_frequency (float): Sampling frequency of the data
            frame_start_time (float): Start time of the frame
            frame_end_time (float): End time of the frame

        Returns:
            np.ndarray: The initial frame data with injected signal

        Notes:
            Assumes consistent sampling frequency between signal and frame data.
        """

        if self.waveform_arguments['earth_rotation']:

            t_gps = np.arange(parameters['geocent_time'] - hp.duration,
                              parameters['geocent_time'] +
                              self.waveform_arguments['earth_rotation_timestep'],
                              self.waveform_arguments['earth_rotation_timestep'])

            repeat_count = int(self.waveform_arguments['earth_rotation_timestep'] *
                               self.waveform_arguments['sampling_frequency'])

            Fp, Fc = detector.antenna_pattern(
                right_ascension=parameters['right_ascension'],
                declination=parameters['declination'],
                polarization=parameters['polarization_angle'],
                t_gps=t_gps
            )
            Fp = np.repeat(Fp, repeat_count)[:len(hp)]
            Fc = np.repeat(Fc, repeat_count)[:len(hp)]

            if self.waveform_arguments['time_dependent_timedelay']:

                tdelayArr = detector.time_delay_from_earth_center(
                    parameters['right_ascension'], parameters['declination'], t_gps)
                tdelayArr = np.repeat(tdelayArr, repeat_count)[:len(hp)]

                # Now evaluate h at u = t + tau(t)
                u = hp.sample_times.data + tdelayArr

                hp.data = interp1d(hp.sample_times.data, hp.data,
                                   kind='cubic', fill_value='extrapolate')(u)
                hc.data = interp1d(hc.sample_times.data, hc.data,
                                   kind='cubic', fill_value='extrapolate')(u)
        else:

            # Compute antenna patterns and signal
            Fp, Fc = detector.antenna_pattern(
                right_ascension=parameters['right_ascension'],
                declination=parameters['declination'],
                polarization=parameters['polarization_angle'],
                t_gps=parameters['geocent_time']
            )

        signal = Fp*hp + Fc*hc

        # Define signal start and end time
        signal_start_time = signal.sample_times[0]
        signal_end_time = signal.sample_times[-1]

        # Check if signal is outside the frame
        if signal_end_time < frame_start_time or signal_start_time > frame_end_time:
            raise ValueError(
                "Signal time range does not overlap with frame time range. No injection performed.")

        # Select frame and signal indices
        idx_frame_start = max(0, int((signal_start_time - frame_start_time) * sampling_frequency))
        idx_frame_end = min(len(frame_data),
                            int((signal_end_time - frame_start_time) * sampling_frequency))
        idx_signal_start = max(0, int((frame_start_time - signal_start_time) * sampling_frequency))
        idx_signal_end = min(len(signal),
                             int((frame_end_time - signal_start_time) * sampling_frequency))

        # Check for valid overlap
        if idx_frame_start >= len(frame_data) or idx_signal_start >= len(signal):
            raise ValueError(
                "Computed indices indicate no valid overlap between signal and frame data.")
        if idx_frame_end <= idx_frame_start or idx_signal_end <= idx_signal_start:
            raise ValueError(
                "Computed slice lengths are zero or negative, indicating no valid overlap between signal and frame data.")

        # Adjust for length mismatch
        frame_slice_length = idx_frame_end - idx_frame_start
        signal_slice_length = idx_signal_end - idx_signal_start
        if signal_slice_length > frame_slice_length:
            idx_signal_end = idx_signal_start + frame_slice_length
        elif signal_slice_length < frame_slice_length:
            idx_frame_end = idx_frame_start + signal_slice_length

        try:
            frame_data[idx_frame_start: idx_frame_end] += signal.data[idx_signal_start: idx_signal_end]
        except ValueError as e:
            raise ValueError(
                f"Error injecting signal: {str(e)}. Check array dimensions and sampling consistency.")

        return frame_data

    def next(self) -> np.ndarray:
        """
        Generate a frame of data for each detector containig the events of the population file.

        Returns:
            np.ndarray: time series for each detector
        """
        # Select the events that are in the data frame
        injection_df = self.select_events_in_frame(self.population_df,
                                                   self.start_time,
                                                   self.end_time)
        # Initialize empty frame
        frame_data = np.zeros((self.N_det, int(self.duration * self.sampling_frequency)))

        for idx, row in injection_df.iterrows():

            # Extract the parameters
            parameters = self._adjust_parameters_to_pycbc_convention(row.to_dict())

            # Compute the polarizations
            hp, hc = self.get_polarization_at_time(parameters, self.waveform_arguments)

            # Inject the event
            for i, ifo in enumerate(self.detectors):
                frame_data[i, :] = self.inject_signal_in_frame(hp, hc,
                                                               parameters,
                                                               ifo,
                                                               frame_data[i, :],
                                                               self.sampling_frequency,
                                                               self.start_time,
                                                               self.end_time)
        return frame_data

    def update_state(self) -> None:
        """Update the internal state for the next batch."""
        self.sample_counter += 1
        self.start_time += self.duration

    def save_batch(self, batch: np.ndarray, file_name: str | Path, overwrite: bool = False, **kwargs) -> None:
        """
        Args:
            batch (np.ndarray): One batch of data with shape (D, N), where D is the number of detectors and N the number of data points.
            file_name (str | Path): File name.
            overwrite (bool, optional): If True, overwrite existing file. Defaults to False.
            **kwargs: Optional keyword arguments, e.g., 'dataset_name' for HDF5 or 'channel' for GWF.

        Raises:
            ValueError: If the file suffix is not supported (supported: '.h5', '.hdf5', '.gwf').
            ValueError: If file_name does not start with 'E-'.
        """
        file_name = Path(file_name)

        if batch.shape[0] != self.N_det:
            raise ValueError(
                f"Batch first dimension ({batch.shape[0]}) must match number of detectors ({self.N_det}).")

        for i, det_name in enumerate(self.detector_names):
            # Adjust filename per detector
            det_file_name = self._adjust_filename(file_name=file_name, insert=det_name)

            if file_name.suffix in [".h5", ".hdf5"]:
                # Prepare dataset name
                dataset_name = kwargs.get("dataset_name", "strain")
                det_dataset_name = f"{det_name}:{dataset_name}"
                self._save_batch_hdf5(
                    batch=batch[i, :],
                    file_name=det_file_name,
                    overwrite=overwrite,
                    dataset_name=det_dataset_name
                )
            elif file_name.suffix == ".gwf":
                # Prepare channel
                channel = kwargs.get("channel", "strain")
                det_channel = f"{det_name}:{channel}"
                self._save_batch_gwf(
                    batch=batch[i, :],
                    file_name=det_file_name,
                    overwrite=overwrite,
                    channel=det_channel
                )
            else:
                raise ValueError(
                    f"Suffix of file_name = {file_name} is not supported. Use ['.h5', '.hdf5'] for HDF5 files,"
                    "and '.gwf' for frame files."
                )

    def _adjust_filename(self, file_name: Path, insert: str) -> Path:
        """If the file name starts with 'E-', insert `insert` right after it."""
        stem = file_name.stem
        if stem.startswith("E-"):
            stem = f"E-{insert}_{stem[2:]}"
        else:
            raise ValueError(f"Invalid filename '{file_name.name}'. Must start with 'E-'.")
        return file_name.with_stem(stem)
