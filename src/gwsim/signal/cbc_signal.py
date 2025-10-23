from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries


from ..detectors import Detectors
from ..waveform import CBCWaveform
from .base import BaseSignal


class CBCSignal(Generator):

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
        self.end_time = self.start_time + self.duration
        self.approximant = approximant
        self.flow = flow
        self.waveform_arguments = dict(approximant=self.approximant,
                                       flow=self.flow,
                                       sampling_frequency=self.sampling_frequency)
        if not Path(population_file).is_file():
            raise FileNotFoundError(f"Population file {population_file} not found.")
        self.population_df = self._read_population_file(self.population_file)

    def _read_population_file(self, filename: str) -> pd.DataFrame:
        """
        Read the PyCBC population file into a pandas DataFrame.

        Args:
            filename (str): Path to the PyCBC population file.

        Returns:
            pandas.DataFrame: DataFrame with event parameters as columns and events as rows.

        Raises:
            FileNotFoundError: If the population file does not exist.
            ValueError: If the file is missing required columns.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Population file {filename} not found.")

        population_df = pd.read_csv(filename)  # TODO: Adjust based on input from gwsim#11

        # Check for missing parameters
        required_columns = ['mass_1', 'mass_2', 'geocent_time', 'luminosity_distance',
                            'spin_1x', 'spin_1y', 'spin_1z', 'spin_2x', 'spin_2y', 'spin_2z',
                            'right_ascension', 'declination', 'polarization_angle', 'iota', 'phase']
        if not all(col in population_df.columns for col in required_columns):
            raise ValueError(f"Population file must contain columns: {required_columns}")

        return population_df  # TODO: the output data frame must have as column the parameters name, and as rows the event names. Each row corresponds to an event

    def select_events_in_frame(events_df: pandas.DataFrame, start_time: float, end_time: float) -> pandas.DataFrame:
        """
        Function to select the events that overlap with the frame. This takes events with a piece of signal or full duration in the frame.

        Args:
            events_df (pandas.DataFrame): Data frame of all the event
            start_time (float): Start time of the frame
            end_time (float): End time of the frame

        Returns:
            pandas.DataFrame: Data frame containing the events in the frame, sorted by geocentric time
        """
        # An event overlaps the frame if it starts before the frame ends and ends after the frame starts
        time_mask = (events_df['geocent_time'] > start_time) & (
            events_df['geocent_time'] - events_df['duration'] < end_time)

        return events_df[time_mask].sort_values(by='geocent_time', ascending=True)

    def _adjust_parameters_to_pycbc_convention(self, parameters: dict) -> dict:
        """
        Auxiliary function to adjust the parameters of the event according to the PyCBC convention.

        Args:
            parameters (dict): Dictionary with the parameters of the event

        Returns:
            dict: Dictionary with the parameters of the event adjusted according to the PyCBC convention
        """
        # TODO

        return parameters_adjusted

    def get_polarization_at_time(self, parameters: dict, waveform_arguments: dict) -> (pycbc.TimeSeries, pycbc.TimeSeries):
        """
        Function to make the polarization of the events at the correct time (# TO CHECK: Earth rotation)

        Args:
            parameters (dict): Dictionary with the parameters of the event
            waveform_arguments (dict): Dictionary with the waveform arguments

        Returns:
            (TimeSeries, TimeSeries): PyCBC TimeSeries of the plus and cross polarization, hp and hc
        """

        # TODO: Adjust parmaters for NS events, adding tidal deformability

        hp, hc = get_td_waveform(approximant=waveform_arguments['approximant'],
                                 mass1=parameters['mass_1']*(1+parameters['redshift']),
                                 mass2=parameters['mass_2']*(1+parameters['redshift']),
                                 spin1x=parameters['spin_1x'],
                                 spin1y=parameters['spin_1y'],
                                 spin1z=parameters['spin_1z'],
                                 spin2x=parameters['spin_2x'],
                                 spin2y=parameters['spin_2y'],
                                 spin2z=parameters['spin_2z'],
                                 distance=parameters['luminosity_distance'],
                                 coa_phase=parameters['phase'],
                                 inclination=parameters['iota'],
                                 f_lower=waveform_arguments['flow'],
                                 delta_t=1./waveform_arguments['sampling_frequency'])

        hp.start_time += parameters['geocent_time']
        hc.start_time += parameters['geocent_time']

        return hp, hc

    def inject_signal_in_frame(hp: pycbc.TimeSeries,
                               hc: pycbc.TimeSeries,
                               parameters: dict,
                               detector: Detector,
                               frame_data: np.ndarray,
                               sampling_frequency: float,
                               frame_start_time: float,
                               frame_end_time: float) -> np.ndarray:
        """
        Inject a signal into the frame data (# TO CHECK: the function does NOT include effects of Earth rotation).

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
        # TODO: Incorporate Earth rotation effects if required

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
            frame_data[idx_frame_start: idx_frame_end] += signal[idx_signal_start: idx_signal_end]
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
            hp, hc = get_polarization_at_time(parameters, self.waveform_arguments)

            # Inject the event
            for i, ifo in enumerate(self.detectors):
                frame_data[i, :] = inject_signal_in_frame(hp, hc,
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
