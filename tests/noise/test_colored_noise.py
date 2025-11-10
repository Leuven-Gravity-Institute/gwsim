from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import kstest
from scipy.signal import get_window, welch
from scipy.linalg import cholesky
from scipy.sparse import block_diag, coo_matrix
from etmdc.noise_curve import load_ET_PSD
from pathlib import Path
import tempfile

import pytest

from gwsim.utils.random import get_state
from gwsim.noise.colored_noise import ColoredNoise


# Fixtures


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path


@pytest.fixture
def mock_psd(tmp_path):
    """Create PSD array and save it as .npy files."""
    # freqs = np.linspace(0, 6000, 3000)
    # psd = np.column_stack([freqs, np.ones_like(freqs)])
    psd = load_ET_PSD()
    psd_file = tmp_path / "psd.npy"
    np.save(psd_file, psd)
    return str(psd_file)

# Unit-level tests


def test_gaussianity(mock_psd):
    """ Test that whitened noise series in gaussian """
    # Define parameters
    detector_names = ['E1', 'E2']
    N_det = len(detector_names)
    fs = 4096
    duration = 512
    flow = 2
    start_time = 0.0

    # Create colored noise instance
    simulator = ColoredNoise(
        detector_names=detector_names,
        psd=mock_psd,
        sampling_frequency=fs,
        duration=duration,
        flow=flow,
        start_time=start_time,
        seed=123
    )
    noise_ts = simulator.next()

    # Time and frequency properties
    dt = 1 / fs
    df = 1 / duration
    N = int(duration * fs)
    frequency = np.arange(0.0, int(duration * fs) / 2.0 + 1) * df
    kmin = int(flow / df)
    kmax = int((fs / 2) / df)
    N_freq = len(frequency[kmin:kmax])

    # Compute the target PSD
    psd = np.load(mock_psd)
    target_psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                          fill_value="extrapolate")(frequency[kmin: kmax])
    d1 = np.zeros_like(target_psd)

    # Compute the target spectral matrix
    target_spectral_matrix = np.empty((N_freq, N_det, N_det))
    for n in range(N_freq):
        submatrix = np.array(
            [
                [target_psd[n] if row == col else d1[n] for row in range(N_det)]
                for col in range(N_det)
            ]
        )
        target_spectral_matrix[n, :, :] = cholesky(submatrix, lower=True)
    target_spectral_matrix = block_diag(target_spectral_matrix, format="coo")

    # Transform back time series to frequency domain
    noise_fs = np.fft.rfft(noise_ts, axis=-1) * dt

    # Whiten the noise frequency series with the target spectral matrix
    whitened_noise_fs = np.zeros_like(noise_fs, dtype=np.complex128)
    size = N_det * N_det
    for i in range(N_freq):
        L = np.array(target_spectral_matrix.data[i * size: (i + 1) * size]).reshape(N_det, N_det)
        whitened_noise_fs[:, kmin +
                          i] = np.linalg.inv(L) @ noise_fs[:, kmin + i] / np.sqrt(0.5 / df)

    whitened_noise_fs = whitened_noise_fs[:, kmin: kmax]

    # Test the gaussianity of the whitened data
    joint_fs = np.concatenate((whitened_noise_fs.real, whitened_noise_fs.imag), axis=None)
    stats = kstest(joint_fs, cdf="norm", args=(0, np.sqrt(0.5)))

    assert stats.pvalue > 0.05, f'Gaussianity test NOT passed, p-value is: {stats.pvalue:.3f}'


def test_gaussianity_connecting_region(mock_psd):
    """ Test that the connecting regions of two adjacent frames in gaussian """
    # Define parameters
    detector_names = ['E1', 'E2']
    N_det = len(detector_names)
    fs = 4096
    duration = 512
    flow = 2
    start_time = 0.0

    # Create colored noise instance
    simulator = ColoredNoise(
        detector_names=detector_names,
        psd=mock_psd,
        sampling_frequency=fs,
        duration=duration,
        flow=flow,
        start_time=start_time,
        seed=123
    )
    noise_ts_1 = simulator.next()
    simulator.update_state()
    noise_ts_2 = simulator.next()

    T_connecting_region = 10
    N_connecting_region = int(10 * fs)
    connecting_region_ts = np.concatenate(
        (noise_ts_1[:, -N_connecting_region:], noise_ts_2[:, :N_connecting_region]), axis=1)

    # Time and frequency properties
    dt = 1 / fs
    df = 1 / (2 * T_connecting_region)
    N = int((2 * T_connecting_region) * fs)
    frequency = np.arange(0.0, int((2 * T_connecting_region) * fs) / 2.0 + 1) * df
    kmin = int(flow / df)
    kmax = int((fs / 2) / df)
    N_freq = len(frequency[kmin:kmax])

    # Compute the target PSD
    psd = np.load(mock_psd)
    target_psd = interp1d(psd[:, 0], psd[:, 1], bounds_error=False,
                          fill_value="extrapolate")(frequency[kmin: kmax])
    d1 = np.zeros_like(target_psd)

    # Compute the target spectral matrix
    target_spectral_matrix = np.empty((N_freq, N_det, N_det))
    for n in range(N_freq):
        submatrix = np.array(
            [
                [target_psd[n] if row == col else d1[n] for row in range(N_det)]
                for col in range(N_det)
            ]
        )
        target_spectral_matrix[n, :, :] = cholesky(submatrix, lower=True)
    target_spectral_matrix = block_diag(target_spectral_matrix, format="coo")

    # Transform back time series to frequency domain
    connecting_region_fs = np.fft.rfft(connecting_region_ts, axis=-1) * dt

    # Whiten the noise frequency series with the target spectral matrix
    whitened_connecting_region_fs = np.zeros_like(connecting_region_fs, dtype=np.complex128)
    size = N_det * N_det
    for i in range(N_freq):
        L = np.array(target_spectral_matrix.data[i * size: (i + 1) * size]).reshape(N_det, N_det)
        whitened_connecting_region_fs[:, kmin +
                                      i] = np.linalg.inv(L) @ connecting_region_fs[:, kmin + i] / np.sqrt(0.5 / df)

    whitened_connecting_region_fs = whitened_connecting_region_fs[:, kmin: kmax]

    # Test the gaussianity of the whitened data
    joint_fs = np.concatenate((whitened_connecting_region_fs.real,
                              whitened_connecting_region_fs.imag), axis=None)
    stats = kstest(joint_fs, cdf="norm", args=(0, np.sqrt(0.5)))

    assert stats.pvalue > 0.05, f'Gaussianity test NOT passed, p-value is: {stats.pvalue:.3f}'


def test_psd_matching(mock_psd):
    """Test that the PSD of the generated noise matches the target PSD."""
    # Define parameters (same as other tests for consistency)
    detector_names = ['E1', 'E2']
    fs = 4096
    duration = 512
    flow = 2
    start_time = 0.0

    # Create colored noise instance
    simulator = ColoredNoise(
        detector_names=detector_names,
        psd=mock_psd,
        sampling_frequency=fs,
        duration=duration,
        flow=flow,
        start_time=start_time,
        seed=123  # Fixed seed for reproducibility
    )
    noise_ts = simulator.next()

    # Load target PSD
    psd = np.load(mock_psd)
    target_interp = interp1d(psd[:, 0], psd[:, 1], bounds_error=False, fill_value="extrapolate")

    # Check each detector independently (since uncorrelated)
    for i, det in enumerate(detector_names):
        # Estimate PSD with Welch (density scaling to match units; nperseg for good low-freq resolution)
        f, Pxx = welch(
            noise_ts[i],
            fs=fs,
            nperseg=int(fs * 4)
        )

        # Interpolate target to estimated frequencies
        target_psd = target_interp(f)

        # Mask to relevant frequencies (≥ flow, ≤ Nyquist)
        mask = (f >= flow + 1) & (f <= (fs / 2) - 100)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(f[mask], Pxx[mask])
        ax.loglog(f[mask], target_psd[mask])
        plt.show()

        # Assert close (stochastic, so use rtol; atol for tiny PSD values like in ET)
        assert np.allclose(
            Pxx[mask], target_psd[mask], rtol=0.5, atol=1e-50
        ), f"PSD mismatch for {det}: max rel diff {np.max(np.abs(Pxx[mask] - target_psd[mask]) / target_psd[mask])}"
