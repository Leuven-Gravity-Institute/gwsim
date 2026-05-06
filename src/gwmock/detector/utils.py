"""Utilities for loading gravitational wave interferometer configurations."""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np

# The default base path for detector configuration files
DEFAULT_DETECTOR_BASE_PATH = Path(__file__).parent / "detectors"


def _bilby_to_pycbc_detector_parameters(bilby_params: dict) -> dict:
    """
    Convert Bilby detector parameters to PyCBC-compatible parameters.

    This function handles the conversion of units and conventions between Bilby and PyCBC,
    including latitude/longitude to radians, length from km to meters, and azimuth adjustments
    due to different reference conventions (Bilby: from East counterclockwise; PyCBC/LAL: from North clockwise).

    Args:
        bilby_params (dict): Dictionary of Bilby parameters (e.g., 'latitude', 'xarm_azimuth', etc.).

    Returns:
        dict: Dictionary of converted PyCBC parameters.
    """
    pycbc_params = {
        "name": bilby_params["name"],
        "latitude": np.deg2rad(bilby_params["latitude"]),
        "longitude": np.deg2rad(bilby_params["longitude"]),
        "height": bilby_params["elevation"],
        "xangle": (np.pi / 2 - np.deg2rad(bilby_params["xarm_azimuth"])) % (2 * np.pi),
        "yangle": (np.pi / 2 - np.deg2rad(bilby_params["yarm_azimuth"])) % (2 * np.pi),
        "xaltitude": bilby_params["xarm_tilt"],
        "yaltitude": bilby_params["yarm_tilt"],
        "xlength": bilby_params["length"] * 1000,
        "ylength": bilby_params["length"] * 1000,
    }

    return pycbc_params


def resolve_interferometer_config_path(config_file: str | Path) -> Path:
    """Resolve one interferometer config path against the built-in detector directory."""
    config_path = Path(config_file)
    if config_path.is_file():
        return config_path

    default_path = DEFAULT_DETECTOR_BASE_PATH / config_path
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(f"Config file {config_file} not found.")


def read_interferometer_config(config_file: str | Path, encoding: str = "utf-8") -> dict:
    """Read one ``.interferometer`` file into its Bilby-style parameter mapping."""
    resolved_config_file = resolve_interferometer_config_path(config_file)

    bilby_params = {}
    with resolved_config_file.open(encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            if key == "power_spectral_density":
                continue
            value = literal_eval("=".join(split_line[1:]))
            bilby_params[key] = value

    return bilby_params


def interferometer_config_to_custom_detector(config_file: str | Path, encoding: str = "utf-8"):
    """Convert one ``.interferometer`` file into a public ``gwmock_signal`` detector."""
    from gwmock_signal import CustomDetector  # noqa: PLC0415

    bilby_params = read_interferometer_config(config_file=config_file, encoding=encoding)
    return CustomDetector(
        name=str(bilby_params["name"]),
        latitude_rad=float(np.deg2rad(float(bilby_params["latitude"]))),
        longitude_rad=float(np.deg2rad(float(bilby_params["longitude"]))),
        elevation_m=float(bilby_params["elevation"]),
        xarm_azimuth_rad=float(np.deg2rad(float(bilby_params["xarm_azimuth"]))),
        yarm_azimuth_rad=float(np.deg2rad(float(bilby_params["yarm_azimuth"]))),
        xarm_tilt_rad=float(bilby_params.get("xarm_tilt", 0.0)),
        yarm_tilt_rad=float(bilby_params.get("yarm_tilt", 0.0)),
    )


def load_interferometer_config(config_file: str | Path, encoding: str = "utf-8") -> str:
    """
    Load a .interferometer config file and add its detector using pycbc.detector.add_detector_on_earth.

    Args:
        config_file: The path to the config file.
        encoding: The file encoding to use when reading the config file. Default is 'utf-8'.

    Returns:
        str: Added detector name (e.g., "E1").
    """
    bilby_params = read_interferometer_config(config_file=config_file, encoding=encoding)

    params = _bilby_to_pycbc_detector_parameters(bilby_params)
    det_name = params["name"]

    try:
        from pycbc.detector import add_detector_on_earth  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycbc is required to register detector configurations from '.interferometer' files."
        ) from exc

    add_detector_on_earth(
        name=det_name,
        latitude=params["latitude"],
        longitude=params["longitude"],
        height=params["height"],
        xangle=params["xangle"],
        yangle=params["yangle"],
        xaltitude=params["xaltitude"],
        yaltitude=params["yaltitude"],
        xlength=params["xlength"],
        ylength=params["ylength"],
    )

    return det_name
