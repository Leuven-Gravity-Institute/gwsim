from __future__ import annotations
import configparser
from pathlib import Path

from pycbc.detector import Detector as BaseDetector
from pycbc.detector import get_available_detectors, add_detector_on_earth

# Store the original for reference
_original_get_available_detectors = get_available_detectors

# Define the path to the available .interferometer config files
detectors_dir = str(Path(__file__).parent / "detectors")
if not Path(detectors_dir).exists() or not Path(detectors_dir).is_dir():
    print(
        f"Warning: Detector config directory {path.absolute()} does not exist or is not a directory.")


def load_interferometer_config(config_name: str, config_path: str = detectors_dir) -> list[str]:
    """
    Load a .interferometer config file and add its individual detectors using pycbc.detector.add_detector_on_earth.

    Args:
        config_name (str): The base name of the config file (e.g., "ET_Triangle_Sardinia").
        config_path (str, optional): Directory where .interferometer files are stored (default: './detectors').

    Returns:
        list: List of added detector names (e.g., ["E1", "E2", "E3"]).
    """
    # Load the .interferometer config file
    path = Path(config_path)
    file_path = path / f"{config_name}.interferometer"
    if not file_path.exists():
        raise FileNotFoundError(f"Config file {file_path} not found.")

    config = configparser.ConfigParser()
    config.read(file_path)

    added_detectors = []

    for section in config.sections():
        params = config[section]
        det_name = section

        # Parse parameters (assume radians for angles/lat/lon, meters for lengths/heights)
        latitude = float(params['LATITUDE'].split(';')[0].strip())
        longitude = float(params['LONGITUDE'].split(';')[0].strip())
        height = float(params.get('ELEVATION', 0).split(';')[0].strip())
        xangle = float(params['X_AZIMUTH'].split(';')[0].strip())
        yangle = float(params['Y_AZIMUTH'].split(';')[0].strip())
        xaltitude = float(params.get('X_ALTITUDE', 0).split(';')[0].strip())
        yaltitude = float(params.get('Y_ALTITUDE', 0).split(';')[0].strip())
        xlength = float(params.get('X_LENGTH', 10000).split(';')[0].strip())
        ylength = float(params.get('Y_LENGTH', 10000).split(';')[0].strip())

        # Add detector configuration
        add_detector_on_earth(
            name=det_name,
            latitude=latitude,
            longitude=longitude,
            height=height,
            xangle=xangle,
            yangle=yangle,
            xaltitude=xaltitude,
            yaltitude=yaltitude,
            xlength=xlength,
            ylength=ylength
        )
        added_detectors.append(det_name)

    return added_detectors


def extended_get_available_detectors(config_path: str = detectors_dir) -> list[str]:
    """ Extended version of pycbc.detector.get_available_detectors that includes both built-in detectors and available .interferometer config names. """
    built_in_dets = _original_get_available_detectors()
    path = Path(config_path)
    config_files = [f.stem for f in path.glob('*.interferometer')]

    return sorted(set(built_in_dets + config_files))


# Monkey-patch get_available_detectors to include config/group names
get_available_detectors = extended_get_available_detectors


def Detector(name: str, config_path: str = detectors_dir) -> BaseDetector | tuple[BaseDetector, ...]:
    """
    Factory function to create pycbc.detector.Detector objects.
    If `name` is a PyCBC built-in detector (e.g., 'V1'), returns a single pycbc.detector.Detector instance.
    If `name` is a config/group name (e.g., 'ET_Triangle_Sardinia'), automatically loads the config, adds the detectors, and returns a tuple of pycbc.detector.Detector instances for the group.

    Args:
        name (str): The detector name or config/group name.
        config_path (str, optional): Directory where .interferometer files are stored (default: './detectors').

    Returns:
        pycbc.detector.Detector or tuple of pycbc.detector.Detector instances
    """
    built_in_dets = _original_get_available_detectors()
    if name in built_in_dets:
        return BaseDetector(name)
    else:
        # Load the config and add detector configuration
        added_names = load_interferometer_config(name, config_path=config_path)
        if not added_names:
            raise ValueError(f"No detectors loaded from config '{name}'.")

        # Return tuple of pycbc.detector.Detector instances
        detectors = tuple(BaseDetector(det_name) for det_name in added_names)
        if len(detectors) == 1:
            # Return single if only one
            return detectors[0]
        return detectors
