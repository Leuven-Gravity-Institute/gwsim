""" Script to test whether the ET detector manually computed and added with PyCBC is equivalent to the LAL detector configuration"""

import numpy as np
import pymap3d as pm

from pycbc.detector import add_detector_on_earth, Detector


def add_ET_Triangular_detector_at_location(E1_latitude: float, E1_longitude: float, E1_height: float, location_name: str, ETArmL: float = 10000) -> (Detector, Detector, Detector):
    """
    Add the triangular Einstein Telescope detector with PyCBC at a given location and height.
    The ET triangular configuration follows T1400308. The arms 1 and 2 of E1 are defined on the tangent plane at the E1 vertex position.
    The arm 1 has the same azimuth angle and altitude of the Virgo arm 1 in the local horizontal coordinate system center at the E1 vertex.

    Args:
        E1_latitude (float): E1 vertex latitude (rad)
        E1_longitude (float): E1 vertex longitude (rad)
        E1_height (float): E1 vertex height above the standard reference ellipsoidal earth (meters)
        location_name (str): Name of the ET location (e.g., Sardinia, EMR, Cascina, ...)
        ETArmL (float, optional): ET arm length (meters). Default to 10000 meters.

    Returns:
        (Detector, Detector, Detector): pycbc.detector.Detector objects for E1, E2 and E3.
    """
    # E1 vertex location in geocentic (ECEF) coordinates
    E1_ellipsoid = [E1_latitude, E1_longitude, E1_height]
    E1 = np.array(pm.geodetic2ecef(*E1_ellipsoid, deg=False))

    # Normal vector to the tangent plane at the E1 vertex (ECEF coordinates)
    E1normVec = np.array([
        np.cos(E1_latitude) * np.cos(E1_longitude),
        np.cos(E1_latitude) * np.sin(E1_longitude),
        np.sin(E1_latitude)
    ])

    # Azimuth and altitude of Virgo arm 1 from LAL
    Virgo_lal = Detector('V1')
    V1Arm1_az = Virgo_lal.info['xangle']
    V1Arm1_alt = Virgo_lal.info['xaltitude']

    # Define the arm 1 of E1 with the same azimuth and altitude of the Virgo arm 1 (ECEF coordinates)
    E1Arm1 = np.array(pm.aer2ecef(
        az=V1Arm1_az,
        el=V1Arm1_alt,
        srange=1,
        lat0=E1_latitude,
        lon0=E1_longitude,
        alt0=E1_height,
        deg=False
    ) - E1)

    # E2 vertex location
    E2 = E1 + (ETArmL * E1Arm1)

    # Calculating rotation matrix to define E2 and E3 arms
    ux, uy, uz = E1normVec
    theta = 60
    cosT = np.cos(np.deg2rad(theta))
    sinT = np.sin(np.deg2rad(theta))
    RE1 = np.array([
        [cosT + ux**2 * (1 - cosT), ux * uy * (1 - cosT) - uz *
         sinT, ux * uz * (1 - cosT) + uy * sinT],
        [ux * uy * (1 - cosT) + uz * sinT, cosT + uy**2 *
         (1 - cosT), uy * uz * (1 - cosT) - ux * sinT],
        [ux * uz * (1 - cosT) - uy * sinT, uy * uz * (1 - cosT) +
         ux * sinT, cosT + uz**2 * (1 - cosT)]
    ])

    # Apply rotational matrix to E1 arm 1 vector to define E1 arm 2
    E1Arm2 = RE1 @ E1Arm1

    # E3 vertex location
    E3 = E1 + (ETArmL * E1Arm2)

    # E2 arm vectors
    E2Arm1 = -E1Arm1 + E1Arm2
    E2Arm2 = -E1Arm1

    # E3 arm vectors
    E3Arm1 = -E1Arm2
    E3Arm2 = -E2Arm1

    # Calculate the vertex positions in geodetic (ellipsoidal) coordinates
    E2_ellipsoid = np.array(pm.ecef2geodetic(*E2, deg=False))
    E3_ellipsoid = np.array(pm.ecef2geodetic(*E3, deg=False))

    # Calculate the unit vector angles (azimuth and altitude)
    E1Arm1Angles = np.array(pm.ecef2aer(*(E1Arm1 + E1), *E1_ellipsoid, deg=False))
    E1Arm2Angles = np.array(pm.ecef2aer(*(E1Arm2 + E1), *E1_ellipsoid, deg=False))
    E2Arm1Angles = np.array(pm.ecef2aer(*(E2Arm1 + E2), *E2_ellipsoid, deg=False))
    E2Arm2Angles = np.array(pm.ecef2aer(*(E2Arm2 + E2), *E2_ellipsoid, deg=False))
    E3Arm1Angles = np.array(pm.ecef2aer(*(E3Arm1 + E3), *E3_ellipsoid, deg=False))
    E3Arm2Angles = np.array(pm.ecef2aer(*(E3Arm2 + E3), *E3_ellipsoid, deg=False))

    # Add detectors with PyCBC
    add_detector_on_earth(
        name="E1_60deg_"+location_name,
        latitude=E1_ellipsoid[0],
        longitude=E1_ellipsoid[1],
        height=E1_ellipsoid[2],
        xangle=E1Arm1Angles[0],
        yangle=E1Arm2Angles[0],
        xaltitude=E1Arm1Angles[1],
        yaltitude=E1Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )
    add_detector_on_earth(
        name="E2_60deg_"+location_name,
        latitude=E2_ellipsoid[0],
        longitude=E2_ellipsoid[1],
        height=E2_ellipsoid[2],
        xangle=E2Arm1Angles[0],
        yangle=E2Arm2Angles[0],
        xaltitude=E2Arm1Angles[1],
        yaltitude=E2Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )
    add_detector_on_earth(
        name="E3_60deg_"+location_name,
        latitude=E3_ellipsoid[0],
        longitude=E3_ellipsoid[1],
        height=E3_ellipsoid[2],
        xangle=E3Arm1Angles[0],
        yangle=E3Arm2Angles[0],
        xaltitude=E3Arm1Angles[1],
        yaltitude=E3Arm2Angles[1],
        xlength=ETArmL,
        ylength=ETArmL
    )

    return Detector("E1_60deg_"+location_name), Detector("E2_60deg_"+location_name), Detector("E3_60deg_"+location_name)


if __name__ == "__main__":

    E1_PyCBC = Detector('E1')

    ifo1, ifo2, ifo3 = add_ET_Triangular_detector_at_location(
        E1_latitude=E1_PyCBC.latitude,
        E1_longitude=E1_PyCBC.longitude,
        E1_height=E1_PyCBC.info['height'],
        location_name="Cascina",
        ETArmL=10000
    )

    print("\n=== Difference E1 hard code vs PyCBC ===")
    for key, val in ifo1.info.items():
        print(key, "\n", val - Detector('E1').info[key])

    print("\n=== Difference E2 hard code vs PyCBC ===")
    for key, val in ifo2.info.items():
        print(key, "\n", val - Detector('E2').info[key])

    print("\n=== Difference E3 hard code vs PyCBC ===")
    for key, val in ifo3.info.items():
        print(key, "\n", val - Detector('E3').info[key])

    print(f"\nHard code Null Stream:\n", ifo1.response + ifo2.response + ifo3.response)
    print(f"PyCBC Null Stream:\n", Detector('E1').response +
          Detector('E2').response + Detector('E3').response)
