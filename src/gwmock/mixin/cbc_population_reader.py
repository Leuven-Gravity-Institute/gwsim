"""Mixin for reading compact binary coalescence (CBC) population data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gwmock.mixin.population_reader import PopulationReaderMixin

logger = logging.getLogger("gwmock")

CBC_COMMON_PARAMETER_NAME_MAPPER = {
    # Masses
    "m1": "detector_frame_mass_1",
    "mass1": "detector_frame_mass_1",
    "mass_1": "detector_frame_mass_1",
    "m2": "detector_frame_mass_2",
    "mass2": "detector_frame_mass_2",
    "mass_2": "detector_frame_mass_2",
    "m1_source": "source_frame_mass_1",
    "mass1_source": "source_frame_mass_1",
    "mass_1_source": "source_frame_mass_1",
    "srcmass1": "source_frame_mass_1",
    "m2_source": "source_frame_mass_2",
    "mass2_source": "source_frame_mass_2",
    "mass_2_source": "source_frame_mass_2",
    "srcmass2": "source_frame_mass_2",
    # Spins
    "chi1x": "spin_1x",
    "chi1y": "spin_1y",
    "chi1z": "spin_1z",
    "chi2x": "spin_2x",
    "chi2y": "spin_2y",
    "chi2z": "spin_2z",
    "spin1x": "spin_1x",
    "spin1y": "spin_1y",
    "spin1z": "spin_1z",
    "spin2x": "spin_2x",
    "spin2y": "spin_2y",
    "spin2z": "spin_2z",
    # Tidal deformabilities
    "Lambda1": "lambda_1",
    "Lambda2": "lambda_2",
    # Luminosity Distance
    "dL": "distance",
    "luminosity_distance": "distance",
    # Coalescence phase
    "Phicoal": "coa_phase",
    # Inclination angle
    "iota": "inclination",
    # Coalescence time
    "tGPS": "coa_time",
    "tc": "coa_time",
    # Sky position
    "ra": "right_ascension",
    "dec": "declination",
    # Polarization angle
    "polarization": "polarization_angle",
    "psi": "polarization_angle",
    # Redshift
    "z": "redshift",
}


class CBCPopulationReaderMixin(PopulationReaderMixin):
    """Mixin class for reading compact binary coalescence (CBC) population data."""

    def __init__(
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        **kwargs,
    ):
        """Initialize the CBC population reader mixin.

        Args:
            population_file: Path or URL to the population data file.
            population_parameter_name_mapper: Optional dictionary to map population parameter names to standard names.
            population_cache_dir: Optional directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300 seconds.
            **kwargs: Additional arguments absorbed by parent classes.
        """
        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_sort_by="coa_time",
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            **kwargs,
        )

    def _population_get_default_parameter_name_mapper(self) -> dict[str, str]:
        """Get the default parameter name mapper for CBC populations.

        Returns:
            A dictionary mapping common CBC population parameter names to standard names.
        """
        return CBC_COMMON_PARAMETER_NAME_MAPPER.copy()

    def _population_post_process_population_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the population data after reading.

        Compute mass1 and mass2 from srcmass1, srcmass2 and redshift if they are not present in the data.

        Args:
            df: DataFrame containing the population data.

        Returns:
            DataFrame with post-processed population data.
        """
        if "detector_frame_mass_1" not in df.columns:
            if "source_frame_mass_1" in df.columns and "redshift" in df.columns:
                df["detector_frame_mass_1"] = df["source_frame_mass_1"] * (1 + df["redshift"])
                logger.info("Computed detector_frame_mass_1 from source_frame_mass_1 and redshift.")
            else:
                raise ValueError(
                    "detector_frame_mass_1 is not in population data, and cannot be computed "
                    "from source_frame_mass_1 and redshift."
                )
        if "detector_frame_mass_2" not in df.columns:
            if "source_frame_mass_2" in df.columns and "redshift" in df.columns:
                df["detector_frame_mass_2"] = df["source_frame_mass_2"] * (1 + df["redshift"])
                logger.info("Computed detector_frame_mass_2 from source_frame_mass_2 and redshift.")
            else:
                raise ValueError(
                    "detector_frame_mass_2 is not in population data, and cannot be computed "
                    "from source_frame_mass_2 and redshift."
                )
        return df
