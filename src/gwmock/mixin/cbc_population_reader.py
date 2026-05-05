"""Mixin for reading compact binary coalescence (CBC) population data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gwmock.mixin.population_reader import PopulationReaderMixin


class CBCPopulationReaderMixin(PopulationReaderMixin):
    """Mixin class for reading compact binary coalescence (CBC) population data."""

    def __init__(
        self,
        population_file: str | Path,
        population_parameter_name_mapper: dict[str, str] | None = None,
        population_cache_dir: str | Path | None = None,
        population_download_timeout: int = 300,
        source_type: str = "bbh",
        **kwargs,
    ):
        """Initialize the CBC population reader mixin.

        Args:
            population_file: Path or URL to the population data file.
            population_parameter_name_mapper: Optional explicit column overrides forwarded to gwmock-pop.
            population_cache_dir: Optional directory to cache downloaded population files.
            population_download_timeout: Timeout in seconds for downloading population files. Default is 300 seconds.
            source_type: Public gwmock-pop routing key for the CBC catalogue. Defaults to ``"bbh"``.
            **kwargs: Additional arguments absorbed by parent classes.
        """
        self.source_type = source_type
        super().__init__(
            population_file=population_file,
            population_parameter_name_mapper=population_parameter_name_mapper,
            population_sort_by="coa_time",
            population_cache_dir=population_cache_dir,
            population_download_timeout=population_download_timeout,
            **kwargs,
        )

    def _population_get_default_parameter_name_mapper(self) -> dict[str, str]:
        """CBC canonicalization now lives in gwmock-pop's file loader."""
        return {}

    def _population_post_process_population_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """CBC derived-quantity handling now lives in gwmock-pop's file loader."""
        return df
