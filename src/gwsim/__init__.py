"""A package to simulate a population of gravitational waves."""

from __future__ import annotations

from gwsim.utils.log import setup_logger
from gwsim.version import __version__

setup_logger()

__all__ = ["__version__"]
