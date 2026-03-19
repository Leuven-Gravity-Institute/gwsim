"""A package to simulate a population of gravitational waves."""

from __future__ import annotations

import warnings

from . import utils
from .utils.log import setup_logger
from .version import __version__

setup_logger()


warnings.warn(
    "This package `gwsim` has been renamed to `gwmock`. Please update your requirements and imports.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from gwmock import *  # noqa: F403
except ImportError as e:
    # This handles the edge case where the dependency installation failed
    raise ImportError("gwsim requires gwmock to be installed. Please run 'pip install gwmock'.") from e

__all__ = ["__version__", "utils"]
