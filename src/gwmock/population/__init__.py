"""Population helpers and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from gwmock_pop import GWPopSimulator

from gwmock.cli.utils.backend_resolver import instantiate_backend

from .adapter import PopulationAdapter


def instantiate_population_backend(
    backend_name: str,
    *,
    init_kwargs: Mapping[str, Any] | None = None,
) -> GWPopSimulator:
    """Resolve, instantiate, and validate a population backend."""
    return cast(
        "GWPopSimulator",
        instantiate_backend("population", backend_name, init_kwargs=init_kwargs),
    )


__all__ = ["PopulationAdapter", "instantiate_population_backend"]
