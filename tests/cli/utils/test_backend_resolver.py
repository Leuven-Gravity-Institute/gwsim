"""Tests for backend discovery and validation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from gwmock_noise import BaseNoiseSimulator, NoiseConfig, SimulationResult
from gwmock_signal import DetectorStrainStack, GWSimulator

from gwmock.cli.utils.backend_resolver import instantiate_backend, resolve_backend_class, validate_backend


class ModulePopulationBackend:
    """Protocol-conformant backend used for import-path tests."""

    parameter_names = ("mass_1",)
    source_type = "bbh"

    def simulate(self, n_samples: int, **_kwargs):
        return {"mass_1": np.ones(n_samples)}


class LegacyPopulationBackend(ModulePopulationBackend):
    """Protocol-conformant backend used for legacy dotted-path tests."""


class InvalidPopulationBackend:
    """Backend missing required protocol members."""

    parameter_names = ("mass_1",)


class DuckSignalBackend:
    """Duck-typed signal backend with the public ``simulate`` surface."""

    required_params = frozenset({"coa_time"})

    def simulate(
        self,
        params: Mapping[str, object],
        detector_names,
        background=None,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        earth_rotation: bool = True,
        interpolate_if_offset: bool = True,
    ) -> DetectorStrainStack:
        _ = background, minimum_frequency, earth_rotation, interpolate_if_offset
        return DetectorStrainStack.from_mapping(
            detector_names,
            {detector: np.zeros(int(sampling_frequency)) + float(params["coa_time"]) for detector in detector_names},
        )


class ProtocolNoiseBackend:
    """Runtime-checkable noise backend used for validation tests."""

    def __init__(self) -> None:
        self.duration = 4.0
        self.sampling_frequency = 8.0
        self.detectors = ["H1"]
        self.seed = None

    def generate(self, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None):
        _ = duration, sampling_frequency, seed
        return {detector: np.zeros(4) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        _ = chunk_duration, sampling_frequency, seed
        yield {detector: np.zeros(4) for detector in detectors}

    @property
    def metadata(self) -> dict[str, object]:
        return {"kind": "protocol"}


class RunOnlyNoiseBackend(BaseNoiseSimulator):
    """BaseNoiseSimulator compatibility should remain valid for orchestration."""

    def run(self, config: NoiseConfig) -> SimulationResult:
        return SimulationResult(output_paths={}, config=config)


def test_resolve_population_builtin_alias():
    """Built-in aliases should resolve before other lookup modes."""
    backend_class = resolve_backend_class("population", "file")

    assert backend_class.__name__ == "FilePopulationLoader"


def test_resolve_backend_entry_point_from_installed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Entry-point discovery should work for an installed third-party package."""
    package_dir = tmp_path / "plugin-src"
    site_packages = tmp_path / "site-packages"
    (package_dir / "tiny_backend_pkg").mkdir(parents=True)
    (package_dir / "tiny_backend_pkg" / "__init__.py").write_text("")
    (package_dir / "tiny_backend_pkg" / "population.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class EntryPointPopulationBackend:
                parameter_names = ("mass_1",)
                source_type = "bbh"

                def simulate(self, n_samples: int, **_kwargs):
                    return {"mass_1": np.full(n_samples, 7.0)}
            """
        )
    )
    (package_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "tiny-backend-pkg"
            version = "0.0.1"

            [project.entry-points."gwmock.population"]
            tiny_alias = "tiny_backend_pkg.population:EntryPointPopulationBackend"
            """
        )
    )

    uv_path = shutil.which("uv")
    if uv_path is None:  # pragma: no cover - repository tests run with uv available
        raise AssertionError("uv executable is required for entry-point installation tests.")
    subprocess.run(  # noqa: S603
        [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--quiet",
            "--no-deps",
            "--target",
            str(site_packages),
            str(package_dir),
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(site_packages))

    backend = instantiate_backend("population", "tiny_alias")

    assert backend.__class__.__name__ == "EntryPointPopulationBackend"
    assert backend.simulate(1)["mass_1"][0] == pytest.approx(7.0)


def test_resolve_module_class_literal():
    """Explicit ``module:Class`` paths should resolve directly."""
    backend = instantiate_backend(
        "population",
        "tests.cli.utils.test_backend_resolver:ModulePopulationBackend",
    )

    assert backend.__class__ is ModulePopulationBackend


def test_resolve_legacy_dotted_path_warns_once(monkeypatch):
    """Legacy dotted import paths should warn once and continue to work."""
    from gwmock.cli.utils import backend_resolver

    monkeypatch.setattr(backend_resolver, "_LEGACY_PATH_WARNINGS", set())
    with pytest.warns(DeprecationWarning, match="use 'tests.cli.utils.test_backend_resolver:LegacyPopulationBackend'"):
        first = resolve_backend_class("population", "tests.cli.utils.test_backend_resolver.LegacyPopulationBackend")
    second = resolve_backend_class("population", "tests.cli.utils.test_backend_resolver.LegacyPopulationBackend")

    assert first is LegacyPopulationBackend
    assert second is LegacyPopulationBackend


def test_invalid_population_backend_names_missing_member():
    """Validation failures should name the missing or mismatched protocol member."""
    with pytest.raises(TypeError, match="source_type"):
        instantiate_backend("population", "tests.cli.utils.test_backend_resolver:InvalidPopulationBackend")


def test_validate_signal_backend_accepts_duck_typed_simulator():
    """Signal backends may match by public surface without subclassing ``GWSimulator``."""
    backend = DuckSignalBackend()

    validate_backend("signal", "duck", DuckSignalBackend, backend)


def test_validate_signal_backend_accepts_gwsimulator_subclass():
    """Subclass-based signal backends remain valid."""

    class ConcreteSignalBackend(GWSimulator):
        @property
        def required_params(self) -> frozenset[str]:
            return frozenset({"coa_time"})

        def simulate(
            self,
            params,
            detector_names,
            background=None,
            *,
            sampling_frequency,
            minimum_frequency,
            earth_rotation=True,
            interpolate_if_offset=True,
        ):
            _ = params, background, sampling_frequency, minimum_frequency, earth_rotation, interpolate_if_offset
            return DetectorStrainStack.from_mapping(
                detector_names, {detector: np.zeros(4) for detector in detector_names}
            )

    validate_backend("signal", "concrete", ConcreteSignalBackend, ConcreteSignalBackend())


def test_validate_noise_backend_accepts_protocol_instance():
    """Noise backends may match the runtime-checkable public protocol."""
    validate_backend("noise", "protocol", ProtocolNoiseBackend, ProtocolNoiseBackend())


def test_validate_noise_backend_accepts_run_boundary_class():
    """Run-boundary adapters remain compatible during the orchestration transition."""
    validate_backend("noise", "run-only", RunOnlyNoiseBackend, RunOnlyNoiseBackend())
