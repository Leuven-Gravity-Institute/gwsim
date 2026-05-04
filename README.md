# gwmock

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock/graph/badge.svg?token=GLW2LEFKW7)](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock)](https://pypi.org/project/gwmock/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock)](https://pypi.org/project/gwmock/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Leuven-Gravity-Institute/gwmock/blob/main/LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1115995501.svg)](https://doi.org/10.5281/zenodo.17925458)

A Python package for generating Mock Data Challenge (MDC) datasets for the
gravitational-wave (GW) community. It simulates strain data for detectors like
Einstein Telescope, providing a unified interface for reproducible GW data
generation.

## Features

- **Modular Design**: Uses mixins for flexible simulator composition
- **Detector Support**: Built-in support for various GW detectors with custom
  configuration options
- **Waveform Generation**: Integrates with PyCBC and LALSuite for accurate
  signal simulation
- **Noise Models**: Supports colored and correlated noise generation
  (In-Progress)
- **Population Models**: Handles injection populations for signals and glitches
- **Data Formats**: Outputs in standard GW formats (GWF frames)
- **CLI**: Command-line tools for easy simulation workflows

## Installation

We recommend using `uv` to manage virtual environments for installing gwmock.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

**Note:** The package is built and tested against Python 3.12-3.13. When
creating a virtual environment with `uv`, specify the Python version to ensure
compatibility: `uv venv --python 3.12` (replace `3.12` with your preferred
version in the 3.12-3.13 range). This avoids potential issues with unsupported
Python versions.

### From PyPI

```bash
# Create a virtual environment (recommended with uv)
uv venv gwmock-env --python 3.12
source gwmock-env/bin/activate  # On Windows: gwmock-env\Scripts\activate
uv pip install gwmock
```

### From Source

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
ce gwmock
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

## Quick Start

### Command Line

```bash
# Generate simulated data
gwmock simulate config.yaml
```

## Configuration

gwmock uses YAML configuration files for reproducible simulations. The primary
CLI path is now the adapter-backed `orchestration` surface, which keeps backend
selection explicit without asking users to provide internal Python class paths.

Key configuration sections:

| Section                    | Purpose                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| `globals`                  | Shared orchestration parameters such as sampling rate, segment duration, start time, and output roots |
| `orchestration.population` | Public `gwmock-pop` backend or loader plus its arguments and explicit event count                     |
| `orchestration.signal`     | Public `gwmock-signal` routing inputs, detector network, and signal output settings                   |
| `orchestration.noise`      | Public `gwmock-noise` adapter arguments and noise output settings                                     |

The legacy `simulators.*.class` configuration remains available for
compatibility and metadata reproduction, but it is deprecated for fresh runs;
new configs should prefer the adapter-backed `orchestration` flow. See
`examples/default_config/config.yaml` and
`examples/signal/bbh/et_triangle_sardinia/config.yaml` for concrete examples.

The first protocol-based compatibility release is intentionally scoped to the
path covered by the end-to-end tests:

- file-backed CBC population catalogues loaded through the public `gwmock-pop`
  contract,
- transient CBC signal backends resolved by `source-type` through public
  `gwmock-signal` APIs,
- stateless segment generation through the public `gwmock-noise` run boundary,
  with gwmock still owning orchestration, metadata, and output layout.

Deferred behavior is explicit rather than silent:

- fresh `simulators.*.class` configs are deprecated and retained only for
  backwards compatibility plus metadata reproduction,
- exact hidden-filter continuation across noise segments is still out of scope
  until `gwmock-noise` exposes a public stateful continuation protocol,
- non-transient signal backends that do not expose `generate_polarizations()`
  are not part of the initial compatibility contract.

## Documentation

Full documentation to be available at
[https://leuven-gravity-institute.github.io/gwmock](https://leuven-gravity-institute.github.io/gwmock).

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a merge request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock/releases).

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock/issues/new) or
contact the maintainers.
